"""
Data Pipeline Manager

Integrates and orchestrates data collection, processing, and delivery between different system components.
Connects sentiment analysis with data acquisition and implements caching, scheduling, and validation.
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json
import os
import uuid
from enum import Enum
import aiofiles
from aiohttp import ClientSession
import pandas as pd
import numpy as np
from functools import lru_cache
import pickle
import hashlib
from pathlib import Path

# Import AI Trading Agent components
from ai_trading_agent.data_acquisition.data_service import DataService
from ai_trading_agent.sentiment_analysis.sentiment_service import SentimentService
from ai_trading_agent.sentiment_analysis.analyzer import SentimentAnalyzer
from ai_trading_agent.sentiment_analysis.collector import (
    TwitterCollector, RedditCollector, NewsCollector, FearGreedIndexCollector
)

# Logging setup
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source types."""
    MARKET_DATA = "market_data"
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    FEAR_GREED = "fear_greed"
    FUNDAMENTALS = "fundamentals"
    TECHNICAL = "technical"


class ScheduleFrequency(Enum):
    """Schedule frequency options."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class DataPipelineManager:
    """
    Manages the flow of data between different components of the system.
    
    Responsibilities:
    1. Connect sentiment analysis system with data acquisition module
    2. Set up and manage scheduled data collection tasks
    3. Implement caching layer for performance optimization
    4. Add data validation and error recovery mechanisms
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data pipeline manager.
        
        Args:
            config: Configuration dictionary with settings for data sources,
                   caching, scheduling, and error handling
        """
        self.config = config
        self.cache_dir = Path(config.get("cache_dir", "data/cache"))
        self.data_dir = Path(config.get("data_dir", "data"))
        
        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Task tracking
        self._scheduled_tasks: Dict[str, asyncio.Task] = {}
        self._active_collectors: Dict[str, Any] = {}
        self._task_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Cache settings
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_ttl = config.get("cache_ttl", {})  # TTL in seconds per data source
        
        # Error handling
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 5)  # seconds
        self.error_thresholds = config.get("error_thresholds", {})
        self._error_counters: Dict[str, int] = {}
        
        # Rate limiting
        self.rate_limits = config.get("rate_limits", {})
        self._last_api_call_time: Dict[str, datetime] = {}
        
        logger.info("Data Pipeline Manager initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        # Initialize data service
        data_service_config = self.config.get("data_service", {})
        self.data_service = DataService(data_service_config)
        
        # Initialize sentiment service
        sentiment_service_config = self.config.get("sentiment_service", {})
        self.sentiment_service = SentimentService(sentiment_service_config)
        
        # Initialize sentiment analyzer
        analyzer_config = self.config.get("sentiment_analyzer", {})
        self.sentiment_analyzer = SentimentAnalyzer(analyzer_config)
    
    async def start_scheduled_tasks(self) -> None:
        """Start all configured scheduled tasks."""
        # Get schedules from config
        schedules = self.config.get("schedules", {})
        
        # Start tasks for each configured schedule
        for source_name, schedule_config in schedules.items():
            if not schedule_config.get("enabled", True):
                logger.info(f"Schedule for {source_name} is disabled, skipping")
                continue
                
            try:
                source_type = DataSource(source_name)
                frequency = ScheduleFrequency(schedule_config.get("frequency", "1h"))
                symbols = schedule_config.get("symbols", [])
                params = schedule_config.get("params", {})
                
                # Create and start task
                task_id = f"{source_name}_{frequency.value}_{uuid.uuid4().hex[:8]}"
                task = asyncio.create_task(
                    self._run_scheduled_collection(task_id, source_type, frequency, symbols, params)
                )
                
                # Track the task
                self._scheduled_tasks[task_id] = task
                self._task_history[task_id] = []
                
                logger.info(f"Started scheduled task {task_id} for {source_type.value} data at {frequency.value} frequency")
                
            except ValueError as e:
                logger.error(f"Invalid configuration for scheduled task {source_name}: {e}")
            except Exception as e:
                logger.error(f"Error starting scheduled task for {source_name}: {e}")
    
    async def stop_scheduled_tasks(self) -> None:
        """Stop all running scheduled tasks."""
        for task_id, task in list(self._scheduled_tasks.items()):
            if not task.done():
                logger.info(f"Canceling scheduled task {task_id}")
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Task {task_id} canceled successfully")
                except Exception as e:
                    logger.error(f"Error during task {task_id} cancellation: {e}")
            
            # Remove from tracking
            del self._scheduled_tasks[task_id]
        
        logger.info("All scheduled tasks stopped")
    
    async def _run_scheduled_collection(
        self, 
        task_id: str, 
        source_type: DataSource, 
        frequency: ScheduleFrequency,
        symbols: List[str],
        params: Dict[str, Any]
    ) -> None:
        """
        Run scheduled data collection task.
        
        Args:
            task_id: Unique identifier for this task
            source_type: Type of data source to collect
            frequency: Collection frequency
            symbols: List of symbols to collect data for
            params: Additional parameters for the collection
        """
        try:
            # Calculate seconds between runs
            seconds_map = {
                ScheduleFrequency.MINUTE_1: 60,
                ScheduleFrequency.MINUTE_5: 300,
                ScheduleFrequency.MINUTE_15: 900,
                ScheduleFrequency.MINUTE_30: 1800,
                ScheduleFrequency.HOUR_1: 3600,
                ScheduleFrequency.HOUR_4: 14400,
                ScheduleFrequency.DAY_1: 86400,
                ScheduleFrequency.WEEK_1: 604800
            }
            interval_seconds = seconds_map.get(frequency, 3600)
            
            while True:
                start_time = time.time()
                
                try:
                    # Record task execution start
                    execution_record = {
                        "start_time": datetime.now().isoformat(),
                        "status": "running",
                        "source_type": source_type.value,
                        "symbols": symbols,
                        "frequency": frequency.value
                    }
                    self._task_history[task_id].append(execution_record)
                    
                    # Collect data based on source type
                    if source_type == DataSource.MARKET_DATA:
                        await self._collect_market_data(symbols, params)
                    elif source_type == DataSource.TWITTER:
                        await self._collect_twitter_data(symbols, params)
                    elif source_type == DataSource.REDDIT:
                        await self._collect_reddit_data(symbols, params)
                    elif source_type == DataSource.NEWS:
                        await self._collect_news_data(symbols, params)
                    elif source_type == DataSource.FEAR_GREED:
                        await self._collect_fear_greed_data(params)
                    elif source_type == DataSource.FUNDAMENTALS:
                        await self._collect_fundamental_data(symbols, params)
                    elif source_type == DataSource.TECHNICAL:
                        await self._collect_technical_data(symbols, params)
                    
                    # Update execution record with success status
                    execution_record.update({
                        "end_time": datetime.now().isoformat(),
                        "status": "completed",
                        "duration": time.time() - start_time
                    })
                    
                except Exception as e:
                    logger.error(f"Error in scheduled task {task_id}: {e}")
                    
                    # Update execution record with failure status
                    execution_record.update({
                        "end_time": datetime.now().isoformat(),
                        "status": "failed",
                        "error": str(e),
                        "duration": time.time() - start_time
                    })
                    
                    # Increment error counter
                    source_name = source_type.value
                    self._error_counters[source_name] = self._error_counters.get(source_name, 0) + 1
                    
                    # Check if we need to pause collection for this source
                    if self._should_pause_collection(source_name):
                        logger.warning(f"Pausing collection for {source_name} due to excessive errors")
                        # Wait for the retry delay before trying again
                        await asyncio.sleep(self.retry_delay * 10)  # Longer delay for paused collection
                
                # Keep track of the most recent records (limit to 100 per task)
                if len(self._task_history[task_id]) > 100:
                    self._task_history[task_id] = self._task_history[task_id][-100:]
                
                # Calculate time to wait for next execution
                elapsed = time.time() - start_time
                wait_time = max(0, interval_seconds - elapsed)
                
                await asyncio.sleep(wait_time)
                
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was canceled")
            raise
        except Exception as e:
            logger.error(f"Unhandled error in scheduled task {task_id}: {e}")
            raise
    
    def _should_pause_collection(self, source_name: str) -> bool:
        """Determine if collection should be paused based on error history."""
        error_count = self._error_counters.get(source_name, 0)
        threshold = self.error_thresholds.get(source_name, 5)
        return error_count >= threshold
    
    async def _collect_market_data(self, symbols: List[str], params: Dict[str, Any]) -> None:
        """Collect market data for specified symbols."""
        timeframe = params.get("timeframe", "1h")
        limit = params.get("limit", 100)
        
        for symbol in symbols:
            try:
                logger.info(f"Collecting market data for {symbol} ({timeframe})")
                
                # Check cache first
                cache_key = f"market_data_{symbol}_{timeframe}"
                cached_data = self._get_from_cache(cache_key)
                
                if cached_data is not None:
                    logger.info(f"Using cached market data for {symbol}")
                    # In a real implementation, we might update this with just the latest data
                    continue
                
                # Collect fresh data
                data = await self.data_service.get_historical_data(symbol, timeframe, limit)
                
                # Store in cache
                self._store_in_cache(cache_key, data)
                
                # Save to data directory
                await self._save_market_data(symbol, timeframe, data)
                
                logger.info(f"Successfully collected market data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting market data for {symbol}: {e}")
                raise
    
    async def _collect_twitter_data(self, symbols: List[str], params: Dict[str, Any]) -> None:
        """Collect Twitter sentiment data for specified symbols."""
        limit = params.get("limit", 100)
        lang = params.get("language", "en")
        
        # Initialize collector if needed
        if "twitter" not in self._active_collectors:
            self._active_collectors["twitter"] = TwitterCollector(self.config.get("twitter", {}))
        
        collector = self._active_collectors["twitter"]
        
        for symbol in symbols:
            try:
                logger.info(f"Collecting Twitter data for {symbol}")
                
                # Check cache first
                cache_key = f"twitter_{symbol}_{lang}"
                cached_data = self._get_from_cache(cache_key)
                
                if cached_data is not None:
                    # For social media, we typically want fresh data each time
                    # but we might use cache if the last collection was very recent
                    last_collection = self._get_last_collection_time(cache_key)
                    if last_collection and (datetime.now() - last_collection).total_seconds() < 300:  # 5 minutes
                        logger.info(f"Using recent cached Twitter data for {symbol}")
                        continue
                
                # Respect rate limits
                await self._enforce_rate_limit("twitter")
                
                # Collect fresh data
                data = await collector.fetch_sentiment_data(symbol, limit=limit, lang=lang)
                
                # Process through sentiment analyzer
                processed_data = await self.sentiment_analyzer.analyze_batch(data)
                
                # Store in cache
                self._store_in_cache(cache_key, processed_data)
                
                # Save to data directory
                await self._save_sentiment_data("twitter", symbol, processed_data)
                
                logger.info(f"Successfully collected Twitter data for {symbol}: {len(processed_data)} tweets")
                
            except Exception as e:
                logger.error(f"Error collecting Twitter data for {symbol}: {e}")
                raise
    
    async def _collect_reddit_data(self, symbols: List[str], params: Dict[str, Any]) -> None:
        """Collect Reddit sentiment data for specified symbols."""
        subreddits = params.get("subreddits", ["cryptocurrency", "wallstreetbets", "investing"])
        limit = params.get("limit", 100)
        time_filter = params.get("time_filter", "day")
        
        # Initialize collector if needed
        if "reddit" not in self._active_collectors:
            self._active_collectors["reddit"] = RedditCollector(self.config.get("reddit", {}))
        
        collector = self._active_collectors["reddit"]
        
        for symbol in symbols:
            try:
                logger.info(f"Collecting Reddit data for {symbol}")
                
                # Check cache first
                cache_key = f"reddit_{symbol}_{time_filter}"
                cached_data = self._get_from_cache(cache_key)
                
                if cached_data is not None:
                    last_collection = self._get_last_collection_time(cache_key)
                    if last_collection and (datetime.now() - last_collection).total_seconds() < 1800:  # 30 minutes
                        logger.info(f"Using recent cached Reddit data for {symbol}")
                        continue
                
                # Respect rate limits
                await self._enforce_rate_limit("reddit")
                
                # Collect fresh data
                data = await collector.fetch_sentiment_data(
                    symbol, 
                    subreddits=subreddits, 
                    limit=limit, 
                    time_filter=time_filter
                )
                
                # Process through sentiment analyzer
                processed_data = await self.sentiment_analyzer.analyze_batch(data)
                
                # Store in cache
                self._store_in_cache(cache_key, processed_data)
                
                # Save to data directory
                await self._save_sentiment_data("reddit", symbol, processed_data)
                
                logger.info(f"Successfully collected Reddit data for {symbol}: {len(processed_data)} posts")
                
            except Exception as e:
                logger.error(f"Error collecting Reddit data for {symbol}: {e}")
                raise
    
    async def _collect_news_data(self, symbols: List[str], params: Dict[str, Any]) -> None:
        """Collect News sentiment data for specified symbols."""
        days = params.get("days", 1)
        limit = params.get("limit", 50)
        
        # Initialize collector if needed
        if "news" not in self._active_collectors:
            self._active_collectors["news"] = NewsCollector(self.config.get("news", {}))
        
        collector = self._active_collectors["news"]
        
        for symbol in symbols:
            try:
                logger.info(f"Collecting News data for {symbol}")
                
                # Check cache first
                cache_key = f"news_{symbol}_{days}d"
                cached_data = self._get_from_cache(cache_key)
                
                if cached_data is not None:
                    last_collection = self._get_last_collection_time(cache_key)
                    if last_collection and (datetime.now() - last_collection).total_seconds() < 3600:  # 1 hour
                        logger.info(f"Using recent cached News data for {symbol}")
                        continue
                
                # Respect rate limits
                await self._enforce_rate_limit("news")
                
                # Collect fresh data
                data = await collector.fetch_sentiment_data(symbol, days=days, limit=limit)
                
                # Process through sentiment analyzer
                processed_data = await self.sentiment_analyzer.analyze_batch(data)
                
                # Store in cache
                self._store_in_cache(cache_key, processed_data)
                
                # Save to data directory
                await self._save_sentiment_data("news", symbol, processed_data)
                
                logger.info(f"Successfully collected News data for {symbol}: {len(processed_data)} articles")
                
            except Exception as e:
                logger.error(f"Error collecting News data for {symbol}: {e}")
                raise
    
    async def _collect_fear_greed_data(self, params: Dict[str, Any]) -> None:
        """Collect Fear & Greed Index data."""
        days = params.get("days", 30)
        
        # Initialize collector if needed
        if "fear_greed" not in self._active_collectors:
            self._active_collectors["fear_greed"] = FearGreedIndexCollector(self.config.get("fear_greed", {}))
        
        collector = self._active_collectors["fear_greed"]
        
        try:
            logger.info("Collecting Fear & Greed Index data")
            
            # Check cache first
            cache_key = f"fear_greed_{days}d"
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data is not None:
                last_collection = self._get_last_collection_time(cache_key)
                if last_collection and (datetime.now() - last_collection).total_seconds() < 3600:  # 1 hour
                    logger.info("Using recent cached Fear & Greed data")
                    return
            
            # Respect rate limits
            await self._enforce_rate_limit("fear_greed")
            
            # Collect fresh data
            data = await collector.fetch_sentiment_data(days=days)
            
            # Store in cache
            self._store_in_cache(cache_key, data)
            
            # Save to data directory
            await self._save_sentiment_data("fear_greed", "index", data)
            
            logger.info(f"Successfully collected Fear & Greed data: {len(data)} days")
            
        except Exception as e:
            logger.error(f"Error collecting Fear & Greed data: {e}")
            raise
    
    async def _collect_fundamental_data(self, symbols: List[str], params: Dict[str, Any]) -> None:
        """Collect fundamental data for specified symbols."""
        # This would connect to a fundamental data provider
        logger.info(f"Collecting fundamental data for {len(symbols)} symbols")
        # Placeholder for future implementation
        pass
    
    async def _collect_technical_data(self, symbols: List[str], params: Dict[str, Any]) -> None:
        """Collect technical indicator data for specified symbols."""
        timeframe = params.get("timeframe", "1d")
        indicators = params.get("indicators", ["sma", "ema", "rsi", "macd"])
        
        for symbol in symbols:
            try:
                logger.info(f"Generating technical indicators for {symbol} ({timeframe})")
                
                # Check cache first
                cache_key = f"technical_{symbol}_{timeframe}"
                cached_data = self._get_from_cache(cache_key)
                
                if cached_data is not None:
                    logger.info(f"Using cached technical data for {symbol}")
                    continue
                
                # Get historical data
                market_data = await self.data_service.get_historical_data(symbol, timeframe, 200)
                
                # Convert to pandas DataFrame
                df = pd.DataFrame(market_data)
                
                # Generate technical indicators
                technical_data = {}
                for indicator in indicators:
                    if indicator == "sma":
                        for period in [20, 50, 200]:
                            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
                            technical_data[f"sma_{period}"] = df[f"sma_{period}"].to_dict()
                    
                    elif indicator == "ema":
                        for period in [12, 26]:
                            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
                            technical_data[f"ema_{period}"] = df[f"ema_{period}"].to_dict()
                    
                    elif indicator == "rsi":
                        # Calculate RSI using pandas
                        delta = df["close"].diff()
                        gain = delta.where(delta > 0, 0)
                        loss = -delta.where(delta < 0, 0)
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss
                        df["rsi"] = 100 - (100 / (1 + rs))
                        technical_data["rsi"] = df["rsi"].to_dict()
                    
                    elif indicator == "macd":
                        # Calculate MACD using pandas
                        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
                        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
                        df["macd"] = ema_12 - ema_26
                        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
                        df["macd_hist"] = df["macd"] - df["macd_signal"]
                        
                        technical_data["macd"] = df["macd"].to_dict()
                        technical_data["macd_signal"] = df["macd_signal"].to_dict()
                        technical_data["macd_hist"] = df["macd_hist"].to_dict()
                
                # Store in cache
                self._store_in_cache(cache_key, technical_data)
                
                # Save to data directory
                await self._save_technical_data(symbol, timeframe, technical_data)
                
                logger.info(f"Successfully generated technical data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error generating technical data for {symbol}: {e}")
                raise
    
    async def get_sentiment_data(
        self, 
        symbol: str, 
        source_types: List[DataSource] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get sentiment data for a symbol from specified sources.
        
        Args:
            symbol: Symbol to get sentiment data for
            source_types: List of data sources to include (default: all sentiment sources)
            start_time: Start time for data filtering
            end_time: End time for data filtering
            
        Returns:
            Dictionary with sentiment data from each source
        """
        if source_types is None:
            source_types = [
                DataSource.TWITTER, 
                DataSource.REDDIT, 
                DataSource.NEWS, 
                DataSource.FEAR_GREED
            ]
            
        result = {}
        
        for source_type in source_types:
            try:
                source_name = source_type.value
                
                # Default file path based on source and symbol
                file_path = self.data_dir / f"{source_name}_{symbol}.json"
                
                # For Fear & Greed index, the file name doesn't include a symbol
                if source_type == DataSource.FEAR_GREED:
                    file_path = self.data_dir / "fear_greed_index.json"
                
                # Load data if file exists
                if file_path.exists():
                    async with aiofiles.open(file_path, "r") as f:
                        data = json.loads(await f.read())
                        
                    # Filter by time range if specified
                    if start_time or end_time:
                        filtered_data = []
                        for item in data:
                            timestamp = datetime.fromisoformat(item.get("timestamp", ""))
                            
                            if start_time and timestamp < start_time:
                                continue
                                
                            if end_time and timestamp > end_time:
                                continue
                                
                            filtered_data.append(item)
                            
                        result[source_name] = filtered_data
                    else:
                        result[source_name] = data
                else:
                    logger.warning(f"No {source_name} data found for {symbol}")
                    result[source_name] = []
                    
            except Exception as e:
                logger.error(f"Error getting {source_type.value} data for {symbol}: {e}")
                result[source_type.value] = []
        
        return result
    
    async def get_aggregated_sentiment(
        self,
        symbol: str,
        source_types: List[DataSource] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment score for a symbol.
        
        Args:
            symbol: Symbol to get sentiment for
            source_types: List of data sources to include
            start_time: Start time for data filtering
            end_time: End time for data filtering
            weights: Dictionary of source weights (default: equal weighting)
            
        Returns:
            Dictionary with aggregated sentiment metrics
        """
        # Get raw sentiment data
        sentiment_data = await self.get_sentiment_data(symbol, source_types, start_time, end_time)
        
        if not any(sentiment_data.values()):
            logger.warning(f"No sentiment data found for {symbol}")
            return {
                "symbol": symbol,
                "score": 0,
                "source_scores": {},
                "count": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Default weights if not provided
        if weights is None:
            num_sources = len([s for s in sentiment_data.keys() if sentiment_data[s]])
            weights = {source: 1.0 / max(num_sources, 1) for source in sentiment_data.keys()}
        
        # Calculate sentiment scores for each source
        source_scores = {}
        source_counts = {}
        
        for source, items in sentiment_data.items():
            if not items:
                source_scores[source] = 0
                source_counts[source] = 0
                continue
                
            total_score = sum(item.get("sentiment_score", 0) for item in items)
            source_scores[source] = total_score / len(items) if items else 0
            source_counts[source] = len(items)
        
        # Calculate weighted average
        weighted_score = sum(
            source_scores[source] * weights.get(source, 0)
            for source in source_scores.keys()
        )
        
        # Calculate total count
        total_count = sum(source_counts.values())
        
        return {
            "symbol": symbol,
            "score": weighted_score,
            "source_scores": source_scores,
            "count": total_count,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _save_market_data(self, symbol: str, timeframe: str, data: List[Dict[str, Any]]) -> None:
        """Save market data to file."""
        file_path = self.data_dir / f"market_{symbol}_{timeframe}.json"
        
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(data, default=str))
    
    async def _save_sentiment_data(self, source: str, symbol: str, data: List[Dict[str, Any]]) -> None:
        """Save sentiment data to file."""
        file_path = self.data_dir / f"{source}_{symbol}.json"
        
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(data, default=str))
    
    async def _save_technical_data(self, symbol: str, timeframe: str, data: Dict[str, Any]) -> None:
        """Save technical indicator data to file."""
        file_path = self.data_dir / f"technical_{symbol}_{timeframe}.json"
        
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(data, default=str))
    
    def _get_from_cache(self, key: str) -> Any:
        """Get data from cache if available and not expired."""
        if not self.cache_enabled:
            return None
            
        # Determine cache file path
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        
        if not cache_file.exists():
            return None
            
        try:
            # Load cache file
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                
            # Check if cache has expired
            timestamp = cache_data.get("timestamp", 0)
            ttl = self._get_cache_ttl_for_key(key)
            
            if time.time() - timestamp > ttl:
                logger.debug(f"Cache expired for key: {key}")
                return None
                
            return cache_data.get("data")
            
        except Exception as e:
            logger.error(f"Error reading from cache for key {key}: {e}")
            return None
    
    def _store_in_cache(self, key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        if not self.cache_enabled:
            return
            
        # Determine cache file path
        cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        
        try:
            # Create cache data with timestamp
            cache_data = {
                "timestamp": time.time(),
                "data": data
            }
            
            # Write to cache file
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.error(f"Error writing to cache for key {key}: {e}")
    
    def _get_cache_ttl_for_key(self, key: str) -> int:
        """Get TTL for a cache key based on data source."""
        # Default TTL of 1 hour
        default_ttl = 3600
        
        # Extract source from key
        parts = key.split("_")
        if not parts:
            return default_ttl
            
        source = parts[0]
        
        # Get TTL from config
        return self.cache_ttl.get(source, default_ttl)
    
    def _get_last_collection_time(self, key: str) -> Optional[datetime]:
        """Get the timestamp of the last data collection."""
        cache_data = self._get_from_cache(key)
        
        if cache_data is None:
            return None
            
        # This is a simplification - in a real implementation we'd store the collection timestamp
        # separately from the cache expiration timestamp
        if isinstance(cache_data, dict) and "timestamp" in cache_data:
            return datetime.fromtimestamp(cache_data["timestamp"])
        
        return None
    
    @staticmethod
    def _hash_key(key: str) -> str:
        """Create a filename-safe hash of a cache key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    async def _enforce_rate_limit(self, source: str) -> None:
        """
        Enforce rate limits for API calls.
        Will sleep until it's safe to make another API call.
        """
        # Get rate limit for source
        rate_limit = self.rate_limits.get(source, {"calls": 1, "period": 1})
        calls_allowed = rate_limit.get("calls", 1)
        period_seconds = rate_limit.get("period", 1)
        
        current_time = datetime.now()
        
        # If this is the first call, just record the time and return
        if source not in self._last_api_call_time:
            self._last_api_call_time[source] = current_time
            return
            
        last_call_time = self._last_api_call_time[source]
        time_since_last_call = (current_time - last_call_time).total_seconds()
        
        # Calculate time to wait
        time_per_call = period_seconds / calls_allowed
        time_to_wait = max(0, time_per_call - time_since_last_call)
        
        if time_to_wait > 0:
            logger.debug(f"Rate limiting {source} API call, waiting {time_to_wait:.2f} seconds")
            await asyncio.sleep(time_to_wait)
        
        # Update last call time
        self._last_api_call_time[source] = datetime.now()


# Create a singleton instance
_data_pipeline_manager_instance = None

def get_data_pipeline_manager(config: Dict[str, Any] = None) -> DataPipelineManager:
    """
    Get or create the data pipeline manager singleton instance.
    
    Args:
        config: Configuration dictionary (optional, used only when creating new instance)
        
    Returns:
        DataPipelineManager instance
    """
    global _data_pipeline_manager_instance
    
    if _data_pipeline_manager_instance is None:
        if config is None:
            config = {}  # Default empty config
        _data_pipeline_manager_instance = DataPipelineManager(config)
        
    return _data_pipeline_manager_instance