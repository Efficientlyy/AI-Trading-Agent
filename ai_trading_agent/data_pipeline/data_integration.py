"""
Data Integration Module

This module provides a unified interface for collecting and processing data from various sources,
including Alpha Vantage for sentiment data and Twelve Data for market data.
"""

import logging
import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal

from ..sentiment_analysis.alpha_vantage_connector import AlphaVantageSentimentConnector
from ..data_acquisition.twelve_data_connector import TwelveDataConnector

logger = logging.getLogger(__name__)


class DataIntegrationManager:
    """
    Manager for integrating data from various sources.
    
    This class provides a unified interface for collecting, processing, and storing
    data from different sources, including sentiment data and market data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data integration manager.
        
        Args:
            config: Configuration dictionary for the manager
        """
        self.config = config or {}
        
        # Initialize connectors
        self.sentiment_connector = AlphaVantageSentimentConnector(
            config=self.config.get("alpha_vantage", {})
        )
        
        self.market_connector = TwelveDataConnector(
            config=self.config.get("twelve_data", {})
        )
        
        # Initialize data cache
        self.sentiment_cache = {}
        self.market_data_cache = {}
        self.cache_expiry = {}
        
        # Default cache expiry times (in seconds)
        self.sentiment_cache_expiry = int(self.config.get("sentiment_cache_expiry", 3600))  # 1 hour
        self.market_data_cache_expiry = int(self.config.get("market_data_cache_expiry", 60))  # 1 minute
        
        # Initialize scheduled tasks
        self.scheduled_tasks = {}
        self.running = False
        
        # Initialize callbacks
        self.data_callbacks = {}
        
        # Create cache directory if it doesn't exist
        self.cache_dir = self.config.get("cache_dir", "data_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("Initialized DataIntegrationManager")
    
    async def start(self) -> None:
        """Start the data integration manager."""
        if self.running:
            logger.warning("DataIntegrationManager is already running")
            return
        
        self.running = True
        
        # Start WebSocket connection for real-time market data
        await self.market_connector.connect()
        
        # Start scheduled tasks
        self._start_scheduled_tasks()
        
        logger.info("Started DataIntegrationManager")
    
    async def stop(self) -> None:
        """Stop the data integration manager."""
        if not self.running:
            logger.warning("DataIntegrationManager is not running")
            return
        
        self.running = False
        
        # Stop scheduled tasks
        for task_name, task in self.scheduled_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled scheduled task: {task_name}")
        
        # Close WebSocket connection
        await self.market_connector.disconnect()
        
        logger.info("Stopped DataIntegrationManager")
    
    def _start_scheduled_tasks(self) -> None:
        """Start scheduled tasks for data collection."""
        # Schedule sentiment data collection
        sentiment_interval = int(self.config.get("sentiment_collection_interval", 900))  # 15 minutes
        self.scheduled_tasks["sentiment_collection"] = asyncio.create_task(
            self._scheduled_sentiment_collection(sentiment_interval)
        )
        
        # Schedule market data collection (for historical data)
        market_interval = int(self.config.get("market_collection_interval", 3600))  # 1 hour
        self.scheduled_tasks["market_collection"] = asyncio.create_task(
            self._scheduled_market_collection(market_interval)
        )
        
        # Schedule cache cleanup
        cleanup_interval = int(self.config.get("cache_cleanup_interval", 3600))  # 1 hour
        self.scheduled_tasks["cache_cleanup"] = asyncio.create_task(
            self._scheduled_cache_cleanup(cleanup_interval)
        )
        
        logger.info("Started scheduled tasks")
    
    async def _scheduled_sentiment_collection(self, interval: int) -> None:
        """
        Scheduled task for collecting sentiment data.
        
        Args:
            interval: Collection interval in seconds
        """
        try:
            while self.running:
                logger.info("Running scheduled sentiment collection")
                
                # Get symbols to collect sentiment for
                symbols = self.config.get("sentiment_symbols", ["BTC", "ETH"])
                
                # Collect sentiment data for each symbol
                for symbol in symbols:
                    try:
                        await self.get_sentiment_data(symbol)
                    except Exception as e:
                        logger.error(f"Error collecting sentiment data for {symbol}: {e}")
                
                # Wait for next collection
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Sentiment collection task cancelled")
        except Exception as e:
            logger.error(f"Error in sentiment collection task: {e}")
    
    async def _scheduled_market_collection(self, interval: int) -> None:
        """
        Scheduled task for collecting historical market data.
        
        Args:
            interval: Collection interval in seconds
        """
        try:
            while self.running:
                logger.info("Running scheduled market data collection")
                
                # Get symbols to collect market data for
                symbols = self.config.get("market_symbols", ["BTC/USD", "ETH/USD"])
                
                # Collect market data for each symbol
                for symbol in symbols:
                    try:
                        # Get daily OHLCV data
                        await self.get_historical_market_data(symbol, "1day", 30)
                    except Exception as e:
                        logger.error(f"Error collecting market data for {symbol}: {e}")
                
                # Wait for next collection
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Market collection task cancelled")
        except Exception as e:
            logger.error(f"Error in market collection task: {e}")
    
    async def _scheduled_cache_cleanup(self, interval: int) -> None:
        """
        Scheduled task for cleaning up expired cache entries.
        
        Args:
            interval: Cleanup interval in seconds
        """
        try:
            while self.running:
                logger.info("Running scheduled cache cleanup")
                
                # Get current time
                current_time = time.time()
                
                # Clean up sentiment cache
                for key in list(self.sentiment_cache.keys()):
                    if key in self.cache_expiry and self.cache_expiry[key] < current_time:
                        del self.sentiment_cache[key]
                        del self.cache_expiry[key]
                        logger.debug(f"Removed expired sentiment cache entry: {key}")
                
                # Clean up market data cache
                for key in list(self.market_data_cache.keys()):
                    if key in self.cache_expiry and self.cache_expiry[key] < current_time:
                        del self.market_data_cache[key]
                        del self.cache_expiry[key]
                        logger.debug(f"Removed expired market data cache entry: {key}")
                
                # Wait for next cleanup
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Cache cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in cache cleanup task: {e}")
    
    async def get_sentiment_data(self, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get sentiment data for a symbol.
        
        Args:
            symbol: Symbol to get sentiment data for
            force_refresh: Whether to force a refresh from the API
            
        Returns:
            DataFrame containing sentiment data
        """
        cache_key = f"sentiment_{symbol}"
        
        # Check cache if not forcing refresh
        if not force_refresh and cache_key in self.sentiment_cache:
            # Check if cache is still valid
            if cache_key in self.cache_expiry and self.cache_expiry[cache_key] > time.time():
                logger.debug(f"Using cached sentiment data for {symbol}")
                return self.sentiment_cache[cache_key]
        
        try:
            # Try to get sentiment data by crypto ticker
            logger.info(f"Fetching sentiment data for {symbol}")
            sentiment_data = self.sentiment_connector.get_sentiment_by_crypto(symbol, days_back=7)
            
            # Check if we got valid data
            if not sentiment_data or "error" in sentiment_data:
                # Fall back to topic-based sentiment
                logger.info(f"Falling back to topic-based sentiment for {symbol}")
                topics = ["blockchain", "cryptocurrency"]
                sentiment_data = self.sentiment_connector.get_sentiment_by_topic(topics, days_back=7)
            
            # Process sentiment data into DataFrame
            df = self._process_sentiment_data(sentiment_data, symbol)
            
            # Cache the data
            self.sentiment_cache[cache_key] = df
            self.cache_expiry[cache_key] = time.time() + self.sentiment_cache_expiry
            
            # Save to disk cache
            self._save_to_cache(cache_key, df)
            
            # Notify callbacks
            self._notify_callbacks("sentiment", symbol, df)
            
            return df
        except Exception as e:
            logger.error(f"Error getting sentiment data for {symbol}: {e}")
            
            # Try to load from disk cache as fallback
            df = self._load_from_cache(cache_key)
            if df is not None:
                logger.info(f"Loaded sentiment data for {symbol} from disk cache")
                return df
            
            # Return empty DataFrame if all else fails
            return pd.DataFrame()
    
    def _process_sentiment_data(self, sentiment_data: Dict[str, Any], symbol: str) -> pd.DataFrame:
        """
        Process raw sentiment data into a DataFrame.
        
        Args:
            sentiment_data: Raw sentiment data from Alpha Vantage
            symbol: Symbol the data is for
            
        Returns:
            Processed DataFrame
        """
        # Check if we have valid data
        if not sentiment_data or "feed" not in sentiment_data:
            return pd.DataFrame()
        
        # Extract feed items
        feed = sentiment_data.get("feed", [])
        
        # Create DataFrame
        rows = []
        for item in feed:
            # Extract sentiment scores
            sentiment = item.get("overall_sentiment_score", 0)
            
            # Extract time
            time_str = item.get("time_published", "")
            try:
                timestamp = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
            except ValueError:
                timestamp = datetime.now()
            
            # Create row
            row = {
                "timestamp": timestamp,
                "compound_score": sentiment,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "topics": ", ".join(item.get("topics", [])),
                "symbol": symbol
            }
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Sort by timestamp
        if not df.empty:
            df = df.sort_values("timestamp", ascending=False)
        
        return df
    
    async def get_real_time_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get real-time price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price or None if not available
        """
        try:
            # Check if we're subscribed to this symbol
            if not self.market_connector.is_subscribed(symbol):
                # Subscribe to the symbol
                await self.market_connector.subscribe(symbol, "price")
            
            # Get the latest price
            price = self.market_connector.get_latest_price(symbol)
            
            if price is not None:
                return Decimal(str(price))
            
            return None
        except Exception as e:
            logger.error(f"Error getting real-time price for {symbol}: {e}")
            return None
    
    async def get_historical_market_data(
        self, 
        symbol: str, 
        interval: str = "1day", 
        days: int = 30,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            interval: Time interval (e.g., "1min", "1hour", "1day")
            days: Number of days of data to get
            force_refresh: Whether to force a refresh from the API
            
        Returns:
            DataFrame containing market data
        """
        cache_key = f"market_{symbol}_{interval}_{days}"
        
        # Check cache if not forcing refresh
        if not force_refresh and cache_key in self.market_data_cache:
            # Check if cache is still valid
            if cache_key in self.cache_expiry and self.cache_expiry[cache_key] > time.time():
                logger.debug(f"Using cached market data for {symbol} ({interval})")
                return self.market_data_cache[cache_key]
        
        try:
            # Get historical data
            logger.info(f"Fetching historical market data for {symbol} ({interval})")
            data = await self.market_connector.get_time_series(symbol, interval, days)
            
            # Process data into DataFrame
            df = self._process_market_data(data, symbol)
            
            # Cache the data
            self.market_data_cache[cache_key] = df
            self.cache_expiry[cache_key] = time.time() + self.market_data_cache_expiry
            
            # Save to disk cache
            self._save_to_cache(cache_key, df)
            
            # Notify callbacks
            self._notify_callbacks("market", symbol, df)
            
            return df
        except Exception as e:
            logger.error(f"Error getting historical market data for {symbol}: {e}")
            
            # Try to load from disk cache as fallback
            df = self._load_from_cache(cache_key)
            if df is not None:
                logger.info(f"Loaded market data for {symbol} from disk cache")
                return df
            
            # Return empty DataFrame if all else fails
            return pd.DataFrame()
    
    def _process_market_data(self, market_data: Dict[str, Any], symbol: str) -> pd.DataFrame:
        """
        Process raw market data into a DataFrame.
        
        Args:
            market_data: Raw market data from Twelve Data
            symbol: Symbol the data is for
            
        Returns:
            Processed DataFrame
        """
        # Check if we have valid data
        if not market_data or "values" not in market_data:
            return pd.DataFrame()
        
        # Extract values
        values = market_data.get("values", [])
        
        # Create DataFrame
        df = pd.DataFrame(values)
        
        # Rename columns
        if not df.empty:
            # Convert types
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            # Convert datetime
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            
            # Add symbol column
            df["symbol"] = symbol
            
            # Sort by datetime
            df = df.sort_index()
        
        return df
    
    def _save_to_cache(self, key: str, df: pd.DataFrame) -> None:
        """
        Save DataFrame to disk cache.
        
        Args:
            key: Cache key
            df: DataFrame to save
        """
        try:
            if df.empty:
                return
            
            # Create file path
            file_path = os.path.join(self.cache_dir, f"{key}.csv")
            
            # Save to CSV
            df.to_csv(file_path)
            logger.debug(f"Saved {key} to disk cache")
        except Exception as e:
            logger.error(f"Error saving {key} to disk cache: {e}")
    
    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            DataFrame or None if not found
        """
        try:
            # Create file path
            file_path = os.path.join(self.cache_dir, f"{key}.csv")
            
            # Check if file exists
            if not os.path.exists(file_path):
                return None
            
            # Load from CSV
            df = pd.read_csv(file_path)
            
            # Convert datetime index
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            
            logger.debug(f"Loaded {key} from disk cache")
            return df
        except Exception as e:
            logger.error(f"Error loading {key} from disk cache: {e}")
            return None
    
    def register_callback(self, data_type: str, callback: Callable[[str, pd.DataFrame], None]) -> None:
        """
        Register a callback for data updates.
        
        Args:
            data_type: Type of data to receive updates for ("sentiment" or "market")
            callback: Callback function that takes (symbol, data)
        """
        if data_type not in self.data_callbacks:
            self.data_callbacks[data_type] = []
        
        self.data_callbacks[data_type].append(callback)
        logger.info(f"Registered callback for {data_type} data")
    
    def _notify_callbacks(self, data_type: str, symbol: str, data: pd.DataFrame) -> None:
        """
        Notify callbacks of data updates.
        
        Args:
            data_type: Type of data being updated
            symbol: Symbol the data is for
            data: Updated data
        """
        if data_type in self.data_callbacks:
            for callback in self.data_callbacks[data_type]:
                try:
                    callback(symbol, data)
                except Exception as e:
                    logger.error(f"Error in {data_type} callback: {e}")
    
    async def get_combined_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get combined market and sentiment data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            days: Number of days of data to get
            
        Returns:
            DataFrame containing combined data
        """
        try:
            # Get market data
            market_symbol = f"{symbol}/USD" if "/" not in symbol else symbol
            market_data = await self.get_historical_market_data(market_symbol, "1day", days)
            
            # Get sentiment data
            sentiment_data = await self.get_sentiment_data(symbol.split("/")[0] if "/" in symbol else symbol)
            
            # Check if we have both types of data
            if market_data.empty or sentiment_data.empty:
                logger.warning(f"Missing data for combined view of {symbol}")
                return market_data  # Return whatever we have
            
            # Resample sentiment data to daily
            sentiment_data = sentiment_data.set_index("timestamp") if "timestamp" in sentiment_data.columns else sentiment_data
            daily_sentiment = sentiment_data["compound_score"].resample("D").mean()
            
            # Combine data
            combined = market_data.copy()
            combined["sentiment"] = None
            
            # Match sentiment to market data dates
            for date in combined.index:
                date_str = date.strftime("%Y-%m-%d")
                if date_str in daily_sentiment.index:
                    combined.loc[date, "sentiment"] = daily_sentiment[date_str]
            
            # Forward fill sentiment (carry forward last known sentiment)
            combined["sentiment"] = combined["sentiment"].ffill()
            
            return combined
        except Exception as e:
            logger.error(f"Error getting combined data for {symbol}: {e}")
            return pd.DataFrame()
