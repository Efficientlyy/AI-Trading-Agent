"""
Sentiment Data Collector and Storage System.

This module provides functionality to collect, store, and retrieve historical sentiment data
from various sources including Fear & Greed Index, news sentiment, social media sentiment,
and on-chain metrics. The data is stored in a structured format that can be easily
accessed for backtesting sentiment-based trading strategies.
"""

import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
from asyncio import Task

from src.analysis_agents.sentiment.market_sentiment import FearGreedClient
from src.analysis_agents.sentiment.news_sentiment import NewsSentimentAgent
from src.analysis_agents.sentiment.social_media_sentiment import SocialMediaSentimentAgent
from src.analysis_agents.sentiment.onchain_sentiment import OnchainSentimentAgent
from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class SentimentCollector:
    """
    Collects and stores historical sentiment data from various sources.
    
    This class provides functionality to:
    1. Collect sentiment data from multiple sources
    2. Store sentiment data in a structured format (CSV/JSON)
    3. Load historical sentiment data for backtesting
    4. Provide timestamp-aligned sentiment data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the SentimentCollector.
        
        Args:
            config: Configuration dictionary (loads from config if not provided)
        """
        self.config = config or ConfigManager().get_config("sentiment_analysis")
        self.data_dir = Path(self.config.get("historical_data_path", "data/historical/sentiment"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentiment sources based on config
        self.sources = {}
        self.collection_tasks = {}
        self._initialize_sources()
        
    def _initialize_sources(self):
        """Initialize sentiment data sources based on configuration."""
        # Initialize Fear & Greed Index client if enabled
        if self.config.get("fear_greed", {}).get("enabled", False):
            self.sources["fear_greed"] = FearGreedClient(self.config.get("fear_greed", {}))
        
        # Initialize News Sentiment if enabled
        if self.config.get("news", {}).get("enabled", False):
            self.sources["news"] = NewsSentimentAgent(self.config.get("news", {}))
        
        # Initialize Social Media Sentiment if enabled
        if self.config.get("social_media", {}).get("enabled", False):
            self.sources["social_media"] = SocialMediaSentimentAgent(self.config.get("social_media", {}))
        
        # Initialize On-chain Sentiment if enabled
        if self.config.get("onchain", {}).get("enabled", False):
            self.sources["onchain"] = OnchainSentimentAgent(self.config.get("onchain", {}))
    
    async def collect_historical_data(self, 
                                     source: str, 
                                     symbol: str, 
                                     start_date: datetime.datetime,
                                     end_date: Optional[datetime.datetime] = None,
                                     save: bool = True) -> Dict[str, Any]:
        """
        Collect historical sentiment data for a specific source and symbol.
        
        Args:
            source: Sentiment source ("fear_greed", "news", "social_media", "onchain")
            symbol: Trading symbol to collect data for (e.g., "BTC", "ETH")
            start_date: Start date for historical data collection
            end_date: End date for historical data collection (defaults to now)
            save: Whether to save the collected data to disk
            
        Returns:
            Dictionary of collected sentiment data
        """
        if source not in self.sources:
            raise ValueError(f"Source {source} not initialized or not supported")
        
        end_date = end_date or datetime.datetime.now()
        
        # Collect data from the source
        source_client = self.sources[source]
        
        if source == "fear_greed":
            # Fear & Greed has its own historical data method
            data = await source_client.get_historical_data(start_date, end_date)
        elif hasattr(source_client, "get_historical_sentiment"):
            # Use specific historical method if available
            data = await source_client.get_historical_sentiment(symbol, start_date, end_date)
        else:
            raise NotImplementedError(f"Historical data collection not implemented for {source}")
        
        # Format the data
        formatted_data = self._format_sentiment_data(data, source, symbol)
        
        # Save to disk if requested
        if save:
            self._save_data(formatted_data, source, symbol, start_date, end_date)
        
        return formatted_data
    
    def _format_sentiment_data(self, 
                              data: Union[Dict, List], 
                              source: str, 
                              symbol: str) -> Dict[str, Any]:
        """
        Format sentiment data into a standardized structure.
        
        Args:
            data: Raw sentiment data
            source: Data source name
            symbol: Trading symbol
            
        Returns:
            Formatted sentiment data
        """
        # Standard sentiment data format
        formatted = {
            "source": source,
            "symbol": symbol,
            "timestamp": [],
            "sentiment_value": [],
            "metadata": {},
        }
        
        # Source-specific formatting
        if source == "fear_greed":
            for item in data:
                formatted["timestamp"].append(item.get("timestamp"))
                formatted["sentiment_value"].append(item.get("value") / 100)  # Normalize to 0-1
            formatted["metadata"] = {"classification": [item.get("classification") for item in data]}
        
        elif source == "news":
            # Assuming news sentiment returns data with timestamp and sentiment_value
            for item in data:
                formatted["timestamp"].append(item.get("timestamp"))
                formatted["sentiment_value"].append(item.get("sentiment_value"))
            # Add any news-specific metadata
            if "category" in data[0]:
                formatted["metadata"]["category"] = [item.get("category") for item in data]
            if "headline" in data[0]:
                formatted["metadata"]["headline"] = [item.get("headline") for item in data]
                
        elif source == "social_media":
            # Format social media sentiment data
            for item in data:
                formatted["timestamp"].append(item.get("timestamp"))
                formatted["sentiment_value"].append(item.get("sentiment_value"))
            # Add any social media specific metadata
            if "platform" in data[0]:
                formatted["metadata"]["platform"] = [item.get("platform") for item in data]
            if "volume" in data[0]:
                formatted["metadata"]["volume"] = [item.get("volume") for item in data]
                
        elif source == "onchain":
            # Format on-chain sentiment data
            for item in data:
                formatted["timestamp"].append(item.get("timestamp"))
                formatted["sentiment_value"].append(item.get("sentiment_value"))
            # Add any on-chain specific metadata
            if "metric" in data[0]:
                formatted["metadata"]["metric"] = [item.get("metric") for item in data]
        
        return formatted
    
    def _save_data(self, 
                  data: Dict[str, Any], 
                  source: str, 
                  symbol: str, 
                  start_date: datetime.datetime,
                  end_date: datetime.datetime):
        """
        Save sentiment data to disk.
        
        Args:
            data: Formatted sentiment data
            source: Data source name
            symbol: Trading symbol
            start_date: Start date of the data
            end_date: End date of the data
        """
        # Create a filename based on parameters
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        filename = f"{symbol}_{source}_{start_str}_{end_str}.json"
        
        # Ensure directory exists
        source_dir = self.data_dir / source
        source_dir.mkdir(exist_ok=True)
        
        # Save data to JSON
        with open(source_dir / filename, 'w') as f:
            json.dump(data, f, default=str)
        
        logger.info(f"Saved {source} sentiment data for {symbol} to {source_dir / filename}")
    
    def start_collection(self, 
                        source: str, 
                        symbol: str, 
                        interval_minutes: int = 60):
        """
        Start collecting sentiment data at regular intervals.
        
        Args:
            source: Sentiment source to collect from
            symbol: Trading symbol to collect data for
            interval_minutes: Collection interval in minutes
        """
        if source not in self.sources:
            raise ValueError(f"Source {source} not initialized or not supported")
        
        # Create a unique key for this collection task
        task_key = f"{source}_{symbol}_{interval_minutes}"
        
        # Check if already collecting
        if task_key in self.collection_tasks and not self.collection_tasks[task_key].done():
            logger.warning(f"Already collecting {source} data for {symbol}")
            return
        
        # Start collection task
        self.collection_tasks[task_key] = asyncio.create_task(
            self._collection_loop(source, symbol, interval_minutes)
        )
        
        logger.info(f"Started {source} data collection for {symbol} every {interval_minutes} minutes")
    
    def stop_collection(self, source: str, symbol: str, interval_minutes: int = 60):
        """
        Stop collecting sentiment data.
        
        Args:
            source: Sentiment source to stop collecting from
            symbol: Trading symbol to stop collecting for
            interval_minutes: Collection interval that was used
        """
        task_key = f"{source}_{symbol}_{interval_minutes}"
        
        if task_key in self.collection_tasks and not self.collection_tasks[task_key].done():
            self.collection_tasks[task_key].cancel()
            logger.info(f"Stopped {source} data collection for {symbol}")
    
    async def _collection_loop(self, source: str, symbol: str, interval_minutes: int):
        """
        Collection loop that runs at regular intervals.
        
        Args:
            source: Sentiment source to collect from
            symbol: Trading symbol to collect data for
            interval_minutes: Collection interval in minutes
        """
        try:
            while True:
                # Get current timestamp
                now = datetime.datetime.now()
                
                # Collect sentiment data
                source_client = self.sources[source]
                
                if source == "fear_greed":
                    # Fear & Greed data
                    data = await source_client.get_data()
                    data = [data]  # Wrap in list for consistent formatting
                elif hasattr(source_client, "get_sentiment"):
                    # Use standard sentiment method
                    data = await source_client.get_sentiment(symbol)
                    if not isinstance(data, list):
                        data = [data]
                else:
                    logger.error(f"Cannot collect data from {source}, no suitable method found")
                    break
                
                # Format and save data
                formatted_data = self._format_sentiment_data(data, source, symbol)
                today = now.strftime("%Y%m%d")
                self._save_data(
                    formatted_data, 
                    source, 
                    symbol, 
                    now.replace(hour=0, minute=0, second=0, microsecond=0),
                    now
                )
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
        except asyncio.CancelledError:
            logger.info(f"Cancelled {source} data collection for {symbol}")
        except Exception as e:
            logger.error(f"Error in {source} data collection for {symbol}: {e}")
    
    def load_historical_data(self, 
                            source: str, 
                            symbol: str, 
                            start_date: datetime.datetime,
                            end_date: Optional[datetime.datetime] = None) -> pd.DataFrame:
        """
        Load historical sentiment data from disk.
        
        Args:
            source: Sentiment source to load data for
            symbol: Trading symbol to load data for
            start_date: Start date for the data
            end_date: End date for the data (defaults to now)
            
        Returns:
            DataFrame containing the sentiment data
        """
        end_date = end_date or datetime.datetime.now()
        
        # Create source directory path
        source_dir = self.data_dir / source
        
        if not source_dir.exists():
            raise FileNotFoundError(f"No data directory found for {source}")
        
        # Find all files matching the source and symbol
        pattern = f"{symbol}_{source}_*.json"
        matching_files = list(source_dir.glob(pattern))
        
        if not matching_files:
            raise FileNotFoundError(f"No data files found for {source} and {symbol}")
        
        # Filter files by date range
        data_frames = []
        for file_path in matching_files:
            # Extract dates from filename
            filename = file_path.name
            try:
                file_start_str = filename.split('_')[-2]
                file_end_str = filename.split('_')[-1].replace('.json', '')
                
                file_start = datetime.datetime.strptime(file_start_str, "%Y%m%d")
                file_end = datetime.datetime.strptime(file_end_str, "%Y%m%d")
                
                # Check if file date range overlaps with requested date range
                if file_end >= start_date and file_start <= end_date:
                    # Load and parse the file
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame({
                        'timestamp': pd.to_datetime(data['timestamp']),
                        'sentiment_value': data['sentiment_value'],
                        'source': source,
                        'symbol': symbol
                    })
                    
                    # Add metadata columns if available
                    for meta_key, meta_values in data.get('metadata', {}).items():
                        if len(meta_values) == len(df):
                            df[meta_key] = meta_values
                    
                    # Filter to requested date range
                    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                    
                    data_frames.append(df)
            except Exception as e:
                logger.warning(f"Error parsing file {filename}: {e}")
        
        if not data_frames:
            raise ValueError(f"No data found for {source} and {symbol} in the specified date range")
        
        # Combine all data frames and sort by timestamp
        combined_df = pd.concat(data_frames)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        return combined_df
    
    def combine_sentiment_sources(self, 
                                  symbol: str, 
                                  start_date: datetime.datetime,
                                  end_date: Optional[datetime.datetime] = None,
                                  sources: Optional[List[str]] = None,
                                  resample_freq: str = '1H') -> pd.DataFrame:
        """
        Combine multiple sentiment sources into a single DataFrame.
        
        Args:
            symbol: Trading symbol to load data for
            start_date: Start date for the data
            end_date: End date for the data (defaults to now)
            sources: List of sources to combine (defaults to all available sources)
            resample_freq: Frequency to resample the data to (e.g. '1H', '15min')
            
        Returns:
            DataFrame containing combined sentiment data
        """
        end_date = end_date or datetime.datetime.now()
        sources = sources or list(self.sources.keys())
        
        # Load data for each source
        dfs = []
        for source in sources:
            try:
                df = self.load_historical_data(source, symbol, start_date, end_date)
                dfs.append(df)
            except FileNotFoundError:
                logger.warning(f"No data found for {source} and {symbol}")
        
        if not dfs:
            raise ValueError(f"No data found for {symbol} in any of the specified sources")
        
        # Combine all dataframes
        combined = pd.concat(dfs)
        
        # Create a pivot table with sources as columns
        pivot = combined.pivot_table(
            index='timestamp', 
            columns='source', 
            values='sentiment_value',
            aggfunc='mean'
        )
        
        # Resample to consistent frequency
        resampled = pivot.resample(resample_freq).mean()
        
        # Forward fill missing values (carries forward last known sentiment)
        filled = resampled.fillna(method='ffill')
        
        # Add a combined sentiment column (simple average)
        filled['combined'] = filled.mean(axis=1)
        
        return filled

async def main():
    """Test the SentimentCollector."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize collector
    collector = SentimentCollector()
    
    # Test historical data collection
    start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    try:
        fear_greed_data = await collector.collect_historical_data(
            "fear_greed", "BTC", start_date
        )
        print(f"Collected Fear & Greed data: {len(fear_greed_data['timestamp'])} data points")
    except Exception as e:
        print(f"Error collecting Fear & Greed data: {e}")
    
    # Test loading historical data
    try:
        df = collector.load_historical_data("fear_greed", "BTC", start_date)
        print(f"Loaded Fear & Greed data: {len(df)} rows")
        print(df.head())
    except Exception as e:
        print(f"Error loading Fear & Greed data: {e}")

if __name__ == "__main__":
    asyncio.run(main())