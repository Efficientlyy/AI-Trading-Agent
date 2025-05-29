"""
Market Data Provider for AI Trading Agent.

This module provides data fetching capabilities from various sources including
Alpha Vantage and Yahoo Finance, with proper caching and error handling.
"""

import logging
import pandas as pd
import numpy as np
import aiohttp
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import os
from io import StringIO
import yfinance as yf

from ..common.utils import get_logger

# Create logger
logger = get_logger(__name__)

class MarketDataProvider:
    """
    Market Data Provider that fetches data from various sources.
    
    Supports multiple data sources with fallback capabilities.
    Primary sources include Alpha Vantage and Yahoo Finance.
    """
    
    def __init__(
        self,
        primary_source: str = "alpha_vantage",
        fallback_source: str = "yahoo_finance",
        cache_timeout_minutes: int = 15
    ):
        """
        Initialize the Market Data Provider.
        
        Args:
            primary_source: Primary data source (alpha_vantage or yahoo_finance)
            fallback_source: Fallback data source if primary fails
            cache_timeout_minutes: How long to cache data in minutes
        """
        self.primary_source = primary_source.lower()
        self.fallback_source = fallback_source.lower()
        self.cache_timeout_minutes = cache_timeout_minutes
        
        # Initialize cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # API keys
        self.api_keys = self._load_api_keys()
        
        # Async session
        self.session = None
        
        logger.info(f"Initialized MarketDataProvider with primary source: {primary_source}")
        
        # Validate configuration
        self._validate_configuration()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """
        Load API keys from environment variables or configuration files.
        
        Returns:
            Dictionary of API keys by provider
        """
        # Try to load from environment variables first
        api_keys = {
            "alpha_vantage": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
            "polygon": os.environ.get("POLYGON_API_KEY", ""),
            "iex": os.environ.get("IEX_API_KEY", ""),
            "finnhub": os.environ.get("FINNHUB_API_KEY", ""),
        }
        
        # Check for config file
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'api_keys.json')
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_keys = json.load(f)
                    # Update keys from config file
                    for provider, key in config_keys.items():
                        if key and provider in api_keys:
                            api_keys[provider] = key
        except Exception as e:
            logger.error(f"Error loading API keys from config file: {e}")
        
        # Log available keys (without revealing the actual keys)
        available_providers = [provider for provider, key in api_keys.items() if key]
        logger.info(f"API keys available for: {', '.join(available_providers) if available_providers else 'None'}")
        
        return api_keys
    
    def _validate_configuration(self) -> None:
        """Validate the configuration and log warnings if needed."""
        valid_sources = ["alpha_vantage", "yahoo_finance", "polygon", "iex", "finnhub"]
        
        if self.primary_source not in valid_sources:
            logger.warning(f"Invalid primary source: {self.primary_source}. Valid options are: {', '.join(valid_sources)}")
        
        if self.fallback_source not in valid_sources:
            logger.warning(f"Invalid fallback source: {self.fallback_source}. Valid options are: {', '.join(valid_sources)}")
        
        # Check API keys for sources that need them
        if self.primary_source == "alpha_vantage" and not self.api_keys.get("alpha_vantage"):
            logger.warning("Alpha Vantage selected as primary source but no API key provided")
        
        if self.primary_source == "polygon" and not self.api_keys.get("polygon"):
            logger.warning("Polygon selected as primary source but no API key provided")
            
        if self.primary_source == "iex" and not self.api_keys.get("iex"):
            logger.warning("IEX selected as primary source but no API key provided")
            
        if self.primary_source == "finnhub" and not self.api_keys.get("finnhub"):
            logger.warning("Finnhub selected as primary source but no API key provided")
    
    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is initialized."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_timestamps:
            return False
        
        # Check if cache has expired
        timestamp = self.cache_timestamps[cache_key]
        now = datetime.now()
        cache_age = (now - timestamp).total_seconds() / 60  # minutes
        
        return cache_age < self.cache_timeout_minutes
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available and valid."""
        if cache_key in self.cache and self._is_cache_valid(cache_key):
            logger.debug(f"Using cached data for {cache_key}")
            return self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: Any) -> None:
        """Cache data with current timestamp."""
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now()
        logger.debug(f"Cached data for {cache_key}")
    
    async def fetch_historical_data(
        self,
        symbols: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical market data for the specified symbols.
        
        Args:
            symbols: List of symbols to fetch data for
            start_date: Start date for historical data (defaults to 1 year ago)
            end_date: End date for historical data (defaults to today)
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data
        """
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
            
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Format dates for API calls
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        result = {}
        
        for symbol in symbols:
            # Generate cache key
            cache_key = f"{symbol}_{interval}_{start_str}_{end_str}"
            
            # Check cache
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                result[symbol] = cached_data
                continue
            
            # Try primary source
            try:
                if self.primary_source == "alpha_vantage":
                    data = await self._fetch_alpha_vantage(symbol, interval, start_date, end_date)
                elif self.primary_source == "yahoo_finance":
                    data = await self._fetch_yahoo_finance(symbol, interval, start_date, end_date)
                else:
                    logger.warning(f"Unsupported primary source: {self.primary_source}, falling back")
                    raise ValueError(f"Unsupported primary source: {self.primary_source}")
                
                if data is not None and not data.empty:
                    result[symbol] = data
                    self._cache_data(cache_key, data)
                    continue
            
            except Exception as e:
                logger.warning(f"Error fetching data from primary source for {symbol}: {e}")
            
            # Try fallback source if primary failed
            try:
                logger.info(f"Trying fallback source {self.fallback_source} for {symbol}")
                
                if self.fallback_source == "alpha_vantage":
                    data = await self._fetch_alpha_vantage(symbol, interval, start_date, end_date)
                elif self.fallback_source == "yahoo_finance":
                    data = await self._fetch_yahoo_finance(symbol, interval, start_date, end_date)
                else:
                    logger.warning(f"Unsupported fallback source: {self.fallback_source}")
                    raise ValueError(f"Unsupported fallback source: {self.fallback_source}")
                
                if data is not None and not data.empty:
                    result[symbol] = data
                    self._cache_data(cache_key, data)
                    continue
            
            except Exception as e:
                logger.error(f"Error fetching data from fallback source for {symbol}: {e}")
            
            logger.error(f"Failed to fetch data for {symbol} from all sources")
        
        return result
    
    async def _fetch_alpha_vantage(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            interval: Time interval for data
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Ensure API key is available
        api_key = self.api_keys.get("alpha_vantage")
        if not api_key:
            logger.error("Alpha Vantage API key not available")
            return None
        
        # Map intervals to Alpha Vantage format
        interval_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "60min",
            "4h": "60min",  # Alpha Vantage doesn't have 4h, use 60min and resample
            "1d": "daily",
            "1w": "weekly"
        }
        
        av_interval = interval_map.get(interval, "daily")
        
        # Determine function based on interval
        if av_interval in ["1min", "5min", "15min", "30min", "60min"]:
            function = "TIME_SERIES_INTRADAY"
            output_size = "full"
        elif av_interval == "daily":
            function = "TIME_SERIES_DAILY"
            output_size = "full"
        elif av_interval == "weekly":
            function = "TIME_SERIES_WEEKLY"
            output_size = "full"
        else:
            logger.error(f"Unsupported interval for Alpha Vantage: {interval}")
            return None
        
        # Construct URL
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": output_size,
            "datatype": "json"
        }
        
        # Add interval parameter if using intraday
        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = av_interval
        
        await self._ensure_session()
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Alpha Vantage API error: {response.status}")
                    return None
                
                data = await response.json()
                
                # Check for error messages
                if "Error Message" in data:
                    logger.error(f"Alpha Vantage error: {data['Error Message']}")
                    return None
                
                if "Note" in data:
                    logger.warning(f"Alpha Vantage API limit note: {data['Note']}")
                
                # Extract time series data
                time_series_key = None
                for key in data.keys():
                    if "Time Series" in key:
                        time_series_key = key
                        break
                
                if not time_series_key:
                    logger.error("No time series data found in Alpha Vantage response")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
                
                # Rename columns
                rename_map = {
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. volume": "volume"
                }
                df = df.rename(columns=rename_map)
                
                # Convert to numeric
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Set index to datetime
                df.index = pd.to_datetime(df.index)
                
                # Sort by date
                df = df.sort_index()
                
                # Filter by date range
                df = df.loc[start_date:end_date]
                
                # Resample if needed (for 4h from 1h)
                if interval == "4h" and av_interval == "60min":
                    df = df.resample('4H').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                
                # Add timestamp column for compatibility
                df["timestamp"] = df.index
                
                return df
        
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage: {e}")
            return None
    
    async def _fetch_yahoo_finance(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            interval: Time interval for data
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Map intervals to Yahoo Finance format
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",  # Yahoo Finance doesn't directly support 4h
            "1d": "1d",
            "1w": "1wk"
        }
        
        yf_interval = interval_map.get(interval, "1d")
        
        # Fetch data using yfinance (synchronously as yfinance doesn't support async)
        try:
            # Need to run in a separate thread as yfinance is synchronous
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    symbol,
                    start=start_date,
                    end=end_date + timedelta(days=1),  # Add a day to include end_date
                    interval=yf_interval,
                    progress=False
                )
            )
            
            if data.empty:
                logger.warning(f"No data returned from Yahoo Finance for {symbol}")
                return None
            
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure all expected columns exist
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in data.columns:
                    logger.error(f"Missing column {col} in Yahoo Finance data")
                    return None
            
            # Add timestamp column for compatibility
            data["timestamp"] = data.index
            
            # Resample if needed (for 4h)
            if interval == "4h" and yf_interval != "4h":
                # Use 1h data and resample
                data = data.resample('4H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'timestamp': 'first'
                }).dropna()
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            return None
    
    async def fetch_latest_data(self, symbols: List[str]) -> Dict[str, pd.Series]:
        """
        Fetch the latest data for the specified symbols.
        
        Args:
            symbols: List of symbols to fetch data for
            
        Returns:
            Dictionary mapping symbols to Series with the latest data
        """
        result = {}
        
        # Fetch last day of data and extract the most recent entry
        historical_data = await self.fetch_historical_data(
            symbols,
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now(),
            interval="1d"
        )
        
        for symbol, data in historical_data.items():
            if not data.empty:
                result[symbol] = data.iloc[-1]
        
        return result
    
    async def fetch_data_since(
        self,
        symbols: List[str],
        latest_timestamps: Dict[str, datetime],
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data since the latest timestamps for each symbol.
        
        Args:
            symbols: List of symbols to fetch data for
            latest_timestamps: Dictionary mapping symbols to their latest timestamps
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to DataFrames with the new data
        """
        result = {}
        
        for symbol in symbols:
            start_date = latest_timestamps.get(symbol)
            
            # If no latest timestamp, use a default (e.g., 1 week ago)
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7)
            else:
                # Add a small buffer to avoid duplicates
                start_date = start_date + timedelta(minutes=1)
            
            # Fetch data
            data = await self.fetch_historical_data(
                [symbol],
                start_date=start_date,
                end_date=datetime.now(),
                interval=interval
            )
            
            if symbol in data and not data[symbol].empty:
                result[symbol] = data[symbol]
        
        return result
