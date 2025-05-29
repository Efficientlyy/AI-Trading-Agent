"""
Market Data Provider

This module provides a unified interface for market data acquisition from different
sources, with MEXC as the primary exchange. It manages the connection between 
data sources and the technical analysis agent.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta

from ..common import get_logger
from ..config.mexc_config import MEXC_CONFIG, TRADING_PAIRS
from .mexc_trading_connector import MexcTradingConnector
from .mexc_spot_v3_client import MexcSpotV3Client

logger = get_logger(__name__)

class MarketDataProvider:
    """
    Unified market data provider that connects to MEXC and other data sources.
    
    This class provides market data to the technical analysis agent, handling
    the acquisition, transformation, and caching of data from multiple sources.
    
    Attributes:
        mexc_connector (MexcTradingConnector): MEXC trading connector
        data_cache (Dict): Cache of market data
        is_initialized (bool): Whether the provider is initialized
    """
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize the market data provider.
        
        Args:
            symbols (List[str], optional): Symbols to provide data for. Defaults to None.
        """
        self.symbols = symbols or TRADING_PAIRS
        self.mexc_connector = MexcTradingConnector(symbols=self.symbols)
        self.data_cache = {}
        self.is_initialized = False
        self.last_update_time = {}
        
        logger.info(f"Initialized MarketDataProvider for symbols: {self.symbols}")
    
    async def initialize(self):
        """Initialize the market data provider and establish connections."""
        if self.is_initialized:
            return
        
        try:
            # Initialize MEXC connection and subscribe to data
            await self.mexc_connector.subscribe_to_klines(self.symbols, ["1m", "5m", "15m", "1h", "4h", "1d"])
            await self.mexc_connector.subscribe_to_tickers(self.symbols)
            
            # Register cache update callback
            self.mexc_connector.register_kline_callback(self._update_kline_cache)
            
            self.is_initialized = True
            logger.info("MarketDataProvider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MarketDataProvider: {e}")
            raise
    
    async def _update_kline_cache(self, symbol: str, interval: str, data: Dict[str, Any]):
        """
        Update the kline cache with new data.
        
        Args:
            symbol (str): Symbol of the kline data
            interval (str): Interval of the kline data
            data (Dict[str, Any]): Kline data
        """
        cache_key = f"{symbol}_{interval}"
        
        if cache_key not in self.data_cache:
            self.data_cache[cache_key] = []
        
        # Convert kline data to OHLCV format
        try:
            kline = [
                data["t"],  # Open time
                float(data["o"]),  # Open
                float(data["h"]),  # High
                float(data["l"]),  # Low
                float(data["c"]),  # Close
                float(data["v"]),  # Volume
            ]
            
            # Add or update the kline in the cache
            self.data_cache[cache_key].append(kline)
            
            # Keep only the last 1000 klines
            self.data_cache[cache_key] = self.data_cache[cache_key][-1000:]
            
            # Update last update time
            self.last_update_time[cache_key] = datetime.now()
            
            logger.debug(f"Updated kline cache for {cache_key}")
        except Exception as e:
            logger.error(f"Error updating kline cache for {cache_key}: {e}")
    
    async def get_market_data(
        self,
        symbols: Optional[List[str]] = None,
        interval: str = "1h",
        limit: int = 100,
        include_current: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get market data for the specified symbols and interval.
        
        Args:
            symbols (List[str], optional): Symbols to get data for. Defaults to None.
            interval (str, optional): Kline interval. Defaults to "1h".
            limit (int, optional): Number of klines to get. Defaults to 100.
            include_current (bool, optional): Whether to include the current kline. Defaults to True.
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to DataFrames of market data
        """
        if not self.is_initialized:
            await self.initialize()
        
        symbols = symbols or self.symbols
        result = {}
        
        for symbol in symbols:
            try:
                # Try to get data from cache first
                cache_key = f"{symbol}_{interval}"
                cached_data = self.data_cache.get(cache_key, [])
                
                # If we have enough cached data, use it
                if len(cached_data) >= limit:
                    df = self._convert_klines_to_dataframe(cached_data[-limit:])
                    result[symbol] = df
                    continue
                
                # Otherwise, fetch from API
                klines = await self.mexc_connector.get_klines(symbol, interval, limit)
                
                # Convert to DataFrame
                df = self._convert_klines_to_dataframe(klines)
                
                # If the current kline should be excluded, drop the last row
                if not include_current and not df.empty:
                    df = df.iloc[:-1]
                
                result[symbol] = df
                
                logger.debug(f"Got {len(df)} klines for {symbol} ({interval})")
            except Exception as e:
                logger.error(f"Error getting market data for {symbol} ({interval}): {e}")
        
        return result
    
    def _convert_klines_to_dataframe(self, klines: List[List[Any]]) -> pd.DataFrame:
        """
        Convert klines list to a pandas DataFrame.
        
        Args:
            klines (List[List[Any]]): List of klines
            
        Returns:
            pd.DataFrame: DataFrame of klines
        """
        if not klines:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Set timestamp as index
        df.set_index("timestamp", inplace=True)
        
        return df
    
    async def get_current_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get current prices for the specified symbols.
        
        Args:
            symbols (List[str], optional): Symbols to get prices for. Defaults to None.
            
        Returns:
            Dict[str, float]: Dictionary mapping symbols to current prices
        """
        if not self.is_initialized:
            await self.initialize()
        
        symbols = symbols or self.symbols
        result = {}
        
        for symbol in symbols:
            try:
                ticker = await self.mexc_connector.get_ticker(symbol)
                
                # Extract the price
                if "price" in ticker:
                    result[symbol] = float(ticker["price"])
                elif "lastPrice" in ticker:
                    result[symbol] = float(ticker["lastPrice"])
                elif "close" in ticker:
                    result[symbol] = float(ticker["close"])
                else:
                    logger.warning(f"Could not extract price from ticker for {symbol}")
            except Exception as e:
                logger.error(f"Error getting current price for {symbol}: {e}")
        
        return result
    
    async def get_orderbook(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol (str): Symbol to get order book for
            limit (int, optional): Limit of results. Defaults to 10.
            
        Returns:
            Dict[str, Any]: Order book data
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.mexc_connector.get_orderbook(symbol, limit)
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {}
    
    async def close(self):
        """Close all connections and resources."""
        try:
            await self.mexc_connector.close()
            logger.info("MarketDataProvider closed")
        except Exception as e:
            logger.error(f"Error closing MarketDataProvider: {e}")

# Singleton instance
_market_data_provider = None

def get_market_data_provider(symbols: Optional[List[str]] = None) -> MarketDataProvider:
    """
    Get the singleton instance of the market data provider.
    
    Args:
        symbols (List[str], optional): Symbols to provide data for. Defaults to None.
        
    Returns:
        MarketDataProvider: Market data provider instance
    """
    global _market_data_provider
    
    if _market_data_provider is None:
        _market_data_provider = MarketDataProvider(symbols)
    
    return _market_data_provider
