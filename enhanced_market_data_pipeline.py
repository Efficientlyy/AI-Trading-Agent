#!/usr/bin/env python
"""
Enhanced Market Data Pipeline with Robust Error Handling

This module enhances the market data pipeline with improved error handling,
standardized symbol formats, and fallback mechanisms for API failures.
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from multi_asset_data_service import MultiAssetDataService
from symbol_standardization import SymbolStandardizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("market_data_pipeline.log")
    ]
)

logger = logging.getLogger("market_data_pipeline")

class EnhancedMarketDataPipeline:
    """Enhanced market data pipeline with robust error handling and fallbacks"""
    
    def __init__(self, symbols=None, timeframes=None, cache_dir=None, mock_data_dir=None):
        """Initialize enhanced market data pipeline
        
        Args:
            symbols: List of symbols to track (default: ["BTC/USDC", "ETH/USDC", "SOL/USDC"])
            timeframes: List of timeframes to support (default: ["1m", "5m", "15m", "1h", "4h", "1d"])
            cache_dir: Directory for caching data (default: "./cache")
            mock_data_dir: Directory for mock data (default: "./test_data")
        """
        # Initialize symbol standardizer
        self.standardizer = SymbolStandardizer()
        
        # Standardize input symbols to internal format
        self.symbols = symbols or ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        self.symbols = [self.standardizer.for_internal(s) for s in self.symbols]
        
        # Initialize timeframes
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Initialize data service with standardized symbols
        # Fix: Pass timeframes as a separate parameter, not as part of symbols
        self.data_service = MultiAssetDataService(supported_assets=self.symbols)
        
        # Set up caching
        self.cache_dir = cache_dir or "./cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up mock data
        self.mock_data_dir = mock_data_dir or "./test_data"
        os.makedirs(self.mock_data_dir, exist_ok=True)
        
        # Initialize data stores with TTL cache
        self.market_data_cache = {}
        self.cache_ttl = {}
        self.default_ttl = 60  # seconds
        
        # Initialize status tracking
        self.last_update = {}
        self.error_counts = {}
        self.max_errors = 3
        self.status = "initialized"
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Enhanced market data pipeline initialized with {len(self.symbols)} symbols")
    
    def get_market_data(self, symbol, timeframe="5m", limit=100, use_cache=True, fallback_to_mock=True):
        """Get market data with enhanced error handling and fallbacks
        
        Args:
            symbol: Symbol to get data for (any format)
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to return
            use_cache: Whether to use cached data if available
            fallback_to_mock: Whether to fall back to mock data if live data fails
            
        Returns:
            list: Market data as list of candles
        """
        # Standardize symbol to internal format
        internal_symbol = self.standardizer.for_internal(symbol)
        
        # Check if symbol is supported
        if internal_symbol not in self.symbols:
            logger.warning(f"Symbol not supported: {symbol} (internal: {internal_symbol})")
            return []
        
        # Generate cache key
        cache_key = f"{internal_symbol}_{timeframe}_{limit}"
        
        # Check cache if enabled
        if use_cache and cache_key in self.market_data_cache:
            # Check if cache is still valid
            if time.time() < self.cache_ttl.get(cache_key, 0):
                logger.debug(f"Using cached data for {symbol} {timeframe}")
                return self.market_data_cache[cache_key]
        
        # Try to get live data
        try:
            # Convert to API format for data service
            api_symbol = self.standardizer.for_api(internal_symbol)
            
            # Get data from service
            with self.lock:
                self.status = "fetching"
                data = self.data_service.get_klines(api_symbol, timeframe, limit)
            
            # Validate data
            if data and len(data) > 0:
                # Update cache
                self.market_data_cache[cache_key] = data
                self.cache_ttl[cache_key] = time.time() + self.default_ttl
                self.last_update[cache_key] = time.time()
                self.error_counts[cache_key] = 0
                self.status = "success"
                
                logger.info(f"Successfully fetched {len(data)} candles for {symbol} {timeframe}")
                return data
            else:
                logger.warning(f"Empty data returned for {symbol} {timeframe}")
                # Increment error count
                self.error_counts[cache_key] = self.error_counts.get(cache_key, 0) + 1
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {str(e)}")
            # Increment error count
            self.error_counts[cache_key] = self.error_counts.get(cache_key, 0) + 1
        
        # If we reach here, live data failed
        self.status = "error"
        
        # Check if we should fall back to mock data
        if fallback_to_mock:
            logger.info(f"Falling back to mock data for {symbol} {timeframe}")
            mock_data = self._get_mock_data(internal_symbol, timeframe, limit)
            if mock_data:
                logger.info(f"Using mock data for {symbol} {timeframe}")
                return mock_data
        
        # If we still have cached data, return it even if expired
        if cache_key in self.market_data_cache:
            logger.warning(f"Using expired cached data for {symbol} {timeframe}")
            return self.market_data_cache[cache_key]
        
        # If all else fails, return empty list
        logger.error(f"No data available for {symbol} {timeframe}")
        return []
    
    def _get_mock_data(self, symbol, timeframe, limit):
        """Get mock data for testing
        
        Args:
            symbol: Symbol to get data for (internal format)
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to return
            
        Returns:
            list: Mock market data
        """
        # Generate mock data filename
        api_symbol = self.standardizer.for_api(symbol)
        filename = f"{api_symbol}_{timeframe}_mock.json"
        filepath = os.path.join(self.mock_data_dir, filename)
        
        # Check if mock data file exists
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Return only the requested number of candles
                return data[-limit:] if len(data) > limit else data
            except Exception as e:
                logger.error(f"Error loading mock data: {str(e)}")
                return None
        else:
            # If no mock data file exists, generate one
            logger.info(f"Generating mock data for {symbol} {timeframe}")
            mock_data = self._generate_mock_data(symbol, timeframe, 1000)  # Generate extra for future use
            
            # Save mock data
            try:
                with open(filepath, 'w') as f:
                    json.dump(mock_data, f)
                
                logger.info(f"Mock data saved to {filepath}")
                
                # Return only the requested number of candles
                return mock_data[-limit:] if len(mock_data) > limit else mock_data
            except Exception as e:
                logger.error(f"Error saving mock data: {str(e)}")
                return mock_data[-limit:] if len(mock_data) > limit else mock_data
    
    def _generate_mock_data(self, symbol, timeframe, count):
        """Generate mock market data
        
        Args:
            symbol: Symbol to generate data for
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            count: Number of candles to generate
            
        Returns:
            list: Generated mock market data
        """
        # Parse timeframe to get interval in minutes
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            minutes = int(timeframe[:-1]) * 60 * 24
        else:
            minutes = 5  # Default to 5m
        
        # Generate mock data
        mock_data = []
        
        # Start time (now - count * interval)
        start_time = datetime.now() - timedelta(minutes=count * minutes)
        
        # Initial price based on symbol
        if "BTC" in symbol:
            price = 50000 + (hash(symbol) % 10000)  # Random starting price around 50k
        elif "ETH" in symbol:
            price = 3000 + (hash(symbol) % 1000)  # Random starting price around 3k
        else:
            price = 100 + (hash(symbol) % 100)  # Random starting price around 100
        
        # Generate candles
        for i in range(count):
            # Calculate candle time
            candle_time = start_time + timedelta(minutes=i * minutes)
            timestamp = int(candle_time.timestamp() * 1000)
            
            # Generate price movement (random walk with drift)
            price_change = (hash(f"{symbol}_{i}") % 200 - 100) / 1000 * price
            price += price_change
            
            # Generate candle
            open_price = price
            high_price = open_price * (1 + (hash(f"{symbol}_high_{i}") % 100) / 10000)
            low_price = open_price * (1 - (hash(f"{symbol}_low_{i}") % 100) / 10000)
            close_price = price * (1 + (hash(f"{symbol}_close_{i}") % 200 - 100) / 10000)
            
            # Ensure high is highest and low is lowest
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            # Generate volume
            volume = (hash(f"{symbol}_vol_{i}") % 1000) / 10
            
            # Create candle
            candle = {
                "time": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "close_time": timestamp + minutes * 60 * 1000,
                "quote_volume": volume * close_price,
                "trades": hash(f"{symbol}_trades_{i}") % 100 + 10
            }
            
            mock_data.append(candle)
            
            # Update price for next candle
            price = close_price
        
        return mock_data
    
    def get_status(self):
        """Get pipeline status
        
        Returns:
            dict: Pipeline status
        """
        return {
            "status": self.status,
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "last_update": self.last_update,
            "error_counts": self.error_counts
        }
    
    def clear_cache(self):
        """Clear data cache"""
        with self.lock:
            self.market_data_cache = {}
            self.cache_ttl = {}
        
        logger.info("Cache cleared")

# Example usage
if __name__ == "__main__":
    pipeline = EnhancedMarketDataPipeline()
    
    # Test with different symbol formats
    symbols = ["BTC/USDC", "BTCUSDC", "ETH-USDC"]
    
    for symbol in symbols:
        print(f"Getting data for {symbol}...")
        data = pipeline.get_market_data(symbol)
        print(f"Got {len(data)} candles")
        
        # Print first and last candle
        if data:
            print(f"First candle: {data[0]}")
            print(f"Last candle: {data[-1]}")
        
        print("---")
