"""
Binance API client for cryptocurrency data.

This module provides a client for the Binance API to fetch real-time
and historical cryptocurrency prices with proper error handling and rate limiting.
"""

import os
import time
import logging
import requests
import json
from typing import Dict, Optional, Union, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BinanceClient:
    """Client for the Binance API to fetch cryptocurrency data."""
    
    # API endpoints
    BASE_URL = "https://api.binance.com"
    TICKER_PRICE_ENDPOINT = "/api/v3/ticker/price"
    KLINES_ENDPOINT = "/api/v3/klines"
    
    # Symbol mapping for Binance (which uses different formats)
    SYMBOL_MAPPING = {
        'BTC/USD': 'BTCUSDT',
        'ETH/USD': 'ETHUSDT',
        'BNB/USD': 'BNBUSDT',
        'XRP/USD': 'XRPUSDT',
        'ADA/USD': 'ADAUSDT',
        'SOL/USD': 'SOLUSDT',
        'DOGE/USD': 'DOGEUSDT',
        'DOT/USD': 'DOTUSDT',
        'AVAX/USD': 'AVAXUSDT',
        'SHIB/USD': 'SHIBUSDT'
    }
    
    def __init__(self):
        """Initialize the Binance client."""
        self.session = requests.Session()
        self.last_request_time = 0
        self.rate_limit = 1.0  # Max 1 request per second for public endpoints
        self.request_timeout = 10  # 10 seconds timeout
        
        # Initialize cache
        self.price_cache = {}
        self.cache_ttl = 30  # 30 seconds TTL
        self.cache_timestamps = {}
        
        logger.info("Initialized Binance client")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting to avoid being throttled by Binance."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # If less than the minimum interval has passed, sleep
        if elapsed < self.rate_limit and self.last_request_time > 0:
            sleep_time = self.rate_limit - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _map_symbol(self, symbol: str) -> str:
        """Map standard symbol format to Binance format."""
        return self.SYMBOL_MAPPING.get(symbol, symbol.replace('/', ''))
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached price for a symbol is still valid."""
        if symbol not in self.cache_timestamps:
            return False
        
        return (time.time() - self.cache_timestamps[symbol]) < self.cache_ttl
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol from Binance.
        
        Args:
            symbol: The symbol to get price for (e.g., 'BTC/USD')
            
        Returns:
            Current price or None if unavailable
        """
        # Check cache first
        if symbol in self.price_cache and self._is_cache_valid(symbol):
            logger.debug(f"Using cached Binance price for {symbol}: ${self.price_cache[symbol]:.2f}")
            return self.price_cache[symbol]
        
        # Map symbol to Binance format
        binance_symbol = self._map_symbol(symbol)
        
        # Enforce rate limit
        self._enforce_rate_limit()
        
        try:
            # Make request to Binance API
            url = f"{self.BASE_URL}{self.TICKER_PRICE_ENDPOINT}"
            params = {"symbol": binance_symbol}
            
            logger.debug(f"Fetching price from Binance for {binance_symbol}")
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                if "price" in data:
                    price = float(data["price"])
                    logger.debug(f"Binance response for {symbol}: ${price:.2f}")
                    
                    # Update cache
                    self.price_cache[symbol] = price
                    self.cache_timestamps[symbol] = time.time()
                    
                    return price
                else:
                    logger.warning(f"No price data in Binance response for {symbol}: {data}")
            else:
                logger.warning(f"Binance API error ({response.status_code}): {response.text}")
        
        except requests.RequestException as e:
            logger.warning(f"Binance request error for {symbol}: {e}")
        except ValueError as e:
            logger.warning(f"Error parsing Binance response for {symbol}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error getting Binance price for {symbol}: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30, interval: str = 'daily') -> Optional[Any]:
        """
        Get historical data for a symbol from Binance.
        
        Args:
            symbol: The symbol to get data for (e.g., 'BTC/USD')
            days: Number of days of historical data
            interval: Data interval ('daily', 'hourly', etc.)
            
        Returns:
            Pandas DataFrame with historical data or None if unavailable
        """
        try:
            import pandas as pd
            
            # Map symbol to Binance format
            binance_symbol = self._map_symbol(symbol)
            
            # Map interval to Binance format
            interval_map = {
                'daily': '1d',
                'hourly': '1h',
                '15min': '15m',
                '1min': '1m'
            }
            binance_interval = interval_map.get(interval, '1d')
            
            # Calculate start time
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            start_time = end_time - (days * 24 * 60 * 60 * 1000)  # days back in milliseconds
            
            # Enforce rate limit
            self._enforce_rate_limit()
            
            # Make request to Binance API
            url = f"{self.BASE_URL}{self.KLINES_ENDPOINT}"
            params = {
                "symbol": binance_symbol,
                "interval": binance_interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": 1000  # Maximum allowed
            }
            
            logger.debug(f"Fetching historical data from Binance for {binance_symbol}")
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Parse Binance kline data
                # Columns: [0] Open time, [1] Open, [2] High, [3] Low, [4] Close, [5] Volume
                if data and isinstance(data, list):
                    df = pd.DataFrame(data, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'trades', 
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    
                    # Convert timestamps to datetime
                    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
                    
                    # Convert price columns to float
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    # Rename columns to match expected format
                    df = df[['date', 'close', 'volume']].rename(columns={'close': 'price'})
                    
                    logger.info(f"Got {len(df)} historical data points from Binance for {symbol}")
                    return df
                else:
                    logger.warning(f"No historical data in Binance response for {symbol}")
            else:
                logger.warning(f"Binance API error ({response.status_code}): {response.text}")
        
        except requests.RequestException as e:
            logger.warning(f"Binance request error for historical data: {e}")
        except ValueError as e:
            logger.warning(f"Error parsing Binance historical response: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error getting Binance historical data: {e}")
        
        return None
