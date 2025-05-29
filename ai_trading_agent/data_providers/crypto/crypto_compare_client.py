"""
CryptoCompare API client for cryptocurrency data.

This module provides a client for the CryptoCompare API to fetch real-time
and historical cryptocurrency prices with proper error handling and rate limiting.
CryptoCompare offers better reliability and historical data with an API key.
"""

import os
import time
import logging
import requests
import json
from typing import Dict, Optional, Union, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CryptoCompareClient:
    """Client for the CryptoCompare API to fetch cryptocurrency data."""
    
    # API endpoints
    BASE_URL = "https://min-api.cryptocompare.com/data"
    PRICE_ENDPOINT = "/price"
    HISTORICAL_DAILY_ENDPOINT = "/histoday"
    HISTORICAL_HOURLY_ENDPOINT = "/histohour"
    
    # Symbol mapping for CryptoCompare
    SYMBOL_MAPPING = {
        'BTC/USD': {'fsym': 'BTC', 'tsym': 'USD'},
        'ETH/USD': {'fsym': 'ETH', 'tsym': 'USD'},
        'BNB/USD': {'fsym': 'BNB', 'tsym': 'USD'},
        'XRP/USD': {'fsym': 'XRP', 'tsym': 'USD'},
        'ADA/USD': {'fsym': 'ADA', 'tsym': 'USD'},
        'SOL/USD': {'fsym': 'SOL', 'tsym': 'USD'},
        'DOGE/USD': {'fsym': 'DOGE', 'tsym': 'USD'},
        'DOT/USD': {'fsym': 'DOT', 'tsym': 'USD'},
        'AVAX/USD': {'fsym': 'AVAX', 'tsym': 'USD'},
        'SHIB/USD': {'fsym': 'SHIB', 'tsym': 'USD'}
    }
    
    def __init__(self, api_key=None):
        """
        Initialize the CryptoCompare client.
        
        Args:
            api_key: Optional API key for authenticated access
        """
        self.api_key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
        self.session = requests.Session()
        
        # Set up headers if API key is provided
        if self.api_key:
            self.session.headers.update({
                'authorization': f'Apikey {self.api_key}'
            })
        
        self.last_request_time = 0
        self.rate_limit = 0.25  # Rate limit (seconds between requests)
        self.request_timeout = 10  # 10 seconds timeout
        
        # Initialize cache
        self.price_cache = {}
        self.cache_ttl = 30  # 30 seconds TTL
        self.cache_timestamps = {}
        
        logger.info(f"Initialized CryptoCompare client with API key: {'Yes' if self.api_key else 'No'}")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting to avoid being throttled by CryptoCompare."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # If less than the minimum interval has passed, sleep
        if elapsed < self.rate_limit and self.last_request_time > 0:
            sleep_time = self.rate_limit - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _map_symbol(self, symbol: str) -> Dict[str, str]:
        """Map standard symbol format to CryptoCompare parameters."""
        if symbol in self.SYMBOL_MAPPING:
            return self.SYMBOL_MAPPING[symbol]
        
        # If not found in mapping, split by '/'
        if '/' in symbol:
            fsym, tsym = symbol.split('/')
            return {'fsym': fsym, 'tsym': tsym}
        
        # Fallback: assume it's a crypto symbol and use USD as target
        return {'fsym': symbol, 'tsym': 'USD'}
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached price for a symbol is still valid."""
        if symbol not in self.cache_timestamps:
            return False
        
        return (time.time() - self.cache_timestamps[symbol]) < self.cache_ttl
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol from CryptoCompare.
        
        Args:
            symbol: The symbol to get price for (e.g., 'BTC/USD')
            
        Returns:
            Current price or None if unavailable
        """
        # Check cache first
        if symbol in self.price_cache and self._is_cache_valid(symbol):
            logger.debug(f"Using cached CryptoCompare price for {symbol}: ${self.price_cache[symbol]:.2f}")
            return self.price_cache[symbol]
        
        # Map symbol to CryptoCompare format
        symbol_params = self._map_symbol(symbol)
        
        # Enforce rate limit
        self._enforce_rate_limit()
        
        try:
            # Make request to CryptoCompare API
            url = f"{self.BASE_URL}{self.PRICE_ENDPOINT}"
            params = {
                "fsym": symbol_params['fsym'],
                "tsyms": symbol_params['tsym']
            }
            
            logger.debug(f"Fetching price from CryptoCompare for {symbol}")
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                target_symbol = symbol_params['tsym']
                
                if target_symbol in data:
                    price = float(data[target_symbol])
                    logger.debug(f"CryptoCompare response for {symbol}: ${price:.2f}")
                    
                    # Update cache
                    self.price_cache[symbol] = price
                    self.cache_timestamps[symbol] = time.time()
                    
                    return price
                else:
                    logger.warning(f"No price data in CryptoCompare response for {symbol}: {data}")
            else:
                logger.warning(f"CryptoCompare API error ({response.status_code}): {response.text}")
        
        except requests.RequestException as e:
            logger.warning(f"CryptoCompare request error for {symbol}: {e}")
        except ValueError as e:
            logger.warning(f"Error parsing CryptoCompare response for {symbol}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error getting CryptoCompare price for {symbol}: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30, interval: str = 'daily') -> Optional[Any]:
        """
        Get historical data for a symbol from CryptoCompare.
        
        Args:
            symbol: The symbol to get data for (e.g., 'BTC/USD')
            days: Number of days of historical data
            interval: Data interval ('daily', 'hourly', etc.)
            
        Returns:
            Pandas DataFrame with historical data or None if unavailable
        """
        try:
            import pandas as pd
            
            # Map symbol to CryptoCompare parameters
            symbol_params = self._map_symbol(symbol)
            
            # Choose the right endpoint based on interval
            if interval == 'daily':
                endpoint = self.HISTORICAL_DAILY_ENDPOINT
            elif interval == 'hourly':
                endpoint = self.HISTORICAL_HOURLY_ENDPOINT
            else:
                logger.warning(f"Unsupported interval {interval}, using daily")
                endpoint = self.HISTORICAL_DAILY_ENDPOINT
            
            # Enforce rate limit
            self._enforce_rate_limit()
            
            # Calculate limit (number of data points)
            limit = days
            if interval == 'hourly':
                limit = days * 24
            
            # Make request to CryptoCompare API
            url = f"{self.BASE_URL}{endpoint}"
            params = {
                "fsym": symbol_params['fsym'],
                "tsym": symbol_params['tsym'],
                "limit": min(limit, 2000),  # CryptoCompare limit is 2000
                "aggregate": 1
            }
            
            logger.debug(f"Fetching historical data from CryptoCompare for {symbol}")
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                if 'Data' in data and data['Data']:
                    # Create DataFrame from the data
                    df = pd.DataFrame(data['Data'])
                    
                    # Convert timestamp to datetime
                    df['date'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Extract price as close price and volume
                    df = df[['date', 'close', 'volumefrom']].rename(
                        columns={'close': 'price', 'volumefrom': 'volume'})
                    
                    logger.info(f"Got {len(df)} historical data points from CryptoCompare for {symbol}")
                    return df
                else:
                    logger.warning(f"No data in CryptoCompare historical response for {symbol}")
            else:
                logger.warning(f"CryptoCompare API error ({response.status_code}): {response.text}")
        
        except requests.RequestException as e:
            logger.warning(f"CryptoCompare request error for historical data: {e}")
        except ValueError as e:
            logger.warning(f"Error parsing CryptoCompare historical response: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error getting CryptoCompare historical data: {e}")
        
        return None
