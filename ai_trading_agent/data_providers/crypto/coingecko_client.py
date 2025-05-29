"""
CoinGecko API client for cryptocurrency data.

This module provides a client for the CoinGecko API to fetch real-time
and historical cryptocurrency prices with proper error handling and rate limiting.
CoinGecko offers a free API tier which requires no authentication.
"""

import os
import time
import logging
import requests
import json
from typing import Dict, Optional, Union, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CoinGeckoClient:
    """Client for the CoinGecko API to fetch cryptocurrency data."""
    
    # API endpoints
    BASE_URL = "https://api.coingecko.com/api/v3"
    PRICE_ENDPOINT = "/simple/price"
    COIN_LIST_ENDPOINT = "/coins/list"
    HISTORY_ENDPOINT = "/coins/{id}/market_chart"
    
    # Symbol mapping for CoinGecko (which uses IDs instead of symbols)
    SYMBOL_TO_ID = {
        'BTC/USD': 'bitcoin',
        'ETH/USD': 'ethereum',
        'BNB/USD': 'binancecoin',
        'XRP/USD': 'ripple',
        'ADA/USD': 'cardano',
        'SOL/USD': 'solana',
        'DOGE/USD': 'dogecoin',
        'DOT/USD': 'polkadot',
        'AVAX/USD': 'avalanche-2',
        'SHIB/USD': 'shiba-inu'
    }
    
    def __init__(self):
        """Initialize the CoinGecko client."""
        self.session = requests.Session()
        self.last_request_time = 0
        self.rate_limit = 1.3  # Max 50 requests per minute for free tier (1.2 seconds per request)
        self.request_timeout = 10  # 10 seconds timeout
        
        # Initialize cache
        self.price_cache = {}
        self.cache_ttl = 30  # 30 seconds TTL
        self.cache_timestamps = {}
        
        # Initialize ID mapping
        self._initialize_id_mapping()
        
        logger.info("Initialized CoinGecko client")
    
    def _initialize_id_mapping(self):
        """
        Initialize a mapping of symbols to CoinGecko IDs.
        This is needed because CoinGecko uses IDs (like 'bitcoin') instead of symbols.
        """
        # We already have a hardcoded mapping for common coins
        # But for completeness we could fetch the full list from API
        # if we need to support more coins.
        pass
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting to avoid being throttled by CoinGecko."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # If less than the minimum interval has passed, sleep
        if elapsed < self.rate_limit and self.last_request_time > 0:
            sleep_time = self.rate_limit - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _map_symbol_to_id(self, symbol: str) -> str:
        """Map standard symbol format to CoinGecko ID."""
        # Remove USD to get base currency
        base_currency = symbol.split('/')[0] if '/' in symbol else symbol
        
        # First check our mapping dictionary
        if symbol in self.SYMBOL_TO_ID:
            return self.SYMBOL_TO_ID[symbol]
        
        # Try a lowercase version of the base currency
        return base_currency.lower()
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached price for a symbol is still valid."""
        if symbol not in self.cache_timestamps:
            return False
        
        return (time.time() - self.cache_timestamps[symbol]) < self.cache_ttl
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol from CoinGecko.
        
        Args:
            symbol: The symbol to get price for (e.g., 'BTC/USD')
            
        Returns:
            Current price or None if unavailable
        """
        # Check cache first
        if symbol in self.price_cache and self._is_cache_valid(symbol):
            logger.debug(f"Using cached CoinGecko price for {symbol}: ${self.price_cache[symbol]:.2f}")
            return self.price_cache[symbol]
        
        # Map symbol to CoinGecko ID
        coin_id = self._map_symbol_to_id(symbol)
        
        # Enforce rate limit
        self._enforce_rate_limit()
        
        try:
            # Convert USD to 'usd' for the API
            vs_currency = 'usd'
            
            # Make request to CoinGecko API
            url = f"{self.BASE_URL}{self.PRICE_ENDPOINT}"
            params = {
                "ids": coin_id,
                "vs_currencies": vs_currency,
                "include_market_cap": "false",
                "include_24hr_vol": "false",
                "include_24hr_change": "false",
                "include_last_updated_at": "false"
            }
            
            logger.debug(f"Fetching price from CoinGecko for {coin_id}")
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                if coin_id in data and vs_currency in data[coin_id]:
                    price = float(data[coin_id][vs_currency])
                    logger.debug(f"CoinGecko response for {symbol}: ${price:.2f}")
                    
                    # Update cache
                    self.price_cache[symbol] = price
                    self.cache_timestamps[symbol] = time.time()
                    
                    return price
                else:
                    logger.warning(f"No price data in CoinGecko response for {symbol}: {data}")
            else:
                logger.warning(f"CoinGecko API error ({response.status_code}): {response.text}")
        
        except requests.RequestException as e:
            logger.warning(f"CoinGecko request error for {symbol}: {e}")
        except ValueError as e:
            logger.warning(f"Error parsing CoinGecko response for {symbol}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error getting CoinGecko price for {symbol}: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30, interval: str = 'daily') -> Optional[Any]:
        """
        Get historical data for a symbol from CoinGecko.
        
        Args:
            symbol: The symbol to get data for (e.g., 'BTC/USD')
            days: Number of days of historical data
            interval: Data interval ('daily', 'hourly', etc.)
            
        Returns:
            Pandas DataFrame with historical data or None if unavailable
        """
        try:
            import pandas as pd
            
            # Map symbol to CoinGecko ID
            coin_id = self._map_symbol_to_id(symbol)
            
            # Map interval to CoinGecko interval
            interval_map = {
                'daily': 'daily',
                'hourly': 'hourly'
            }
            coingecko_interval = interval_map.get(interval, 'daily')
            
            # Enforce rate limit
            self._enforce_rate_limit()
            
            # Make request to CoinGecko API
            url = f"{self.BASE_URL}{self.HISTORY_ENDPOINT.format(id=coin_id)}"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": coingecko_interval
            }
            
            logger.debug(f"Fetching historical data from CoinGecko for {coin_id}")
            response = self.session.get(url, params=params, timeout=self.request_timeout)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                if 'prices' in data and data['prices']:
                    # CoinGecko returns prices as a list of [timestamp, price] pairs
                    prices = data['prices']
                    volumes = data.get('total_volumes', [[0, 0]] * len(prices))
                    
                    # Create DataFrame
                    df = pd.DataFrame({
                        'timestamp': [p[0] for p in prices],
                        'price': [p[1] for p in prices],
                        'volume': [v[1] for v in volumes[:len(prices)]]  # In case lengths differ
                    })
                    
                    # Convert timestamp to datetime
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Select relevant columns
                    df = df[['date', 'price', 'volume']]
                    
                    logger.info(f"Got {len(df)} historical data points from CoinGecko for {symbol}")
                    return df
                else:
                    logger.warning(f"No price data in CoinGecko historical response for {symbol}")
            else:
                logger.warning(f"CoinGecko API error ({response.status_code}): {response.text}")
        
        except requests.RequestException as e:
            logger.warning(f"CoinGecko request error for historical data: {e}")
        except ValueError as e:
            logger.warning(f"Error parsing CoinGecko historical response: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error getting CoinGecko historical data: {e}")
        
        return None
