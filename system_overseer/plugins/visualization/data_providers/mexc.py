#!/usr/bin/env python
"""
MEXC Data Provider

This module implements the DataProvider interface for the MEXC exchange.
"""

import os
import time
import logging
import requests
import hmac
import hashlib
import pandas as pd
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from .base import DataProvider

logger = logging.getLogger("system_overseer.plugins.visualization.data_providers.mexc")

class MexcDataProvider(DataProvider):
    """MEXC exchange data provider implementation."""
    
    def __init__(self):
        """Initialize MEXC data provider."""
        self.api_key = None
        self.api_secret = None
        self.base_url = "https://api.mexc.com"
        self.spot_api_url = self.base_url + "/api/v3"
        self.initialized = False
        
        # Available intervals mapping
        self.intervals = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1w",
            "1M": "1M"
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the data provider with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        logger.info("Initializing MEXC data provider")
        
        try:
            # Load API credentials from environment if not in config
            if "api_key" in config and "api_secret" in config:
                self.api_key = config["api_key"]
                self.api_secret = config["api_secret"]
            else:
                # Try to load from environment
                load_dotenv('.env-secure/.env')
                self.api_key = os.getenv('MEXC_API_KEY')
                self.api_secret = os.getenv('MEXC_API_SECRET')
            
            if not self.api_key or not self.api_secret:
                logger.error("MEXC API credentials not found")
                return False
            
            # Test connection
            symbols = self.get_available_symbols()
            if not symbols:
                logger.error("Failed to retrieve symbols from MEXC API")
                return False
            
            logger.info(f"MEXC data provider initialized successfully with {len(symbols)} symbols")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MEXC data provider: {e}")
            return False
    
    def generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for MEXC API.
        
        Args:
            query_string: Query string to sign
            
        Returns:
            str: Signature
        """
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get klines (candlestick data) for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            interval: Time interval (e.g., '1m', '5m', '1h')
            limit: Maximum number of klines to retrieve
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with columns [timestamp, open, high, low, close, volume]
                                   or None if data retrieval failed
        """
        if not self.initialized:
            logger.error("MEXC data provider not initialized")
            return None
        
        # Map interval to MEXC format
        mexc_interval = self.intervals.get(interval, "1m")
        
        # Endpoint
        endpoint = '/klines'
        url = self.spot_api_url + endpoint
        
        # Parameters
        params = {
            'symbol': symbol,
            'interval': mexc_interval,
            'limit': min(limit, 1000)  # MEXC has a max limit of 1000
        }
        
        try:
            # Send request
            response = requests.get(url, params=params)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                
                # Log the actual data structure for debugging
                if data and len(data) > 0:
                    logger.info(f"MEXC klines data structure: {len(data[0])} columns")
                    logger.debug(f"First kline data: {data[0]}")
                else:
                    logger.error("Empty data returned from MEXC API")
                    return None
                
                try:
                    # Create DataFrame directly from the raw data
                    raw_df = pd.DataFrame(data)
                    
                    # Ensure we have enough columns for OHLCV data
                    if raw_df.shape[1] < 6:
                        logger.error(f"Insufficient columns in MEXC API response: {raw_df.shape[1]}")
                        return None
                    
                    # Create a new DataFrame with only the columns we need
                    result_df = pd.DataFrame()
                    result_df['timestamp'] = pd.to_datetime(raw_df.iloc[:, 0], unit='ms')
                    result_df['open'] = pd.to_numeric(raw_df.iloc[:, 1])
                    result_df['high'] = pd.to_numeric(raw_df.iloc[:, 2])
                    result_df['low'] = pd.to_numeric(raw_df.iloc[:, 3])
                    result_df['close'] = pd.to_numeric(raw_df.iloc[:, 4])
                    result_df['volume'] = pd.to_numeric(raw_df.iloc[:, 5])
                    
                    # Log the shape of the resulting DataFrame
                    logger.info(f"Processed klines DataFrame shape: {result_df.shape}")
                    
                    # Verify we have data
                    if result_df.empty:
                        logger.error("Empty result DataFrame after processing")
                        return None
                    
                    return result_df
                    
                except Exception as e:
                    logger.error(f"Error processing klines data: {e}")
                    return None
            else:
                logger.error(f"Failed to get klines from MEXC API: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting klines from MEXC API: {e}")
            return None
    
    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary with ticker data or None if retrieval failed
        """
        if not self.initialized:
            logger.error("MEXC data provider not initialized")
            return None
        
        # Endpoint
        endpoint = '/ticker/24hr'
        url = self.spot_api_url + endpoint
        
        # Parameters
        params = {
            'symbol': symbol
        }
        
        try:
            # Send request
            response = requests.get(url, params=params)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                logger.error(f"Failed to get ticker from MEXC API: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting ticker from MEXC API: {e}")
            return None
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDC')
            limit: Maximum number of orders to retrieve for each side
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary with order book data or None if retrieval failed
        """
        if not self.initialized:
            logger.error("MEXC data provider not initialized")
            return None
        
        # Endpoint
        endpoint = '/depth'
        url = self.spot_api_url + endpoint
        
        # Parameters
        params = {
            'symbol': symbol,
            'limit': min(limit, 5000)  # MEXC has a max limit
        }
        
        try:
            # Send request
            response = requests.get(url, params=params)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                logger.error(f"Failed to get order book from MEXC API: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting order book from MEXC API: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pair symbols.
        
        Returns:
            List[str]: List of available symbols
        """
        # Endpoint
        endpoint = '/exchangeInfo'
        url = self.spot_api_url + endpoint
        
        try:
            # Send request
            response = requests.get(url)
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                symbols = [symbol['symbol'] for symbol in data.get('symbols', [])]
                return symbols
            else:
                logger.error(f"Failed to get symbols from MEXC API: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting symbols from MEXC API: {e}")
            return []
    
    def get_available_intervals(self) -> List[str]:
        """Get list of available time intervals.
        
        Returns:
            List[str]: List of available intervals
        """
        return list(self.intervals.keys())
