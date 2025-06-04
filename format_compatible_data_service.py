#!/usr/bin/env python
"""
Fixed Multi-Asset Data Service for Trading-Agent System

This module provides a data service for fetching and managing market data
for cryptocurrency assets from MEXC API with improved symbol handling and
robust error handling for klines data with format compatibility.
"""

import os
import json
import time
import hmac
import hashlib
import requests
import threading
import websocket
import logging
import urllib.parse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multi_asset_data_service")

# Import environment loader
try:
    from env_loader import load_environment_variables
except ImportError:
    # Define a simple fallback if env_loader is not available
    def load_environment_variables(env_path=None):
        """Simple fallback for loading environment variables"""
        env_vars = {}
        try:
            if env_path is None:
                # Check for .env-secure/.env first, then fallback to .env
                if os.path.exists('.env-secure/.env'):
                    env_path = '.env-secure/.env'
                else:
                    env_path = '.env'
            
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            env_vars[key] = value
        except Exception as e:
            logger.error(f"Error loading environment variables: {str(e)}")
        
        return env_vars

class MultiAssetDataService:
    """Data service for cryptocurrency assets with improved symbol handling"""
    
    def __init__(self, supported_assets=None, env_path=None):
        """Initialize multi-asset data service
        
        Args:
            supported_assets: List of supported assets (default: BTC/USDC, ETH/USDC, SOL/USDC)
            env_path: Path to .env file (optional)
        """
        # Default supported assets
        self.supported_assets = supported_assets or ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        
        # Create symbol mappings
        self.symbol_map = {}
        for asset in self.supported_assets:
            # Remove slash for API format
            api_symbol = asset.replace('/', '')
            self.symbol_map[asset] = api_symbol
            # Also map the API format back to the internal format
            self.symbol_map[api_symbol] = asset
        
        self.current_asset = self.supported_assets[0] if self.supported_assets else None
        
        # Load API credentials
        env_vars = load_environment_variables(env_path)
        self.api_key = env_vars.get('MEXC_API_KEY')
        self.api_secret = env_vars.get('MEXC_SECRET_KEY')
        
        logger.info(f"Loaded API credentials - Key: {self.api_key[:5] if self.api_key else None}... Secret: {self.api_secret[:5] if self.api_secret else None}...")
        
        # API endpoints
        self.base_url = "https://api.mexc.com"
        self.ws_url = "wss://wbs.mexc.com/ws"
        
        # Initialize cache for each asset with default values
        self.cache = {}
        for asset in self.supported_assets:
            api_symbol = self.symbol_map[asset]
            # Ensure orderbook has valid structure with empty lists
            self.cache[asset] = {
                "ticker": {"price": 0, "symbol": asset, "timestamp": 0},
                "orderbook": {"asks": [], "bids": []},
                "trades": [],
                "klines": {},  # Keyed by interval
                "patterns": []
            }
            # Also cache with API symbol as key for direct lookups
            self.cache[api_symbol] = self.cache[asset]
        
        # WebSocket connections
        self.ws_connections = {}
        self.ws_threads = {}
        
        # Supported intervals for klines
        self.supported_intervals = ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1w']
        
        logger.info(f"Initialized MultiAssetDataService with {len(self.supported_assets)} assets")
        logger.info(f"Symbol mappings: {self.symbol_map}")
    
    def switch_asset(self, asset):
        """Switch current asset
        
        Args:
            asset: Asset to switch to
            
        Returns:
            bool: True if switch successful, False otherwise
        """
        if asset in self.supported_assets:
            self.current_asset = asset
            logger.info(f"Switched to asset: {asset}")
            return True
        
        logger.warning(f"Asset not supported: {asset}")
        return False
    
    def get_current_asset(self):
        """Get current asset
        
        Returns:
            str: Current asset
        """
        return self.current_asset
    
    def get_supported_assets(self):
        """Get supported assets
        
        Returns:
            list: Supported assets
        """
        return self.supported_assets
    
    def _generate_signature(self, params):
        """Generate signature for authenticated requests
        
        Args:
            params: Request parameters
            
        Returns:
            str: HMAC SHA256 signature
        """
        if not self.api_secret:
            return ""
        
        # MEXC requires parameters to be sorted alphabetically
        query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
        
        # Generate HMAC SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _generate_signature_v2(self, params):
        """Generate signature for authenticated requests using MEXC's exact method
        
        Args:
            params: Request parameters
            
        Returns:
            str: HMAC SHA256 signature
        """
        if not self.api_secret:
            return ""
        
        # MEXC requires parameters to be in the exact order they were added
        # Do not sort the parameters
        query_string = urllib.parse.urlencode(params)
        
        # Generate HMAC SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _get_timestamp(self):
        """Get current timestamp in milliseconds
        
        Returns:
            int: Current timestamp
        """
        return int(time.time() * 1000)
    
    def _get_symbol_for_api(self, asset):
        """Convert internal asset format to API symbol format
        
        Args:
            asset: Asset in internal format (e.g., BTC/USDC)
            
        Returns:
            str: Symbol in API format (e.g., BTCUSDC)
        """
        # If already in API format, return as is
        if '/' not in asset:
            return asset
        
        # Use symbol map if available
        if asset in self.symbol_map:
            return self.symbol_map[asset]
        
        # Fallback to simple replacement
        return asset.replace('/', '')
    
    def _get_asset_from_api_symbol(self, symbol):
        """Convert API symbol format to internal asset format
        
        Args:
            symbol: Symbol in API format (e.g., BTCUSDC)
            
        Returns:
            str: Asset in internal format (e.g., BTC/USDC)
        """
        # If already in internal format, return as is
        if '/' in symbol:
            return symbol
        
        # Use symbol map if available
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        
        # Fallback to default mapping logic
        for quote in ['USDC', 'USDT', 'USD', 'BUSD']:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return f"{base}/{quote}"
        
        # If no match, return as is
        return symbol
    
    def get_ticker(self, asset=None):
        """Get ticker for specified asset
        
        Args:
            asset: Asset to get ticker for (default: current asset)
            
        Returns:
            dict: Ticker data
        """
        target_asset = asset or self.current_asset
        symbol = self._get_symbol_for_api(target_asset)
        
        url = f"{self.base_url}/api/v3/ticker/price?symbol={symbol}"
        try:
            logger.info(f"Fetching ticker for {target_asset} (API symbol: {symbol})")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                ticker = {
                    "price": float(data["price"]),
                    "symbol": target_asset,
                    "timestamp": self._get_timestamp()
                }
                self.cache[target_asset]["ticker"] = ticker
                return ticker
            else:
                logger.error(f"Error fetching ticker for {target_asset} (API symbol: {symbol}): {response.status_code}, {response.text}")
                return self.cache[target_asset]["ticker"]
        except Exception as e:
            logger.error(f"Exception fetching ticker for {target_asset} (API symbol: {symbol}): {str(e)}")
            return self.cache[target_asset]["ticker"]
    
    def get_orderbook(self, asset=None, limit=20):
        """Get orderbook for specified asset
        
        Args:
            asset: Asset to get orderbook for (default: current asset)
            limit: Number of entries to return
            
        Returns:
            dict: Orderbook data with 'bids' and 'asks' keys
        """
        target_asset = asset or self.current_asset
        symbol = self._get_symbol_for_api(target_asset)
        
        # Ensure we have a valid cache entry with proper structure
        if target_asset not in self.cache:
            self.cache[target_asset] = {
                "ticker": {"price": 0, "symbol": target_asset, "timestamp": 0},
                "orderbook": {"asks": [], "bids": []},
                "trades": [],
                "klines": {},
                "patterns": []
            }
        
        # Ensure orderbook has valid structure
        if "orderbook" not in self.cache[target_asset]:
            self.cache[target_asset]["orderbook"] = {"asks": [], "bids": []}
        
        url = f"{self.base_url}/api/v3/depth?symbol={symbol}&limit={limit}"
        try:
            logger.info(f"Fetching orderbook for {target_asset} (API symbol: {symbol})")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                if "asks" not in data or "bids" not in data:
                    logger.error(f"Invalid orderbook response for {target_asset}: missing asks or bids")
                    return self.cache[target_asset]["orderbook"]
                
                # Format order book data for FlashTradingSignals compatibility
                # FlashTradingSignals expects raw arrays, not dictionaries
                orderbook = {
                    "asks": data["asks"][:limit],
                    "bids": data["bids"][:limit]
                }
                
                # Update cache
                self.cache[target_asset]["orderbook"] = orderbook
                return orderbook
            else:
                logger.error(f"Error fetching orderbook for {target_asset} (API symbol: {symbol}): {response.status_code}, {response.text}")
                return self.cache[target_asset]["orderbook"]
        except Exception as e:
            logger.error(f"Exception fetching orderbook for {target_asset} (API symbol: {symbol}): {str(e)}")
            return self.cache[target_asset]["orderbook"]
    
    # For compatibility with FlashTradingSignals
    def get_order_book(self, asset=None, limit=20):
        """Alias for get_orderbook to ensure compatibility with FlashTradingSignals
        
        Args:
            asset: Asset to get orderbook for (default: current asset)
            limit: Number of entries to return
            
        Returns:
            dict: Orderbook data with 'bids' and 'asks' keys
        """
        return self.get_orderbook(asset, limit)
    
    def get_trades(self, asset=None, limit=50):
        """Get recent trades for specified asset
        
        Args:
            asset: Asset to get trades for (default: current asset)
            limit: Number of trades to return
            
        Returns:
            list: Recent trades
        """
        target_asset = asset or self.current_asset
        symbol = self._get_symbol_for_api(target_asset)
        
        url = f"{self.base_url}/api/v3/trades?symbol={symbol}&limit={limit}"
        try:
            logger.info(f"Fetching trades for {target_asset} (API symbol: {symbol})")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Format trades data
                trades = []
                for trade in data:
                    trades.append({
                        "price": float(trade["price"]),
                        "quantity": float(trade["qty"]),
                        "time": int(trade["time"]),
                        "isBuyerMaker": trade["isBuyerMaker"]
                    })
                
                self.cache[target_asset]["trades"] = trades
                return trades
            else:
                logger.error(f"Error fetching trades for {target_asset} (API symbol: {symbol}): {response.status_code}, {response.text}")
                return self.cache[target_asset]["trades"]
        except Exception as e:
            logger.error(f"Exception fetching trades for {target_asset} (API symbol: {symbol}): {str(e)}")
            return self.cache[target_asset]["trades"]
    
    def _normalize_interval(self, interval):
        """Normalize interval to MEXC supported format
        
        Args:
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            
        Returns:
            str: Normalized interval
        """
        # MEXC supports these intervals directly
        if interval in ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1w', '1M']:
            return interval
            
        # Map common interval formats to MEXC supported formats
        interval_map = {
            '1h': '60m',
            '2h': '120m',
            '4h': '4h',
            '6h': '6h',
            '12h': '12h',
            '1d': '1d',
            '1w': '1w',
            '1M': '1M'
        }
        
        # Return mapped interval or original if not in map
        return interval_map.get(interval, interval)
    
    def get_klines(self, asset=None, interval="1m", limit=100):
        """Get klines (candlestick data) for specified asset with robust error handling
        and format compatibility for MEXC API
        
        Args:
            asset: Asset to get klines for (default: current asset)
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of klines to return
            
        Returns:
            list: Klines data or empty list if error
        """
        target_asset = asset or self.current_asset
        symbol = self._get_symbol_for_api(target_asset)
        
        # Normalize interval
        normalized_interval = self._normalize_interval(interval)
        
        # Check if interval is supported
        if normalized_interval not in self.supported_intervals:
            logger.warning(f"Unsupported interval: {interval}, using fallback interval '5m'")
            normalized_interval = '5m'
        
        # Check cache first
        cache_key = f"{normalized_interval}"
        if target_asset in self.cache and "klines" in self.cache[target_asset] and cache_key in self.cache[target_asset]["klines"]:
            cached_klines = self.cache[target_asset]["klines"][cache_key]
            # Only use cache if it's not empty and was updated recently (within 5 minutes)
            if cached_klines and len(cached_klines) > 0 and time.time() - cached_klines[0].get("cache_time", 0) < 300:
                return cached_klines
        
        # Ensure we have a valid cache structure
        if target_asset not in self.cache:
            self.cache[target_asset] = {
                "ticker": {"price": 0, "symbol": target_asset, "timestamp": 0},
                "orderbook": {"asks": [], "bids": []},
                "trades": [],
                "klines": {},
                "patterns": []
            }
        
        if "klines" not in self.cache[target_asset]:
            self.cache[target_asset]["klines"] = {}
        
        url = f"{self.base_url}/api/v3/klines?symbol={symbol}&interval={normalized_interval}&limit={limit}"
        try:
            logger.info(f"Fetching klines for {target_asset} (API symbol: {symbol}, interval: {normalized_interval})")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response data
                if not isinstance(data, list):
                    logger.error(f"Invalid klines response for {target_asset}: not a list, got {type(data)}")
                    return []
                
                if len(data) == 0:
                    logger.warning(f"Empty klines response for {target_asset}")
                    return []
                
                # Log the first kline to understand the format
                logger.debug(f"First kline format: {data[0]}")
                
                # Format klines data with robust error handling for MEXC format
                # MEXC returns 8 elements instead of 9 (missing trades count)
                klines = []
                for kline in data:
                    # Validate kline structure
                    if not isinstance(kline, list):
                        logger.warning(f"Invalid kline format (not a list): {kline}")
                        continue
                    
                    # MEXC format has 8 elements:
                    # [open_time, open, high, low, close, volume, close_time, quote_volume]
                    if len(kline) < 7:  # Minimum required elements
                        logger.warning(f"Invalid kline format (too few elements): {kline}")
                        continue
                    
                    try:
                        kline_data = {
                            "time": int(kline[0]),
                            "open": float(kline[1]),
                            "high": float(kline[2]),
                            "low": float(kline[3]),
                            "close": float(kline[4]),
                            "volume": float(kline[5]),
                            "close_time": int(kline[6]),
                            "quote_volume": float(kline[7]) if len(kline) > 7 else 0.0,
                            "trades": 0,  # Default value since MEXC doesn't provide this
                            "cache_time": time.time()
                        }
                        klines.append(kline_data)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing kline: {e}, kline: {kline}")
                        continue
                
                # Update cache
                self.cache[target_asset]["klines"][cache_key] = klines
                
                logger.info(f"Successfully fetched {len(klines)} klines for {target_asset}")
                return klines
            else:
                logger.error(f"Error fetching klines for {target_asset} (API symbol: {symbol}, interval: {normalized_interval}): {response.status_code}, {response.text}")
                # Return empty list if no cached data
                if target_asset not in self.cache or "klines" not in self.cache[target_asset] or cache_key not in self.cache[target_asset]["klines"]:
                    return []
                return self.cache[target_asset]["klines"].get(cache_key, [])
        except Exception as e:
            logger.error(f"Exception fetching klines for {target_asset} (API symbol: {symbol}, interval: {normalized_interval}): {str(e)}")
            # Return empty list if no cached data
            if target_asset not in self.cache or "klines" not in self.cache[target_asset] or cache_key not in self.cache[target_asset]["klines"]:
                return []
            return self.cache[target_asset]["klines"].get(cache_key, [])
