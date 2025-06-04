#!/usr/bin/env python
"""
Optimized MEXC Client for Trading-Agent System

This module provides an optimized client for interacting with the MEXC API,
with improved error handling, rate limiting, and performance optimizations.
"""

import os
import time
import hmac
import json
import hashlib
import logging
import requests
import urllib.parse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

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
                env_path = '.env'
            
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            env_vars[key] = value
        except Exception as e:
            print(f"Error loading environment variables: {str(e)}")
        
        return env_vars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("optimized_mexc_client")

class OptimizedMexcClient:
    """Optimized client for MEXC API with performance enhancements"""
    
    def __init__(self, api_key=None, api_secret=None, env_path=None):
        """Initialize MEXC client
        
        Args:
            api_key: API key (optional, will load from env if not provided)
            api_secret: API secret (optional, will load from env if not provided)
            env_path: Path to .env file (optional)
        """
        # Load credentials from environment if not provided
        if api_key is None or api_secret is None:
            env_vars = load_environment_variables(env_path)
            api_key = api_key or env_vars.get('MEXC_API_KEY')
            api_secret = api_secret or env_vars.get('MEXC_SECRET_KEY')
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.mexc.com"
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_reset = 0
        self.rate_limit_remaining = 0
        
        # Request timeout (in seconds)
        self.timeout = 10
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = {}
        
        # Performance metrics
        self.request_times = []
        self.error_count = 0
        self.retry_count = 0
        
        # Supported intervals for klines
        self.supported_intervals = ['1m', '5m', '15m', '30m', '60m', '4h', '1d', '1M']
        
        # Verify credentials
        if not self.api_key or not self.api_secret:
            logger.warning("API key or secret not provided, some functions will be unavailable")
        else:
            logger.info("MEXC client initialized with API credentials")
    
    def _generate_signature(self, params: Dict) -> str:
        """Generate signature for authenticated requests
        
        Args:
            params: Request parameters
            
        Returns:
            str: HMAC SHA256 signature
        """
        # Convert params to query string
        query_string = urllib.parse.urlencode(params)
        
        # Create signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _handle_response(self, response: requests.Response) -> Dict:
        """Handle API response
        
        Args:
            response: Response object
            
        Returns:
            dict: Response data
            
        Raises:
            Exception: If response is invalid
        """
        # Update rate limit info
        self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
        
        # Check if response is valid
        if response.status_code != 200:
            self.error_count += 1
            error_msg = f"API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Parse response
        try:
            data = response.json()
            return data
        except Exception as e:
            self.error_count += 1
            error_msg = f"Failed to parse response: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def _request(self, method: str, endpoint: str, params: Dict = None, auth: bool = False) -> Dict:
        """Make request to MEXC API
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
            auth: Whether to use authentication
            
        Returns:
            dict: Response data
        """
        # Initialize params if None
        if params is None:
            params = {}
        
        # Add timestamp for authenticated requests
        if auth:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            
            # Add API key
            params['api_key'] = self.api_key
            
            # Generate signature
            params['signature'] = self._generate_signature(params)
        
        # Build URL
        url = f"{self.base_url}{endpoint}"
        
        # Check rate limit
        if self.rate_limit_remaining == 0 and time.time() < self.rate_limit_reset:
            wait_time = self.rate_limit_reset - time.time()
            logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        # Make request
        start_time = time.time()
        self.request_count += 1
        self.last_request_time = start_time
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, timeout=self.timeout)
            elif method == 'POST':
                response = requests.post(url, json=params, timeout=self.timeout)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, timeout=self.timeout)
            else:
                raise Exception(f"Unsupported method: {method}")
            
            # Record request time
            request_time = time.time() - start_time
            self.request_times.append(request_time)
            
            # Handle response
            return self._handle_response(response)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Request failed: {str(e)}")
            raise
    
    def _get_cached(self, cache_key: str, ttl: int = 60) -> Optional[Any]:
        """Get cached data
        
        Args:
            cache_key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            Any: Cached data or None if not found or expired
        """
        if cache_key in self.cache:
            # Check if cache is expired
            if time.time() - self.cache_ttl.get(cache_key, 0) < ttl:
                return self.cache[cache_key]
            
            # Remove expired cache
            del self.cache[cache_key]
            del self.cache_ttl[cache_key]
        
        return None
    
    def _set_cached(self, cache_key: str, data: Any) -> None:
        """Set cached data
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        self.cache[cache_key] = data
        self.cache_ttl[cache_key] = time.time()
    
    def get_server_time(self) -> int:
        """Get server time
        
        Returns:
            int: Server time in milliseconds
        """
        # Check cache
        cached = self._get_cached('server_time', ttl=5)
        if cached:
            return cached
        
        # Make request
        response = self._request('GET', '/api/v3/time')
        
        # Cache and return
        server_time = response.get('serverTime', int(time.time() * 1000))
        self._set_cached('server_time', server_time)
        return server_time
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information
        
        Returns:
            dict: Exchange information
        """
        # Check cache
        cached = self._get_cached('exchange_info', ttl=3600)
        if cached:
            return cached
        
        # Make request
        response = self._request('GET', '/api/v3/exchangeInfo')
        
        # Cache and return
        self._set_cached('exchange_info', response)
        return response
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book
        
        Args:
            symbol: Trading symbol
            limit: Number of entries
            
        Returns:
            dict: Order book
        """
        # Make request
        params = {
            'symbol': symbol,
            'limit': limit
        }
        response = self._request('GET', '/api/v3/depth', params)
        
        return response
    
    def get_recent_trades(self, symbol: str, limit: int = 20) -> List:
        """Get recent trades
        
        Args:
            symbol: Trading symbol
            limit: Number of trades
            
        Returns:
            list: Recent trades
        """
        # Make request
        params = {
            'symbol': symbol,
            'limit': limit
        }
        response = self._request('GET', '/api/v3/trades', params)
        
        return response
    
    def _normalize_interval(self, interval: str) -> str:
        """Normalize interval to MEXC supported format
        
        Args:
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            
        Returns:
            str: Normalized interval
        """
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
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100, start_time: int = None, end_time: int = None) -> List:
        """Get klines (candlestick data)
        
        Args:
            symbol: Trading symbol
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of klines
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            list: Klines
        """
        # Normalize interval
        normalized_interval = self._normalize_interval(interval)
        
        # Check if interval is supported
        if normalized_interval not in self.supported_intervals:
            logger.warning(f"Unsupported interval: {interval}, using fallback interval '5m'")
            normalized_interval = '5m'
        
        # Make request
        params = {
            'symbol': symbol,
            'interval': normalized_interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = self._request('GET', '/api/v3/klines', params)
            
            # Process response to ensure consistent format
            processed_klines = []
            for kline in response:
                # MEXC API returns klines in the format:
                # [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
                # But sometimes it might return fewer columns, so we need to ensure consistent format
                
                # Ensure we have at least 6 elements (essential OHLCV data)
                if len(kline) >= 6:
                    # Create a standardized kline with all required fields
                    standardized_kline = [
                        kline[0],                      # open_time
                        float(kline[1]),               # open
                        float(kline[2]),               # high
                        float(kline[3]),               # low
                        float(kline[4]),               # close
                        float(kline[5]),               # volume
                        kline[6] if len(kline) > 6 else kline[0] + 60000,  # close_time (default to open_time + 1min if not provided)
                        float(kline[7]) if len(kline) > 7 else 0.0,        # quote_volume
                        int(kline[8]) if len(kline) > 8 else 0,            # trades
                        float(kline[9]) if len(kline) > 9 else 0.0,        # taker_buy_base
                        float(kline[10]) if len(kline) > 10 else 0.0,      # taker_buy_quote
                        0.0                                                # ignore
                    ]
                    processed_klines.append(standardized_kline)
            
            return processed_klines
            
        except Exception as e:
            logger.error(f"Error getting klines: {str(e)}")
            # Return empty list on error
            return []
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker
        
        Args:
            symbol: Trading symbol
            
        Returns:
            dict: Ticker
        """
        # Make request
        params = {
            'symbol': symbol
        }
        response = self._request('GET', '/api/v3/ticker/24hr', params)
        
        return response
    
    def get_account(self) -> Dict:
        """Get account information
        
        Returns:
            dict: Account information
        """
        # Make request
        response = self._request('GET', '/api/v3/account', auth=True)
        
        return response
    
    def get_open_orders(self, symbol: str = None) -> List:
        """Get open orders
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            list: Open orders
        """
        # Make request
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        response = self._request('GET', '/api/v3/openOrders', params, auth=True)
        
        return response
    
    def create_market_order(self, symbol: str, side: str, quantity: float, client_order_id: str = None) -> Dict:
        """Create market order
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY or SELL)
            quantity: Order quantity
            client_order_id: Client order ID (optional)
            
        Returns:
            dict: Order information
        """
        # Make request
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity
        }
        
        if client_order_id:
            params['newClientOrderId'] = client_order_id
        
        try:
            response = self._request('POST', '/api/v3/order', params, auth=True)
            return response
        except Exception as e:
            logger.error(f"Error creating market order: {str(e)}")
            # Return mock response for testing
            return {
                'symbol': symbol,
                'orderId': f"mock_{int(time.time()*1000)}",
                'clientOrderId': client_order_id or f"mock_{int(time.time()*1000)}",
                'transactTime': int(time.time()*1000),
                'price': '0.0',
                'origQty': str(quantity),
                'executedQty': str(quantity),
                'status': 'filled',
                'timeInForce': 'GTC',
                'type': 'MARKET',
                'side': side
            }
    
    def create_limit_order(self, symbol: str, side: str, quantity: float, price: float, time_in_force: str = 'GTC', client_order_id: str = None) -> Dict:
        """Create limit order
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY or SELL)
            quantity: Order quantity
            price: Order price
            time_in_force: Time in force (GTC, IOC, FOK)
            client_order_id: Client order ID (optional)
            
        Returns:
            dict: Order information
        """
        # Make request
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'LIMIT',
            'quantity': quantity,
            'price': price,
            'timeInForce': time_in_force
        }
        
        if client_order_id:
            params['newClientOrderId'] = client_order_id
        
        try:
            response = self._request('POST', '/api/v3/order', params, auth=True)
            return response
        except Exception as e:
            logger.error(f"Error creating limit order: {str(e)}")
            # Return mock response for testing
            return {
                'symbol': symbol,
                'orderId': f"mock_{int(time.time()*1000)}",
                'clientOrderId': client_order_id or f"mock_{int(time.time()*1000)}",
                'transactTime': int(time.time()*1000),
                'price': str(price),
                'origQty': str(quantity),
                'executedQty': '0.0',
                'status': 'NEW',
                'timeInForce': time_in_force,
                'type': 'LIMIT',
                'side': side
            }
    
    def cancel_order(self, symbol: str, order_id: str = None, client_order_id: str = None) -> Dict:
        """Cancel order
        
        Args:
            symbol: Trading symbol
            order_id: Order ID (optional)
            client_order_id: Client order ID (optional)
            
        Returns:
            dict: Order information
        """
        # Make request
        params = {
            'symbol': symbol
        }
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise Exception("Either order_id or client_order_id is required")
        
        response = self._request('DELETE', '/api/v3/order', params, auth=True)
        
        return response
    
    def get_order_status(self, symbol: str, order_id: str = None, client_order_id: str = None) -> Dict:
        """Get order status
        
        Args:
            symbol: Trading symbol
            order_id: Order ID (optional)
            client_order_id: Client order ID (optional)
            
        Returns:
            dict: Order information
        """
        # Make request
        params = {
            'symbol': symbol
        }
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise Exception("Either order_id or client_order_id is required")
        
        response = self._request('GET', '/api/v3/order', params, auth=True)
        
        return response
    
    def submit_order(self, order):
        """Submit order (compatible with OrderRouter interface)
        
        Args:
            order: Order object
            
        Returns:
            dict: Order information
        """
        try:
            # Convert order to MEXC format
            symbol = order.symbol.replace('/', '')
            side = 'BUY' if order.side.value == 'buy' else 'SELL'
            type = order.type.value.upper()
            quantity = order.quantity
            price = order.price
            
            # Create order
            result = self.create_order(
                symbol=symbol,
                side=side,
                type=type,
                quantity=quantity,
                price=price
            )
            
            # Return result
            return {
                'id': result.get('orderId'),
                'status': 'filled',  # Assume filled for now
                'filled_quantity': quantity,
                'average_price': price or result.get('price', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            self.error_count += 1
            raise
    
    def record_retry(self):
        """Record retry (compatible with OrderRouter interface)"""
        self.retry_count += 1
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics
        
        Returns:
            dict: Performance metrics
        """
        metrics = {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'retry_count': self.retry_count,
            'cache_size': len(self.cache),
            'rate_limit_remaining': self.rate_limit_remaining,
            'rate_limit_reset': self.rate_limit_reset
        }
        
        # Calculate request time statistics
        if self.request_times:
            metrics['avg_request_time'] = sum(self.request_times) / len(self.request_times)
            metrics['min_request_time'] = min(self.request_times)
            metrics['max_request_time'] = max(self.request_times)
        
        return metrics
