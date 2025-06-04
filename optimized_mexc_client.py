#!/usr/bin/env python
"""
Optimized MEXC API Client with Rate Limit Compliance

This module provides an optimized client for interacting with the MEXC API,
featuring robust rate limit compliance, exponential backoff, and comprehensive
error handling.
"""

import os
import json
import time
import hmac
import hashlib
import logging
import threading
import queue
import urllib.parse
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mexc_api_client.log")
    ]
)

logger = logging.getLogger("mexc_api_client")

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, requests_per_second=5, burst_limit=10):
        """Initialize rate limiter
        
        Args:
            requests_per_second: Maximum requests per second
            burst_limit: Maximum burst of requests allowed
        """
        self.requests_per_second = requests_per_second
        self.burst_limit = burst_limit
        self.tokens = burst_limit
        self.last_refill_time = time.time()
        self.lock = threading.RLock()
    
    def refill_tokens(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill_time
        
        # Calculate new tokens to add
        new_tokens = elapsed * self.requests_per_second
        
        # Update tokens and last refill time
        with self.lock:
            self.tokens = min(self.tokens + new_tokens, self.burst_limit)
            self.last_refill_time = now
    
    def consume(self, tokens=1):
        """Consume tokens for a request
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            float: Time to wait in seconds before making request, or 0 if can proceed immediately
        """
        self.refill_tokens()
        
        with self.lock:
            if self.tokens >= tokens:
                # Enough tokens, consume and proceed
                self.tokens -= tokens
                return 0
            else:
                # Not enough tokens, calculate wait time
                wait_time = (tokens - self.tokens) / self.requests_per_second
                return wait_time

class OptimizedMEXCClient:
    """Optimized client for MEXC API with rate limit compliance"""
    
    # API endpoints
    BASE_URL = "https://api.mexc.com"
    
    # Rate limits (as of March 25, 2025)
    SPOT_ORDER_RATE_LIMIT = 5  # orders per second
    MARKET_DATA_RATE_LIMIT = 10  # estimated safe limit for market data requests
    
    # Request categories for rate limiting
    CATEGORY_MARKET_DATA = "market_data"
    CATEGORY_ORDER = "order"
    CATEGORY_ACCOUNT = "account"
    
    def __init__(self, api_key=None, api_secret=None, env_path=None):
        """Initialize MEXC client
        
        Args:
            api_key: MEXC API key (optional, will load from env if not provided)
            api_secret: MEXC API secret (optional, will load from env if not provided)
            env_path: Path to .env file (optional)
        """
        # Load API credentials
        self.api_key = api_key
        self.api_secret = api_secret
        
        if not self.api_key or not self.api_secret:
            self._load_credentials(env_path)
        
        # Initialize rate limiters for different request categories
        self.rate_limiters = {
            self.CATEGORY_MARKET_DATA: RateLimiter(requests_per_second=10, burst_limit=20),
            self.CATEGORY_ORDER: RateLimiter(requests_per_second=5, burst_limit=10),
            self.CATEGORY_ACCOUNT: RateLimiter(requests_per_second=2, burst_limit=5)
        }
        
        # Request queue and processing thread
        self.request_queue = queue.Queue()
        self.request_thread = threading.Thread(target=self._process_request_queue, daemon=True)
        self.request_thread.start()
        
        # Request tracking
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = 0
        self.last_error_time = 0
        self.last_429_time = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Optimized MEXC client initialized")
    
    def _load_credentials(self, env_path=None):
        """Load API credentials from environment variables or .env file
        
        Args:
            env_path: Path to .env file (optional)
        """
        # Try to load from environment variables first
        self.api_key = os.environ.get('MEXC_API_KEY')
        self.api_secret = os.environ.get('MEXC_SECRET_KEY')
        
        # If not found, try to load from .env file
        if not self.api_key or not self.api_secret:
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
                                if key == 'MEXC_API_KEY':
                                    self.api_key = value
                                elif key == 'MEXC_SECRET_KEY':
                                    self.api_secret = value
            except Exception as e:
                logger.error(f"Error loading credentials from .env file: {str(e)}")
        
        # Log credential status
        if self.api_key and self.api_secret:
            logger.info(f"API credentials loaded successfully: {self.api_key[:5]}...")
        else:
            logger.warning("API credentials not found or incomplete")
    
    def _generate_signature(self, params):
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
    
    def _process_request_queue(self):
        """Process the request queue with rate limiting"""
        while True:
            try:
                # Get request from queue
                category, method, url, params, headers, result_queue = self.request_queue.get()
                
                # Apply rate limiting
                rate_limiter = self.rate_limiters.get(category, self.rate_limiters[self.CATEGORY_MARKET_DATA])
                wait_time = rate_limiter.consume()
                
                if wait_time > 0:
                    logger.debug(f"Rate limiting: Waiting {wait_time:.3f}s before {category} request")
                    time.sleep(wait_time)
                
                # Execute request with exponential backoff
                max_retries = 3
                base_delay = 1  # second
                
                for retry in range(max_retries + 1):
                    try:
                        # Update last request time
                        with self.lock:
                            self.last_request_time = time.time()
                            self.request_count += 1
                        
                        # Execute request
                        if method.upper() == 'GET':
                            response = requests.get(url, params=params, headers=headers, timeout=10)
                        elif method.upper() == 'POST':
                            response = requests.post(url, json=params, headers=headers, timeout=10)
                        elif method.upper() == 'DELETE':
                            response = requests.delete(url, params=params, headers=headers, timeout=10)
                        else:
                            raise ValueError(f"Unsupported HTTP method: {method}")
                        
                        # Check for rate limit error
                        if response.status_code == 429:
                            with self.lock:
                                self.last_429_time = time.time()
                                self.error_count += 1
                            
                            # Calculate retry delay with exponential backoff
                            delay = base_delay * (2 ** retry)
                            logger.warning(f"Rate limit exceeded (429), retrying in {delay}s (attempt {retry + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                        
                        # Check for other errors
                        if response.status_code >= 400:
                            with self.lock:
                                self.last_error_time = time.time()
                                self.error_count += 1
                            
                            # Only retry on 5xx errors or specific 4xx errors
                            if response.status_code >= 500 or response.status_code in [408, 429]:
                                if retry < max_retries:
                                    delay = base_delay * (2 ** retry)
                                    logger.warning(f"Request failed with status {response.status_code}, retrying in {delay}s (attempt {retry + 1}/{max_retries})")
                                    time.sleep(delay)
                                    continue
                            
                            # Return error response
                            result_queue.put((False, response))
                            break
                        
                        # Return successful response
                        result_queue.put((True, response))
                        break
                    
                    except requests.exceptions.RequestException as e:
                        with self.lock:
                            self.last_error_time = time.time()
                            self.error_count += 1
                        
                        if retry < max_retries:
                            delay = base_delay * (2 ** retry)
                            logger.warning(f"Request exception: {str(e)}, retrying in {delay}s (attempt {retry + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            logger.error(f"Request failed after {max_retries} retries: {str(e)}")
                            result_queue.put((False, str(e)))
                
                # Mark task as done
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in request queue processor: {str(e)}")
                time.sleep(1)  # Prevent tight loop in case of persistent errors
    
    def _queue_request(self, category, method, endpoint, params=None, headers=None, authenticated=False):
        """Queue a request with rate limiting
        
        Args:
            category: Request category for rate limiting
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            headers: Request headers
            authenticated: Whether request requires authentication
            
        Returns:
            requests.Response: Response object
        """
        # Prepare URL
        url = f"{self.BASE_URL}{endpoint}"
        
        # Prepare headers
        if headers is None:
            headers = {}
        
        # Add authentication if required
        if authenticated:
            if not self.api_key or not self.api_secret:
                raise ValueError("API key and secret required for authenticated requests")
            
            # Add API key to headers
            headers["X-MEXC-APIKEY"] = self.api_key
            
            # Add timestamp and signature to params
            if params is None:
                params = {}
            
            params["timestamp"] = self._get_timestamp()
            
            # Generate signature
            signature = self._generate_signature(params)
            params["signature"] = signature
        
        # Create result queue
        result_queue = queue.Queue()
        
        # Add request to queue
        self.request_queue.put((category, method, url, params, headers, result_queue))
        
        # Wait for result
        success, result = result_queue.get()
        
        if success:
            return result
        elif isinstance(result, requests.Response):
            # Log error details
            try:
                error_data = result.json()
                logger.error(f"API error: {result.status_code} - {error_data}")
            except:
                logger.error(f"API error: {result.status_code} - {result.text}")
            
            # Raise exception with status code
            result.raise_for_status()
            return result  # This line won't be reached due to the raise_for_status() call
        else:
            raise Exception(f"Request failed: {result}")
    
    def get_ticker(self, symbol):
        """Get ticker for specified symbol
        
        Args:
            symbol: Symbol in API format (e.g., BTCUSDC)
            
        Returns:
            dict: Ticker data
        """
        endpoint = "/api/v3/ticker/price"
        params = {"symbol": symbol}
        
        response = self._queue_request(
            category=self.CATEGORY_MARKET_DATA,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.json()
    
    def get_orderbook(self, symbol, limit=20):
        """Get orderbook for specified symbol
        
        Args:
            symbol: Symbol in API format (e.g., BTCUSDC)
            limit: Number of entries to return
            
        Returns:
            dict: Orderbook data
        """
        endpoint = "/api/v3/depth"
        params = {"symbol": symbol, "limit": limit}
        
        response = self._queue_request(
            category=self.CATEGORY_MARKET_DATA,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.json()
    
    def get_trades(self, symbol, limit=50):
        """Get recent trades for specified symbol
        
        Args:
            symbol: Symbol in API format (e.g., BTCUSDC)
            limit: Number of trades to return
            
        Returns:
            list: Recent trades
        """
        endpoint = "/api/v3/trades"
        params = {"symbol": symbol, "limit": limit}
        
        response = self._queue_request(
            category=self.CATEGORY_MARKET_DATA,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.json()
    
    def get_klines(self, symbol, interval="5m", limit=100):
        """Get klines (candlestick data) for specified symbol
        
        Args:
            symbol: Symbol in API format (e.g., BTCUSDC)
            interval: Kline interval (1m, 5m, 15m, 60m, 4h, 1d, etc.)
            limit: Number of klines to return
            
        Returns:
            list: Klines data
        """
        endpoint = "/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        
        response = self._queue_request(
            category=self.CATEGORY_MARKET_DATA,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.json()
    
    def get_exchange_info(self):
        """Get exchange information including symbol details
        
        Returns:
            dict: Exchange information
        """
        endpoint = "/api/v3/exchangeInfo"
        
        response = self._queue_request(
            category=self.CATEGORY_MARKET_DATA,
            method="GET",
            endpoint=endpoint
        )
        
        return response.json()
    
    def get_account_info(self):
        """Get account information (requires API key)
        
        Returns:
            dict: Account information
        """
        endpoint = "/api/v3/account"
        
        response = self._queue_request(
            category=self.CATEGORY_ACCOUNT,
            method="GET",
            endpoint=endpoint,
            params={},
            authenticated=True
        )
        
        return response.json()
    
    def create_order(self, symbol, side, order_type, quantity, price=None, time_in_force="GTC"):
        """Create a new order
        
        Args:
            symbol: Symbol in API format (e.g., BTCUSDC)
            side: Order side (BUY or SELL)
            order_type: Order type (LIMIT, MARKET, STOP_LOSS, STOP_LOSS_LIMIT, etc.)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            
        Returns:
            dict: Order response
        """
        endpoint = "/api/v3/order"
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity
        }
        
        # Add price for LIMIT orders
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            params["price"] = price
            params["timeInForce"] = time_in_force
        
        response = self._queue_request(
            category=self.CATEGORY_ORDER,
            method="POST",
            endpoint=endpoint,
            params=params,
            authenticated=True
        )
        
        return response.json()
    
    def cancel_order(self, symbol, order_id=None, client_order_id=None):
        """Cancel an existing order
        
        Args:
            symbol: Symbol in API format (e.g., BTCUSDC)
            order_id: Order ID (optional if client_order_id is provided)
            client_order_id: Client order ID (optional if order_id is provided)
            
        Returns:
            dict: Cancel response
        """
        endpoint = "/api/v3/order"
        
        # Prepare parameters
        params = {"symbol": symbol}
        
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")
        
        response = self._queue_request(
            category=self.CATEGORY_ORDER,
            method="DELETE",
            endpoint=endpoint,
            params=params,
            authenticated=True
        )
        
        return response.json()
    
    def get_order(self, symbol, order_id=None, client_order_id=None):
        """Get order status
        
        Args:
            symbol: Symbol in API format (e.g., BTCUSDC)
            order_id: Order ID (optional if client_order_id is provided)
            client_order_id: Client order ID (optional if order_id is provided)
            
        Returns:
            dict: Order status
        """
        endpoint = "/api/v3/order"
        
        # Prepare parameters
        params = {"symbol": symbol}
        
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")
        
        response = self._queue_request(
            category=self.CATEGORY_ORDER,
            method="GET",
            endpoint=endpoint,
            params=params,
            authenticated=True
        )
        
        return response.json()
    
    def get_open_orders(self, symbol=None):
        """Get all open orders
        
        Args:
            symbol: Symbol in API format (optional)
            
        Returns:
            list: Open orders
        """
        endpoint = "/api/v3/openOrders"
        
        # Prepare parameters
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        response = self._queue_request(
            category=self.CATEGORY_ORDER,
            method="GET",
            endpoint=endpoint,
            params=params,
            authenticated=True
        )
        
        return response.json()
    
    def get_status(self):
        """Get client status
        
        Returns:
            dict: Client status
        """
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "last_request_time": datetime.fromtimestamp(self.last_request_time).isoformat() if self.last_request_time else None,
            "last_error_time": datetime.fromtimestamp(self.last_error_time).isoformat() if self.last_error_time else None,
            "last_429_time": datetime.fromtimestamp(self.last_429_time).isoformat() if self.last_429_time else None,
            "queue_size": self.request_queue.qsize()
        }
    
    def wait_for_queue_empty(self, timeout=None):
        """Wait for request queue to be empty
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if queue is empty, False if timeout occurred
        """
        try:
            self.request_queue.join(timeout=timeout)
            return True
        except:
            return False

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = OptimizedMEXCClient()
    
    # Test with BTCUSDC
    symbol = "BTCUSDC"
    
    # Get ticker
    try:
        ticker = client.get_ticker(symbol)
        print(f"Ticker: {ticker}")
    except Exception as e:
        print(f"Error getting ticker: {str(e)}")
    
    # Get klines
    try:
        klines = client.get_klines(symbol, interval="5m", limit=10)
        print(f"Klines: {len(klines)} candles")
        if klines:
            print(f"First candle: {klines[0]}")
            print(f"Last candle: {klines[-1]}")
    except Exception as e:
        print(f"Error getting klines: {str(e)}")
    
    # Get client status
    status = client.get_status()
    print(f"Client status: {status}")
