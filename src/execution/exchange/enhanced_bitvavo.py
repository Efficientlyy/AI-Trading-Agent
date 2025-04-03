"""
Enhanced Bitvavo Exchange Connector

This module provides an enhanced connector for the Bitvavo cryptocurrency exchange,
with improved reliability, monitoring, and performance.
"""

import time
import hmac
import hashlib
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Import base connector
from src.execution.exchange.bitvavo import BitvavoConnector

# Import enhanced components
from src.monitoring.error_monitor import ErrorMonitor
from src.monitoring.rate_limit_monitor import RateLimitMonitor
from src.common.cache.data_cache import DataCache
from src.common.network.fallback_manager import BitvavoFallbackManager, FallbackTrigger, FallbackStrategy
from src.common.network.connection_pool import ConnectionPool, RequestBatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBitvavoConnector(BitvavoConnector):
    """
    Enhanced connector for the Bitvavo cryptocurrency exchange.
    
    This class extends the base BitvavoConnector with additional features:
    - Connection pooling for improved performance
    - Error monitoring with circuit breakers
    - Rate limit monitoring with alerts
    - Data caching for reduced API calls
    - Fallback strategies for API outages
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "",
                 connection_pool: ConnectionPool = None,
                 error_monitor: ErrorMonitor = None,
                 rate_limit_monitor: RateLimitMonitor = None,
                 data_cache: DataCache = None,
                 fallback_manager: BitvavoFallbackManager = None):
        """
        Initialize the enhanced Bitvavo connector.
        
        Args:
            api_key: Bitvavo API key
            api_secret: Bitvavo API secret
            connection_pool: Connection pool for HTTP requests
            error_monitor: Error monitor for tracking API errors
            rate_limit_monitor: Rate limit monitor for tracking API rate limits
            data_cache: Data cache for caching API responses
            fallback_manager: Fallback manager for handling API outages
        """
        # Initialize base connector
        super().__init__(api_key, api_secret)
        
        # Store enhanced components
        self.connection_pool = connection_pool
        self.error_monitor = error_monitor
        self.rate_limit_monitor = rate_limit_monitor
        self.data_cache = data_cache
        self.fallback_manager = fallback_manager
        
        # Create components if not provided
        if not self.connection_pool:
            self.connection_pool = ConnectionPool()
            logger.info("Created default connection pool")
            
        if not self.error_monitor:
            self.error_monitor = ErrorMonitor()
            logger.info("Created default error monitor")
            
        if not self.rate_limit_monitor:
            self.rate_limit_monitor = RateLimitMonitor()
            logger.info("Created default rate limit monitor")
            
        if not self.data_cache:
            self.data_cache = DataCache()
            logger.info("Created default data cache")
            
        if not self.fallback_manager:
            self.fallback_manager = BitvavoFallbackManager(
                cache_handler=self._cache_handler,
                mock_handler=self._mock_handler
            )
            logger.info("Created default fallback manager")
            
        # Register with monitors
        if self.rate_limit_monitor:
            self.rate_limit_monitor.register_connector("bitvavo", self)
            
        # Start monitors
        if self.error_monitor:
            self.error_monitor.start()
            
        if self.rate_limit_monitor:
            self.rate_limit_monitor.start()
            
        logger.info("Enhanced Bitvavo connector initialized")
        
    def initialize(self) -> bool:
        """
        Initialize the connector by testing the connection and loading market data.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Test connection
            time_response = self.get_time()
            if not time_response or 'time' not in time_response:
                logger.error("Failed to get time from Bitvavo API")
                return False
            
            # Load markets
            markets = self.get_markets()
            if not markets:
                logger.error("Failed to get markets from Bitvavo API")
                return False
            
            # Build symbol mapping
            for market in markets:
                if 'market' in market:
                    standard_symbol = self.standardize_symbol(market['market'])
                    self.symbol_mapping[standard_symbol] = market['market']
            
            logger.info(f"Loaded {len(self.symbol_mapping)} markets from Bitvavo")
            return True
        except Exception as e:
            logger.error(f"Error initializing Bitvavo connector: {e}")
            return False
            
    def _request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        Make a request to the Bitvavo API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            
        Returns:
            Dict: Response data
        """
        url = f"{self.base_url}{endpoint}"
        
        # Create cache key
        cache_key = self._create_cache_key(method, endpoint, params, data)
        
        # Check cache
        if method == 'GET' and self.data_cache:
            cached_data = self.data_cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {method} {endpoint}")
                return cached_data
        
        # Check circuit breaker
        if self.error_monitor and self.error_monitor.is_circuit_open("bitvavo", endpoint):
            logger.warning(f"Circuit breaker open for {endpoint}, using fallback")
            
            # Use fallback
            if self.fallback_manager:
                success, result = self.fallback_manager.handle_fallback(
                    "bitvavo",
                    FallbackTrigger.CIRCUIT_OPEN,
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "params": params,
                        "data": data
                    }
                )
                
                if success:
                    return result
            
            return {'error': 'Circuit breaker open'}
        
        # Add authentication if API key is provided
        headers = {}
        if self.api_key:
            timestamp = int(time.time() * 1000)
            signature = self._generate_signature(timestamp, method, endpoint, data)
            
            headers.update({
                'Bitvavo-Access-Key': self.api_key,
                'Bitvavo-Access-Timestamp': str(timestamp),
                'Bitvavo-Access-Signature': signature
            })
        
        try:
            # Use connection pool if available
            if self.connection_pool:
                response = self.connection_pool.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                    session_name="bitvavo"
                )
            else:
                # Fall back to base implementation
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers
                )
            
            # Handle response
            result = self._handle_response(response)
            
            # Cache result if successful
            if method == 'GET' and self.data_cache and 'error' not in result:
                # Determine TTL based on endpoint
                ttl = self._get_cache_ttl(endpoint)
                self.data_cache.set(cache_key, result, ttl)
            
            return result
        except requests.exceptions.Timeout:
            # Handle timeout
            logger.warning(f"Request timeout for {method} {endpoint}")
            
            # Register error
            if self.error_monitor:
                self.error_monitor.register_error(
                    "bitvavo",
                    "timeout",
                    f"Request timeout for {method} {endpoint}"
                )
            
            # Use fallback
            if self.fallback_manager:
                success, result = self.fallback_manager.handle_fallback(
                    "bitvavo",
                    FallbackTrigger.TIMEOUT,
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "params": params,
                        "data": data
                    }
                )
                
                if success:
                    return result
            
            return {'error': 'Request timeout'}
        except requests.exceptions.ConnectionError:
            # Handle connection error
            logger.warning(f"Connection error for {method} {endpoint}")
            
            # Register error
            if self.error_monitor:
                self.error_monitor.register_error(
                    "bitvavo",
                    "connection_error",
                    f"Connection error for {method} {endpoint}"
                )
            
            # Use fallback
            if self.fallback_manager:
                success, result = self.fallback_manager.handle_fallback(
                    "bitvavo",
                    FallbackTrigger.CONNECTION_ERROR,
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "params": params,
                        "data": data
                    }
                )
                
                if success:
                    return result
            
            return {'error': 'Connection error'}
        except Exception as e:
            # Handle other errors
            logger.error(f"Error making request: {e}")
            
            # Register error
            if self.error_monitor:
                self.error_monitor.register_error(
                    "bitvavo",
                    "request_error",
                    str(e)
                )
            
            return {'error': str(e)}
            
    def _handle_response(self, response: requests.Response) -> Dict:
        """
        Handle API response, including rate limiting and error handling.
        
        Args:
            response: Response from API request
            
        Returns:
            Dict: Response data
        """
        # Update rate limit info
        if 'Bitvavo-Ratelimit-Remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['Bitvavo-Ratelimit-Remaining'])
        
        if 'Bitvavo-Ratelimit-ResetAt' in response.headers:
            self.rate_limit_reset = int(response.headers['Bitvavo-Ratelimit-ResetAt'])
        
        # Check for rate limiting
        if self.rate_limit_remaining <= 0:
            reset_time = self.rate_limit_reset / 1000  # Convert to seconds
            current_time = time.time()
            sleep_time = max(0, reset_time - current_time)
            
            logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
            
            # Register error
            if self.error_monitor:
                self.error_monitor.register_error(
                    "bitvavo",
                    "rate_limit",
                    f"Rate limit reached, reset in {sleep_time:.2f} seconds"
                )
            
            # Use fallback
            if self.fallback_manager:
                success, result = self.fallback_manager.handle_fallback(
                    "bitvavo",
                    FallbackTrigger.RATE_LIMIT,
                    {
                        "reset_time": reset_time
                    }
                )
                
                if success:
                    return result
            
            time.sleep(sleep_time)
        
        # Check for errors
        if response.status_code != 200:
            error_message = response.text
            
            logger.error(f"API error: {response.status_code} - {error_message}")
            
            # Register error
            if self.error_monitor:
                self.error_monitor.register_error(
                    "bitvavo",
                    "api_error",
                    f"API error: {response.status_code} - {error_message}"
                )
            
            # Use fallback
            if self.fallback_manager:
                success, result = self.fallback_manager.handle_fallback(
                    "bitvavo",
                    FallbackTrigger.API_ERROR,
                    {
                        "status_code": response.status_code,
                        "error_message": error_message
                    }
                )
                
                if success:
                    return result
            
            return {'error': error_message}
        
        # Parse response
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            
            # Register error
            if self.error_monitor:
                self.error_monitor.register_error(
                    "bitvavo",
                    "parse_error",
                    f"Error parsing response: {e}"
                )
            
            return {'error': 'Error parsing response'}
            
    def _create_cache_key(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> str:
        """
        Create a cache key for a request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            
        Returns:
            str: Cache key
        """
        # Create key components
        key_parts = [
            "bitvavo",
            method,
            endpoint
        ]
        
        # Add params
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            key_parts.append(param_str)
        
        # Add data
        if data:
            data_str = json.dumps(data, sort_keys=True)
            key_parts.append(data_str)
        
        # Join parts
        return ":".join(key_parts)
        
    def _get_cache_ttl(self, endpoint: str) -> int:
        """
        Get cache TTL for an endpoint.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            int: Cache TTL in seconds
        """
        # Define TTLs for different endpoints
        ttls = {
            '/time': 60,  # 1 minute
            '/markets': 3600,  # 1 hour
            '/ticker/price': 10,  # 10 seconds
            '/orderbook': 5,  # 5 seconds
            '/trades': 30,  # 30 seconds
            '/candles': 60,  # 1 minute
            '/balance': 30,  # 30 seconds
            '/account': 300,  # 5 minutes
            '/orders': 10,  # 10 seconds
        }
        
        # Get TTL for endpoint
        for pattern, ttl in ttls.items():
            if endpoint.startswith(pattern):
                return ttl
        
        # Default TTL
        return 60  # 1 minute
        
    def _cache_handler(self, source: str, request_data: Dict[str, Any]) -> Any:
        """
        Cache handler for fallback manager.
        
        Args:
            source: Source name
            request_data: Request data
            
        Returns:
            Any: Cached data or None
        """
        if not self.data_cache:
            return None
            
        # Create cache key
        method = request_data.get('method', 'GET')
        endpoint = request_data.get('endpoint', '')
        params = request_data.get('params', None)
        data = request_data.get('data', None)
        
        cache_key = self._create_cache_key(method, endpoint, params, data)
        
        # Get from cache
        return self.data_cache.get(cache_key)
        
    def _mock_handler(self, source: str, request_data: Dict[str, Any]) -> Any:
        """
        Mock handler for fallback manager.
        
        Args:
            source: Source name
            request_data: Request data
            
        Returns:
            Any: Mock data
        """
        # Get request details
        method = request_data.get('method', 'GET')
        endpoint = request_data.get('endpoint', '')
        
        # Generate mock data based on endpoint
        if endpoint == '/time':
            return {'time': int(time.time() * 1000)}
        elif endpoint == '/markets':
            return [
                {'market': 'BTC-EUR', 'status': 'trading'},
                {'market': 'ETH-EUR', 'status': 'trading'},
                {'market': 'XRP-EUR', 'status': 'trading'}
            ]
        elif endpoint == '/ticker/price':
            return {'market': 'BTC-EUR', 'price': '50000'}
        elif endpoint == '/orderbook':
            return {
                'market': 'BTC-EUR',
                'bids': [['50000', '1.0']],
                'asks': [['51000', '1.0']]
            }
        elif endpoint == '/trades':
            return [
                {
                    'id': '123456',
                    'timestamp': int(time.time() * 1000),
                    'amount': '1.0',
                    'price': '50000',
                    'side': 'buy'
                }
            ]
        elif endpoint == '/candles':
            return [
                [
                    int(time.time() * 1000),  # timestamp
                    '50000',  # open
                    '51000',  # high
                    '49000',  # low
                    '50500',  # close
                    '100'  # volume
                ]
            ]
        elif endpoint == '/balance':
            return [
                {'symbol': 'EUR', 'available': '10000', 'inOrder': '0'},
                {'symbol': 'BTC', 'available': '1.0', 'inOrder': '0'}
            ]
        elif endpoint == '/account':
            return {'fees': {'taker': '0.0025', 'maker': '0.0015'}}
        elif endpoint == '/orders':
            return []
        else:
            return None
            
    def close(self):
        """Close the connector and release resources."""
        # Close connection pool
        if self.connection_pool:
            self.connection_pool.close()
            
        # Stop monitors
        if self.error_monitor:
            self.error_monitor.stop()
            
        if self.rate_limit_monitor:
            self.rate_limit_monitor.stop()
            
        logger.info("Enhanced Bitvavo connector closed")