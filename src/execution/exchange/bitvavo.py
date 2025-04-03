"""
Bitvavo Exchange Connector

This module provides a connector for the Bitvavo cryptocurrency exchange.
"""

import time
import hmac
import hashlib
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BitvavoConnector:
    """
    Connector for the Bitvavo cryptocurrency exchange.
    
    This class provides methods for interacting with the Bitvavo API,
    including authentication, market data, and trading operations.
    """
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        """
        Initialize the Bitvavo connector.
        
        Args:
            api_key: Bitvavo API key
            api_secret: Bitvavo API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.bitvavo.com/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Trading-Agent/1.0'
        })
        
        if api_key:
            self.session.headers.update({
                'Bitvavo-Access-Key': api_key
            })
        
        # Rate limiting
        self.rate_limit_remaining = 1000
        self.rate_limit_reset = 0
        
        # Symbol mapping
        self.symbol_mapping = {}
        
        logger.info("Bitvavo connector initialized")
    
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
    
    def _generate_signature(self, timestamp: int, method: str, url_path: str, body: Dict = None) -> str:
        """
        Generate HMAC-SHA256 signature for API request.
        
        Args:
            timestamp: Current timestamp in milliseconds
            method: HTTP method (GET, POST, etc.)
            url_path: API endpoint path
            body: Request body (for POST requests)
            
        Returns:
            str: HMAC-SHA256 signature
        """
        if body is None:
            body = {}
        
        # Create signature string
        signature_string = str(timestamp) + method + url_path
        if body:
            signature_string += json.dumps(body)
        
        # Create HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
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
            time.sleep(sleep_time)
        
        # Check for errors
        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return {'error': response.text}
        
        # Parse response
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {'error': 'Error parsing response'}
    
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
        
        # Add authentication if API key is provided
        headers = {}
        if self.api_key:
            timestamp = int(time.time() * 1000)
            signature = self._generate_signature(timestamp, method, endpoint, data)
            
            headers.update({
                'Bitvavo-Access-Timestamp': str(timestamp),
                'Bitvavo-Access-Signature': signature
            })
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers
            )
            
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Error making request: {e}")
            return {'error': str(e)}
    
    def get_time(self) -> Dict:
        """
        Get server time from Bitvavo API.
        
        Returns:
            Dict: Server time
        """
        return self._request('GET', '/time')
    
    def get_markets(self) -> List[Dict]:
        """
        Get available markets from Bitvavo API.
        
        Returns:
            List[Dict]: Available markets
        """
        return self._request('GET', '/markets')
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get ticker for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/EUR)
            
        Returns:
            Dict: Ticker data
        """
        bitvavo_symbol = self.normalize_symbol(symbol)
        return self._request('GET', f'/ticker/price?market={bitvavo_symbol}')
    
    def get_order_book(self, symbol: str, depth: int = 25) -> Dict:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/EUR)
            depth: Order book depth
            
        Returns:
            Dict: Order book data
        """
        bitvavo_symbol = self.normalize_symbol(symbol)
        return self._request('GET', f'/orderbook?market={bitvavo_symbol}&depth={depth}')
    
    def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/EUR)
            limit: Number of trades to return
            
        Returns:
            List[Dict]: Recent trades
        """
        bitvavo_symbol = self.normalize_symbol(symbol)
        return self._request('GET', f'/trades?market={bitvavo_symbol}&limit={limit}')
    
    def get_candles(self, symbol: str, interval: str = '1h', limit: int = 100) -> List[List]:
        """
        Get candles for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/EUR)
            interval: Candle interval (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d)
            limit: Number of candles to return
            
        Returns:
            List[List]: Candles data
        """
        bitvavo_symbol = self.normalize_symbol(symbol)
        return self._request('GET', f'/candles?market={bitvavo_symbol}&interval={interval}&limit={limit}')
    
    def get_balance(self) -> List[Dict]:
        """
        Get account balance.
        
        Returns:
            List[Dict]: Account balance
        """
        return self._request('GET', '/balance')
    
    def get_account(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Dict: Account information
        """
        return self._request('GET', '/account')
    
    def create_order(self, symbol: str, side: str, order_type: str, amount: float, price: float = None) -> Dict:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/EUR)
            side: Order side (buy or sell)
            order_type: Order type (limit or market)
            amount: Order amount
            price: Order price (required for limit orders)
            
        Returns:
            Dict: Order data
        """
        bitvavo_symbol = self.normalize_symbol(symbol)
        
        data = {
            'market': bitvavo_symbol,
            'side': side,
            'orderType': order_type,
            'amount': str(amount)
        }
        
        if order_type == 'limit' and price is not None:
            data['price'] = str(price)
        
        return self._request('POST', '/order', data=data)
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """
        Cancel an order.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/EUR)
            order_id: Order ID
            
        Returns:
            Dict: Cancellation result
        """
        bitvavo_symbol = self.normalize_symbol(symbol)
        return self._request('DELETE', f'/order?market={bitvavo_symbol}&orderId={order_id}')
    
    def get_orders(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get open orders for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/EUR)
            limit: Number of orders to return
            
        Returns:
            List[Dict]: Open orders
        """
        bitvavo_symbol = self.normalize_symbol(symbol)
        return self._request('GET', f'/orders?market={bitvavo_symbol}&limit={limit}')
    
    def get_order(self, symbol: str, order_id: str) -> Dict:
        """
        Get order details.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/EUR)
            order_id: Order ID
            
        Returns:
            Dict: Order details
        """
        bitvavo_symbol = self.normalize_symbol(symbol)
        return self._request('GET', f'/order?market={bitvavo_symbol}&orderId={order_id}')
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize a symbol to Bitvavo format.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/EUR)
            
        Returns:
            str: Normalized symbol (e.g., BTC-EUR)
        """
        if '/' in symbol:
            base, quote = symbol.split('/')
            return f"{base}-{quote}"
        return symbol
    
    def standardize_symbol(self, symbol: str) -> str:
        """
        Standardize a Bitvavo symbol to standard format.
        
        Args:
            symbol: Bitvavo symbol (e.g., BTC-EUR)
            
        Returns:
            str: Standardized symbol (e.g., BTC/EUR)
        """
        if '-' in symbol:
            base, quote = symbol.split('-')
            return f"{base}/{quote}"
        return symbol