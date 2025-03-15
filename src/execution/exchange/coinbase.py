"""Coinbase Advanced Trade API connector implementation.

This module provides a concrete implementation of the BaseExchangeConnector
for the Coinbase cryptocurrency exchange, using their Advanced Trade API.
"""

import asyncio
import base64
import hmac
import hashlib
import json
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urlencode

try:
    import aiohttp
except ImportError:
    raise ImportError("aiohttp is required for CoinbaseExchangeConnector. Install it with 'pip install aiohttp'")

from src.execution.exchange.base import BaseExchangeConnector
from src.models.order import Order, OrderType, OrderStatus, OrderSide, TimeInForce
from src.common.logging import get_logger

logger = get_logger("exchange.coinbase")


class CoinbaseExchangeConnector(BaseExchangeConnector):
    """Coinbase exchange connector implementation.
    
    This class provides methods to interact with the Coinbase Advanced Trade API
    for trading and market data operations.
    """
    
    # API endpoints for Coinbase Advanced Trade API
    BASE_URL = "https://api.coinbase.com"
    API_V3 = "/api/v3"
    
    # Public endpoints
    PRODUCTS_ENDPOINT = "/api/v3/brokerage/products"
    PRODUCT_BOOK_ENDPOINT = "/api/v3/brokerage/product_book"
    MARKET_TRADES_ENDPOINT = "/api/v3/brokerage/market_trades"
    PRODUCT_CANDLES_ENDPOINT = "/api/v3/brokerage/products/{product_id}/candles"
    TICKER_ENDPOINT = "/api/v3/brokerage/products/{product_id}/ticker"
    
    # Private endpoints
    ACCOUNTS_ENDPOINT = "/api/v3/brokerage/accounts"
    CREATE_ORDER_ENDPOINT = "/api/v3/brokerage/orders"
    CANCEL_ORDERS_ENDPOINT = "/api/v3/brokerage/orders/batch_cancel"
    LIST_ORDERS_ENDPOINT = "/api/v3/brokerage/orders/historical/batch"
    GET_ORDER_ENDPOINT = "/api/v3/brokerage/orders/historical/{order_id}"
    FILLS_ENDPOINT = "/api/v3/brokerage/orders/historical/fills"
    
    # Order status mapping
    ORDER_STATUS_MAP = {
        "OPEN": OrderStatus.OPEN,
        "PENDING": OrderStatus.PENDING,
        "FILLED": OrderStatus.FILLED,
        "CANCELLED": OrderStatus.CANCELLED,
        "EXPIRED": OrderStatus.EXPIRED,
        "FAILED": OrderStatus.REJECTED,
    }
    
    # Reverse order status mapping
    REVERSE_ORDER_STATUS_MAP = {v: k for k, v in ORDER_STATUS_MAP.items()}
    
    # Order side mapping
    ORDER_SIDE_MAP = {
        OrderSide.BUY: "BUY",
        OrderSide.SELL: "SELL"
    }
    
    # Order type mapping
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "MARKET",
        OrderType.LIMIT: "LIMIT",
        OrderType.STOP: "STOP",
        OrderType.STOP_LIMIT: "STOP_LIMIT"
    }
    
    # Time in force mapping
    TIME_IN_FORCE_MAP = {
        TimeInForce.GTC: "GOOD_UNTIL_CANCELLED",
        TimeInForce.IOC: "IMMEDIATE_OR_CANCEL",
        TimeInForce.FOK: "FILL_OR_KILL",
        TimeInForce.DAY: "GOOD_UNTIL_DATE"
    }
    
    def __init__(self, 
                 exchange_id: str = "coinbase", 
                 api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None,
                 sandbox: bool = False):
        """Initialize the Coinbase exchange connector.
        
        Args:
            exchange_id: Unique identifier for this exchange (default: "coinbase")
            api_key: Coinbase API key for authenticated requests
            api_secret: Coinbase API secret for authenticated requests
            sandbox: Whether to use the Coinbase sandbox environment
        """
        super().__init__(exchange_id, api_key, api_secret)
        
        # Use sandbox if specified
        if sandbox:
            self.BASE_URL = "https://api-public.sandbox.exchange.coinbase.com"
        
        # API rate limits
        self.request_weight_per_second = 10
        self.orders_per_second = 5
        
        # Session for API requests
        self._session = None
        self._initialized = False
        
        # Cache for exchange information
        self._exchange_info = {}
        self._products = {}
        
        # Request timestamps for rate limiting
        self._last_request_timestamps = []
        
    async def initialize(self) -> bool:
        """Initialize the exchange connector.
        
        Establishes connection to Coinbase and validates API credentials.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Create a new aiohttp session if needed
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            
            # Get products information
            products_info = await self._public_request("GET", self.PRODUCTS_ENDPOINT)
            
            if not products_info or "products" not in products_info:
                logger.error("Failed to retrieve products information")
                return False
            
            # Build product info cache
            for product in products_info["products"]:
                symbol = product["product_id"]
                self._products[symbol] = product
            
            # Test authenticated endpoints if API key provided
            if self.api_key and self.api_secret:
                try:
                    accounts = await self._private_request("GET", self.ACCOUNTS_ENDPOINT)
                    if "accounts" not in accounts:
                        logger.warning("Failed to validate API credentials")
                        return False
                except Exception as e:
                    logger.warning(f"Failed to validate API credentials: {str(e)}")
                    return False
            
            self._initialized = True
            logger.info(f"Initialized Coinbase connector with {len(self._products)} products")
            return True
            
        except Exception as e:
            self._initialized = False
            logger.error(f"Error initializing Coinbase connector: {str(e)}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the exchange connector.
        
        Closes aiohttp session and cleans up resources.
        """
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._initialized = False
        logger.info("Coinbase connector shut down")
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information and trading rules.
        
        Returns:
            Dict containing exchange information
        """
        if not self._products:
            products_info = await self._public_request("GET", self.PRODUCTS_ENDPOINT)
            if products_info and "products" in products_info:
                for product in products_info["products"]:
                    symbol = product["product_id"]
                    self._products[symbol] = product
        
        return {
            "timezone": "UTC",
            "serverTime": int(time.time() * 1000),
            "symbols": list(self._products.values()),
            "exchangeFilters": [],
            "rateLimits": [
                {
                    "rateLimitType": "REQUEST_WEIGHT",
                    "interval": "SECOND",
                    "intervalNum": 1,
                    "limit": self.request_weight_per_second
                },
                {
                    "rateLimitType": "ORDERS",
                    "interval": "SECOND",
                    "intervalNum": 1,
                    "limit": self.orders_per_second
                }
            ]
        }
    
    async def get_account_balance(self) -> Dict[str, Decimal]:
        """Get account balances for all assets.
        
        Returns:
            Dict mapping asset symbol to balance amount
        """
        if not (self.api_key and self.api_secret):
            logger.error("API credentials required for account balance")
            return {}
        
        try:
            accounts_response = await self._private_request("GET", self.ACCOUNTS_ENDPOINT)
            
            balances = {}
            if "accounts" in accounts_response:
                for account in accounts_response["accounts"]:
                    currency = account["currency"]
                    available = Decimal(account.get("available_balance", {}).get("value", "0"))
                    hold = Decimal(account.get("hold", {}).get("value", "0"))
                    total = available + hold
                    balances[currency] = total
            
            return balances
            
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            
        Returns:
            Dict containing ticker information
        """
        normalized_symbol = self.normalize_symbol(symbol)
        
        try:
            endpoint = self.TICKER_ENDPOINT.format(product_id=normalized_symbol)
            ticker = await self._public_request("GET", endpoint)
            
            if not ticker:
                logger.error(f"Failed to get ticker for {symbol}")
                return {}
            
            # Format ticker data to match our standard format
            return {
                "symbol": normalized_symbol,
                "price": ticker.get("price", "0"),
                "volume": ticker.get("volume_24h", "0"),
                "high": ticker.get("high_24h", "0"),
                "low": ticker.get("low_24h", "0"),
                "timestamp": ticker.get("time", datetime.utcnow().isoformat())
            }
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            return {}
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get current order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            limit: Maximum number of bids/asks to return
            
        Returns:
            Dict containing order book data
        """
        normalized_symbol = self.normalize_symbol(symbol)
        
        try:
            params = {
                "product_id": normalized_symbol,
                "limit": min(limit, 1000)  # Coinbase max limit is 1000
            }
            
            orderbook = await self._public_request("GET", self.PRODUCT_BOOK_ENDPOINT, params=params)
            
            if not orderbook:
                logger.error(f"Failed to get orderbook for {symbol}")
                return {}
            
            # Format orderbook data to match our standard format
            result = {
                "symbol": normalized_symbol,
                "timestamp": int(time.time() * 1000),
                "bids": [],
                "asks": []
            }
            
            if "bids" in orderbook:
                result["bids"] = [[Decimal(bid["price"]), Decimal(bid["size"])] for bid in orderbook["bids"]]
            
            if "asks" in orderbook:
                result["asks"] = [[Decimal(ask["price"]), Decimal(ask["size"])] for ask in orderbook["asks"]]
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {str(e)}")
            return {}
    
    async def create_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[str]]:
        """Submit an order to the exchange.
        
        Args:
            order: Order object containing order details
            
        Returns:
            Tuple of (success, exchange_order_id, error_message)
        """
        if not (self.api_key and self.api_secret):
            return False, None, "API credentials required for creating orders"
        
        try:
            # Normalize symbol
            normalized_symbol = self.normalize_symbol(order.symbol)
            
            # Prepare order request
            order_data = {
                "client_order_id": order.client_order_id or order.id,
                "product_id": normalized_symbol,
                "side": self.ORDER_SIDE_MAP[order.side],
                "order_configuration": {}
            }
            
            # Set order type specific configuration
            if order.order_type == OrderType.MARKET:
                order_data["order_configuration"]["market_market_ioc"] = {
                    "quote_size" if order.side == OrderSide.BUY else "base_size": str(order.quantity)
                }
            elif order.order_type == OrderType.LIMIT:
                if not order.price:
                    return False, None, "Price is required for limit orders"
                
                time_in_force = self.TIME_IN_FORCE_MAP.get(order.time_in_force, "GOOD_UNTIL_CANCELLED")
                
                order_data["order_configuration"]["limit_limit_gtc"] = {
                    "base_size": str(order.quantity),
                    "limit_price": str(order.price),
                    "post_only": order.is_post_only
                }
            elif order.order_type == OrderType.STOP:
                if not order.stop_price:
                    return False, None, "Stop price is required for stop orders"
                
                order_data["order_configuration"]["stop_limit_stop_limit_gtc"] = {
                    "base_size": str(order.quantity),
                    "limit_price": str(order.price or order.stop_price),
                    "stop_price": str(order.stop_price),
                    "stop_direction": "STOP_DIRECTION_STOP_DOWN" if order.side == OrderSide.SELL else "STOP_DIRECTION_STOP_UP"
                }
            else:
                return False, None, f"Unsupported order type: {order.order_type}"
            
            # Submit the order
            response = await self._private_request("POST", self.CREATE_ORDER_ENDPOINT, data=order_data)
            
            if not response or "success" not in response or not response["success"]:
                error_msg = response.get("error_response", {}).get("message", "Unknown error")
                return False, None, f"Order creation failed: {error_msg}"
            
            if "order_id" not in response:
                return False, None, "Order created but no order ID returned"
            
            return True, response["order_id"], None
            
        except Exception as e:
            logger.error(f"Error creating order: {str(e)}")
            return False, None, f"Error creating order: {str(e)}"
    
    async def cancel_order(self, order_id: str, symbol: str) -> Tuple[bool, Optional[str]]:
        """Cancel an existing order.
        
        Args:
            order_id: Exchange order ID to cancel
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (success, error_message)
        """
        if not (self.api_key and self.api_secret):
            return False, "API credentials required for cancelling orders"
        
        try:
            # Prepare cancel request
            cancel_data = {
                "order_ids": [order_id]
            }
            
            # Submit the cancel request
            response = await self._private_request("POST", self.CANCEL_ORDERS_ENDPOINT, data=cancel_data)
            
            if not response or "success" not in response or not response["success"]:
                error_msg = response.get("error_response", {}).get("message", "Unknown error")
                return False, f"Order cancellation failed: {error_msg}"
            
            # Check if there were any failures in the response
            results = response.get("results", [])
            if not results:
                return False, "No results returned from cancel request"
            
            for result in results:
                if result.get("order_id") == order_id:
                    if result.get("success"):
                        return True, None
                    else:
                        return False, result.get("failure_reason", "Unknown failure reason")
            
            return False, "Order ID not found in cancel response"
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False, f"Error cancelling order: {str(e)}"
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific order.
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            
        Returns:
            Dict containing order information or None if not found
        """
        if not (self.api_key and self.api_secret):
            logger.error("API credentials required for getting order details")
            return None
        
        try:
            endpoint = self.GET_ORDER_ENDPOINT.format(order_id=order_id)
            response = await self._private_request("GET", endpoint)
            
            if not response or "order" not in response:
                logger.error(f"Failed to get order {order_id}")
                return None
            
            order_data = response["order"]
            
            # Map the response to our standard format
            status = self.ORDER_STATUS_MAP.get(order_data.get("status", "UNKNOWN"), OrderStatus.PENDING)
            
            # Extract filled quantity and price
            filled_quantity = Decimal("0")
            avg_price = None
            
            if "filled_size" in order_data:
                filled_quantity = Decimal(order_data["filled_size"])
            
            if "average_filled_price" in order_data and order_data["average_filled_price"] != "0":
                avg_price = Decimal(order_data["average_filled_price"])
            
            return {
                "symbol": order_data.get("product_id", symbol),
                "order_id": order_id,
                "client_order_id": order_data.get("client_order_id"),
                "price": Decimal(order_data.get("limit_price", "0")),
                "original_quantity": Decimal(order_data.get("base_size", "0")),
                "executed_quantity": filled_quantity,
                "status": status,
                "side": OrderSide.BUY if order_data.get("side") == "BUY" else OrderSide.SELL,
                "type": OrderType.LIMIT if "limit_price" in order_data else OrderType.MARKET,
                "time_in_force": TimeInForce.GTC,  # Default since Coinbase doesn't always provide this
                "created_at": datetime.fromisoformat(order_data.get("created_time", "").replace("Z", "+00:00")),
                "updated_at": datetime.fromisoformat(order_data.get("created_time", "").replace("Z", "+00:00")),
                "average_price": avg_price,
                "fills": []  # Would need a separate request to get fills
            }
            
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {str(e)}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders.
        
        Args:
            symbol: Optional trading pair symbol to filter by
            
        Returns:
            List of dictionaries containing order information
        """
        if not (self.api_key and self.api_secret):
            logger.error("API credentials required for getting open orders")
            return []
        
        try:
            params = {
                "order_status": ["OPEN", "PENDING"],
                "limit": 100  # Maximum number of orders to return
            }
            
            if symbol:
                params["product_id"] = self.normalize_symbol(symbol)
            
            response = await self._private_request("GET", self.LIST_ORDERS_ENDPOINT, params=params)
            
            if not response or "orders" not in response:
                logger.error("Failed to get open orders")
                return []
            
            orders = []
            for order_data in response["orders"]:
                # Map the response to our standard format
                status = self.ORDER_STATUS_MAP.get(order_data.get("status", "UNKNOWN"), OrderStatus.PENDING)
                
                # Extract filled quantity and price
                filled_quantity = Decimal("0")
                avg_price = None
                
                if "filled_size" in order_data:
                    filled_quantity = Decimal(order_data["filled_size"])
                
                if "average_filled_price" in order_data and order_data["average_filled_price"] != "0":
                    avg_price = Decimal(order_data["average_filled_price"])
                
                orders.append({
                    "symbol": order_data.get("product_id"),
                    "order_id": order_data.get("order_id"),
                    "client_order_id": order_data.get("client_order_id"),
                    "price": Decimal(order_data.get("limit_price", "0")),
                    "original_quantity": Decimal(order_data.get("base_size", "0")),
                    "executed_quantity": filled_quantity,
                    "status": status,
                    "side": OrderSide.BUY if order_data.get("side") == "BUY" else OrderSide.SELL,
                    "type": OrderType.LIMIT if "limit_price" in order_data else OrderType.MARKET,
                    "time_in_force": TimeInForce.GTC,  # Default since Coinbase doesn't always provide this
                    "created_at": datetime.fromisoformat(order_data.get("created_time", "").replace("Z", "+00:00")),
                    "updated_at": datetime.fromisoformat(order_data.get("created_time", "").replace("Z", "+00:00")),
                    "average_price": avg_price
                })
            
            return orders
            
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return []
    
    async def get_trade_history(
        self, 
        symbol: str, 
        limit: int = 100,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get trade history for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of trades to return
            from_time: Optional start time for trades
            to_time: Optional end time for trades
            
        Returns:
            List of dictionaries containing trade information
        """
        normalized_symbol = self.normalize_symbol(symbol)
        
        try:
            params = {
                "product_id": normalized_symbol,
                "limit": min(limit, 1000)  # Coinbase max limit is 1000
            }
            
            response = await self._public_request("GET", self.MARKET_TRADES_ENDPOINT, params=params)
            
            if not response or "trades" not in response:
                logger.error(f"Failed to get trade history for {symbol}")
                return []
            
            trades = []
            for trade in response["trades"]:
                trade_time = datetime.fromisoformat(trade.get("time", "").replace("Z", "+00:00"))
                
                # Filter by time if specified
                if from_time and trade_time < from_time:
                    continue
                if to_time and trade_time > to_time:
                    continue
                
                trades.append({
                    "id": trade.get("trade_id"),
                    "symbol": normalized_symbol,
                    "price": Decimal(trade.get("price", "0")),
                    "quantity": Decimal(trade.get("size", "0")),
                    "time": trade_time,
                    "side": OrderSide.BUY if trade.get("side") == "BUY" else OrderSide.SELL,
                    "is_best_match": True
                })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trade history for {symbol}: {str(e)}")
            return []
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize a symbol to Coinbase format.
        
        Args:
            symbol: Symbol in standard format (e.g., "BTC/USD")
            
        Returns:
            Symbol in Coinbase format (e.g., "BTC-USD")
        """
        return symbol.replace("/", "-")
    
    def standardize_symbol(self, symbol: str) -> str:
        """Convert a symbol from Coinbase format to standard format.
        
        Args:
            symbol: Symbol in Coinbase format (e.g., "BTC-USD")
            
        Returns:
            Symbol in standard format (e.g., "BTC/USD")
        """
        return symbol.replace("-", "/")
    
    async def _public_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a public request to the Coinbase API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Optional query parameters
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        # Add query parameters if provided
        if params and method == "GET":
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query_string}"
        
        try:
            # Ensure we have a session
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            
            # Make the request
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            data = None
            if method != "GET" and params:
                data = json.dumps(params)
            
            response = await self._session.request(method, url, headers=headers, data=data)
            
            # Check for HTTP errors
            if response.status >= 400:
                error_text = await response.text()
                logger.error(f"Coinbase API error ({response.status}): {error_text}")
                return {}
            
            # Parse JSON response
            response_data = await response.json()
            return response_data
            
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during Coinbase public request: {str(e)}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error during Coinbase public request: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error during Coinbase public request: {str(e)}")
            return {}
    
    async def _private_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                              data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an authenticated request to the Coinbase API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Optional query parameters
            data: Optional request body
            
        Returns:
            Response data as dictionary
        """
        if not (self.api_key and self.api_secret):
            logger.error("API key and secret required for private requests")
            return {}
        
        url = f"{self.BASE_URL}{endpoint}"
        
        # Add query parameters if provided
        if params and method == "GET":
            query_string = urlencode(params)
            url = f"{url}?{query_string}"
        
        try:
            # Ensure we have a session
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            
            # Prepare the request
            timestamp = str(int(time.time()))
            request_body = ""
            
            if data:
                request_body = json.dumps(data)
            
            # Create the message to sign
            message = timestamp + method + endpoint
            if request_body:
                message += request_body
            
            # Sign the message
            signature = self._generate_signature(message)
            
            # Set headers
            headers = {
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Make the request
            response = await self._session.request(method, url, headers=headers, data=request_body if data else None)
            
            # Check for HTTP errors
            if response.status >= 400:
                error_text = await response.text()
                logger.error(f"Coinbase API error ({response.status}): {error_text}")
                return {}
            
            # Parse JSON response
            response_data = await response.json()
            return response_data
            
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during Coinbase private request: {str(e)}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error during Coinbase private request: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error during Coinbase private request: {str(e)}")
            return {}
    
    def _generate_signature(self, message: str) -> str:
        """Generate HMAC-SHA256 signature for Coinbase API authentication.
        
        Args:
            message: Message to sign
            
        Returns:
            Base64-encoded HMAC-SHA256 signature
        """
        if not self.api_secret:
            raise ValueError("API secret is required for generating signatures")
            
        secret_bytes = base64.b64decode(self.api_secret)
        message_bytes = message.encode('utf-8')
        
        hmac_signature = hmac.new(secret_bytes, message_bytes, hashlib.sha256)
        return base64.b64encode(hmac_signature.digest()).decode('utf-8') 