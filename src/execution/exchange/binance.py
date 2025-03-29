"""Binance exchange connector implementation.

This module provides a concrete implementation of the BaseExchangeConnector
for the Binance cryptocurrency exchange.
"""

import asyncio
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
    raise ImportError("aiohttp is required for BinanceExchangeConnector. Install it with 'pip install aiohttp'")

from src.execution.exchange.base import BaseExchangeConnector
from src.models.order import Order, OrderType, OrderStatus, OrderSide, TimeInForce


class BinanceExchangeConnector(BaseExchangeConnector):
    """Binance exchange connector implementation.
    
    This class provides methods to interact with the Binance API for
    trading and market data operations.
    """
    
    # API endpoints
    BASE_URL = "https://api.binance.com"
    API_V3 = "/api/v3"
    
    # Endpoint paths
    PING_ENDPOINT = "/ping"
    TIME_ENDPOINT = "/time"
    EXCHANGE_INFO_ENDPOINT = "/exchangeInfo"
    DEPTH_ENDPOINT = "/depth"
    TRADES_ENDPOINT = "/trades"
    HISTORICAL_TRADES = "/historicalTrades"
    TICKER_ENDPOINT = "/ticker/24hr"
    TICKER_PRICE_ENDPOINT = "/ticker/price"
    TICKER_BOOK_ENDPOINT = "/ticker/bookTicker"
    
    # Account and order endpoints (requires authentication)
    ACCOUNT_ENDPOINT = "/account"
    ORDER_ENDPOINT = "/order"
    OPEN_ORDERS_ENDPOINT = "/openOrders"
    ALL_ORDERS_ENDPOINT = "/allOrders"
    TEST_ORDER_ENDPOINT = "/order/test"
    
    def __init__(self, 
                 exchange_id: str = "binance", 
                 api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None,
                 testnet: bool = False):
        """Initialize the Binance exchange connector.
        
        Args:
            exchange_id: Unique identifier for this exchange (default: "binance")
            api_key: Binance API key for authenticated requests
            api_secret: Binance API secret for authenticated requests
            testnet: Whether to use the Binance testnet instead of production
        """
        super().__init__(exchange_id, api_key, api_secret)
        
        # Use testnet if specified
        if testnet:
            self.BASE_URL = "https://testnet.binance.vision"
            
        # Trading rate limits from Binance API
        self.request_weight_per_minute = 1200  # Default weight limit
        self.orders_per_minute = 50            # Orders per minute per symbol
        self.orders_per_second = 10            # Orders per second 
        self.orders_per_day = 160000           # Orders per day
        
        # Session for API requests
        self._session = None
        self._initialized = False
        
        # Cache for exchange information
        self._exchange_info = None
        self._symbol_info = {}
        
        # Request timestamps for rate limiting
        self._last_request_timestamps = []
    
    async def initialize(self) -> bool:
        """Initialize the exchange connector.
        
        Establishes connection to Binance and validates API credentials.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Create a new aiohttp session if needed
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            
            # Test connection with ping request
            await self._public_request(self.PING_ENDPOINT, params=None)
            
            # Get exchange information
            self._exchange_info = await self._public_request(self.EXCHANGE_INFO_ENDPOINT, params=None)
            
            # Build symbol info cache
            if self._exchange_info and "symbols" in self._exchange_info:
                for symbol_data in self._exchange_info["symbols"]:
                    self._symbol_info[symbol_data["symbol"]] = symbol_data
            
            # Test authenticated endpoints if API key provided
            if self.api_key:
                account_info = await self._private_request(self.ACCOUNT_ENDPOINT, params={})
                if "balances" not in account_info:
                    raise ValueError("Invalid API credentials")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self._initialized = False
            print(f"Error initializing Binance connector: {str(e)}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the exchange connector.
        
        Closes the aiohttp session and performs cleanup.
        """
        if self._session and not self._session.closed:
            await self._session.close()
        self._initialized = False
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information and trading rules.
        
        Returns:
            Dict containing exchange information with trading rules
        """
        if not self._exchange_info:
            self._exchange_info = await self._public_request(self.EXCHANGE_INFO_ENDPOINT, params=None)
        return self._exchange_info
    
    async def get_account_balance(self) -> Dict[str, Decimal]:
        """Get account balances for all assets.
        
        Returns:
            Dict mapping asset symbol to balance amount
        """
        account_info = await self._private_request(self.ACCOUNT_ENDPOINT, params={})
        
        # Extract balances and convert to Decimal
        balances = {}
        for balance in account_info.get("balances", []):
            asset = balance["asset"]
            # Combine free and locked balances
            free = Decimal(balance["free"])
            locked = Decimal(balance["locked"])
            total = free + locked
            
            # Only include non-zero balances
            if total > 0:
                balances[asset] = total
        
        return balances
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Dict containing ticker information
        """
        binance_symbol = self.normalize_symbol(symbol)
        params = {"symbol": binance_symbol}
        
        ticker = await self._public_request(self.TICKER_ENDPOINT, params=params)
        
        # Standardize the response format
        return {
            "symbol": symbol,
            "bid": Decimal(str(ticker["bidPrice"])) if "bidPrice" in ticker else None,
            "ask": Decimal(str(ticker["askPrice"])) if "askPrice" in ticker else None,
            "last": Decimal(str(ticker["lastPrice"])) if "lastPrice" in ticker else None,
            "high": Decimal(str(ticker["highPrice"])) if "highPrice" in ticker else None,
            "low": Decimal(str(ticker["lowPrice"])) if "lowPrice" in ticker else None,
            "volume": Decimal(str(ticker["volume"])) if "volume" in ticker else None,
            "quote_volume": Decimal(str(ticker["quoteVolume"])) if "quoteVolume" in ticker else None,
            "change_24h": Decimal(str(ticker["priceChange"])) if "priceChange" in ticker else None,
            "change_percent_24h": Decimal(str(ticker["priceChangePercent"])) if "priceChangePercent" in ticker else None,
            "timestamp": datetime.fromtimestamp(ticker["closeTime"] / 1000) if "closeTime" in ticker else datetime.now(),
        }
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Get current order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            limit: Maximum number of bids/asks to return
            
        Returns:
            Dict containing order book data
        """
        binance_symbol = self.normalize_symbol(symbol)
        
        # Limit must be one of: 5, 10, 20, 50, 100, 500, 1000, 5000
        valid_limits = [5, 10, 20, 50, 100, 500, 1000, 5000]
        adjusted_limit = min(valid_limits, key=lambda x: abs(x - limit))
        
        params = {
            "symbol": binance_symbol,
            "limit": adjusted_limit
        }
        
        orderbook = await self._public_request(self.DEPTH_ENDPOINT, params=params)
        
        # Convert string values to Decimal
        bids = [[Decimal(str(price)), Decimal(str(amount))] for price, amount in orderbook["bids"]]
        asks = [[Decimal(str(price)), Decimal(str(amount))] for price, amount in orderbook["asks"]]
        
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.fromtimestamp(orderbook["lastUpdateId"] / 1000) if "lastUpdateId" in orderbook else datetime.now(),
        }
    
    async def create_order(self, order: Order) -> Tuple[bool, Optional[str], Optional[str]]:
        """Submit an order to the exchange.
        
        Args:
            order: Order object containing order details
            
        Returns:
            Tuple of (success, exchange_order_id, error_message)
        """
        binance_symbol = self.normalize_symbol(order.symbol)
        
        # Build the order parameters
        params = {
            "symbol": binance_symbol,
            "side": order.side.value.upper(),
            "quantity": float(order.quantity),
            "timestamp": int(time.time() * 1000)
        }
        
        # Set client order ID if provided
        if order.client_order_id:
            params["newClientOrderId"] = order.client_order_id
        
        # Map order type and add necessary parameters
        if order.order_type == OrderType.MARKET:
            params["type"] = "MARKET"
        elif order.order_type == OrderType.LIMIT:
            params["type"] = "LIMIT"
            params["price"] = float(order.price) if order.price is not None else None
            
            # Handle time in force
            time_in_force = "GTC"
            if order.time_in_force == TimeInForce.IOC:
                time_in_force = "IOC"
            elif order.time_in_force == TimeInForce.FOK:
                time_in_force = "FOK"
            params["timeInForce"] = time_in_force
            
        elif order.order_type == OrderType.STOP:
            params["type"] = "STOP_LOSS"
            params["stopPrice"] = float(order.stop_price) if order.stop_price is not None else None
        elif order.order_type == OrderType.STOP_LIMIT:
            params["type"] = "STOP_LOSS_LIMIT"
            params["price"] = float(order.price) if order.price is not None else None
            params["stopPrice"] = float(order.stop_price) if order.stop_price is not None else None
            params["timeInForce"] = "GTC"
        elif order.order_type == OrderType.TRAILING_STOP:
            # Binance doesn't directly support trailing stops through the API
            # This would need to be implemented client-side
            return False, None, "Trailing stop orders not directly supported"
        
        # Add post-only flag if needed
        if order.is_post_only and order.order_type == OrderType.LIMIT:
            params["newOrderRespType"] = "ACK"
        
        # Execute the order
        try:
            response = await self._private_request(self.ORDER_ENDPOINT, params=params, method="POST")
            
            # Extract the exchange order ID
            exchange_order_id = response.get("orderId")
            
            return True, exchange_order_id, None
        except Exception as e:
            return False, None, str(e)
    
    async def cancel_order(self, order_id: str, symbol: str) -> Tuple[bool, Optional[str]]:
        """Cancel an existing order.
        
        Args:
            order_id: Exchange order ID to cancel
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (success, error_message)
        """
        binance_symbol = self.normalize_symbol(symbol)
        
        params = {
            "symbol": binance_symbol,
            "orderId": order_id,
            "timestamp": int(time.time() * 1000)
        }
        
        try:
            response = await self._private_request(self.ORDER_ENDPOINT, params=params, method="DELETE")
            return True, None
        except Exception as e:
            return False, str(e)
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific order.
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            
        Returns:
            Dict containing order information or None if not found
        """
        binance_symbol = self.normalize_symbol(symbol)
        
        params = {
            "symbol": binance_symbol,
            "orderId": order_id,
            "timestamp": int(time.time() * 1000)
        }
        
        try:
            order_info = await self._private_request(self.ORDER_ENDPOINT, params=params)
            return self._convert_order_response(order_info)
        except Exception as e:
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all open orders.
        
        Args:
            symbol: Optional trading pair symbol to filter by
            
        Returns:
            List of dictionaries containing order information
        """
        params = {"timestamp": int(time.time() * 1000)}
        
        if symbol:
            params["symbol"] = self.normalize_symbol(symbol)
        
        try:
            orders = await self._private_request(self.OPEN_ORDERS_ENDPOINT, params=params)
            return [self._convert_order_response(order) for order in orders]
        except Exception as e:
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
        binance_symbol = self.normalize_symbol(symbol)
        
        params = {
            "symbol": binance_symbol,
            "limit": min(1000, limit),  # Binance maximum is 1000
            "timestamp": int(time.time() * 1000)
        }
        
        # Add time range if specified
        if from_time:
            params["startTime"] = int(from_time.timestamp() * 1000)
        if to_time:
            params["endTime"] = int(to_time.timestamp() * 1000)
        
        try:
            trades = await self._private_request(self.ALL_ORDERS_ENDPOINT, params=params)
            
            # Convert to standardized format
            result = []
            for trade in trades:
                if trade.get("status") == "FILLED":
                    result.append({
                        "id": trade["orderId"],
                        "symbol": self.standardize_symbol(trade["symbol"]),
                        "side": trade["side"].lower(),
                        "price": Decimal(str(trade["price"])),
                        "amount": Decimal(str(trade["executedQty"])),
                        "cost": Decimal(str(trade["cummulativeQuoteQty"])),
                        "fee": None,  # Fee info not included in this endpoint
                        "timestamp": datetime.fromtimestamp(trade["time"] / 1000),
                    })
            
            return result
        except Exception as e:
            return []
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize a symbol to Binance format.
        
        Args:
            symbol: Symbol in standard format (e.g., "BTC/USDT")
            
        Returns:
            Symbol in Binance format (e.g., "BTCUSDT")
        """
        return symbol.replace("/", "")
    
    def standardize_symbol(self, symbol: str) -> str:
        """Convert a symbol from Binance format to standard format.
        
        Args:
            symbol: Symbol in Binance format (e.g., "BTCUSDT")
            
        Returns:
            Symbol in standard format (e.g., "BTC/USDT")
        """
        # For Binance, we need to identify the quote currency
        # This is a simplification and might not work for all pairs
        common_quote_currencies = ["USDT", "USD", "BTC", "ETH", "BNB", "BUSD"]
        
        for quote in sorted(common_quote_currencies, key=len, reverse=True):
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return f"{base}/{quote}"
        
        # Fallback if no known quote currency is found
        if len(symbol) >= 6:
            # Try a 3/3 split as a guess (e.g., BTCUSD)
            return f"{symbol[:-3]}/{symbol[-3:]}"
        elif len(symbol) >= 5:
            # Try a 2/3 split as a guess (e.g., BTETH)
            return f"{symbol[:-3]}/{symbol[-3:]}"
        
        # If all else fails, return as is
        return symbol
    
    async def _public_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, method: str = "GET") -> Dict:
        """Make a public API request.
        
        Args:
            endpoint: API endpoint path
            params: Optional request parameters
            method: HTTP method (GET, POST, DELETE)
            
        Returns:
            Response data as dict
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()
            
        url = self.BASE_URL + self.API_V3 + endpoint
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        
        # Apply rate limiting
        self._respect_rate_limits()
        
        async with self._session.request(method, url, params=params, headers=headers) as response:
            text = response.text()
            if not response.ok:
                raise ValueError(f"Error in public request: {text}")
            
            return json.loads(text) if text else {}
    
    async def _private_request(self, endpoint: str, params: Dict[str, Any], method: str = "GET") -> Dict:
        """Make a private (authenticated) API request.
        
        Args:
            endpoint: API endpoint path
            params: Request parameters
            method: HTTP method (GET, POST, DELETE)
            
        Returns:
            Response data as dict
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()
            
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret required for private requests")
        
        # Add timestamp if not already present
        if "timestamp" not in params:
            params["timestamp"] = int(time.time() * 1000)
        
        # Create signature
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        # Add signature to parameters
        params["signature"] = signature
        
        url = self.BASE_URL + self.API_V3 + endpoint
        headers = {"X-MBX-APIKEY": self.api_key}
        
        # Apply rate limiting
        self._respect_rate_limits()
        
        if method == "GET":
            async with self._session.get(url, params=params, headers=headers) as response:
                text = response.text()
                if not response.ok:
                    raise ValueError(f"Error in private request: {text}")
                
                return json.loads(text) if text else {}
        elif method == "POST":
            async with self._session.post(url, params=params, headers=headers) as response:
                text = response.text()
                if not response.ok:
                    raise ValueError(f"Error in private request: {text}")
                
                return json.loads(text) if text else {}
        elif method == "DELETE":
            async with self._session.delete(url, params=params, headers=headers) as response:
                text = response.text()
                if not response.ok:
                    raise ValueError(f"Error in private request: {text}")
                
                return json.loads(text) if text else {}
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    
    async def _respect_rate_limits(self):
        """Apply rate limiting to avoid exceeding Binance API limits."""
        now = time.time()
        
        # Remove timestamps older than 1 minute
        self._last_request_timestamps = [ts for ts in self._last_request_timestamps if now - ts <= 60]
        
        # Check if we're approaching the rate limit
        if len(self._last_request_timestamps) >= self.request_weight_per_minute - 5:
            # If close to limit, wait until oldest timestamp is more than 1 minute old
            if self._last_request_timestamps:
                oldest = self._last_request_timestamps[0]
                wait_time = max(0, 60 - (now - oldest))
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
        
        # Add current timestamp to the list
        self._last_request_timestamps.append(time.time())
    
    def _convert_order_response(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Binance order response to standardized format.
        
        Args:
            order_data: Order data from Binance API
            
        Returns:
            Standardized order information
        """
        # Map Binance status to our OrderStatus enum
        status_map = {
            "NEW": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "PENDING_CANCEL": OrderStatus.PENDING,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED
        }
        
        # Map Binance order type to our OrderType enum
        type_map = {
            "LIMIT": OrderType.LIMIT,
            "MARKET": OrderType.MARKET,
            "STOP_LOSS": OrderType.STOP,
            "STOP_LOSS_LIMIT": OrderType.STOP_LIMIT,
            "TAKE_PROFIT": OrderType.STOP,
            "TAKE_PROFIT_LIMIT": OrderType.STOP_LIMIT
        }
        
        # Map Binance side to our OrderSide enum
        side_map = {
            "BUY": OrderSide.BUY,
            "SELL": OrderSide.SELL
        }
        
        # Build standardized order info
        return {
            "id": str(order_data["orderId"]),
            "client_order_id": order_data.get("clientOrderId"),
            "exchange_order_id": str(order_data["orderId"]),
            "exchange": "binance",
            "symbol": self.standardize_symbol(order_data["symbol"]),
            "order_type": type_map.get(order_data["type"], OrderType.LIMIT).value,
            "side": side_map.get(order_data["side"], OrderSide.BUY).value,
            "quantity": float(order_data["origQty"]),
            "price": float(order_data["price"]) if "price" in order_data and order_data["price"] != "0" else None,
            "stop_price": float(order_data["stopPrice"]) if "stopPrice" in order_data and order_data["stopPrice"] != "0" else None,
            "status": status_map.get(order_data["status"], OrderStatus.CREATED).value,
            "created_at": datetime.fromtimestamp(order_data["time"] / 1000) if "time" in order_data else None,
            "updated_at": datetime.fromtimestamp(order_data["updateTime"] / 1000) if "updateTime" in order_data else None,
            "filled_quantity": float(order_data["executedQty"]) if "executedQty" in order_data else 0.0,
            "average_fill_price": float(order_data["avgPrice"]) if "avgPrice" in order_data and order_data["avgPrice"] != "0" else None,
        } 