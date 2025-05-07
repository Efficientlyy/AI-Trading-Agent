"""
Binance Exchange Connector

This module provides a connector for the Binance cryptocurrency exchange,
allowing the trading system to interact with Binance for order execution.
"""

import logging
import asyncio
import json
import os
import time
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import aiohttp
from dotenv import load_dotenv

from .exchange_connector import ExchangeConnector
from .models import Order, OrderSide, OrderType, Position, Portfolio

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class BinanceConnector(ExchangeConnector):
    """
    Connector for the Binance cryptocurrency exchange.
    
    This class implements the ExchangeConnector interface for Binance,
    providing methods for placing orders, getting market data, and managing positions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Binance connector.
        
        Args:
            config: Configuration dictionary for the connector
        """
        super().__init__(config)
        self.name = "Binance"
        
        # API configuration
        self.api_key = self.config.get("api_key") or os.getenv("BINANCE_API_KEY")
        self.api_secret = self.config.get("api_secret") or os.getenv("BINANCE_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            logger.warning("No Binance API credentials provided. API calls requiring authentication will fail.")
        
        # API URLs
        self.use_testnet = self.config.get("use_testnet", True)
        
        if self.use_testnet:
            self.base_url = "https://testnet.binance.vision/api"
            logger.info("Using Binance testnet")
        else:
            self.base_url = "https://api.binance.com/api"
            logger.info("Using Binance mainnet")
        
        # Exchange info
        self.exchange_info = {}
        self.symbols_info = {}
        self.assets_info = {}
        
        # Rate limiting
        self.rate_limits = {}
        self.last_request_time = {}
        
        logger.info("Initialized Binance connector")
    
    async def load_exchange_info(self) -> None:
        """Load exchange information."""
        try:
            # Get exchange info
            url = f"{self.base_url}/v3/exchangeInfo"
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error getting exchange info: {response.status}")
                    return
                
                data = await response.json()
                self.exchange_info = data
                
                # Extract symbols info
                for symbol_info in data.get("symbols", []):
                    symbol = symbol_info.get("symbol")
                    if symbol:
                        self.symbols_info[symbol] = symbol_info
                        self.supported_symbols.append(symbol)
                        
                        # Extract precision info
                        for filter_info in symbol_info.get("filters", []):
                            if filter_info.get("filterType") == "PRICE_FILTER":
                                tick_size = filter_info.get("tickSize", "0.00000001")
                                # Count decimal places in tick size
                                decimal_places = len(tick_size.split(".")[1].rstrip("0"))
                                self.price_precision[symbol] = decimal_places
                            
                            if filter_info.get("filterType") == "LOT_SIZE":
                                step_size = filter_info.get("stepSize", "0.00000001")
                                # Count decimal places in step size
                                decimal_places = len(step_size.split(".")[1].rstrip("0"))
                                self.quantity_precision[symbol] = decimal_places
                                
                                min_qty = filter_info.get("minQty", "0")
                                self.min_order_sizes[symbol] = Decimal(min_qty)
                
                # Extract rate limits
                for rate_limit in data.get("rateLimits", []):
                    limit_type = rate_limit.get("rateLimitType")
                    if limit_type:
                        self.rate_limits[limit_type] = rate_limit
                
                logger.info(f"Loaded exchange info: {len(self.supported_symbols)} symbols")
        except Exception as e:
            logger.error(f"Error loading exchange info: {e}")
    
    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                           data: Optional[Dict[str, Any]] = None, auth: bool = False) -> Dict[str, Any]:
        """
        Make a request to the Binance API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Query parameters
            data: Request data
            auth: Whether authentication is required
            
        Returns:
            Response data
            
        Raises:
            Exception: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        # Add authentication if required
        if auth:
            if not self.api_key or not self.api_secret:
                raise ValueError("API credentials required but not provided")
            
            # Add timestamp
            timestamp = int(time.time() * 1000)
            if params is None:
                params = {}
            params["timestamp"] = timestamp
            
            # Create signature
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            
            params["signature"] = signature
            headers["X-MBX-APIKEY"] = self.api_key
        
        # Make request
        try:
            if method == "GET":
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error making request: {response.status} - {error_text}")
                        return {"error": error_text}
                    
                    return await response.json()
            elif method == "POST":
                async with self.session.post(url, params=params, json=data, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error making request: {response.status} - {error_text}")
                        return {"error": error_text}
                    
                    return await response.json()
            elif method == "DELETE":
                async with self.session.delete(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error making request: {response.status} - {error_text}")
                        return {"error": error_text}
                    
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except Exception as e:
            logger.error(f"Error making request: {e}")
            return {"error": str(e)}
    
    async def get_market_price(self, symbol: str) -> Decimal:
        """
        Get the current market price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current market price
            
        Raises:
            ValueError: If the symbol is not supported
        """
        if symbol not in self.supported_symbols:
            raise ValueError(f"Symbol {symbol} is not supported")
        
        # Get ticker price
        response = await self._make_request("GET", "/v3/ticker/price", {"symbol": symbol})
        
        if "error" in response:
            raise ValueError(f"Error getting market price: {response['error']}")
        
        price = response.get("price")
        if price is None:
            raise ValueError("Price not found in response")
        
        return Decimal(price)
    
    async def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get the order book for a symbol.
        
        Args:
            symbol: Symbol to get order book for
            limit: Number of levels to get
            
        Returns:
            Order book data
            
        Raises:
            ValueError: If the symbol is not supported
        """
        if symbol not in self.supported_symbols:
            raise ValueError(f"Symbol {symbol} is not supported")
        
        # Get order book
        response = await self._make_request("GET", "/v3/depth", {"symbol": symbol, "limit": limit})
        
        if "error" in response:
            raise ValueError(f"Error getting order book: {response['error']}")
        
        return response
    
    async def get_balance(self) -> Decimal:
        """
        Get the account balance.
        
        Returns:
            Account balance
        """
        # Get account info
        response = await self._make_request("GET", "/v3/account", auth=True)
        
        if "error" in response:
            raise ValueError(f"Error getting account info: {response['error']}")
        
        # Calculate total balance in USDT
        total_balance = Decimal("0")
        
        for balance in response.get("balances", []):
            asset = balance.get("asset")
            free = Decimal(balance.get("free", "0"))
            locked = Decimal(balance.get("locked", "0"))
            
            total = free + locked
            
            if total > 0:
                # Convert to USDT
                if asset == "USDT":
                    total_balance += total
                else:
                    try:
                        # Try to get price for asset/USDT
                        symbol = f"{asset}USDT"
                        price = await self.get_market_price(symbol)
                        total_balance += total * price
                    except Exception:
                        logger.warning(f"Could not convert {asset} to USDT")
        
        return total_balance
    
    async def get_asset_balance(self, asset: str) -> Decimal:
        """
        Get the balance for a specific asset.
        
        Args:
            asset: Asset to get balance for
            
        Returns:
            Asset balance
            
        Raises:
            ValueError: If the asset is not supported
        """
        # Get account info
        response = await self._make_request("GET", "/v3/account", auth=True)
        
        if "error" in response:
            raise ValueError(f"Error getting account info: {response['error']}")
        
        # Find asset balance
        for balance in response.get("balances", []):
            if balance.get("asset") == asset:
                free = Decimal(balance.get("free", "0"))
                locked = Decimal(balance.get("locked", "0"))
                return free + locked
        
        return Decimal("0")
    
    async def get_positions(self) -> Dict[str, Position]:
        """
        Get all open positions.
        
        Returns:
            Dictionary of positions, keyed by symbol
        """
        # Get account info
        response = await self._make_request("GET", "/v3/account", auth=True)
        
        if "error" in response:
            raise ValueError(f"Error getting account info: {response['error']}")
        
        # Find positions
        positions = {}
        
        for balance in response.get("balances", []):
            asset = balance.get("asset")
            free = Decimal(balance.get("free", "0"))
            locked = Decimal(balance.get("locked", "0"))
            
            total = free + locked
            
            if total > 0 and asset != "USDT":
                # Try to get price for asset/USDT
                try:
                    symbol = f"{asset}USDT"
                    price = await self.get_market_price(symbol)
                    
                    # Create position
                    positions[asset] = Position(
                        symbol=asset,
                        quantity=total,
                        entry_price=price,  # We don't know the actual entry price
                        current_price=price,
                        unrealized_pnl=Decimal("0"),  # We don't know the actual PnL
                        realized_pnl=Decimal("0")
                    )
                except Exception:
                    logger.warning(f"Could not get price for {asset}/USDT")
        
        return positions
    
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """
        Place an order on the exchange.
        
        Args:
            order: Order to place
            
        Returns:
            Dictionary with order result
            
        Raises:
            ValueError: If the order is invalid
        """
        # Validate order
        self._validate_order(order)
        
        # Prepare order parameters
        params = {
            "symbol": order.symbol,
            "side": order.side.value.upper(),
            "type": order.order_type.value.upper(),
            "quantity": str(self._round_quantity(order.symbol, order.quantity))
        }
        
        # Add price for limit orders
        if order.order_type == OrderType.LIMIT:
            params["price"] = str(self._round_price(order.symbol, Decimal(str(order.price))))
            params["timeInForce"] = "GTC"  # Good Till Cancelled
        
        # Place order
        response = await self._make_request("POST", "/v3/order", params=params, auth=True)
        
        if "error" in response:
            return {"success": False, "error": response["error"]}
        
        return {
            "success": True,
            "order_id": response.get("orderId"),
            "client_order_id": response.get("clientOrderId"),
            "symbol": response.get("symbol"),
            "price": response.get("price"),
            "quantity": response.get("origQty"),
            "side": response.get("side"),
            "type": response.get("type"),
            "status": response.get("status"),
            "timestamp": response.get("transactTime")
        }
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an order on the exchange.
        
        Args:
            order_id: ID of the order to cancel
            symbol: Symbol of the order
            
        Returns:
            Dictionary with cancel result
            
        Raises:
            ValueError: If the order ID is invalid
        """
        # Validate symbol
        if symbol not in self.supported_symbols:
            raise ValueError(f"Symbol {symbol} is not supported")
        
        # Cancel order
        params = {
            "symbol": symbol,
            "orderId": order_id
        }
        
        response = await self._make_request("DELETE", "/v3/order", params=params, auth=True)
        
        if "error" in response:
            return {"success": False, "error": response["error"]}
        
        return {
            "success": True,
            "order_id": response.get("orderId"),
            "client_order_id": response.get("clientOrderId"),
            "symbol": response.get("symbol"),
            "status": "cancelled"
        }
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order to get status for
            symbol: Symbol of the order
            
        Returns:
            Dictionary with order status
            
        Raises:
            ValueError: If the order ID is invalid
        """
        # Validate symbol
        if symbol not in self.supported_symbols:
            raise ValueError(f"Symbol {symbol} is not supported")
        
        # Get order status
        params = {
            "symbol": symbol,
            "orderId": order_id
        }
        
        response = await self._make_request("GET", "/v3/order", params=params, auth=True)
        
        if "error" in response:
            return {"success": False, "error": response["error"]}
        
        return {
            "success": True,
            "order": {
                "order_id": response.get("orderId"),
                "client_order_id": response.get("clientOrderId"),
                "symbol": response.get("symbol"),
                "price": response.get("price"),
                "quantity": response.get("origQty"),
                "executed_quantity": response.get("executedQty"),
                "side": response.get("side"),
                "type": response.get("type"),
                "status": response.get("status"),
                "time": response.get("time"),
                "update_time": response.get("updateTime")
            }
        }
    
    async def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get order history.
        
        Args:
            symbol: Symbol to get history for (None for all)
            limit: Maximum number of orders to get
            
        Returns:
            List of orders
        """
        # Prepare parameters
        params = {
            "limit": limit
        }
        
        if symbol:
            # Validate symbol
            if symbol not in self.supported_symbols:
                raise ValueError(f"Symbol {symbol} is not supported")
            
            params["symbol"] = symbol
        
        # Get order history
        response = await self._make_request("GET", "/v3/allOrders", params=params, auth=True)
        
        if isinstance(response, dict) and "error" in response:
            raise ValueError(f"Error getting order history: {response['error']}")
        
        # Format orders
        orders = []
        for order in response:
            orders.append({
                "order_id": order.get("orderId"),
                "client_order_id": order.get("clientOrderId"),
                "symbol": order.get("symbol"),
                "price": order.get("price"),
                "quantity": order.get("origQty"),
                "executed_quantity": order.get("executedQty"),
                "side": order.get("side"),
                "type": order.get("type"),
                "status": order.get("status"),
                "time": order.get("time"),
                "update_time": order.get("updateTime")
            })
        
        return orders
    
    async def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Args:
            symbol: Symbol to get history for (None for all)
            limit: Maximum number of trades to get
            
        Returns:
            List of trades
        """
        # Prepare parameters
        params = {
            "limit": limit
        }
        
        if symbol:
            # Validate symbol
            if symbol not in self.supported_symbols:
                raise ValueError(f"Symbol {symbol} is not supported")
            
            params["symbol"] = symbol
        
        # Get trade history
        response = await self._make_request("GET", "/v3/myTrades", params=params, auth=True)
        
        if isinstance(response, dict) and "error" in response:
            raise ValueError(f"Error getting trade history: {response['error']}")
        
        # Format trades
        trades = []
        for trade in response:
            trades.append({
                "id": trade.get("id"),
                "order_id": trade.get("orderId"),
                "symbol": trade.get("symbol"),
                "price": trade.get("price"),
                "quantity": trade.get("qty"),
                "commission": trade.get("commission"),
                "commission_asset": trade.get("commissionAsset"),
                "time": trade.get("time"),
                "is_buyer": trade.get("isBuyer"),
                "is_maker": trade.get("isMaker")
            })
        
        return trades
