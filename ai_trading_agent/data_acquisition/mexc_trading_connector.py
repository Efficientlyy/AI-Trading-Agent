"""
MEXC Trading Connector

This module provides a high-level interface for trading with MEXC exchange,
integrating the MEXC Spot V3 API client with the AI Trading Agent system.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from .mexc_spot_v3_client import MexcSpotV3Client
from ..common import get_logger
from ..config.mexc_config import MEXC_CONFIG, TRADING_PAIRS
from ..portfolio.portfolio_manager import PortfolioManager
from ..trading.order import Order, OrderStatus, OrderType, OrderSide
from ..trading.execution_handler import ExecutionHandler

logger = get_logger(__name__)

class MexcTradingConnector:
    """
    High-level connector for trading with MEXC exchange.
    
    This class provides methods for executing trades, managing orders,
    and retrieving market data from MEXC, integrating with the AI Trading Agent's
    portfolio management and execution systems.
    
    Attributes:
        client (MexcSpotV3Client): MEXC Spot V3 API client
        portfolio_manager (PortfolioManager): Portfolio manager instance
        execution_handler (ExecutionHandler): Execution handler instance
        symbols (List[str]): List of trading symbols
        orderbook_cache (Dict[str, Dict]): Cache of order book data
        ticker_cache (Dict[str, Dict]): Cache of ticker data
        kline_cache (Dict[str, List]): Cache of kline data
        ws_tasks (Dict[str, asyncio.Task]): WebSocket tasks
        callbacks (Dict[str, List[Callable]]): Callbacks for WebSocket events
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        portfolio_manager: Optional[PortfolioManager] = None,
        execution_handler: Optional[ExecutionHandler] = None,
        symbols: Optional[List[str]] = None,
    ):
        """
        Initialize the MEXC Trading Connector.
        
        Args:
            api_key (str, optional): MEXC API key. Defaults to None.
            api_secret (str, optional): MEXC API secret. Defaults to None.
            portfolio_manager (PortfolioManager, optional): Portfolio manager. Defaults to None.
            execution_handler (ExecutionHandler, optional): Execution handler. Defaults to None.
            symbols (List[str], optional): Trading symbols. Defaults to None.
        """
        # Initialize MEXC Spot V3 API client
        self.client = MexcSpotV3Client(api_key, api_secret)
        
        # Set portfolio manager and execution handler
        self.portfolio_manager = portfolio_manager
        self.execution_handler = execution_handler
        
        # Set trading symbols
        self.symbols = symbols or TRADING_PAIRS
        
        # Initialize caches
        self.orderbook_cache = {}
        self.ticker_cache = {}
        self.kline_cache = {}
        
        # Initialize WebSocket tasks
        self.ws_tasks = {}
        
        # Initialize callbacks
        self.callbacks = {
            "ticker": [],
            "kline": [],
            "depth": [],
            "trade": [],
            "user_data": []
        }
        
        logger.info("Initialized MEXC Trading Connector")
    
    # ---------- MARKET DATA METHODS ----------
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Dict[str, Any]: Ticker data
        """
        # Check cache first
        if symbol in self.ticker_cache:
            return self.ticker_cache[symbol]
        
        # Fetch from API
        try:
            ticker = await self.client.get_ticker(symbol)
            
            # Cache the result
            if isinstance(ticker, dict):
                self.ticker_cache[symbol] = ticker
                return ticker
            elif isinstance(ticker, list) and len(ticker) > 0:
                for t in ticker:
                    if t.get("symbol") == symbol:
                        self.ticker_cache[symbol] = t
                        return t
            
            logger.warning(f"Ticker data not found for {symbol}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            limit (int, optional): Limit of order book entries. Defaults to 100.
            
        Returns:
            Dict[str, Any]: Order book data
        """
        try:
            orderbook = await self.client.get_orderbook(symbol, limit)
            
            # Cache the result
            self.orderbook_cache[symbol] = orderbook
            
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            
            # Return cached data if available
            if symbol in self.orderbook_cache:
                return self.orderbook_cache[symbol]
            
            return {}
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List[Any]]:
        """
        Get klines (candlestick) data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            interval (str): Kline interval
            limit (int, optional): Limit of klines. Defaults to 500.
            start_time (int, optional): Start time in milliseconds. Defaults to None.
            end_time (int, optional): End time in milliseconds. Defaults to None.
            
        Returns:
            List[List[Any]]: Klines data
        """
        cache_key = f"{symbol}_{interval}"
        
        try:
            klines = await self.client.get_klines(symbol, interval, start_time, end_time, limit)
            
            # Cache the result
            self.kline_cache[cache_key] = klines
            
            return klines
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol} ({interval}): {e}")
            
            # Return cached data if available
            if cache_key in self.kline_cache:
                return self.kline_cache[cache_key]
            
            return []
    
    # ---------- TRADING METHODS ----------
    
    async def create_order(self, order: Order) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            order (Order): Order to create
            
        Returns:
            Dict[str, Any]: Order data
        """
        try:
            # Check if we have a portfolio manager
            if self.portfolio_manager:
                # Check if the order meets risk management criteria
                if not self.portfolio_manager.validate_order(order):
                    logger.warning(f"Order {order.id} rejected by portfolio manager")
                    order.status = OrderStatus.REJECTED
                    return {"error": "Order rejected by portfolio manager"}
            
            # Prepare order parameters
            params = {
                "symbol": order.symbol,
                "side": order.side.value.upper(),
                "type": order.order_type.value.upper(),
                "quantity": order.quantity,
            }
            
            # Add price for limit orders
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                params["price"] = order.price
            
            # Add stop price for stop orders
            if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
                params["stopPrice"] = order.stop_price
            
            # Add client order ID if available
            if order.client_order_id:
                params["newClientOrderId"] = order.client_order_id
            
            # Create the order
            response = await self.client.create_order(**params)
            
            # Update order with response data
            order.exchange_order_id = response.get("orderId")
            order.status = OrderStatus.OPEN if response.get("status") == "NEW" else order.status
            order.filled_quantity = float(response.get("executedQty", 0))
            order.average_price = float(response.get("price", 0))
            
            # Update portfolio if we have a portfolio manager
            if self.portfolio_manager and order.status == OrderStatus.FILLED:
                self.portfolio_manager.update_on_order_fill(order)
            
            return response
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            order.status = OrderStatus.REJECTED
            return {"error": str(e)}
    
    async def cancel_order(self, order: Order) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order (Order): Order to cancel
            
        Returns:
            Dict[str, Any]: Cancelled order data
        """
        try:
            # Prepare parameters
            params = {
                "symbol": order.symbol,
            }
            
            # Use exchange order ID if available
            if order.exchange_order_id:
                params["orderId"] = order.exchange_order_id
            # Otherwise use client order ID
            elif order.client_order_id:
                params["origClientOrderId"] = order.client_order_id
            else:
                logger.error("Cannot cancel order without exchange order ID or client order ID")
                return {"error": "Missing order ID"}
            
            # Cancel the order
            response = await self.client.cancel_order(**params)
            
            # Update order status
            order.status = OrderStatus.CANCELED
            
            return response
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {"error": str(e)}
    
    async def get_order(self, order: Order) -> Dict[str, Any]:
        """
        Get order details.
        
        Args:
            order (Order): Order to get details for
            
        Returns:
            Dict[str, Any]: Order data
        """
        try:
            # Prepare parameters
            params = {
                "symbol": order.symbol,
            }
            
            # Use exchange order ID if available
            if order.exchange_order_id:
                params["orderId"] = order.exchange_order_id
            # Otherwise use client order ID
            elif order.client_order_id:
                params["origClientOrderId"] = order.client_order_id
            else:
                logger.error("Cannot get order without exchange order ID or client order ID")
                return {"error": "Missing order ID"}
            
            # Get the order
            response = await self.client.get_order(**params)
            
            # Update order with response data
            order.status = OrderStatus(response.get("status", "").lower())
            order.filled_quantity = float(response.get("executedQty", 0))
            order.average_price = float(response.get("price", 0))
            
            # Update portfolio if we have a portfolio manager and the order is filled
            if self.portfolio_manager and order.status == OrderStatus.FILLED:
                self.portfolio_manager.update_on_order_fill(order)
            
            return response
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return {"error": str(e)}
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            symbol (str, optional): Symbol to get open orders for. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: Open orders
        """
        try:
            return await self.client.get_open_orders(symbol)
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict[str, Any]: Account information
        """
        try:
            return await self.client.get_account_info()
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    # ---------- WEBSOCKET METHODS ----------
    
    async def _ticker_callback(self, data: Dict[str, Any]) -> None:
        """
        Process ticker WebSocket data.
        
        Args:
            data (Dict[str, Any]): Ticker data
        """
        try:
            if "data" in data and "symbol" in data.get("data", {}):
                symbol = data["data"]["symbol"]
                ticker_data = data["data"]
                
                # Update cache
                self.ticker_cache[symbol] = ticker_data
                
                # Call callbacks
                for callback in self.callbacks["ticker"]:
                    try:
                        await callback(symbol, ticker_data)
                    except Exception as e:
                        logger.error(f"Error in ticker callback: {e}")
        except Exception as e:
            logger.error(f"Error processing ticker data: {e}")
    
    async def _kline_callback(self, data: Dict[str, Any]) -> None:
        """
        Process kline WebSocket data.
        
        Args:
            data (Dict[str, Any]): Kline data
        """
        try:
            if "data" in data and "s" in data.get("data", {}):
                symbol = data["data"]["s"]
                interval = data["data"]["i"]
                kline_data = data["data"]
                
                # Create cache key
                cache_key = f"{symbol}_{interval}"
                
                # Update cache
                if cache_key not in self.kline_cache:
                    self.kline_cache[cache_key] = []
                
                # Add new kline to cache
                self.kline_cache[cache_key].append([
                    kline_data["t"],  # Open time
                    float(kline_data["o"]),  # Open
                    float(kline_data["h"]),  # High
                    float(kline_data["l"]),  # Low
                    float(kline_data["c"]),  # Close
                    float(kline_data["v"]),  # Volume
                    kline_data["T"],  # Close time
                    float(kline_data["q"]),  # Quote asset volume
                    kline_data["n"],  # Number of trades
                    float(kline_data["V"]),  # Taker buy base asset volume
                    float(kline_data["Q"]),  # Taker buy quote asset volume
                ])
                
                # Limit cache size
                self.kline_cache[cache_key] = self.kline_cache[cache_key][-500:]
                
                # Call callbacks
                for callback in self.callbacks["kline"]:
                    try:
                        await callback(symbol, interval, kline_data)
                    except Exception as e:
                        logger.error(f"Error in kline callback: {e}")
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
    
    async def _depth_callback(self, data: Dict[str, Any]) -> None:
        """
        Process depth WebSocket data.
        
        Args:
            data (Dict[str, Any]): Depth data
        """
        try:
            if "data" in data and "s" in data.get("data", {}):
                symbol = data["data"]["s"]
                depth_data = data["data"]
                
                # Update cache
                if symbol not in self.orderbook_cache:
                    self.orderbook_cache[symbol] = {
                        "lastUpdateId": 0,
                        "bids": [],
                        "asks": []
                    }
                
                # Update the order book
                orderbook = self.orderbook_cache[symbol]
                
                # Check if this is a newer update
                if depth_data["u"] <= orderbook["lastUpdateId"]:
                    return
                
                # Update last update ID
                orderbook["lastUpdateId"] = depth_data["u"]
                
                # Update bids
                for bid in depth_data.get("b", []):
                    price, quantity = float(bid[0]), float(bid[1])
                    
                    # Remove price level if quantity is 0
                    if quantity == 0:
                        orderbook["bids"] = [b for b in orderbook["bids"] if b[0] != price]
                    else:
                        # Update existing price level or add new one
                        updated = False
                        for i, b in enumerate(orderbook["bids"]):
                            if b[0] == price:
                                orderbook["bids"][i] = [price, quantity]
                                updated = True
                                break
                        
                        if not updated:
                            orderbook["bids"].append([price, quantity])
                
                # Update asks
                for ask in depth_data.get("a", []):
                    price, quantity = float(ask[0]), float(ask[1])
                    
                    # Remove price level if quantity is 0
                    if quantity == 0:
                        orderbook["asks"] = [a for a in orderbook["asks"] if a[0] != price]
                    else:
                        # Update existing price level or add new one
                        updated = False
                        for i, a in enumerate(orderbook["asks"]):
                            if a[0] == price:
                                orderbook["asks"][i] = [price, quantity]
                                updated = True
                                break
                        
                        if not updated:
                            orderbook["asks"].append([price, quantity])
                
                # Sort bids (descending) and asks (ascending)
                orderbook["bids"].sort(key=lambda x: x[0], reverse=True)
                orderbook["asks"].sort(key=lambda x: x[0])
                
                # Call callbacks
                for callback in self.callbacks["depth"]:
                    try:
                        await callback(symbol, orderbook)
                    except Exception as e:
                        logger.error(f"Error in depth callback: {e}")
        except Exception as e:
            logger.error(f"Error processing depth data: {e}")
    
    async def _trade_callback(self, data: Dict[str, Any]) -> None:
        """
        Process trade WebSocket data.
        
        Args:
            data (Dict[str, Any]): Trade data
        """
        try:
            if "data" in data and "s" in data.get("data", {}):
                symbol = data["data"]["s"]
                trade_data = data["data"]
                
                # Call callbacks
                for callback in self.callbacks["trade"]:
                    try:
                        await callback(symbol, trade_data)
                    except Exception as e:
                        logger.error(f"Error in trade callback: {e}")
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    async def _user_data_callback(self, data: Dict[str, Any]) -> None:
        """
        Process user data WebSocket data.
        
        Args:
            data (Dict[str, Any]): User data
        """
        try:
            # Handle account update
            if "data" in data and data.get("e") == "outboundAccountPosition":
                # Update portfolio if we have a portfolio manager
                if self.portfolio_manager:
                    balances = data["data"]["B"]
                    self.portfolio_manager.update_balances(balances)
            
            # Handle order update
            elif "data" in data and data.get("e") == "executionReport":
                order_data = data["data"]
                
                # Create order object
                order = Order(
                    symbol=order_data["s"],
                    order_type=OrderType(order_data["o"].lower()),
                    side=OrderSide(order_data["S"].lower()),
                    quantity=float(order_data["q"]),
                    price=float(order_data["p"]) if order_data["o"] != "MARKET" else None,
                    client_order_id=order_data["c"],
                    exchange_order_id=order_data["i"],
                    status=OrderStatus(order_data["X"].lower()),
                    filled_quantity=float(order_data["z"]),
                    average_price=float(order_data["L"]) if order_data["L"] != "0" else None,
                )
                
                # Update portfolio if we have a portfolio manager and the order is filled
                if self.portfolio_manager and order.status == OrderStatus.FILLED:
                    self.portfolio_manager.update_on_order_fill(order)
                
                # Call callbacks
                for callback in self.callbacks["user_data"]:
                    try:
                        await callback(order)
                    except Exception as e:
                        logger.error(f"Error in user data callback: {e}")
        except Exception as e:
            logger.error(f"Error processing user data: {e}")
    
    def register_ticker_callback(self, callback: Callable) -> None:
        """
        Register a callback for ticker updates.
        
        Args:
            callback (Callable): Callback function
        """
        self.callbacks["ticker"].append(callback)
    
    def register_kline_callback(self, callback: Callable) -> None:
        """
        Register a callback for kline updates.
        
        Args:
            callback (Callable): Callback function
        """
        self.callbacks["kline"].append(callback)
    
    def register_depth_callback(self, callback: Callable) -> None:
        """
        Register a callback for depth updates.
        
        Args:
            callback (Callable): Callback function
        """
        self.callbacks["depth"].append(callback)
    
    def register_trade_callback(self, callback: Callable) -> None:
        """
        Register a callback for trade updates.
        
        Args:
            callback (Callable): Callback function
        """
        self.callbacks["trade"].append(callback)
    
    def register_user_data_callback(self, callback: Callable) -> None:
        """
        Register a callback for user data updates.
        
        Args:
            callback (Callable): Callback function
        """
        self.callbacks["user_data"].append(callback)
    
    async def subscribe_to_tickers(self, symbols: Optional[List[str]] = None) -> None:
        """
        Subscribe to ticker updates.
        
        Args:
            symbols (List[str], optional): Symbols to subscribe to. Defaults to None.
        """
        symbols = symbols or self.symbols
        
        for symbol in symbols:
            try:
                task = await self.client.subscribe_ticker(symbol, self._ticker_callback)
                self.ws_tasks[f"ticker_{symbol}"] = task
                logger.info(f"Subscribed to ticker updates for {symbol}")
            except Exception as e:
                logger.error(f"Error subscribing to ticker for {symbol}: {e}")
    
    async def subscribe_to_klines(
        self,
        symbols: Optional[List[str]] = None,
        intervals: Optional[List[str]] = None,
    ) -> None:
        """
        Subscribe to kline updates.
        
        Args:
            symbols (List[str], optional): Symbols to subscribe to. Defaults to None.
            intervals (List[str], optional): Intervals to subscribe to. Defaults to ["1m"].
        """
        symbols = symbols or self.symbols
        intervals = intervals or ["1m"]
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    task = await self.client.subscribe_kline(symbol, interval, self._kline_callback)
                    self.ws_tasks[f"kline_{symbol}_{interval}"] = task
                    logger.info(f"Subscribed to kline updates for {symbol} ({interval})")
                except Exception as e:
                    logger.error(f"Error subscribing to klines for {symbol} ({interval}): {e}")
    
    async def subscribe_to_depth(self, symbols: Optional[List[str]] = None) -> None:
        """
        Subscribe to depth updates.
        
        Args:
            symbols (List[str], optional): Symbols to subscribe to. Defaults to None.
        """
        symbols = symbols or self.symbols
        
        for symbol in symbols:
            try:
                task = await self.client.subscribe_depth(symbol, self._depth_callback)
                self.ws_tasks[f"depth_{symbol}"] = task
                logger.info(f"Subscribed to depth updates for {symbol}")
            except Exception as e:
                logger.error(f"Error subscribing to depth for {symbol}: {e}")
    
    async def subscribe_to_trades(self, symbols: Optional[List[str]] = None) -> None:
        """
        Subscribe to trade updates.
        
        Args:
            symbols (List[str], optional): Symbols to subscribe to. Defaults to None.
        """
        symbols = symbols or self.symbols
        
        for symbol in symbols:
            try:
                task = await self.client.subscribe_trades(symbol, self._trade_callback)
                self.ws_tasks[f"trades_{symbol}"] = task
                logger.info(f"Subscribed to trade updates for {symbol}")
            except Exception as e:
                logger.error(f"Error subscribing to trades for {symbol}: {e}")
    
    async def subscribe_to_user_data(self) -> None:
        """Subscribe to user data updates."""
        try:
            task = await self.client.start_user_data_stream(self._user_data_callback)
            self.ws_tasks["user_data"] = task
            logger.info("Subscribed to user data updates")
        except Exception as e:
            logger.error(f"Error subscribing to user data: {e}")
    
    async def subscribe_all(
        self,
        include_user_data: bool = True,
        symbols: Optional[List[str]] = None,
        intervals: Optional[List[str]] = None,
    ) -> None:
        """
        Subscribe to all WebSocket streams.
        
        Args:
            include_user_data (bool, optional): Include user data stream. Defaults to True.
            symbols (List[str], optional): Symbols to subscribe to. Defaults to None.
            intervals (List[str], optional): Intervals to subscribe to. Defaults to ["1m"].
        """
        # Subscribe to market data
        await self.subscribe_to_tickers(symbols)
        await self.subscribe_to_klines(symbols, intervals)
        await self.subscribe_to_depth(symbols)
        await self.subscribe_to_trades(symbols)
        
        # Subscribe to user data if requested
        if include_user_data:
            await self.subscribe_to_user_data()
    
    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all WebSocket streams."""
        for name, task in self.ws_tasks.items():
            try:
                task.cancel()
                logger.info(f"Unsubscribed from {name}")
            except Exception as e:
                logger.error(f"Error unsubscribing from {name}: {e}")
        
        self.ws_tasks = {}
    
    # ---------- CLEANUP ----------
    
    async def close(self) -> None:
        """Close all connections and resources."""
        # Unsubscribe from all WebSocket streams
        await self.unsubscribe_all()
        
        # Close client
        await self.client.close()
        
        logger.info("Closed MEXC Trading Connector")
