"""
MEXC Spot V3 API Client

This module provides a comprehensive client for interacting with the MEXC Spot V3 API.
It includes support for both REST API endpoints and WebSocket streams.

MEXC API Documentation: https://mexcdevelop.github.io/apidocs/spot_v3_en/
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import aiohttp
import requests

from ..common import get_logger
from ..config.mexc_config import MEXC_CONFIG

logger = get_logger(__name__)

class MexcSpotV3Client:
    """
    Client for the MEXC Spot V3 API.
    
    This class provides methods for interacting with both REST and WebSocket endpoints
    of the MEXC Spot V3 API, following their official documentation:
    https://mexcdevelop.github.io/apidocs/spot_v3_en/
    
    Attributes:
        api_key (str): MEXC API key
        api_secret (str): MEXC API secret
        base_url (str): Base URL for REST API requests
        ws_url (str): URL for WebSocket connections
        recv_window (int): Receive window for requests (milliseconds)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        ws_url: Optional[str] = None,
        recv_window: int = 5000,
    ):
        """
        Initialize the MEXC Spot V3 API client.
        
        Args:
            api_key (str, optional): MEXC API key. Defaults to None.
            api_secret (str, optional): MEXC API secret. Defaults to None.
            base_url (str, optional): Base URL for REST API. Defaults to MEXC API URL.
            ws_url (str, optional): URL for WebSocket. Defaults to MEXC WebSocket URL.
            recv_window (int, optional): Receive window in milliseconds. Defaults to 5000.
        """
        # Use provided credentials or get from config
        self.api_key = api_key or MEXC_CONFIG.get("API_KEY", "")
        self.api_secret = api_secret or MEXC_CONFIG.get("API_SECRET", "")
        
        # API URLs
        self.base_url = base_url or "https://api.mexc.com"
        self.ws_url = ws_url or "wss://stream.mexc.com/ws"
        
        # Request settings
        self.recv_window = recv_window
        
        # Session for API requests
        self._session = None
        self._ws_connections = {}
        self._listen_key = None
        self._listen_key_keep_alive_task = None
        
        # API rate limiting
        self._request_weight = 0
        self._last_request_timestamp = 0
        
        logger.info("Initialized MEXC Spot V3 API client")
    
    async def _init_session(self) -> None:
        """Initialize an aiohttp session for async requests."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
    
    async def close_session(self) -> None:
        """Close the aiohttp session."""
        if self._session is not None:
            await self._session.close()
            self._session = None
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC SHA256 signature for API request.
        
        Args:
            params (Dict[str, Any]): Request parameters
            
        Returns:
            str: HMAC SHA256 signature
        """
        # Convert params to query string
        query_string = urllib.parse.urlencode(params)
        
        # Create signature using HMAC SHA256
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
    
    def _handle_response_error(self, response_data: Dict[str, Any]) -> None:
        """
        Handle API response errors.
        
        Args:
            response_data (Dict[str, Any]): Response data from API
            
        Raises:
            ValueError: If API returns an error
        """
        if "code" in response_data and response_data["code"] != 200:
            error_msg = f"API Error: {response_data.get('code')} - {response_data.get('msg', 'Unknown error')}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _prepare_request_params(self, params: Dict[str, Any], signed: bool = False) -> Dict[str, Any]:
        """
        Prepare parameters for API request.
        
        Args:
            params (Dict[str, Any]): Request parameters
            signed (bool, optional): Whether the request needs a signature. Defaults to False.
            
        Returns:
            Dict[str, Any]: Prepared parameters
        """
        # Create a copy of the params to avoid modifying the original
        request_params = params.copy() if params else {}
        
        if signed:
            # Add timestamp and receive window
            request_params["timestamp"] = self._get_timestamp()
            request_params["recvWindow"] = self.recv_window
            
            # Add signature
            request_params["signature"] = self._generate_signature(request_params)
        
        return request_params

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        api_version: str = "v3",
        api_type: str = "api",
    ) -> Dict[str, Any]:
        """
        Make an async request to the MEXC API.
        
        Args:
            method (str): HTTP method (GET, POST, DELETE, etc.)
            endpoint (str): API endpoint
            params (Dict[str, Any], optional): Request parameters. Defaults to None.
            signed (bool, optional): Whether the request needs a signature. Defaults to False.
            api_version (str, optional): API version. Defaults to "v3".
            api_type (str, optional): API type (api, sapi). Defaults to "api".
            
        Returns:
            Dict[str, Any]: Response data
            
        Raises:
            ValueError: If API returns an error
        """
        await self._init_session()
        
        # Prepare URL
        url = f"{self.base_url}/{api_type}/{api_version}/{endpoint}"
        
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-MEXC-APIKEY"] = self.api_key
        
        # Prepare request parameters
        request_params = self._prepare_request_params(params, signed)
        
        # Make request
        try:
            if method == "GET":
                async with self._session.get(url, params=request_params, headers=headers) as response:
                    response_data = await response.json()
            elif method == "POST":
                async with self._session.post(url, json=request_params, headers=headers) as response:
                    response_data = await response.json()
            elif method == "DELETE":
                async with self._session.delete(url, params=request_params, headers=headers) as response:
                    response_data = await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Handle response
            if isinstance(response_data, dict):
                self._handle_response_error(response_data)
                
            return response_data
            
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise

    # Synchronous version of the request method for convenience
    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        api_version: str = "v3",
        api_type: str = "api",
    ) -> Dict[str, Any]:
        """
        Make a synchronous request to the MEXC API.
        
        Args:
            method (str): HTTP method (GET, POST, DELETE, etc.)
            endpoint (str): API endpoint
            params (Dict[str, Any], optional): Request parameters. Defaults to None.
            signed (bool, optional): Whether the request needs a signature. Defaults to False.
            api_version (str, optional): API version. Defaults to "v3".
            api_type (str, optional): API type (api, sapi). Defaults to "api".
            
        Returns:
            Dict[str, Any]: Response data
            
        Raises:
            ValueError: If API returns an error
        """
        # Prepare URL
        url = f"{self.base_url}/{api_type}/{api_version}/{endpoint}"
        
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-MEXC-APIKEY"] = self.api_key
        
        # Prepare request parameters
        request_params = self._prepare_request_params(params, signed)
        
        # Make request
        try:
            if method == "GET":
                response = requests.get(url, params=request_params, headers=headers)
            elif method == "POST":
                response = requests.post(url, json=request_params, headers=headers)
            elif method == "DELETE":
                response = requests.delete(url, params=request_params, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Parse response
            response_data = response.json()
            
            # Handle response
            if isinstance(response_data, dict):
                self._handle_response_error(response_data)
                
            return response_data
            
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise
    
    # ---------- REST API ENDPOINTS ----------

    # System Status
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status.
        
        Returns:
            Dict[str, Any]: System status
        """
        return await self._request("GET", "system/status")
    
    # Exchange Information
    async def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Args:
            symbol (str, optional): Symbol to get info for. Defaults to None.
            
        Returns:
            Dict[str, Any]: Exchange information
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
            
        return await self._request("GET", "exchangeInfo", params)
    
    # Market Data - Ticker
    async def get_ticker(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get ticker for a symbol or all symbols.
        
        Args:
            symbol (str, optional): Symbol to get ticker for. Defaults to None.
            
        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Ticker data
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
            
        return await self._request("GET", "ticker/24hr", params)
    
    # Market Data - Order Book
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol (str): Symbol to get order book for
            limit (int, optional): Limit of results. Defaults to 100.
            
        Returns:
            Dict[str, Any]: Order book data
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
            
        return await self._request("GET", "depth", params)
    
    # Market Data - Recent Trades
    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol (str): Symbol to get trades for
            limit (int, optional): Limit of results. Defaults to 500.
            
        Returns:
            List[Dict[str, Any]]: Recent trades
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
            
        return await self._request("GET", "trades", params)
    
    # Market Data - Klines (Candlestick)
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> List[List[Any]]:
        """
        Get klines (candlestick) data for a symbol.
        
        Args:
            symbol (str): Symbol to get klines for
            interval (str): Interval of klines (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_time (int, optional): Start time in milliseconds. Defaults to None.
            end_time (int, optional): End time in milliseconds. Defaults to None.
            limit (int, optional): Limit of results. Defaults to 500.
            
        Returns:
            List[List[Any]]: Klines data
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
        
        if end_time:
            params["endTime"] = end_time
            
        return await self._request("GET", "klines", params)
    
    # Account - Create Order
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        new_client_order_id: Optional[str] = None,
        stop_price: Optional[float] = None,
        iceberg_qty: Optional[float] = None,
        new_order_resp_type: str = "RESULT",
    ) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            symbol (str): Symbol to create order for
            side (str): Order side (BUY, SELL)
            order_type (str): Order type (LIMIT, MARKET, LIMIT_MAKER, STOP_LOSS, STOP_LOSS_LIMIT)
            quantity (float, optional): Order quantity. Required for MARKET orders when quoteOrderQty is not specified.
            price (float, optional): Order price. Required for LIMIT orders.
            time_in_force (str, optional): Time in force (GTC, IOC, FOK). Defaults to "GTC".
            new_client_order_id (str, optional): Client order ID. Defaults to None.
            stop_price (float, optional): Stop price. Required for STOP_LOSS orders. Defaults to None.
            iceberg_qty (float, optional): Iceberg quantity. Defaults to None.
            new_order_resp_type (str, optional): Response type (ACK, RESULT, FULL). Defaults to "RESULT".
            
        Returns:
            Dict[str, Any]: Order data
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
        }
        
        if quantity:
            params["quantity"] = quantity
            
        if price:
            params["price"] = price
            
        params["timeInForce"] = time_in_force
        
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id
            
        if stop_price:
            params["stopPrice"] = stop_price
            
        if iceberg_qty:
            params["icebergQty"] = iceberg_qty
            
        params["newOrderRespType"] = new_order_resp_type
        
        return await self._request("POST", "order", params, signed=True)
    
    # Account - Cancel Order
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            symbol (str): Symbol of the order
            order_id (int, optional): Order ID. Defaults to None.
            orig_client_order_id (str, optional): Original client order ID. Defaults to None.
            
        Returns:
            Dict[str, Any]: Cancelled order data
        """
        params = {
            "symbol": symbol
        }
        
        if order_id:
            params["orderId"] = order_id
            
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
            
        return await self._request("DELETE", "order", params, signed=True)
    
    # Account - Get Order
    async def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get order details.
        
        Args:
            symbol (str): Symbol of the order
            order_id (int, optional): Order ID. Defaults to None.
            orig_client_order_id (str, optional): Original client order ID. Defaults to None.
            
        Returns:
            Dict[str, Any]: Order data
        """
        params = {
            "symbol": symbol
        }
        
        if order_id:
            params["orderId"] = order_id
            
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
            
        return await self._request("GET", "order", params, signed=True)
    
    # Account - Get Open Orders
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            symbol (str, optional): Symbol to get open orders for. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: Open orders
        """
        params = {}
        
        if symbol:
            params["symbol"] = symbol
            
        return await self._request("GET", "openOrders", params, signed=True)
    
    # Account - Get All Orders
    async def get_all_orders(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Get all orders.
        
        Args:
            symbol (str): Symbol to get orders for
            order_id (int, optional): Order ID. Defaults to None.
            start_time (int, optional): Start time in milliseconds. Defaults to None.
            end_time (int, optional): End time in milliseconds. Defaults to None.
            limit (int, optional): Limit of results. Defaults to 500.
            
        Returns:
            List[Dict[str, Any]]: Orders
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        if order_id:
            params["orderId"] = order_id
            
        if start_time:
            params["startTime"] = start_time
            
        if end_time:
            params["endTime"] = end_time
            
        return await self._request("GET", "allOrders", params, signed=True)
    
    # Account - Get Account Information
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict[str, Any]: Account information
        """
        return await self._request("GET", "account", signed=True)
    
    # Account - Get Account Trades
    async def get_account_trades(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        from_id: Optional[int] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Get account trades.
        
        Args:
            symbol (str): Symbol to get trades for
            start_time (int, optional): Start time in milliseconds. Defaults to None.
            end_time (int, optional): End time in milliseconds. Defaults to None.
            from_id (int, optional): Trade ID to fetch from. Defaults to None.
            limit (int, optional): Limit of results. Defaults to 500.
            
        Returns:
            List[Dict[str, Any]]: Trades
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
            
        if end_time:
            params["endTime"] = end_time
            
        if from_id:
            params["fromId"] = from_id
            
        return await self._request("GET", "myTrades", params, signed=True)
    
    # User Data Stream - Create Listen Key
    async def create_listen_key(self) -> str:
        """
        Create a listen key for user data stream.
        
        Returns:
            str: Listen key
        """
        response = await self._request("POST", "userDataStream")
        return response.get("listenKey", "")
    
    # User Data Stream - Keep-Alive Listen Key
    async def keep_alive_listen_key(self, listen_key: str) -> Dict[str, Any]:
        """
        Keep alive a listen key.
        
        Args:
            listen_key (str): Listen key to keep alive
            
        Returns:
            Dict[str, Any]: Response data
        """
        params = {"listenKey": listen_key}
        return await self._request("PUT", "userDataStream", params)
    
    # User Data Stream - Close Listen Key
    async def close_listen_key(self, listen_key: str) -> Dict[str, Any]:
        """
        Close a listen key.
        
        Args:
            listen_key (str): Listen key to close
            
        Returns:
            Dict[str, Any]: Response data
        """
        params = {"listenKey": listen_key}
        return await self._request("DELETE", "userDataStream", params)
    
    # ---------- WEBSOCKET STREAMS ----------
    
    async def _connect_websocket(self, stream_name: str, callback) -> None:
        """
        Connect to a WebSocket stream.
        
        Args:
            stream_name (str): Stream name
            callback: Callback function for messages
        """
        while True:
            try:
                url = f"{self.ws_url}"
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        logger.info(f"Connected to WebSocket stream: {stream_name}")
                        
                        # Subscribe to the stream
                        await ws.send_str(json.dumps({
                            "method": "SUBSCRIPTION",
                            "params": [stream_name],
                            "id": int(time.time() * 1000)
                        }))
                        
                        # Save the connection
                        self._ws_connections[stream_name] = ws
                        
                        # Process messages
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    await callback(data)
                                except Exception as e:
                                    logger.error(f"Error processing WebSocket message: {e}")
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket connection closed with error: {ws.exception()}")
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.warning("WebSocket connection closed")
                                break
                        
                        # Remove the connection
                        if stream_name in self._ws_connections:
                            del self._ws_connections[stream_name]
                            
                        logger.warning(f"WebSocket connection lost for {stream_name}, reconnecting...")
            
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {e}")
                
            # Wait before reconnecting
            await asyncio.sleep(5)
    
    async def subscribe_ticker(self, symbol: str, callback) -> asyncio.Task:
        """
        Subscribe to ticker updates.
        
        Args:
            symbol (str): Symbol to subscribe to
            callback: Callback function for messages
            
        Returns:
            asyncio.Task: WebSocket task
        """
        stream_name = f"spot@public.ticker.v3.api@{symbol}"
        task = asyncio.create_task(self._connect_websocket(stream_name, callback))
        return task
    
    async def subscribe_kline(self, symbol: str, interval: str, callback) -> asyncio.Task:
        """
        Subscribe to kline (candlestick) updates.
        
        Args:
            symbol (str): Symbol to subscribe to
            interval (str): Kline interval
            callback: Callback function for messages
            
        Returns:
            asyncio.Task: WebSocket task
        """
        stream_name = f"spot@public.kline.v3.api@{symbol}@{interval}"
        task = asyncio.create_task(self._connect_websocket(stream_name, callback))
        return task
    
    async def subscribe_depth(self, symbol: str, callback) -> asyncio.Task:
        """
        Subscribe to depth (order book) updates.
        
        Args:
            symbol (str): Symbol to subscribe to
            callback: Callback function for messages
            
        Returns:
            asyncio.Task: WebSocket task
        """
        stream_name = f"spot@public.increase.depth.v3.api@{symbol}"
        task = asyncio.create_task(self._connect_websocket(stream_name, callback))
        return task
    
    async def subscribe_trades(self, symbol: str, callback) -> asyncio.Task:
        """
        Subscribe to trade updates.
        
        Args:
            symbol (str): Symbol to subscribe to
            callback: Callback function for messages
            
        Returns:
            asyncio.Task: WebSocket task
        """
        stream_name = f"spot@public.deals.v3.api@{symbol}"
        task = asyncio.create_task(self._connect_websocket(stream_name, callback))
        return task
    
    # User Data Stream
    async def start_user_data_stream(self, callback) -> None:
        """
        Start user data stream.
        
        Args:
            callback: Callback function for messages
        """
        # Create listen key
        self._listen_key = await self.create_listen_key()
        
        if not self._listen_key:
            logger.error("Failed to create listen key for user data stream")
            return
        
        # Start keep-alive task
        if self._listen_key_keep_alive_task is None:
            self._listen_key_keep_alive_task = asyncio.create_task(self._keep_listen_key_alive())
        
        # Connect to user data stream
        stream_name = f"spot@private.universal.v3.api"
        task = asyncio.create_task(self._connect_websocket(stream_name, callback))
        return task
    
    async def _keep_listen_key_alive(self) -> None:
        """Keep the listen key alive by pinging it every 30 minutes."""
        while self._listen_key:
            try:
                # Keep alive the listen key every 30 minutes
                await asyncio.sleep(30 * 60)
                if self._listen_key:
                    await self.keep_alive_listen_key(self._listen_key)
                    logger.debug("Successfully kept listen key alive")
            except Exception as e:
                logger.error(f"Error keeping listen key alive: {e}")
    
    async def stop_user_data_stream(self) -> None:
        """Stop user data stream."""
        if self._listen_key:
            try:
                await self.close_listen_key(self._listen_key)
                self._listen_key = None
                logger.info("Closed listen key for user data stream")
            except Exception as e:
                logger.error(f"Error closing listen key: {e}")
        
        if self._listen_key_keep_alive_task:
            self._listen_key_keep_alive_task.cancel()
            self._listen_key_keep_alive_task = None
            
        for stream_name, ws in self._ws_connections.items():
            if "private" in stream_name:
                try:
                    await ws.close()
                    logger.info(f"Closed WebSocket connection for {stream_name}")
                except Exception as e:
                    logger.error(f"Error closing WebSocket connection: {e}")
    
    # ---------- CLEANUP ----------
    
    async def close(self) -> None:
        """Close all connections and resources."""
        # Stop user data stream
        await self.stop_user_data_stream()
        
        # Close all WebSocket connections
        for stream_name, ws in self._ws_connections.items():
            try:
                await ws.close()
                logger.info(f"Closed WebSocket connection for {stream_name}")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
        
        # Clear connections
        self._ws_connections = {}
        
        # Close HTTP session
        await self.close_session()
        
        logger.info("Closed MEXC Spot V3 API client")
