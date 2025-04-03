"""Binance exchange connector for the AI Crypto Trading System.

This module implements the exchange connector for the Binance cryptocurrency exchange,
providing methods for fetching market data and subscribing to real-time updates.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union

import aiohttp
from aiohttp import ClientSession, ClientWebSocketResponse

from src.common.logging import get_logger
from src.data_collection.exchange_connector import ExchangeConnector
from src.models.market_data import CandleData, OrderBookData, TimeFrame, TradeData


class BinanceConnector(ExchangeConnector):
    """Exchange connector for Binance."""
    
    REST_API_URL = "https://api.binance.com"
    WS_API_URL = "wss://stream.binance.com:9443/ws"
    
    TIMEFRAME_MAP = {
        TimeFrame.MINUTE_1: "1m",
        TimeFrame.MINUTE_5: "5m",
        TimeFrame.MINUTE_15: "15m",
        TimeFrame.MINUTE_30: "30m",
        TimeFrame.HOUR_1: "1h",
        TimeFrame.HOUR_4: "4h",
        TimeFrame.DAY_1: "1d",
        TimeFrame.WEEK_1: "1w",
    }
    
    def __init__(self, exchange_id: str = "binance"):
        """Initialize the Binance connector."""
        super().__init__(exchange_id)
        self.logger = get_logger("data_collection", "binance")
        self.session: Optional[ClientSession] = None
        self.ws: Optional[ClientWebSocketResponse] = None
        self.ws_subscriptions: Set[str] = set()
        self.ws_task: Optional[asyncio.Task] = None
        self.candle_polling_tasks: Dict[str, asyncio.Task] = {}
    
    async def _initialize(self) -> None:
        """Initialize the Binance connector."""
        self.logger.info("Initializing Binance connector")
    
    async def _init_rest_client(self) -> None:
        """Initialize the REST API client."""
        self.session = aiohttp.ClientSession()
        self.logger.debug("REST client session initialized")
    
    async def _init_websocket_client(self) -> None:
        """Initialize the WebSocket client."""
        if self.ws:
            self._close_websocket_client()
        
        self.ws = await self.session.ws_connect(self.WS_API_URL)
        self.logger.debug("WebSocket connection established")
        
        # Start WebSocket message processing task
        self.ws_task = self.create_task(self._process_ws_messages())
        
        # Subscribe to all requested data
        self._update_subscriptions()
    
    async def _close_websocket_client(self) -> None:
        """Close the WebSocket client."""
        if self.ws_task and not self.ws_task.done():
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
            self.ws_task = None
        
        if self.ws:
            await self.ws.close()
            self.ws = None
            self.ws_subscriptions.clear()
            self.logger.debug("WebSocket connection closed")
    
    async def _start(self) -> None:
        """Start the Binance connector."""
        self.logger.info("Starting Binance connector")
    
    async def _stop(self) -> None:
        """Stop the Binance connector."""
        self.logger.info("Stopping Binance connector")
        
        # Cancel all polling tasks
        for task in self.candle_polling_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        if self.candle_polling_tasks:
            await asyncio.gather(*self.candle_polling_tasks.values(), return_exceptions=True)
            self.candle_polling_tasks.clear()
        
        # Close the REST session
        if self.session:
            await self.session.close()
            self.session = None
    
    async def fetch_available_symbols(self) -> Set[str]:
        """Fetch the list of available trading pairs from Binance.
        
        Returns:
            Set[str]: Set of available symbols in the format "BTC/USDT"
        """
        url = f"{self.REST_API_URL}/api/v3/exchangeInfo"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    self.logger.error("Failed to fetch exchange info", status=response.status)
                    return set()
                
                data = response.json()
                
                # Extract symbols and convert to standardized format
                symbols = set()
                for symbol_info in data.get("symbols", []):
                    status = symbol_info.get("status")
                    if status != "TRADING":
                        continue
                    
                    base = symbol_info.get("baseAsset")
                    quote = symbol_info.get("quoteAsset")
                    if base and quote:
                        symbols.add(f"{base}/{quote}")
                
                self.logger.info("Fetched available symbols", count=len(symbols))
                return symbols
                
        except Exception as e:
            self.logger.error("Error fetching available symbols", error=str(e))
            return set()
    
    async def fetch_candles(
        self, symbol: str, timeframe: TimeFrame, 
        since: Optional[datetime] = None, limit: int = 500
    ) -> List[CandleData]:
        """Fetch historical candle data for a symbol and timeframe.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
            since: Optional start time for the candles
            limit: Maximum number of candles to fetch
            
        Returns:
            List[CandleData]: List of candle data objects
        """
        # Convert symbol to Binance format
        binance_symbol = self._convert_to_binance_symbol(symbol)
        binance_interval = self.TIMEFRAME_MAP.get(timeframe)
        
        if not binance_symbol or not binance_interval:
            self.logger.error("Invalid symbol or timeframe", 
                            symbol=symbol, timeframe=timeframe.value)
            return []
        
        url = f"{self.REST_API_URL}/api/v3/klines"
        
        params = {
            "symbol": binance_symbol,
            "interval": binance_interval,
            "limit": limit
        }
        
        if since:
            params["startTime"] = int(since.timestamp() * 1000)
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error("Failed to fetch candles", 
                                    status=response.status, 
                                    symbol=symbol, 
                                    timeframe=timeframe.value)
                    return []
                
                data = response.json()
                
                # Convert Binance candles to our format
                candles = []
                for candle_data in data:
                    if len(candle_data) < 11:
                        continue
                    
                    try:
                        timestamp = datetime.fromtimestamp(candle_data[0] / 1000)
                        candle = CandleData(
                            symbol=symbol,
                            exchange=self.exchange_id,
                            timestamp=timestamp,
                            timeframe=timeframe,
                            open=float(candle_data[1]),
                            high=float(candle_data[2]),
                            low=float(candle_data[3]),
                            close=float(candle_data[4]),
                            volume=float(candle_data[5])
                        )
                        candles.append(candle)
                    except (ValueError, TypeError) as e:
                        self.logger.error("Error parsing candle data", 
                                        error=str(e), 
                                        data=candle_data)
                
                self.logger.debug("Fetched candles", 
                                symbol=symbol, 
                                timeframe=timeframe.value, 
                                count=len(candles))
                return candles
                
        except Exception as e:
            self.logger.error("Error fetching candles", 
                            error=str(e), 
                            symbol=symbol, 
                            timeframe=timeframe.value)
            return []
    
    async def fetch_orderbook(self, symbol: str, limit: int = 100) -> Optional[OrderBookData]:
        """Fetch the current order book for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            limit: Maximum number of orders to fetch on each side
            
        Returns:
            Optional[OrderBookData]: Order book data object or None if failed
        """
        # Convert symbol to Binance format
        binance_symbol = self._convert_to_binance_symbol(symbol)
        
        if not binance_symbol:
            self.logger.error("Invalid symbol", symbol=symbol)
            return None
        
        url = f"{self.REST_API_URL}/api/v3/depth"
        
        params = {
            "symbol": binance_symbol,
            "limit": limit
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error("Failed to fetch order book", 
                                    status=response.status, 
                                    symbol=symbol)
                    return None
                
                data = response.json()
                
                # Convert to our format
                timestamp = datetime.utcnow()
                
                bids = [
                    {"price": float(bid[0]), "size": float(bid[1])}
                    for bid in data.get("bids", [])
                ]
                
                asks = [
                    {"price": float(ask[0]), "size": float(ask[1])}
                    for ask in data.get("asks", [])
                ]
                
                orderbook = OrderBookData(
                    symbol=symbol,
                    exchange=self.exchange_id,
                    timestamp=timestamp,
                    bids=bids,
                    asks=asks
                )
                
                self.logger.debug("Fetched order book", 
                                symbol=symbol, 
                                bids_count=len(bids), 
                                asks_count=len(asks))
                return orderbook
                
        except Exception as e:
            self.logger.error("Error fetching order book", 
                            error=str(e), 
                            symbol=symbol)
            return None
    
    async def fetch_trades(
        self, symbol: str, since: Optional[datetime] = None, limit: int = 500
    ) -> List[TradeData]:
        """Fetch recent trades for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            since: Optional start time for the trades
            limit: Maximum number of trades to fetch
            
        Returns:
            List[TradeData]: List of trade data objects
        """
        # Convert symbol to Binance format
        binance_symbol = self._convert_to_binance_symbol(symbol)
        
        if not binance_symbol:
            self.logger.error("Invalid symbol", symbol=symbol)
            return []
        
        url = f"{self.REST_API_URL}/api/v3/trades"
        
        params = {
            "symbol": binance_symbol,
            "limit": limit
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error("Failed to fetch trades", 
                                    status=response.status, 
                                    symbol=symbol)
                    return []
                
                data = response.json()
                
                # Convert to our format
                trades = []
                for trade_data in data:
                    try:
                        timestamp = datetime.fromtimestamp(trade_data["time"] / 1000)
                        side = "buy" if trade_data["isBuyerMaker"] else "sell"
                        
                        trade = TradeData(
                            symbol=symbol,
                            exchange=self.exchange_id,
                            timestamp=timestamp,
                            price=float(trade_data["price"]),
                            size=float(trade_data["qty"]),
                            side=side
                        )
                        trades.append(trade)
                    except (KeyError, ValueError, TypeError) as e:
                        self.logger.error("Error parsing trade data", 
                                        error=str(e), 
                                        data=trade_data)
                
                self.logger.debug("Fetched trades", 
                                symbol=symbol, 
                                count=len(trades))
                return trades
                
        except Exception as e:
            self.logger.error("Error fetching trades", 
                            error=str(e), 
                            symbol=symbol)
            return []
    
    async def _start_polling_tasks(self) -> None:
        """Start polling tasks for REST-based data."""
        # Start candle polling tasks for subscribed symbols and timeframes
        for symbol in self.subscribed_symbols:
            if symbol in self.subscribed_timeframes:
                for timeframe in self.subscribed_timeframes[symbol]:
                    await self._start_candle_polling(symbol, timeframe)
    
    async def _start_candle_polling(self, symbol: str, timeframe: TimeFrame) -> None:
        """Start a polling task for candle data.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
        """
        task_key = f"{symbol}_{timeframe.value}"
        
        # Skip if already polling
        if task_key in self.candle_polling_tasks:
            return
        
        # Create and start the polling task
        task = self.create_task(self._poll_candles(symbol, timeframe))
        self.candle_polling_tasks[task_key] = task
        self.logger.debug("Started candle polling task", 
                        symbol=symbol, 
                        timeframe=timeframe.value)
    
    async def _stop_candle_polling(self, symbol: str, timeframe: TimeFrame) -> None:
        """Stop a polling task for candle data.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
        """
        task_key = f"{symbol}_{timeframe.value}"
        
        # Skip if not polling
        if task_key not in self.candle_polling_tasks:
            return
        
        # Cancel and remove the polling task
        task = self.candle_polling_tasks[task_key]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        del self.candle_polling_tasks[task_key]
        self.logger.debug("Stopped candle polling task", 
                        symbol=symbol, 
                        timeframe=timeframe.value)
    
    async def _poll_candles(self, symbol: str, timeframe: TimeFrame) -> None:
        """Poll for candle data and publish updates.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The candle timeframe
        """
        poll_interval = self._get_poll_interval(timeframe)
        
        try:
            while True:
                # Fetch the latest candles
                candles = await self.fetch_candles(symbol, timeframe, limit=5)
                
                if candles:
                    # Publish the latest candle
                    latest_candle = candles[-1]
                    await self.publish_candle_data(latest_candle)
                
                # Wait for the next poll
                await asyncio.sleep(poll_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("Candle polling task cancelled", 
                            symbol=symbol, 
                            timeframe=timeframe.value)
            raise
        except Exception as e:
            self.logger.error("Error in candle polling task", 
                            symbol=symbol, 
                            timeframe=timeframe.value, 
                            error=str(e))
    
    def _get_poll_interval(self, timeframe: TimeFrame) -> float:
        """Get the polling interval for a timeframe.
        
        Args:
            timeframe: The candle timeframe
            
        Returns:
            float: The polling interval in seconds
        """
        # Set a reasonable polling interval based on the timeframe
        if timeframe == TimeFrame.MINUTE_1:
            return 30.0  # Poll every 30 seconds
        elif timeframe == TimeFrame.MINUTE_5:
            return 60.0  # Poll every minute
        elif timeframe == TimeFrame.MINUTE_15:
            return 5 * 60.0  # Poll every 5 minutes
        elif timeframe == TimeFrame.MINUTE_30:
            return 10 * 60.0  # Poll every 10 minutes
        elif timeframe == TimeFrame.HOUR_1:
            return 15 * 60.0  # Poll every 15 minutes
        elif timeframe == TimeFrame.HOUR_4:
            return 30 * 60.0  # Poll every 30 minutes
        elif timeframe == TimeFrame.DAY_1:
            return 60 * 60.0  # Poll every hour
        elif timeframe == TimeFrame.WEEK_1:
            return 6 * 60 * 60.0  # Poll every 6 hours
        else:
            return 60.0  # Default to polling every minute
    
    async def _update_subscriptions(self) -> None:
        """Update WebSocket subscriptions based on current subscription state."""
        if not self.ws:
            self.logger.warning("Cannot update subscriptions: WebSocket not connected")
            return
        
        # Calculate new subscriptions based on current state
        new_subscriptions = set()
        
        # Add orderbook subscriptions
        for symbol in self.subscribed_orderbooks:
            binance_symbol = self._convert_to_binance_symbol(symbol)
            if binance_symbol:
                new_subscriptions.add(f"{binance_symbol.lower()}@depth")
        
        # Add trade subscriptions
        for symbol in self.subscribed_trades:
            binance_symbol = self._convert_to_binance_symbol(symbol)
            if binance_symbol:
                new_subscriptions.add(f"{binance_symbol.lower()}@trade")
        
        # Update WebSocket subscriptions
        to_subscribe = new_subscriptions - self.ws_subscriptions
        to_unsubscribe = self.ws_subscriptions - new_subscriptions
        
        # Process unsubscriptions
        if to_unsubscribe:
            streams = list(to_unsubscribe)
            msg = {
                "method": "UNSUBSCRIBE",
                "params": streams,
                "id": int(time.time() * 1000)
            }
            await self.ws.send_json(msg)
            self.logger.debug("Sent unsubscribe request", streams=streams)
            
            # Remove from tracked subscriptions
            self.ws_subscriptions -= to_unsubscribe
        
        # Process new subscriptions
        if to_subscribe:
            streams = list(to_subscribe)
            msg = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": int(time.time() * 1000)
            }
            await self.ws.send_json(msg)
            self.logger.debug("Sent subscribe request", streams=streams)
            
            # Add to tracked subscriptions
            self.ws_subscriptions |= to_subscribe
    
    async def _process_ws_messages(self) -> None:
        """Process WebSocket messages."""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_message(data)
                    except json.JSONDecodeError:
                        self.logger.error("Failed to parse WebSocket message", data=msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error("WebSocket connection closed with exception", 
                                    exception=self.ws.exception())
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    self.logger.info("WebSocket connection closed")
                    break
        except asyncio.CancelledError:
            self.logger.debug("WebSocket message processing task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error processing WebSocket messages", error=str(e))
    
    async def _handle_ws_message(self, data: Dict) -> None:
        """Handle a WebSocket message.
        
        Args:
            data: The message data
        """
        # Handle subscription responses
        if "id" in data:
            self.logger.debug("Received subscription response", data=data)
            return
        
        # Handle data updates
        if "e" in data:  # Regular stream
            event_type = data.get("e")
            if event_type == "trade":
                await self._handle_trade_event(data)
            # Add handling for other event types as needed
        elif "stream" in data:  # Combined stream
            stream = data.get("stream")
            stream_data = data.get("data", {})
            
            if "@depth" in stream:
                await self._handle_orderbook_update(stream, stream_data)
            elif "@trade" in stream:
                await self._handle_trade_event(stream_data)
            # Add handling for other stream types as needed
    
    async def _handle_trade_event(self, data: Dict) -> None:
        """Handle a trade event.
        
        Args:
            data: The trade event data
        """
        try:
            # Extract symbol from data
            binance_symbol = data.get("s")
            if not binance_symbol:
                return
            
            symbol = self._convert_from_binance_symbol(binance_symbol)
            if not symbol:
                return
            
            # Extract trade data
            timestamp = datetime.fromtimestamp(data.get("T", 0) / 1000)
            price = float(data.get("p", 0))
            quantity = float(data.get("q", 0))
            is_buyer_maker = data.get("m", False)
            side = "sell" if is_buyer_maker else "buy"
            
            # Create and publish trade data
            trade = TradeData(
                symbol=symbol,
                exchange=self.exchange_id,
                timestamp=timestamp,
                price=price,
                size=quantity,
                side=side
            )
            
            await self.publish_trade_data(trade)
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error("Error handling trade event", error=str(e), data=data)
    
    async def _handle_orderbook_update(self, stream: str, data: Dict) -> None:
        """Handle an order book update event.
        
        Args:
            stream: The stream name
            data: The order book update data
        """
        try:
            # Extract symbol from stream name
            binance_symbol = stream.split("@")[0].upper()
            symbol = self._convert_from_binance_symbol(binance_symbol)
            if not symbol:
                return
            
            # Skip partial order book updates for now
            # In a production system, you would maintain a local order book copy
            # and apply these updates to it
            
            # Instead, trigger a fetch of the full order book
            orderbook = await self.fetch_orderbook(symbol)
            if orderbook:
                await self.publish_orderbook_data(orderbook)
            
        except Exception as e:
            self.logger.error("Error handling order book update", 
                            error=str(e), stream=stream, data=data)
    
    def _convert_to_binance_symbol(self, symbol: str) -> Optional[str]:
        """Convert a standard symbol to Binance format.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Optional[str]: The Binance symbol (e.g., "BTCUSDT") or None if invalid
        """
        try:
            base, quote = symbol.split("/")
            return f"{base}{quote}"
        except ValueError:
            self.logger.error("Invalid symbol format", symbol=symbol)
            return None
    
    def _convert_from_binance_symbol(self, binance_symbol: str) -> Optional[str]:
        """Convert a Binance symbol to standard format.
        
        This is a simplified implementation and assumes common quote assets.
        A more robust implementation would need to check against available symbols.
        
        Args:
            binance_symbol: The Binance symbol (e.g., "BTCUSDT")
            
        Returns:
            Optional[str]: The standard symbol (e.g., "BTC/USDT") or None if invalid
        """
        # Common quote assets, ordered by length to avoid incorrect splits
        quote_assets = ["USDT", "BTC", "ETH", "BNB", "BUSD", "USDC", "DAI", "USD"]
        
        for quote in quote_assets:
            if binance_symbol.endswith(quote):
                base = binance_symbol[:-len(quote)]
                return f"{base}/{quote}"
        
        # If no common quote asset found, make a best guess
        # This could be improved by using the exchange info from Binance
        for i in range(len(binance_symbol) - 1, 0, -1):
            base = binance_symbol[:i]
            quote = binance_symbol[i:]
            
            # Check if both parts look like valid symbols (uppercase letters)
            if base.isalpha() and quote.isalpha():
                return f"{base}/{quote}"
        
        self.logger.error("Could not convert Binance symbol to standard format", 
                        binance_symbol=binance_symbol)
        return None 