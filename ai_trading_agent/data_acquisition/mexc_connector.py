"""
MEXC Exchange Connector

This module provides direct WebSocket connection to MEXC exchange
for real-time market data and trading capabilities.
"""

import asyncio
import json
import logging
import time
import hmac
import hashlib
import base64
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
import websockets
import aiohttp
from urllib.parse import urlencode

from ..common import get_logger
from ..config.mexc_config import MEXC_CONFIG, API_ENDPOINTS, TRADING_PAIRS

# Configure logger
logger = get_logger(__name__)

class MexcConnector:
    """
    MEXC Exchange connector providing direct WebSocket access for real-time
    market data and trading capabilities, specifically optimized for BTC/USDC.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize the MEXC connector with API credentials."""
        self.api_key = api_key
        self.api_secret = api_secret
        
        # WebSocket connection
        self.ws = None
        self.ws_connected = False
        self.ws_task = None
        self.ping_task = None
        
        # Data storage
        self.ticker_data = {}
        self.orderbook_data = {}
        self.kline_data = {}
        self.trade_data = {}
        
        # Callbacks
        self.ticker_callbacks = []
        self.orderbook_callbacks = []
        self.kline_callbacks = []
        self.trade_callbacks = []
        
        # Default trading pair
        self.default_pair = MEXC_CONFIG['default_pair']
        
        # Subscription tracking
        self.active_subscriptions = set()
        
        logger.info(f"Initialized MexcConnector with default pair: {self.default_pair}")
    
    async def connect(self) -> bool:
        """
        Connect to MEXC WebSocket API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.ws_connected:
            logger.info("Already connected to MEXC WebSocket")
            return True
        
        try:
            # Fix: Use correct WebSocket URL key from config
            ws_url = API_ENDPOINTS.get('public_ws', 'wss://stream.mexc.com/ws')
            logger.info(f"Connecting to MEXC WebSocket URL: {ws_url}")
            self.ws = await websockets.connect(ws_url)
            self.ws_connected = True
            
            # Start WebSocket handler task
            self.ws_task = asyncio.create_task(self._ws_handler())
            
            # Start ping task to keep connection alive
            self.ping_task = asyncio.create_task(self._ping_loop())
            
            logger.info("Connected to MEXC WebSocket")
            
            # Subscribe to default pair
            await self.subscribe_ticker(self.default_pair)
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to MEXC WebSocket: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MEXC WebSocket."""
        if not self.ws_connected:
            return
        
        try:
            # Cancel tasks
            if self.ws_task:
                self.ws_task.cancel()
            
            if self.ping_task:
                self.ping_task.cancel()
            
            # Close WebSocket
            if self.ws:
                await self.ws.close()
            
            self.ws_connected = False
            self.ws = None
            
            logger.info("Disconnected from MEXC WebSocket")
        except Exception as e:
            logger.error(f"Error disconnecting from MEXC WebSocket: {e}")
    
    async def _ws_handler(self):
        """Handle WebSocket messages."""
        try:
            while True:
                if not self.ws:
                    break
                
                message = await self.ws.recv()
                
                try:
                    data = json.loads(message)
                    
                    # Handle ping message
                    if 'ping' in data:
                        await self.ws.send(json.dumps({"pong": data['ping']}))
                        continue
                    
                    # Handle subscribed data
                    if 'channel' in data and 'data' in data:
                        channel = data['channel']
                        channel_data = data['data']
                        
                        if channel.startswith('spot@public.ticker.v3'):
                            self._handle_ticker_update(channel, channel_data)
                        elif channel.startswith('spot@public.bookTicker.v3'):
                            self._handle_orderbook_update(channel, channel_data)
                        elif channel.startswith('spot@public.kline.v3'):
                            self._handle_kline_update(channel, channel_data)
                        elif channel.startswith('spot@public.deals.v3'):
                            self._handle_trade_update(channel, channel_data)
                        else:
                            logger.debug(f"Received data for unknown channel: {channel}")
                    
                    # Handle subscription response
                    elif 'code' in data and 'channel' in data:
                        if data['code'] == 0:
                            logger.info(f"Successfully subscribed to {data['channel']}")
                            self.active_subscriptions.add(data['channel'])
                        else:
                            logger.error(f"Subscription error for {data['channel']}: {data.get('msg', 'Unknown error')}")
                    
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
        
        except asyncio.CancelledError:
            logger.info("WebSocket handler task cancelled")
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
            
            # Try to reconnect
            self.ws_connected = False
            asyncio.create_task(self._reconnect())
    
    async def _ping_loop(self):
        """Send periodic pings to keep connection alive."""
        try:
            while self.ws_connected:
                if self.ws and not self.ws.closed:
                    try:
                        # Send ping with current timestamp
                        ping_data = {"ping": int(time.time() * 1000)}
                        await self.ws.send(json.dumps(ping_data))
                    except Exception as e:
                        logger.error(f"Error sending ping: {e}")
                
                # Wait 30 seconds before next ping
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            logger.info("Ping task cancelled")
        except Exception as e:
            logger.error(f"Ping loop error: {e}")
    
    async def _reconnect(self, delay: int = 5):
        """Reconnect to WebSocket after connection failure."""
        if self.ws_connected:
            return
        
        logger.info(f"Attempting to reconnect in {delay} seconds...")
        await asyncio.sleep(delay)
        
        try:
            # Store current subscriptions to resubscribe after reconnect
            subscriptions = list(self.active_subscriptions)
            
            # Connect again
            success = await self.connect()
            
            if success and subscriptions:
                # Resubscribe to channels
                for channel in subscriptions:
                    if channel.startswith('spot@public.ticker.v3'):
                        # Extract symbol from channel name
                        symbol = channel.split('.')[-1]
                        await self.subscribe_ticker(symbol)
                    elif channel.startswith('spot@public.bookTicker.v3'):
                        symbol = channel.split('.')[-1]
                        await self.subscribe_orderbook(symbol)
                    elif channel.startswith('spot@public.kline.v3'):
                        # Extract symbol and interval
                        parts = channel.split('.')
                        symbol = parts[-1]
                        interval = parts[-2]
                        await self.subscribe_kline(symbol, interval)
                    elif channel.startswith('spot@public.deals.v3'):
                        symbol = channel.split('.')[-1]
                        await self.subscribe_trades(symbol)
        except Exception as e:
            logger.error(f"Reconnection error: {e}")
            # Try again with exponential backoff
            next_delay = min(delay * 2, 60)  # Cap at 60 seconds
            asyncio.create_task(self._reconnect(next_delay))
    
    def _handle_ticker_update(self, channel: str, data: Dict[str, Any]):
        """Handle ticker data updates."""
        symbol = channel.split('.')[-1]
        
        # Update ticker data
        self.ticker_data[symbol] = {
            'symbol': symbol,
            'price': float(data.get('c', 0)),  # Last price
            'open': float(data.get('o', 0)),
            'high': float(data.get('h', 0)),
            'low': float(data.get('l', 0)),
            'volume': float(data.get('v', 0)),
            'timestamp': int(time.time() * 1000),
        }
        
        # Notify callbacks
        for callback in self.ticker_callbacks:
            try:
                callback(symbol, self.ticker_data[symbol])
            except Exception as e:
                logger.error(f"Error in ticker callback: {e}")
    
    def _handle_orderbook_update(self, channel: str, data: Dict[str, Any]):
        """Handle orderbook data updates."""
        symbol = channel.split('.')[-1]
        
        # Update orderbook data
        self.orderbook_data[symbol] = {
            'symbol': symbol,
            'bid': float(data.get('b', 0)),
            'ask': float(data.get('a', 0)),
            'bid_qty': float(data.get('B', 0)),
            'ask_qty': float(data.get('A', 0)),
            'timestamp': int(time.time() * 1000),
        }
        
        # Notify callbacks
        for callback in self.orderbook_callbacks:
            try:
                callback(symbol, self.orderbook_data[symbol])
            except Exception as e:
                logger.error(f"Error in orderbook callback: {e}")
    
    def _handle_kline_update(self, channel: str, data: Dict[str, Any]):
        """Handle kline (candlestick) data updates."""
        parts = channel.split('.')
        symbol = parts[-1]
        interval = parts[-2]
        
        key = f"{symbol}_{interval}"
        
        # Update kline data
        candle = {
            'symbol': symbol,
            'interval': interval,
            'open': float(data.get('o', 0)),
            'high': float(data.get('h', 0)),
            'low': float(data.get('l', 0)),
            'close': float(data.get('c', 0)),
            'volume': float(data.get('v', 0)),
            'timestamp': int(data.get('t', time.time() * 1000)),
            'is_closed': data.get('e', False),  # Whether the candle is closed
        }
        
        # Store kline data
        if key not in self.kline_data:
            self.kline_data[key] = []
        
        # Update existing candle or add new one
        if self.kline_data[key] and self.kline_data[key][-1]['timestamp'] == candle['timestamp']:
            self.kline_data[key][-1] = candle
        else:
            self.kline_data[key].append(candle)
            # Keep only the last 1000 candles
            if len(self.kline_data[key]) > 1000:
                self.kline_data[key] = self.kline_data[key][-1000:]
        
        # Notify callbacks
        for callback in self.kline_callbacks:
            try:
                callback(symbol, interval, candle)
            except Exception as e:
                logger.error(f"Error in kline callback: {e}")
    
    def _handle_trade_update(self, channel: str, data: Dict[str, Any]):
        """Handle trade data updates."""
        symbol = channel.split('.')[-1]
        
        # Update trade data
        trade = {
            'symbol': symbol,
            'id': data.get('d', ''),
            'price': float(data.get('p', 0)),
            'amount': float(data.get('v', 0)),
            'side': 'buy' if data.get('S', '') == 'BUY' else 'sell',
            'timestamp': int(data.get('t', time.time() * 1000)),
        }
        
        # Store trade data
        if symbol not in self.trade_data:
            self.trade_data[symbol] = []
        
        self.trade_data[symbol].append(trade)
        # Keep only the last 1000 trades
        if len(self.trade_data[symbol]) > 1000:
            self.trade_data[symbol] = self.trade_data[symbol][-1000:]
        
        # Notify callbacks
        for callback in self.trade_callbacks:
            try:
                callback(symbol, trade)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")
    
    async def subscribe_ticker(self, symbol: str) -> bool:
        """
        Subscribe to ticker data for a symbol.
        
        Args:
            symbol: Trading pair in format like 'BTC/USDC'
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        if not self.ws_connected:
            logger.error("Cannot subscribe to ticker, not connected")
            return False
        
        # Convert symbol format (BTC/USDC -> BTCUSDC)
        formatted_symbol = symbol.replace('/', '')
        
        try:
            channel = f"spot@public.ticker.v3.api.{formatted_symbol}"
            
            # Check if already subscribed
            if channel in self.active_subscriptions:
                logger.info(f"Already subscribed to ticker for {symbol}")
                return True
            
            # Subscribe to ticker
            subscription = {
                "method": "SUBSCRIPTION",
                "params": [channel]
            }
            
            await self.ws.send(json.dumps(subscription))
            logger.info(f"Subscribed to ticker for {symbol}")
            
            return True
        except Exception as e:
            logger.error(f"Error subscribing to ticker for {symbol}: {e}")
            return False
    
    async def subscribe_orderbook(self, symbol: str) -> bool:
        """
        Subscribe to orderbook data for a symbol.
        
        Args:
            symbol: Trading pair in format like 'BTC/USDC'
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        if not self.ws_connected:
            logger.error("Cannot subscribe to orderbook, not connected")
            return False
        
        # Convert symbol format (BTC/USDC -> BTCUSDC)
        formatted_symbol = symbol.replace('/', '')
        
        try:
            channel = f"spot@public.bookTicker.v3.api.{formatted_symbol}"
            
            # Check if already subscribed
            if channel in self.active_subscriptions:
                logger.info(f"Already subscribed to orderbook for {symbol}")
                return True
            
            # Subscribe to orderbook
            subscription = {
                "method": "SUBSCRIPTION",
                "params": [channel]
            }
            
            await self.ws.send(json.dumps(subscription))
            logger.info(f"Subscribed to orderbook for {symbol}")
            
            return True
        except Exception as e:
            logger.error(f"Error subscribing to orderbook for {symbol}: {e}")
            return False
    
    async def subscribe_kline(self, symbol: str, interval: str = '1m') -> bool:
        """
        Subscribe to kline (candlestick) data for a symbol.
        
        Args:
            symbol: Trading pair in format like 'BTC/USDC'
            interval: Timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        if not self.ws_connected:
            logger.error("Cannot subscribe to kline, not connected")
            return False
        
        # Convert symbol format (BTC/USDC -> BTCUSDC)
        formatted_symbol = symbol.replace('/', '')
        
        try:
            channel = f"spot@public.kline.v3.api.{interval}.{formatted_symbol}"
            
            # Check if already subscribed
            if channel in self.active_subscriptions:
                logger.info(f"Already subscribed to kline for {symbol} ({interval})")
                return True
            
            # Subscribe to kline
            subscription = {
                "method": "SUBSCRIPTION",
                "params": [channel]
            }
            
            await self.ws.send(json.dumps(subscription))
            logger.info(f"Subscribed to kline for {symbol} ({interval})")
            
            return True
        except Exception as e:
            logger.error(f"Error subscribing to kline for {symbol} ({interval}): {e}")
            return False
    
    async def subscribe_trades(self, symbol: str) -> bool:
        """
        Subscribe to trade data for a symbol.
        
        Args:
            symbol: Trading pair in format like 'BTC/USDC'
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        if not self.ws_connected:
            logger.error("Cannot subscribe to trades, not connected")
            return False
        
        # Convert symbol format (BTC/USDC -> BTCUSDC)
        formatted_symbol = symbol.replace('/', '')
        
        try:
            channel = f"spot@public.deals.v3.api.{formatted_symbol}"
            
            # Check if already subscribed
            if channel in self.active_subscriptions:
                logger.info(f"Already subscribed to trades for {symbol}")
                return True
            
            # Subscribe to trades
            subscription = {
                "method": "SUBSCRIPTION",
                "params": [channel]
            }
            
            await self.ws.send(json.dumps(subscription))
            logger.info(f"Subscribed to trades for {symbol}")
            
            return True
        except Exception as e:
            logger.error(f"Error subscribing to trades for {symbol}: {e}")
            return False
    
    def register_ticker_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a callback for ticker updates."""
        self.ticker_callbacks.append(callback)
    
    def register_orderbook_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a callback for orderbook updates."""
        self.orderbook_callbacks.append(callback)
    
    def register_kline_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]):
        """Register a callback for kline updates."""
        self.kline_callbacks.append(callback)
    
    def register_trade_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a callback for trade updates."""
        self.trade_callbacks.append(callback)
    
    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest ticker data for a symbol."""
        return self.ticker_data.get(symbol)
    
    def get_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest orderbook data for a symbol."""
        return self.orderbook_data.get(symbol)
    
    def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100) -> List[Dict[str, Any]]:
        """Get kline data for a symbol and interval."""
        key = f"{symbol}_{interval}"
        if key not in self.kline_data:
            return []
        
        # Return the requested number of klines, or all available if less than limit
        klines = self.kline_data[key]
        if limit > 0 and len(klines) > limit:
            return klines[-limit:]
        return klines
    
    def get_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade data for a symbol."""
        if symbol not in self.trade_data:
            return []
        
        # Return the requested number of trades, or all available if less than limit
        trades = self.trade_data[symbol]
        if limit > 0 and len(trades) > limit:
            return trades[-limit:]
        return trades
    
    async def fetch_historical_klines(self, symbol: str, interval: str = '1m', 
                                     limit: int = 500) -> List[Dict[str, Any]]:
        """
        Fetch historical kline data from MEXC REST API.
        
        Args:
            symbol: Trading pair in format like 'BTC/USDC'
            interval: Timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
            limit: Number of klines to fetch (max 1000)
            
        Returns:
            List of kline data
        """
        # Convert symbol format (BTC/USDC -> BTCUSDC)
        formatted_symbol = symbol.replace('/', '')
        
        try:
            url = f"{API_ENDPOINTS['spot']}/api/v3/klines"
            params = {
                'symbol': formatted_symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # MEXC limit is 1000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Format kline data
                        klines = []
                        for item in data:
                            klines.append({
                                'symbol': symbol,
                                'interval': interval,
                                'timestamp': item[0],
                                'open': float(item[1]),
                                'high': float(item[2]),
                                'low': float(item[3]),
                                'close': float(item[4]),
                                'volume': float(item[5]),
                                'is_closed': True,
                            })
                        
                        return klines
                    else:
                        logger.error(f"Error fetching historical klines: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching historical klines: {e}")
            return []
    
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order on MEXC.
        
        Args:
            symbol: Trading pair in format like 'BTC/USDC'
            side: 'buy' or 'sell'
            order_type: 'limit' or 'market'
            quantity: Order quantity
            price: Order price (required for limit orders)
            
        Returns:
            Order response from MEXC
        """
        if not self.api_key or not self.api_secret:
            logger.error("API key and secret required for trading")
            return {'error': 'API credentials not provided'}
        
        # Convert symbol format (BTC/USDC -> BTCUSDC)
        formatted_symbol = symbol.replace('/', '')
        
        try:
            # Prepare order parameters
            params = {
                'symbol': formatted_symbol,
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': str(quantity),
                'timestamp': int(time.time() * 1000)
            }
            
            # Add price for limit orders
            if order_type.lower() == 'limit' and price is not None:
                params['price'] = str(price)
            
            # Add signature
            params['signature'] = self._generate_signature(params)
            
            # Place order
            url = f"{API_ENDPOINTS['spot']}/api/v3/order"
            headers = {'X-MEXC-APIKEY': self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=params, headers=headers) as response:
                    result = await response.json()
                    
                    if response.status == 200:
                        logger.info(f"Order placed successfully: {result}")
                        return result
                    else:
                        logger.error(f"Error placing order: {result}")
                        return {'error': result.get('msg', 'Unknown error')}
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'error': str(e)}
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate signature for authenticated requests."""
        if not self.api_secret:
            return ''
        
        # Convert params to query string
        query_string = urlencode(params)
        
        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
