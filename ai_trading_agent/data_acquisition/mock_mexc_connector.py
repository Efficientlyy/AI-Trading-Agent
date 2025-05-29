"""
Mock MEXC Exchange Connector

This module provides a mock implementation of the MEXC connector
that can be used when the real connection fails, allowing the backend
to continue running without crashing.
"""

import asyncio
import json
import logging
import time
import random
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta

from ..common import get_logger
from ..config.mexc_config import TRADING_PAIRS

# Configure logger
logger = get_logger(__name__)

class MockMexcConnector:
    """
    Mock MEXC Exchange connector that mimics the behavior of the real connector
    but uses generated data instead of real exchange data.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize the mock MEXC connector."""
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Connection state
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
        self.default_pair = "BTC/USDC"
        
        # Subscription tracking
        self.active_subscriptions = set()
        
        # Generate initial mock data
        self._generate_mock_data()
        
        logger.info(f"Initialized MockMexcConnector with default pair: {self.default_pair}")
    
    def _generate_mock_data(self):
        """Generate mock data for all trading pairs."""
        for symbol in TRADING_PAIRS:
            # Generate ticker data
            self.ticker_data[symbol] = {
                'symbol': symbol,
                'price': 50000.0 + random.uniform(-1000, 1000),
                'volume': 10000.0 + random.uniform(-1000, 1000),
                'timestamp': int(time.time() * 1000)
            }
            
            # Generate orderbook data
            self.orderbook_data[symbol] = {
                'symbol': symbol,
                'bids': [[50000.0 - i * 10, 1.0 - i * 0.1] for i in range(10)],
                'asks': [[50000.0 + i * 10, 1.0 - i * 0.1] for i in range(10)],
                'timestamp': int(time.time() * 1000)
            }
            
            # Generate kline data
            base_price = 50000.0
            for interval in ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']:
                key = f"{symbol}:{interval}"
                self.kline_data[key] = {
                    'symbol': symbol,
                    'interval': interval,
                    'open': base_price,
                    'high': base_price + random.uniform(0, 100),
                    'low': base_price - random.uniform(0, 100),
                    'close': base_price + random.uniform(-50, 50),
                    'volume': 100.0 + random.uniform(0, 50),
                    'timestamp': int(time.time() * 1000)
                }
    
    async def connect(self) -> bool:
        """
        Connect to mock MEXC WebSocket API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.ws_connected:
            logger.info("Already connected to mock MEXC WebSocket")
            return True
        
        try:
            self.ws_connected = True
            
            # Start WebSocket handler task
            self.ws_task = asyncio.create_task(self._ws_handler())
            
            # Start ping task to simulate keep-alive
            self.ping_task = asyncio.create_task(self._ping_loop())
            
            logger.info("Connected to mock MEXC WebSocket")
            
            # Subscribe to default pair
            await self.subscribe_ticker(self.default_pair)
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to mock MEXC WebSocket: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from mock MEXC WebSocket."""
        if not self.ws_connected:
            return
        
        try:
            # Cancel tasks
            if self.ws_task:
                self.ws_task.cancel()
            
            if self.ping_task:
                self.ping_task.cancel()
            
            self.ws_connected = False
            
            logger.info("Disconnected from mock MEXC WebSocket")
        except Exception as e:
            logger.error(f"Error disconnecting from mock MEXC WebSocket: {e}")
    
    async def _ws_handler(self):
        """Simulate WebSocket message handling by periodically updating mock data."""
        try:
            while True:
                # Update mock data
                self._update_mock_data()
                
                # Trigger callbacks
                await self._trigger_callbacks()
                
                # Wait for next update
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("WebSocket handler task cancelled")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")
    
    async def _ping_loop(self):
        """Simulate ping loop to keep connection alive."""
        try:
            while True:
                await asyncio.sleep(30)
                logger.debug("Sent ping to mock MEXC WebSocket")
        except asyncio.CancelledError:
            logger.info("Ping task cancelled")
        except Exception as e:
            logger.error(f"Error in ping loop: {e}")
    
    def _update_mock_data(self):
        """Update mock data with random changes."""
        current_time = int(time.time() * 1000)
        
        for symbol in TRADING_PAIRS:
            # Update ticker data
            if symbol in self.ticker_data:
                price = self.ticker_data[symbol]['price']
                self.ticker_data[symbol]['price'] = price * (1 + random.uniform(-0.001, 0.001))
                self.ticker_data[symbol]['volume'] += random.uniform(-10, 10)
                self.ticker_data[symbol]['timestamp'] = current_time
            
            # Update orderbook data
            if symbol in self.orderbook_data:
                self.orderbook_data[symbol]['timestamp'] = current_time
                
                # Update bids and asks
                for i in range(len(self.orderbook_data[symbol]['bids'])):
                    self.orderbook_data[symbol]['bids'][i][0] += random.uniform(-1, 1)
                    self.orderbook_data[symbol]['asks'][i][0] += random.uniform(-1, 1)
            
            # Update kline data
            for interval in ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']:
                key = f"{symbol}:{interval}"
                if key in self.kline_data:
                    kline = self.kline_data[key]
                    close = kline['close'] * (1 + random.uniform(-0.001, 0.001))
                    high = max(kline['high'], close)
                    low = min(kline['low'], close)
                    
                    self.kline_data[key] = {
                        'symbol': symbol,
                        'interval': interval,
                        'open': kline['open'],
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': kline['volume'] + random.uniform(-1, 1),
                        'timestamp': current_time
                    }
    
    async def _trigger_callbacks(self):
        """Trigger callbacks for subscribed data."""
        for symbol in self.active_subscriptions:
            # Trigger ticker callbacks
            if symbol in self.ticker_data:
                for callback in self.ticker_callbacks:
                    try:
                        await callback(symbol, self.ticker_data[symbol])
                    except Exception as e:
                        logger.error(f"Error in ticker callback: {e}")
            
            # Trigger orderbook callbacks
            if symbol in self.orderbook_data:
                for callback in self.orderbook_callbacks:
                    try:
                        await callback(symbol, self.orderbook_data[symbol])
                    except Exception as e:
                        logger.error(f"Error in orderbook callback: {e}")
            
            # Trigger kline callbacks
            for interval in ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']:
                key = f"{symbol}:{interval}"
                if key in self.kline_data:
                    for callback in self.kline_callbacks:
                        try:
                            await callback(symbol, interval, self.kline_data[key])
                        except Exception as e:
                            logger.error(f"Error in kline callback: {e}")
    
    async def subscribe_ticker(self, symbol: str) -> bool:
        """
        Subscribe to ticker data for a symbol.
        
        Args:
            symbol: Trading pair in format like 'BTC/USDC'
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        try:
            self.active_subscriptions.add(symbol)
            logger.info(f"Subscribed to mock ticker data for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to mock ticker data for {symbol}: {e}")
            return False
    
    async def subscribe_orderbook(self, symbol: str) -> bool:
        """
        Subscribe to orderbook data for a symbol.
        
        Args:
            symbol: Trading pair in format like 'BTC/USDC'
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        try:
            self.active_subscriptions.add(symbol)
            logger.info(f"Subscribed to mock orderbook data for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to mock orderbook data for {symbol}: {e}")
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
        try:
            self.active_subscriptions.add(symbol)
            logger.info(f"Subscribed to mock kline data for {symbol} ({interval})")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to mock kline data for {symbol} ({interval}): {e}")
            return False
    
    async def subscribe_trades(self, symbol: str) -> bool:
        """
        Subscribe to trade data for a symbol.
        
        Args:
            symbol: Trading pair in format like 'BTC/USDC'
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        try:
            self.active_subscriptions.add(symbol)
            logger.info(f"Subscribed to mock trade data for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to mock trade data for {symbol}: {e}")
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
    
    def get_ticker(self, symbol: str):
        """Get the latest ticker data for a symbol."""
        return self.ticker_data.get(symbol)
    
    def get_orderbook(self, symbol: str):
        """Get the latest orderbook data for a symbol."""
        return self.orderbook_data.get(symbol)
    
    def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100):
        """Get kline data for a symbol and interval."""
        key = f"{symbol}:{interval}"
        kline = self.kline_data.get(key)
        
        if kline:
            # Generate historical klines
            klines = []
            current_time = int(time.time() * 1000)
            
            for i in range(limit):
                timestamp = current_time - i * self._get_interval_ms(interval)
                klines.append({
                    'symbol': symbol,
                    'interval': interval,
                    'timestamp': timestamp,
                    'open': kline['open'] * (1 + random.uniform(-0.001 * i, 0.001 * i)),
                    'high': kline['high'] * (1 + random.uniform(-0.001 * i, 0.001 * i)),
                    'low': kline['low'] * (1 + random.uniform(-0.001 * i, 0.001 * i)),
                    'close': kline['close'] * (1 + random.uniform(-0.001 * i, 0.001 * i)),
                    'volume': kline['volume'] * (1 + random.uniform(-0.001 * i, 0.001 * i)),
                    'is_closed': True,
                })
            
            return klines
        
        return []
    
    def _get_interval_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds."""
        intervals = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }
        
        return intervals.get(interval, 60 * 1000)