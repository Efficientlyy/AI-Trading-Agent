#!/usr/bin/env python
"""
Multi-Asset Data Service for Trading-Agent System

This module provides data retrieval and processing capabilities for multiple assets
(BTC, ETH, SOL) with support for different timeframes and real-time updates.
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import exchange client
from optimized_mexc_client import OptimizedMEXCClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_service")

class DataCache:
    """Cache for market data"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """Initialize data cache
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time to live in seconds
        """
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        
        logger.info(f"Initialized DataCache with max_size={max_size}, ttl={ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found or expired
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check if item is expired
            if time.time() - self.timestamps[key] > self.ttl:
                self.cache.pop(key)
                self.timestamps.pop(key)
                return None
            
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set item in cache
        
        Args:
            key: Cache key
            value: Item to cache
        """
        with self.lock:
            # If cache is full, remove oldest item
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.timestamps, key=self.timestamps.get)
                self.cache.pop(oldest_key)
                self.timestamps.pop(oldest_key)
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            
            logger.info("Cache cleared")

class MultiAssetDataService:
    """Data service for multiple assets"""
    
    def __init__(self, symbols: List[str] = None, timeframes: List[str] = None):
        """Initialize multi-asset data service
        
        Args:
            symbols: List of symbols to track (default: ["BTC/USDC", "ETH/USDC", "SOL/USDC"])
            timeframes: List of timeframes to support (default: ["1m", "5m", "15m", "1h", "4h", "1d"])
        """
        self.symbols = symbols or ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Initialize exchange client
        self.client = OptimizedMEXCClient()
        
        # Initialize caches
        self.klines_cache = DataCache(max_size=100, ttl=60)  # 1 minute TTL for klines
        self.ticker_cache = DataCache(max_size=100, ttl=10)  # 10 seconds TTL for tickers
        self.orderbook_cache = DataCache(max_size=100, ttl=5)  # 5 seconds TTL for orderbooks
        
        # Initialize WebSocket connections
        self.ws_connections = {}
        self.ws_callbacks = {}
        
        # Initialize data buffers
        self.klines_buffer = {}
        self.ticker_buffer = {}
        self.trades_buffer = {}
        
        for symbol in self.symbols:
            self.klines_buffer[symbol] = {}
            for timeframe in self.timeframes:
                self.klines_buffer[symbol][timeframe] = pd.DataFrame()
            
            self.ticker_buffer[symbol] = None
            self.trades_buffer[symbol] = []
        
        logger.info(f"Initialized MultiAssetDataService with symbols={self.symbols}, timeframes={self.timeframes}")
    
    def get_klines(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Get historical klines (candlestick) data
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            timeframe: Timeframe (e.g., "1m", "5m", "15m", "1h", "4h", "1d")
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check if symbol is supported
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not in supported symbols {self.symbols}")
            return pd.DataFrame()
        
        # Check if timeframe is supported
        if timeframe not in self.timeframes:
            logger.warning(f"Timeframe {timeframe} not in supported timeframes {self.timeframes}")
            return pd.DataFrame()
        
        # Check cache
        cache_key = f"{symbol}_{timeframe}_{limit}"
        cached_data = self.klines_cache.get(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Get klines from exchange
            klines = self.client.get_klines(symbol, timeframe, limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Set timestamp as index
            df.set_index("timestamp", inplace=True)
            
            # Convert columns to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            
            # Update buffer
            self.klines_buffer[symbol][timeframe] = df
            
            # Update cache
            self.klines_cache.set(cache_key, df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting klines for {symbol} {timeframe}: {e}")
            
            # Return buffer data if available
            if not self.klines_buffer[symbol][timeframe].empty:
                logger.info(f"Returning buffered klines for {symbol} {timeframe}")
                return self.klines_buffer[symbol][timeframe]
            
            return pd.DataFrame()
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            
        Returns:
            Dictionary with ticker data
        """
        # Check if symbol is supported
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not in supported symbols {self.symbols}")
            return {}
        
        # Check cache
        cached_data = self.ticker_cache.get(symbol)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Get ticker from exchange
            ticker = self.client.get_ticker(symbol)
            
            # Update buffer
            self.ticker_buffer[symbol] = ticker
            
            # Update cache
            self.ticker_cache.set(symbol, ticker)
            
            return ticker
        
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            
            # Return buffer data if available
            if self.ticker_buffer[symbol] is not None:
                logger.info(f"Returning buffered ticker for {symbol}")
                return self.ticker_buffer[symbol]
            
            return {}
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get current orderbook
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            limit: Number of levels to retrieve
            
        Returns:
            Dictionary with orderbook data
        """
        # Check if symbol is supported
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not in supported symbols {self.symbols}")
            return {"bids": [], "asks": []}
        
        # Check cache
        cache_key = f"{symbol}_{limit}"
        cached_data = self.orderbook_cache.get(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            # Get orderbook from exchange
            orderbook = self.client.get_orderbook(symbol, limit)
            
            # Update cache
            self.orderbook_cache.set(cache_key, orderbook)
            
            return orderbook
        
        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {e}")
            return {"bids": [], "asks": []}
    
    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent trades
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            limit: Number of trades to retrieve
            
        Returns:
            List of trade dictionaries
        """
        # Check if symbol is supported
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not in supported symbols {self.symbols}")
            return []
        
        try:
            # Get trades from exchange
            trades = self.client.get_trades(symbol, limit)
            
            # Update buffer
            self.trades_buffer[symbol] = trades
            
            return trades
        
        except Exception as e:
            logger.error(f"Error getting trades for {symbol}: {e}")
            
            # Return buffer data if available
            if self.trades_buffer[symbol]:
                logger.info(f"Returning buffered trades for {symbol}")
                return self.trades_buffer[symbol]
            
            return []
    
    def subscribe_ticker(self, symbol: str, callback: Callable[[Dict], None]) -> bool:
        """Subscribe to ticker updates
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            callback: Callback function to handle ticker updates
            
        Returns:
            True if subscription successful, False otherwise
        """
        # Check if symbol is supported
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not in supported symbols {self.symbols}")
            return False
        
        # Register callback
        if "ticker" not in self.ws_callbacks:
            self.ws_callbacks["ticker"] = {}
        
        if symbol not in self.ws_callbacks["ticker"]:
            self.ws_callbacks["ticker"][symbol] = []
        
        self.ws_callbacks["ticker"][symbol].append(callback)
        
        # Start WebSocket connection if not already running
        if "ticker" not in self.ws_connections or not self.ws_connections["ticker"].get(symbol):
            self._start_ticker_websocket(symbol)
        
        return True
    
    def subscribe_klines(self, symbol: str, timeframe: str, callback: Callable[[Dict], None]) -> bool:
        """Subscribe to klines updates
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            timeframe: Timeframe (e.g., "1m", "5m", "15m", "1h", "4h", "1d")
            callback: Callback function to handle klines updates
            
        Returns:
            True if subscription successful, False otherwise
        """
        # Check if symbol is supported
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not in supported symbols {self.symbols}")
            return False
        
        # Check if timeframe is supported
        if timeframe not in self.timeframes:
            logger.warning(f"Timeframe {timeframe} not in supported timeframes {self.timeframes}")
            return False
        
        # Register callback
        if "klines" not in self.ws_callbacks:
            self.ws_callbacks["klines"] = {}
        
        key = f"{symbol}_{timeframe}"
        
        if key not in self.ws_callbacks["klines"]:
            self.ws_callbacks["klines"][key] = []
        
        self.ws_callbacks["klines"][key].append(callback)
        
        # Start WebSocket connection if not already running
        if "klines" not in self.ws_connections or not self.ws_connections["klines"].get(key):
            self._start_klines_websocket(symbol, timeframe)
        
        return True
    
    def subscribe_trades(self, symbol: str, callback: Callable[[Dict], None]) -> bool:
        """Subscribe to trades updates
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            callback: Callback function to handle trades updates
            
        Returns:
            True if subscription successful, False otherwise
        """
        # Check if symbol is supported
        if symbol not in self.symbols:
            logger.warning(f"Symbol {symbol} not in supported symbols {self.symbols}")
            return False
        
        # Register callback
        if "trades" not in self.ws_callbacks:
            self.ws_callbacks["trades"] = {}
        
        if symbol not in self.ws_callbacks["trades"]:
            self.ws_callbacks["trades"][symbol] = []
        
        self.ws_callbacks["trades"][symbol].append(callback)
        
        # Start WebSocket connection if not already running
        if "trades" not in self.ws_connections or not self.ws_connections["trades"].get(symbol):
            self._start_trades_websocket(symbol)
        
        return True
    
    def _start_ticker_websocket(self, symbol: str):
        """Start WebSocket connection for ticker updates
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
        """
        try:
            # Initialize WebSocket connections dictionary if not exists
            if "ticker" not in self.ws_connections:
                self.ws_connections["ticker"] = {}
            
            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                self.client.get_ws_endpoint(),
                on_message=lambda ws, msg: self._on_ticker_message(ws, msg, symbol),
                on_error=lambda ws, err: self._on_error(ws, err, "ticker", symbol),
                on_close=lambda ws, close_status_code, close_msg: self._on_close(ws, close_status_code, close_msg, "ticker", symbol),
                on_open=lambda ws: self._on_open(ws, "ticker", symbol)
            )
            
            # Store connection
            self.ws_connections["ticker"][symbol] = ws
            
            # Start WebSocket connection in a separate thread
            threading.Thread(target=ws.run_forever).start()
            
            logger.info(f"Started ticker WebSocket for {symbol}")
        
        except Exception as e:
            logger.error(f"Error starting ticker WebSocket for {symbol}: {e}")
    
    def _start_klines_websocket(self, symbol: str, timeframe: str):
        """Start WebSocket connection for klines updates
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
            timeframe: Timeframe (e.g., "1m", "5m", "15m", "1h", "4h", "1d")
        """
        try:
            # Initialize WebSocket connections dictionary if not exists
            if "klines" not in self.ws_connections:
                self.ws_connections["klines"] = {}
            
            key = f"{symbol}_{timeframe}"
            
            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                self.client.get_ws_endpoint(),
                on_message=lambda ws, msg: self._on_klines_message(ws, msg, symbol, timeframe),
                on_error=lambda ws, err: self._on_error(ws, err, "klines", key),
                on_close=lambda ws, close_status_code, close_msg: self._on_close(ws, close_status_code, close_msg, "klines", key),
                on_open=lambda ws: self._on_open(ws, "klines", key)
            )
            
            # Store connection
            self.ws_connections["klines"][key] = ws
            
            # Start WebSocket connection in a separate thread
            threading.Thread(target=ws.run_forever).start()
            
            logger.info(f"Started klines WebSocket for {symbol} {timeframe}")
        
        except Exception as e:
            logger.error(f"Error starting klines WebSocket for {symbol} {timeframe}: {e}")
    
    def _start_trades_websocket(self, symbol: str):
        """Start WebSocket connection for trades updates
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDC")
        """
        try:
            # Initialize WebSocket connections dictionary if not exists
            if "trades" not in self.ws_connections:
                self.ws_connections["trades"] = {}
            
            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                self.client.get_ws_endpoint(),
                on_message=lambda ws, msg: self._on_trades_message(ws, msg, symbol),
                on_error=lambda ws, err: self._on_error(ws, err, "trades", symbol),
                on_close=lambda ws, close_status_code, close_msg: self._on_close(ws, close_status_code, close_msg, "trades", symbol),
                on_open=lambda ws: self._on_open(ws, "trades", symbol)
            )
            
            # Store connection
            self.ws_connections["trades"][symbol] = ws
            
            # Start WebSocket connection in a separate thread
            threading.Thread(target=ws.run_forever).start()
            
            logger.info(f"Started trades WebSocket for {symbol}")
        
        except Exception as e:
            logger.error(f"Error starting trades WebSocket for {symbol}: {e}")
    
    def _on_ticker_message(self, ws, message: str, symbol: str):
        """Handle ticker WebSocket message
        
        Args:
            ws: WebSocket connection
            message: WebSocket message
            symbol: Trading pair symbol
        """
        try:
            # Parse message
            data = json.loads(message)
            
            # Process ticker data
            ticker = self.client.parse_ws_ticker(data)
            
            # Update buffer
            self.ticker_buffer[symbol] = ticker
            
            # Update cache
            self.ticker_cache.set(symbol, ticker)
            
            # Call callbacks
            if "ticker" in self.ws_callbacks and symbol in self.ws_callbacks["ticker"]:
                for callback in self.ws_callbacks["ticker"][symbol]:
                    try:
                        callback(ticker)
                    except Exception as e:
                        logger.error(f"Error in ticker callback for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing ticker message for {symbol}: {e}")
    
    def _on_klines_message(self, ws, message: str, symbol: str, timeframe: str):
        """Handle klines WebSocket message
        
        Args:
            ws: WebSocket connection
            message: WebSocket message
            symbol: Trading pair symbol
            timeframe: Timeframe
        """
        try:
            # Parse message
            data = json.loads(message)
            
            # Process klines data
            kline = self.client.parse_ws_kline(data)
            
            if not kline:
                return
            
            # Update buffer
            df = self.klines_buffer[symbol][timeframe]
            
            # Convert to DataFrame row
            kline_df = pd.DataFrame([kline], columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            # Convert timestamp to datetime
            kline_df["timestamp"] = pd.to_datetime(kline_df["timestamp"], unit="ms")
            
            # Set timestamp as index
            kline_df.set_index("timestamp", inplace=True)
            
            # Convert columns to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                kline_df[col] = pd.to_numeric(kline_df[col])
            
            # Update buffer
            if df.empty:
                self.klines_buffer[symbol][timeframe] = kline_df
            else:
                # If timestamp already exists, update row
                if kline_df.index[0] in df.index:
                    df.loc[kline_df.index[0]] = kline_df.iloc[0]
                else:
                    # Append new row
                    df = pd.concat([df, kline_df])
                
                # Keep only the latest 1000 candles
                if len(df) > 1000:
                    df = df.iloc[-1000:]
                
                self.klines_buffer[symbol][timeframe] = df
            
            # Update cache
            cache_key = f"{symbol}_{timeframe}_1000"
            self.klines_cache.set(cache_key, self.klines_buffer[symbol][timeframe])
            
            # Call callbacks
            key = f"{symbol}_{timeframe}"
            if "klines" in self.ws_callbacks and key in self.ws_callbacks["klines"]:
                for callback in self.ws_callbacks["klines"][key]:
                    try:
                        callback(kline)
                    except Exception as e:
                        logger.error(f"Error in klines callback for {symbol} {timeframe}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing klines message for {symbol} {timeframe}: {e}")
    
    def _on_trades_message(self, ws, message: str, symbol: str):
        """Handle trades WebSocket message
        
        Args:
            ws: WebSocket connection
            message: WebSocket message
            symbol: Trading pair symbol
        """
        try:
            # Parse message
            data = json.loads(message)
            
            # Process trades data
            trade = self.client.parse_ws_trade(data)
            
            if not trade:
                return
            
            # Update buffer
            self.trades_buffer[symbol].append(trade)
            
            # Keep only the latest 1000 trades
            if len(self.trades_buffer[symbol]) > 1000:
                self.trades_buffer[symbol] = self.trades_buffer[symbol][-1000:]
            
            # Call callbacks
            if "trades" in self.ws_callbacks and symbol in self.ws_callbacks["trades"]:
                for callback in self.ws_callbacks["trades"][symbol]:
                    try:
                        callback(trade)
                    except Exception as e:
                        logger.error(f"Error in trades callback for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing trades message for {symbol}: {e}")
    
    def _on_open(self, ws, channel: str, key: str):
        """Handle WebSocket connection open
        
        Args:
            ws: WebSocket connection
            channel: Channel name
            key: Subscription key
        """
        logger.info(f"WebSocket connection opened for {channel} {key}")
        
        # Subscribe to channel
        if channel == "ticker":
            ws.send(json.dumps({
                "method": "SUBSCRIBE",
                "params": [f"{key.lower()}@ticker"],
                "id": int(time.time() * 1000)
            }))
        
        elif channel == "klines":
            symbol, timeframe = key.split("_")
            ws.send(json.dumps({
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@kline_{timeframe}"],
                "id": int(time.time() * 1000)
            }))
        
        elif channel == "trades":
            ws.send(json.dumps({
                "method": "SUBSCRIBE",
                "params": [f"{key.lower()}@trade"],
                "id": int(time.time() * 1000)
            }))
    
    def _on_error(self, ws, error, channel: str, key: str):
        """Handle WebSocket error
        
        Args:
            ws: WebSocket connection
            error: Error
            channel: Channel name
            key: Subscription key
        """
        logger.error(f"WebSocket error for {channel} {key}: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg, channel: str, key: str):
        """Handle WebSocket connection close
        
        Args:
            ws: WebSocket connection
            close_status_code: Close status code
            close_msg: Close message
            channel: Channel name
            key: Subscription key
        """
        logger.info(f"WebSocket connection closed for {channel} {key}: {close_status_code} {close_msg}")
        
        # Reconnect after delay
        threading.Timer(5.0, self._reconnect_websocket, args=[channel, key]).start()
    
    def _reconnect_websocket(self, channel: str, key: str):
        """Reconnect WebSocket
        
        Args:
            channel: Channel name
            key: Subscription key
        """
        logger.info(f"Reconnecting WebSocket for {channel} {key}")
        
        if channel == "ticker":
            self._start_ticker_websocket(key)
        
        elif channel == "klines":
            symbol, timeframe = key.split("_")
            self._start_klines_websocket(symbol, timeframe)
        
        elif channel == "trades":
            self._start_trades_websocket(key)
    
    def close(self):
        """Close all WebSocket connections"""
        # Close ticker WebSockets
        if "ticker" in self.ws_connections:
            for symbol, ws in self.ws_connections["ticker"].items():
                try:
                    ws.close()
                    logger.info(f"Closed ticker WebSocket for {symbol}")
                except Exception as e:
                    logger.error(f"Error closing ticker WebSocket for {symbol}: {e}")
        
        # Close klines WebSockets
        if "klines" in self.ws_connections:
            for key, ws in self.ws_connections["klines"].items():
                try:
                    ws.close()
                    logger.info(f"Closed klines WebSocket for {key}")
                except Exception as e:
                    logger.error(f"Error closing klines WebSocket for {key}: {e}")
        
        # Close trades WebSockets
        if "trades" in self.ws_connections:
            for symbol, ws in self.ws_connections["trades"].items():
                try:
                    ws.close()
                    logger.info(f"Closed trades WebSocket for {symbol}")
                except Exception as e:
                    logger.error(f"Error closing trades WebSocket for {symbol}: {e}")
        
        logger.info("Closed all WebSocket connections")

if __name__ == "__main__":
    # Example usage
    data_service = MultiAssetDataService()
    
    # Get BTC/USDC klines
    btc_klines = data_service.get_klines("BTC/USDC", "1h", 100)
    print(f"BTC/USDC 1h klines: {len(btc_klines)} candles")
    
    # Get BTC/USDC ticker
    btc_ticker = data_service.get_ticker("BTC/USDC")
    print(f"BTC/USDC ticker: {btc_ticker}")
    
    # Get BTC/USDC orderbook
    btc_orderbook = data_service.get_orderbook("BTC/USDC", 10)
    print(f"BTC/USDC orderbook: {len(btc_orderbook['bids'])} bids, {len(btc_orderbook['asks'])} asks")
    
    # Subscribe to BTC/USDC ticker updates
    def on_ticker(ticker):
        print(f"BTC/USDC ticker update: {ticker}")
    
    data_service.subscribe_ticker("BTC/USDC", on_ticker)
    
    # Keep script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        data_service.close()
