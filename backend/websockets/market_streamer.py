"""
Market Data Streaming Service

This module provides real-time market data streaming functionality for the
AI Trading Agent platform. It handles fetching market data from various sources
and broadcasting it to subscribed WebSocket clients.
"""

import asyncio
import logging
import json
import random
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import time

from .manager import (
    connection_manager,
    WebSocketMessage,
    MessageType
)
from ai_trading_agent.data_acquisition.data_service import data_service, DataService
from ai_trading_agent.data_acquisition.base_provider import BaseDataProvider

# Setup logging
logger = logging.getLogger(__name__)


class MarketDataStreamer:
    """
    Service for streaming real-time market data to WebSocket clients
    """
    
    def __init__(self):
        self.running = False
        self.tasks = {}  # Map of symbol:interval to task
        self.last_data = {}  # Cache of last data point for each symbol:interval
        self.data_service = None  # Will be initialized in startup
        self.mock_mode = False  # Whether to use mock data or real data
        self.update_intervals = {
            "1m": 60,  # 1 minute
            "5m": 300,  # 5 minutes
            "15m": 900,  # 15 minutes
            "1h": 3600,  # 1 hour
            "4h": 14400,  # 4 hours
            "1d": 86400,  # 1 day
        }
        # In mock mode, we'll accelerate the updates for better testing
        self.mock_update_intervals = {
            "1m": 5,  # 5 seconds
            "5m": 10,  # 10 seconds
            "15m": 15,  # 15 seconds
            "1h": 20,  # 20 seconds
            "4h": 30,  # 30 seconds
            "1d": 60,  # 60 seconds
        }
    
    async def startup(self, data_service: Optional[DataService] = None, mock_mode: bool = False):
        """
        Initialize the market data streamer
        
        Args:
            data_service: Optional data service to use
            mock_mode: Whether to use mock data
        """
        self.mock_mode = mock_mode
        
        # Initialize data service if not provided
        if data_service:
            self.data_service = data_service
        else:
            # Import here to avoid circular imports
            from ai_trading_agent.data_acquisition.data_service import data_service
            self.data_service = data_service
        
        self.running = True
        logger.info(f"Market data streamer started (mock_mode={mock_mode})")
    
    async def shutdown(self):
        """Stop all streaming tasks and clean up resources"""
        self.running = False
        
        # Cancel all streaming tasks
        for key, task in list(self.tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.tasks.clear()
        logger.info("Market data streamer stopped")
    
    def get_stream_key(self, symbol: str, interval: str) -> str:
        """
        Generate a unique key for a market data stream
        
        Args:
            symbol: Asset symbol
            interval: Data interval
            
        Returns:
            str: Stream key
        """
        return f"market_data:{symbol}:{interval}"
    
    def get_subscription_topic(self, symbol: str, interval: str) -> str:
        """
        Get the subscription topic for a market data stream
        
        Args:
            symbol: Asset symbol
            interval: Data interval
            
        Returns:
            str: Subscription topic
        """
        return self.get_stream_key(symbol, interval)
    
    async def start_stream(self, symbol: str, interval: str = "1m") -> bool:
        """
        Start streaming market data for a symbol and interval
        
        Args:
            symbol: Asset symbol to stream
            interval: Data interval
            
        Returns:
            bool: True if stream started successfully
        """
        stream_key = self.get_stream_key(symbol, interval)
        
        # Skip if already streaming
        if stream_key in self.tasks and not self.tasks[stream_key].done():
            logger.debug(f"Stream already active: {stream_key}")
            return True
        
        # Create and start the streaming task
        task = asyncio.create_task(self._stream_data(symbol, interval))
        self.tasks[stream_key] = task
        
        logger.info(f"Started market data stream: {stream_key}")
        return True
    
    async def stop_stream(self, symbol: str, interval: str = "1m") -> bool:
        """
        Stop streaming market data for a symbol and interval
        
        Args:
            symbol: Asset symbol to stop streaming
            interval: Data interval
            
        Returns:
            bool: True if stream stopped successfully
        """
        stream_key = self.get_stream_key(symbol, interval)
        
        # Skip if not streaming
        if stream_key not in self.tasks:
            return True
        
        # Cancel the streaming task
        task = self.tasks[stream_key]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Remove from tasks
        del self.tasks[stream_key]
        
        logger.info(f"Stopped market data stream: {stream_key}")
        return True
    
    async def _fetch_market_data(self, symbol: str, interval: str) -> Dict[str, Any]:
        """
        Fetch market data for a symbol and interval
        
        Args:
            symbol: Asset symbol
            interval: Data interval
            
        Returns:
            Dict[str, Any]: Market data
        """
        if self.mock_mode:
            return self._generate_mock_data(symbol, interval)
        
        try:
            # Get the latest candlestick data
            ohlcv_data = await self.data_service.get_ohlcv(
                symbol=symbol, 
                timeframe=interval,
                limit=1  # Only get the latest candle
            )
            
            if not ohlcv_data or len(ohlcv_data) == 0:
                logger.warning(f"No market data available for {symbol} ({interval})")
                # Return mock data as fallback
                return self._generate_mock_data(symbol, interval)
            
            # Get the latest candle
            latest_candle = ohlcv_data[-1]
            
            # Format the data
            return {
                "symbol": symbol,
                "interval": interval,
                "timestamp": latest_candle[0],
                "open": latest_candle[1],
                "high": latest_candle[2],
                "low": latest_candle[3],
                "close": latest_candle[4],
                "volume": latest_candle[5],
                "is_mock": False
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol} ({interval}): {str(e)}")
            # Return mock data as fallback
            return self._generate_mock_data(symbol, interval, error=True)
    
    def _generate_mock_data(self, symbol: str, interval: str, error: bool = False) -> Dict[str, Any]:
        """
        Generate mock market data
        
        Args:
            symbol: Asset symbol
            interval: Data interval
            error: Whether this is fallback data due to an error
            
        Returns:
            Dict[str, Any]: Mock market data
        """
        # Get last data point for this symbol:interval or create a new one
        last_key = f"{symbol}:{interval}"
        last_data = self.last_data.get(last_key, None)
        
        now = datetime.now()
        
        if not last_data:
            # Initial values depend on the symbol
            if symbol.startswith("BTC"):
                base_price = 60000 + random.uniform(-5000, 5000)
                volatility = 500
            elif symbol.startswith("ETH"):
                base_price = 3500 + random.uniform(-300, 300)
                volatility = 50
            elif any(s in symbol for s in ["AAPL", "MSFT", "GOOGL", "AMZN"]):
                base_price = 150 + random.uniform(-20, 20)
                volatility = 2
            else:
                base_price = 100 + random.uniform(-10, 10)
                volatility = 1
            
            # Initial mock data
            mock_data = {
                "symbol": symbol,
                "interval": interval,
                "timestamp": int(now.timestamp() * 1000),
                "open": base_price,
                "high": base_price * (1 + random.uniform(0, 0.02)),
                "low": base_price * (1 - random.uniform(0, 0.02)),
                "close": base_price * (1 + random.uniform(-0.015, 0.015)),
                "volume": random.uniform(100, 1000),
                "is_mock": True,
                "is_error_fallback": error
            }
        else:
            # Generate next candle based on previous close
            prev_close = last_data["close"]
            price_change_pct = random.uniform(-0.02, 0.02)  # -2% to +2%
            new_close = prev_close * (1 + price_change_pct)
            
            # Calculate other fields
            new_open = prev_close
            new_high = max(new_open, new_close) * (1 + random.uniform(0, 0.01))
            new_low = min(new_open, new_close) * (1 - random.uniform(0, 0.01))
            new_volume = last_data["volume"] * (1 + random.uniform(-0.3, 0.3))
            
            mock_data = {
                "symbol": symbol,
                "interval": interval,
                "timestamp": int(now.timestamp() * 1000),
                "open": new_open,
                "high": new_high,
                "low": new_low,
                "close": new_close,
                "volume": new_volume,
                "is_mock": True,
                "is_error_fallback": error
            }
        
        # Save this data point
        self.last_data[last_key] = mock_data
        return mock_data
    
    async def _stream_data(self, symbol: str, interval: str):
        """
        Stream market data for a symbol and interval
        
        Args:
            symbol: Asset symbol
            interval: Data interval
        """
        topic = self.get_subscription_topic(symbol, interval)
        
        # Determine update interval
        if self.mock_mode:
            update_seconds = self.mock_update_intervals.get(interval, 5)
        else:
            update_seconds = self.update_intervals.get(interval, 60)
        
        try:
            while self.running:
                # Check if there are any subscribers
                if topic in connection_manager.subscriptions and connection_manager.subscriptions[topic]:
                    # Fetch market data
                    market_data = await self._fetch_market_data(symbol, interval)
                    
                    # Create WebSocket message
                    message = WebSocketMessage(
                        type=MessageType.MARKET_DATA,
                        data=market_data
                    ).dict()
                    
                    # Broadcast to subscribers
                    sent_count = await connection_manager.broadcast(message, topic)
                    
                    if sent_count > 0:
                        logger.debug(
                            f"Broadcasted market data for {symbol} ({interval}) "
                            f"to {sent_count} connections"
                        )
                
                # Wait for next update
                await asyncio.sleep(update_seconds)
        
        except asyncio.CancelledError:
            # Task was cancelled
            logger.info(f"Market data stream cancelled: {symbol} ({interval})")
        
        except Exception as e:
            logger.error(f"Error in market data stream {symbol} ({interval}): {str(e)}")


# Create a singleton instance of the market data streamer
market_data_streamer = MarketDataStreamer()


async def get_market_data_streamer() -> MarketDataStreamer:
    """Dependency for getting the market data streamer"""
    return market_data_streamer


async def startup_market_data_streamer(mock_mode: bool = False):
    """Start the market data streamer"""
    await market_data_streamer.startup(mock_mode=mock_mode)


async def shutdown_market_data_streamer():
    """Stop the market data streamer"""
    await market_data_streamer.shutdown()