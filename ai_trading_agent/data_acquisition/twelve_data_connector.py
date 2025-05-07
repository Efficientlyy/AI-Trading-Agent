"""
Twelve Data Connector Module

This module provides a connector for the Twelve Data API, which offers real-time
and historical market data for cryptocurrencies and other financial instruments.
It supports both REST API calls for historical data and WebSocket connections for real-time data.
"""

import logging
import asyncio
import json
import os
import time
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import websockets
import aiohttp
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class TwelveDataConnector:
    """
    Connector for the Twelve Data API.
    
    This class provides methods for accessing real-time and historical market data
    from the Twelve Data API, with support for both REST API and WebSocket connections.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Twelve Data connector.
        
        Args:
            config: Configuration dictionary for the connector
        """
        self.config = config or {}
        
        # API configuration
        self.api_key = self.config.get("api_key") or os.getenv("TWELVE_DATA_API_KEY")
        if not self.api_key:
            logger.warning("No Twelve Data API key provided. API calls will likely fail.")
        
        self.base_url = self.config.get("base_url", "https://api.twelvedata.com")
        self.ws_url = self.config.get("ws_url", "wss://ws.twelvedata.com/v1")
        
        # WebSocket connection
        self.ws = None
        self.ws_connected = False
        self.ws_task = None
        self.ws_reconnect_delay = 1  # Initial reconnect delay in seconds
        self.ws_max_reconnect_delay = 60  # Maximum reconnect delay in seconds
        
        # Subscriptions
        self.subscriptions = {}
        
        # Data storage
        self.latest_prices = {}
        self.price_history = {}
        self.last_update_time = {}
        
        # Callbacks
        self.price_callbacks = []
        self.bar_callbacks = []
        
        logger.info("Initialized TwelveDataConnector")
    
    async def connect(self) -> bool:
        """
        Connect to the Twelve Data WebSocket API.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.ws_connected:
            logger.info("Already connected to Twelve Data WebSocket")
            return True
        
        try:
            # Connect to WebSocket
            self.ws = await websockets.connect(self.ws_url)
            self.ws_connected = True
            
            # Start WebSocket task
            self.ws_task = asyncio.create_task(self._ws_handler())
            
            # Reset reconnect delay
            self.ws_reconnect_delay = 1
            
            logger.info("Connected to Twelve Data WebSocket")
            
            # Resubscribe to previous subscriptions
            if self.subscriptions:
                await self._resubscribe()
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to Twelve Data WebSocket: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the Twelve Data WebSocket API."""
        if not self.ws_connected:
            logger.info("Not connected to Twelve Data WebSocket")
            return
        
        try:
            # Cancel WebSocket task
            if self.ws_task and not self.ws_task.done():
                self.ws_task.cancel()
            
            # Close WebSocket connection
            if self.ws:
                await self.ws.close()
            
            self.ws_connected = False
            self.ws = None
            
            logger.info("Disconnected from Twelve Data WebSocket")
        except Exception as e:
            logger.error(f"Error disconnecting from Twelve Data WebSocket: {e}")
    
    async def _ws_handler(self) -> None:
        """Handle WebSocket messages."""
        try:
            async for message in self.ws:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Handle different message types
                    if "event" in data:
                        event = data["event"]
                        
                        if event == "price":
                            await self._handle_price_event(data)
                        elif event == "bar":
                            await self._handle_bar_event(data)
                        elif event == "heartbeat":
                            # Just log heartbeat
                            logger.debug("Received heartbeat")
                        else:
                            logger.debug(f"Received unknown event: {event}")
                    else:
                        logger.debug(f"Received message without event: {data}")
                except json.JSONDecodeError:
                    logger.error(f"Error decoding WebSocket message: {message}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
        except asyncio.CancelledError:
            logger.info("WebSocket handler task cancelled")
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
            
            # Reconnect if not intentionally disconnected
            if self.ws_connected:
                asyncio.create_task(self._reconnect())
    
    async def _reconnect(self) -> None:
        """Reconnect to the WebSocket with exponential backoff."""
        # Set connection status to disconnected
        self.ws_connected = False
        
        # Wait before reconnecting
        logger.info(f"Reconnecting in {self.ws_reconnect_delay} seconds...")
        await asyncio.sleep(self.ws_reconnect_delay)
        
        # Increase reconnect delay with exponential backoff
        self.ws_reconnect_delay = min(self.ws_reconnect_delay * 2, self.ws_max_reconnect_delay)
        
        # Try to reconnect
        success = await self.connect()
        
        # If reconnection failed, try again
        if not success:
            asyncio.create_task(self._reconnect())
    
    async def _resubscribe(self) -> None:
        """Resubscribe to previous subscriptions after reconnect."""
        for symbol, events in self.subscriptions.items():
            for event in events:
                await self.subscribe(symbol, event)
    
    async def _handle_price_event(self, data: Dict[str, Any]) -> None:
        """
        Handle price event from WebSocket.
        
        Args:
            data: Price event data
        """
        if "symbol" not in data or "price" not in data:
            logger.warning(f"Invalid price event data: {data}")
            return
        
        symbol = data["symbol"]
        price = float(data["price"])
        timestamp = data.get("timestamp", int(time.time()))
        
        # Store latest price
        self.latest_prices[symbol] = price
        self.last_update_time[symbol] = timestamp
        
        # Store in price history (keep last 1000 prices)
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            "timestamp": timestamp,
            "price": price
        })
        
        # Limit history size
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
        
        # Notify callbacks
        for callback in self.price_callbacks:
            try:
                callback(symbol, price, timestamp)
            except Exception as e:
                logger.error(f"Error in price callback: {e}")
    
    async def _handle_bar_event(self, data: Dict[str, Any]) -> None:
        """
        Handle bar event from WebSocket.
        
        Args:
            data: Bar event data
        """
        if "symbol" not in data or "bar" not in data:
            logger.warning(f"Invalid bar event data: {data}")
            return
        
        symbol = data["symbol"]
        bar = data["bar"]
        
        # Notify callbacks
        for callback in self.bar_callbacks:
            try:
                callback(symbol, bar)
            except Exception as e:
                logger.error(f"Error in bar callback: {e}")
    
    async def subscribe(self, symbol: str, event_type: str) -> bool:
        """
        Subscribe to a symbol for real-time updates.
        
        Args:
            symbol: Symbol to subscribe to
            event_type: Event type ("price" or "bar")
            
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.ws_connected:
            logger.warning("Not connected to WebSocket, cannot subscribe")
            return False
        
        try:
            # Create subscription message
            subscription = {
                "action": "subscribe",
                "params": {
                    "symbols": symbol,
                    "events": [event_type]
                }
            }
            
            # Add API key if available
            if self.api_key:
                subscription["params"]["api_key"] = self.api_key
            
            # Send subscription message
            await self.ws.send(json.dumps(subscription))
            
            # Update subscriptions
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = []
            
            if event_type not in self.subscriptions[symbol]:
                self.subscriptions[symbol].append(event_type)
            
            logger.info(f"Subscribed to {symbol} for {event_type} events")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")
            return False
    
    async def unsubscribe(self, symbol: str, event_type: Optional[str] = None) -> bool:
        """
        Unsubscribe from a symbol.
        
        Args:
            symbol: Symbol to unsubscribe from
            event_type: Event type to unsubscribe from (None for all)
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        if not self.ws_connected:
            logger.warning("Not connected to WebSocket, cannot unsubscribe")
            return False
        
        try:
            # Create unsubscription message
            events = [event_type] if event_type else self.subscriptions.get(symbol, [])
            unsubscription = {
                "action": "unsubscribe",
                "params": {
                    "symbols": symbol,
                    "events": events
                }
            }
            
            # Send unsubscription message
            await self.ws.send(json.dumps(unsubscription))
            
            # Update subscriptions
            if symbol in self.subscriptions:
                if event_type:
                    if event_type in self.subscriptions[symbol]:
                        self.subscriptions[symbol].remove(event_type)
                    
                    if not self.subscriptions[symbol]:
                        del self.subscriptions[symbol]
                else:
                    del self.subscriptions[symbol]
            
            logger.info(f"Unsubscribed from {symbol} for {event_type or 'all'} events")
            return True
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {e}")
            return False
    
    def is_subscribed(self, symbol: str, event_type: Optional[str] = None) -> bool:
        """
        Check if subscribed to a symbol.
        
        Args:
            symbol: Symbol to check
            event_type: Event type to check (None for any)
            
        Returns:
            True if subscribed, False otherwise
        """
        if symbol not in self.subscriptions:
            return False
        
        if event_type:
            return event_type in self.subscriptions[symbol]
        
        return bool(self.subscriptions[symbol])
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Latest price or None if not available
        """
        return self.latest_prices.get(symbol)
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get price history for a symbol.
        
        Args:
            symbol: Symbol to get history for
            limit: Maximum number of items to return
            
        Returns:
            List of price history items
        """
        history = self.price_history.get(symbol, [])
        return history[-limit:] if limit < len(history) else history
    
    def register_price_callback(self, callback: Callable[[str, float, int], None]) -> None:
        """
        Register a callback for price updates.
        
        Args:
            callback: Callback function that takes (symbol, price, timestamp)
        """
        self.price_callbacks.append(callback)
    
    def register_bar_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register a callback for bar updates.
        
        Args:
            callback: Callback function that takes (symbol, bar)
        """
        self.bar_callbacks.append(callback)
    
    async def get_time_series(
        self, 
        symbol: str, 
        interval: str = "1day", 
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get time series data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            interval: Time interval (e.g., "1min", "1hour", "1day")
            days: Number of days of data to get
            
        Returns:
            Dictionary containing time series data
        """
        try:
            # Calculate number of bars based on interval and days
            outputsize = self._calculate_outputsize(interval, days)
            
            # Build API URL
            url = f"{self.base_url}/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "format": "JSON"
            }
            
            if self.api_key:
                params["apikey"] = self.api_key
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Error getting time series data: {response.status}")
                        return {}
                    
                    data = await response.json()
                    
                    # Check for API errors
                    if "status" in data and data["status"] == "error":
                        logger.error(f"API error: {data.get('message', 'Unknown error')}")
                        return {}
                    
                    return data
        except Exception as e:
            logger.error(f"Error getting time series data: {e}")
            return {}
    
    def _calculate_outputsize(self, interval: str, days: int) -> int:
        """
        Calculate the number of bars needed based on interval and days.
        
        Args:
            interval: Time interval
            days: Number of days
            
        Returns:
            Number of bars
        """
        # Map interval to minutes
        interval_minutes = {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "30min": 30,
            "45min": 45,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "1day": 1440,
            "1week": 10080
        }
        
        # Get minutes for this interval
        minutes = interval_minutes.get(interval)
        if not minutes:
            # Default to daily
            minutes = 1440
        
        # Calculate bars
        minutes_in_day = 1440
        trading_minutes_in_day = 1440  # For crypto, market is 24/7
        
        bars_per_day = trading_minutes_in_day / minutes
        total_bars = int(bars_per_day * days)
        
        # Limit to 5000 (API limit)
        return min(total_bars, 5000)
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get a quote for a symbol.
        
        Args:
            symbol: Symbol to get quote for
            
        Returns:
            Dictionary containing quote data
        """
        try:
            # Build API URL
            url = f"{self.base_url}/quote"
            params = {
                "symbol": symbol,
                "format": "JSON"
            }
            
            if self.api_key:
                params["apikey"] = self.api_key
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Error getting quote: {response.status}")
                        return {}
                    
                    data = await response.json()
                    
                    # Check for API errors
                    if "status" in data and data["status"] == "error":
                        logger.error(f"API error: {data.get('message', 'Unknown error')}")
                        return {}
                    
                    return data
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return {}
