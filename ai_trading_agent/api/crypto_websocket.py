"""
Crypto WebSocket API

This module implements a WebSocket endpoint for streaming real-time
cryptocurrency data from Twelve Data to frontend clients.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Depends
from pydantic import BaseModel

from ..data_acquisition.twelve_data_connector import TwelveDataConnector
from ..common import get_logger

# Configure logger
logger = get_logger(__name__)

# Create router
router = APIRouter()

# Store active connections
active_connections: Dict[str, List[WebSocket]] = {}
# Store price history (limited to last 300 candles per symbol+interval)
price_history: Dict[str, List[Dict[str, Any]]] = {}
# Store latest prices
latest_prices: Dict[str, float] = {}

# Initialize TwelveData connector
twelve_data = TwelveDataConnector()

# Connect to Twelve Data WebSocket (should be called on startup)
async def connect_to_twelve_data():
    """Connect to Twelve Data WebSocket API on startup."""
    try:
        success = await twelve_data.connect()
        if success:
            logger.info("Successfully connected to Twelve Data WebSocket API")
            return True
        else:
            logger.error("Failed to connect to Twelve Data WebSocket API")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Twelve Data WebSocket API: {e}")
        return False

# WebSocket connection handler
@router.websocket("/ws/crypto/{symbol}/{interval}")
async def websocket_endpoint(
    websocket: WebSocket,
    symbol: str,
    interval: str = "1m"
):
    """WebSocket endpoint for streaming real-time crypto data."""
    # Convert symbol format (BTC-USD to BTC/USD)
    formatted_symbol = symbol.replace("-", "/")
    connection_id = f"{formatted_symbol}:{interval}"
    
    await websocket.accept()
    
    # Add to active connections
    if connection_id not in active_connections:
        active_connections[connection_id] = []
    active_connections[connection_id].append(websocket)
    
    # Subscribe to Twelve Data WebSocket for this symbol if not already subscribed
    if not twelve_data.is_subscribed(formatted_symbol, "price"):
        success = await twelve_data.subscribe(formatted_symbol, "price")
        if not success:
            logger.error(f"Failed to subscribe to price updates for {formatted_symbol}")
            # Send error message to client
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Failed to subscribe to price updates"
            }))
    
    # Also subscribe to bar data if interval is supported
    if interval in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
        if not twelve_data.is_subscribed(formatted_symbol, "bar"):
            success = await twelve_data.subscribe(formatted_symbol, "bar")
            if not success:
                logger.error(f"Failed to subscribe to bar updates for {formatted_symbol}")
                # Send error message to client
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Failed to subscribe to bar updates"
                }))
    
    # Send historical data if available
    history_key = f"{formatted_symbol}:{interval}"
    if history_key in price_history and price_history[history_key]:
        for candle in price_history[history_key]:
            try:
                await websocket.send_text(json.dumps({
                    "type": "ohlcv",
                    **candle
                }))
            except Exception as e:
                logger.error(f"Error sending historical data: {e}")
    
    # Send latest price if available
    if formatted_symbol in latest_prices:
        try:
            await websocket.send_text(json.dumps({
                "type": "price",
                "symbol": formatted_symbol,
                "price": latest_prices[formatted_symbol],
                "timestamp": int(asyncio.get_event_loop().time() * 1000)
            }))
        except Exception as e:
            logger.error(f"Error sending latest price: {e}")
    
    try:
        # Keep connection open and handle client messages
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle client messages (e.g. changing timeframe)
                if "action" in message:
                    if message["action"] == "change_interval":
                        # Handle interval change
                        new_interval = message.get("interval", interval)
                        # Remove from old connection group
                        active_connections[connection_id].remove(websocket)
                        # Add to new connection group
                        new_connection_id = f"{formatted_symbol}:{new_interval}"
                        if new_connection_id not in active_connections:
                            active_connections[new_connection_id] = []
                        active_connections[new_connection_id].append(websocket)
                        # Update connection_id
                        connection_id = new_connection_id
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
    except WebSocketDisconnect:
        # Remove from active connections
        if connection_id in active_connections:
            active_connections[connection_id].remove(websocket)
            # Clean up empty lists
            if not active_connections[connection_id]:
                del active_connections[connection_id]
                # Unsubscribe if no more clients for this symbol
                symbol_still_active = False
                for conn_id in active_connections:
                    if formatted_symbol in conn_id:
                        symbol_still_active = True
                        break
                if not symbol_still_active:
                    await twelve_data.unsubscribe(formatted_symbol)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Attempt to remove from active connections
        try:
            if connection_id in active_connections:
                active_connections[connection_id].remove(websocket)
        except:
            pass

# Register Twelve Data callbacks
def register_twelve_data_callbacks():
    """Register callbacks for Twelve Data events."""
    # Price update callback
    def price_callback(symbol: str, price: float, timestamp: int):
        # Store latest price
        latest_prices[symbol] = price
        
        # Send to all connected clients
        for conn_id, connections in active_connections.items():
            if symbol in conn_id:
                for connection in connections:
                    asyncio.create_task(connection.send_text(json.dumps({
                        "type": "price",
                        "symbol": symbol,
                        "price": price,
                        "timestamp": timestamp
                    })))
    
    # Bar update callback
    def bar_callback(symbol: str, bar: Dict[str, Any]):
        # Determine interval from bar
        interval = bar.get("interval", "1m")
        
        # Format as OHLCV data
        ohlcv_data = {
            "symbol": symbol,
            "timestamp": bar["timestamp"],
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": bar["close"],
            "volume": bar["volume"]
        }
        
        # Store in price history
        history_key = f"{symbol}:{interval}"
        if history_key not in price_history:
            price_history[history_key] = []
        
        # Check if we need to update the last candle or add a new one
        if price_history[history_key] and price_history[history_key][-1]["timestamp"] == bar["timestamp"]:
            price_history[history_key][-1] = ohlcv_data
        else:
            price_history[history_key].append(ohlcv_data)
            # Limit history size
            if len(price_history[history_key]) > 300:
                price_history[history_key] = price_history[history_key][-300:]
        
        # Send to all connected clients for this symbol and interval
        conn_id = f"{symbol}:{interval}"
        if conn_id in active_connections:
            for connection in active_connections[conn_id]:
                asyncio.create_task(connection.send_text(json.dumps({
                    "type": "ohlcv",
                    **ohlcv_data
                })))
    
    # Register callbacks
    twelve_data.register_price_callback(price_callback)
    twelve_data.register_bar_callback(bar_callback)

# Initialize module
async def initialize():
    """Initialize the module by connecting to Twelve Data and registering callbacks."""
    connected = await connect_to_twelve_data()
    if connected:
        register_twelve_data_callbacks()
        return True
    return False

# Clean up on shutdown
async def shutdown():
    """Clean up resources on shutdown."""
    await twelve_data.disconnect()
