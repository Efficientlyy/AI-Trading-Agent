"""
MEXC WebSocket API

This module implements a WebSocket endpoint for streaming real-time
cryptocurrency data specifically from MEXC exchange to frontend clients,
with a focus on BTC/USDC trading pair.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Depends
from pydantic import BaseModel

from ..config.mexc_config import MEXC_CONFIG, TRADING_PAIRS
# Import factory function later to avoid circular imports
from ..common import get_logger

# Configure logger
logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/ws/mexc", tags=["mexc"])

# Store active connections
active_connections: Dict[str, List[WebSocket]] = {}
# Track subscribed symbols
subscribed_symbols: Set[str] = set()

# Initialize MEXC connector (will be set in connect_to_mexc)
mexc_connector = None

# Connect to MEXC WebSocket (should be called on startup)
async def connect_to_mexc():
    """Connect to MEXC WebSocket API on startup."""
    global mexc_connector
    
    try:
        logger.info("Attempting to connect to MEXC WebSocket API with credentials...")
        logger.info(f"API Key available: {'Yes' if MEXC_CONFIG.get('API_KEY') else 'No'}")
        logger.info(f"API Secret available: {'Yes' if MEXC_CONFIG.get('API_SECRET') else 'No'}")
        
        # Import the factory here to avoid circular imports
        from ..data_acquisition.mexc_connector_factory import create_mexc_connector
        
        # Use the connector factory to get a connector (real or mock)
        mexc_connector = await create_mexc_connector()
        
        if mexc_connector is None:
            logger.error("Failed to create MEXC connector")
            return False
            
        logger.info(f"Using {'mock' if 'Mock' in mexc_connector.__class__.__name__ else 'real'} MEXC connector")
        
        # Auto-subscribe to default pairs
        await subscribe_to_default_pairs()
        
        return True
    except Exception as e:
        logger.error(f"Error connecting to MEXC WebSocket API: {e}")
        logger.exception("Detailed traceback:")
        
        # Create a simplified hard-coded mock connector as last resort
        try:
            # Create a very simple mock with in-place implementation to avoid import issues
            class SimpleMockConnector:
                def __init__(self):
                    self.ticker_callbacks = []
                    self.orderbook_callbacks = []
                    self.kline_callbacks = []
                    self.trade_callbacks = []
                
                async def connect(self):
                    return True
                
                async def disconnect(self):
                    pass
                
                async def subscribe_ticker(self, symbol):
                    return True
                    
                async def subscribe_orderbook(self, symbol):
                    return True
                    
                async def subscribe_kline(self, symbol, interval='1m'):
                    return True
                    
                async def subscribe_trades(self, symbol):
                    return True
                
                def register_ticker_callback(self, callback):
                    self.ticker_callbacks.append(callback)
                
                def register_orderbook_callback(self, callback):
                    self.orderbook_callbacks.append(callback)
                
                def register_kline_callback(self, callback):
                    self.kline_callbacks.append(callback)
                
                def register_trade_callback(self, callback):
                    self.trade_callbacks.append(callback)
                
                def get_ticker(self, symbol):
                    return None
                
                def get_orderbook(self, symbol):
                    return None
            
            mexc_connector = SimpleMockConnector()
            await mexc_connector.connect()
            logger.warning("Using emergency simple mock connector after connection failure")
            return True
        except Exception as mock_error:
            logger.error(f"Even simple mock connector failed: {mock_error}")
            return False

async def subscribe_to_default_pairs():
    """Subscribe to default trading pairs."""
    for pair in TRADING_PAIRS:
        try:
            # Subscribe to ticker (price updates)
            await mexc_connector.subscribe_ticker(pair)
            # Subscribe to kline data (candlesticks)
            await mexc_connector.subscribe_kline(pair, '1m')
            # Track subscribed symbols
            subscribed_symbols.add(pair)
        except Exception as e:
            logger.error(f"Error subscribing to {pair}: {e}")

# WebSocket connection handler for ticker data
@router.websocket("/ticker/{symbol}")
async def websocket_ticker(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for streaming real-time ticker data from MEXC."""
    await websocket.accept()
    
    # Convert symbol format if needed (btc-usdc to BTC/USDC)
    formatted_symbol = symbol.replace("-", "/").upper()
    connection_id = f"ticker:{formatted_symbol}"
    
    # Add to active connections
    if connection_id not in active_connections:
        active_connections[connection_id] = []
    active_connections[connection_id].append(websocket)
    
    logger.info(f"New WebSocket connection for MEXC ticker: {formatted_symbol}")
    
    # Subscribe to ticker if not already subscribed
    if formatted_symbol not in subscribed_symbols:
        success = await mexc_connector.subscribe_ticker(formatted_symbol)
        if success:
            subscribed_symbols.add(formatted_symbol)
        else:
            # Send error message to client
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Failed to subscribe to ticker for {formatted_symbol}"
            }))
    
    # Register callback for this specific client
    async def ticker_callback(symbol: str, data: Dict[str, Any]):
        if symbol == formatted_symbol and connection_id in active_connections:
            try:
                for conn in active_connections[connection_id]:
                    await conn.send_text(json.dumps({
                        "type": "ticker",
                        "symbol": symbol,
                        "data": data
                    }))
            except Exception as e:
                logger.error(f"Error sending ticker data: {e}")
    
    # Add callback
    mexc_connector.register_ticker_callback(ticker_callback)
    
    # Send initial data if available
    initial_data = mexc_connector.get_ticker(formatted_symbol)
    if initial_data:
        await websocket.send_text(json.dumps({
            "type": "ticker",
            "symbol": formatted_symbol,
            "data": initial_data
        }))
    
    try:
        # Keep connection open and handle client messages
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle client messages if needed
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
    except WebSocketDisconnect:
        # Remove from active connections
        if connection_id in active_connections:
            active_connections[connection_id].remove(websocket)
            # Clean up empty lists
            if not active_connections[connection_id]:
                del active_connections[connection_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Remove from active connections
        try:
            if connection_id in active_connections:
                active_connections[connection_id].remove(websocket)
        except:
            pass

# WebSocket connection handler for kline (candlestick) data
@router.websocket("/kline/{symbol}/{interval}")
async def websocket_kline(
    websocket: WebSocket,
    symbol: str,
    interval: str = "1m"
):
    """WebSocket endpoint for streaming real-time kline data from MEXC."""
    await websocket.accept()
    
    # Convert symbol format if needed (btc-usdc to BTC/USDC)
    formatted_symbol = symbol.replace("-", "/").upper()
    connection_id = f"kline:{formatted_symbol}:{interval}"
    
    # Add to active connections
    if connection_id not in active_connections:
        active_connections[connection_id] = []
    active_connections[connection_id].append(websocket)
    
    logger.info(f"New WebSocket connection for MEXC kline: {formatted_symbol} ({interval})")
    
    # Subscribe to kline if not already subscribed
    subscription_key = f"{formatted_symbol}_{interval}"
    if subscription_key not in [f"{s}_{interval}" for s in subscribed_symbols]:
        success = await mexc_connector.subscribe_kline(formatted_symbol, interval)
        if success:
            subscribed_symbols.add(formatted_symbol)
        else:
            # Send error message to client
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Failed to subscribe to kline for {formatted_symbol} ({interval})"
            }))
    
    # Register callback for this specific client
    async def kline_callback(symbol: str, timeframe: str, data: Dict[str, Any]):
        if symbol == formatted_symbol and timeframe == interval and connection_id in active_connections:
            try:
                for conn in active_connections[connection_id]:
                    await conn.send_text(json.dumps({
                        "type": "kline",
                        "symbol": symbol,
                        "interval": timeframe,
                        "data": data
                    }))
            except Exception as e:
                logger.error(f"Error sending kline data: {e}")
    
    # Add callback
    mexc_connector.register_kline_callback(kline_callback)
    
    # Send historical data
    try:
        historical_data = await mexc_connector.fetch_historical_klines(
            formatted_symbol, interval, 100
        )
        if historical_data:
            await websocket.send_text(json.dumps({
                "type": "historical_klines",
                "symbol": formatted_symbol,
                "interval": interval,
                "data": historical_data
            }))
    except Exception as e:
        logger.error(f"Error sending historical data: {e}")
    
    try:
        # Keep connection open and handle client messages
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle client messages if needed
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
    except WebSocketDisconnect:
        # Remove from active connections
        if connection_id in active_connections:
            active_connections[connection_id].remove(websocket)
            # Clean up empty lists
            if not active_connections[connection_id]:
                del active_connections[connection_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Remove from active connections
        try:
            if connection_id in active_connections:
                active_connections[connection_id].remove(websocket)
        except:
            pass

# WebSocket connection handler for orderbook data
@router.websocket("/orderbook/{symbol}")
async def websocket_orderbook(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for streaming real-time orderbook data from MEXC."""
    await websocket.accept()
    
    # Convert symbol format if needed (btc-usdc to BTC/USDC)
    formatted_symbol = symbol.replace("-", "/").upper()
    connection_id = f"orderbook:{formatted_symbol}"
    
    # Add to active connections
    if connection_id not in active_connections:
        active_connections[connection_id] = []
    active_connections[connection_id].append(websocket)
    
    logger.info(f"New WebSocket connection for MEXC orderbook: {formatted_symbol}")
    
    # Subscribe to orderbook if not already subscribed
    if formatted_symbol not in subscribed_symbols:
        success = await mexc_connector.subscribe_orderbook(formatted_symbol)
        if success:
            subscribed_symbols.add(formatted_symbol)
        else:
            # Send error message to client
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Failed to subscribe to orderbook for {formatted_symbol}"
            }))
    
    # Register callback for this specific client
    async def orderbook_callback(symbol: str, data: Dict[str, Any]):
        if symbol == formatted_symbol and connection_id in active_connections:
            try:
                for conn in active_connections[connection_id]:
                    await conn.send_text(json.dumps({
                        "type": "orderbook",
                        "symbol": symbol,
                        "data": data
                    }))
            except Exception as e:
                logger.error(f"Error sending orderbook data: {e}")
    
    # Add callback
    mexc_connector.register_orderbook_callback(orderbook_callback)
    
    # Send initial data if available
    initial_data = mexc_connector.get_orderbook(formatted_symbol)
    if initial_data:
        await websocket.send_text(json.dumps({
            "type": "orderbook",
            "symbol": formatted_symbol,
            "data": initial_data
        }))
    
    try:
        # Keep connection open and handle client messages
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle client messages if needed
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
    except WebSocketDisconnect:
        # Remove from active connections
        if connection_id in active_connections:
            active_connections[connection_id].remove(websocket)
            # Clean up empty lists
            if not active_connections[connection_id]:
                del active_connections[connection_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Remove from active connections
        try:
            if connection_id in active_connections:
                active_connections[connection_id].remove(websocket)
        except:
            pass

# Initialize module
async def initialize():
    """Initialize the module by connecting to MEXC and registering callbacks."""
    try:
        connected = await connect_to_mexc()
        return connected
    except Exception as e:
        logger.error(f"Failed to initialize MEXC WebSocket module: {e}")
        logger.exception("Detailed error traceback:")
        # Return False but don't crash the server
        return False

# Clean up on shutdown
async def shutdown():
    """Clean up resources on shutdown."""
    await mexc_connector.disconnect()
