"""
WebSocket Module

This module provides WebSocket functionality for the AI Trading Agent platform.
"""

from .manager import (
    connection_manager,
    get_connection_manager,
    WebSocketMessage,
    MessageType,
    ConnectionStatus
)
from .endpoints import (
    router as websocket_router,
    startup_websocket_manager,
    shutdown_websocket_manager
)
from .market_streamer import (
    market_data_streamer,
    get_market_data_streamer,
    startup_market_data_streamer,
    shutdown_market_data_streamer
)

__all__ = [
    # Manager
    "connection_manager",
    "get_connection_manager",
    "WebSocketMessage",
    "MessageType",
    "ConnectionStatus",
    
    # Endpoints
    "websocket_router",
    "startup_websocket_manager",
    "shutdown_websocket_manager",
    
    # Market Streamer
    "market_data_streamer",
    "get_market_data_streamer",
    "startup_market_data_streamer",
    "shutdown_market_data_streamer",
]