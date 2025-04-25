"""
WebSocket Endpoints

This module provides FastAPI WebSocket endpoints for real-time data streaming.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .manager import (
    connection_manager, 
    get_connection_manager, 
    ConnectionManager,
    WebSocketMessage, 
    MessageType
)
from backend.security.audit_logging import log_security_event, SecurityEventType
from backend.auth.jwt import get_current_user_from_token, get_token_data
from backend.auth.dependencies import get_current_user, get_current_active_user

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time data streaming
    """
    connection_id = None
    try:
        # Accept connection and get connection ID
        connection_id = await connection_manager.connect(websocket)
        
        # Send welcome message with connection ID
        welcome_msg = WebSocketMessage(
            type=MessageType.SYSTEM_STATUS,
            data={
                "message": "Connected to AI Trading Agent WebSocket Server",
                "connection_id": connection_id,
                "status": "connected"
            }
        ).dict()
        
        await websocket.send_json(welcome_msg)
        
        # Wait for authentication or messages
        authenticated = False
        while True:
            # Receive message
            try:
                message = await websocket.receive_json()
            except ValueError as e:
                # Invalid JSON
                error_msg = WebSocketMessage(
                    type=MessageType.ERROR,
                    data={
                        "message": "Invalid JSON message",
                        "error": str(e)
                    }
                ).dict()
                await websocket.send_json(error_msg)
                continue
            
            # Process message based on action
            action = message.get("action")
            
            # Authentication required for most actions
            if action != "authenticate" and not authenticated:
                error_msg = WebSocketMessage(
                    type=MessageType.ERROR,
                    data={
                        "message": "Authentication required",
                        "error": "unauthorized"
                    }
                ).dict()
                await websocket.send_json(error_msg)
                continue
            
            # Process different actions
            if action == "authenticate":
                token = message.get("token")
                if not token:
                    error_msg = WebSocketMessage(
                        type=MessageType.ERROR,
                        data={
                            "message": "Authentication failed",
                            "error": "missing_token"
                        }
                    ).dict()
                    await websocket.send_json(error_msg)
                    continue
                
                # Try to authenticate
                auth_success = await connection_manager.authenticate(connection_id, token)
                
                if auth_success:
                    authenticated = True
                    try:
                        # Get user ID for logging
                        token_data = await get_token_data(token)
                        user_id = token_data.user_id
                        
                        # Log successful WebSocket authentication
                        log_security_event(
                            event_type=SecurityEventType.AUTH_SUCCESS,
                            message=f"WebSocket authentication successful for user: {user_id}",
                            user_id=str(user_id),
                            details={
                                "connection_id": connection_id,
                                "websocket": True,
                            }
                        )
                    except Exception as e:
                        # Continue even if logging fails
                        logger.error(f"Error logging WebSocket auth: {str(e)}")
                    
                    # Send success message
                    auth_msg = WebSocketMessage(
                        type=MessageType.SYSTEM_STATUS,
                        data={
                            "message": "Authentication successful",
                            "status": "authenticated"
                        }
                    ).dict()
                    await websocket.send_json(auth_msg)
                else:
                    # Authentication failed
                    error_msg = WebSocketMessage(
                        type=MessageType.ERROR,
                        data={
                            "message": "Authentication failed",
                            "error": "invalid_token"
                        }
                    ).dict()
                    await websocket.send_json(error_msg)
                    
                    # Log failed authentication
                    log_security_event(
                        event_type=SecurityEventType.AUTH_FAILURE,
                        message=f"WebSocket authentication failed",
                        details={
                            "connection_id": connection_id,
                            "websocket": True,
                            "error": "invalid_token"
                        },
                        severity="WARNING"
                    )
            
            elif action == "subscribe":
                # Subscribe to topics
                topics = message.get("topics", [])
                if not topics:
                    error_msg = WebSocketMessage(
                        type=MessageType.ERROR,
                        data={
                            "message": "No topics specified",
                            "error": "missing_topics"
                        }
                    ).dict()
                    await websocket.send_json(error_msg)
                    continue
                
                # Subscribe to topics
                subscribed = await connection_manager.subscribe(connection_id, topics)
                
                # Send subscription confirmation
                sub_msg = WebSocketMessage(
                    type=MessageType.SYSTEM_STATUS,
                    data={
                        "message": "Subscription updated",
                        "subscribed": subscribed
                    }
                ).dict()
                await websocket.send_json(sub_msg)
            
            elif action == "unsubscribe":
                # Unsubscribe from topics
                topics = message.get("topics", [])
                if not topics:
                    # If no topics specified, unsubscribe from all
                    topics = connection_manager.get_subscriptions(connection_id)
                
                # Unsubscribe from topics
                unsubscribed = await connection_manager.unsubscribe(connection_id, topics)
                
                # Send unsubscription confirmation
                unsub_msg = WebSocketMessage(
                    type=MessageType.SYSTEM_STATUS,
                    data={
                        "message": "Unsubscribed from topics",
                        "unsubscribed": unsubscribed
                    }
                ).dict()
                await websocket.send_json(unsub_msg)
            
            elif action == "ping":
                # Simple ping-pong for connection testing
                pong_msg = WebSocketMessage(
                    type=MessageType.SYSTEM_STATUS,
                    data={
                        "message": "pong",
                        "timestamp": datetime.now().isoformat()
                    }
                ).dict()
                await websocket.send_json(pong_msg)
            
            elif action == "get_subscriptions":
                # Get current subscriptions
                subs = connection_manager.get_subscriptions(connection_id)
                
                # Send subscriptions list
                subs_msg = WebSocketMessage(
                    type=MessageType.SYSTEM_STATUS,
                    data={
                        "message": "Current subscriptions",
                        "subscriptions": subs
                    }
                ).dict()
                await websocket.send_json(subs_msg)
            
            else:
                # Unknown action
                error_msg = WebSocketMessage(
                    type=MessageType.ERROR,
                    data={
                        "message": f"Unknown action: {action}",
                        "error": "unknown_action"
                    }
                ).dict()
                await websocket.send_json(error_msg)
    
    except WebSocketDisconnect:
        # Client disconnected
        if connection_id:
            logger.info(f"WebSocket client disconnected: {connection_id}")
            await connection_manager.disconnect(connection_id)
    
    except Exception as e:
        # Unexpected error
        logger.error(f"WebSocket error: {str(e)}")
        if connection_id:
            await connection_manager.disconnect(connection_id)


@router.websocket("/ws/market/{symbol}")
async def market_data_websocket(
    websocket: WebSocket,
    symbol: str,
    interval: Optional[str] = Query("1m", description="Data interval (1m, 5m, 15m, 1h, 1d)")
):
    """
    WebSocket endpoint for streaming real-time market data for a specific symbol
    """
    connection_id = None
    topic = f"market_data:{symbol}:{interval}"
    
    try:
        # Accept connection and get connection ID
        connection_id = await connection_manager.connect(websocket)
        
        # Auto-subscribe to the symbol's market data
        await connection_manager.subscribe(connection_id, [topic])
        
        # Send initial message
        welcome_msg = WebSocketMessage(
            type=MessageType.SYSTEM_STATUS,
            data={
                "message": f"Connected to {symbol} market data stream",
                "connection_id": connection_id,
                "symbol": symbol,
                "interval": interval,
                "status": "connected"
            }
        ).dict()
        
        await websocket.send_json(welcome_msg)
        
        # Keep connection open and handle messages
        while True:
            try:
                message = await websocket.receive_json()
                # Simple ping-pong for now
                if message.get("action") == "ping":
                    pong_msg = WebSocketMessage(
                        type=MessageType.SYSTEM_STATUS,
                        data={
                            "message": "pong",
                            "timestamp": datetime.now().isoformat()
                        }
                    ).dict()
                    await websocket.send_json(pong_msg)
            except ValueError:
                # Invalid JSON, ignore
                pass
    
    except WebSocketDisconnect:
        # Client disconnected
        if connection_id:
            logger.info(f"Market data WebSocket client disconnected: {connection_id}")
            await connection_manager.disconnect(connection_id)
    
    except Exception as e:
        # Unexpected error
        logger.error(f"Market data WebSocket error: {str(e)}")
        if connection_id:
            await connection_manager.disconnect(connection_id)


# Export the WebSocket startup and shutdown functions
async def startup_websocket_manager():
    """Start the WebSocket manager and heartbeat task"""
    await connection_manager.start_heartbeat(interval_seconds=30)
    logger.info("WebSocket manager started")


async def shutdown_websocket_manager():
    """Stop the WebSocket manager and heartbeat task"""
    await connection_manager.stop_heartbeat()
    logger.info("WebSocket manager stopped")