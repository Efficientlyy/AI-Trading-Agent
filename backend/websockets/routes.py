"""
WebSocket Routes

This module defines the WebSocket endpoints for real-time data streaming in the AI Trading Agent platform.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel, ValidationError

from backend.websockets.manager import (
    ConnectionManager, MessageType, WebSocketMessage, 
    get_connection_manager
)
from backend.integration.bridge import get_trading_bridge
from backend.auth.jwt import get_current_user_from_token

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


class SubscriptionRequest(BaseModel):
    """Model for subscription requests"""
    topics: List[str]


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """
    Main WebSocket endpoint for real-time data streaming.
    
    Args:
        websocket: The WebSocket connection
        token: Optional JWT token for authentication
        manager: WebSocket connection manager
    """
    # Accept the connection
    connection_id = await manager.connect(websocket)
    
    # Authenticate if token provided
    authenticated = False
    if token:
        authenticated = await manager.authenticate(connection_id, token)
        if authenticated:
            # Send welcome message
            await manager.send_personal_message(
                WebSocketMessage(
                    type=MessageType.SYSTEM_STATUS,
                    data={"status": "authenticated", "message": "Authentication successful"}
                ).dict(),
                connection_id
            )
        else:
            # Send authentication failure message
            await manager.send_personal_message(
                WebSocketMessage(
                    type=MessageType.ERROR,
                    data={"code": "auth_failed", "message": "Authentication failed"}
                ).dict(),
                connection_id
            )
            await websocket.close(code=1008)  # Policy violation
            return
    
    try:
        # Process messages until disconnection
        while True:
            # Wait for a message from the client
            raw_data = await websocket.receive_text()
            
            try:
                # Parse message JSON
                data = json.loads(raw_data)
                
                # Handle command messages
                if "command" in data:
                    command = data["command"]
                    
                    if command == "subscribe" and "topics" in data:
                        # Subscribe to topics
                        topics = data["topics"]
                        if isinstance(topics, list):
                            subscribed_topics = await manager.subscribe(connection_id, topics)
                            await manager.send_personal_message(
                                WebSocketMessage(
                                    type=MessageType.SYSTEM_STATUS,
                                    data={
                                        "command": "subscribe",
                                        "success": True,
                                        "topics": subscribed_topics
                                    }
                                ).dict(),
                                connection_id
                            )
                    
                    elif command == "unsubscribe" and "topics" in data:
                        # Unsubscribe from topics
                        topics = data["topics"]
                        if isinstance(topics, list):
                            unsubscribed_topics = await manager.unsubscribe(connection_id, topics)
                            await manager.send_personal_message(
                                WebSocketMessage(
                                    type=MessageType.SYSTEM_STATUS,
                                    data={
                                        "command": "unsubscribe",
                                        "success": True,
                                        "topics": unsubscribed_topics
                                    }
                                ).dict(),
                                connection_id
                            )
                    
                    elif command == "get_subscriptions":
                        # Get current subscriptions
                        subscriptions = manager.get_subscriptions(connection_id)
                        await manager.send_personal_message(
                            WebSocketMessage(
                                type=MessageType.SYSTEM_STATUS,
                                data={
                                    "command": "get_subscriptions",
                                    "success": True,
                                    "subscriptions": subscriptions
                                }
                            ).dict(),
                            connection_id
                        )
                    
                    else:
                        # Unknown command
                        await manager.send_personal_message(
                            WebSocketMessage(
                                type=MessageType.ERROR,
                                data={
                                    "code": "unknown_command",
                                    "message": f"Unknown command: {command}"
                                }
                            ).dict(),
                            connection_id
                        )
                
                else:
                    # Not a command, just acknowledge receipt
                    await manager.send_personal_message(
                        WebSocketMessage(
                            type=MessageType.SYSTEM_STATUS,
                            data={"status": "received", "message": "Message received"}
                        ).dict(),
                        connection_id
                    )
            
            except json.JSONDecodeError:
                # Invalid JSON
                await manager.send_personal_message(
                    WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"code": "invalid_json", "message": "Invalid JSON"}
                    ).dict(),
                    connection_id
                )
            
            except Exception as e:
                # Other error
                logger.error(f"Error processing WebSocket message: {str(e)}")
                await manager.send_personal_message(
                    WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"code": "internal_error", "message": "Internal server error"}
                    ).dict(),
                    connection_id
                )
    
    except WebSocketDisconnect:
        # Client disconnected
        logger.info(f"WebSocket client disconnected: {connection_id}")
    
    finally:
        # Clean up connection
        await manager.disconnect(connection_id)


@router.websocket("/ws/market/{symbol}")
async def market_data_websocket(
    websocket: WebSocket,
    symbol: str,
    token: Optional[str] = Query(None),
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """
    WebSocket endpoint for streaming market data for a specific symbol.
    
    Args:
        websocket: The WebSocket connection
        symbol: The market symbol (e.g., "BTC/USD")
        token: Optional JWT token for authentication
        manager: WebSocket connection manager
    """
    # Accept the connection
    connection_id = await manager.connect(websocket)
    
    # Authenticate if token provided
    authenticated = False
    if token:
        authenticated = await manager.authenticate(connection_id, token)
        if not authenticated:
            await manager.send_personal_message(
                WebSocketMessage(
                    type=MessageType.ERROR,
                    data={"code": "auth_failed", "message": "Authentication failed"}
                ).dict(),
                connection_id
            )
            await websocket.close(code=1008)
            return
    
    # Subscribe to market data topic for this symbol
    market_topic = f"market.{symbol}"
    await manager.subscribe(connection_id, [market_topic])
    
    await manager.send_personal_message(
        WebSocketMessage(
            type=MessageType.SYSTEM_STATUS,
            data={
                "status": "subscribed", 
                "message": f"Subscribed to market data for {symbol}",
                "symbol": symbol
            }
        ).dict(),
        connection_id
    )
    
    try:
        # Keep connection open until disconnection
        while True:
            # Just wait for disconnection or ping/pong
            data = await websocket.receive_text()
            
            # If client sends a ping, respond with pong
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        # Client disconnected
        logger.info(f"Market data WebSocket client disconnected: {connection_id}")
    
    finally:
        # Clean up connection
        await manager.unsubscribe(connection_id, [market_topic])
        await manager.disconnect(connection_id)


@router.websocket("/ws/portfolio")
async def portfolio_websocket(
    websocket: WebSocket,
    token: str = Query(...),  # Token is required for portfolio data
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """
    WebSocket endpoint for streaming portfolio updates.
    
    Args:
        websocket: The WebSocket connection
        token: JWT token for authentication (required)
        manager: WebSocket connection manager
    """
    # Accept the connection
    connection_id = await manager.connect(websocket)
    
    # Authenticate (required for portfolio data)
    authenticated = await manager.authenticate(connection_id, token)
    if not authenticated:
        await manager.send_personal_message(
            WebSocketMessage(
                type=MessageType.ERROR,
                data={"code": "auth_required", "message": "Authentication required for portfolio data"}
            ).dict(),
            connection_id
        )
        await websocket.close(code=1008)
        return
    
    # Subscribe to portfolio updates
    portfolio_topic = "portfolio"
    await manager.subscribe(connection_id, [portfolio_topic])
    
    await manager.send_personal_message(
        WebSocketMessage(
            type=MessageType.SYSTEM_STATUS,
            data={"status": "subscribed", "message": "Subscribed to portfolio updates"}
        ).dict(),
        connection_id
    )
    
    try:
        # Get trading bridge to access portfolio data
        trading_bridge = get_trading_bridge()
        
        # Send initial portfolio state
        try:
            portfolio_summary = await trading_bridge.get_portfolio()
            await manager.send_personal_message(
                WebSocketMessage(
                    type=MessageType.PORTFOLIO_UPDATE,
                    data=portfolio_summary.dict()
                ).dict(),
                connection_id
            )
        except Exception as e:
            logger.error(f"Error getting initial portfolio: {str(e)}")
        
        # Keep connection open until disconnection
        while True:
            data = await websocket.receive_text()
            
            # If client requests a refresh, send updated portfolio
            if data == "refresh":
                try:
                    portfolio_summary = await trading_bridge.get_portfolio()
                    await manager.send_personal_message(
                        WebSocketMessage(
                            type=MessageType.PORTFOLIO_UPDATE,
                            data=portfolio_summary.dict()
                        ).dict(),
                        connection_id
                    )
                except Exception as e:
                    logger.error(f"Error refreshing portfolio: {str(e)}")
                    await manager.send_personal_message(
                        WebSocketMessage(
                            type=MessageType.ERROR,
                            data={"code": "refresh_failed", "message": "Failed to refresh portfolio"}
                        ).dict(),
                        connection_id
                    )
    
    except WebSocketDisconnect:
        # Client disconnected
        logger.info(f"Portfolio WebSocket client disconnected: {connection_id}")
    
    finally:
        # Clean up connection
        await manager.unsubscribe(connection_id, [portfolio_topic])
        await manager.disconnect(connection_id)


@router.websocket("/ws/sentiment/{symbol}")
async def sentiment_websocket(
    websocket: WebSocket,
    symbol: str,
    token: Optional[str] = Query(None),
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """
    WebSocket endpoint for streaming sentiment data for a specific symbol.
    
    Args:
        websocket: The WebSocket connection
        symbol: The market symbol (e.g., "BTC/USD")
        token: Optional JWT token for authentication
        manager: WebSocket connection manager
    """
    # Accept the connection
    connection_id = await manager.connect(websocket)
    
    # Authenticate if token provided
    authenticated = False
    if token:
        authenticated = await manager.authenticate(connection_id, token)
        if not authenticated:
            await manager.send_personal_message(
                WebSocketMessage(
                    type=MessageType.ERROR,
                    data={"code": "auth_failed", "message": "Authentication failed"}
                ).dict(),
                connection_id
            )
            await websocket.close(code=1008)
            return
    
    # Subscribe to sentiment data topic for this symbol
    sentiment_topic = f"sentiment.{symbol}"
    await manager.subscribe(connection_id, [sentiment_topic])
    
    await manager.send_personal_message(
        WebSocketMessage(
            type=MessageType.SYSTEM_STATUS,
            data={
                "status": "subscribed", 
                "message": f"Subscribed to sentiment data for {symbol}",
                "symbol": symbol
            }
        ).dict(),
        connection_id
    )
    
    try:
        # Keep connection open until disconnection
        while True:
            data = await websocket.receive_text()
            
            # Just respond to keep connection alive
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        # Client disconnected
        logger.info(f"Sentiment WebSocket client disconnected: {connection_id}")
    
    finally:
        # Clean up connection
        await manager.unsubscribe(connection_id, [sentiment_topic])
        await manager.disconnect(connection_id)