"""
WebSocket handler for real-time updates.

This module provides a WebSocket connection for real-time updates to the frontend.
"""

import logging
from typing import Dict, List, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
import json
import asyncio
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["websocket"])

# Store active connections
active_connections: Dict[str, WebSocket] = {}

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections[session_id] = websocket
    logger.info(f"WebSocket connection established for session {session_id}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep the connection alive with periodic heartbeats
        while True:
            # Wait for messages from the client
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                logger.info(f"Received message from session {session_id}: {message}")
                
                # Echo the message back to the client
                await websocket.send_json({
                    "type": "echo",
                    "data": message,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from session {session_id}")
                continue
                
            # Send a heartbeat every 30 seconds
            await asyncio.sleep(30)
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat()
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for session {session_id}")
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        if session_id in active_connections:
            del active_connections[session_id]

# Function to send a message to a specific session
async def send_message_to_session(session_id: str, message: Dict[str, Any]) -> bool:
    """Send a message to a specific session."""
    if session_id in active_connections:
        try:
            await active_connections[session_id].send_json(message)
            return True
        except Exception as e:
            logger.error(f"Error sending message to session {session_id}: {e}")
            return False
    return False

# Function to broadcast a message to all sessions
async def broadcast_message(message: Dict[str, Any]) -> int:
    """Broadcast a message to all sessions."""
    sent_count = 0
    for session_id, connection in active_connections.items():
        try:
            await connection.send_json(message)
            sent_count += 1
        except Exception as e:
            logger.error(f"Error broadcasting message to session {session_id}: {e}")
    return sent_count
