"""
WebSocket handler for real-time updates.

This module provides a WebSocket connection for real-time updates to the frontend.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging

# Import event bus module consistently
from ..common.event_bus import get_event_bus

# Import technical analysis agent API
from .data_source_api import get_ta_agent
from ..agent.technical_analysis_agent import DataMode

# Set up logging
logger = logging.getLogger(__name__)

# Create router for main WebSocket endpoints
router = APIRouter(tags=["websocket"])

# Create separate routers for different WebSocket endpoints
from .crypto_websocket import router as crypto_router
from .mexc_websocket import router as mexc_router

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
                
                # Process message based on action
                if 'action' in message:
                    action = message['action']
                    
                    # Handle set_data_mode action
                    if action == 'set_data_mode' and 'mode' in message:
                        ta_agent = get_ta_agent()
                        data_mode = message['mode']
                        
                        # Update the technical analysis agent's data mode
                        if data_mode == 'mock' or data_mode == 'real':
                            # Update data source config via the agent's method
                            current_mode = ta_agent.get_data_source_type()
                            
                            # Only toggle if the requested mode is different from current mode
                            if current_mode != data_mode:
                                new_mode = ta_agent.toggle_data_source()
                                logger.info(f"Changed Technical Analysis Agent data mode from {current_mode} to {new_mode}")
                                
                                # Publish event to notify orchestrator and other components
                                event_bus = get_event_bus()
                                event_bus.publish(
                                    'data_source_toggled',
                                    {'is_mock': new_mode == 'mock'},
                                    source='websocket'
                                )
                                logger.info(f"Published data_source_toggled event: is_mock={new_mode == 'mock'}")
                            else:
                                logger.info(f"Technical Analysis Agent already in {data_mode} mode")
                        
                        # Send confirmation back to client
                        await websocket.send_json({
                            "type": "data_mode_update",
                            "mode": ta_agent.get_data_source_type(),  # Get the actual current mode
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    # Handle ping action
                    elif action == 'ping':
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    # Handle other actions - echo back for now
                    else:
                        await websocket.send_json({
                            "type": "echo",
                            "data": message,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                # If no action specified, just echo back
                else:
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
