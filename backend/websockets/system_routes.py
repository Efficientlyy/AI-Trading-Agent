"""
System Control WebSocket Routes

This module defines WebSocket endpoints for real-time system control and monitoring.
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
from backend.system_control import SYSTEM_STATUS, MOCK_AGENTS
from backend.auth.jwt import get_current_user_from_token

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define topic names
SYSTEM_STATUS_TOPIC = "system.status"
AGENT_STATUS_TOPIC = "agent.status"
SESSION_STATUS_TOPIC = "session.status"


@router.websocket("/ws/system")
async def system_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """
    WebSocket endpoint for system control and monitoring.
    
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
    
    # Subscribe to system status topics
    await manager.subscribe(connection_id, [SYSTEM_STATUS_TOPIC, AGENT_STATUS_TOPIC, SESSION_STATUS_TOPIC])
    
    # Send initial system status
    await manager.send_personal_message(
        WebSocketMessage(
            type=MessageType.SYSTEM_STATUS,
            data=SYSTEM_STATUS
        ).dict(),
        connection_id
    )
    
    # Send initial agent statuses
    await manager.send_personal_message(
        WebSocketMessage(
            type=MessageType.AGENT_STATUS,
            data={"agents": MOCK_AGENTS}
        ).dict(),
        connection_id
    )
    
    try:
        # Process messages until disconnection
        while True:
            # Wait for a message from the client
            raw_data = await websocket.receive_text()
            
            try:
                # Parse message JSON
                data = json.loads(raw_data)
                
                # Handle command messages
                if "action" in data:
                    action = data["action"]
                    
                    if action == "get_system_status":
                        # Send current system status
                        await manager.send_personal_message(
                            WebSocketMessage(
                                type=MessageType.SYSTEM_STATUS,
                                data=SYSTEM_STATUS
                            ).dict(),
                            connection_id
                        )
                    
                    elif action == "get_agent_status":
                        # Send current agent statuses
                        await manager.send_personal_message(
                            WebSocketMessage(
                                type=MessageType.AGENT_STATUS,
                                data={"agents": MOCK_AGENTS}
                            ).dict(),
                            connection_id
                        )
                    
                    elif action == "start_system":
                        # This would call the actual system start API
                        # For now, just acknowledge the command
                        await manager.send_personal_message(
                            WebSocketMessage(
                                type=MessageType.SYSTEM_STATUS,
                                data={
                                    "action": "start_system",
                                    "status": "acknowledged",
                                    "message": "System start command received"
                                }
                            ).dict(),
                            connection_id
                        )
                    
                    elif action == "stop_system":
                        # This would call the actual system stop API
                        # For now, just acknowledge the command
                        await manager.send_personal_message(
                            WebSocketMessage(
                                type=MessageType.SYSTEM_STATUS,
                                data={
                                    "action": "stop_system",
                                    "status": "acknowledged",
                                    "message": "System stop command received"
                                }
                            ).dict(),
                            connection_id
                        )
                    
                    elif action == "start_agent" and "agent_id" in data:
                        # This would call the actual agent start API
                        # For now, just acknowledge the command
                        await manager.send_personal_message(
                            WebSocketMessage(
                                type=MessageType.AGENT_STATUS,
                                data={
                                    "action": "start_agent",
                                    "agent_id": data["agent_id"],
                                    "status": "acknowledged",
                                    "message": f"Agent start command received for {data['agent_id']}"
                                }
                            ).dict(),
                            connection_id
                        )
                    
                    elif action == "stop_agent" and "agent_id" in data:
                        # This would call the actual agent stop API
                        # For now, just acknowledge the command
                        await manager.send_personal_message(
                            WebSocketMessage(
                                type=MessageType.AGENT_STATUS,
                                data={
                                    "action": "stop_agent",
                                    "agent_id": data["agent_id"],
                                    "status": "acknowledged",
                                    "message": f"Agent stop command received for {data['agent_id']}"
                                }
                            ).dict(),
                            connection_id
                        )
                    
                    else:
                        # Unknown action
                        await manager.send_personal_message(
                            WebSocketMessage(
                                type=MessageType.ERROR,
                                data={
                                    "code": "unknown_action",
                                    "message": f"Unknown action: {action}"
                                }
                            ).dict(),
                            connection_id
                        )
                
            except json.JSONDecodeError:
                # Invalid JSON
                await manager.send_personal_message(
                    WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"code": "invalid_json", "message": "Invalid JSON message"}
                    ).dict(),
                    connection_id
                )
            
            except Exception as e:
                # Other errors
                logger.error(f"Error processing message: {str(e)}")
                await manager.send_personal_message(
                    WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"code": "processing_error", "message": f"Error processing message: {str(e)}"}
                    ).dict(),
                    connection_id
                )
    
    except WebSocketDisconnect:
        # Client disconnected
        logger.info(f"System WebSocket client disconnected: {connection_id}")
    
    finally:
        # Clean up connection
        await manager.unsubscribe(connection_id, [SYSTEM_STATUS_TOPIC, AGENT_STATUS_TOPIC, SESSION_STATUS_TOPIC])
        await manager.disconnect(connection_id)


# Function to broadcast system status updates to all subscribers
async def broadcast_system_status(
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """Broadcast current system status to all subscribers"""
    message = WebSocketMessage(
        type=MessageType.SYSTEM_STATUS,
        data=SYSTEM_STATUS
    ).dict()
    
    await manager.broadcast(message, SYSTEM_STATUS_TOPIC)


# Function to broadcast agent status updates to all subscribers
async def broadcast_agent_status(
    agent_id: str,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """Broadcast agent status update to all subscribers"""
    if agent_id not in MOCK_AGENTS:
        logger.error(f"Cannot broadcast status for unknown agent: {agent_id}")
        return
    
    message = WebSocketMessage(
        type=MessageType.AGENT_UPDATE,
        data=MOCK_AGENTS[agent_id]
    ).dict()
    
    await manager.broadcast(message, f"agent.{agent_id}")
    await manager.broadcast(message, AGENT_STATUS_TOPIC)


# Function to broadcast session status updates to all subscribers
async def broadcast_session_status(
    session_id: str,
    session_data: Dict[str, Any],
    manager: ConnectionManager = Depends(get_connection_manager)
):
    """Broadcast session status update to all subscribers"""
    message = WebSocketMessage(
        type=MessageType.SESSION_UPDATE,
        data=session_data
    ).dict()
    
    await manager.broadcast(message, f"session.{session_id}")
    await manager.broadcast(message, SESSION_STATUS_TOPIC)
