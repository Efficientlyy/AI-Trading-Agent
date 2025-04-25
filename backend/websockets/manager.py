"""
WebSocket Manager

This module provides WebSocket connection management and real-time data streaming
for the AI Trading Agent platform. It handles authentication, connection lifecycle,
and message distribution for market data, portfolio updates, and trading events.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Set, Any, Optional, Callable, Awaitable
from datetime import datetime
import uuid
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect, Depends, status
from pydantic import BaseModel, ValidationError

from backend.database.models import User
from backend.auth.jwt import get_current_user_from_token
from ai_trading_agent.trading_engine.models import OrderStatus, Order, Trade

# Setup logging
logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """Connection status for WebSocket clients"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"


class MessageType(str, Enum):
    """Types of messages that can be sent over WebSockets"""
    MARKET_DATA = "market_data"
    ORDER_UPDATE = "order_update"
    TRADE_UPDATE = "trade_update"
    PORTFOLIO_UPDATE = "portfolio_update"
    STRATEGY_SIGNAL = "strategy_signal"
    SENTIMENT_UPDATE = "sentiment_update"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages"""
    type: MessageType
    timestamp: datetime = None
    data: Any
    
    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class ConnectionManager:
    """
    WebSocket connection manager for handling client connections,
    authentication, and message broadcasting
    """
    
    def __init__(self):
        # Map of connection_id to WebSocket instance
        self.active_connections: Dict[str, WebSocket] = {}
        # Map of connection_id to user_id
        self.connection_to_user: Dict[str, int] = {}
        # Map of user_id to set of connection_ids
        self.user_connections: Dict[int, Set[str]] = {}
        # Map of subscriptions (topic) to set of connection_ids
        self.subscriptions: Dict[str, Set[str]] = {}
        # Connection status
        self.connection_status: Dict[str, ConnectionStatus] = {}
        # Last activity timestamp
        self.last_activity: Dict[str, float] = {}
        # Heartbeat task
        self.heartbeat_task = None
        
    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection and return connection ID
        
        Args:
            websocket: The WebSocket connection to accept
            
        Returns:
            str: Unique connection ID
        """
        # Generate unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Accept the connection
        await websocket.accept()
        
        # Store connection
        self.active_connections[connection_id] = websocket
        self.connection_status[connection_id] = ConnectionStatus.CONNECTED
        self.last_activity[connection_id] = time.time()
        
        logger.info(f"New WebSocket connection established: {connection_id}")
        return connection_id
    
    async def authenticate(self, connection_id: str, token: str) -> bool:
        """
        Authenticate a WebSocket connection using JWT token
        
        Args:
            connection_id: Connection ID to authenticate
            token: JWT token
            
        Returns:
            bool: True if authentication is successful
        """
        if connection_id not in self.active_connections:
            logger.error(f"Cannot authenticate non-existent connection: {connection_id}")
            return False
        
        try:
            # Verify token and get user
            user = await get_current_user_from_token(token)
            user_id = user.id
            
            # Associate connection with user
            self.connection_to_user[connection_id] = user_id
            
            # Add connection to user's connections
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            
            self.user_connections[user_id].add(connection_id)
            self.connection_status[connection_id] = ConnectionStatus.AUTHENTICATED
            self.last_activity[connection_id] = time.time()
            
            logger.info(f"Connection {connection_id} authenticated for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed for connection {connection_id}: {str(e)}")
            return False
    
    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect a WebSocket connection and clean up resources
        
        Args:
            connection_id: ID of the connection to disconnect
        """
        # Skip if connection doesn't exist
        if connection_id not in self.active_connections:
            return
        
        logger.info(f"Disconnecting WebSocket connection: {connection_id}")
        
        # Remove from user connections if authenticated
        if connection_id in self.connection_to_user:
            user_id = self.connection_to_user[connection_id]
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            del self.connection_to_user[connection_id]
        
        # Remove from subscriptions
        for topic, subscribers in list(self.subscriptions.items()):
            if connection_id in subscribers:
                subscribers.discard(connection_id)
                if not subscribers:
                    del self.subscriptions[topic]
        
        # Clean up status and activity
        self.connection_status[connection_id] = ConnectionStatus.DISCONNECTED
        if connection_id in self.last_activity:
            del self.last_activity[connection_id]
        
        # Remove from active connections
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
    
    async def subscribe(self, connection_id: str, topics: List[str]) -> List[str]:
        """
        Subscribe a connection to one or more topics
        
        Args:
            connection_id: ID of the connection to subscribe
            topics: List of topics to subscribe to
            
        Returns:
            List[str]: List of topics successfully subscribed to
        """
        # Only authenticated connections can subscribe
        if connection_id not in self.connection_to_user:
            logger.warning(f"Unauthenticated connection {connection_id} attempted to subscribe")
            return []
        
        # Only active connections can subscribe
        if connection_id not in self.active_connections:
            logger.warning(f"Inactive connection {connection_id} attempted to subscribe")
            return []
        
        successful_subscriptions = []
        
        # Subscribe to each topic
        for topic in topics:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            
            self.subscriptions[topic].add(connection_id)
            successful_subscriptions.append(topic)
            
        logger.info(f"Connection {connection_id} subscribed to topics: {successful_subscriptions}")
        self.last_activity[connection_id] = time.time()
        
        return successful_subscriptions
    
    async def unsubscribe(self, connection_id: str, topics: List[str]) -> List[str]:
        """
        Unsubscribe a connection from one or more topics
        
        Args:
            connection_id: ID of the connection to unsubscribe
            topics: List of topics to unsubscribe from
            
        Returns:
            List[str]: List of topics successfully unsubscribed from
        """
        successful_unsubscriptions = []
        
        # Unsubscribe from each topic
        for topic in topics:
            if topic in self.subscriptions and connection_id in self.subscriptions[topic]:
                self.subscriptions[topic].discard(connection_id)
                successful_unsubscriptions.append(topic)
                
                # Clean up empty subscription sets
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
        
        logger.info(f"Connection {connection_id} unsubscribed from topics: {successful_unsubscriptions}")
        self.last_activity[connection_id] = time.time()
        
        return successful_unsubscriptions
    
    def get_subscriptions(self, connection_id: str) -> List[str]:
        """
        Get all topics a connection is subscribed to
        
        Args:
            connection_id: ID of the connection
            
        Returns:
            List[str]: List of subscribed topics
        """
        subscribed_topics = []
        
        for topic, subscribers in self.subscriptions.items():
            if connection_id in subscribers:
                subscribed_topics.append(topic)
        
        return subscribed_topics
    
    async def send_personal_message(self, message: Dict, connection_id: str) -> bool:
        """
        Send a message to a specific connection
        
        Args:
            message: The message to send
            connection_id: ID of the connection to send to
            
        Returns:
            bool: True if message was sent successfully
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Attempted to send message to non-existent connection: {connection_id}")
            return False
        
        websocket = self.active_connections[connection_id]
        
        try:
            # Convert message to JSON and send
            await websocket.send_json(message)
            self.last_activity[connection_id] = time.time()
            return True
        except Exception as e:
            logger.error(f"Error sending message to connection {connection_id}: {str(e)}")
            # Disconnect the problematic connection
            await self.disconnect(connection_id)
            return False
    
    async def broadcast(self, message: Dict, topic: Optional[str] = None) -> int:
        """
        Broadcast a message to all subscribers of a topic or all connections
        
        Args:
            message: The message to broadcast
            topic: Optional topic to broadcast to (if None, broadcast to all)
            
        Returns:
            int: Number of connections message was sent to
        """
        # Determine target connections
        target_connections = set()
        
        if topic:
            # Send to subscribers of the topic
            if topic in self.subscriptions:
                target_connections = self.subscriptions[topic].copy()
        else:
            # Send to all authenticated connections
            target_connections = set(self.connection_to_user.keys())
        
        # No connections to send to
        if not target_connections:
            return 0
        
        # Keep track of successful sends
        successful_sends = 0
        
        # Send message to each connection
        for connection_id in target_connections:
            if await self.send_personal_message(message, connection_id):
                successful_sends += 1
        
        return successful_sends
    
    async def broadcast_to_user(self, message: Dict, user_id: int) -> int:
        """
        Broadcast a message to all connections for a specific user
        
        Args:
            message: The message to broadcast
            user_id: ID of the user to send to
            
        Returns:
            int: Number of connections message was sent to
        """
        if user_id not in self.user_connections:
            return 0
        
        user_connections = self.user_connections[user_id].copy()
        successful_sends = 0
        
        for connection_id in user_connections:
            if await self.send_personal_message(message, connection_id):
                successful_sends += 1
        
        return successful_sends
    
    async def start_heartbeat(self, interval_seconds: int = 30) -> None:
        """
        Start sending heartbeat messages to all connections at regular intervals
        
        Args:
            interval_seconds: Interval between heartbeats in seconds
        """
        if self.heartbeat_task:
            # Already running
            return
        
        async def heartbeat_loop():
            try:
                while True:
                    # Send heartbeat to active connections
                    timestamp = datetime.now()
                    
                    heartbeat_message = WebSocketMessage(
                        type=MessageType.HEARTBEAT,
                        timestamp=timestamp,
                        data={"timestamp": timestamp.isoformat()}
                    ).dict()
                    
                    # For each connection, check if it's been too long since activity
                    current_time = time.time()
                    inactive_connections = []
                    
                    for conn_id, last_active in self.last_activity.items():
                        # If inactive for 2 minutes, mark for disconnection
                        if current_time - last_active > 120:
                            inactive_connections.append(conn_id)
                        else:
                            # Send heartbeat to active connections
                            await self.send_personal_message(heartbeat_message, conn_id)
                    
                    # Disconnect inactive connections
                    for conn_id in inactive_connections:
                        logger.info(f"Disconnecting inactive connection: {conn_id}")
                        await self.disconnect(conn_id)
                    
                    # Wait for next interval
                    await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                # Task was cancelled
                logger.info("Heartbeat task cancelled")
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
        
        # Start the heartbeat task
        self.heartbeat_task = asyncio.create_task(heartbeat_loop())
        logger.info(f"Started heartbeat task with interval {interval_seconds}s")
    
    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat task"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
            logger.info("Stopped heartbeat task")


# Create a singleton instance of the connection manager
connection_manager = ConnectionManager()


async def get_connection_manager() -> ConnectionManager:
    """Dependency for getting the connection manager"""
    return connection_manager