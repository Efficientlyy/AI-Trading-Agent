"""
WebSocket Manager

This module provides a WebSocket manager for handling real-time data updates.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("websocket_manager")

class WebSocketManager:
    """
    WebSocket manager for handling real-time data updates.
    """
    
    def __init__(self):
        """
        Initialize the WebSocket manager.
        """
        # Active connections
        self.active_connections: Dict[str, Any] = {}
        
        # Subscription channels
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # Message queue
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Data sources
        self.data_sources: Dict[str, Any] = {}
        
        # Last update time
        self.last_update_time: Dict[str, float] = {}
        
        # Update intervals (in seconds)
        self.update_intervals: Dict[str, float] = {
            'dashboard': 5.0,
            'trades': 2.0,
            'positions': 3.0,
            'performance': 10.0,
            'alerts': 1.0
        }
        
        logger.info("WebSocket manager initialized")
    
    async def start(self):
        """
        Start the WebSocket manager.
        """
        # Start background tasks
        self.background_tasks.append(asyncio.create_task(self.process_message_queue()))
        self.background_tasks.append(asyncio.create_task(self.send_periodic_updates()))
        
        logger.info("WebSocket manager started")
    
    async def stop(self):
        """
        Stop the WebSocket manager.
        """
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Clear active connections
        self.active_connections.clear()
        
        # Clear subscriptions
        self.subscriptions.clear()
        
        logger.info("WebSocket manager stopped")
    
    async def connect(self, websocket, client_id: str):
        """
        Handle a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            client_id: The client ID
        """
        # Store connection
        self.active_connections[client_id] = websocket
        
        # Initialize subscriptions
        self.subscriptions[client_id] = set()
        
        # Log connection
        logger.info(f"Client connected: {client_id}")
        
        # Send welcome message
        await self.send_message(client_id, 'system', {
            'message': 'Connected to real-time updates server',
            'client_id': client_id,
            'timestamp': datetime.now().isoformat()
        })
    
    async def disconnect(self, client_id: str):
        """
        Handle a WebSocket disconnection.
        
        Args:
            client_id: The client ID
        """
        # Remove connection
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove subscriptions
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        
        # Log disconnection
        logger.info(f"Client disconnected: {client_id}")
    
    async def receive_message(self, client_id: str, message: str):
        """
        Handle a message from a client.
        
        Args:
            client_id: The client ID
            message: The message
        """
        try:
            # Parse message
            data = json.loads(message)
            
            # Check if message has type
            if 'type' not in data:
                logger.warning(f"Received message without type from client {client_id}")
                return
            
            # Handle message based on type
            message_type = data['type']
            
            if message_type == 'subscribe':
                await self.handle_subscribe(client_id, data.get('data', {}))
            elif message_type == 'unsubscribe':
                await self.handle_unsubscribe(client_id, data.get('data', {}))
            elif message_type == 'ping':
                await self.handle_ping(client_id)
            else:
                # Add message to queue for processing
                await self.message_queue.put({
                    'client_id': client_id,
                    'type': message_type,
                    'data': data.get('data', {}),
                    'timestamp': data.get('timestamp', datetime.now().isoformat())
                })
        except json.JSONDecodeError:
            logger.warning(f"Received invalid JSON from client {client_id}")
        except Exception as e:
            logger.error(f"Error handling message from client {client_id}: {e}")
    
    async def handle_subscribe(self, client_id: str, data: Dict[str, Any]):
        """
        Handle a subscribe message.
        
        Args:
            client_id: The client ID
            data: The message data
        """
        # Get channels to subscribe to
        channels = data.get('channels', [])
        
        # Subscribe to channels
        if client_id in self.subscriptions:
            for channel in channels:
                self.subscriptions[client_id].add(channel)
        
        # Log subscription
        logger.info(f"Client {client_id} subscribed to channels: {channels}")
        
        # Send confirmation
        await self.send_message(client_id, 'subscription_confirmed', {
            'channels': list(self.subscriptions.get(client_id, set())),
            'timestamp': datetime.now().isoformat()
        })
    
    async def handle_unsubscribe(self, client_id: str, data: Dict[str, Any]):
        """
        Handle an unsubscribe message.
        
        Args:
            client_id: The client ID
            data: The message data
        """
        # Get channels to unsubscribe from
        channels = data.get('channels', [])
        
        # Unsubscribe from channels
        if client_id in self.subscriptions:
            for channel in channels:
                self.subscriptions[client_id].discard(channel)
        
        # Log unsubscription
        logger.info(f"Client {client_id} unsubscribed from channels: {channels}")
        
        # Send confirmation
        await self.send_message(client_id, 'unsubscription_confirmed', {
            'channels': list(self.subscriptions.get(client_id, set())),
            'timestamp': datetime.now().isoformat()
        })
    
    async def handle_ping(self, client_id: str):
        """
        Handle a ping message.
        
        Args:
            client_id: The client ID
        """
        # Send pong response
        await self.send_message(client_id, 'pong', {
            'timestamp': datetime.now().isoformat()
        })
    
    async def send_message(self, client_id: str, message_type: str, data: Dict[str, Any]):
        """
        Send a message to a client.
        
        Args:
            client_id: The client ID
            message_type: The message type
            data: The message data
        """
        # Check if client is connected
        if client_id not in self.active_connections:
            logger.warning(f"Cannot send message to client {client_id}: not connected")
            return
        
        try:
            # Create message
            message = {
                'type': message_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send message
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {e}")
    
    async def broadcast_message(self, message_type: str, data: Dict[str, Any], channel: Optional[str] = None):
        """
        Broadcast a message to all clients or clients subscribed to a channel.
        
        Args:
            message_type: The message type
            data: The message data
            channel: The channel to broadcast to (optional)
        """
        # Get clients to broadcast to
        clients = []
        
        if channel:
            # Broadcast to clients subscribed to the channel
            for client_id, channels in self.subscriptions.items():
                if channel in channels:
                    clients.append(client_id)
        else:
            # Broadcast to all clients
            clients = list(self.active_connections.keys())
        
        # Send message to each client
        for client_id in clients:
            await self.send_message(client_id, message_type, data)
    
    async def process_message_queue(self):
        """
        Process messages in the message queue.
        """
        while True:
            try:
                # Get message from queue
                message = await self.message_queue.get()
                
                # Process message
                client_id = message['client_id']
                message_type = message['type']
                data = message['data']
                
                # Handle message based on type
                # This is where you would implement custom message handling
                
                # Mark message as processed
                self.message_queue.task_done()
            except asyncio.CancelledError:
                # Task was cancelled
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def send_periodic_updates(self):
        """
        Send periodic updates to clients.
        """
        while True:
            try:
                # Get current time
                current_time = time.time()
                
                # Check each channel for updates
                for channel, interval in self.update_intervals.items():
                    # Check if it's time to send an update
                    last_update = self.last_update_time.get(channel, 0)
                    if current_time - last_update >= interval:
                        # Send update
                        await self.send_channel_update(channel)
                        
                        # Update last update time
                        self.last_update_time[channel] = current_time
                
                # Sleep for a short time
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # Task was cancelled
                break
            except Exception as e:
                logger.error(f"Error sending periodic updates: {e}")
                
                # Sleep for a short time to avoid tight loop
                await asyncio.sleep(1)
    
    async def send_channel_update(self, channel: str):
        """
        Send an update for a channel.
        
        Args:
            channel: The channel to send an update for
        """
        # Check if there are any subscribers
        has_subscribers = False
        for channels in self.subscriptions.values():
            if channel in channels:
                has_subscribers = True
                break
        
        if not has_subscribers:
            # No subscribers, skip update
            return
        
        # Get data for channel
        data = await self.get_channel_data(channel)
        
        if data:
            # Broadcast update
            await self.broadcast_message(f"{channel}_update", data, channel)
    
    async def get_channel_data(self, channel: str) -> Dict[str, Any]:
        """
        Get data for a channel.
        
        Args:
            channel: The channel to get data for
            
        Returns:
            The channel data
        """
        # Check if there's a data source for this channel
        if channel in self.data_sources:
            try:
                # Get data from data source
                return await self.data_sources[channel].get_data()
            except Exception as e:
                logger.error(f"Error getting data for channel {channel}: {e}")
        
        # Return empty data
        return {}
    
    def register_data_source(self, channel: str, data_source: Any):
        """
        Register a data source for a channel.
        
        Args:
            channel: The channel to register the data source for
            data_source: The data source
        """
        self.data_sources[channel] = data_source
        logger.info(f"Registered data source for channel: {channel}")
    
    def set_update_interval(self, channel: str, interval: float):
        """
        Set the update interval for a channel.
        
        Args:
            channel: The channel to set the update interval for
            interval: The update interval in seconds
        """
        self.update_intervals[channel] = interval
        logger.info(f"Set update interval for channel {channel} to {interval} seconds")