# Real-time Data Updates

This document provides an overview of the real-time data updates system implemented in Phase 3 of the Real Data Integration project.

## Overview

The real-time updates system enables live data streaming to the dashboard UI through WebSocket connections, eliminating the need for manual page refreshes and providing immediate visibility into market changes and system events.

## Architecture

The real-time updates system consists of the following components:

1. **WebSocket Server**: Implemented in the backend using FastAPI's WebSocket support, this server manages client connections and broadcasts data updates to connected clients.

2. **WebSocket Manager**: A utility class that handles WebSocket connections, message processing, and data broadcasting.

3. **Data Sources**: Classes that provide data for different channels (dashboard, trades, positions, performance, alerts).

4. **WebSocket Client**: JavaScript code that establishes and maintains the WebSocket connection, processes incoming messages, and updates the UI accordingly.

## Features

- **Persistent Connections**: Maintains a persistent connection between the client and server for real-time data streaming.
- **Automatic Reconnection**: Automatically reconnects if the connection is lost.
- **Channel-based Subscriptions**: Allows clients to subscribe to specific data channels.
- **Efficient Data Transfer**: Sends only the data that has changed, minimizing bandwidth usage.
- **Visual Feedback**: Provides visual feedback when data is updated.
- **Connection Status Indicator**: Shows the current connection status (connected, disconnected, error).

## Data Channels

The system supports the following data channels:

1. **Dashboard**: Updates to dashboard summary metrics.
2. **Trades**: Real-time trade notifications.
3. **Positions**: Updates to current positions.
4. **Performance**: Updates to performance metrics.
5. **Alerts**: System alerts and notifications.

## Implementation Details

### Backend

The backend implementation consists of:

1. **WebSocket Endpoint**: A FastAPI WebSocket endpoint that handles client connections and message routing.

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"client_{id(websocket)}"
    
    try:
        # Connect client
        await ws_manager.connect(websocket, client_id)
        
        # Handle messages
        while True:
            data = await websocket.receive_text()
            await ws_manager.receive_message(client_id, data)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Disconnect client
        await ws_manager.disconnect(client_id)
```

2. **WebSocket Manager**: A class that manages WebSocket connections and message broadcasting.

```python
class WebSocketManager:
    def __init__(self):
        self.active_connections = {}
        self.subscriptions = {}
        
    async def connect(self, websocket, client_id):
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        
    async def disconnect(self, client_id):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
            
    async def broadcast_message(self, message_type, data, channel=None):
        # Broadcast message to all clients or clients subscribed to a channel
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
```

3. **Data Sources**: Classes that provide data for different channels.

```python
class DashboardDataSource:
    async def get_data(self):
        # Get dashboard data
        return {
            'total_value': '$50,000.00',
            'daily_pnl': '$1,200.00',
            'open_positions': 5,
            'win_rate': '58.3%'
        }
```

### Frontend

The frontend implementation consists of:

1. **WebSocket Client**: A JavaScript class that establishes and maintains the WebSocket connection.

```javascript
class WebSocketManager {
    constructor(url) {
        this.url = url;
        this.socket = null;
        this.isConnected = false;
        this.messageHandlers = {};
        
        // Connect to WebSocket server
        this.connect();
    }
    
    connect() {
        this.socket = new WebSocket(this.url);
        
        this.socket.onopen = () => {
            this.isConnected = true;
            console.log('Connected to WebSocket server');
        };
        
        this.socket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };
        
        this.socket.onclose = () => {
            this.isConnected = false;
            console.log('Disconnected from WebSocket server');
            
            // Reconnect after a delay
            setTimeout(() => this.connect(), 3000);
        };
    }
    
    handleMessage(message) {
        // Handle message based on type
        const handlers = this.messageHandlers[message.type] || [];
        handlers.forEach(handler => handler(message.data));
    }
    
    on(type, handler) {
        // Register message handler
        if (!this.messageHandlers[type]) {
            this.messageHandlers[type] = [];
        }
        this.messageHandlers[type].push(handler);
    }
    
    send(type, data) {
        // Send message to server
        if (this.isConnected) {
            const message = {
                type,
                data,
                timestamp: new Date().toISOString()
            };
            this.socket.send(JSON.stringify(message));
        }
    }
}
```

2. **UI Updates**: JavaScript code that updates the UI based on received messages.

```javascript
// Handle dashboard update
wsManager.on('dashboard_update', (data) => {
    // Update dashboard metrics
    document.getElementById('total-value').textContent = data.total_value;
    document.getElementById('daily-pnl').textContent = data.daily_pnl;
    document.getElementById('open-positions').textContent = data.open_positions;
    document.getElementById('win-rate').textContent = data.win_rate;
});
```

## Usage

To use the real-time updates system:

1. **Backend Setup**:
   - Initialize the WebSocket manager
   - Register data sources
   - Set up the WebSocket endpoint

2. **Frontend Setup**:
   - Initialize the WebSocket client
   - Register message handlers
   - Subscribe to channels

## Configuration

The real-time updates system can be configured through the following settings:

- **Update Intervals**: The frequency at which data is sent for each channel.
- **Reconnection Settings**: The delay and maximum attempts for reconnection.
- **Channel Subscriptions**: Which channels to subscribe to.

## Performance Considerations

- **Message Size**: Keep messages small to minimize bandwidth usage.
- **Update Frequency**: Balance between real-time updates and server load.
- **Connection Management**: Properly handle connection errors and disconnections.
- **Browser Compatibility**: Ensure compatibility with all major browsers.

## Security Considerations

- **Authentication**: Ensure that only authenticated users can connect to the WebSocket server.
- **Authorization**: Verify that users have permission to access the requested data.
- **Data Validation**: Validate all incoming messages to prevent injection attacks.
- **Rate Limiting**: Implement rate limiting to prevent abuse.

## Future Enhancements

- **Bidirectional Communication**: Enable clients to send commands to the server.
- **Message Compression**: Implement message compression for larger data sets.
- **Selective Updates**: Send only the data that has changed.
- **Offline Support**: Cache data for offline use.
- **Mobile Optimization**: Optimize for mobile devices with limited bandwidth.