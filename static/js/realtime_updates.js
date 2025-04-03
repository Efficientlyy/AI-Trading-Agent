/**
 * Real-time Updates JavaScript
 * 
 * Handles WebSocket connections for real-time data updates, including:
 * - Establishing and maintaining WebSocket connections
 * - Processing incoming data updates
 * - Updating UI components with real-time data
 * - Handling connection errors and reconnection
 */

// Check if WebSocketManager is already defined
if (typeof window.WebSocketManager === 'undefined') {
    // WebSocket manager singleton
    window.WebSocketManager = (function() {
        let instance;

        function createInstance(options) {
            try {
                // Default options
                options = options || {};
                const reconnectInterval = options.reconnectInterval || 3000;
                const maxReconnectAttempts = options.maxReconnectAttempts || 10;
                const debug = options.debug || false;

                // Create WebSocket manager instance
                const wsManager = {
                    socket: null,
                    isConnected: false,
                    reconnectAttempts: 0,
                    reconnectTimer: null,
                    connectionCallbacks: {},
                    messageCallbacks: {},
                    debug: debug,

                    // Log message if debug is enabled
                    log: function(message) {
                        if (this.debug) {
                            console.log(`[WebSocketManager] ${message}`);
                        }
                    },

                    // Connect to WebSocket server
                    connect: function(url) {
                        try {
                            this.log(`Connecting to WebSocket server: ${url}`);
                            this.url = url;
                            
                            // Create WebSocket connection
                            try {
                                this.socket = new WebSocket(url);
                            } catch (e) {
                                console.error('Error creating WebSocket:', e);
                                this.handleReconnect();
                                return;
                            }
                            
                            // Set event handlers
                            if (this.socket) {
                                this.socket.onopen = this.handleOpen.bind(this);
                                this.socket.onclose = this.handleClose.bind(this);
                                this.socket.onerror = this.handleError.bind(this);
                                this.socket.onmessage = this.handleMessage.bind(this);
                            } else {
                                console.error('WebSocket initialization failed');
                                this.handleReconnect();
                            }
                        } catch (error) {
                            console.error('Error connecting to WebSocket server:', error);
                            this.handleReconnect();
                        }
                    },

                    // Handle WebSocket open event
                    handleOpen: function() {
                        try {
                            this.log('WebSocket connection established');
                            this.isConnected = true;
                            this.reconnectAttempts = 0;
                            
                            // Call connection callbacks
                            this.fireEvent('connect');
                        } catch (error) {
                            console.error('Error handling WebSocket open event:', error);
                        }
                    },

                    // Handle WebSocket close event
                    handleClose: function(event) {
                        try {
                            this.log(`WebSocket connection closed: [${event.code}] ${event.reason}`);
                            this.isConnected = false;
                            
                            // Call connection callbacks
                            this.fireEvent('disconnect');
                            
                            // Attempt to reconnect
                            this.handleReconnect();
                        } catch (error) {
                            console.error('Error handling WebSocket close event:', error);
                        }
                    },

                    // Handle WebSocket error event
                    handleError: function(error) {
                        try {
                            this.log('WebSocket error: ' + (error ? error.message : 'Unknown error'));
                            
                            // Call connection callbacks
                            this.fireEvent('error', error);
                        } catch (error) {
                            console.error('Error handling WebSocket error event:', error);
                        }
                    },

                    // Handle WebSocket message event
                    handleMessage: function(event) {
                        try {
                            this.log(`WebSocket message received: ${event.data}`);
                            
                            // Parse message
                            let message;
                            try {
                                message = JSON.parse(event.data);
                            } catch (error) {
                                this.log('Error parsing message:', error);
                                return;
                            }
                            
                            // Check if message has type
                            if (!message || !message.type) {
                                this.log('Invalid message received');
                                return;
                            }
                            
                            // Fire event
                            this.fireEvent(message.type, message.data);
                        } catch (error) {
                            console.error('Error handling WebSocket message:', error);
                        }
                    },

                    // Handle reconnection
                    handleReconnect: function() {
                        try {
                            if (this.reconnectTimer) {
                                clearTimeout(this.reconnectTimer);
                                this.reconnectTimer = null;
                            }
                            
                            if (this.reconnectAttempts < maxReconnectAttempts) {
                                this.reconnectAttempts++;
                                this.log(`Reconnecting (${this.reconnectAttempts}/${maxReconnectAttempts}) in ${reconnectInterval}ms...`);
                                
                                this.reconnectTimer = setTimeout(() => {
                                    if (this.url) {
                                        this.connect(this.url);
                                    }
                                }, reconnectInterval);
                            } else {
                                this.log('Max reconnect attempts reached');
                                this.fireEvent('reconnect_failed');
                            }
                        } catch (error) {
                            console.error('Error handling reconnection:', error);
                        }
                    },

                    // Register connection event callback
                    onConnection: function(event, callback) {
                        try {
                            if (!event || typeof callback !== 'function') {
                                return false;
                            }
                            
                            if (!this.connectionCallbacks[event]) {
                                this.connectionCallbacks[event] = [];
                            }
                            
                            this.connectionCallbacks[event].push(callback);
                            return true;
                        } catch (error) {
                            console.error('Error registering connection event callback:', error);
                            return false;
                        }
                    },

                    // Register message event callback
                    on: function(event, callback) {
                        try {
                            if (!event || typeof callback !== 'function') {
                                return false;
                            }
                            
                            if (!this.messageCallbacks[event]) {
                                this.messageCallbacks[event] = [];
                            }
                            
                            this.messageCallbacks[event].push(callback);
                            return true;
                        } catch (error) {
                            console.error('Error registering message event callback:', error);
                            return false;
                        }
                    },

                    // Fire event
                    fireEvent: function(event, data) {
                        try {
                            // Check if event is a connection event
                            if (this.connectionCallbacks[event]) {
                                this.connectionCallbacks[event].forEach(callback => {
                                    try {
                                        callback(data);
                                    } catch (error) {
                                        console.error(`Error in connection callback for event ${event}:`, error);
                                    }
                                });
                            }
                            
                            // Check if event is a message event
                            if (this.messageCallbacks[event]) {
                                this.messageCallbacks[event].forEach(callback => {
                                    try {
                                        callback(data);
                                    } catch (error) {
                                        console.error(`Error in message callback for event ${event}:`, error);
                                    }
                                });
                            }
                        } catch (error) {
                            console.error('Error firing event:', error);
                        }
                    },

                    // Send message to server
                    send: function(type, data) {
                        try {
                            if (!this.isConnected || !this.socket) {
                                this.log('Cannot send message: Not connected');
                                return false;
                            }
                            
                            // Create message
                            const message = JSON.stringify({
                                type: type,
                                data: data
                            });
                            
                            // Send message
                            this.socket.send(message);
                            this.log(`Message sent: ${message}`);
                            return true;
                        } catch (error) {
                            console.error('Error sending message:', error);
                            return false;
                        }
                    },

                    // Close WebSocket connection
                    close: function() {
                        try {
                            if (this.socket) {
                                this.socket.close();
                            }
                            
                            if (this.reconnectTimer) {
                                clearTimeout(this.reconnectTimer);
                                this.reconnectTimer = null;
                            }
                            
                            this.isConnected = false;
                            this.log('WebSocket connection closed');
                        } catch (error) {
                            console.error('Error closing WebSocket connection:', error);
                        }
                    }
                };
                
                return wsManager;
            } catch (error) {
                console.error('Error creating WebSocketManager instance:', error);
                // Return a minimal object with empty methods to prevent null reference errors
                return {
                    connect: function() { console.error('WebSocketManager failed to initialize properly'); },
                    onConnection: function() { return false; },
                    on: function() { return false; },
                    send: function() { return false; },
                    close: function() {}
                };
            }
        }

        return {
            getInstance: function(options) {
                try {
                    if (!instance) {
                        instance = createInstance(options);
                    }
                    return instance;
                } catch (error) {
                    console.error('Error getting WebSocketManager instance:', error);
                    return {
                        connect: function() { console.error('WebSocketManager failed to initialize properly'); },
                        onConnection: function() { return false; },
                        on: function() { return false; },
                        send: function() { return false; },
                        close: function() {}
                    };
                }
            }
        };
    })();
}

// Real-time Updates Controller
if (typeof window.RealTimeUpdatesController === 'undefined') {
    window.RealTimeUpdatesController = class RealTimeUpdatesController {
        constructor() {
            try {
                // WebSocket manager
                this.wsManager = window.WebSocketManager.getInstance({
                    reconnectInterval: 3000,
                    maxReconnectAttempts: 5,
                    debug: true
                });

                // UI components that can be updated in real-time
                this.components = {
                    dashboard: document.getElementById('dashboard-summary'),
                    trades: document.getElementById('recent-trades'),
                    positions: document.getElementById('current-positions'),
                    performance: document.getElementById('performance-metrics'),
                    alerts: document.getElementById('system-alerts')
                };

                // Connection status indicator
                this.statusIndicator = document.getElementById('connection-status');

                // Initialize
                this.init();
            } catch (error) {
                console.error('Error initializing RealTimeUpdatesController:', error);
            }
        }

        /**
         * Initialize real-time updates
         */
        init() {
            try {
                // Check if wsManager is defined
                if (!this.wsManager) {
                    console.error('WebSocket manager is not initialized');
                    return;
                }

                // Check if required methods exist
                if (typeof this.wsManager.onConnection !== 'function') {
                    console.error('WebSocket manager onConnection method is not available');
                    return;
                }

                if (typeof this.wsManager.on !== 'function') {
                    console.error('WebSocket manager on method is not available');
                    return;
                }

                // Register connection event handlers safely
                this.wsManager.onConnection('connect', this.handleConnect.bind(this));
                this.wsManager.onConnection('disconnect', this.handleDisconnect.bind(this));
                this.wsManager.onConnection('error', this.handleError.bind(this));

                // Register message handlers safely
                this.wsManager.on('dashboard_update', this.handleDashboardUpdate.bind(this));
                this.wsManager.on('trade_update', this.handleTradeUpdate.bind(this));
                this.wsManager.on('position_update', this.handlePositionUpdate.bind(this));
                this.wsManager.on('performance_update', this.handlePerformanceUpdate.bind(this));
                this.wsManager.on('alert', this.handleAlert.bind(this));

                // Connect to WebSocket server
                this.connectToWebSocket();

                // Add event listener for visibility change to reconnect when tab becomes visible
                document.addEventListener('visibilitychange', () => {
                    if (document.visibilityState === 'visible' && this.wsManager && !this.wsManager.isConnected) {
                        this.connectToWebSocket();
                    }
                });
            } catch (error) {
                console.error('Error initializing real-time updates:', error);
            }
        }

        /**
         * Connect to WebSocket server
         */
        connectToWebSocket() {
            try {
                console.log("Attempting to connect to real-time updates server...");
                
                // Try Socket.IO first (preferred)
                if (typeof io !== 'undefined') {
                    this.connectWithSocketIO();
                } 
                // Fall back to native WebSocket
                else {
                    this.connectWithNativeWebSocket();
                }
            } catch (error) {
                console.error('Error connecting to real-time updates server:', error);
            }
        }
        
        /**
         * Connect using Socket.IO
         */
        connectWithSocketIO() {
            console.log("Using Socket.IO for real-time updates");
            
            // Get Socket.IO URL from data attribute or use current host
            const socketUrl = document.body.dataset.socketUrl || window.location.origin;
            
            console.log(`Connecting to Socket.IO server at: ${socketUrl}`);
            
            // Connect to Socket.IO server
            const socket = io(socketUrl, {
                reconnection: true,
                reconnectionAttempts: 5,
                reconnectionDelay: 1000,
                reconnectionDelayMax: 5000,
                timeout: 20000,
                // Important: Set transports to use websocket first, then polling
                transports: ['websocket', 'polling']
            });
            
            // Store connection type
            this.connectionType = 'socketio';
            
            // Store the socket
            this.socket = socket;
            
            // Setup event handlers
            socket.on('connect', () => {
                console.log('Connected to Socket.IO server');
                this.handleConnect();
                
                // Subscribe to channels
                socket.emit('subscribe', {
                    channels: ['dashboard', 'trades', 'positions', 'performance', 'alerts']
                });
            });
            
            socket.on('disconnect', () => {
                console.log('Disconnected from Socket.IO server');
                this.handleDisconnect();
            });
            
            socket.on('connect_error', (error) => {
                console.error('Socket.IO connection error:', error);
                this.handleError(error);
            });
            
            // Custom events
            socket.on('dashboard_update', (data) => this.handleDashboardUpdate(data));
            socket.on('trade_update', (data) => this.handleTradeUpdate(data));
            socket.on('position_update', (data) => this.handlePositionUpdate(data));
            socket.on('performance_update', (data) => this.handlePerformanceUpdate(data));
            socket.on('alert', (data) => this.handleAlert(data));
        }
        
        /**
         * Connect using native WebSocket
         */
        connectWithNativeWebSocket() {
            console.log("Using native WebSocket for real-time updates");
            
            // Get WebSocket URL from data attribute or use default
            // Use both secure/non-secure options based on current protocol
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = document.body.dataset.wsUrl || 
                          `${protocol}//${window.location.host}/ws`;
            
            console.log(`Connecting to WebSocket server at: ${wsUrl}`);
            
            // Create WebSocket
            const ws = new WebSocket(wsUrl);
            
            // Store connection type
            this.connectionType = 'websocket';
            
            // Store the WebSocket
            this.socket = ws;
            
            // Setup event handlers
            ws.addEventListener('open', () => {
                console.log('Connected to WebSocket server');
                this.handleConnect();
                
                // Subscribe to channels
                this.sendMessage('subscribe', {
                    channels: ['dashboard', 'trades', 'positions', 'performance', 'alerts']
                });
            });
            
            ws.addEventListener('close', () => {
                console.log('Disconnected from WebSocket server');
                this.handleDisconnect();
                
                // Try to reconnect after a delay
                setTimeout(() => {
                    this.connectToWebSocket();
                }, 3000);
            });
            
            ws.addEventListener('error', (error) => {
                console.error('WebSocket error:', error);
                this.handleError(error);
            });
            
            ws.addEventListener('message', (event) => {
                try {
                    const message = JSON.parse(event.data);
                    const type = message.type;
                    const data = message.data;
                    
                    // Handle message based on type
                    switch(type) {
                        case 'dashboard_update':
                            this.handleDashboardUpdate(data);
                            break;
                        case 'trade_update':
                            this.handleTradeUpdate(data);
                            break;
                        case 'position_update':
                            this.handlePositionUpdate(data);
                            break;
                        case 'performance_update':
                            this.handlePerformanceUpdate(data);
                            break;
                        case 'alert':
                            this.handleAlert(data);
                            break;
                        default:
                            console.log(`Received message of unknown type: ${type}`, data);
                    }
                } catch (error) {
                    console.error('Error handling WebSocket message:', error);
                }
            });
        }
        
        /**
         * Send message to the server
         */
        sendMessage(event, data) {
            try {
                if (!this.socket) {
                    console.warn('Cannot send message: Not connected to server');
                    return false;
                }
                
                if (this.connectionType === 'socketio') {
                    // Socket.IO message
                    this.socket.emit(event, data);
                } else {
                    // Native WebSocket message
                    const message = JSON.stringify({
                        type: event,
                        data: data
                    });
                    this.socket.send(message);
                }
                
                return true;
            } catch (error) {
                console.error('Error sending message:', error);
                return false;
            }
        }

        /**
         * Handle WebSocket connect event
         */
        handleConnect() {
            console.log('Connected to real-time updates server');

            // Update connection status indicator
            if (this.statusIndicator) {
                this.statusIndicator.className = 'connection-status connected';
                this.statusIndicator.title = 'Connected to real-time updates server';
            }

            // Subscribe to updates
            this.sendMessage('subscribe', {
                channels: ['dashboard', 'trades', 'positions', 'performance', 'alerts']
            });
        }

        /**
         * Handle WebSocket disconnect event
         */
        handleDisconnect() {
            console.log('Disconnected from real-time updates server');

            // Update connection status indicator
            if (this.statusIndicator) {
                this.statusIndicator.className = 'connection-status disconnected';
                this.statusIndicator.title = 'Disconnected from real-time updates server';
            }
        }

        /**
         * Handle WebSocket error event
         */
        handleError() {
            console.log('Error connecting to real-time updates server');

            // Update connection status indicator
            if (this.statusIndicator) {
                this.statusIndicator.className = 'connection-status error';
                this.statusIndicator.title = 'Error connecting to real-time updates server';
            }
        }

        /**
         * Handle dashboard update message
         * @param {object} data - The dashboard update data
         */
        handleDashboardUpdate(data) {
            console.log('Received dashboard update:', data);

            // Update dashboard summary
            if (this.components.dashboard && data) {
                // Update dashboard metrics
                const metrics = this.components.dashboard.querySelectorAll('.metric-value');
                metrics.forEach(metric => {
                    const metricId = metric.dataset.metricId;
                    if (metricId && data[metricId] !== undefined) {
                        metric.textContent = data[metricId];

                        // Add animation class to highlight the update
                        metric.classList.add('updated');
                        setTimeout(() => {
                            metric.classList.remove('updated');
                        }, 1000);
                    }
                });
            }
        }

        /**
         * Handle trade update message
         * @param {object} data - The trade update data
         */
        handleTradeUpdate(data) {
            console.log('Received trade update:', data);

            // Update recent trades
            if (this.components.trades && data) {
                // Check if it's a new trade or update to existing trade
                const tradeRow = this.components.trades.querySelector(`tr[data-trade-id="${data.id}"]`);

                if (tradeRow) {
                    // Update existing trade
                    this.updateTradeRow(tradeRow, data);
                } else {
                    // Add new trade
                    this.addTradeRow(data);
                }
            }
        }

        /**
         * Handle position update message
         * @param {object} data - The position update data
         */
        handlePositionUpdate(data) {
            console.log('Received position update:', data);

            // Update current positions
            if (this.components.positions && data) {
                // Check if it's a new position or update to existing position
                const positionRow = this.components.positions.querySelector(`tr[data-position-id="${data.id}"]`);

                if (positionRow) {
                    // Update existing position
                    this.updatePositionRow(positionRow, data);
                } else {
                    // Add new position
                    this.addPositionRow(data);
                }
            }
        }

        /**
         * Handle performance update message
         * @param {object} data - The performance update data
         */
        handlePerformanceUpdate(data) {
            console.log('Received performance update:', data);

            // Update performance metrics
            if (this.components.performance && data) {
                // Update performance metrics
                const metrics = this.components.performance.querySelectorAll('.metric-value');
                metrics.forEach(metric => {
                    const metricId = metric.dataset.metricId;
                    if (metricId && data[metricId] !== undefined) {
                        metric.textContent = data[metricId];

                        // Add animation class to highlight the update
                        metric.classList.add('updated');
                        setTimeout(() => {
                            metric.classList.remove('updated');
                        }, 1000);
                    }
                });
            }
        }

        /**
         * Handle alert message
         * @param {object} data - The alert data
         */
        handleAlert(data) {
            console.log('Received alert:', data);

            // Show alert notification
            if (window.showToast) {
                window.showToast(data.message, data.type || 'info');
            }

            // Add alert to alerts list
            if (this.components.alerts && data) {
                this.addAlertItem(data);
            }
        }

        /**
         * Update trade row with new data
         * @param {HTMLElement} row - The trade row element
         * @param {object} data - The trade data
         */
        updateTradeRow(row, data) {
            // Update trade data
            const cells = row.querySelectorAll('td');

            // Update each cell based on data
            cells.forEach(cell => {
                const field = cell.dataset.field;
                if (field && data[field] !== undefined) {
                    cell.textContent = data[field];

                    // Add animation class to highlight the update
                    cell.classList.add('updated');
                    setTimeout(() => {
                        cell.classList.remove('updated');
                    }, 1000);
                }
            });
        }

        /**
         * Add new trade row
         * @param {object} data - The trade data
         */
        addTradeRow(data) {
            // Get trades table body
            const tableBody = this.components.trades.querySelector('tbody');
            if (!tableBody) return;

            // Create new row
            const row = document.createElement('tr');
            row.dataset.tradeId = data.id;

            // Add cells
            row.innerHTML = `
                <td data-field="time">${data.time}</td>
                <td data-field="symbol">${data.symbol}</td>
                <td data-field="side" class="${data.side.toLowerCase()}">${data.side}</td>
                <td data-field="price">${data.price}</td>
                <td data-field="quantity">${data.quantity}</td>
                <td data-field="value">${data.value}</td>
            `;

            // Add to table
            tableBody.insertBefore(row, tableBody.firstChild);

            // Add animation class to highlight the new row
            row.classList.add('new-row');
            setTimeout(() => {
                row.classList.remove('new-row');
            }, 1000);

            // Remove last row if table is too long
            const rows = tableBody.querySelectorAll('tr');
            if (rows.length > 10) {
                tableBody.removeChild(rows[rows.length - 1]);
            }
        }

        /**
         * Update position row with new data
         * @param {HTMLElement} row - The position row element
         * @param {object} data - The position data
         */
        updatePositionRow(row, data) {
            // Update position data
            const cells = row.querySelectorAll('td');

            // Update each cell based on data
            cells.forEach(cell => {
                const field = cell.dataset.field;
                if (field && data[field] !== undefined) {
                    cell.textContent = data[field];

                    // Add animation class to highlight the update
                    cell.classList.add('updated');
                    setTimeout(() => {
                        cell.classList.remove('updated');
                    }, 1000);
                }
            });

            // Update PnL class
            const pnlCell = row.querySelector('td[data-field="pnl"]');
            if (pnlCell && data.pnl !== undefined) {
                pnlCell.className = data.pnl >= 0 ? 'positive' : 'negative';
            }
        }

        /**
         * Add new position row
         * @param {object} data - The position data
         */
        addPositionRow(data) {
            // Get positions table body
            const tableBody = this.components.positions.querySelector('tbody');
            if (!tableBody) return;

            // Create new row
            const row = document.createElement('tr');
            row.dataset.positionId = data.id;

            // Add cells
            row.innerHTML = `
                <td data-field="symbol">${data.symbol}</td>
                <td data-field="side" class="${data.side.toLowerCase()}">${data.side}</td>
                <td data-field="quantity">${data.quantity}</td>
                <td data-field="entry_price">${data.entry_price}</td>
                <td data-field="current_price">${data.current_price}</td>
                <td data-field="pnl" class="${data.pnl >= 0 ? 'positive' : 'negative'}">${data.pnl}</td>
            `;

            // Add to table
            tableBody.appendChild(row);

            // Add animation class to highlight the new row
            row.classList.add('new-row');
            setTimeout(() => {
                row.classList.remove('new-row');
            }, 1000);
        }

        /**
         * Add new alert item
         * @param {object} data - The alert data
         */
        addAlertItem(data) {
            // Get alerts list
            const alertsList = this.components.alerts.querySelector('.alerts-list');
            if (!alertsList) return;

            // Create new alert item
            const alertItem = document.createElement('div');
            alertItem.className = `alert-item ${data.type || 'info'}`;

            // Add alert content
            alertItem.innerHTML = `
                <div class="alert-time">${data.time || new Date().toLocaleTimeString()}</div>
                <div class="alert-message">${data.message}</div>
            `;

            // Add to alerts list
            alertsList.insertBefore(alertItem, alertsList.firstChild);

            // Add animation class to highlight the new alert
            alertItem.classList.add('new-alert');
            setTimeout(() => {
                alertItem.classList.remove('new-alert');
            }, 1000);

            // Remove last alert if list is too long
            const alerts = alertsList.querySelectorAll('.alert-item');
            if (alerts.length > 10) {
                alertsList.removeChild(alerts[alerts.length - 1]);
            }
        }
    }
}

// Initialize real-time updates when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global real-time updates instance
    window.realTimeUpdates = new RealTimeUpdatesController();
});