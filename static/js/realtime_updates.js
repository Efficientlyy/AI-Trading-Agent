/**
 * Real-time Updates JavaScript
 * 
 * Handles WebSocket connections for real-time data updates, including:
 * - Establishing and maintaining WebSocket connections
 * - Processing incoming data updates
 * - Updating UI components with real-time data
 * - Handling connection errors and reconnection
 */

// WebSocket Manager
class WebSocketManager {
    constructor(options = {}) {
        // Configuration
        this.options = {
            reconnectInterval: 2000,
            maxReconnectAttempts: 10,
            debug: false,
            ...options
        };

        // State
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
        this.messageHandlers = {};
        this.connectionHandlers = {
            onConnect: [],
            onDisconnect: [],
            onError: []
        };

        // Bind methods
        this.connect = this.connect.bind(this);
        this.disconnect = this.disconnect.bind(this);
        this.reconnect = this.reconnect.bind(this);
        this.handleOpen = this.handleOpen.bind(this);
        this.handleMessage = this.handleMessage.bind(this);
        this.handleError = this.handleError.bind(this);
        this.handleClose = this.handleClose.bind(this);

        // Initialize if autoConnect is true
        if (this.options.autoConnect) {
            this.connect(this.options.url);
        }
    }

    /**
     * Connect to WebSocket server
     * @param {string} url - The WebSocket server URL
     */
    connect(url) {
        if (!url) {
            this.log('Error: WebSocket URL is required');
            return;
        }

        // Store URL for reconnection
        this.options.url = url;

        // Create WebSocket connection
        try {
            this.log(`Connecting to WebSocket server: ${url}`);
            this.socket = new WebSocket(url);

            // Set up event handlers
            this.socket.addEventListener('open', this.handleOpen);
            this.socket.addEventListener('message', this.handleMessage);
            this.socket.addEventListener('error', this.handleError);
            this.socket.addEventListener('close', this.handleClose);
        } catch (error) {
            this.log(`Error creating WebSocket connection: ${error.message}`);
            this.handleError(error);
        }
    }

    /**
     * Disconnect from WebSocket server
     */
    disconnect() {
        if (!this.socket) return;

        this.log('Disconnecting from WebSocket server');

        // Clear reconnect timer
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        // Close connection
        try {
            this.socket.close();
        } catch (error) {
            this.log(`Error closing WebSocket connection: ${error.message}`);
        }

        // Reset state
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
    }

    /**
     * Reconnect to WebSocket server
     */
    reconnect() {
        // Check if max reconnect attempts reached
        if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            this.log('Max reconnect attempts reached, giving up');
            return;
        }

        // Increment reconnect attempts
        this.reconnectAttempts++;

        // Clear reconnect timer
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        // Set reconnect timer
        this.log(`Reconnecting in ${this.options.reconnectInterval}ms (attempt ${this.reconnectAttempts}/${this.options.maxReconnectAttempts})`);
        this.reconnectTimer = setTimeout(() => {
            this.connect(this.options.url);
        }, this.options.reconnectInterval);
    }

    /**
     * Send data to WebSocket server
     * @param {string} type - The message type
     * @param {object} data - The message data
     */
    send(type, data = {}) {
        if (!this.socket || !this.isConnected) {
            this.log('Cannot send message: WebSocket is not connected');
            return false;
        }

        try {
            const message = JSON.stringify({
                type,
                data,
                timestamp: new Date().toISOString()
            });

            this.socket.send(message);
            return true;
        } catch (error) {
            this.log(`Error sending message: ${error.message}`);
            return false;
        }
    }

    /**
     * Register message handler
     * @param {string} type - The message type to handle
     * @param {function} handler - The handler function
     */
    on(type, handler) {
        if (typeof handler !== 'function') {
            this.log('Error: Handler must be a function');
            return;
        }

        // Initialize handlers array for this type if needed
        if (!this.messageHandlers[type]) {
            this.messageHandlers[type] = [];
        }

        // Add handler
        this.messageHandlers[type].push(handler);
    }

    /**
     * Register connection event handler
     * @param {string} event - The connection event (connect, disconnect, error)
     * @param {function} handler - The handler function
     */
    onConnection(event, handler) {
        if (typeof handler !== 'function') {
            this.log('Error: Handler must be a function');
            return;
        }

        // Add handler based on event type
        switch (event) {
            case 'connect':
                this.connectionHandlers.onConnect.push(handler);
                break;
            case 'disconnect':
                this.connectionHandlers.onDisconnect.push(handler);
                break;
            case 'error':
                this.connectionHandlers.onError.push(handler);
                break;
            default:
                this.log(`Error: Unknown connection event: ${event}`);
        }
    }

    /**
     * Handle WebSocket open event
     * @param {Event} event - The open event
     */
    handleOpen(event) {
        this.log('WebSocket connection established');

        // Update state
        this.isConnected = true;
        this.reconnectAttempts = 0;

        // Call connect handlers
        this.connectionHandlers.onConnect.forEach(handler => {
            try {
                handler(event);
            } catch (error) {
                this.log(`Error in connect handler: ${error.message}`);
            }
        });
    }

    /**
     * Handle WebSocket message event
     * @param {MessageEvent} event - The message event
     */
    handleMessage(event) {
        let message;

        // Parse message
        try {
            message = JSON.parse(event.data);
        } catch (error) {
            this.log(`Error parsing message: ${error.message}`);
            return;
        }

        // Log message if debug is enabled
        if (this.options.debug) {
            this.log(`Received message: ${JSON.stringify(message)}`);
        }

        // Check if message has type
        if (!message.type) {
            this.log('Received message without type');
            return;
        }

        // Call handlers for this message type
        const handlers = this.messageHandlers[message.type] || [];
        handlers.forEach(handler => {
            try {
                handler(message.data, message);
            } catch (error) {
                this.log(`Error in message handler for type ${message.type}: ${error.message}`);
            }
        });

        // Call handlers for 'all' message type
        const allHandlers = this.messageHandlers['all'] || [];
        allHandlers.forEach(handler => {
            try {
                handler(message.data, message);
            } catch (error) {
                this.log(`Error in 'all' message handler: ${error.message}`);
            }
        });
    }

    /**
     * Handle WebSocket error event
     * @param {Event} event - The error event
     */
    handleError(event) {
        this.log('WebSocket error occurred');

        // Call error handlers
        this.connectionHandlers.onError.forEach(handler => {
            try {
                handler(event);
            } catch (error) {
                this.log(`Error in error handler: ${error.message}`);
            }
        });
    }

    /**
     * Handle WebSocket close event
     * @param {CloseEvent} event - The close event
     */
    handleClose(event) {
        this.log(`WebSocket connection closed: ${event.code} ${event.reason}`);

        // Update state
        this.isConnected = false;

        // Call disconnect handlers
        this.connectionHandlers.onDisconnect.forEach(handler => {
            try {
                handler(event);
            } catch (error) {
                this.log(`Error in disconnect handler: ${error.message}`);
            }
        });

        // Reconnect if not closed cleanly
        if (event.code !== 1000) {
            this.reconnect();
        }
    }

    /**
     * Log message if debug is enabled
     * @param {string} message - The message to log
     */
    log(message) {
        if (this.options.debug) {
            console.log(`[WebSocketManager] ${message}`);
        }
    }
}

// Real-time Updates Controller
class RealTimeUpdatesController {
    constructor() {
        // WebSocket manager
        this.wsManager = null;

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
    }

    /**
     * Initialize real-time updates
     */
    init() {
        // Create WebSocket manager
        this.wsManager = new WebSocketManager({
            reconnectInterval: 3000,
            maxReconnectAttempts: 5,
            debug: true
        });

        // Register connection event handlers
        this.wsManager.onConnection('connect', this.handleConnect.bind(this));
        this.wsManager.onConnection('disconnect', this.handleDisconnect.bind(this));
        this.wsManager.onConnection('error', this.handleError.bind(this));

        // Register message handlers
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
    }

    /**
     * Connect to WebSocket server
     */
    connectToWebSocket() {
        // Get WebSocket URL from data attribute or use default
        const wsUrl = document.body.dataset.wsUrl || `ws://${window.location.host}/ws`;

        // Connect to WebSocket server
        this.wsManager.connect(wsUrl);
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
        this.wsManager.send('subscribe', {
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

// Initialize real-time updates when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global real-time updates instance
    window.realTimeUpdates = new RealTimeUpdatesController();
});