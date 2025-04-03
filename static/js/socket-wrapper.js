/**
 * Safe Socket.IO wrapper for the AI Trading Agent Dashboard
 * 
 * This wrapper provides a safe interface to Socket.IO, with proper error handling
 * and DOM-safe operations
 */

// Global socket handler with safe operation methods
const SocketHandler = {
    // The Socket.IO instance
    socket: null,

    // Connected flag
    connected: false,

    // Initialize the Socket.IO connection
    init: function(url) {
        try {
            // Create Socket.IO instance
            this.socket = io(url, {
                reconnection: true,
                reconnectionAttempts: 5,
                reconnectionDelay: 1000,
                reconnectionDelayMax: 5000,
                timeout: 10000
            });

            // Set up event handlers
            this.setupEventHandlers();

            return true;
        } catch (error) {
            console.error('Error initializing Socket.IO:', error);
            return false;
        }
    },

    // Set up Socket.IO event handlers
    setupEventHandlers: function() {
        if (!this.socket) return;

        // Connection events
        this.socket.on('connect', () => {
            console.log('Socket.IO connected');
            this.connected = true;
            this.updateConnectionStatus('connected');
        });

        this.socket.on('disconnect', () => {
            console.log('Socket.IO disconnected');
            this.connected = false;
            this.updateConnectionStatus('disconnected');
        });

        this.socket.on('connect_error', (error) => {
            console.error('Socket.IO connection error:', error);
            this.connected = false;
            this.updateConnectionStatus('error');
        });

        // Custom events
        this.socket.on('notification', (data) => {
            this.handleNotification(data);
        });

        this.socket.on('trade', (data) => {
            this.handleTrade(data);
        });

        this.socket.on('position', (data) => {
            this.handlePosition(data);
        });

        this.socket.on('market_data', (data) => {
            this.handleMarketData(data);
        });

        this.socket.on('system_alert', (data) => {
            this.handleSystemAlert(data);
        });
    },

    // Update connection status
    updateConnectionStatus: function(status) {
        try {
            const statusIndicator = document.getElementById('connection-status');
            if (!statusIndicator) return;

            // Remove all status classes safely
            const statusClasses = ['connected', 'disconnected', 'error'];
            statusClasses.forEach(cls => {
                if (statusIndicator.classList) {
                    statusIndicator.classList.remove(cls);
                }
            });

            // Add the new status class safely
            if (statusIndicator.classList) {
                statusIndicator.classList.add(status);
            }

            // Update title
            switch (status) {
                case 'connected':
                    statusIndicator.title = 'Connected to server';
                    break;
                case 'disconnected':
                    statusIndicator.title = 'Disconnected from server';
                    break;
                case 'error':
                    statusIndicator.title = 'Connection error';
                    break;
            }
        } catch (error) {
            console.error('Error updating connection status:', error);
        }
    },

    // Handle notification event
    handleNotification: function(data) {
        try {
            console.log('Received notification:', data);

            // Use NotificationHandler if available
            if (window.NotificationHandler && typeof window.NotificationHandler.addNotification === 'function') {
                window.NotificationHandler.addNotification(data);
            }
            // Fallback to showNotification if available
            else if (window.NotificationHandler && typeof window.NotificationHandler.showNotification === 'function') {
                window.NotificationHandler.showNotification(data.message, data.type || 'info');
            }
            // Last resort, use alert
            else {
                console.warn('Notification received but no handler available:', data);
            }
        } catch (error) {
            console.error('Error handling notification:', error);
        }
    },

    // Handle trade event
    handleTrade: function(data) {
        try {
            console.log('Received trade data:', data);
            // Add your trade handling logic here
            // Use DOM safe operations, e.g., DOMHandler.getElementById(), etc.
        } catch (error) {
            console.error('Error handling trade data:', error);
        }
    },

    // Handle position event
    handlePosition: function(data) {
        try {
            console.log('Received position data:', data);
            // Add your position handling logic here
            // Use DOM safe operations, e.g., DOMHandler.getElementById(), etc.
        } catch (error) {
            console.error('Error handling position data:', error);
        }
    },

    // Handle market data event
    handleMarketData: function(data) {
        try {
            console.log('Received market data:', data);
            // Add your market data handling logic here
            // Use DOM safe operations, e.g., DOMHandler.getElementById(), etc.
        } catch (error) {
            console.error('Error handling market data:', error);
        }
    },

    // Handle system alert event
    handleSystemAlert: function(data) {
        try {
            console.log('Received system alert:', data);
            
            // Use NotificationHandler if available
            if (window.NotificationHandler && typeof window.NotificationHandler.showNotification === 'function') {
                window.NotificationHandler.showNotification(data.message, 'alert', 10000);
            } else {
                console.warn('System alert received but no handler available:', data);
            }
        } catch (error) {
            console.error('Error handling system alert:', error);
        }
    },

    // Send a message to the server
    send: function(event, data) {
        try {
            if (!this.socket || !this.connected) {
                console.warn('Cannot send message: Socket is not connected');
                return false;
            }

            this.socket.emit(event, data);
            return true;
        } catch (error) {
            console.error('Error sending message:', error);
            return false;
        }
    },

    // Subscribe to a channel
    subscribe: function(channel) {
        return this.send('subscribe', { channel });
    },

    // Unsubscribe from a channel
    unsubscribe: function(channel) {
        return this.send('unsubscribe', { channel });
    }
};

// Make socket handler globally available
window.SocketHandler = SocketHandler;
