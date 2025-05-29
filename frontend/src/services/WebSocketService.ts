/**
 * WebSocket service for real-time updates.
 * This service provides a connection to the backend WebSocket server
 * for real-time updates on paper trading status, portfolio, trades, and alerts.
 */

import { v4 as uuidv4 } from 'uuid';

// WebSocket topics
export enum WebSocketTopic {
  STATUS = 'status',
  PORTFOLIO = 'portfolio',
  TRADES = 'trades',
  ALERTS = 'alerts',
  PERFORMANCE = 'performance',
  AGENT_STATUS = 'agent_status',
  PAPER_TRADING = 'paper_trading',
  DRAWDOWN = 'drawdown',
  TRADE_STATS = 'trade_stats',
  SENTIMENT_PIPELINE = 'sentiment_pipeline'
}

// WebSocket message handler type
export type WebSocketEventHandler = (data: any) => void;

// Connection status handler type
export type ConnectionStatusHandler = (isConnected: boolean, errorMessage?: string) => void;

// WebSocket data type
export interface WebSocketData {
  [key: string]: any;
}

// WebSocket message type
export interface WebSocketMessage {
  type: string;
  topic?: string;
  topics?: string[];
  data?: any;
  timestamp?: number;
}

/**
 * WebSocket service for real-time updates
 */
export class WebSocketService {
  private socket: WebSocket | null = null;
  private sessionId: string | null = null;
  private isConnected: boolean = false;
  private isConnecting: boolean = false;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 10; // Increased from 5 to 10
  private reconnectDelay: number = 1000; // Base delay in ms (reduced from 2000)
  private pingInterval: number = 30000; // 30 seconds
  private pingTimer: NodeJS.Timeout | null = null;
  private topicHandlers: Map<string, Set<WebSocketEventHandler>> = new Map();
  private messageHandlers: Set<(message: WebSocketMessage) => void> = new Set();
  private connectionStatusHandlers: Set<ConnectionStatusHandler> = new Set();
  private lastConnectionError: string | null = null;

  /**
   * Initialize the WebSocket service with a session ID
   * 
   * @param sessionId - The session ID to use for the WebSocket connection
   */
  initialize(sessionId: string): void {
    this.sessionId = sessionId;
  }

  /**
   * Connect to the WebSocket server
   * 
   * @param topics - Optional topics to subscribe to
   * @param sessionId - Optional session ID to use
   * @returns A promise that resolves when the connection is established
   */
  connect(topics?: string[] | string, sessionId?: string): Promise<void> {
    // Handle backward compatibility with old API
    if (sessionId) {
      this.initialize(sessionId);
    }
    
    // If topics are provided, store them for subscription after connection
    if (topics) {
      if (Array.isArray(topics)) {
        topics.forEach(topic => {
          this.subscribe(topic, () => {});
        });
      } else {
        this.subscribe(topics, () => {});
      }
    }
    
    // If already connected, return success
    if (this.isConnected) {
      console.log('WebSocket already connected, reusing existing connection');
      return Promise.resolve();
    }

    // If already connecting, return a pending promise
    if (this.isConnecting) {
      console.log('WebSocket connection already in progress, waiting...');
      return Promise.reject(new Error('WebSocket connection already in progress'));
    }

    // Reset reconnect attempts when manually connecting
    this.reconnectAttempts = 0;
    this.isConnecting = true;
    
    // Notify that connection is being attempted
    this.notifyConnectionStatusHandlers(false, 'Connecting to server...');

    return new Promise<void>((resolve, reject) => {
      try {
        // Create WebSocket connection with direct URL
        // Use a hardcoded WebSocket URL for development to ensure it works
        const defaultSessionId = '4fb71ca9-351d-46cb-996f-2c3bc4d90b70';
        const sessionIdToUse = this.sessionId || defaultSessionId;

        if (!this.sessionId) {
          this.sessionId = defaultSessionId;
        }

        // Direct WebSocket URL
        // Check if we're in development or production
        const host = window.location.hostname;
        const port = '8000'; // Backend port
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        
        // Construct WebSocket URL
        const wsUrl = `${wsProtocol}//${host}:${port}/ws/${sessionIdToUse}`;

        console.log('Connecting to WebSocket URL:', wsUrl);
        console.log('Session ID:', sessionIdToUse);

        // Set a timeout to reject the promise if connection takes too long
        const connectionTimeout = setTimeout(() => {
          if (!this.isConnected) {
            this.isConnecting = false;
            console.error('WebSocket connection timeout. Please ensure the backend server is running.');
            this.notifyConnectionStatusHandlers(false, 'Connection timeout. Please ensure the backend server is running.');
            reject(new Error('WebSocket connection timeout. Please ensure the backend server is running.'));
          }
        }, 5000); // 5 seconds

        // Create WebSocket connection
        this.socket = new WebSocket(wsUrl);

        // Set up event handlers
        this.socket.onopen = () => {
          console.log('WebSocket connection established');
          this.isConnected = true;
          this.isConnecting = false;
          this.reconnectAttempts = 0;

          // Start ping timer
          this.startPingTimer();

          // Send subscription message
          this.sendSubscription();

          // Notify connection status handlers
          this.notifyConnectionStatusHandlers(true);

          // Clear the timeout and resolve the promise
          clearTimeout(connectionTimeout);
          resolve();

          // Log success message
          console.log('WebSocket connection successfully established with session ID:', this.sessionId);
        };

        this.socket.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data) as WebSocketMessage;
            console.log('WebSocket message received:', message);
            this.handleMessage(message);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
            this.notifyConnectionStatusHandlers(false, 'Error parsing WebSocket message');
          }
        };

        this.socket.onclose = (event) => {
          console.log('WebSocket connection closed:', event.code, event.reason);
          this.isConnected = false;
          this.isConnecting = false;

          // Stop ping timer
          this.stopPingTimer();

          // Notify connection status handlers
          this.notifyConnectionStatusHandlers(false, `WebSocket connection closed: ${event.code} ${event.reason}`);

          // Attempt to reconnect if not closed cleanly
          if (event.code !== 1000 && event.code !== 1001) {
            this.reconnect();
          }
        };

        this.socket.onerror = (event) => {
          const errorMessage = 'WebSocket connection error - Unable to connect to backend server';
          console.error(errorMessage, event);
          this.isConnected = false;
          this.isConnecting = false;

          // Stop ping timer
          this.stopPingTimer();

          // Notify connection status handlers
          this.notifyConnectionStatusHandlers(false, errorMessage);

          // Clear the timeout and reject the promise
          clearTimeout(connectionTimeout);
          reject(new Error(errorMessage));
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }

    this.isConnected = false;
    this.isConnecting = false;
    this.stopPingTimer();
  }

  /**
   * Subscribe to a topic
   * 
   * @param topic - The topic to subscribe to
   * @param handler - The handler to call when a message is received for the topic
   */
  subscribe(topic: string, handler: WebSocketEventHandler): void {
    // For backward compatibility
    this.on(topic, handler);
  }
  
  /**
   * Alias for subscribe (backward compatibility)
   * 
   * @param topic - The topic to subscribe to
   * @param handler - The handler to call when a message is received for the topic
   */
  on(topic: string, handler: WebSocketEventHandler): void {
    if (!this.topicHandlers.has(topic)) {
      this.topicHandlers.set(topic, new Set());
    }

    const handlers = this.topicHandlers.get(topic);
    if (handlers) {
      handlers.add(handler);
    }

    // If connected, send subscription message
    if (this.isConnected && this.socket) {
      this.sendSubscription();
    }
  }

  /**
   * Unsubscribe from a topic
   * 
   * @param topic - The topic to unsubscribe from
   * @param handler - The handler to remove
   */
  unsubscribe(topic: string, handler: WebSocketEventHandler): void {
    // For backward compatibility
    this.off(topic, handler);
  }
  
  /**
   * Alias for unsubscribe (backward compatibility)
   * 
   * @param topic - The topic to unsubscribe from
   * @param handler - The handler to remove
   */
  off(topic: string, handler: WebSocketEventHandler): void {
    if (this.topicHandlers.has(topic)) {
      const handlers = this.topicHandlers.get(topic);
      if (handlers) {
        handlers.delete(handler);

        if (handlers.size === 0) {
          this.topicHandlers.delete(topic);
        }
      }
    }
  }

  /**
   * Register a handler for all messages
   * 
   * @param handler - The handler to call when any message is received
   */
  onMessage(handler: (message: WebSocketMessage) => void): void {
    this.messageHandlers.add(handler);
  }

  /**
   * Unregister a message handler
   * 
   * @param handler - The handler to remove
   */
  offMessage(handler: (message: WebSocketMessage) => void): void {
    this.messageHandlers.delete(handler);
  }

  /**
   * Register a connection status handler
   * 
   * @param handler - The handler to call when the connection status changes
   */
  onConnectionStatus(handler: ConnectionStatusHandler): void {
    this.connectionStatusHandlers.add(handler);

    // Immediately call the handler with the current connection status
    handler(this.isConnected, this.isConnected ? undefined : 'Not connected');
  }

  /**
   * Unregister a connection status handler
   * 
   * @param handler - The handler to remove
   */
  offConnectionStatus(handler: ConnectionStatusHandler): void {
    this.connectionStatusHandlers.delete(handler);
  }

/**
 * Send a message to the WebSocket server
 * 
 * @param message - The message to send
 */
public send(message: WebSocketMessage): void {
  if (this.socket && this.isConnected) {
    try {
      this.socket.send(JSON.stringify(message));
      console.log('Sent message to WebSocket server:', message);
    } catch (error) {
      console.error('Error sending message to WebSocket server:', error);
    }
  } else {
    console.warn('Cannot send message, WebSocket not connected');
  }
}

/**
 * Send a subscription message for all registered topics
 */
private sendSubscription(): void {
  if (this.socket && this.isConnected) {
    const topics = Array.from(this.topicHandlers.keys());

    // If no topics, subscribe to 'performance' by default
    if (topics.length === 0) {
      topics.push('performance');
    }

    this.send({
      type: 'subscribe',
      data: {
        topics
      }
    });
  }
}

/**
 * Handle a message received from the WebSocket server
 * 
 * @param message - The message received
 */
private handleMessage(message: WebSocketMessage): void {
  // Notify all message handlers
  this.messageHandlers.forEach((handler) => {
    try {
      handler(message);
    } catch (error) {
      console.error('Error in message handler:', error);
    }
  });

  // If message has a topic, notify topic handlers
  if (message.topic && this.topicHandlers.has(message.topic)) {
    const handlers = this.topicHandlers.get(message.topic);

    if (handlers) {
      handlers.forEach((handler) => {
        try {
          handler(message.data);
        } catch (error) {
          console.error(`Error in handler for topic ${message.topic}:`, error);
        }
      });
    }
  }
}

/**
 * Start the ping timer to keep the connection alive
 */
private startPingTimer(): void {
  this.stopPingTimer();

  this.pingTimer = setInterval(() => {
    if (this.socket && this.isConnected) {
      this.send({
        type: 'ping',
        data: {
          timestamp: Date.now()
        }
      });
    }
  }, this.pingInterval);
}

/**
 * Stop the ping timer
 */
private stopPingTimer(): void {
  if (this.pingTimer) {
    clearInterval(this.pingTimer);
    this.pingTimer = null;
  }
}

/**
 * Attempt to reconnect to the WebSocket server with exponential backoff
 */
private reconnect(): void {
  if (this.reconnectAttempts >= this.maxReconnectAttempts) {
    console.error(`Maximum reconnect attempts (${this.maxReconnectAttempts}) reached`);
    this.notifyConnectionStatusHandlers(false, `Failed to reconnect after ${this.maxReconnectAttempts} attempts. Please reload the page.`);
    return;
  }

  this.reconnectAttempts++;

  // Use exponential backoff with jitter for reconnection attempts
  const baseDelay = this.reconnectDelay;
  const exponentialDelay = baseDelay * Math.pow(1.5, this.reconnectAttempts - 1);
  const jitter = Math.random() * 0.3 * exponentialDelay; // Add up to 30% jitter
  const delay = Math.min(exponentialDelay + jitter, 30000); // Cap at 30 seconds
  
  console.log(`Attempting to reconnect in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
  
  // Notify users about reconnection attempts
  this.notifyConnectionStatusHandlers(
    false, 
    `Connection lost. Reconnecting in ${Math.round(delay / 1000)} seconds... (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`
  );

  setTimeout(() => {
    if (!this.isConnected && !this.isConnecting) {
      console.log(`Reconnecting... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      this.connect().catch((error) => {
        console.error('Reconnection failed:', error);
      });
    }
  }, delay);
}

/**
 * Notify all connection status handlers of a change in connection status
 * 
 * @param isConnected - Whether the connection is established
 * @param errorMessage - Optional error message if the connection failed
 */
private notifyConnectionStatusHandlers(isConnected: boolean, errorMessage?: string): void {
  // Log connection status changes
  if (isConnected) {
    console.log('WebSocket connected successfully');
  } else if (errorMessage) {
    console.warn(`WebSocket connection issue: ${errorMessage}`);
  }
  
  this.connectionStatusHandlers.forEach((handler) => {
    try {
      handler(isConnected, errorMessage);
    } catch (error) {
      console.error('Error in connection status handler:', error);
    }
  });
}

/**
 * Check if the WebSocket connection is open
 * @returns True if the connection is open, false otherwise
 */
public isConnectionOpen(): boolean {
  return this.isConnected;
}

/**
 * Add a handler for a specific topic
 * @param topic - The topic to add a handler for
 * @param handler - The handler function
 */
public addTopicHandler(topic: string | WebSocketTopic, handler: WebSocketEventHandler): void {
  // Create a set for this topic if it doesn't exist
  if (!this.topicHandlers.has(topic)) {
    this.topicHandlers.set(topic, new Set());
  }
  
  // Add the handler
  const handlers = this.topicHandlers.get(topic);
  if (handlers) {
    handlers.add(handler);
  }
}

/**
 * Remove a handler for a specific topic
 * @param topic - The topic to remove a handler for
 * @param handler - The handler function to remove
 */
public removeTopicHandler(topic: string | WebSocketTopic, handler: WebSocketEventHandler): void {
  // Remove the handler if the topic exists
  const handlers = this.topicHandlers.get(topic);
  if (handlers) {
    handlers.delete(handler);
  }
}

/**
 * Add a connection status handler
 * @param handler - The handler function
 */
public addConnectionStatusHandler(handler: ConnectionStatusHandler): void {
  this.connectionStatusHandlers.add(handler);
  
  // Immediately notify with current status
  handler(this.isConnected, this.lastConnectionError || undefined);
}

/**
 * Remove a connection status handler
 * @param handler - The handler function to remove
 */
public removeConnectionStatusHandler(handler: ConnectionStatusHandler): void {
  this.connectionStatusHandlers.delete(handler);
}

}

// Create singleton instance
export default new WebSocketService();
