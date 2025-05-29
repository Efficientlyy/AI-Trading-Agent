// MEXC WebSocket Service for real-time data

// WebSocket message types
export type MessageType = 
  | 'ping'
  | 'sub.symbol'
  | 'sub.kline'
  | 'sub.depth'
  | 'sub.trade'
  | 'sub.ticker';

// WebSocket connection status
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

// WebSocket subscription type
export interface Subscription {
  type: MessageType;
  symbol: string;
  interval?: string;
  callback: (data: any) => void;
}

class MexcWebSocketService {
  private ws: WebSocket | null = null;
  private readonly url = 'wss://wbs.mexc.com/ws';
  private connectionStatus: ConnectionStatus = 'disconnected';
  private subscriptions: Map<string, Subscription> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private pingInterval: NodeJS.Timeout | null = null;
  private statusListeners: ((status: ConnectionStatus) => void)[] = [];

  // Connect to WebSocket
  connect(): Promise<void> {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return Promise.resolve();
    }

    this.setConnectionStatus('connecting');
    
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          this.setConnectionStatus('connected');
          this.reconnectAttempts = 0;
          this.setupPingInterval();
          this.resubscribeAll();
          resolve();
        };

        this.ws.onclose = () => {
          this.setConnectionStatus('disconnected');
          this.clearPingInterval();
          this.handleReconnect();
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.setConnectionStatus('disconnected');
          reject(error);
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };
      } catch (error) {
        this.setConnectionStatus('disconnected');
        console.error('WebSocket connection error:', error);
        reject(error);
      }
    });
  }

  // Add a status listener
  addStatusListener(listener: (status: ConnectionStatus) => void): void {
    this.statusListeners.push(listener);
    // Immediately notify with current status
    listener(this.connectionStatus);
  }

  // Remove a status listener
  removeStatusListener(listener: (status: ConnectionStatus) => void): void {
    this.statusListeners = this.statusListeners.filter(l => l !== listener);
  }

  // Set connection status and notify listeners
  private setConnectionStatus(status: ConnectionStatus): void {
    this.connectionStatus = status;
    this.statusListeners.forEach(listener => listener(status));
  }

  // Setup ping interval to keep connection alive
  private setupPingInterval(): void {
    this.clearPingInterval();
    this.pingInterval = setInterval(() => {
      this.sendPing();
    }, 30000); // Send ping every 30 seconds
  }

  // Clear ping interval
  private clearPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  // Send ping to keep connection alive
  private sendPing(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const pingMessage = {
        method: 'PING'
      };
      this.ws.send(JSON.stringify(pingMessage));
    }
  }

  // Handle WebSocket reconnection
  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      this.setConnectionStatus('reconnecting');
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      
      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
        this.connect().catch(error => {
          console.error('Reconnection failed:', error);
        });
      }, delay);
    } else {
      console.error('Max reconnection attempts reached. Please reconnect manually.');
    }
  }

  // Handle incoming WebSocket messages with throttling
  private lastUpdateTime: Record<string, number> = {};
  private pendingMessages: Record<string, any> = {};
  private pendingCallbacks: Map<string, NodeJS.Timeout> = new Map();
  
  // Determine appropriate update interval based on channel type
  private getUpdateInterval(channel: string): number {
    switch (channel) {
      case 'depth':      // Order book updates
        return 500;      // 500ms (2 updates per second)
      case 'trade':      // Trade updates
        return 1000;     // 1000ms (1 update per second)
      case 'kline':      // Candlestick updates
        return 2000;     // 2000ms (0.5 updates per second)
      case 'ticker':     // Ticker updates
        return 2000;     // 2000ms (0.5 updates per second)
      default:
        return 1000;     // Default 1000ms
    }
  }
  
  // Throttle control for updates
  private readonly updateThrottleMs = 500; // Limit updates to once per 500ms per subscription
  
  private handleMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data);
      
      // Handle ping response
      if (data.channel === 'pong') {
        return;
      }
      
      // Find the appropriate subscription and call its callback
      if (data.channel && data.symbol) {
        const key = `${data.channel}.${data.symbol}${data.interval ? `.${data.interval}` : ''}`;
        const subscription = this.subscriptions.get(key);
        
        if (subscription) {
          // Throttle updates - each channel type has different update frequency needs
          const now = Date.now();
          const minInterval = this.getUpdateInterval(data.channel);
          
          // If we have a pending callback for this key, clear it
          if (this.pendingCallbacks.has(key)) {
            clearTimeout(this.pendingCallbacks.get(key)!);
          }
          
          // Store the latest data
          this.pendingMessages[key] = data;
          
          if (!this.lastUpdateTime[key] || now - this.lastUpdateTime[key] > minInterval) {
            // Use requestAnimationFrame for smoother UI
            window.requestAnimationFrame(() => {
              try {
                const currentTime = Date.now();
                const key = `${subscription.type}.${subscription.symbol}`;
                const lastUpdate = this.lastUpdateTime[key] || 0;
                
                if (currentTime - lastUpdate >= this.updateThrottleMs) {
                  this.lastUpdateTime[key] = currentTime;
                  try {
                    subscription.callback(data);
                  } catch (callbackError) {
                    console.error('Error processing subscription callback:', callbackError);
                  }
                }
              } catch (error) {
                console.error('Error in subscription callback:', error);
              }
            });
          } else {
            // Otherwise, schedule an update for later
            const timeToWait = minInterval - (now - this.lastUpdateTime[key]);
            const timeout = setTimeout(() => {
              window.requestAnimationFrame(() => {
                try {
                  if (this.pendingMessages[key]) {
                    subscription.callback(this.pendingMessages[key]);
                    this.lastUpdateTime[key] = Date.now();
                    delete this.pendingMessages[key];
                    this.pendingCallbacks.delete(key);
                  }
                } catch (callbackError) {
                  console.error('Error in delayed subscription callback:', callbackError);
                }
              });
            }, timeToWait);
            
            this.pendingCallbacks.set(key, timeout);
          }
        }
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error, event.data);
    }
  }

  // Resubscribe to all subscriptions after reconnect
  private resubscribeAll(): void {
    this.subscriptions.forEach(subscription => {
      this.sendSubscription(subscription);
    });
  }

  // Send subscription message to WebSocket
  private sendSubscription(subscription: Subscription): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.connect().then(() => {
        this.sendSubscription(subscription);
      }).catch(error => {
        console.error('Failed to connect for subscription:', error);
      });
      return;
    }

    const message: any = {
      method: 'SUBSCRIPTION',
      params: []
    };

    switch (subscription.type) {
      case 'sub.kline':
        message.params = [`spot@public.kline.v3.api@${subscription.symbol}@${subscription.interval}`];
        break;
      case 'sub.depth':
        message.params = [`spot@public.depth.v3.api@${subscription.symbol}`];
        break;
      case 'sub.trade':
        message.params = [`spot@public.deals.v3.api@${subscription.symbol}`];
        break;
      case 'sub.ticker':
        message.params = [`spot@public.ticker.v3.api@${subscription.symbol}`];
        break;
      default:
        console.error('Unknown subscription type:', subscription.type);
        return;
    }

    this.ws.send(JSON.stringify(message));
  }

  // Subscribe to market data
  subscribe(type: MessageType, symbol: string, callback: (data: any) => void, interval?: string): void {
    const key = `${type}.${symbol}${interval ? `.${interval}` : ''}`;
    
    // Store subscription
    this.subscriptions.set(key, { type, symbol, callback, interval });
    
    // Send subscription to server if connected
    if (this.connectionStatus === 'connected') {
      this.sendSubscription(this.subscriptions.get(key)!);
    } else {
      // Connect first, then subscribe
      this.connect().then(() => {
        this.sendSubscription(this.subscriptions.get(key)!);
      }).catch(error => {
        console.error('Failed to connect for subscription:', error);
      });
    }
  }

  // Unsubscribe from market data
  unsubscribe(type: MessageType, symbol: string, interval?: string): void {
    const key = `${type}.${symbol}${interval ? `.${interval}` : ''}`;
    
    if (!this.subscriptions.has(key)) {
      return;
    }
    
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const message: any = {
        method: 'UNSUBSCRIPTION',
        params: []
      };

      switch (type) {
        case 'sub.kline':
          message.params = [`spot@public.kline.v3.api@${symbol}@${interval}`];
          break;
        case 'sub.depth':
          message.params = [`spot@public.depth.v3.api@${symbol}`];
          break;
        case 'sub.trade':
          message.params = [`spot@public.deals.v3.api@${symbol}`];
          break;
        case 'sub.ticker':
          message.params = [`spot@public.ticker.v3.api@${symbol}`];
          break;
      }

      this.ws.send(JSON.stringify(message));
    }
    
    // Remove subscription
    this.subscriptions.delete(key);
  }

  // Disconnect WebSocket
  disconnect(): void {
    this.clearPingInterval();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    this.setConnectionStatus('disconnected');
    this.subscriptions.clear();
  }

  // Get current connection status
  getConnectionStatus(): ConnectionStatus {
    return this.connectionStatus;
  }
}

export const mexcWebSocketService = new MexcWebSocketService();
export default mexcWebSocketService;