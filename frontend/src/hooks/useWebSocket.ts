import { useCallback, useEffect, useState } from 'react';
import { useAuthToken } from './useAuthToken';

// WebSocket data structure
interface WebSocketData {
  ohlcv?: {
    symbol: string;
    timeframe: string;
    data: any;
  };
  portfolio?: any;
  trades?: any;
  agent_status?: {
    status: string;
    reasoning: string;
    timestamp: string;
    regime_label?: string;
    adaptive_reason?: string;
  };
  [key: string]: any;
}

/**
 * Hook for WebSocket communication with the backend
 */
export const useWebSocket = (topics: string[] = []) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const [data, setData] = useState<WebSocketData>({});
  const { getToken } = useAuthToken();

  // Use WebSocket URL from environment or default to localhost
  const wsUrl = process.env.REACT_APP_WS_URL ||
    (window.location.protocol === 'https:'
      ? `wss://${window.location.host}/ws/updates`
      : `ws://${window.location.host}/ws/updates`);

  // Initialize WebSocket connection
  useEffect(() => {
    const token = getToken();
    if (!token) return; // Don't connect if not authenticated

    const socketUrl = `${wsUrl}?token=${token}`;
    const ws = new WebSocket(socketUrl);

    // Set up event handlers
    ws.onopen = () => {
      console.log('WebSocket connected');
      setStatus('connected');

      // Subscribe to initial topics
      topics.forEach(topic => {
        ws.send(JSON.stringify({ action: 'subscribe', topic }));
      });

      // Start ping to keep connection alive
      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: 'ping' }));
        }
      }, 30000); // Ping every 30 seconds

      // Clean up interval on unmount
      return () => clearInterval(pingInterval);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setStatus('disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        // Handle different message types
        if (message.topic) {
          // Handle topic-based messages (like OHLCV, portfolio, etc.)
          setData(prevData => ({
            ...prevData,
            [message.topic]: message
          }));
        }
        else if (message.action === 'pong') {
          // Handle pong (response to ping)
          // console.log('Received pong from server');
        }
        else if (message.error) {
          console.error('WebSocket error message:', message.error);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    setSocket(ws);

    // Clean up on unmount
    return () => {
      ws.close();
    };
  }, [wsUrl, getToken]);

  // Subscribe to a topic
  const subscribe = useCallback((topic: string) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: 'subscribe', topic }));
    }
  }, [socket]);

  // Unsubscribe from a topic
  const unsubscribe = useCallback((topic: string) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: 'unsubscribe', topic }));
    }
  }, [socket]);

  // Get OHLCV stream for a specific symbol
  const getOHLCVStream = useCallback((symbol: string, timeframe: string = '1m') => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        action: 'subscribe',
        topic: 'ohlcv',
        symbol,
        timeframe
      }));
    }
  }, [socket]);

  // Stop OHLCV stream for a specific symbol
  const stopOHLCVStream = useCallback((symbol: string) => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        action: 'unsubscribe',
        topic: 'ohlcv',
        symbol
      }));
    }
  }, [socket]);

  // Start agent
  const startAgent = useCallback(() => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: 'start_agent' }));
    }
  }, [socket]);

  // Stop agent
  const stopAgent = useCallback(() => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: 'stop_agent' }));
    }
  }, [socket]);

  return {
    status,
    data,
    subscribe,
    unsubscribe,
    getOHLCVStream,
    stopOHLCVStream,
    startAgent,
    stopAgent
  };
};
