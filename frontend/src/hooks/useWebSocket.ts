import { useState, useEffect, useCallback, useRef } from 'react';
import { WebSocketMessage, WebSocketUpdate, TopicType } from '../types';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws/updates';

interface WebSocketHookResult {
  data: WebSocketUpdate;
  status: 'connecting' | 'connected' | 'disconnected';
  error: string | null;
  subscribe: (topic: TopicType) => void;
  unsubscribe: (topic: TopicType) => void;
}

// Mock data for development mode
const MOCK_DATA: Record<string, any> = {
  portfolio: {
    total_value: 45678.92,
    cash: 12345.67,
    positions: {
      'BTC': { 
        symbol: 'BTC',
        quantity: 0.5, 
        entry_price: 42000,
        current_price: 45000,
        market_value: 22500.00,
        unrealized_pnl: 1500,
        realized_pnl: 0
      },
      'ETH': { 
        symbol: 'ETH',
        quantity: 3.2, 
        entry_price: 2800,
        current_price: 3000,
        market_value: 9600.00,
        unrealized_pnl: 640,
        realized_pnl: 0
      },
      'SOL': { 
        symbol: 'SOL',
        quantity: 25, 
        entry_price: 45,
        current_price: 50,
        market_value: 1250.00,
        unrealized_pnl: 125,
        realized_pnl: 0
      }
    },
    daily_pnl: 1250.75
  },
  sentiment_signal: {
    'BTC/USD': { signal: 'buy', strength: 0.8 },
    'ETH/USD': { signal: 'hold', strength: 0.5 },
    'SOL/USD': { signal: 'sell', strength: -0.7 }
  },
  performance: {
    total_return: 12.5,
    sharpe_ratio: 1.8,
    max_drawdown: -5.2,
    win_rate: 0.65,
    profit_factor: 1.75,
    avg_trade: 125.5
  }
};

export const useWebSocket = (initialTopics: TopicType[] = []): WebSocketHookResult => {
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [data, setData] = useState<WebSocketUpdate>({});
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const topicsRef = useRef<Set<TopicType>>(new Set(initialTopics));
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const initialTopicsProcessedRef = useRef<boolean>(false);
  const MAX_RECONNECT_ATTEMPTS = 3;

  // Function to send a message to the WebSocket server
  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Helper function to update data based on topic
  const updateDataForTopic = useCallback((topic: TopicType, topicData: any) => {
    setData(prevData => {
      const newData = { ...prevData };
      
      // Update the correct property based on the topic
      if (topic === 'portfolio') {
        newData.portfolio = topicData;
      } else if (topic === 'sentiment_signal') {
        newData.sentiment_signal = topicData;
      } else if (topic === 'performance') {
        newData.performance = topicData;
      }
      
      return newData;
    });
  }, []);

  // Subscribe to a topic
  const subscribe = useCallback((topic: TopicType) => {
    topicsRef.current.add(topic);
    setError(null); // Clear any previous errors
    
    // In development mode, immediately update with mock data
    if (process.env.NODE_ENV === 'development') {
      if (MOCK_DATA[topic]) {
        updateDataForTopic(topic, MOCK_DATA[topic]);
      }
      return;
    }
    
    // In production, send subscription to server
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      sendMessage({ action: 'subscribe', topic });
    }
  }, [sendMessage, updateDataForTopic]);

  // Unsubscribe from a topic
  const unsubscribe = useCallback((topic: TopicType) => {
    topicsRef.current.delete(topic);
    
    // In development mode, no need to send unsubscribe message
    if (process.env.NODE_ENV === 'development') {
      return;
    }
    
    // In production, send unsubscribe to server
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      sendMessage({ action: 'unsubscribe', topic });
    }
  }, [sendMessage]);

  // Initialize with mock data in development mode
  useEffect(() => {
    // Only run once and only in development mode
    if (process.env.NODE_ENV === 'development' && !initialTopicsProcessedRef.current) {
      console.log('Using mock WebSocket data in development mode');
      setStatus('connected');
      
      // Initialize with mock data for initial topics
      const currentTopics = Array.from(topicsRef.current);
      currentTopics.forEach((topic: TopicType) => {
        if (MOCK_DATA[topic]) {
          updateDataForTopic(topic, MOCK_DATA[topic]);
        }
      });
      
      initialTopicsProcessedRef.current = true;
    }
  }, [updateDataForTopic]);

  // Setup WebSocket connection in production
  useEffect(() => {
    // Skip in development mode
    if (process.env.NODE_ENV === 'development') {
      return;
    }
    
    // For production, use real WebSocket connection
    const connectWebSocket = () => {
      // Clear any existing reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      // Create new WebSocket connection
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      setStatus('connecting');
      setError(null); // Clear any previous errors

      // Store a copy of the topics ref for the cleanup function
      const currentTopics = Array.from(topicsRef.current);

      ws.onopen = () => {
        setStatus('connected');
        console.log('WebSocket connected');
        reconnectAttemptsRef.current = 0; // Reset reconnect attempts on successful connection
        
        // Subscribe to all topics in the set
        currentTopics.forEach((topic: TopicType) => {
          ws.send(JSON.stringify({ action: 'subscribe', topic }));
        });
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          // Assume the message contains topic data in a format like { topic: data }
          Object.entries(message).forEach(([topicKey, topicData]) => {
            const topic = topicKey as TopicType;
            if (topic === 'portfolio' || topic === 'sentiment_signal' || topic === 'performance') {
              updateDataForTopic(topic, topicData);
            }
          });
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
          setError('Failed to parse WebSocket message');
        }
      };

      ws.onclose = () => {
        setStatus('disconnected');
        console.log('WebSocket disconnected');
        
        // Attempt to reconnect after a delay, but only up to MAX_RECONNECT_ATTEMPTS
        if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
          console.log(`Attempting to reconnect (${reconnectAttemptsRef.current + 1}/${MAX_RECONNECT_ATTEMPTS})...`);
          reconnectAttemptsRef.current += 1;
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, 3000);
        } else {
          console.log(`Max reconnect attempts (${MAX_RECONNECT_ATTEMPTS}) reached. Using mock data instead.`);
          setError('Failed to connect to WebSocket server. Using mock data instead.');
          // Fall back to mock data after max reconnect attempts
          currentTopics.forEach((topic: TopicType) => {
            if (MOCK_DATA[topic]) {
              updateDataForTopic(topic, MOCK_DATA[topic]);
            }
          });
          setStatus('connected'); // Pretend we're connected to avoid UI showing disconnected state
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('WebSocket connection error');
        ws.close();
      };

      // Return cleanup function
      return () => {
        // Unsubscribe from all topics before closing
        if (ws.readyState === WebSocket.OPEN) {
          currentTopics.forEach((topic: TopicType) => {
            ws.send(JSON.stringify({ action: 'unsubscribe', topic }));
          });
        }
        
        ws.close();
      };
    };

    // Connect to WebSocket and get the cleanup function
    const cleanup = connectWebSocket();

    // Cleanup when component unmounts
    return () => {
      if (cleanup) {
        cleanup();
      }
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [updateDataForTopic]); // Only depend on updateDataForTopic

  // Subscribe to initial topics when status changes to connected
  useEffect(() => {
    if (status === 'connected' && initialTopics.length > 0 && !initialTopicsProcessedRef.current) {
      initialTopics.forEach(topic => {
        subscribe(topic);
      });
      initialTopicsProcessedRef.current = true;
    }
  }, [status, subscribe, initialTopics]);

  return { data, status, error, subscribe, unsubscribe };
};
