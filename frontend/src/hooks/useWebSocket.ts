import { useState, useEffect, useCallback, useRef } from 'react';
import { WebSocketMessage, WebSocketUpdate, TopicType, OHLCVLiveUpdate } from '../types';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws/updates';

interface WebSocketHookResult {
  data: WebSocketUpdate;
  status: 'connecting' | 'connected' | 'disconnected';
  error: string | null;
  subscribe: (topic: TopicType, options?: { symbol?: string; timeframe?: string }) => void;
  unsubscribe: (topic: TopicType, options?: { symbol?: string; timeframe?: string }) => void;
  getOHLCVStream: (symbol: string, timeframe: string) => void;
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
  // Store symbol/timeframe subscriptions for ohlcv
  const ohlcvSubscriptions = useRef<{ [key: string]: { symbol: string; timeframe: string } }>({});
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [data, setData] = useState<WebSocketUpdate>({});
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const topicsRef = useRef<Set<TopicType>>(new Set(initialTopics));
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const initialTopicsProcessedRef = useRef<boolean>(false);
  const reconnectAttemptsRef = useRef<number>(0);
  const MAX_RECONNECT_ATTEMPTS = 3;

  // Enhanced subscribe to support symbol/timeframe for ohlcv
  const subscribe = useCallback((topic: TopicType, options?: { symbol?: string; timeframe?: string }) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const msg: any = { action: 'subscribe', topic };
      if (topic === 'ohlcv' && options?.symbol) {
        msg.symbol = options.symbol;
        msg.timeframe = options.timeframe || '1m';
        ohlcvSubscriptions.current[`${options.symbol}:${msg.timeframe}`] = { symbol: options.symbol, timeframe: msg.timeframe };
      }
      wsRef.current.send(JSON.stringify(msg));
      topicsRef.current.add(topic);
    }
  }, []);

  const unsubscribe = useCallback((topic: TopicType, options?: { symbol?: string; timeframe?: string }) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const msg: any = { action: 'unsubscribe', topic };
      if (topic === 'ohlcv' && options?.symbol) {
        msg.symbol = options.symbol;
        msg.timeframe = options.timeframe || '1m';
        delete ohlcvSubscriptions.current[`${options.symbol}:${msg.timeframe}`];
      }
      wsRef.current.send(JSON.stringify(msg));
      topicsRef.current.delete(topic);
    }
  }, []);

  // Helper to get live OHLCV stream
  const getOHLCVStream = useCallback((symbol: string, timeframe: string) => {
    subscribe('ohlcv', { symbol, timeframe });
  }, [subscribe]);

  // Update data for a topic
  const updateDataForTopic = useCallback((topic: TopicType, topicData: any) => {
    setData(prev => {
      const newData = { ...prev };
      if (topic === 'portfolio') {
        newData.portfolio = topicData;
      } else if (topic === 'sentiment_signal') {
        newData.sentiment_signal = topicData;
      } else if (topic === 'performance') {
        newData.performance = topicData;
      } else if (topic === 'ohlcv') {
        if (Array.isArray(topicData.data)) {
          newData.ohlcv = topicData;
        } else if (typeof topicData.data === 'object' && topicData.data !== null) {
          // Defensive: ensure topic is present
          newData.ohlcv = { ...newData.ohlcv, ...topicData, topic: 'ohlcv' };
        }
      }
      return newData;
    });
  }, []);

  // WebSocket connection logic
  useEffect(() => {
    let ws: WebSocket;
    let currentTopics = new Set(topicsRef.current);

    const connectWebSocket = () => {
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      setStatus('connecting');

      ws.onopen = () => {
        setStatus('connected');
        setError(null);
        currentTopics.forEach((topic: TopicType) => {
          ws.send(JSON.stringify({ action: 'subscribe', topic }));
        });
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.topic === 'ohlcv') {
            setData(prev => ({ ...prev, ohlcv: msg }));
          } else if (msg.topic) {
            updateDataForTopic(msg.topic, msg);
          }
        } catch (e) {
          console.error('WebSocket message parse error:', e);
          setError('Failed to parse WebSocket message');
        }
      };

      ws.onclose = () => {
        setStatus('disconnected');
        if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttemptsRef.current += 1;
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, 3000);
        } else {
          setError('Failed to connect to WebSocket server. Using mock data instead.');
          currentTopics.forEach((topic: TopicType) => {
            if (MOCK_DATA[topic]) {
              updateDataForTopic(topic, MOCK_DATA[topic]);
            }
          });
          setStatus('connected');
        }
      };

      ws.onerror = (error) => {
        setError('WebSocket connection error');
        ws.close();
      };

      return () => {
        if (ws.readyState === WebSocket.OPEN) {
          currentTopics.forEach((topic: TopicType) => {
            ws.send(JSON.stringify({ action: 'unsubscribe', topic }));
          });
        }
        ws.close();
      };
    };

    const cleanup = connectWebSocket();

    return () => {
      if (cleanup) cleanup();
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [updateDataForTopic]);

  // Subscribe to initial topics when status changes to connected
  useEffect(() => {
    if (status === 'connected' && initialTopics.length > 0 && !initialTopicsProcessedRef.current) {
      initialTopics.forEach(topic => {
        subscribe(topic);
      });
      initialTopicsProcessedRef.current = true;
    }
  }, [status, subscribe, initialTopics]);

  return { data, status, error, subscribe, unsubscribe, getOHLCVStream };
};
