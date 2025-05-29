import { useState, useEffect, useCallback, useRef } from 'react';
import mexcService, { MexcTicker, MexcOrderBook, MexcTrade } from '../api/mexcService';
import mexcWebSocketService, { ConnectionStatus } from '../api/mexcWebSocketService';

// Using real data with optimizations to prevent browser freezing
const USE_MOCK_DATA = false;

// Mock data generators
const generateMockTicker = (symbol: string): MexcTicker => {
  const basePrice = 30000 + Math.random() * 2000;
  const change = (Math.random() * 2 - 1) * 2; // Random change between -2% and 2%
  
  return {
    symbol: symbol.replace('/', '_'),
    lastPrice: basePrice.toFixed(2),
    priceChange: (basePrice * change / 100).toFixed(2),
    priceChangePercent: change.toFixed(2),
    volume: (Math.random() * 1000 + 500).toFixed(2),
    quoteVolume: (Math.random() * 10000000 + 5000000).toFixed(2),
    high: (basePrice * 1.02).toFixed(2),
    low: (basePrice * 0.98).toFixed(2)
  };
};

const generateMockOrderBook = (symbol: string): MexcOrderBook => {
  const basePrice = 30000;
  const bids: [string, string][] = [];
  const asks: [string, string][] = [];
  
  // Generate bids (buy orders) - slightly below current price
  for (let i = 0; i < 20; i++) {
    const price = (basePrice * (1 - 0.0001 * (i + 1))).toFixed(2);
    const quantity = (Math.random() * 2 + 0.1).toFixed(6);
    bids.push([price, quantity]);
  }
  
  // Generate asks (sell orders) - slightly above current price
  for (let i = 0; i < 20; i++) {
    const price = (basePrice * (1 + 0.0001 * (i + 1))).toFixed(2);
    const quantity = (Math.random() * 2 + 0.1).toFixed(6);
    asks.push([price, quantity]);
  }
  
  return {
    symbol: symbol.replace('/', '_'),
    bids,
    asks,
    timestamp: Date.now()
  };
};

const generateMockTrades = (symbol: string, count: number = 30): MexcTrade[] => {
  const basePrice = 30000;
  const trades: MexcTrade[] = [];
  
  for (let i = 0; i < count; i++) {
    // Random price around base price with small variations
    const price = (basePrice * (1 + (Math.random() * 0.002 - 0.001))).toFixed(2);
    
    // Random quantity between 0.01 and 2 BTC
    const qty = (Math.random() * 1.99 + 0.01).toFixed(6);
    
    // Random timestamp within the last 10 minutes
    const time = Date.now() - Math.floor(Math.random() * 10 * 60 * 1000);
    
    // About 50/50 buys and sells
    const isBuyerMaker = Math.random() > 0.5;
    
    trades.push({
      id: 10000000 + i,
      price,
      qty,
      time,
      isBuyerMaker
    });
  }
  
  // Sort by time, most recent first
  return trades.sort((a, b) => b.time - a.time);
};

const generateMockKlineData = (symbol: string, timeframe: string, count: number = 100): any[] => {
  const basePrice = 30000;
  const klineData: any[] = [];
  
  // Determine the timeframe in milliseconds
  let interval: number;
  switch (timeframe) {
    case '1m': interval = 60 * 1000; break;
    case '5m': interval = 5 * 60 * 1000; break;
    case '15m': interval = 15 * 60 * 1000; break;
    case '30m': interval = 30 * 60 * 1000; break;
    case '1h': interval = 60 * 60 * 1000; break;
    case '4h': interval = 4 * 60 * 60 * 1000; break;
    case '1d': interval = 24 * 60 * 60 * 1000; break;
    case '1w': interval = 7 * 24 * 60 * 60 * 1000; break;
    default: interval = 60 * 60 * 1000; // Default to 1h
  }
  
  let currentPrice = basePrice;
  let currentTime = Date.now() - (count * interval);
  
  for (let i = 0; i < count; i++) {
    // Generate a random price movement (-2% to +2%)
    const priceChange = currentPrice * (Math.random() * 0.04 - 0.02);
    
    // Calculate candle values
    const open = currentPrice;
    const close = currentPrice + priceChange;
    
    // High is the maximum of open and close, plus a little extra
    const high = Math.max(open, close) * (1 + Math.random() * 0.01);
    
    // Low is the minimum of open and close, minus a little extra
    const low = Math.min(open, close) * (1 - Math.random() * 0.01);
    
    // Volume is random but higher during bigger price movements
    const volume = Math.abs(priceChange) * (Math.random() * 10 + 5);
    
    klineData.push({
      time: currentTime,
      open,
      high,
      low,
      close,
      volume
    });
    
    // Update for next candle
    currentPrice = close;
    currentTime += interval;
  }
  
  return klineData;
};

interface MexcData {
  ticker: MexcTicker | null;
  orderBook: MexcOrderBook | null;
  trades: MexcTrade[];
  klineData: any[];
  connectionStatus: ConnectionStatus;
  isLoading: boolean;
  error: string | null;
}

export function useMexcData(symbol: string, timeframe: string) {
  const [data, setData] = useState<MexcData>({
    ticker: null,
    orderBook: null,
    trades: [],
    klineData: [],
    connectionStatus: 'disconnected',
    isLoading: true,
    error: null
  });

  // Use mock data update interval - simulates WebSocket updates
  const mockDataIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initial data fetch
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        if (USE_MOCK_DATA) {
          // Use mock data
          const ticker = generateMockTicker(symbol);
          const orderBook = generateMockOrderBook(symbol);
          const trades = generateMockTrades(symbol, 30);
          const klineData = generateMockKlineData(symbol, timeframe, 100);
          
          setData(prev => ({
            ...prev,
            ticker,
            orderBook,
            trades,
            klineData,
            connectionStatus: 'connected',
            isLoading: false
          }));

          // Set up mock data update interval
          if (mockDataIntervalRef.current) {
            clearInterval(mockDataIntervalRef.current);
          }

          mockDataIntervalRef.current = setInterval(() => {
            // Update ticker every 2 seconds
            const newTicker = generateMockTicker(symbol);
            
            // Update order book occasionally
            const shouldUpdateOrderBook = Math.random() > 0.5;
            const newOrderBook = shouldUpdateOrderBook ? generateMockOrderBook(symbol) : data.orderBook;
            
            // Add new trades occasionally
            const shouldAddTrade = Math.random() > 0.3;
            let newTrades = [...(data.trades || [])];
            if (shouldAddTrade) {
              const newTrade = generateMockTrades(symbol, 1)[0];
              newTrades = [newTrade, ...newTrades].slice(0, 30);
            }
            
            // Update the most recent candle
            let newKlineData = [...(data.klineData || [])];
            if (newKlineData.length > 0) {
              const lastCandle = { ...newKlineData[newKlineData.length - 1] };
              lastCandle.close = parseFloat(newTicker.lastPrice);
              lastCandle.high = Math.max(lastCandle.high, lastCandle.close);
              lastCandle.low = Math.min(lastCandle.low, lastCandle.close);
              newKlineData[newKlineData.length - 1] = lastCandle;
            }
            
            setData(prev => ({
              ...prev,
              ticker: newTicker,
              orderBook: newOrderBook,
              trades: newTrades,
              klineData: newKlineData
            }));
          }, 2000);
        } else {
          // Use real API data
          const ticker = await mexcService.getSymbolData(symbol);
          const orderBook = await mexcService.getOrderBook(symbol);
          const trades = await mexcService.getRecentTrades(symbol);
          const klineData = mexcService.formatKlineData(await mexcService.getKlines(symbol, timeframe));
          
          setData(prev => ({
            ...prev,
            ticker,
            orderBook,
            trades,
            klineData,
            isLoading: false
          }));
        }
      } catch (error) {
        console.error('Error fetching MEXC data:', error);
        setData(prev => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error ? error.message : String(error)
        }));
      }
    };

    fetchInitialData();

    // Cleanup function
    return () => {
      if (mockDataIntervalRef.current) {
        clearInterval(mockDataIntervalRef.current);
        mockDataIntervalRef.current = null;
      }
    };
  }, [symbol, timeframe]);

  // Create refs to store data without causing re-renders - only used with real WebSocket
  const dataRef = useRef({
    ticker: null as MexcTicker | null,
    orderBook: null as MexcOrderBook | null,
    trades: [] as MexcTrade[],
    klineData: [] as any[],
  });
  
  // Create debounced state update function
  const scheduledUpdateRef = useRef<NodeJS.Timeout | null>(null);
  
  // Combined state update function to batch updates
  const updateDataState = useCallback(() => {
    if (scheduledUpdateRef.current) {
      clearTimeout(scheduledUpdateRef.current);
      scheduledUpdateRef.current = null;
    }
    
    // Schedule state update with requestAnimationFrame for better performance
    scheduledUpdateRef.current = setTimeout(() => {
      window.requestAnimationFrame(() => {
        setData(prev => ({
          ...prev,
          ticker: dataRef.current.ticker,
          orderBook: dataRef.current.orderBook,
          trades: dataRef.current.trades,
          klineData: dataRef.current.klineData,
        }));
        scheduledUpdateRef.current = null;
      });
    }, 100); // Batch updates together with 100ms delay
  }, []);

  // Set up WebSocket connections (only if not using mock data)
  useEffect(() => {
    if (USE_MOCK_DATA) return; // Skip WebSocket setup if using mock data

    // Add status listener
    mexcWebSocketService.addStatusListener((status) => {
      setData(prev => ({ ...prev, connectionStatus: status }));
    });

    // Connect to WebSocket
    mexcWebSocketService.connect().catch(error => {
      console.error('Failed to connect to MEXC WebSocket:', error);
    });

    // Subscribe to ticker updates
    mexcWebSocketService.subscribe('sub.ticker', symbol, (tickerData) => {
      dataRef.current.ticker = tickerData.data;
      updateDataState();
    });

    // Subscribe to order book updates
    mexcWebSocketService.subscribe('sub.depth', symbol, (depthData) => {
      dataRef.current.orderBook = depthData.data;
      updateDataState();
    });

    // Subscribe to trade updates
    mexcWebSocketService.subscribe('sub.trade', symbol, (tradeData) => {
      // Keep only the most recent trades
      dataRef.current.trades = [tradeData.data, ...dataRef.current.trades].slice(0, 30);
      updateDataState();
    });

    // Subscribe to kline updates
    mexcWebSocketService.subscribe('sub.kline', symbol, (klineData) => {
      // Update existing kline data with new candle
      const newData = [...dataRef.current.klineData];
      const lastIndex = newData.findIndex(candle => candle.time === klineData.data.t);
      
      const updatedCandle = {
        time: klineData.data.t,
        open: klineData.data.o,
        high: klineData.data.h,
        low: klineData.data.l,
        close: klineData.data.c,
        volume: klineData.data.v
      };
      
      if (lastIndex >= 0) {
        // Update existing candle
        newData[lastIndex] = updatedCandle;
      } else {
        // Add new candle - limit to 100 candles maximum to prevent memory issues
        newData.push(updatedCandle);
        
        // Sort by time and limit size
        newData.sort((a, b) => a.time - b.time);
        if (newData.length > 100) {
          newData.splice(0, newData.length - 100);
        }
      }
      
      dataRef.current.klineData = newData;
      updateDataState();
    }, timeframe);

    // Cleanup function
    return () => {
      if (scheduledUpdateRef.current) {
        clearTimeout(scheduledUpdateRef.current);
      }
      mexcWebSocketService.unsubscribe('sub.ticker', symbol);
      mexcWebSocketService.unsubscribe('sub.depth', symbol);
      mexcWebSocketService.unsubscribe('sub.trade', symbol);
      mexcWebSocketService.unsubscribe('sub.kline', symbol, timeframe);
    };
  }, [symbol, timeframe, updateDataState]);

  // Function to refresh data manually
  const refreshData = async () => {
    try {
      setData(prev => ({ ...prev, isLoading: true, error: null }));
      
      if (USE_MOCK_DATA) {
        // Refresh with new mock data
        const ticker = generateMockTicker(symbol);
        const orderBook = generateMockOrderBook(symbol);
        const trades = generateMockTrades(symbol, 30);
        const klineData = generateMockKlineData(symbol, timeframe, 100);
        
        setData(prev => ({
          ...prev,
          ticker,
          orderBook,
          trades,
          klineData,
          isLoading: false
        }));
      } else {
        // Refresh with real API data
        const ticker = await mexcService.getSymbolData(symbol);
        const orderBook = await mexcService.getOrderBook(symbol);
        const trades = await mexcService.getRecentTrades(symbol);
        const klineData = mexcService.formatKlineData(await mexcService.getKlines(symbol, timeframe));
        
        setData(prev => ({
          ...prev,
          ticker,
          orderBook,
          trades,
          klineData,
          isLoading: false
        }));
      }
    } catch (error) {
      console.error('Error refreshing MEXC data:', error);
      setData(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : String(error)
      }));
    }
  };

  return { ...data, refreshData };
}

export default useMexcData;