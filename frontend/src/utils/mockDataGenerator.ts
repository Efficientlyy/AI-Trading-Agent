/**
 * Mock data generator for the MEXC dashboard
 * This provides consistent yet realistic-looking data for testing without the API load
 */

import { MexcTicker, MexcOrderBook, MexcTrade } from '../api/mexcService';

/**
 * Generate a mock ticker with realistic price movements
 */
export const generateMockTicker = (basePrice?: number): MexcTicker => {
  const base = basePrice || 30000 + Math.random() * 2000;
  const change = (Math.random() * 2 - 1) * 2; // Random change between -2% and 2%
  
  return {
    symbol: 'BTC_USDT',
    lastPrice: base.toFixed(2),
    priceChange: (base * change / 100).toFixed(2),
    priceChangePercent: change.toFixed(2),
    volume: (Math.random() * 1000 + 500).toFixed(2),
    quoteVolume: (Math.random() * 10000000 + 5000000).toFixed(2),
    high: (base * (1 + Math.random() * 0.03)).toFixed(2),
    low: (base * (1 - Math.random() * 0.03)).toFixed(2)
  };
};

/**
 * Generate a mock order book with realistic depth
 */
export const generateMockOrderBook = (basePrice?: number): MexcOrderBook => {
  const base = basePrice || 30000;
  const bids: [string, string][] = [];
  const asks: [string, string][] = [];
  
  // Generate bids (buy orders) - slightly below current price
  for (let i = 0; i < 20; i++) {
    const price = (base * (1 - 0.0001 * (i + 1))).toFixed(2);
    const quantity = (Math.random() * 2 + 0.1).toFixed(6);
    bids.push([price, quantity]);
  }
  
  // Generate asks (sell orders) - slightly above current price
  for (let i = 0; i < 20; i++) {
    const price = (base * (1 + 0.0001 * (i + 1))).toFixed(2);
    const quantity = (Math.random() * 2 + 0.1).toFixed(6);
    asks.push([price, quantity]);
  }
  
  return {
    symbol: 'BTC_USDT',
    bids,
    asks,
    timestamp: Date.now()
  };
};

/**
 * Generate mock trades with realistic patterns
 */
export const generateMockTrades = (count: number = 30, basePrice?: number): MexcTrade[] => {
  const base = basePrice || 30000;
  const trades: MexcTrade[] = [];
  
  for (let i = 0; i < count; i++) {
    // Random price around base price with small variations
    const price = (base * (1 + (Math.random() * 0.002 - 0.001))).toFixed(2);
    
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

/**
 * Generate mock kline (candlestick) data
 */
export const generateMockKlineData = (count: number = 100, timeframe: string = '1h', basePrice?: number): any[] => {
  const base = basePrice || 30000;
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
  
  let currentPrice = base;
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
