import { OHLCV } from '../../types';

// Generate random price data with a trend
const generateMockOHLCVData = (
  symbol: string,
  days: number,
  startPrice: number,
  volatility: number,
  trend: number
): OHLCV[] => {
  const data: OHLCV[] = [];
  let currentPrice = startPrice;
  const now = new Date();
  
  for (let i = days; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    date.setHours(0, 0, 0, 0);
    
    // Add some randomness with trend
    const changePercent = (Math.random() - 0.5) * volatility + trend;
    const change = currentPrice * (changePercent / 100);
    
    const open = currentPrice;
    const close = open + change;
    
    // High and low are random values between open and close, with some extra range
    const max = Math.max(open, close);
    const min = Math.min(open, close);
    const range = Math.abs(close - open) + (currentPrice * volatility / 200);
    
    const high = max + (Math.random() * range);
    const low = min - (Math.random() * range);
    
    // Volume is random but correlated with price change
    const volume = 1000000 + Math.abs(change) * 10000 * (0.5 + Math.random());
    
    data.push({
      timestamp: date.toISOString(),
      open,
      high,
      low,
      close,
      volume,
    });
    
    currentPrice = close;
  }
  
  return data;
};

// Generate mock data for different timeframes
const generateTimeframeData = (
  symbol: string,
  timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w'
): OHLCV[] => {
  // Configure parameters based on timeframe
  let dataPoints: number;
  let volatility: number;
  let trend: number;
  
  switch (timeframe) {
    case '1m':
      dataPoints = 24 * 60; // 1 day of minute data
      volatility = 0.1;
      trend = 0.01;
      break;
    case '5m':
      dataPoints = 3 * 24 * 12; // 3 days of 5-minute data
      volatility = 0.2;
      trend = 0.02;
      break;
    case '15m':
      dataPoints = 7 * 24 * 4; // 7 days of 15-minute data
      volatility = 0.3;
      trend = 0.03;
      break;
    case '30m':
      dataPoints = 14 * 24 * 2; // 14 days of 30-minute data
      volatility = 0.4;
      trend = 0.04;
      break;
    case '1h':
      dataPoints = 30 * 24; // 30 days of hourly data
      volatility = 0.5;
      trend = 0.05;
      break;
    case '4h':
      dataPoints = 60 * 6; // 60 days of 4-hour data
      volatility = 0.8;
      trend = 0.08;
      break;
    case '1d':
      dataPoints = 180; // 180 days of daily data
      volatility = 1.2;
      trend = 0.1;
      break;
    case '1w':
      dataPoints = 52; // 52 weeks of weekly data
      volatility = 2.0;
      trend = 0.2;
      break;
    default:
      dataPoints = 180;
      volatility = 1.2;
      trend = 0.1;
  }
  
  // Starting prices for different symbols
  const startPrices: Record<string, number> = {
    'BTC': 35000,
    'ETH': 2200,
    'AAPL': 180,
    'MSFT': 350,
    'GOOGL': 140,
    'AMZN': 130,
    'TSLA': 220,
    'NVDA': 800,
    'META': 450,
    'JPM': 180,
  };
  
  // Trends for different symbols (percentage)
  const trends: Record<string, number> = {
    'BTC': 0.15,
    'ETH': 0.12,
    'AAPL': 0.05,
    'MSFT': 0.08,
    'GOOGL': 0.06,
    'AMZN': 0.07,
    'TSLA': -0.03,
    'NVDA': 0.10,
    'META': 0.04,
    'JPM': 0.02,
  };
  
  const startPrice = startPrices[symbol] || 100;
  const symbolTrend = trends[symbol] || trend;
  
  return generateMockOHLCVData(symbol, dataPoints, startPrice, volatility, symbolTrend);
};

// Mock API function to get historical data
export const getMockHistoricalData = (
  symbol: string,
  timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' = '1d'
): Promise<OHLCV[]> => {
  return new Promise((resolve) => {
    // Simulate API delay
    setTimeout(() => {
      resolve(generateTimeframeData(symbol, timeframe));
    }, 500);
  });
};

// Export a function to get data for multiple symbols
export const getMockHistoricalDataForSymbols = (
  symbols: string[],
  timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' = '1d'
): Promise<Record<string, OHLCV[]>> => {
  return new Promise((resolve) => {
    const result: Record<string, OHLCV[]> = {};
    
    // Simulate API delay
    setTimeout(() => {
      symbols.forEach((symbol) => {
        result[symbol] = generateTimeframeData(symbol, timeframe);
      });
      resolve(result);
    }, 800);
  });
};
