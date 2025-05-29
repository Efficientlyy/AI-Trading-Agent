import axios from 'axios';

// Base URLs for MEXC API
const MEXC_REST_BASE_URL = 'https://api.mexc.com';
const MEXC_WS_BASE_URL = 'wss://wbs.mexc.com/ws';

// API keys (only needed for private endpoints, which we're not using for the dashboard)
const API_KEY = 'mx0vglFrxgsCpIXJo9';
const API_SECRET = '70497df5aacc47c6be0c7114b921d4c1';

// Types for API responses
export interface MexcKline {
  s: string;      // Symbol
  c: number;      // Close price
  h: number;      // High price
  i: string;      // Interval
  l: number;      // Low price
  o: number;      // Open price
  t: number;      // Timestamp
  v: number;      // Volume
}

export interface MexcTicker {
  symbol: string;
  lastPrice: string;
  priceChange: string;
  priceChangePercent: string;
  volume: string;
  quoteVolume: string;
  high: string;
  low: string;
}

export interface MexcOrderBookItem {
  price: string;
  quantity: string;
}

export interface MexcOrderBook {
  symbol: string;
  bids: [string, string][];  // [price, quantity]
  asks: [string, string][];  // [price, quantity]
  timestamp: number;
}

export interface MexcTrade {
  id: number;
  price: string;
  qty: string;
  time: number;
  isBuyerMaker: boolean;
}

// MEXC API Service
class MexcService {
  // Get market data for a specific symbol
  async getSymbolData(symbol: string) {
    try {
      // Format symbol correctly (remove / if present)
      const formattedSymbol = symbol.replace('/', '');
      const response = await axios.get(`${MEXC_REST_BASE_URL}/api/v3/ticker/24hr`, {
        params: { symbol: formattedSymbol }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching symbol data:', error);
      throw error;
    }
  }

  // Get ticker data for all symbols or a specific one
  async getTickers(symbol?: string) {
    try {
      const params: Record<string, any> = {};
      if (symbol) {
        params.symbol = symbol.replace('/', '');
      }
      const response = await axios.get(`${MEXC_REST_BASE_URL}/api/v3/ticker/24hr`, { params });
      return response.data;
    } catch (error) {
      console.error('Error fetching tickers:', error);
      throw error;
    }
  }

  // Get order book for a symbol
  async getOrderBook(symbol: string, limit: number = 20) {
    try {
      const response = await axios.get(`${MEXC_REST_BASE_URL}/api/v3/depth`, {
        params: {
          symbol: symbol.replace('/', ''),
          limit: limit
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching order book:', error);
      throw error;
    }
  }

  // Get recent trades
  async getRecentTrades(symbol: string, limit: number = 20) {
    try {
      const response = await axios.get(`${MEXC_REST_BASE_URL}/api/v3/trades`, {
        params: {
          symbol: symbol.replace('/', ''),
          limit: limit
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching recent trades:', error);
      throw error;
    }
  }

  // Get candlestick data
  async getKlines(symbol: string, interval: string, limit: number = 100) {
    try {
      const response = await axios.get(`${MEXC_REST_BASE_URL}/api/v3/klines`, {
        params: {
          symbol: symbol.replace('/', ''),
          interval: interval,
          limit: limit
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching klines:', error);
      throw error;
    }
  }

  // Format raw kline data to a more usable format
  formatKlineData(data: any[]): any[] {
    return data.map(item => ({
      time: item[0],
      open: parseFloat(item[1]),
      high: parseFloat(item[2]),
      low: parseFloat(item[3]),
      close: parseFloat(item[4]),
      volume: parseFloat(item[5]),
    }));
  }
}

export const mexcService = new MexcService();
export default mexcService;