import { createAuthenticatedClient } from './client';
import { Asset, OHLCV, HistoricalDataRequest } from '../types';

export const marketApi = {
  getAssets: async (): Promise<{ assets: Asset[] }> => {
    // MOCKED asset data for offline/dev use
    return {
      assets: [
        { symbol: 'BTC/USD', name: 'Bitcoin', type: 'crypto', price: 48000, change_24h: 2.1, volume_24h: 10000 },
        { symbol: 'ETH/USD', name: 'Ethereum', type: 'crypto', price: 3500, change_24h: 1.5, volume_24h: 8000 },
        { symbol: 'AAPL', name: 'Apple Inc.', type: 'stock', price: 180, change_24h: 0.8, volume_24h: 12000 },
        { symbol: 'TSLA', name: 'Tesla Inc.', type: 'stock', price: 700, change_24h: -1.2, volume_24h: 9000 },
        { symbol: 'MSFT', name: 'Microsoft Corp.', type: 'stock', price: 320, change_24h: 0.3, volume_24h: 11000 },
      ]
    };
  },
  
  getHistoricalData: async (request: HistoricalDataRequest): Promise<{ data: OHLCV[] }> => {
    // MOCKED OHLCV data for offline/dev use
    // Generates 30 days of fake daily candles for any symbol
    const now = Date.now();
    const dayMs = 24 * 60 * 60 * 1000;
    // Ensure at least some price movement for SMA to work
    let prevClose = 40000 + Math.floor(Math.random() * 10000);
    const data: OHLCV[] = Array.from({ length: 30 }).map((_, i) => {
      // Simulate a realistic walk so SMA is not flat/null
      const drift = Math.floor((Math.random() - 0.5) * 500);
      const open = prevClose + drift;
      const close = open + Math.floor((Math.random() - 0.5) * 1000);
      const high = Math.max(open, close) + Math.floor(Math.random() * 200);
      const low = Math.min(open, close) - Math.floor(Math.random() * 200);
      const volume = 10 + Math.floor(Math.random() * 5);
      prevClose = close;
      return {
        timestamp: new Date(now - (29 - i) * dayMs).toISOString(),
        open,
        high,
        low,
        close,
        volume,
      };
    });
    return { data };
  },
  
  getSentiment: async (symbol?: string): Promise<{ sentiment: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ sentiment: string }>('/sentiment', {
      params: { symbol }
    });
    return response.data;
  },
  
  // Get historical prices for a symbol
  getHistoricalPrices: async (symbol: string, timeframe: string = '1d', limit: number = 100): Promise<{ data: OHLCV[] }> => {
    // This function uses the same mock data generator as getHistoricalData
    // but with a different interface to match existing code
    return marketApi.getHistoricalData({
      symbol,
      timeframe,
      limit
    });
  },
};
