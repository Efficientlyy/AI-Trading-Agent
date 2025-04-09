import { createAuthenticatedClient } from './client';
import { Asset, OHLCV, HistoricalDataRequest } from '../types';

export const marketApi = {
  getAssets: async (): Promise<{ assets: Asset[] }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ assets: Asset[] }>('/assets');
    return response.data;
  },
  
  getHistoricalData: async (request: HistoricalDataRequest): Promise<{ data: OHLCV[] }> => {
    const client = createAuthenticatedClient();
    const { symbol, start, end, timeframe } = request;
    const response = await client.get<{ data: OHLCV[] }>('/history', {
      params: { symbol, start, end, timeframe }
    });
    return response.data;
  },
  
  getSentiment: async (symbol?: string): Promise<{ sentiment: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ sentiment: string }>('/sentiment', {
      params: { symbol }
    });
    return response.data;
  },
};
