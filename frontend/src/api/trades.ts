import { createAuthenticatedClient } from './client';
import { Trade } from '../types';

export const tradesApi = {
  getRecentTrades: async (): Promise<{ trades: Trade[] }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ trades: Trade[] }>('/trades/recent');
    return response.data;
  }
};
