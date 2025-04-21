import { Trade } from '../types';

export const tradesApi = {
  getRecentTrades: async (): Promise<{ trades: Trade[] }> => {
    // Always use mock data for trades
    const { getMockTrades } = await import('./mockData/mockTrades');
    return getMockTrades();
  }
};
