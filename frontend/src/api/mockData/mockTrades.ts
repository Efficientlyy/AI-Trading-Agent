// mockTrades.ts
import { OrderSide, Trade } from '../../types';

// Export this function to fix the import errors in Dashboard.tsx and other files
export const getMockTrades = async (): Promise<{ trades: Trade[] }> => {
  return mockTradeApi.getRecentTrades();
};

export const mockTradeApi = {
  getRecentTrades: async (): Promise<{ trades: Trade[] }> => {
    return {
      trades: [
        {
          id: '1',
          symbol: 'AAPL',
          side: OrderSide.BUY,
          quantity: 10,
          price: 180.34,
          status: 'filled',
          timestamp: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
          realized_pnl: 50,
        },
        {
          id: '2',
          symbol: 'MSFT',
          side: OrderSide.SELL,
          quantity: 5,
          price: 321.12,
          status: 'partial',
          timestamp: new Date(Date.now() - 1000 * 60 * 120).toISOString(),
          realized_pnl: 20,
        },
        {
          id: '3',
          symbol: 'TSLA',
          side: OrderSide.BUY,
          quantity: 3,
          price: 702.50,
          status: 'pending',
          timestamp: new Date(Date.now() - 1000 * 60 * 180).toISOString(),
          realized_pnl: 30,
        },
        {
          id: '4',
          symbol: 'GOOG',
          side: OrderSide.BUY,
          quantity: 2,
          price: 2800.10,
          status: 'cancelled',
          timestamp: new Date(Date.now() - 1000 * 60 * 240).toISOString(),
          realized_pnl: 0,
        },
        {
          id: '5',
          symbol: 'AMZN',
          side: OrderSide.SELL,
          quantity: 1,
          price: 3400.00,
          status: 'filled',
          timestamp: new Date(Date.now() - 1000 * 60 * 300).toISOString(),
          realized_pnl: 100,
        },
      ],
    };
  },
};
