// mockTrades.ts
import { Trade } from '../../types';

export const getMockTrades = async (): Promise<{ trades: Trade[] }> => {
  return {
    trades: [
      {
        id: '1',
        symbol: 'AAPL',
        side: 'buy',
        quantity: 10,
        price: 180.34,
        status: 'filled',
        timestamp: Date.now() - 1000 * 60 * 60,
        realized_pnl: 50,
      },
      {
        id: '2',
        symbol: 'MSFT',
        side: 'sell',
        quantity: 5,
        price: 321.12,
        status: 'partial',
        timestamp: Date.now() - 1000 * 60 * 120,
        realized_pnl: 20,
      },
      {
        id: '3',
        symbol: 'TSLA',
        side: 'buy',
        quantity: 3,
        price: 702.50,
        status: 'pending',
        timestamp: Date.now() - 1000 * 60 * 180,
        realized_pnl: 30,
      },
      {
        id: '4',
        symbol: 'GOOG',
        side: 'buy',
        quantity: 2,
        price: 2800.10,
        status: 'cancelled',
        timestamp: Date.now() - 1000 * 60 * 240,
        realized_pnl: 0,
      },
      {
        id: '5',
        symbol: 'AMZN',
        side: 'sell',
        quantity: 1,
        price: 3400.00,
        status: 'filled',
        timestamp: Date.now() - 1000 * 60 * 300,
        realized_pnl: 100,
      },
    ],
  };
};
