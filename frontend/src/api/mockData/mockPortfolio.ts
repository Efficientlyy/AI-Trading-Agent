// mockPortfolio.ts
import { Portfolio } from '../../types';

export const getMockPortfolio = async (): Promise<{ portfolio: Portfolio }> => {
  // Example mock data structure with all required Position fields
  return {
    portfolio: {
      total_value: 100000,
      cash: 25000,
      daily_pnl: 350,
      margin_multiplier: 2,
      positions: {
        AAPL: {
          symbol: 'AAPL',
          quantity: 10,
          market_value: 1800,
          entry_price: 170,
          current_price: 180,
          unrealized_pnl: 100,
          realized_pnl: 50
        },
        MSFT: {
          symbol: 'MSFT',
          quantity: 5,
          market_value: 1600,
          entry_price: 300,
          current_price: 320,
          unrealized_pnl: 100,
          realized_pnl: 20
        },
        TSLA: {
          symbol: 'TSLA',
          quantity: 3,
          market_value: 2100,
          entry_price: 650,
          current_price: 700,
          unrealized_pnl: 150,
          realized_pnl: 30
        },
      },
    },
  };
};
