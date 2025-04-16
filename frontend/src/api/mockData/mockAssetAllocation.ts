// mockAssetAllocation.ts
import { Portfolio } from '../../types';

export const getMockAssetAllocation = async (): Promise<{ portfolio: Portfolio }> => {
  // Use the same mock structure as mockPortfolio, but with different values for variety
  return {
    portfolio: {
      total_value: 120000,
      cash: 20000,
      daily_pnl: 400,
      margin_multiplier: 2,
      positions: {
        AAPL: {
          symbol: 'AAPL',
          quantity: 15,
          market_value: 2700,
          entry_price: 170,
          current_price: 180,
          unrealized_pnl: 150,
          realized_pnl: 70
        },
        MSFT: {
          symbol: 'MSFT',
          quantity: 8,
          market_value: 2560,
          entry_price: 300,
          current_price: 320,
          unrealized_pnl: 160,
          realized_pnl: 40
        },
        TSLA: {
          symbol: 'TSLA',
          quantity: 4,
          market_value: 2800,
          entry_price: 650,
          current_price: 700,
          unrealized_pnl: 200,
          realized_pnl: 50
        },
      },
    },
  };
};
