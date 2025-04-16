// mockPerformanceMetrics.ts
export const getMockPerformanceMetrics = async () => {
  return {
    performance: {
      total_return: 0.18,
      sharpe_ratio: 1.45,
      max_drawdown: -8.2,
      win_rate: 62,
      profit_factor: 1.9,
      avg_trade: 220,
    }
  };
};
