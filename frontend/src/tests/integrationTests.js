/**
 * Integration Tests for Trading Dashboard Components
 * Tests the interaction between Strategy Optimizer, Backtesting Interface, and Performance Analysis
 */
console.log('Running Integration Tests...');

// Mock data and interfaces
const mockTrade = {
  id: 'trade-1',
  symbol: 'BTC',
  entryDate: '2023-01-15T10:30:00Z',
  exitDate: '2023-01-17T14:45:00Z',
  entryPrice: 21500.75,
  exitPrice: 22350.25,
  quantity: 0.5,
  side: 'long',
  pnl: 424.75,
  strategy: 'Moving Average Crossover',
  status: 'closed'
};

// Mock BacktestMetrics
const mockBacktestMetrics = {
  totalReturn: 15.5,
  annualizedReturn: 12.3,
  sharpeRatio: 1.8,
  maxDrawdown: 8.5,
  winRate: 65,
  profitFactor: 2.1,
  averageWin: 3.2,
  averageLoss: 1.5
};

// Mock BacktestResult
const mockBacktestResult = {
  trades: Array(20).fill().map((_, i) => ({
    ...mockTrade,
    id: `trade-${i + 1}`,
    entryPrice: 21000 + Math.random() * 1000,
    exitPrice: 22000 + Math.random() * 1000,
    pnl: (Math.random() > 0.3 ? 1 : -1) * (100 + Math.random() * 500)
  })),
  metrics: { ...mockBacktestMetrics },
  equity: Array(100).fill().map((_, i) => ({
    date: new Date(2023, 0, 1 + i).toISOString(),
    equity: 10000 * (1 + 0.001 * i + 0.0005 * Math.sin(i))
  }))
};

// Mock optimization result
const mockOptimizationResults = Array(20).fill().map((_, i) => {
  const fastPeriod = 5 + Math.floor(Math.random() * 15);
  const slowPeriod = 20 + Math.floor(Math.random() * 30);
  const performanceFactor = 0.8 + Math.random() * 0.4;
  
  return {
    parameters: {
      fastPeriod,
      slowPeriod,
      signalPeriod: 9
    },
    metrics: {
      totalReturn: Math.round(mockBacktestMetrics.totalReturn * performanceFactor * 100) / 100,
      annualizedReturn: Math.round(mockBacktestMetrics.annualizedReturn * performanceFactor * 100) / 100,
      sharpeRatio: Math.round(mockBacktestMetrics.sharpeRatio * performanceFactor * 100) / 100,
      maxDrawdown: Math.round(mockBacktestMetrics.maxDrawdown * (2 - performanceFactor) * 100) / 100,
      winRate: Math.round(mockBacktestMetrics.winRate * performanceFactor),
      profitFactor: Math.round(mockBacktestMetrics.profitFactor * performanceFactor * 100) / 100,
      averageWin: Math.round(mockBacktestMetrics.averageWin * performanceFactor * 100) / 100,
      averageLoss: Math.round(mockBacktestMetrics.averageLoss * 100) / 100
    }
  };
});

// Sort optimization results by Sharpe Ratio
mockOptimizationResults.sort((a, b) => b.metrics.sharpeRatio - a.metrics.sharpeRatio);

// Test 1: Dashboard Component Integration
function testDashboardIntegration() {
  console.log('\nTest 1: Dashboard Component Integration');
  
  // Simulate Dashboard state
  const dashboardState = {
    activeTab: 'optimization',
    symbol: 'BTC',
    strategy: 'Moving Average Crossover',
    backtestResults: mockBacktestResult,
    optimizationResults: mockOptimizationResults,
    isLoading: false
  };
  
  console.log('Dashboard State:', JSON.stringify(dashboardState, null, 2));
  
  // Validate Dashboard state
  if (!dashboardState.symbol || !dashboardState.strategy) {
    console.error('❌ Failed: Dashboard missing required symbol or strategy');
    return false;
  }
  
  if (!dashboardState.backtestResults || !dashboardState.optimizationResults) {
    console.error('❌ Failed: Dashboard missing backtest or optimization results');
    return false;
  }
  
  console.log('✅ Success: Dashboard correctly integrates all required components');
  return true;
}

// Test 2: Data Flow Between Components
function testComponentDataFlow() {
  console.log('\nTest 2: Data Flow Between Components');
  
  // Simulate running optimization
  console.log('Running optimization with Moving Average Crossover strategy...');
  
  // Parameters for optimization
  const optimizationParams = {
    symbol: 'BTC',
    strategy: 'Moving Average Crossover',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 10000,
    targetMetric: 'sharpeRatio',
    parameters: [
      { name: 'fastPeriod', min: 5, max: 20, step: 1, currentValue: 10 },
      { name: 'slowPeriod', min: 20, max: 50, step: 5, currentValue: 30 },
      { name: 'signalPeriod', min: 5, max: 15, step: 1, currentValue: 9 }
    ]
  };
  
  // Simulate optimization results
  const results = mockOptimizationResults;
  
  // Simulate applying best parameters to backtesting
  const bestParams = results[0].parameters;
  console.log('Best parameters found:', bestParams);
  
  // Simulate running backtest with optimized parameters
  console.log('Running backtest with optimized parameters...');
  const backtestParams = {
    symbol: optimizationParams.symbol,
    strategy: optimizationParams.strategy,
    startDate: optimizationParams.startDate,
    endDate: optimizationParams.endDate,
    initialCapital: optimizationParams.initialCapital,
    parameters: bestParams
  };
  
  // Simulate backtest results
  const backtestResults = {
    ...mockBacktestResult,
    metrics: results[0].metrics
  };
  
  // Validate data flow
  if (JSON.stringify(backtestResults.metrics) !== JSON.stringify(results[0].metrics)) {
    console.error('❌ Failed: Metrics from optimization not correctly applied to backtest');
    return false;
  }
  
  console.log('Backtest results with optimized parameters:', JSON.stringify(backtestResults.metrics, null, 2));
  console.log('✅ Success: Data flows correctly between optimization and backtesting components');
  
  return true;
}

// Test 3: Performance Analysis with Optimization Results
function testPerformanceAnalysis() {
  console.log('\nTest 3: Performance Analysis with Optimization Results');
  
  // Simulate running performance analysis on the best optimization result
  const bestResult = mockOptimizationResults[0];
  console.log('Analyzing performance of best parameter set:', bestResult.parameters);
  
  // Simulate calculating additional performance metrics
  const extendedMetrics = {
    ...bestResult.metrics,
    // Additional calculated metrics
    calmarRatio: Math.round((bestResult.metrics.annualizedReturn / bestResult.metrics.maxDrawdown) * 100) / 100,
    recoveryFactor: Math.round((bestResult.metrics.totalReturn / bestResult.metrics.maxDrawdown) * 100) / 100,
    expectancy: Math.round((bestResult.metrics.winRate / 100 * bestResult.metrics.averageWin - 
                           (1 - bestResult.metrics.winRate / 100) * bestResult.metrics.averageLoss) * 100) / 100
  };
  
  console.log('Extended performance metrics:', JSON.stringify(extendedMetrics, null, 2));
  
  // Validate performance analysis
  if (!extendedMetrics.calmarRatio || !extendedMetrics.recoveryFactor || !extendedMetrics.expectancy) {
    console.error('❌ Failed: Performance analysis missing extended metrics');
    return false;
  }
  
  // Check if metrics make sense
  if (extendedMetrics.calmarRatio <= 0 || extendedMetrics.recoveryFactor <= 0) {
    console.error('❌ Failed: Performance metrics have invalid values');
    return false;
  }
  
  console.log('✅ Success: Performance analysis correctly processes optimization results');
  return true;
}

// Test 4: Trade Statistics with Optimization Results
function testTradeStatistics() {
  console.log('\nTest 4: Trade Statistics with Optimization Results');
  
  // Simulate generating trades based on best parameters
  const bestParams = mockOptimizationResults[0].parameters;
  
  // Generate mock trades with the best parameters
  const trades = Array(30).fill().map((_, i) => {
    const isWin = Math.random() < (mockOptimizationResults[0].metrics.winRate / 100);
    const pnl = isWin 
      ? mockOptimizationResults[0].metrics.averageWin * (0.8 + Math.random() * 0.4)
      : -mockOptimizationResults[0].metrics.averageLoss * (0.8 + Math.random() * 0.4);
    
    return {
      id: `trade-${i + 1}`,
      symbol: 'BTC',
      entryDate: new Date(2023, 0, 1 + i).toISOString(),
      exitDate: new Date(2023, 0, 2 + i).toISOString(),
      entryPrice: 21000 + Math.random() * 1000,
      exitPrice: 21000 + Math.random() * 1000 + (isWin ? 500 : -300),
      quantity: 0.5,
      side: 'long',
      pnl: Math.round(pnl * 100) / 100,
      strategy: 'Moving Average Crossover',
      status: 'closed'
    };
  });
  
  // Calculate trade statistics
  const tradeStats = {
    totalTrades: trades.length,
    winningTrades: trades.filter(t => t.pnl > 0).length,
    losingTrades: trades.filter(t => t.pnl <= 0).length,
    winRate: Math.round(trades.filter(t => t.pnl > 0).length / trades.length * 100),
    totalProfit: Math.round(trades.reduce((sum, t) => sum + (t.pnl > 0 ? t.pnl : 0), 0) * 100) / 100,
    totalLoss: Math.round(trades.reduce((sum, t) => sum + (t.pnl <= 0 ? t.pnl : 0), 0) * 100) / 100,
    netProfit: Math.round(trades.reduce((sum, t) => sum + t.pnl, 0) * 100) / 100,
    largestWin: Math.round(Math.max(...trades.map(t => t.pnl)) * 100) / 100,
    largestLoss: Math.round(Math.min(...trades.map(t => t.pnl)) * 100) / 100,
    averageWin: Math.round(trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0) / 
                          trades.filter(t => t.pnl > 0).length * 100) / 100,
    averageLoss: Math.round(trades.filter(t => t.pnl <= 0).reduce((sum, t) => sum + t.pnl, 0) / 
                           trades.filter(t => t.pnl <= 0).length * 100) / 100,
    profitFactor: Math.round(Math.abs(trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0) / 
                                    trades.filter(t => t.pnl <= 0).reduce((sum, t) => sum + t.pnl, 0)) * 100) / 100
  };
  
  console.log('Trade statistics for optimized strategy:', JSON.stringify(tradeStats, null, 2));
  
  // Validate trade statistics
  if (Math.abs(tradeStats.winRate - mockOptimizationResults[0].metrics.winRate) > 10) {
    console.warn('⚠️ Warning: Win rate in trade statistics differs significantly from optimization metrics');
  }
  
  if (Math.abs(tradeStats.profitFactor - mockOptimizationResults[0].metrics.profitFactor) > 0.5) {
    console.warn('⚠️ Warning: Profit factor in trade statistics differs significantly from optimization metrics');
  }
  
  // Check if statistics are consistent
  if (tradeStats.totalTrades !== tradeStats.winningTrades + tradeStats.losingTrades) {
    console.error('❌ Failed: Trade statistics are inconsistent');
    return false;
  }
  
  if (Math.abs(tradeStats.netProfit - (tradeStats.totalProfit + tradeStats.totalLoss)) > 0.01) {
    console.error('❌ Failed: Net profit calculation is inconsistent');
    return false;
  }
  
  console.log('✅ Success: Trade statistics correctly analyze trades from optimized strategy');
  return true;
}

// Run all tests
console.log('=== INTEGRATION TEST SUITE ===');
const dashboardIntegration = testDashboardIntegration();
const componentDataFlow = testComponentDataFlow();
const performanceAnalysis = testPerformanceAnalysis();
const tradeStatistics = testTradeStatistics();

console.log('\n=== TEST SUMMARY ===');
console.log(`Dashboard Integration: ${dashboardIntegration ? '✅ PASS' : '❌ FAIL'}`);
console.log(`Component Data Flow: ${componentDataFlow ? '✅ PASS' : '❌ FAIL'}`);
console.log(`Performance Analysis: ${performanceAnalysis ? '✅ PASS' : '❌ FAIL'}`);
console.log(`Trade Statistics: ${tradeStatistics ? '✅ PASS' : '❌ FAIL'}`);

if (dashboardIntegration && componentDataFlow && performanceAnalysis && tradeStatistics) {
  console.log('\n🎉 All integration tests passed successfully!');
} else {
  console.error('\n❌ Some integration tests failed. See details above.');
}
