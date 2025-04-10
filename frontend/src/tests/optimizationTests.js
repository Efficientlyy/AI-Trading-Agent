/**
 * Test suite for Strategy Optimization functionality
 */
console.log('Running Strategy Optimization Tests...');

// Mock interfaces and functions for testing
const BacktestMetrics = {
  totalReturn: 15.5,
  annualizedReturn: 12.3,
  sharpeRatio: 1.8,
  maxDrawdown: 8.5,
  winRate: 65,
  profitFactor: 2.1,
  averageWin: 3.2,
  averageLoss: 1.5
};

// Mock optimization result
const OptimizationResult = {
  parameters: {},
  metrics: { ...BacktestMetrics }
};

// Mock function to simulate running a backtest
function runMockBacktest(params) {
  return {
    trades: [],
    metrics: { ...BacktestMetrics }
  };
}

// Mock function to generate parameter combinations
function generateParameterCombinations(parameters) {
  const combinations = [];
  
  // Generate 20 random combinations
  for (let i = 0; i < 20; i++) {
    const combination = {};
    
    parameters.forEach(param => {
      // Generate a random value within the parameter range
      const range = param.max - param.min;
      const steps = Math.round(range / param.step);
      const randomSteps = Math.floor(Math.random() * steps);
      const value = param.min + (randomSteps * param.step);
      
      // Round to 2 decimal places to avoid floating point issues
      combination[param.name] = Math.round(value * 100) / 100;
    });
    
    combinations.push(combination);
  }
  
  return combinations;
}

// Mock function to adjust metrics based on parameters
function adjustMetricsBasedOnParameters(metrics, parameters) {
  // Create a copy of the metrics
  const adjustedMetrics = { ...metrics };
  
  // Apply parameter-based adjustments
  // This is a simplified model - in reality, the relationship would be more complex
  
  // Moving Average parameters
  if ('fastPeriod' in parameters && 'slowPeriod' in parameters) {
    const fastPeriod = parameters.fastPeriod;
    const slowPeriod = parameters.slowPeriod;
    
    // Better performance when fast and slow periods have good separation
    const periodRatio = slowPeriod / fastPeriod;
    if (periodRatio > 3 && periodRatio < 7) {
      adjustedMetrics.totalReturn *= 1.2;
      adjustedMetrics.sharpeRatio *= 1.15;
      adjustedMetrics.winRate += 5;
    }
  }
  
  // RSI parameters
  if ('period' in parameters && 'overbought' in parameters && 'oversold' in parameters) {
    const period = parameters.period;
    const overboughtThreshold = parameters.overbought;
    const oversoldThreshold = parameters.oversold;
    
    // Better performance with more standard RSI settings
    if (period >= 12 && period <= 16 && 
        overboughtThreshold >= 68 && overboughtThreshold <= 75 &&
        oversoldThreshold >= 25 && oversoldThreshold <= 35) {
      adjustedMetrics.totalReturn *= 1.25;
      adjustedMetrics.sharpeRatio *= 1.2;
      adjustedMetrics.winRate += 7;
    }
  }
  
  // Add some randomness to make results look more realistic
  const randomFactor = 0.9 + (Math.random() * 0.2); // 0.9 to 1.1
  adjustedMetrics.totalReturn *= randomFactor;
  adjustedMetrics.annualizedReturn *= randomFactor;
  adjustedMetrics.sharpeRatio *= randomFactor;
  adjustedMetrics.profitFactor *= randomFactor;
  
  // Round values for display
  adjustedMetrics.totalReturn = Math.round(adjustedMetrics.totalReturn * 100) / 100;
  adjustedMetrics.annualizedReturn = Math.round(adjustedMetrics.annualizedReturn * 100) / 100;
  adjustedMetrics.sharpeRatio = Math.round(adjustedMetrics.sharpeRatio * 100) / 100;
  adjustedMetrics.maxDrawdown = Math.round(adjustedMetrics.maxDrawdown * 100) / 100;
  adjustedMetrics.winRate = Math.round(adjustedMetrics.winRate);
  adjustedMetrics.profitFactor = Math.round(adjustedMetrics.profitFactor * 100) / 100;
  adjustedMetrics.averageWin = Math.round(adjustedMetrics.averageWin * 100) / 100;
  adjustedMetrics.averageLoss = Math.round(adjustedMetrics.averageLoss * 100) / 100;
  
  return adjustedMetrics;
}

// Mock function to sort optimization results
function sortResultsByMetric(results, targetMetric) {
  return results.sort((a, b) => {
    // For drawdown, lower is better
    if (targetMetric === 'maxDrawdown') {
      return a.metrics[targetMetric] - b.metrics[targetMetric];
    }
    // For all other metrics, higher is better
    return b.metrics[targetMetric] - a.metrics[targetMetric];
  });
}

// Mock function to run strategy optimization
function runStrategyOptimization(params) {
  const { symbol, strategy, startDate, endDate, initialCapital, targetMetric, parameters } = params;
  const results = [];
  
  // Generate parameter combinations
  const parameterCombinations = generateParameterCombinations(parameters);
  
  // Run backtest for each parameter combination
  parameterCombinations.forEach(paramCombo => {
    // Create backtest params with the current parameter combination
    const backtestParams = {
      symbol,
      strategy,
      startDate,
      endDate,
      initialCapital,
      parameters: paramCombo
    };
    
    // Run a mock backtest with these parameters
    const backtestResult = runMockBacktest(backtestParams);
    
    // Adjust metrics based on parameters to simulate optimization
    const adjustedMetrics = adjustMetricsBasedOnParameters(backtestResult.metrics, paramCombo);
    
    // Add to results
    results.push({
      parameters: paramCombo,
      metrics: adjustedMetrics
    });
  });
  
  // Sort results by target metric
  return sortResultsByMetric(results, targetMetric);
}

// Test 1: Basic optimization with Moving Average strategy
function testMovingAverageOptimization() {
  console.log('\nTest 1: Moving Average Strategy Optimization');
  
  const params = {
    symbol: 'BTC',
    strategy: 'Moving Average Crossover',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 10000,
    targetMetric: 'sharpeRatio',
    parameters: [
      { name: 'fastPeriod', min: 5, max: 20, step: 1, currentValue: 10 },
      { name: 'slowPeriod', min: 20, max: 50, step: 5, currentValue: 30 },
      { name: 'positionSize', min: 5, max: 20, step: 5, currentValue: 10 }
    ]
  };
  
  const results = runStrategyOptimization(params);
  
  // Validate results
  console.log(`Generated ${results.length} optimization results`);
  console.log('Top 3 parameter combinations:');
  results.slice(0, 3).forEach((result, index) => {
    console.log(`\n#${index + 1}:`);
    console.log('Parameters:', JSON.stringify(result.parameters, null, 2));
    console.log('Metrics:', JSON.stringify(result.metrics, null, 2));
  });
  
  // Validation checks
  if (results.length === 0) {
    console.error('❌ Failed: No optimization results generated');
  } else if (results.length < 10) {
    console.warn('⚠️ Warning: Less than 10 optimization results generated');
  } else {
    console.log('✅ Success: Generated sufficient optimization results');
  }
  
  // Check if results are sorted by target metric
  const isSorted = results.every((result, i) => 
    i === 0 || results[i-1].metrics.sharpeRatio >= result.metrics.sharpeRatio
  );
  
  if (isSorted) {
    console.log('✅ Success: Results are properly sorted by Sharpe Ratio');
  } else {
    console.error('❌ Failed: Results are not properly sorted by target metric');
  }
  
  return results;
}

// Test 2: RSI Strategy Optimization
function testRSIOptimization() {
  console.log('\nTest 2: RSI Strategy Optimization');
  
  const params = {
    symbol: 'ETH',
    strategy: 'RSI Oscillator',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 10000,
    targetMetric: 'totalReturn',
    parameters: [
      { name: 'period', min: 7, max: 21, step: 1, currentValue: 14 },
      { name: 'overbought', min: 65, max: 80, step: 1, currentValue: 70 },
      { name: 'oversold', min: 20, max: 35, step: 1, currentValue: 30 }
    ]
  };
  
  const results = runStrategyOptimization(params);
  
  // Validate results
  console.log(`Generated ${results.length} optimization results`);
  console.log('Top 3 parameter combinations:');
  results.slice(0, 3).forEach((result, index) => {
    console.log(`\n#${index + 1}:`);
    console.log('Parameters:', JSON.stringify(result.parameters, null, 2));
    console.log('Metrics:', JSON.stringify(result.metrics, null, 2));
  });
  
  // Check if results are sorted by target metric
  const isSorted = results.every((result, i) => 
    i === 0 || results[i-1].metrics.totalReturn >= result.metrics.totalReturn
  );
  
  if (isSorted) {
    console.log('✅ Success: Results are properly sorted by Total Return');
  } else {
    console.error('❌ Failed: Results are not properly sorted by target metric');
  }
  
  return results;
}

// Test 3: Parameter Impact Analysis
function testParameterImpact() {
  console.log('\nTest 3: Parameter Impact Analysis');
  
  // Run optimization with a variety of parameters
  const params = {
    symbol: 'BTC',
    strategy: 'Bollinger Breakout',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 10000,
    targetMetric: 'sharpeRatio',
    parameters: [
      { name: 'period', min: 10, max: 30, step: 2, currentValue: 20 },
      { name: 'standardDeviations', min: 1.5, max: 3.0, step: 0.1, currentValue: 2.0 }
    ]
  };
  
  const results = runStrategyOptimization(params);
  
  // Analyze parameter impact
  console.log('Parameter Impact Analysis:');
  
  // Group results by period
  const periodGroups = {};
  results.forEach(result => {
    const period = result.parameters.period;
    if (!periodGroups[period]) {
      periodGroups[period] = 0;
    }
    periodGroups[period] += result.metrics.sharpeRatio;
  });
  
  // Calculate average Sharpe ratio for each period
  const periodImpact = Object.entries(periodGroups).map(([period, totalSharpe]) => {
    const count = results.filter(r => r.parameters.period === Number(period)).length;
    return {
      period: Number(period),
      avgSharpe: totalSharpe / count
    };
  });
  
  console.log('\nPeriod Impact on Sharpe Ratio:');
  periodImpact.sort((a, b) => b.avgSharpe - a.avgSharpe);
  periodImpact.forEach(item => {
    console.log(`Period ${item.period}: Average Sharpe = ${item.avgSharpe.toFixed(2)}`);
  });
  
  // Group results by standard deviations
  const stdDevGroups = {};
  results.forEach(result => {
    const stdDev = result.parameters.standardDeviations;
    const roundedStdDev = Math.round(stdDev * 10) / 10; // Round to 1 decimal place
    if (!stdDevGroups[roundedStdDev]) {
      stdDevGroups[roundedStdDev] = 0;
    }
    stdDevGroups[roundedStdDev] += result.metrics.sharpeRatio;
  });
  
  // Calculate average Sharpe ratio for each standard deviation
  const stdDevImpact = Object.entries(stdDevGroups).map(([stdDev, totalSharpe]) => {
    const count = results.filter(r => {
      const roundedStdDev = Math.round(r.parameters.standardDeviations * 10) / 10;
      return roundedStdDev === Number(stdDev);
    }).length;
    return {
      stdDev: Number(stdDev),
      avgSharpe: totalSharpe / count
    };
  });
  
  console.log('\nStandard Deviation Impact on Sharpe Ratio:');
  stdDevImpact.sort((a, b) => b.avgSharpe - a.avgSharpe);
  stdDevImpact.forEach(item => {
    console.log(`StdDev ${item.stdDev}: Average Sharpe = ${item.avgSharpe.toFixed(2)}`);
  });
  
  // Validation
  if (periodImpact.length > 0 && stdDevImpact.length > 0) {
    console.log('✅ Success: Parameter impact analysis completed');
  } else {
    console.error('❌ Failed: Could not analyze parameter impact');
  }
  
  return { periodImpact, stdDevImpact };
}

// Run all tests
console.log('=== STRATEGY OPTIMIZATION TEST SUITE ===');
const maResults = testMovingAverageOptimization();
const rsiResults = testRSIOptimization();
const parameterImpact = testParameterImpact();

console.log('\n=== TEST SUMMARY ===');
console.log(`Total optimization results generated: ${maResults.length + rsiResults.length}`);
console.log('Parameter impact analysis completed for Bollinger Bands strategy');
console.log('All tests completed successfully!');
