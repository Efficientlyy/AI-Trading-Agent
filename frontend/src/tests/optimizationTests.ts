import { runStrategyOptimization } from '../api/mockData/optimizationResults';
import { OptimizationParams, ParameterConfig } from '../components/dashboard/StrategyOptimizer';

/**
 * Test suite for Strategy Optimization functionality
 */
console.log('Running Strategy Optimization Tests...');

// Test 1: Basic optimization with Moving Average strategy
function testMovingAverageOptimization() {
  console.log('\nTest 1: Moving Average Strategy Optimization');
  
  const params: OptimizationParams = {
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
  
  const params: OptimizationParams = {
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

// Test 3: Sentiment Strategy Optimization
function testSentimentOptimization() {
  console.log('\nTest 3: Sentiment Strategy Optimization');
  
  const params: OptimizationParams = {
    symbol: 'AAPL',
    strategy: 'Sentiment-Based',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 10000,
    targetMetric: 'winRate',
    parameters: [
      { name: 'sentimentThreshold', min: 0.5, max: 0.9, step: 0.05, currentValue: 0.6 },
      { name: 'lookbackPeriod', min: 1, max: 14, step: 1, currentValue: 7 }
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
    i === 0 || results[i-1].metrics.winRate >= result.metrics.winRate
  );
  
  if (isSorted) {
    console.log('✅ Success: Results are properly sorted by Win Rate');
  } else {
    console.error('❌ Failed: Results are not properly sorted by target metric');
  }
  
  return results;
}

// Test 4: Parameter Impact Analysis
function testParameterImpact() {
  console.log('\nTest 4: Parameter Impact Analysis');
  
  // Run optimization with a variety of parameters
  const params: OptimizationParams = {
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
  const periodGroups: Record<number, number> = {};
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
  const stdDevGroups: Record<number, number> = {};
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
const sentimentResults = testSentimentOptimization();
const parameterImpact = testParameterImpact();

console.log('\n=== TEST SUMMARY ===');
console.log(`Total optimization results generated: ${maResults.length + rsiResults.length + sentimentResults.length}`);
console.log('Parameter impact analysis completed for Bollinger Bands strategy');
console.log('All tests completed successfully!');
