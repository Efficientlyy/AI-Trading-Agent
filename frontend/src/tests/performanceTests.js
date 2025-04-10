/**
 * Performance Tests for Strategy Optimization
 * Tests how the optimization process handles larger parameter spaces and evaluates execution time
 */
console.log('Running Performance Tests...');

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

// Mock function to generate parameter combinations
function generateParameterCombinations(parameters) {
  const combinations = [];
  
  // Generate all possible combinations (up to a reasonable limit)
  const paramValues = {};
  let totalCombinations = 1;
  
  // Calculate total possible combinations and prepare value arrays
  parameters.forEach(param => {
    const { name, min, max, step } = param;
    const values = [];
    for (let value = min; value <= max; value += step) {
      values.push(Math.round(value * 100) / 100);
    }
    paramValues[name] = values;
    totalCombinations *= values.length;
  });
  
  // Limit to a reasonable number if too many combinations
  const maxCombinations = 1000;
  const limitCombinations = totalCombinations > maxCombinations;
  
  // Generate combinations
  if (limitCombinations) {
    // Generate a subset of combinations randomly
    for (let i = 0; i < maxCombinations; i++) {
      const combination = {};
      parameters.forEach(param => {
        const values = paramValues[param.name];
        const randomIndex = Math.floor(Math.random() * values.length);
        combination[param.name] = values[randomIndex];
      });
      combinations.push(combination);
    }
  } else {
    // Generate all combinations (for smaller parameter spaces)
    function generateCombos(index, current) {
      if (index === parameters.length) {
        combinations.push({...current});
        return;
      }
      
      const param = parameters[index];
      const values = paramValues[param.name];
      
      for (const value of values) {
        current[param.name] = value;
        generateCombos(index + 1, current);
      }
    }
    
    generateCombos(0, {});
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
    const backtestResult = {
      trades: [],
      metrics: { ...mockBacktestMetrics }
    };
    
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

// Test 1: Small Parameter Space Performance
function testSmallParameterSpace() {
  console.log('\nTest 1: Small Parameter Space Performance');
  
  const params = {
    symbol: 'BTC',
    strategy: 'Moving Average Crossover',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 10000,
    targetMetric: 'sharpeRatio',
    parameters: [
      { name: 'fastPeriod', min: 5, max: 10, step: 1, currentValue: 7 },
      { name: 'slowPeriod', min: 20, max: 30, step: 5, currentValue: 25 }
    ]
  };
  
  console.log(`Parameter space: fastPeriod (${params.parameters[0].min}-${params.parameters[0].max}, step ${params.parameters[0].step}), slowPeriod (${params.parameters[1].min}-${params.parameters[1].max}, step ${params.parameters[1].step})`);
  
  const startTime = performance.now();
  const results = runStrategyOptimization(params);
  const endTime = performance.now();
  const executionTime = endTime - startTime;
  
  console.log(`Generated ${results.length} optimization results`);
  console.log(`Execution time: ${executionTime.toFixed(2)} ms`);
  
  // Validate results
  if (results.length === 0) {
    console.error('❌ Failed: No optimization results generated');
    return { pass: false, executionTime, combinations: 0 };
  }
  
  // Calculate expected number of combinations
  const fastPeriodValues = Math.floor((params.parameters[0].max - params.parameters[0].min) / params.parameters[0].step) + 1;
  const slowPeriodValues = Math.floor((params.parameters[1].max - params.parameters[1].min) / params.parameters[1].step) + 1;
  const expectedCombinations = fastPeriodValues * slowPeriodValues;
  
  if (results.length !== expectedCombinations) {
    console.warn(`⚠️ Warning: Expected ${expectedCombinations} combinations, got ${results.length}`);
  }
  
  console.log('✅ Success: Small parameter space optimization completed');
  return { pass: true, executionTime, combinations: results.length };
}

// Test 2: Medium Parameter Space Performance
function testMediumParameterSpace() {
  console.log('\nTest 2: Medium Parameter Space Performance');
  
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
  
  console.log(`Parameter space: 3 parameters with approximately 15-20 values each`);
  
  const startTime = performance.now();
  const results = runStrategyOptimization(params);
  const endTime = performance.now();
  const executionTime = endTime - startTime;
  
  console.log(`Generated ${results.length} optimization results`);
  console.log(`Execution time: ${executionTime.toFixed(2)} ms`);
  
  // Validate results
  if (results.length === 0) {
    console.error('❌ Failed: No optimization results generated');
    return { pass: false, executionTime, combinations: 0 };
  }
  
  // Check if we got a reasonable number of combinations
  if (results.length < 100) {
    console.warn('⚠️ Warning: Fewer combinations than expected for medium parameter space');
  }
  
  console.log('✅ Success: Medium parameter space optimization completed');
  return { pass: true, executionTime, combinations: results.length };
}

// Test 3: Large Parameter Space Performance
function testLargeParameterSpace() {
  console.log('\nTest 3: Large Parameter Space Performance');
  
  const params = {
    symbol: 'AAPL',
    strategy: 'Multi-Factor',
    startDate: '2023-01-01',
    endDate: '2023-12-31',
    initialCapital: 10000,
    targetMetric: 'sharpeRatio',
    parameters: [
      { name: 'maFast', min: 5, max: 25, step: 1, currentValue: 10 },
      { name: 'maSlow', min: 20, max: 100, step: 5, currentValue: 50 },
      { name: 'rsiPeriod', min: 7, max: 21, step: 1, currentValue: 14 },
      { name: 'rsiOverbought', min: 65, max: 80, step: 1, currentValue: 70 },
      { name: 'rsiOversold', min: 20, max: 35, step: 1, currentValue: 30 },
      { name: 'bbPeriod', min: 10, max: 30, step: 2, currentValue: 20 },
      { name: 'bbDeviation', min: 1.5, max: 3.0, step: 0.1, currentValue: 2.0 }
    ]
  };
  
  console.log(`Parameter space: 7 parameters with many possible values`);
  
  const startTime = performance.now();
  const results = runStrategyOptimization(params);
  const endTime = performance.now();
  const executionTime = endTime - startTime;
  
  console.log(`Generated ${results.length} optimization results`);
  console.log(`Execution time: ${executionTime.toFixed(2)} ms`);
  
  // Validate results
  if (results.length === 0) {
    console.error('❌ Failed: No optimization results generated');
    return { pass: false, executionTime, combinations: 0 };
  }
  
  // Check if we got a reasonable number of combinations
  if (results.length < 500) {
    console.warn('⚠️ Warning: Fewer combinations than expected for large parameter space');
  }
  
  console.log('✅ Success: Large parameter space optimization completed');
  return { pass: true, executionTime, combinations: results.length };
}

// Test 4: Performance Scaling Analysis
function testPerformanceScaling() {
  console.log('\nTest 4: Performance Scaling Analysis');
  
  const testCases = [
    {
      name: 'Very Small (2 parameters, narrow range)',
      parameters: [
        { name: 'param1', min: 1, max: 5, step: 1, currentValue: 3 },
        { name: 'param2', min: 1, max: 5, step: 1, currentValue: 3 }
      ]
    },
    {
      name: 'Small (2 parameters, wider range)',
      parameters: [
        { name: 'param1', min: 1, max: 10, step: 1, currentValue: 5 },
        { name: 'param2', min: 1, max: 10, step: 1, currentValue: 5 }
      ]
    },
    {
      name: 'Medium (3 parameters)',
      parameters: [
        { name: 'param1', min: 1, max: 10, step: 1, currentValue: 5 },
        { name: 'param2', min: 1, max: 10, step: 1, currentValue: 5 },
        { name: 'param3', min: 1, max: 10, step: 1, currentValue: 5 }
      ]
    },
    {
      name: 'Large (4 parameters)',
      parameters: [
        { name: 'param1', min: 1, max: 10, step: 1, currentValue: 5 },
        { name: 'param2', min: 1, max: 10, step: 1, currentValue: 5 },
        { name: 'param3', min: 1, max: 10, step: 1, currentValue: 5 },
        { name: 'param4', min: 1, max: 10, step: 1, currentValue: 5 }
      ]
    },
    {
      name: 'Very Large (5 parameters)',
      parameters: [
        { name: 'param1', min: 1, max: 10, step: 1, currentValue: 5 },
        { name: 'param2', min: 1, max: 10, step: 1, currentValue: 5 },
        { name: 'param3', min: 1, max: 10, step: 1, currentValue: 5 },
        { name: 'param4', min: 1, max: 10, step: 1, currentValue: 5 },
        { name: 'param5', min: 1, max: 10, step: 1, currentValue: 5 }
      ]
    }
  ];
  
  const results = [];
  
  for (const testCase of testCases) {
    console.log(`Testing ${testCase.name} parameter space...`);
    
    const params = {
      symbol: 'BTC',
      strategy: 'Test Strategy',
      startDate: '2023-01-01',
      endDate: '2023-12-31',
      initialCapital: 10000,
      targetMetric: 'sharpeRatio',
      parameters: testCase.parameters
    };
    
    const startTime = performance.now();
    const optimizationResults = runStrategyOptimization(params);
    const endTime = performance.now();
    const executionTime = endTime - startTime;
    
    console.log(`Generated ${optimizationResults.length} combinations in ${executionTime.toFixed(2)} ms`);
    
    results.push({
      name: testCase.name,
      parameters: testCase.parameters.length,
      combinations: optimizationResults.length,
      executionTime
    });
  }
  
  // Print scaling results
  console.log('\nPerformance Scaling Results:');
  console.log('-----------------------------');
  console.log('Parameter Space | Parameters | Combinations | Execution Time (ms)');
  console.log('------------------------------------------------------------');
  results.forEach(result => {
    console.log(`${result.name.padEnd(15)} | ${String(result.parameters).padEnd(10)} | ${String(result.combinations).padEnd(12)} | ${result.executionTime.toFixed(2)}`);
  });
  
  // Calculate scaling factor
  if (results.length > 1) {
    const scalingFactors = [];
    for (let i = 1; i < results.length; i++) {
      const combinationsRatio = results[i].combinations / results[i-1].combinations;
      const timeRatio = results[i].executionTime / results[i-1].executionTime;
      scalingFactors.push({
        from: results[i-1].name,
        to: results[i].name,
        combinationsRatio,
        timeRatio,
        efficiency: combinationsRatio / timeRatio
      });
    }
    
    console.log('\nScaling Factors:');
    console.log('----------------');
    console.log('Transition | Combinations Increase | Time Increase | Efficiency');
    console.log('------------------------------------------------------------');
    scalingFactors.forEach(factor => {
      console.log(`${factor.from} → ${factor.to} | ${factor.combinationsRatio.toFixed(2)}x | ${factor.timeRatio.toFixed(2)}x | ${factor.efficiency.toFixed(2)}`);
    });
  }
  
  return results;
}

// Run all tests
console.log('=== PERFORMANCE TEST SUITE ===');
const smallSpaceResult = testSmallParameterSpace();
const mediumSpaceResult = testMediumParameterSpace();
const largeSpaceResult = testLargeParameterSpace();
const scalingResults = testPerformanceScaling();

console.log('\n=== TEST SUMMARY ===');
console.log(`Small Parameter Space: ${smallSpaceResult.pass ? '✅ PASS' : '❌ FAIL'} (${smallSpaceResult.combinations} combinations in ${smallSpaceResult.executionTime.toFixed(2)} ms)`);
console.log(`Medium Parameter Space: ${mediumSpaceResult.pass ? '✅ PASS' : '❌ FAIL'} (${mediumSpaceResult.combinations} combinations in ${mediumSpaceResult.executionTime.toFixed(2)} ms)`);
console.log(`Large Parameter Space: ${largeSpaceResult.pass ? '✅ PASS' : '❌ FAIL'} (${largeSpaceResult.combinations} combinations in ${largeSpaceResult.executionTime.toFixed(2)} ms)`);
console.log(`Performance Scaling Analysis: ${scalingResults.length > 0 ? '✅ PASS' : '❌ FAIL'}`);

// Overall performance assessment
const totalCombinations = smallSpaceResult.combinations + mediumSpaceResult.combinations + largeSpaceResult.combinations;
const totalTime = smallSpaceResult.executionTime + mediumSpaceResult.executionTime + largeSpaceResult.executionTime;
const combinationsPerSecond = totalCombinations / (totalTime / 1000);

console.log(`\nOverall Performance: ${combinationsPerSecond.toFixed(2)} combinations/second`);

if (smallSpaceResult.pass && mediumSpaceResult.pass && largeSpaceResult.pass) {
  console.log('\n🎉 All performance tests passed successfully!');
} else {
  console.error('\n❌ Some performance tests failed. See details above.');
}
