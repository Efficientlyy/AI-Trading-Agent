import { OptimizationParams, OptimizationResult, ParameterConfig } from '../../components/dashboard/StrategyOptimizer';
import { BacktestMetrics } from '../../components/dashboard/BacktestingInterface';
import { runMockBacktest } from './backtestResults';
import { BacktestParams } from '../../types';

// Generate parameter combinations for optimization
const generateParameterCombinations = (params: OptimizationParams): Record<string, any>[] => {
  const combinations: Record<string, any>[] = [];
  
  // Limit the number of combinations to avoid performance issues
  const MAX_COMBINATIONS = 20;
  
  // For each parameter, generate a set of values within the specified range
  const paramValues: Record<string, number[]> = {};
  
  params.parameters.forEach(param => {
    const { name, min, max, step } = param;
    const values: number[] = [];
    
    // Calculate how many steps we can take
    const numSteps = Math.floor((max - min) / step) + 1;
    
    // If we have too many steps, reduce by sampling
    if (numSteps > 10) {
      // Take fewer samples to reduce combinations
      const samplingRate = Math.ceil(numSteps / 10);
      for (let i = 0; i < numSteps; i += samplingRate) {
        values.push(min + (i * step));
      }
      // Always include max value
      if (values[values.length - 1] < max) {
        values.push(max);
      }
    } else {
      // Use all steps
      for (let value = min; value <= max; value += step) {
        values.push(value);
      }
    }
    
    paramValues[name] = values;
  });
  
  // Generate combinations using a recursive function
  const generateCombinations = (
    currentParams: string[],
    currentIndex: number,
    currentCombination: Record<string, any>
  ) => {
    if (currentIndex === currentParams.length) {
      combinations.push({...currentCombination});
      return;
    }
    
    const paramName = currentParams[currentIndex];
    const values = paramValues[paramName];
    
    for (const value of values) {
      currentCombination[paramName] = value;
      generateCombinations(currentParams, currentIndex + 1, currentCombination);
      
      // If we've reached the maximum number of combinations, stop
      if (combinations.length >= MAX_COMBINATIONS) {
        return;
      }
    }
  };
  
  generateCombinations(
    params.parameters.map(p => p.name),
    0,
    {}
  );
  
  return combinations;
};

// Run a backtest for each parameter combination
const runBacktestsForCombinations = (
  params: OptimizationParams,
  combinations: Record<string, any>[]
): OptimizationResult[] => {
  const results: OptimizationResult[] = [];
  
  combinations.forEach(paramCombination => {
    // Create a backtest params object with the current parameter combination
    const backtestParams: BacktestParams = {
      symbol: 'BTC/USDT', 
      strategy_name: params.strategy,
      start_date: params.startDate,
      end_date: params.endDate,
      initial_capital: params.initialCapital,
      parameters: paramCombination
    };
    
    // Run a mock backtest with these parameters
    const { metrics } = runMockBacktest(backtestParams);
    
    // Add the result
    results.push({
      parameters: paramCombination,
      metrics
    });
  });
  
  return results;
};

// Sort optimization results based on the target metric
const sortOptimizationResults = (
  results: OptimizationResult[],
  targetMetric: keyof BacktestMetrics
): OptimizationResult[] => {
  return [...results].sort((a, b) => {
    // For drawdown, lower is better
    if (targetMetric === 'maxDrawdown') {
      return a.metrics[targetMetric] - b.metrics[targetMetric];
    }
    
    // For all other metrics, higher is better
    return b.metrics[targetMetric] - a.metrics[targetMetric];
  });
};

// Run a complete optimization process
export const runMockOptimization = (params: OptimizationParams): OptimizationResult[] => {
  // Generate parameter combinations
  const combinations = generateParameterCombinations(params);
  
  // Run backtests for each combination
  const results = runBacktestsForCombinations(params, combinations);
  
  // Sort results based on the target metric
  const sortedResults = sortOptimizationResults(results, params.targetMetric);
  
  // Return the top results
  return sortedResults.slice(0, 10);
};
