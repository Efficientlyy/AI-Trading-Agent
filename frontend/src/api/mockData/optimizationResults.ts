import { BacktestMetrics } from '../../components/dashboard/BacktestingInterface';
import { OptimizationParams, OptimizationResult, ParameterConfig } from '../../components/dashboard/StrategyOptimizer';
import { runMockBacktest } from './backtestResults';

/**
 * Generate mock optimization results based on parameter ranges
 * @param params Optimization parameters
 * @returns Array of optimization results sorted by target metric
 */
export const runStrategyOptimization = (params: OptimizationParams): OptimizationResult[] => {
  const { symbol, strategy, startDate, endDate, initialCapital, targetMetric, parameters } = params;
  const results: OptimizationResult[] = [];
  
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
      parameters: paramCombo // Add the parameters property
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
};

/**
 * Generate parameter combinations for optimization
 * @param parameters Parameter configurations
 * @returns Array of parameter combinations
 */
const generateParameterCombinations = (parameters: ParameterConfig[]): Record<string, number>[] => {
  // For simplicity, we'll generate a limited number of combinations
  // In a real system, this would use a more sophisticated algorithm
  const combinations: Record<string, number>[] = [];
  
  // Generate 20 random combinations
  for (let i = 0; i < 20; i++) {
    const combination: Record<string, number> = {};
    
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
};

/**
 * Adjust metrics based on parameter values to simulate optimization
 * @param metrics Original backtest metrics
 * @param parameters Parameter values
 * @returns Adjusted metrics
 */
const adjustMetricsBasedOnParameters = (
  metrics: BacktestMetrics, 
  parameters: Record<string, number>
): BacktestMetrics => {
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
  
  // Bollinger Bands parameters
  if ('period' in parameters && 'standardDeviations' in parameters) {
    const period = parameters.period;
    const stdDev = parameters.standardDeviations;
    
    // Better performance with standard Bollinger settings
    if (period >= 18 && period <= 22 && stdDev >= 1.9 && stdDev <= 2.2) {
      adjustedMetrics.totalReturn *= 1.3;
      adjustedMetrics.sharpeRatio *= 1.25;
      adjustedMetrics.winRate += 6;
    }
  }
  
  // Sentiment parameters
  if ('sentimentThreshold' in parameters && 'lookbackPeriod' in parameters) {
    const threshold = parameters.sentimentThreshold;
    const lookback = parameters.lookbackPeriod;
    
    // Better performance with higher threshold and moderate lookback
    if (threshold >= 0.65 && threshold <= 0.8 && lookback >= 5 && lookback <= 10) {
      adjustedMetrics.totalReturn *= 1.35;
      adjustedMetrics.sharpeRatio *= 1.3;
      adjustedMetrics.winRate += 8;
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
};

/**
 * Sort optimization results by target metric
 * @param results Optimization results
 * @param targetMetric Metric to sort by
 * @returns Sorted results
 */
const sortResultsByMetric = (
  results: OptimizationResult[], 
  targetMetric: keyof BacktestMetrics
): OptimizationResult[] => {
  return results.sort((a, b) => {
    // For drawdown, lower is better
    if (targetMetric === 'maxDrawdown') {
      return a.metrics[targetMetric] - b.metrics[targetMetric];
    }
    // For all other metrics, higher is better
    return b.metrics[targetMetric] - a.metrics[targetMetric];
  });
};
