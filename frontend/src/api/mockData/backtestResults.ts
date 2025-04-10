import { BacktestParams, BacktestResult, BacktestMetrics } from '../../components/dashboard/BacktestingInterface';
import { Trade } from '../../components/dashboard/TradeStatistics';
import { generateMockTrades } from './tradeData';

// Generate mock backtest results
export const generateMockBacktestResults = (params: BacktestParams): BacktestResult[] => {
  const { startDate, endDate, initialCapital } = params;
  
  // Parse dates
  const startDateObj = new Date(startDate);
  const endDateObj = new Date(endDate);
  
  // Generate daily results
  const results: BacktestResult[] = [];
  
  let currentDate = new Date(startDateObj);
  let equity = initialCapital;
  let benchmark = initialCapital;
  let highWatermark = equity;
  
  // Generate random daily returns based on strategy
  const strategyVolatility = getStrategyVolatility(params.strategy);
  const benchmarkVolatility = 0.01; // 1% daily volatility for benchmark
  
  while (currentDate <= endDateObj) {
    // Skip weekends
    if (currentDate.getDay() !== 0 && currentDate.getDay() !== 6) {
      // Generate random daily return with slight positive bias for strategy
      const dailyReturn = (Math.random() * 2 - 0.9) * strategyVolatility;
      
      // Generate random daily return with slight positive bias for benchmark
      const benchmarkReturn = (Math.random() * 2 - 0.9) * benchmarkVolatility;
      
      equity = equity * (1 + dailyReturn);
      benchmark = benchmark * (1 + benchmarkReturn);
      
      // Update high watermark
      highWatermark = Math.max(highWatermark, equity);
      
      // Calculate drawdown
      const drawdown = ((highWatermark - equity) / highWatermark) * 100;
      
      results.push({
        date: currentDate.toISOString().split('T')[0],
        equity: Math.round(equity * 100) / 100,
        benchmark: Math.round(benchmark * 100) / 100,
        drawdown: Math.round(drawdown * 100) / 100
      });
    }
    
    // Move to next day
    currentDate.setDate(currentDate.getDate() + 1);
  }
  
  return results;
};

// Get strategy volatility based on strategy name
const getStrategyVolatility = (strategy: string): number => {
  switch (strategy) {
    case 'Moving Average Crossover':
      return 0.015; // 1.5% daily volatility
    case 'MACD Crossover':
      return 0.018; // 1.8% daily volatility
    case 'RSI Oscillator':
      return 0.016; // 1.6% daily volatility
    case 'Bollinger Breakout':
      return 0.02; // 2% daily volatility
    case 'Sentiment-Based':
      return 0.022; // 2.2% daily volatility
    default:
      return 0.015; // Default 1.5% daily volatility
  }
};

// Generate mock backtest metrics
export const generateMockBacktestMetrics = (params: BacktestParams, results: BacktestResult[]): BacktestMetrics => {
  if (results.length === 0) {
    return {
      totalReturn: 0,
      annualizedReturn: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      winRate: 0,
      profitFactor: 0,
      averageWin: 0,
      averageLoss: 0,
      tradesPerMonth: 0,
      totalTrades: 0
    };
  }
  
  const initialValue = results[0].equity;
  const finalValue = results[results.length - 1].equity;
  const totalReturn = ((finalValue - initialValue) / initialValue) * 100;
  
  // Calculate annualized return
  const startDate = new Date(params.startDate);
  const endDate = new Date(params.endDate);
  const daysDiff = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
  const years = daysDiff / 365;
  
  const annualizedReturn = (Math.pow((finalValue / initialValue), (1 / years)) - 1) * 100;
  
  // Calculate max drawdown
  const maxDrawdown = Math.max(...results.map(r => r.drawdown || 0));
  
  // Generate random metrics based on strategy
  const winRate = getStrategyWinRate(params.strategy);
  const profitFactor = 1 + (totalReturn / 100);
  
  // Generate trade metrics
  const totalTrades = Math.floor(daysDiff / 7 * (1 + Math.random() * 2)); // 1-3 trades per week
  const tradesPerMonth = Math.round((totalTrades / (daysDiff / 30)) * 10) / 10;
  
  // Calculate average win and loss
  const averageWin = (params.initialCapital * 0.02) * (1 + Math.random()); // 2-4% of initial capital
  const averageLoss = (params.initialCapital * 0.01) * (1 + Math.random()); // 1-2% of initial capital
  
  return {
    totalReturn: Math.round(totalReturn * 100) / 100,
    annualizedReturn: Math.round(annualizedReturn * 100) / 100,
    sharpeRatio: Math.round((annualizedReturn / 15) * 100) / 100, // Assuming 15% volatility
    maxDrawdown: Math.round(maxDrawdown * 100) / 100,
    winRate: Math.round(winRate * 100) / 100,
    profitFactor: Math.round(profitFactor * 100) / 100,
    averageWin: Math.round(averageWin * 100) / 100,
    averageLoss: Math.round(averageLoss * 100) / 100,
    tradesPerMonth,
    totalTrades
  };
};

// Get strategy win rate based on strategy name
const getStrategyWinRate = (strategy: string): number => {
  switch (strategy) {
    case 'Moving Average Crossover':
      return 55 + Math.random() * 5; // 55-60% win rate
    case 'MACD Crossover':
      return 58 + Math.random() * 5; // 58-63% win rate
    case 'RSI Oscillator':
      return 60 + Math.random() * 7; // 60-67% win rate
    case 'Bollinger Breakout':
      return 52 + Math.random() * 6; // 52-58% win rate
    case 'Sentiment-Based':
      return 57 + Math.random() * 8; // 57-65% win rate
    default:
      return 55 + Math.random() * 5; // Default 55-60% win rate
  }
};

// Run a mock backtest
export const runMockBacktest = (params: BacktestParams): { 
  results: BacktestResult[]; 
  metrics: BacktestMetrics;
  trades: Trade[];
} => {
  const results = generateMockBacktestResults(params);
  const metrics = generateMockBacktestMetrics(params, results);
  const trades = generateMockTrades(params);
  
  return {
    results,
    metrics,
    trades
  };
};
