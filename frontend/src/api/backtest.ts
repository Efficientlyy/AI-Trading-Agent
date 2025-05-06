import { createAuthenticatedClient } from './client';
import { BacktestParams, BacktestResult, PerformanceMetrics, Order } from '../types';

// Create a simplified mock implementation instead of importing from a file that doesn't exist
interface MockBacktestResult {
  date: string;
  equity: number;
  benchmark: number;
  drawdown: number;
}

interface MockBacktestMetrics {
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  averageWin: number;
  averageLoss: number;
  tradesPerMonth: number;
  totalTrades: number;
}

interface MockTrade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  timestamp: string;
  pnl: number;
}

// Mock data generation functions
const generateMockBacktestResults = (params: BacktestParams): MockBacktestResult[] => {
  const startDate = new Date(params.start_date);
  const endDate = new Date(params.end_date);
  const initialCapital = params.initial_capital;
  
  const results: MockBacktestResult[] = [];
  let currentDate = new Date(startDate);
  let equity = initialCapital;
  let benchmark = initialCapital;
  let peak = initialCapital;
  
  // Generate random daily returns
  const strategyVolatility = 0.015; // 1.5% daily volatility
  const benchmarkVolatility = 0.01; // 1% daily volatility
  
  while (currentDate <= endDate) {
    // Skip weekends
    if (currentDate.getDay() !== 0 && currentDate.getDay() !== 6) {
      // Generate random daily return with slight positive bias
      const dailyReturn = (Math.random() * 2 - 0.9) * strategyVolatility;
      const benchmarkReturn = (Math.random() * 2 - 0.9) * benchmarkVolatility;
      
      equity = equity * (1 + dailyReturn);
      benchmark = benchmark * (1 + benchmarkReturn);
      
      // Update peak
      peak = Math.max(peak, equity);
      
      // Calculate drawdown
      const drawdown = ((peak - equity) / peak) * 100;
      
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

const generateMockTrades = (params: BacktestParams): MockTrade[] => {
  const startDate = new Date(params.start_date);
  const endDate = new Date(params.end_date);
  const daysDiff = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
  
  // Generate 1-3 trades per week
  const numTrades = Math.floor(daysDiff / 7 * (1 + Math.random() * 2));
  const trades: MockTrade[] = [];
  
  for (let i = 0; i < numTrades; i++) {
    const tradeDate = new Date(startDate.getTime() + Math.random() * (endDate.getTime() - startDate.getTime()));
    const side = Math.random() > 0.5 ? 'BUY' : 'SELL';
    const price = 100 + Math.random() * 50;
    const quantity = Math.floor(10 + Math.random() * 90);
    const pnl = side === 'BUY' ? Math.random() * 200 - 50 : Math.random() * 200 - 100;
    
    trades.push({
      id: `trade-${i}`,
      symbol: params.symbol || 'BTC',
      side,
      quantity,
      price,
      timestamp: tradeDate.toISOString(),
      pnl
    });
  }
  
  return trades;
};

const generateMockBacktestMetrics = (params: BacktestParams, results: MockBacktestResult[]): MockBacktestMetrics => {
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
  const startDate = new Date(params.start_date);
  const endDate = new Date(params.end_date);
  const daysDiff = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
  const years = daysDiff / 365;
  
  const annualizedReturn = (Math.pow((finalValue / initialValue), (1 / years)) - 1) * 100;
  
  // Calculate max drawdown
  const maxDrawdown = Math.max(...results.map(r => r.drawdown));
  
  // Generate random metrics
  const winRate = 55 + Math.random() * 5; // 55-60% win rate
  const profitFactor = 1 + (totalReturn / 100);
  
  // Generate trade metrics
  const totalTrades = Math.floor(daysDiff / 7 * (1 + Math.random() * 2)); // 1-3 trades per week
  const tradesPerMonth = Math.round((totalTrades / (daysDiff / 30)) * 10) / 10;
  
  // Calculate average win and loss
  const averageWin = (params.initial_capital * 0.02) * (1 + Math.random()); // 2-4% of initial capital
  const averageLoss = (params.initial_capital * 0.01) * (1 + Math.random()); // 1-2% of initial capital
  
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

const runMockBacktest = (params: BacktestParams) => {
  const results = generateMockBacktestResults(params);
  const metrics = generateMockBacktestMetrics(params, results);
  const trades = generateMockTrades(params);
  
  return {
    results,
    metrics,
    trades
  };
};

// Flag to determine whether to use mock data or real API
const USE_MOCK_DATA = true;

export const backtestApi = {
  startBacktest: async (params: BacktestParams): Promise<{ job_id: string, message: string }> => {
    if (USE_MOCK_DATA) {
      // Generate a random job ID
      const jobId = `mock-job-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      
      // Store the params in sessionStorage for later retrieval
      sessionStorage.setItem(`backtest-params-${jobId}`, JSON.stringify(params));
      
      return {
        job_id: jobId,
        message: 'Mock backtest started successfully'
      };
    }
    
    // Real API call
    const client = createAuthenticatedClient();
    const response = await client.post<{ job_id: string, message: string }>('/backtest/start', params);
    return response.data;
  },
  
  getBacktestStatus: async (jobId: string): Promise<{ status: string, progress: number, message: string }> => {
    if (USE_MOCK_DATA) {
      // Simulate a completed backtest
      return {
        status: 'completed',
        progress: 100,
        message: 'Mock backtest completed successfully'
      };
    }
    
    // Real API call
    const client = createAuthenticatedClient();
    const response = await client.get<{ status: string, progress: number, message: string }>(`/backtest/status/${jobId}`);
    return response.data;
  },
  
  getBacktestResults: async (jobId: string): Promise<BacktestResult> => {
    if (USE_MOCK_DATA) {
      // Retrieve the params from sessionStorage
      const paramsJson = sessionStorage.getItem(`backtest-params-${jobId}`);
      
      if (paramsJson) {
        const params = JSON.parse(paramsJson) as BacktestParams;
        const mockData = runMockBacktest(params);
        
        return {
          id: jobId,
          strategy_id: `strategy-${Date.now()}`,
          start_date: params.start_date,
          end_date: params.end_date,
          initial_capital: params.initial_capital,
          final_capital: mockData.results[mockData.results.length - 1].equity,
          total_return: mockData.metrics.totalReturn,
          annualized_return: mockData.metrics.annualizedReturn,
          max_drawdown: mockData.metrics.maxDrawdown,
          sharpe_ratio: mockData.metrics.sharpeRatio,
          equity_curve: mockData.results.map(r => ({
            timestamp: r.date,
            equity: r.equity
          })),
          parameters: params.parameters || {},
          trades: mockData.trades.map(trade => ({
            id: trade.id,
            symbol: trade.symbol,
            side: trade.side === 'BUY' ? 'BUY' : 'SELL',
            type: 'MARKET',
            quantity: trade.quantity,
            price: trade.price,
            status: 'FILLED',
            created_at: trade.timestamp
          })) as Order[],
          metrics: {
            total_return: mockData.metrics.totalReturn,
            annualized_return: mockData.metrics.annualizedReturn,
            max_drawdown: mockData.metrics.maxDrawdown,
            sharpe_ratio: mockData.metrics.sharpeRatio,
            sortino_ratio: 1.2,
            win_rate: mockData.metrics.winRate,
            profit_factor: mockData.metrics.profitFactor,
            avg_win: mockData.metrics.averageWin,
            avg_loss: mockData.metrics.averageLoss,
            max_consecutive_wins: 5,
            max_consecutive_losses: 2,
            avg_trade: mockData.metrics.tradesPerMonth
          } as PerformanceMetrics,
          created_at: new Date().toISOString(),
          params: {
            strategy_name: params.strategy_name
          }
        };
      }
      
      // Fallback if no params found
      return {
        id: jobId,
        strategy_id: `strategy-${Date.now()}`,
        start_date: '2023-01-01',
        end_date: '2023-12-31',
        initial_capital: 10000,
        final_capital: 12500,
        total_return: 25,
        annualized_return: 25,
        max_drawdown: 10,
        sharpe_ratio: 1.5,
        equity_curve: [],
        parameters: {},
        trades: [],
        metrics: {
          total_return: 25,
          annualized_return: 25,
          sharpe_ratio: 1.5,
          max_drawdown: 10,
          win_rate: 60,
          profit_factor: 1.8,
          avg_win: 500,
          avg_loss: 300,
          avg_trade: 8,
          sortino_ratio: 1.2,
          max_consecutive_wins: 5,
          max_consecutive_losses: 3,
          volatility: 14.2
        },
        created_at: new Date().toISOString(),
        params: {
          strategy_name: 'Unknown Strategy'
        }
      };
    }
    
    // Real API call
    const client = createAuthenticatedClient();
    const response = await client.get<BacktestResult>(`/backtest/results/${jobId}`);
    return response.data;
  },
  
  getAllBacktests: async (): Promise<{ backtests: BacktestResult[] }> => {
    if (USE_MOCK_DATA) {
      // Mock equity curve data for all backtests
      const mockEquityCurve = (() => {
        const startDate = new Date('2023-01-01');
        const endDate = new Date('2023-12-31');
        const result = [];
        let currentDate = new Date(startDate);
        let equity = 10000;
        
        while (currentDate <= endDate) {
          if (currentDate.getDay() !== 0 && currentDate.getDay() !== 6) { // Skip weekends
            const dailyReturn = (Math.random() * 2 - 0.9) * 0.015; // 1.5% daily volatility
            equity = equity * (1 + dailyReturn);
            
            result.push({
              timestamp: currentDate.toISOString().split('T')[0],
              equity: Math.round(equity * 100) / 100
            });
          }
          currentDate.setDate(currentDate.getDate() + 1);
        }
        
        return result;
      })();
      
      // Generate mock backtest list
      const mockBacktests: BacktestResult[] = [
        {
          id: 'mock-backtest-1',
          strategy_id: 'strategy-1',
          start_date: '2023-01-01',
          end_date: '2023-12-31',
          initial_capital: 10000,
          final_capital: 12500,
          total_return: 25,
          annualized_return: 25,
          max_drawdown: 10,
          sharpe_ratio: 1.5,
          parameters: {},
          equity_curve: mockEquityCurve,
          trades: [] as Order[],
          metrics: {
            total_return: 25,
            annualized_return: 25,
            sharpe_ratio: 1.5,
            max_drawdown: 10,
            win_rate: 60,
            profit_factor: 1.8,
            avg_win: 500,
            avg_loss: 300,
            avg_trade: 8,
            sortino_ratio: 1.2,
            max_consecutive_wins: 5,
            max_consecutive_losses: 3
          },
          created_at: '2023-12-31T23:59:59Z',
          params: {
            strategy_name: 'Moving Average Crossover'
          }
        },
        {
          id: 'mock-backtest-2',
          strategy_id: 'strategy-2',
          start_date: '2023-06-01',
          end_date: '2023-12-31',
          initial_capital: 10000,
          final_capital: 11200,
          total_return: 12,
          annualized_return: 24,
          max_drawdown: 8,
          sharpe_ratio: 1.2,
          parameters: {},
          equity_curve: mockEquityCurve,
          trades: [] as Order[],
          metrics: {
            total_return: 12,
            annualized_return: 24,
            sharpe_ratio: 1.2,
            max_drawdown: 8,
            win_rate: 65,
            profit_factor: 1.6,
            avg_win: 400,
            avg_loss: 250,
            avg_trade: 12,
            sortino_ratio: 1.1,
            max_consecutive_wins: 4,
            max_consecutive_losses: 2
          },
          created_at: '2023-12-30T23:59:59Z',
          params: {
            strategy_name: 'RSI Oscillator'
          }
        },
        {
          id: 'mock-backtest-3',
          strategy_id: 'strategy-3',
          start_date: '2023-09-01',
          end_date: '2023-12-31',
          initial_capital: 10000,
          final_capital: 13000,
          total_return: 30,
          annualized_return: 90,
          max_drawdown: 12,
          sharpe_ratio: 2.1,
          parameters: {},
          equity_curve: mockEquityCurve,
          trades: [] as Order[],
          metrics: {
            total_return: 30,
            annualized_return: 90,
            sharpe_ratio: 2.1,
            max_drawdown: 12,
            win_rate: 62,
            profit_factor: 2.0,
            avg_win: 600,
            avg_loss: 300,
            avg_trade: 10,
            sortino_ratio: 1.8,
            max_consecutive_wins: 6,
            max_consecutive_losses: 2
          },
          created_at: '2023-12-29T23:59:59Z',
          params: {
            strategy_name: 'Sentiment-Based'
          }
        }
      ];
      
      return { backtests: mockBacktests };
    }
    
    // Real API call
    const client = createAuthenticatedClient();
    const response = await client.get<{ backtests: BacktestResult[] }>('/backtest/list');
    return response.data;
  },
  
  getPerformanceMetrics: async (): Promise<PerformanceMetrics> => {
    if (USE_MOCK_DATA) {
      // Generate mock performance metrics
      return {
        total_return: 42.5,
        annualized_return: 18.2,
        sharpe_ratio: 1.65,
        max_drawdown: 15.3,
        win_rate: 58.7,
        profit_factor: 1.92,
        avg_trade: 1.2,
        sortino_ratio: 1.5,
        max_consecutive_wins: 5,
        max_consecutive_losses: 3,
        avg_win: 450,
        avg_loss: 250,
        volatility: 14.2
      };
    }
    
    // Real API call
    const client = createAuthenticatedClient();
    const response = await client.get<PerformanceMetrics>('/metrics');
    return response.data;
  },
};
