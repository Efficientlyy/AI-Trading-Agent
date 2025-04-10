import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import TradeStatistics, { Trade } from './TradeStatistics';
import PerformanceAnalysis from './PerformanceAnalysis';

export interface BacktestParams {
  symbol: string;
  strategy: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  parameters: Record<string, any>;
}

export interface BacktestResult {
  date: string;
  equity: number;
  benchmark?: number;
  drawdown?: number;
}

export interface BacktestMetrics {
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

export interface BacktestingInterfaceProps {
  symbol: string;
  availableStrategies?: string[];
  onRunBacktest?: (params: BacktestParams) => void;
  backtestResults?: BacktestResult[];
  backtestMetrics?: BacktestMetrics;
  backtestTrades?: Trade[];
  isLoading?: boolean;
}

const BacktestingInterface: React.FC<BacktestingInterfaceProps> = ({
  symbol,
  availableStrategies = ['Moving Average Crossover', 'RSI Oscillator', 'MACD Divergence', 'Bollinger Bands', 'Sentiment-Based'],
  onRunBacktest,
  backtestResults = [],
  backtestMetrics,
  backtestTrades = [],
  isLoading = false
}) => {
  // State for backtest parameters
  const [backtestParams, setBacktestParams] = useState<BacktestParams>({
    symbol,
    strategy: availableStrategies[0],
    startDate: new Date(new Date().setFullYear(new Date().getFullYear() - 1)).toISOString().split('T')[0], // Default to 1 year ago
    endDate: new Date().toISOString().split('T')[0], // Default to today
    initialCapital: 10000,
    parameters: {}
  });
  
  // State for active tab
  const [activeTab, setActiveTab] = useState<'results' | 'performance' | 'trades'>('results');
  
  // Update symbol when prop changes
  useEffect(() => {
    setBacktestParams(prev => ({
      ...prev,
      symbol
    }));
  }, [symbol]);
  
  // Update strategy parameters based on selected strategy
  useEffect(() => {
    let params: Record<string, any> = {};
    
    switch (backtestParams.strategy) {
      case 'Moving Average Crossover':
        params = {
          fastPeriod: 10,
          slowPeriod: 50,
          signalPeriod: 9
        };
        break;
      case 'RSI Oscillator':
        params = {
          period: 14,
          overbought: 70,
          oversold: 30
        };
        break;
      case 'MACD Divergence':
        params = {
          fastPeriod: 12,
          slowPeriod: 26,
          signalPeriod: 9
        };
        break;
      case 'Bollinger Bands':
        params = {
          period: 20,
          standardDeviations: 2
        };
        break;
      case 'Sentiment-Based':
        params = {
          sentimentThreshold: 0.6,
          useMarketSentiment: true,
          useSocialSentiment: true
        };
        break;
      default:
        params = {};
    }
    
    setBacktestParams(prev => ({
      ...prev,
      parameters: params
    }));
  }, [backtestParams.strategy]);
  
  // Handle parameter change
  const handleParamChange = (field: keyof BacktestParams, value: any) => {
    setBacktestParams(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // Handle strategy parameter change
  const handleStrategyParamChange = (paramName: string, value: any) => {
    const updatedParams = {
      ...backtestParams.parameters,
      [paramName]: typeof backtestParams.parameters[paramName] === 'number' ? parseFloat(value) : value
    };
    
    setBacktestParams(prev => ({
      ...prev,
      parameters: updatedParams
    }));
  };
  
  // Handle strategy selection change
  const handleStrategyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const strategy = e.target.value;
    
    setBacktestParams(prev => ({
      ...prev,
      strategy
    }));
  };
  
  // Handle run backtest
  const handleRunBacktest = () => {
    if (onRunBacktest) {
      onRunBacktest(backtestParams);
    }
  };
  
  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };
  
  // Generate mock backtest results if none provided
  const generateMockResults = (): BacktestResult[] => {
    if (backtestResults.length > 0) return backtestResults;
    
    const mockResults: BacktestResult[] = [];
    const startDate = new Date(backtestParams.startDate);
    const endDate = new Date(backtestParams.endDate);
    
    let currentDate = new Date(startDate);
    let equity = backtestParams.initialCapital;
    let benchmark = backtestParams.initialCapital;
    
    while (currentDate <= endDate) {
      if (currentDate.getDay() !== 0 && currentDate.getDay() !== 6) { // Skip weekends
        const dailyReturn = (Math.random() * 0.02) - 0.01; // Random daily return between -1% and 1%
        const benchmarkReturn = (Math.random() * 0.015) - 0.005; // Random benchmark return between -0.5% and 1%
        
        equity = equity * (1 + dailyReturn);
        benchmark = benchmark * (1 + benchmarkReturn);
        
        const maxEquity = mockResults.length > 0 
          ? Math.max(equity, ...mockResults.map(r => r.equity))
          : equity;
        
        const drawdown = ((maxEquity - equity) / maxEquity) * 100;
        
        mockResults.push({
          date: currentDate.toISOString().split('T')[0],
          equity: Math.round(equity * 100) / 100,
          benchmark: Math.round(benchmark * 100) / 100,
          drawdown: Math.round(drawdown * 100) / 100
        });
      }
      
      currentDate.setDate(currentDate.getDate() + 1);
    }
    
    return mockResults;
  };
  
  // Generate mock metrics if none provided
  const generateMockMetrics = (): BacktestMetrics => {
    if (backtestMetrics) return backtestMetrics;
    
    const results = generateMockResults();
    const initialValue = backtestParams.initialCapital;
    const finalValue = results[results.length - 1].equity;
    const totalReturn = ((finalValue - initialValue) / initialValue) * 100;
    
    const daysDiff = Math.floor((new Date(backtestParams.endDate).getTime() - new Date(backtestParams.startDate).getTime()) / (1000 * 60 * 60 * 24));
    const years = daysDiff / 365;
    
    const annualizedReturn = (Math.pow((finalValue / initialValue), (1 / years)) - 1) * 100;
    
    return {
      totalReturn: Math.round(totalReturn * 100) / 100,
      annualizedReturn: Math.round(annualizedReturn * 100) / 100,
      sharpeRatio: Math.round((annualizedReturn / 15) * 100) / 100, // Assuming 15% volatility
      maxDrawdown: Math.round(Math.max(...results.map(r => r.drawdown || 0)) * 100) / 100,
      winRate: Math.round(Math.random() * 30 + 50), // Random win rate between 50% and 80%
      profitFactor: Math.round((Math.random() * 1 + 1.2) * 100) / 100, // Random profit factor between 1.2 and 2.2
      averageWin: Math.round(Math.random() * 200 + 100) / 100, // Random average win between $1 and $3
      averageLoss: Math.round(Math.random() * 100 + 50) / 100, // Random average loss between $0.5 and $1.5
      tradesPerMonth: Math.round(Math.random() * 50 + 20), // Random trades per month between 20 and 70
      totalTrades: Math.round(Math.random() * 300 + 100) // Random total trades between 100 and 400
    };
  };
  
  const displayResults = generateMockResults();
  const displayMetrics = generateMockMetrics();
  
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Backtesting Interface</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Backtest Parameters */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Backtest Parameters</h3>
          
          <div className="space-y-4">
            {/* Symbol */}
            <div>
              <label htmlFor="backtest-symbol" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Symbol
              </label>
              <input
                id="backtest-symbol"
                type="text"
                value={symbol}
                readOnly
                className="block w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              />
            </div>
            
            {/* Strategy */}
            <div>
              <label htmlFor="backtest-strategy" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Strategy
              </label>
              <select
                id="backtest-strategy"
                value={backtestParams.strategy}
                onChange={handleStrategyChange}
                className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              >
                {availableStrategies.map(strategy => (
                  <option key={strategy} value={strategy}>{strategy}</option>
                ))}
              </select>
            </div>
            
            {/* Date Range */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="backtest-start-date" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  Start Date
                </label>
                <input
                  id="backtest-start-date"
                  type="date"
                  value={backtestParams.startDate}
                  onChange={(e) => handleParamChange('startDate', e.target.value)}
                  className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                />
              </div>
              
              <div>
                <label htmlFor="backtest-end-date" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  End Date
                </label>
                <input
                  id="backtest-end-date"
                  type="date"
                  value={backtestParams.endDate}
                  onChange={(e) => handleParamChange('endDate', e.target.value)}
                  className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                />
              </div>
            </div>
            
            {/* Initial Capital */}
            <div>
              <label htmlFor="backtest-initial-capital" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Initial Capital
              </label>
              <input
                id="backtest-initial-capital"
                type="number"
                min="1000"
                step="1000"
                value={backtestParams.initialCapital}
                onChange={(e) => handleParamChange('initialCapital', parseFloat(e.target.value))}
                className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              />
            </div>
            
            {/* Strategy Parameters */}
            <div>
              <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                Strategy Parameters
              </h4>
              
              <div className="space-y-3">
                {Object.entries(backtestParams.parameters).map(([paramName, paramValue]) => (
                  <div key={paramName} className="grid grid-cols-2 gap-2 items-center">
                    <label htmlFor={`param-${paramName}`} className="text-sm text-gray-600 dark:text-gray-400">
                      {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                    </label>
                    <input
                      id={`param-${paramName}`}
                      type={typeof paramValue === 'number' ? 'number' : 'text'}
                      value={paramValue}
                      onChange={(e) => handleStrategyParamChange(paramName, e.target.value)}
                      className="px-2 py-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                    />
                  </div>
                ))}
              </div>
            </div>
            
            <button
              type="button"
              onClick={handleRunBacktest}
              disabled={isLoading}
              className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {isLoading ? 'Running Backtest...' : 'Run Backtest'}
            </button>
          </div>
        </div>
        
        {/* Backtest Results */}
        <div className="lg:col-span-2">
          {/* Tabs */}
          <div className="border-b border-gray-200 dark:border-gray-700 mb-4">
            <nav className="-mb-px flex space-x-4">
              <button
                onClick={() => setActiveTab('results')}
                className={`pb-2 px-1 ${
                  activeTab === 'results'
                    ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                Results
              </button>
              <button
                onClick={() => setActiveTab('performance')}
                className={`pb-2 px-1 ${
                  activeTab === 'performance'
                    ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                Performance Analysis
              </button>
              <button
                onClick={() => setActiveTab('trades')}
                className={`pb-2 px-1 ${
                  activeTab === 'trades'
                    ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                Trade Statistics
              </button>
            </nav>
          </div>
          
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <p className="text-gray-500 dark:text-gray-400">Running backtest...</p>
            </div>
          ) : backtestResults.length === 0 ? (
            <div className="flex items-center justify-center h-64 border border-gray-200 dark:border-gray-800 rounded-md">
              <p className="text-gray-500 dark:text-gray-400">Run a backtest to see results</p>
            </div>
          ) : (
            <div>
              {activeTab === 'results' && (
                <div>
                  {/* Equity Curve */}
                  <div className="mb-4">
                    <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Equity Curve
                    </h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={displayResults}
                          margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="date" 
                            tick={{ fontSize: 10 }}
                            tickFormatter={(value) => {
                              const date = new Date(value);
                              return `${date.getMonth() + 1}/${date.getDate()}`;
                            }}
                          />
                          <YAxis tick={{ fontSize: 10 }} />
                          <Tooltip 
                            formatter={(value: number) => [`$${value.toFixed(2)}`, '']}
                            labelFormatter={(label) => `Date: ${label}`}
                          />
                          <Legend />
                          <Line type="monotone" dataKey="equity" stroke="#4F46E5" name="Equity" activeDot={{ r: 8 }} />
                          {displayResults[0].benchmark && (
                            <Line type="monotone" dataKey="benchmark" stroke="#9CA3AF" name="Benchmark" dot={false} />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  
                  {/* Metrics */}
                  {displayMetrics && (
                    <div>
                      <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Performance Metrics
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                            Total Return
                          </h4>
                          <div className={`text-lg font-semibold ${
                            displayMetrics.totalReturn >= 0 
                              ? 'text-green-600 dark:text-green-400' 
                              : 'text-red-600 dark:text-red-400'
                          }`}>
                            {displayMetrics.totalReturn >= 0 ? '+' : ''}
                            {displayMetrics.totalReturn}%
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                            Sharpe Ratio
                          </h4>
                          <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                            {displayMetrics.sharpeRatio.toFixed(2)}
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                            Max Drawdown
                          </h4>
                          <div className="text-lg font-semibold text-red-600 dark:text-red-400">
                            -{displayMetrics.maxDrawdown}%
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                            Win Rate
                          </h4>
                          <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                            {displayMetrics.winRate}%
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                            Profit Factor
                          </h4>
                          <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                            {displayMetrics.profitFactor.toFixed(2)}
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                            Avg Win
                          </h4>
                          <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                            {formatCurrency(displayMetrics.averageWin)}
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                            Avg Loss
                          </h4>
                          <div className="text-lg font-semibold text-red-600 dark:text-red-400">
                            {formatCurrency(displayMetrics.averageLoss)}
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                            Total Trades
                          </h4>
                          <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                            {displayMetrics.totalTrades}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {activeTab === 'performance' && (
                <PerformanceAnalysis
                  backtestResults={displayResults}
                  backtestMetrics={displayMetrics}
                />
              )}
              
              {activeTab === 'trades' && (
                <TradeStatistics
                  trades={backtestTrades}
                />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BacktestingInterface;
