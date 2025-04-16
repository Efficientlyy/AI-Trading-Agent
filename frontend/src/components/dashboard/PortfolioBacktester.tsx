import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell, PieChart, Pie, Sector } from 'recharts';
import { PortfolioBacktestParams, PortfolioBacktestResult, AssetPerformance } from '../../api/mockData/portfolioBacktest';

export interface PortfolioBacktesterProps {
  availableAssets: string[];
  onRunBacktest?: (params: PortfolioBacktestParams) => void;
  backtestResult?: PortfolioBacktestResult;
  isLoading?: boolean;
}

const PortfolioBacktester: React.FC<PortfolioBacktesterProps> = ({
  availableAssets = ['BTC', 'ETH', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'GLD'],
  onRunBacktest,
  backtestResult,
  isLoading = false
}) => {
  // State for backtest parameters
  const [backtestParams, setBacktestParams] = useState<PortfolioBacktestParams>({
    assets: ['BTC', 'ETH', 'AAPL', 'MSFT'],
    weights: [0.25, 0.25, 0.25, 0.25],
    startDate: new Date(new Date().setFullYear(new Date().getFullYear() - 1)).toISOString().split('T')[0], // Default to 1 year ago
    endDate: new Date().toISOString().split('T')[0], // Default to today
    initialCapital: 100000,
    rebalancingPeriod: 'monthly',
    strategies: {
      'BTC': 'MACD Crossover',
      'ETH': 'Bollinger Breakout',
      'AAPL': 'RSI Oscillator',
      'MSFT': 'Sentiment-Based'
    },
    riskManagement: {
      maxDrawdown: 20,
      stopLoss: 10,
      trailingStop: true,
      correlationThreshold: 0.7
    }
  });
  
  // State for selected assets
  const [selectedAssets, setSelectedAssets] = useState<string[]>(backtestParams.assets);
  
  // State for active index
  const [activeIndex, setActiveIndex] = useState<number>(0);
  
  // State for advanced settings
  const [showAdvancedSettings, setShowAdvancedSettings] = useState<boolean>(false);
  
  // Handle asset selection
  const handleAssetSelection = (asset: string) => {
    if (selectedAssets.includes(asset)) {
      // Remove asset if already selected
      if (selectedAssets.length > 1) { // Ensure at least one asset remains
        const newSelectedAssets = selectedAssets.filter(a => a !== asset);
        setSelectedAssets(newSelectedAssets);
        
        // Update weights to be equal
        const equalWeight = 1 / newSelectedAssets.length;
        const newWeights = newSelectedAssets.map(() => equalWeight);
        
        setBacktestParams(prev => ({
          ...prev,
          assets: newSelectedAssets,
          weights: newWeights
        }));
      }
    } else {
      // Add asset if not already selected
      const newSelectedAssets = [...selectedAssets, asset];
      
      // Update weights to be equal
      const equalWeight = 1 / newSelectedAssets.length;
      const newWeights = newSelectedAssets.map(() => equalWeight);
      
      setSelectedAssets(newSelectedAssets);
      setBacktestParams(prev => ({
        ...prev,
        assets: newSelectedAssets,
        weights: newWeights,
        strategies: {
          ...prev.strategies,
          [asset]: 'MACD Crossover' // Default strategy for new asset
        }
      }));
    }
  };
  
  // Handle weight change
  const handleWeightChange = (index: number, value: number) => {
    const newWeights = [...backtestParams.weights];
    newWeights[index] = value;
    
    // Normalize weights to sum to 1
    const sum = newWeights.reduce((a, b) => a + b, 0);
    const normalizedWeights = newWeights.map(w => w / sum);
    
    setBacktestParams(prev => ({
      ...prev,
      weights: normalizedWeights
    }));
  };
  
  // Handle strategy change
  const handleStrategyChange = (asset: string, strategy: string) => {
    setBacktestParams(prev => ({
      ...prev,
      strategies: {
        ...prev.strategies,
        [asset]: strategy
      }
    }));
  };
  
  // Handle risk management change
  const handleRiskChange = (field: keyof PortfolioBacktestParams['riskManagement'], value: any) => {
    setBacktestParams(prev => ({
      ...prev,
      riskManagement: {
        ...prev.riskManagement,
        [field]: typeof prev.riskManagement[field] === 'boolean' ? Boolean(value) : Number(value)
      }
    }));
  };
  
  // Handle general parameter change
  const handleParamChange = (field: keyof PortfolioBacktestParams, value: any) => {
    setBacktestParams(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // Run backtest
  const handleRunBacktest = () => {
    if (onRunBacktest) {
      onRunBacktest(backtestParams);
    }
  };
  
  // Format percentage for display
  const formatPercent = (value: number) => {
    return `${value.toFixed(2)}%`;
  };
  
  // Get color for correlation value
  const getCorrelationColor = (value: number) => {
    if (value >= 0.7) return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
    if (value >= 0.3) return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
    if (value >= -0.3) return "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200";
    if (value >= -0.7) return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
  };
  
  // Render active shape for pie chart
  const renderActiveShape = (props: any) => {
    const { cx, cy, midAngle, innerRadius, outerRadius, startAngle, endAngle, fill, payload, percent, value } = props;
    const RADIAN = Math.PI / 180;
    const sin = Math.sin(-RADIAN * midAngle);
    const cos = Math.cos(-RADIAN * midAngle);
    const sx = cx + (outerRadius + 10) * cos;
    const sy = cy + (outerRadius + 10) * sin;
    const mx = cx + (outerRadius + 30) * cos;
    const my = cy + (outerRadius + 30) * sin;
    const ex = mx + (cos >= 0 ? 1 : -1) * 22;
    const ey = my;
    const textAnchor = cos >= 0 ? 'start' : 'end';

    return (
      <g>
        <text x={cx} y={cy} dy={8} textAnchor="middle" fill={fill}>
          {payload.symbol}
        </text>
        <Sector
          cx={cx}
          cy={cy}
          innerRadius={innerRadius}
          outerRadius={outerRadius}
          startAngle={startAngle}
          endAngle={endAngle}
          fill={fill}
        />
        <Sector
          cx={cx}
          cy={cy}
          startAngle={startAngle}
          endAngle={endAngle}
          innerRadius={outerRadius + 6}
          outerRadius={outerRadius + 10}
          fill={fill}
        />
        <path d={`M${sx},${sy}L${mx},${my}L${ex},${ey}`} stroke={fill} fill="none" />
        <circle cx={ex} cy={ey} r={2} fill={fill} stroke="none" />
        <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} textAnchor={textAnchor} fill="#333">{`${payload.symbol}: ${value.toFixed(2)}%`}</text>
        <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} dy={18} textAnchor={textAnchor} fill="#999">
          {`(${(percent * 100).toFixed(2)}%)`}
        </text>
      </g>
    );
  };
  
  // Handle pie chart hover
  const onPieEnter = (_: any, index: number) => {
    setActiveIndex(index);
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Portfolio Backtester</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Backtest Parameters */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Portfolio Settings</h3>
          
          <div className="space-y-4">
            {/* Asset Selection */}
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Select Assets
              </label>
              <div className="flex flex-wrap gap-2">
                {availableAssets.map(asset => (
                  <button
                    key={asset}
                    type="button"
                    onClick={() => handleAssetSelection(asset)}
                    className={`px-3 py-1 text-xs font-medium rounded-full ${
                      selectedAssets.includes(asset)
                        ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                        : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
                    }`}
                  >
                    {asset}
                  </button>
                ))}
              </div>
            </div>
            
            {/* Asset Weights */}
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Asset Allocation
              </label>
              <div className="space-y-2">
                {selectedAssets.map((asset, index) => (
                  <div key={asset} className="flex items-center gap-2">
                    <span className="w-12 text-sm">{asset}</span>
                    <input
                      type="range"
                      min="1"
                      max="100"
                      value={backtestParams.weights[index] * 100}
                      onChange={(e) => handleWeightChange(index, parseInt(e.target.value) / 100)}
                      className="flex-1"
                    />
                    <span className="w-16 text-sm text-right">{formatPercent(backtestParams.weights[index])}</span>
                  </div>
                ))}
              </div>
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
                min="10000"
                step="10000"
                value={backtestParams.initialCapital}
                onChange={(e) => handleParamChange('initialCapital', parseFloat(e.target.value))}
                className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              />
            </div>
            
            {/* Rebalancing Period */}
            <div>
              <label htmlFor="rebalancing-period" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Rebalancing Period
              </label>
              <select
                id="rebalancing-period"
                value={backtestParams.rebalancingPeriod}
                onChange={(e) => handleParamChange('rebalancingPeriod', e.target.value)}
                className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              >
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
                <option value="quarterly">Quarterly</option>
                <option value="yearly">Yearly</option>
                <option value="none">No Rebalancing</option>
              </select>
            </div>
            
            {/* Asset Strategies */}
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Asset Strategies
              </label>
              <div className="space-y-2">
                {selectedAssets.map(asset => (
                  <div key={`strategy-${asset}`} className="grid grid-cols-4 gap-2 items-center">
                    <span className="text-sm col-span-1">{asset}</span>
                    <select
                      value={backtestParams.strategies[asset] || 'MACD Crossover'}
                      onChange={(e) => handleStrategyChange(asset, e.target.value)}
                      className="col-span-3 px-2 py-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                    >
                      <option value="MACD Crossover">MACD Crossover</option>
                      <option value="RSI Oscillator">RSI Oscillator</option>
                      <option value="Bollinger Breakout">Bollinger Breakout</option>
                      <option value="Moving Average Crossover">Moving Average Crossover</option>
                      <option value="Sentiment-Based">Sentiment-Based</option>
                    </select>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Risk Management */}
            <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
              <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                Risk Management
              </h4>
              
              <div className="space-y-3">
                <div>
                  <label htmlFor="max-drawdown" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                    Max Drawdown (%)
                  </label>
                  <input
                    id="max-drawdown"
                    type="number"
                    min="0"
                    max="50"
                    step="1"
                    value={backtestParams.riskManagement.maxDrawdown}
                    onChange={(e) => handleRiskChange('maxDrawdown', parseFloat(e.target.value))}
                    className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  />
                </div>
                
                <div>
                  <label htmlFor="stop-loss" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                    Stop Loss (%)
                  </label>
                  <input
                    id="stop-loss"
                    type="number"
                    min="0"
                    max="20"
                    step="0.5"
                    value={backtestParams.riskManagement.stopLoss}
                    onChange={(e) => handleRiskChange('stopLoss', parseFloat(e.target.value))}
                    className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  />
                </div>
                
                <div className="flex items-center">
                  <input
                    id="trailing-stop"
                    type="checkbox"
                    checked={backtestParams.riskManagement.trailingStop}
                    onChange={(e) => handleRiskChange('trailingStop', e.target.checked)}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <label htmlFor="trailing-stop" className="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                    Use Trailing Stop
                  </label>
                </div>
                
                <div>
                  <label htmlFor="correlation-threshold" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                    Correlation Threshold
                  </label>
                  <input
                    id="correlation-threshold"
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={backtestParams.riskManagement.correlationThreshold}
                    onChange={(e) => handleRiskChange('correlationThreshold', parseFloat(e.target.value))}
                    className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  />
                </div>
              </div>
            </div>
            
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Risk Management</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Max Drawdown (%)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    value={backtestParams.riskManagement.maxDrawdown}
                    onChange={(e) => handleRiskChange('maxDrawdown', e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Stop Loss (%)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    value={backtestParams.riskManagement.stopLoss}
                    onChange={(e) => handleRiskChange('stopLoss', e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                  />
                </div>
              </div>
            </div>
            
            <div className="mb-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Advanced Settings</h3>
                <button
                  type="button"
                  onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                  className="text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-500 dark:hover:text-indigo-300"
                >
                  {showAdvancedSettings ? 'Hide' : 'Show'}
                </button>
              </div>
              
              {showAdvancedSettings && (
                <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Trailing Stop
                    </label>
                    <select
                      value={backtestParams.riskManagement.trailingStop ? 'true' : 'false'}
                      onChange={(e) => handleRiskChange('trailingStop', e.target.value === 'true')}
                      className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                    >
                      <option value="true">Enabled</option>
                      <option value="false">Disabled</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Correlation Threshold
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={backtestParams.riskManagement.correlationThreshold}
                      onChange={(e) => handleRiskChange('correlationThreshold', e.target.value)}
                      className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-800 dark:text-gray-200"
                    />
                  </div>
                </div>
              )}
            </div>
            
            <button
              type="button"
              onClick={handleRunBacktest}
              disabled={isLoading}
              className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {isLoading ? 'Running Backtest...' : 'Run Portfolio Backtest'}
            </button>
          </div>
        </div>
        
        {/* Backtest Results */}
        <div>
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Portfolio Performance</h3>
          
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <p className="text-gray-500 dark:text-gray-400">Running portfolio backtest...</p>
            </div>
          ) : !backtestResult ? (
            <div className="flex items-center justify-center h-64 border border-gray-200 dark:border-gray-800 rounded-md">
              <p className="text-gray-500 dark:text-gray-400">Run a backtest to see portfolio performance</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Equity Curve */}
              <div>
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                  Portfolio Equity Curve
                </h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={backtestResult.equityCurve}
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
                      <Line type="monotone" dataKey="equity" stroke="#4F46E5" name="Portfolio" activeDot={{ r: 8 }} />
                      <Line type="monotone" dataKey="benchmark" stroke="#9CA3AF" name="Benchmark" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              {/* Performance Metrics */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                  <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                    Return Metrics
                  </h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Total Return:</span>
                      <span className={`text-sm font-medium ${
                        backtestResult.metrics.totalReturn >= 0 
                          ? 'text-green-600 dark:text-green-400' 
                          : 'text-red-600 dark:text-red-400'
                      }`}>
                        {backtestResult.metrics.totalReturn >= 0 ? '+' : ''}
                        {backtestResult.metrics.totalReturn}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Annualized Return:</span>
                      <span className={`text-sm font-medium ${
                        backtestResult.metrics.annualizedReturn >= 0 
                          ? 'text-green-600 dark:text-green-400' 
                          : 'text-red-600 dark:text-red-400'
                      }`}>
                        {backtestResult.metrics.annualizedReturn >= 0 ? '+' : ''}
                        {backtestResult.metrics.annualizedReturn}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio:</span>
                      <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {backtestResult.metrics.sharpeRatio}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                  <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                    Risk Metrics
                  </h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown:</span>
                      <span className="text-sm font-medium text-red-600 dark:text-red-400">
                        -{backtestResult.metrics.maxDrawdown}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Win Rate:</span>
                      <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {backtestResult.metrics.winRate}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Profit Factor:</span>
                      <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {backtestResult.metrics.profitFactor}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Asset Performance */}
              <div>
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                  Asset Performance
                </h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={backtestResult.assetPerformance}
                      margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="symbol" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip 
                        formatter={(value: number) => [`${value}%`, '']}
                        labelFormatter={(label) => `Asset: ${label}`}
                      />
                      <Legend />
                      <Bar dataKey="totalReturn" name="Return %" fill="#4F46E5">
                        {backtestResult.assetPerformance.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.totalReturn >= 0 ? '#4F46E5' : '#EF4444'} 
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              {/* Asset Allocation */}
              <div>
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                  Asset Allocation
                </h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        activeIndex={activeIndex}
                        activeShape={renderActiveShape}
                        data={backtestResult.assetPerformance.map(asset => ({
                          symbol: asset.symbol,
                          value: asset.weight * 100
                        }))}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        onMouseEnter={onPieEnter}
                      >
                        {backtestResult.assetPerformance.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={[
                              '#4F46E5', '#10B981', '#F59E0B', '#EF4444', 
                              '#6366F1', '#8B5CF6', '#EC4899', '#14B8A6',
                              '#0EA5E9', '#F97316'
                            ][index % 10]} 
                          />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value: number) => `${value.toFixed(2)}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              {/* Correlation Matrix */}
              <div>
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                  Correlation Matrix
                </h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-800">
                      <tr>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Asset
                        </th>
                        {backtestResult.assetPerformance.map(asset => (
                          <th key={`header-${asset.symbol}`} scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                            {asset.symbol}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                      {backtestResult.assetPerformance.map(asset1 => (
                        <tr key={`row-${asset1.symbol}`}>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                            {asset1.symbol}
                          </td>
                          {backtestResult.assetPerformance.map(asset2 => (
                            <td key={`cell-${asset1.symbol}-${asset2.symbol}`} className="px-3 py-2 whitespace-nowrap text-sm">
                              <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                                getCorrelationColor(backtestResult.correlationMatrix[asset1.symbol][asset2.symbol])
                              }`}>
                                {backtestResult.correlationMatrix[asset1.symbol][asset2.symbol].toFixed(2)}
                              </span>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              
              {/* Drawdown Events */}
              <div>
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                  Drawdown Events
                </h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-800">
                      <tr>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Start Date
                        </th>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          End Date
                        </th>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Depth (%)
                        </th>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Duration (days)
                        </th>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Recovery (days)
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                      {backtestResult.drawdownEvents.map((event, index) => (
                        <tr key={`drawdown-${index}`}>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                            {event.startDate}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                            {event.endDate}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                            <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              event.depth > 20 ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                              event.depth > 10 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                              'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            }`}>
                              {event.depth.toFixed(2)}%
                            </span>
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                            {event.duration}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                            {event.recovery}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PortfolioBacktester;
