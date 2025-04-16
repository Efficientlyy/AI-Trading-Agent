import React, { useState, useMemo } from 'react';
import { Cell, 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, PieChart, Pie
} from 'recharts';

export interface Trade {
  id: string;
  symbol: string;
  type: 'buy' | 'sell';
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  pnl: number;
  pnlPercent: number;
  duration: number; // in days
  strategy: string;
}

export interface TradeStatisticsProps {
  trades: Trade[];
}

const COLORS = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];

const TradeStatistics: React.FC<TradeStatisticsProps> = ({ trades = [] }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'duration' | 'symbols' | 'strategies'>('overview');
  
  // Calculate trade statistics
  const stats = useMemo(() => {
    if (!trades || trades.length === 0) {
      return {
        totalTrades: 0,
        winningTrades: 0,
        losingTrades: 0,
        winRate: 0,
        averageWin: 0,
        averageLoss: 0,
        profitFactor: 0,
        largestWin: 0,
        largestLoss: 0,
        averageDuration: 0,
        totalProfit: 0
      };
    }
    
    const winningTrades = trades.filter(trade => trade.pnl > 0);
    const losingTrades = trades.filter(trade => trade.pnl < 0);
    
    const totalProfit = trades.reduce((sum, trade) => sum + trade.pnl, 0);
    const totalWinAmount = winningTrades.reduce((sum, trade) => sum + trade.pnl, 0);
    const totalLossAmount = Math.abs(losingTrades.reduce((sum, trade) => sum + trade.pnl, 0));
    
    return {
      totalTrades: trades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      winRate: (winningTrades.length / trades.length) * 100,
      averageWin: winningTrades.length > 0 ? totalWinAmount / winningTrades.length : 0,
      averageLoss: losingTrades.length > 0 ? totalLossAmount / losingTrades.length : 0,
      profitFactor: totalLossAmount > 0 ? totalWinAmount / totalLossAmount : totalWinAmount > 0 ? Infinity : 0,
      largestWin: winningTrades.length > 0 ? Math.max(...winningTrades.map(t => t.pnl)) : 0,
      largestLoss: losingTrades.length > 0 ? Math.min(...losingTrades.map(t => t.pnl)) : 0,
      averageDuration: trades.reduce((sum, trade) => sum + trade.duration, 0) / trades.length,
      totalProfit
    };
  }, [trades]);
  
  // Prepare data for charts
  const durationData = useMemo(() => {
    if (!trades || trades.length === 0) return [];
    
    // Group trades by duration ranges
    const durationRanges = [
      { range: '1 day', count: 0, profit: 0 },
      { range: '2-3 days', count: 0, profit: 0 },
      { range: '4-7 days', count: 0, profit: 0 },
      { range: '1-2 weeks', count: 0, profit: 0 },
      { range: '2-4 weeks', count: 0, profit: 0 },
      { range: '1+ month', count: 0, profit: 0 }
    ];
    
    trades.forEach(trade => {
      let rangeIndex = 0;
      
      if (trade.duration <= 1) rangeIndex = 0;
      else if (trade.duration <= 3) rangeIndex = 1;
      else if (trade.duration <= 7) rangeIndex = 2;
      else if (trade.duration <= 14) rangeIndex = 3;
      else if (trade.duration <= 30) rangeIndex = 4;
      else rangeIndex = 5;
      
      durationRanges[rangeIndex].count++;
      durationRanges[rangeIndex].profit += trade.pnl;
    });
    
    return durationRanges;
  }, [trades]);
  
  const symbolData = useMemo(() => {
    if (!trades || trades.length === 0) return [];
    
    // Group trades by symbol
    const symbolMap: Record<string, { symbol: string; count: number; profit: number; winRate: number }> = {};
    
    trades.forEach(trade => {
      if (!symbolMap[trade.symbol]) {
        symbolMap[trade.symbol] = {
          symbol: trade.symbol,
          count: 0,
          profit: 0,
          winRate: 0
        };
      }
      
      symbolMap[trade.symbol].count++;
      symbolMap[trade.symbol].profit += trade.pnl;
    });
    
    // Calculate win rate for each symbol
    Object.keys(symbolMap).forEach(symbol => {
      const symbolTrades = trades.filter(t => t.symbol === symbol);
      const winningTrades = symbolTrades.filter(t => t.pnl > 0);
      symbolMap[symbol].winRate = (winningTrades.length / symbolTrades.length) * 100;
    });
    
    return Object.values(symbolMap).sort((a, b) => b.count - a.count);
  }, [trades]);
  
  const strategyData = useMemo(() => {
    if (!trades || trades.length === 0) return [];
    
    // Group trades by strategy
    const strategyMap: Record<string, { strategy: string; count: number; profit: number; winRate: number }> = {};
    
    trades.forEach(trade => {
      if (!strategyMap[trade.strategy]) {
        strategyMap[trade.strategy] = {
          strategy: trade.strategy,
          count: 0,
          profit: 0,
          winRate: 0
        };
      }
      
      strategyMap[trade.strategy].count++;
      strategyMap[trade.strategy].profit += trade.pnl;
    });
    
    // Calculate win rate for each strategy
    Object.keys(strategyMap).forEach(strategy => {
      const strategyTrades = trades.filter(t => t.strategy === strategy);
      const winningTrades = strategyTrades.filter(t => t.pnl > 0);
      strategyMap[strategy].winRate = (winningTrades.length / strategyTrades.length) * 100;
    });
    
    return Object.values(strategyMap).sort((a, b) => b.count - a.count);
  }, [trades]);
  
  const scatterData = useMemo(() => {
    if (!trades || trades.length === 0) return [];
    
    return trades.map(trade => ({
      duration: trade.duration,
      pnl: trade.pnl,
      pnlPercent: trade.pnlPercent,
      symbol: trade.symbol
    }));
  }, [trades]);
  
  if (!trades || trades.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-3">Trade Statistics</h2>
        <div className="flex items-center justify-center h-64 border border-gray-200 dark:border-gray-800 rounded-md">
          <p className="text-gray-500 dark:text-gray-400">No trade data available</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Trade Statistics</h2>
      
      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700 mb-4">
        <nav className="-mb-px flex space-x-4">
          <button
            onClick={() => setActiveTab('overview')}
            className={`pb-2 px-1 ${
              activeTab === 'overview'
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveTab('duration')}
            className={`pb-2 px-1 ${
              activeTab === 'duration'
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            Duration Analysis
          </button>
          <button
            onClick={() => setActiveTab('symbols')}
            className={`pb-2 px-1 ${
              activeTab === 'symbols'
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            By Symbol
          </button>
          <button
            onClick={() => setActiveTab('strategies')}
            className={`pb-2 px-1 ${
              activeTab === 'strategies'
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            By Strategy
          </button>
        </nav>
      </div>
      
      {/* Content */}
      <div className="h-80">
        {activeTab === 'overview' && (
          <div className="h-full grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Key Metrics */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  Total Trades
                </h4>
                <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {stats.totalTrades}
                </div>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  Win Rate
                </h4>
                <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {stats.winRate.toFixed(1)}%
                </div>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  Profit Factor
                </h4>
                <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {stats.profitFactor === Infinity ? '∞' : stats.profitFactor.toFixed(2)}
                </div>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  Avg Duration
                </h4>
                <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {stats.averageDuration.toFixed(1)} days
                </div>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  Avg Win
                </h4>
                <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                  ${stats.averageWin.toFixed(2)}
                </div>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  Avg Loss
                </h4>
                <div className="text-lg font-semibold text-red-600 dark:text-red-400">
                  -${stats.averageLoss.toFixed(2)}
                </div>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  Largest Win
                </h4>
                <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                  ${stats.largestWin.toFixed(2)}
                </div>
              </div>
              
              <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  Largest Loss
                </h4>
                <div className="text-lg font-semibold text-red-600 dark:text-red-400">
                  ${stats.largestLoss.toFixed(2)}
                </div>
              </div>
            </div>
            
            {/* Win/Loss Ratio Pie Chart */}
            <div>
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Win/Loss Distribution
              </h4>
              <ResponsiveContainer width="100%" height="90%">
                <PieChart>
                  <Pie
                    data={[
                      { name: 'Winning Trades', value: stats.winningTrades },
                      { name: 'Losing Trades', value: stats.losingTrades }
                    ]}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    <Cell fill="#10B981" />
                    <Cell fill="#EF4444" />
                  </Pie>
                  <Tooltip formatter={(value) => [value, 'Trades']} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        
        {activeTab === 'duration' && (
          <div className="h-full">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Trade Performance by Duration
            </h3>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart
                data={durationData}
                margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="left" orientation="left" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} />
                <Tooltip 
                  formatter={(value: any, name: string) => [
                    name === 'count' ? value : `$${Number(value).toFixed(2)}`,
                    name === 'count' ? 'Trades' : 'P&L'
                  ]}
                />
                <Legend />
                <Bar yAxisId="left" dataKey="count" name="Trade Count" fill="#4F46E5" />
                <Bar yAxisId="right" dataKey="profit" name="P&L" fill="#10B981">
                  {durationData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.profit >= 0 ? '#10B981' : '#EF4444'} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        
        {activeTab === 'symbols' && (
          <div className="h-full">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Performance by Symbol
            </h3>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart
                data={symbolData}
                margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="symbol" type="category" tick={{ fontSize: 10 }} width={40} />
                <Tooltip 
                  formatter={(value: any, name: string) => [
                    name === 'count' ? value : name === 'profit' ? `$${Number(value).toFixed(2)}` : `${Number(value).toFixed(1)}%`,
                    name === 'count' ? 'Trades' : name === 'profit' ? 'P&L' : 'Win Rate'
                  ]}
                />
                <Legend />
                <Bar dataKey="count" name="Trade Count" fill="#4F46E5" />
                <Bar dataKey="profit" name="P&L" fill="#10B981">
                  {symbolData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.profit >= 0 ? '#10B981' : '#EF4444'} 
                    />
                  ))}
                </Bar>
                <Bar dataKey="winRate" name="Win Rate %" fill="#F59E0B" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        
        {activeTab === 'strategies' && (
          <div className="h-full">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Performance by Strategy
            </h3>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart
                data={strategyData}
                margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="strategy" type="category" tick={{ fontSize: 10 }} width={80} />
                <Tooltip 
                  formatter={(value: any, name: string) => [
                    name === 'count' ? value : name === 'profit' ? `$${Number(value).toFixed(2)}` : `${Number(value).toFixed(1)}%`,
                    name === 'count' ? 'Trades' : name === 'profit' ? 'P&L' : 'Win Rate'
                  ]}
                />
                <Legend />
                <Bar dataKey="count" name="Trade Count" fill="#4F46E5" />
                <Bar dataKey="profit" name="P&L" fill="#10B981">
                  {strategyData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.profit >= 0 ? '#10B981' : '#EF4444'} 
                    />
                  ))}
                </Bar>
                <Bar dataKey="winRate" name="Win Rate %" fill="#F59E0B" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradeStatistics;
