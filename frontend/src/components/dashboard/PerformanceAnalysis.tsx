import React, { useState } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell, ScatterChart, Scatter, ZAxis
} from 'recharts';
import { BacktestResult, BacktestMetrics } from './BacktestingInterface';

export interface PerformanceAnalysisProps {
  backtestResults?: BacktestResult[];
  backtestMetrics?: BacktestMetrics;
}

// Calculate monthly returns from backtest results
const calculateMonthlyReturns = (results: BacktestResult[]) => {
  const monthlyReturns: { month: string; return: number }[] = [];
  
  if (!results || results.length < 2) return monthlyReturns;
  
  let currentMonth = '';
  let monthStartEquity = 0;
  
  results.forEach((result, index) => {
    const date = new Date(result.date);
    const month = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
    
    if (month !== currentMonth) {
      // New month
      if (currentMonth !== '') {
        // Calculate return for previous month
        const lastResult = results[index - 1];
        const monthReturn = ((lastResult.equity - monthStartEquity) / monthStartEquity) * 100;
        monthlyReturns.push({
          month: currentMonth,
          return: parseFloat(monthReturn.toFixed(2))
        });
      }
      
      // Start new month
      currentMonth = month;
      monthStartEquity = result.equity;
    }
    
    // Handle the last month
    if (index === results.length - 1) {
      const monthReturn = ((result.equity - monthStartEquity) / monthStartEquity) * 100;
      monthlyReturns.push({
        month: currentMonth,
        return: parseFloat(monthReturn.toFixed(2))
      });
    }
  });
  
  return monthlyReturns;
};

// Calculate drawdown periods
const calculateDrawdownPeriods = (results: BacktestResult[]) => {
  const drawdownPeriods: {
    startDate: string;
    endDate: string;
    duration: number;
    depth: number;
  }[] = [];
  
  if (!results || results.length < 2) return drawdownPeriods;
  
  let inDrawdown = false;
  let drawdownStart = '';
  let maxEquity = results[0].equity;
  let minEquity = results[0].equity;
  let currentDrawdownStart = '';
  
  results.forEach((result, index) => {
    // Update maximum equity
    if (result.equity > maxEquity) {
      maxEquity = result.equity;
      
      // If we were in a drawdown and now we've recovered, record it
      if (inDrawdown) {
        const drawdownDepth = ((maxEquity - minEquity) / maxEquity) * 100;
        
        // Only record significant drawdowns (> 5%)
        if (drawdownDepth > 5) {
          const startDate = new Date(drawdownStart);
          const endDate = new Date(results[index - 1].date);
          const durationDays = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
          
          drawdownPeriods.push({
            startDate: drawdownStart,
            endDate: results[index - 1].date,
            duration: durationDays,
            depth: parseFloat(drawdownDepth.toFixed(2))
          });
        }
        
        inDrawdown = false;
      }
    } else if (result.equity < maxEquity) {
      // We're in a drawdown
      if (!inDrawdown) {
        inDrawdown = true;
        drawdownStart = results[index - 1].date;
        minEquity = result.equity;
      } else if (result.equity < minEquity) {
        minEquity = result.equity;
      }
    }
  });
  
  // If we're still in a drawdown at the end, record it
  if (inDrawdown) {
    const drawdownDepth = ((maxEquity - minEquity) / maxEquity) * 100;
    
    if (drawdownDepth > 5) {
      const startDate = new Date(drawdownStart);
      const endDate = new Date(results[results.length - 1].date);
      const durationDays = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
      
      drawdownPeriods.push({
        startDate: drawdownStart,
        endDate: results[results.length - 1].date,
        duration: durationDays,
        depth: parseFloat(drawdownDepth.toFixed(2))
      });
    }
  }
  
  return drawdownPeriods;
};

// Calculate trade statistics by day of week
const calculateDayOfWeekStats = (results: BacktestResult[]) => {
  const dayStats: { day: string; return: number; trades: number }[] = [
    { day: 'Monday', return: 0, trades: 0 },
    { day: 'Tuesday', return: 0, trades: 0 },
    { day: 'Wednesday', return: 0, trades: 0 },
    { day: 'Thursday', return: 0, trades: 0 },
    { day: 'Friday', return: 0, trades: 0 }
  ];
  
  if (!results || results.length < 2) return dayStats;
  
  const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const dayReturns: Record<string, number[]> = {
    'Monday': [],
    'Tuesday': [],
    'Wednesday': [],
    'Thursday': [],
    'Friday': []
  };
  
  // Calculate daily returns
  for (let i = 1; i < results.length; i++) {
    const prevResult = results[i - 1];
    const result = results[i];
    
    const date = new Date(result.date);
    const dayOfWeek = dayNames[date.getDay()];
    
    // Skip weekends
    if (dayOfWeek === 'Saturday' || dayOfWeek === 'Sunday') continue;
    
    const dailyReturn = ((result.equity - prevResult.equity) / prevResult.equity) * 100;
    
    if (dayReturns[dayOfWeek]) {
      dayReturns[dayOfWeek].push(dailyReturn);
    }
  }
  
  // Calculate average returns for each day
  Object.keys(dayReturns).forEach((day, index) => {
    if (dayReturns[day].length > 0) {
      const totalReturn = dayReturns[day].reduce((sum, ret) => sum + ret, 0);
      const avgReturn = totalReturn / dayReturns[day].length;
      
      dayStats[index].return = parseFloat(avgReturn.toFixed(2));
      dayStats[index].trades = dayReturns[day].length;
    }
  });
  
  return dayStats;
};

const PerformanceAnalysis: React.FC<PerformanceAnalysisProps> = ({ backtestResults, backtestMetrics }) => {
  const [activeTab, setActiveTab] = useState<'returns' | 'drawdowns' | 'dayOfWeek'>('returns');
  
  // Calculate analysis data
  const monthlyReturns = backtestResults ? calculateMonthlyReturns(backtestResults) : [];
  const drawdownPeriods = backtestResults ? calculateDrawdownPeriods(backtestResults) : [];
  const dayOfWeekStats = backtestResults ? calculateDayOfWeekStats(backtestResults) : [];
  
  if (!backtestResults || !backtestMetrics) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-3">Performance Analysis</h2>
        <div className="flex items-center justify-center h-64 border border-gray-200 dark:border-gray-800 rounded-md">
          <p className="text-gray-500 dark:text-gray-400">Run a backtest to see performance analysis</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Performance Analysis</h2>
      
      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700 mb-4">
        <nav className="-mb-px flex space-x-4">
          <button
            onClick={() => setActiveTab('returns')}
            className={`pb-2 px-1 ${
              activeTab === 'returns'
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            Monthly Returns
          </button>
          <button
            onClick={() => setActiveTab('drawdowns')}
            className={`pb-2 px-1 ${
              activeTab === 'drawdowns'
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            Drawdown Analysis
          </button>
          <button
            onClick={() => setActiveTab('dayOfWeek')}
            className={`pb-2 px-1 ${
              activeTab === 'dayOfWeek'
                ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            Day of Week
          </button>
        </nav>
      </div>
      
      {/* Content */}
      <div className="h-80">
        {activeTab === 'returns' && (
          <div className="h-full">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Monthly Returns (%)</h3>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart
                data={monthlyReturns}
                margin={{ top: 5, right: 5, left: 5, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="month" 
                  tick={{ fontSize: 10 }}
                  angle={-45}
                  textAnchor="end"
                />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip 
                  formatter={(value: number) => [`${value}%`, 'Return']}
                  labelFormatter={(label) => `Month: ${label}`}
                />
                <Bar dataKey="return" name="Return %" fill="#4F46E5">
                  {monthlyReturns.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.return >= 0 ? '#4F46E5' : '#EF4444'} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        
        {activeTab === 'drawdowns' && (
          <div className="h-full">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Drawdown Periods</h3>
            {drawdownPeriods.length > 0 ? (
              <div className="overflow-auto h-full">
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
                        Duration (Days)
                      </th>
                      <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Depth (%)
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                    {drawdownPeriods.map((period, index) => (
                      <tr key={`drawdown-${index}`}>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                          {period.startDate}
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                          {period.endDate}
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">
                          {period.duration}
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-red-600 dark:text-red-400 font-medium">
                          -{period.depth}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64">
                <p className="text-gray-500 dark:text-gray-400">No significant drawdown periods detected</p>
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'dayOfWeek' && (
          <div className="h-full">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Performance by Day of Week</h3>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart
                data={dayOfWeekStats}
                margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip 
                  formatter={(value: number, name: string) => [
                    name === 'return' ? `${value}%` : value,
                    name === 'return' ? 'Avg Return' : 'Trade Count'
                  ]}
                />
                <Legend />
                <Bar dataKey="return" name="Avg Return %" fill="#4F46E5">
                  {dayOfWeekStats.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.return >= 0 ? '#4F46E5' : '#EF4444'} 
                    />
                  ))}
                </Bar>
                <Bar dataKey="trades" name="Trade Count" fill="#10B981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
      
      {/* Summary Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Best Month
          </h4>
          <div className="text-lg font-semibold text-green-600 dark:text-green-400">
            {monthlyReturns.length > 0 
              ? `+${Math.max(...monthlyReturns.map(m => m.return))}%`
              : 'N/A'
            }
          </div>
        </div>
        
        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Worst Month
          </h4>
          <div className="text-lg font-semibold text-red-600 dark:text-red-400">
            {monthlyReturns.length > 0 
              ? `${Math.min(...monthlyReturns.map(m => m.return))}%`
              : 'N/A'
            }
          </div>
        </div>
        
        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Profitable Months
          </h4>
          <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {monthlyReturns.length > 0 
              ? `${monthlyReturns.filter(m => m.return > 0).length}/${monthlyReturns.length}`
              : 'N/A'
            }
          </div>
        </div>
        
        <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
          <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Avg Monthly Return
          </h4>
          <div className={`text-lg font-semibold ${
            monthlyReturns.length > 0 && (monthlyReturns.reduce((sum, m) => sum + m.return, 0) / monthlyReturns.length) >= 0
              ? 'text-green-600 dark:text-green-400'
              : 'text-red-600 dark:text-red-400'
          }`}>
            {monthlyReturns.length > 0 
              ? `${(monthlyReturns.reduce((sum, m) => sum + m.return, 0) / monthlyReturns.length).toFixed(2)}%`
              : 'N/A'
            }
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceAnalysis;
