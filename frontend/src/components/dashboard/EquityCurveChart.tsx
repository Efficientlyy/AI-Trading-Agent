import React, { useMemo } from 'react';
import { useRenderLogger } from '../../hooks/useRenderLogger';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

interface EquityPoint {
  timestamp: string | number;
  value: number;
  benchmark?: number;
}

interface EquityCurveChartProps {
  data: EquityPoint[] | null;
  isLoading: boolean;
  timeframe?: 'day' | 'week' | 'month' | 'year' | 'all';
  showBenchmark?: boolean;
  benchmarkName?: string;
  onTimeframeChange?: (timeframe: 'day' | 'week' | 'month' | 'year' | 'all') => void;
}

const EquityCurveChart: React.FC<EquityCurveChartProps> = ({
  data,
  isLoading,
  timeframe = 'month',
  showBenchmark = true,
  benchmarkName = 'S&P 500',
  onTimeframeChange,
}) => {
  useRenderLogger('EquityCurveChart', { data, isLoading });

  // Timeframe selector buttons - memoized to prevent unnecessary recreations
  const timeframeOptions = useMemo(() => [
    { value: 'day', label: '1D' },
    { value: 'week', label: '1W' },
    { value: 'month', label: '1M' },
    { value: 'year', label: '1Y' },
    { value: 'all', label: 'All' }
  ], []);

  // Handle timeframe button click
  const handleTimeframeClick = (value: string) => {
    if (onTimeframeChange) {
      onTimeframeChange(value as 'day' | 'week' | 'month' | 'year' | 'all');
    }
  };

  if (isLoading) {
    return (
      <div className="dashboard-widget col-span-3 h-80">
        <h2 className="text-lg font-semibold mb-3">Portfolio Performance</h2>
        <div className="animate-pulse h-64 bg-gray-200 dark:bg-gray-700 rounded"></div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="dashboard-widget col-span-3 h-80">
        <h2 className="text-lg font-semibold mb-3">Portfolio Performance</h2>
        <div className="text-gray-500 dark:text-gray-400 text-center py-24">
          No performance data available
        </div>
      </div>
    );
  }

  // Format date for the x-axis
  const formatDate = (timestamp: string | number) => {
    const date = new Date(timestamp);
    
    switch (timeframe) {
      case 'day':
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
      case 'week':
        return date.toLocaleDateString('en-US', { weekday: 'short', month: 'numeric', day: 'numeric' });
      case 'month':
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      case 'year':
        return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
      case 'all':
        return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
      default:
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
  };

  // Calculate percentage change for the tooltip
  const calculatePercentageChange = (currentValue: number, initialValue: number) => {
    if (initialValue === 0) return '0.00%';
    const percentChange = ((currentValue - initialValue) / initialValue) * 100;
    return `${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(2)}%`;
  };

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const initialValue = data[0].value;
      const currentValue = payload[0].value;
      const percentChange = calculatePercentageChange(currentValue, initialValue);
      
      return (
        <div className="bg-white dark:bg-gray-800 p-3 border border-gray-200 dark:border-gray-700 shadow-md rounded">
          <p className="text-gray-600 dark:text-gray-400">{formatDate(label)}</p>
          <p className="font-medium text-primary">
            Portfolio: ${currentValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            <span className={`ml-2 text-sm ${currentValue >= initialValue ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              {percentChange}
            </span>
          </p>
          {showBenchmark && payload[1] && data[0].benchmark !== undefined && (
            <p className="font-medium text-secondary">
              {benchmarkName}: ${payload[1].value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              <span className={`ml-2 text-sm ${payload[1].value >= data[0].benchmark ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                {calculatePercentageChange(payload[1].value, data[0].benchmark)}
              </span>
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="dashboard-widget col-span-3 h-80">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-lg font-semibold">Portfolio Performance</h2>
        <div className="flex space-x-1">
          {timeframeOptions.map(option => (
            <button
              key={option.value}
              className={`px-2 py-1 text-xs rounded ${
                timeframe === option.value 
                  ? 'bg-primary text-white' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700'
              }`}
              onClick={() => handleTimeframeClick(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis 
              dataKey="timestamp" 
              tickFormatter={formatDate} 
              tick={{ fontSize: 12, fill: '#6b7280' }}
              stroke="#9ca3af"
            />
            <YAxis 
              tickFormatter={(value) => `$${value.toLocaleString('en-US', { notation: 'compact', compactDisplay: 'short' })}`}
              tick={{ fontSize: 12, fill: '#6b7280' }}
              stroke="#9ca3af"
              width={60}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="value"
              name="Portfolio"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, stroke: '#3b82f6', strokeWidth: 2, fill: '#fff' }}
            />
            {showBenchmark && (
              <Line
                type="monotone"
                dataKey="benchmark"
                name={benchmarkName}
                stroke="#9333ea"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 6, stroke: '#9333ea', strokeWidth: 2, fill: '#fff' }}
                strokeDasharray="5 5"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default EquityCurveChart;
