import React, { useEffect, useState } from 'react';
import { useDataSource } from '../../context/DataSourceContext';
import { performanceApi } from '../../api/performance';
import { getMockPerformanceMetrics } from '../../api/mockData/mockPerformanceMetrics';

const PerformanceMetrics: React.FC = () => {
  const { dataSource } = useDataSource();
  const [performance, setPerformance] = useState<{
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate?: number;
    profit_factor?: number;
    avg_trade?: number;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);
    const fetchPerformance = async () => {
      try {
        const data = dataSource === 'mock'
          ? await getMockPerformanceMetrics()
          : await performanceApi.getPerformanceMetrics();
        if (isMounted) setPerformance(data.performance);
      } catch (e) {
        if (isMounted) setPerformance(null);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };
    fetchPerformance();
    return () => { isMounted = false; };
  }, [dataSource]);
  if (isLoading) {
    return (
      <div className="dashboard-widget col-span-1">
        <h2 className="text-lg font-semibold mb-3">Performance Metrics</h2>
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-4"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-4"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3 mb-4"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-4"></div>
        </div>
      </div>
    );
  }

  if (!performance) {
    return (
      <div className="dashboard-widget col-span-1">
        <h2 className="text-lg font-semibold mb-3">Performance Metrics</h2>
        <div className="text-gray-500 dark:text-gray-400 text-center py-8 text-base font-medium">
          No performance data available
        </div>
      </div>
    );
  }

  // Helper function to determine color based on metric value
  const getMetricColor = (value: number, metric: string): string => {
    if (metric === 'total_return' || metric === 'sharpe_ratio' || metric === 'win_rate' || metric === 'profit_factor' || metric === 'avg_trade') {
      return value >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
    } else if (metric === 'max_drawdown') {
      return value <= -15 ? 'text-red-600 dark:text-red-400' : 
             value <= -10 ? 'text-yellow-600 dark:text-yellow-400' : 
             'text-green-600 dark:text-green-400';
    }
    return '';
  };

  // Helper function to format values based on metric type
  const formatMetricValue = (value: number, metric: string): string => {
    if (metric === 'total_return' || metric === 'max_drawdown' || metric === 'win_rate') {
      return `${value.toFixed(2)}%`;
    } else if (metric === 'sharpe_ratio' || metric === 'profit_factor') {
      return value.toFixed(2);
    } else if (metric === 'avg_trade') {
      return `$${value.toFixed(2)}`;
    }
    return value.toString();
  };

  return (
    <div className="dashboard-widget col-span-1">
      <h2 className="text-lg font-semibold mb-3">Performance Metrics</h2>
      
      <div className="grid grid-cols-2 gap-4">
        {/* Total Return */}
        <div className="space-y-1">
          <div className="text-sm text-gray-600 dark:text-gray-400">Total Return</div>
          <div className={`text-lg font-semibold ${getMetricColor(performance.total_return, 'total_return')}`}>
            {formatMetricValue(performance.total_return, 'total_return')}
          </div>
        </div>
        
        {/* Sharpe Ratio */}
        <div className="space-y-1">
          <div className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</div>
          <div className={`text-lg font-semibold ${getMetricColor(performance.sharpe_ratio, 'sharpe_ratio')}`}>
            {formatMetricValue(performance.sharpe_ratio, 'sharpe_ratio')}
          </div>
        </div>
        
        {/* Max Drawdown */}
        <div className="space-y-1">
          <div className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</div>
          <div className={`text-lg font-semibold ${getMetricColor(performance.max_drawdown, 'max_drawdown')}`}>
            {formatMetricValue(performance.max_drawdown, 'max_drawdown')}
          </div>
        </div>
        
        {/* Win Rate (if available) */}
        {performance.win_rate !== undefined && (
          <div className="space-y-1">
            <div className="text-sm text-gray-600 dark:text-gray-400">Win Rate</div>
            <div className={`text-lg font-semibold ${getMetricColor(performance.win_rate, 'win_rate')}`}>
              {formatMetricValue(performance.win_rate, 'win_rate')}
            </div>
          </div>
        )}
        
        {/* Profit Factor (if available) */}
        {performance.profit_factor !== undefined && (
          <div className="space-y-1">
            <div className="text-sm text-gray-600 dark:text-gray-400">Profit Factor</div>
            <div className={`text-lg font-semibold ${getMetricColor(performance.profit_factor, 'profit_factor')}`}>
              {formatMetricValue(performance.profit_factor, 'profit_factor')}
            </div>
          </div>
        )}
        
        {/* Average Trade (if available) */}
        {performance.avg_trade !== undefined && (
          <div className="space-y-1">
            <div className="text-sm text-gray-600 dark:text-gray-400">Avg Trade</div>
            <div className={`text-lg font-semibold ${getMetricColor(performance.avg_trade, 'avg_trade')}`}>
              {formatMetricValue(performance.avg_trade, 'avg_trade')}
            </div>
          </div>
        )}
      </div>
      
      <div className="mt-4">
        <button className="text-sm text-primary hover:text-primary-dark">View Detailed Analysis →</button>
      </div>
    </div>
  );
};

export default PerformanceMetrics;
