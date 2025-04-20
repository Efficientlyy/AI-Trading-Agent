import React, { useState, useEffect, useMemo } from 'react';
import { getCircuitBreakerPerformanceMetrics } from '../../api/utils/circuitBreakerExecutor';
import { PerformanceMetrics } from '../../api/utils/performanceOptimizations';

interface PerformanceDashboardProps {
  refreshInterval?: number;
}

const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  refreshInterval = 5000, // 5 seconds default
}) => {
  const [metrics, setMetrics] = useState<Record<string, PerformanceMetrics>>({});
  const [sortBy, setSortBy] = useState<keyof PerformanceMetrics>('averageExecutionTime');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedFunction, setSelectedFunction] = useState<string | null>(null);

  // Fetch metrics on component mount and at regular intervals
  useEffect(() => {
    const fetchMetrics = () => {
      const currentMetrics = getCircuitBreakerPerformanceMetrics();
      setMetrics(currentMetrics);
    };

    // Initial fetch
    fetchMetrics();

    // Set up interval for refreshing metrics
    const intervalId = setInterval(fetchMetrics, refreshInterval);

    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, [refreshInterval]);

  // Filter and sort metrics
  const filteredAndSortedMetrics = useMemo(() => {
    // Filter metrics by search query
    const filtered = Object.values(metrics).filter(metric => 
      metric.functionName.toLowerCase().includes(searchQuery.toLowerCase())
    );

    // Sort metrics by selected field
    return filtered.sort((a, b) => {
      const aValue = a[sortBy];
      const bValue = b[sortBy];

      if (aValue === undefined || bValue === undefined) {
        return 0;
      }

      if (sortDirection === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });
  }, [metrics, sortBy, sortDirection, searchQuery]);

  // Calculate overall statistics
  const overallStats = useMemo(() => {
    const allMetrics = Object.values(metrics);
    
    if (allMetrics.length === 0) {
      return {
        totalFunctions: 0,
        totalCalls: 0,
        avgExecutionTime: 0,
        slowestFunction: '',
        slowestTime: 0,
        fastestFunction: '',
        fastestTime: 0,
        mostCalledFunction: '',
        mostCalls: 0,
      };
    }

    let totalCalls = 0;
    let totalExecutionTime = 0;
    let slowestFunction = '';
    let slowestTime = 0;
    let fastestFunction = '';
    let fastestTime = Infinity;
    let mostCalledFunction = '';
    let mostCalls = 0;

    allMetrics.forEach(metric => {
      totalCalls += metric.callCount;
      totalExecutionTime += metric.totalExecutionTime;

      if (metric.maxExecutionTime > slowestTime) {
        slowestTime = metric.maxExecutionTime;
        slowestFunction = metric.functionName;
      }

      if (metric.minExecutionTime < fastestTime && metric.minExecutionTime > 0) {
        fastestTime = metric.minExecutionTime;
        fastestFunction = metric.functionName;
      }

      if (metric.callCount > mostCalls) {
        mostCalls = metric.callCount;
        mostCalledFunction = metric.functionName;
      }
    });

    return {
      totalFunctions: allMetrics.length,
      totalCalls,
      avgExecutionTime: totalCalls > 0 ? totalExecutionTime / totalCalls : 0,
      slowestFunction,
      slowestTime,
      fastestFunction,
      fastestTime,
      mostCalledFunction,
      mostCalls,
    };
  }, [metrics]);

  // Handle sort change
  const handleSortChange = (field: keyof PerformanceMetrics) => {
    if (sortBy === field) {
      // Toggle sort direction if clicking the same field
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      // Set new sort field and default to descending
      setSortBy(field);
      setSortDirection('desc');
    }
  };

  // Format time in milliseconds
  const formatTime = (time: number): string => {
    if (time < 1) {
      return `${(time * 1000).toFixed(2)}μs`;
    } else if (time < 1000) {
      return `${time.toFixed(2)}ms`;
    } else {
      return `${(time / 1000).toFixed(2)}s`;
    }
  };

  // Render sort indicator
  const renderSortIndicator = (field: keyof PerformanceMetrics) => {
    if (sortBy !== field) {
      return null;
    }

    return sortDirection === 'asc' ? '▲' : '▼';
  };

  // Get function details
  const selectedFunctionDetails = selectedFunction ? metrics[selectedFunction] : null;

  // Calculate cache efficiency if available
  const cacheEfficiency = useMemo(() => {
    if (!selectedFunctionDetails || 
        selectedFunctionDetails.cacheHitCount === undefined || 
        selectedFunctionDetails.cacheMissCount === undefined) {
      return null;
    }

    const totalCacheCalls = selectedFunctionDetails.cacheHitCount + selectedFunctionDetails.cacheMissCount;
    if (totalCacheCalls === 0) {
      return 0;
    }

    return (selectedFunctionDetails.cacheHitCount / totalCacheCalls) * 100;
  }, [selectedFunctionDetails]);

  // Calculate performance recommendations
  const getPerformanceRecommendations = (metric: PerformanceMetrics): string[] => {
    const recommendations: string[] = [];

    // Check for slow execution time
    if (metric.averageExecutionTime > 100) {
      recommendations.push('Consider optimizing this function to reduce execution time.');
    }

    // Check for high call count
    if (metric.callCount > 100) {
      recommendations.push('This function is called frequently. Consider memoization or caching results.');
    }

    // Check for cache efficiency if available
    if (metric.cacheHitCount !== undefined && 
        metric.cacheMissCount !== undefined) {
      const totalCacheCalls = metric.cacheHitCount + metric.cacheMissCount;
      if (totalCacheCalls > 0) {
        const hitRatio = (metric.cacheHitCount / totalCacheCalls) * 100;
        if (hitRatio < 50) {
          recommendations.push('Low cache hit ratio. Consider adjusting cache settings or key generation.');
        }
      }
    }

    // Check for high variance in execution time
    if (metric.maxExecutionTime > metric.averageExecutionTime * 5) {
      recommendations.push('High variance in execution time. Check for inconsistent performance.');
    }

    return recommendations;
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg overflow-hidden">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-white">Performance Dashboard</h2>
        
        {/* Overall statistics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-blue-800 dark:text-blue-200">Total Functions</h3>
            <p className="text-2xl font-bold text-blue-900 dark:text-blue-100">{overallStats.totalFunctions}</p>
            <p className="text-sm text-blue-700 dark:text-blue-300">
              Total Calls: {overallStats.totalCalls.toLocaleString()}
            </p>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-green-800 dark:text-green-200">Average Execution Time</h3>
            <p className="text-2xl font-bold text-green-900 dark:text-green-100">
              {formatTime(overallStats.avgExecutionTime)}
            </p>
            <p className="text-sm text-green-700 dark:text-green-300">
              Fastest: {formatTime(overallStats.fastestTime)} ({overallStats.fastestFunction})
            </p>
          </div>
          
          <div className="bg-red-50 dark:bg-red-900 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-red-800 dark:text-red-200">Slowest Function</h3>
            <p className="text-2xl font-bold text-red-900 dark:text-red-100">
              {formatTime(overallStats.slowestTime)}
            </p>
            <p className="text-sm text-red-700 dark:text-red-300">
              {overallStats.slowestFunction}
            </p>
          </div>
        </div>
        
        {/* Search */}
        <div className="mt-4">
          <input
            type="text"
            placeholder="Search functions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200"
          />
        </div>
      </div>
      
      {/* Metrics table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSortChange('functionName')}
              >
                Function {renderSortIndicator('functionName')}
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSortChange('callCount')}
              >
                Calls {renderSortIndicator('callCount')}
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSortChange('averageExecutionTime')}
              >
                Avg Time {renderSortIndicator('averageExecutionTime')}
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSortChange('minExecutionTime')}
              >
                Min Time {renderSortIndicator('minExecutionTime')}
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSortChange('maxExecutionTime')}
              >
                Max Time {renderSortIndicator('maxExecutionTime')}
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSortChange('cacheHitRatio')}
              >
                Cache Hit % {renderSortIndicator('cacheHitRatio')}
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {filteredAndSortedMetrics.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-6 py-4 text-center text-gray-500 dark:text-gray-400">
                  No metrics available
                </td>
              </tr>
            ) : (
              filteredAndSortedMetrics.map((metric) => (
                <tr 
                  key={metric.functionName}
                  className={`hover:bg-gray-50 dark:hover:bg-gray-750 cursor-pointer ${
                    selectedFunction === metric.functionName ? 'bg-blue-50 dark:bg-blue-900' : ''
                  }`}
                  onClick={() => setSelectedFunction(metric.functionName)}
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    {metric.functionName}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                    {metric.callCount.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                    {formatTime(metric.averageExecutionTime)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                    {formatTime(metric.minExecutionTime)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                    {formatTime(metric.maxExecutionTime)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                    {metric.cacheHitRatio !== undefined ? `${(metric.cacheHitRatio * 100).toFixed(1)}%` : 'N/A'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      
      {/* Selected function details */}
      {selectedFunctionDetails && (
        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">
            {selectedFunctionDetails.functionName}
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Performance Metrics</h4>
              
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Total Calls:</span>
                  <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                    {selectedFunctionDetails.callCount.toLocaleString()}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Average Execution Time:</span>
                  <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                    {formatTime(selectedFunctionDetails.averageExecutionTime)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Min Execution Time:</span>
                  <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                    {formatTime(selectedFunctionDetails.minExecutionTime)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Max Execution Time:</span>
                  <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                    {formatTime(selectedFunctionDetails.maxExecutionTime)}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Total Execution Time:</span>
                  <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                    {formatTime(selectedFunctionDetails.totalExecutionTime)}
                  </span>
                </div>
              </div>
            </div>
            
            {/* Cache metrics if available */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Cache Metrics</h4>
              
              {selectedFunctionDetails.cacheHitCount !== undefined ? (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Cache Hits:</span>
                    <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                      {selectedFunctionDetails.cacheHitCount.toLocaleString()}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Cache Misses:</span>
                    <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                      {selectedFunctionDetails.cacheMissCount?.toLocaleString() || 0}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Cache Hit Ratio:</span>
                    <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                      {cacheEfficiency !== null ? `${cacheEfficiency.toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                  
                  {/* Cache efficiency visualization */}
                  {cacheEfficiency !== null && (
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2.5">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full" 
                          style={{ width: `${cacheEfficiency}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-gray-500 dark:text-gray-400">No cache metrics available for this function</p>
              )}
            </div>
          </div>
          
          {/* Performance recommendations */}
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Recommendations</h4>
            
            {getPerformanceRecommendations(selectedFunctionDetails).length > 0 ? (
              <ul className="list-disc pl-5 space-y-1">
                {getPerformanceRecommendations(selectedFunctionDetails).map((recommendation, index) => (
                  <li key={index} className="text-sm text-gray-600 dark:text-gray-400">
                    {recommendation}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-green-500 dark:text-green-400">
                No performance issues detected for this function.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceDashboard;
