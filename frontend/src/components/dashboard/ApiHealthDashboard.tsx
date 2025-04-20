import React, { useState, useEffect } from 'react';
import { getApiHealthDashboard, getApiErrorAnalysis } from '../../api/utils/enhancedMonitoring';
import { resetCircuitBreaker } from '../../api/utils/monitoring';

// Define types for the dashboard
type ApiHealthStatus = {
  healthScore: number;
  reliability: 'improving' | 'stable' | 'degrading';
  circuitState: string;
  successRate: number;
};

type ApiHealthDashboardProps = {
  refreshInterval?: number; // in milliseconds
  timeWindow?: number; // in minutes
};

const ApiHealthDashboard: React.FC<ApiHealthDashboardProps> = ({
  refreshInterval = 30000, // 30 seconds default
  timeWindow = 60, // 1 hour default
}) => {
  const [healthData, setHealthData] = useState<Record<string, Record<string, ApiHealthStatus>>>({});
  const [selectedExchange, setSelectedExchange] = useState<string | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);
  const [errorAnalysis, setErrorAnalysis] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Load health data
  const loadHealthData = () => {
    try {
      const dashboard = getApiHealthDashboard(timeWindow);
      setHealthData(dashboard);
      setIsLoading(false);
    } catch (error) {
      console.error('Error loading API health data:', error);
      setIsLoading(false);
    }
  };

  // Load error analysis for selected API
  const loadErrorAnalysis = () => {
    if (selectedExchange && selectedMethod) {
      try {
        const analysis = getApiErrorAnalysis(selectedExchange, selectedMethod, timeWindow);
        setErrorAnalysis(analysis);
      } catch (error) {
        console.error('Error loading API error analysis:', error);
        setErrorAnalysis(null);
      }
    } else {
      setErrorAnalysis(null);
    }
  };

  // Reset circuit breaker for selected API
  const handleResetCircuitBreaker = () => {
    if (selectedExchange && selectedMethod) {
      resetCircuitBreaker(selectedExchange, selectedMethod);
      loadHealthData(); // Reload data after reset
      loadErrorAnalysis(); // Reload error analysis after reset
    }
  };

  // Initialize data and set up refresh interval
  useEffect(() => {
    loadHealthData();
    const interval = setInterval(loadHealthData, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval, timeWindow]);

  // Load error analysis when selection changes
  useEffect(() => {
    loadErrorAnalysis();
  }, [selectedExchange, selectedMethod, timeWindow]);

  // Get health score color based on score value
  const getHealthScoreColor = (score: number): string => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-green-300';
    if (score >= 40) return 'bg-yellow-400';
    if (score >= 20) return 'bg-orange-500';
    return 'bg-red-500';
  };

  // Get reliability trend icon
  const getReliabilityIcon = (trend: string): JSX.Element => {
    switch (trend) {
      case 'improving':
        return <span className="text-green-500">↑</span>;
      case 'degrading':
        return <span className="text-red-500">↓</span>;
      default:
        return <span className="text-gray-500">→</span>;
    }
  };

  // Get circuit state color
  const getCircuitStateColor = (state: string): string => {
    switch (state) {
      case 'closed':
        return 'bg-green-500';
      case 'half-open':
        return 'bg-yellow-400';
      case 'open':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Format success rate as percentage
  const formatSuccessRate = (rate: number): string => {
    return `${(rate * 100).toFixed(1)}%`;
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-4">
      <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
        API Health Dashboard
      </h2>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-40">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      ) : Object.keys(healthData).length === 0 ? (
        <div className="text-center text-gray-500 dark:text-gray-400 py-8">
          No API health data available. Start making API calls to see metrics.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {/* Exchange tabs */}
          <div className="flex overflow-x-auto space-x-2 pb-2">
            {Object.keys(healthData).map((exchange) => (
              <button
                key={exchange}
                className={`px-4 py-2 rounded-lg font-medium ${
                  selectedExchange === exchange
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
                onClick={() => {
                  setSelectedExchange(exchange);
                  setSelectedMethod(null);
                }}
              >
                {exchange}
              </button>
            ))}
          </div>

          {/* API methods table */}
          {selectedExchange && (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-700">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      API Method
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Health Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Reliability
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Circuit State
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Success Rate
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {Object.entries(healthData[selectedExchange]).map(([method, status]) => (
                    <tr
                      key={method}
                      className={`hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer ${
                        selectedMethod === method ? 'bg-blue-50 dark:bg-blue-900' : ''
                      }`}
                      onClick={() => setSelectedMethod(method)}
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                        {method}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 mr-2">
                            <div
                              className={`h-2.5 rounded-full ${getHealthScoreColor(status.healthScore)}`}
                              style={{ width: `${status.healthScore}%` }}
                            ></div>
                          </div>
                          <span>{status.healthScore.toFixed(0)}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                        <div className="flex items-center">
                          {getReliabilityIcon(status.reliability)}
                          <span className="ml-1">{status.reliability}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                        <span
                          className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full text-white ${getCircuitStateColor(
                            status.circuitState
                          )}`}
                        >
                          {status.circuitState}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-300">
                        {formatSuccessRate(status.successRate)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Error analysis panel */}
          {selectedExchange && selectedMethod && errorAnalysis && (
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mt-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                  Error Analysis: {selectedExchange}.{selectedMethod}
                </h3>
                <button
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                  onClick={handleResetCircuitBreaker}
                >
                  Reset Circuit Breaker
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Error Categories
                  </h4>
                  {Object.keys(errorAnalysis.categories).length > 0 ? (
                    <div className="space-y-2">
                      {Object.entries(errorAnalysis.categories).map(([category, count]) => (
                        <div key={category} className="flex justify-between">
                          <span className="text-sm text-gray-600 dark:text-gray-400">
                            {category}
                          </span>
                          <span className="text-sm font-medium text-gray-900 dark:text-white">
                            {count as number}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      No errors recorded in the specified time window.
                    </p>
                  )}
                </div>

                <div>
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Recommendations
                  </h4>
                  {errorAnalysis.recommendations.length > 0 ? (
                    <ul className="list-disc pl-5 space-y-1">
                      {errorAnalysis.recommendations.map((recommendation: string, index: number) => (
                        <li
                          key={index}
                          className="text-sm text-gray-600 dark:text-gray-400"
                        >
                          {recommendation}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      No recommendations available.
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ApiHealthDashboard;
