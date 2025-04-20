import React, { useState } from 'react';
import ApiHealthDashboard from '../components/dashboard/ApiHealthDashboard';
import { resetCircuitBreaker } from '../api/utils/monitoring';

const ApiHealthPage: React.FC = () => {
  const [timeWindow, setTimeWindow] = useState<number>(60); // Default 1 hour
  const [refreshInterval, setRefreshInterval] = useState<number>(30000); // Default 30 seconds

  // Time window options
  const timeWindowOptions = [
    { value: 15, label: '15 minutes' },
    { value: 30, label: '30 minutes' },
    { value: 60, label: '1 hour' },
    { value: 180, label: '3 hours' },
    { value: 360, label: '6 hours' },
    { value: 720, label: '12 hours' },
    { value: 1440, label: '24 hours' },
  ];

  // Refresh interval options
  const refreshIntervalOptions = [
    { value: 10000, label: '10 seconds' },
    { value: 30000, label: '30 seconds' },
    { value: 60000, label: '1 minute' },
    { value: 300000, label: '5 minutes' },
  ];

  // Reset all circuit breakers
  const handleResetAllCircuitBreakers = () => {
    const exchanges = ['Alpaca', 'Binance', 'Coinbase'];
    const methods = [
      'getPortfolio', 
      'getPositions', 
      'getBalance', 
      'createOrder', 
      'cancelOrder',
      'getOrders',
      'getOrderStatus'
    ];

    exchanges.forEach(exchange => {
      methods.forEach(method => {
        resetCircuitBreaker(exchange, method);
      });
    });

    // Show success message
    alert('All circuit breakers have been reset.');
  };

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-white">API Health Monitoring</h1>
        
        <div className="flex space-x-4">
          {/* Time window selector */}
          <div className="flex items-center">
            <label htmlFor="timeWindow" className="mr-2 text-sm font-medium text-gray-700 dark:text-gray-300">
              Time Window:
            </label>
            <select
              id="timeWindow"
              value={timeWindow}
              onChange={(e) => setTimeWindow(Number(e.target.value))}
              className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-1 px-3 text-sm"
            >
              {timeWindowOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Refresh interval selector */}
          <div className="flex items-center">
            <label htmlFor="refreshInterval" className="mr-2 text-sm font-medium text-gray-700 dark:text-gray-300">
              Refresh Every:
            </label>
            <select
              id="refreshInterval"
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-1 px-3 text-sm"
            >
              {refreshIntervalOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Reset all circuit breakers button */}
          <button
            onClick={handleResetAllCircuitBreakers}
            className="bg-red-500 hover:bg-red-600 text-white py-1 px-4 rounded-md text-sm font-medium transition-colors"
          >
            Reset All Circuit Breakers
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {/* API Health Dashboard */}
        <ApiHealthDashboard 
          refreshInterval={refreshInterval} 
          timeWindow={timeWindow} 
        />

        {/* Additional information panel */}
        <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            About Circuit Breaker Pattern
          </h2>
          <div className="prose dark:prose-invert max-w-none">
            <p>
              The circuit breaker pattern is used to detect failures and prevent the application from 
              repeatedly trying to execute an operation that's likely to fail. This helps to maintain
              system stability during partial outages or when external services are unavailable.
            </p>
            
            <h3>Circuit States</h3>
            <ul>
              <li>
                <strong className="text-green-500">Closed:</strong> The circuit is closed and API calls flow normally. 
                Failures are tracked and if they exceed a threshold, the circuit opens.
              </li>
              <li>
                <strong className="text-yellow-500">Half-Open:</strong> After a timeout period, the circuit transitions 
                to half-open state to test if the underlying problem is fixed. A limited number of test requests are allowed.
              </li>
              <li>
                <strong className="text-red-500">Open:</strong> The circuit is open and API calls are prevented from 
                reaching the service. Fallback mechanisms are used instead. This prevents cascading failures.
              </li>
            </ul>

            <h3>Fallback Mechanisms</h3>
            <p>
              When a circuit is open, the following fallback strategies are used:
            </p>
            <ol>
              <li><strong>Primary Fallback:</strong> Alternative API or service</li>
              <li><strong>Cache Fallback:</strong> Return cached data if available</li>
              <li><strong>Secondary Fallback:</strong> Use a different data source</li>
              <li><strong>Graceful Degradation:</strong> Provide limited functionality</li>
            </ol>

            <p>
              The health score is calculated based on success rate, response time, error frequency, and 
              circuit breaker state. A score above 80 indicates excellent health, while below 40 indicates 
              significant issues that require attention.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ApiHealthPage;
