import React, { useState } from 'react';
import PerformanceDashboard from '../components/dashboard/PerformanceDashboard';

const PerformancePage: React.FC = () => {
  const [refreshInterval, setRefreshInterval] = useState<number>(5000); // Default 5 seconds

  // Refresh interval options
  const refreshIntervalOptions = [
    { value: 2000, label: '2 seconds' },
    { value: 5000, label: '5 seconds' },
    { value: 10000, label: '10 seconds' },
    { value: 30000, label: '30 seconds' },
    { value: 60000, label: '1 minute' },
  ];

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-white">Performance Monitoring</h1>
        
        <div className="flex space-x-4">
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
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {/* Performance Dashboard */}
        <PerformanceDashboard refreshInterval={refreshInterval} />

        {/* Additional information panel */}
        <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            About Performance Monitoring
          </h2>
          <div className="prose dark:prose-invert max-w-none">
            <p>
              The Performance Dashboard provides real-time insights into the performance of your trading application.
              It tracks execution times, cache efficiency, and other metrics to help identify bottlenecks and
              optimize your trading experience.
            </p>
            
            <h3>Key Metrics</h3>
            <ul>
              <li>
                <strong>Average Execution Time:</strong> The average time it takes for a function to complete.
              </li>
              <li>
                <strong>Min/Max Execution Time:</strong> The fastest and slowest recorded executions.
              </li>
              <li>
                <strong>Call Count:</strong> How many times a function has been called.
              </li>
              <li>
                <strong>Cache Hit Ratio:</strong> Percentage of calls that were served from cache instead of executing the full function.
              </li>
            </ul>

            <h3>Optimizations</h3>
            <p>
              The AI Trading Agent implements several performance optimizations:
            </p>
            <ul>
              <li><strong>Memoization:</strong> Caching function results to avoid redundant calculations</li>
              <li><strong>Batch Processing:</strong> Grouping similar API calls to reduce network overhead</li>
              <li><strong>Throttling:</strong> Limiting the frequency of certain operations</li>
              <li><strong>Debouncing:</strong> Preventing rapid-fire execution of expensive operations</li>
            </ul>

            <p>
              The dashboard automatically analyzes performance data and provides recommendations
              for further optimization. Click on any function in the table to see detailed metrics
              and specific recommendations.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformancePage;
