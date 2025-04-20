import React, { useState } from 'react';
import ApiLogsDashboard from '../components/dashboard/ApiLogsDashboard';
import { LogLevel } from '../api/utils/enhancedLogging';

const ApiLogsPage: React.FC = () => {
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
        <h1 className="text-2xl font-bold text-gray-800 dark:text-white">API Logs</h1>
        
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
        {/* API Logs Dashboard */}
        <ApiLogsDashboard 
          refreshInterval={refreshInterval} 
          maxLogs={200}
          defaultLevel={LogLevel.INFO}
        />

        {/* Additional information panel */}
        <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            About API Logging
          </h2>
          <div className="prose dark:prose-invert max-w-none">
            <p>
              The API Logs dashboard provides detailed insights into all API interactions across different exchanges.
              This helps with debugging, performance monitoring, and understanding system behavior during normal
              operation and failure scenarios.
            </p>
            
            <h3>Log Levels</h3>
            <ul>
              <li>
                <strong className="text-gray-500">DEBUG:</strong> Detailed information for debugging purposes.
              </li>
              <li>
                <strong className="text-blue-500">INFO:</strong> Normal operational information, successful API calls.
              </li>
              <li>
                <strong className="text-yellow-500">WARNING:</strong> Issues that don't prevent operation but require attention,
                such as fallbacks being used or rate limiting.
              </li>
              <li>
                <strong className="text-red-500">ERROR:</strong> Errors that prevent normal operation but don't cause
                system-wide failures, such as individual API call failures.
              </li>
              <li>
                <strong className="text-purple-500">CRITICAL:</strong> Severe errors that may lead to system-wide failures
                or data corruption.
              </li>
            </ul>

            <h3>Log Details</h3>
            <p>
              Click on any log entry to view detailed information, including:
            </p>
            <ul>
              <li><strong>Request Data:</strong> The data sent in the API request</li>
              <li><strong>Response Data:</strong> The data received from the API</li>
              <li><strong>Error Details:</strong> Detailed error information for failed calls</li>
              <li><strong>Metadata:</strong> Additional context about the operation</li>
            </ul>

            <p>
              Logs are stored in browser local storage and will persist across sessions. You can clear logs
              using the "Clear Logs" button if they become too numerous or are no longer needed.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ApiLogsPage;
