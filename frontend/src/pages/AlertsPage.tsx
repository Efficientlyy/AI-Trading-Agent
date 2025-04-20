import React, { useState } from 'react';
import AlertsDashboard from '../components/dashboard/AlertsDashboard';
import { AlertStatus } from '../api/utils/alerts';
import { startAlertChecks } from '../api/utils/alerts';

const AlertsPage: React.FC = () => {
  const [checkInterval, setCheckInterval] = useState<number>(60000); // Default 1 minute

  // Check interval options
  const checkIntervalOptions = [
    { value: 30000, label: '30 seconds' },
    { value: 60000, label: '1 minute' },
    { value: 300000, label: '5 minutes' },
    { value: 600000, label: '10 minutes' },
  ];

  // Manually trigger alert checks
  const triggerAlertCheck = () => {
    // Stop current checks and start new ones
    const stopFn = startAlertChecks(checkInterval);
    
    // We don't need to call stopFn here as startAlertChecks will
    // handle stopping any existing checks
  };

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-white">Alerts</h1>
        
        <div className="flex space-x-4">
          {/* Check interval selector */}
          <div className="flex items-center">
            <label htmlFor="checkInterval" className="mr-2 text-sm font-medium text-gray-700 dark:text-gray-300">
              Check Every:
            </label>
            <select
              id="checkInterval"
              value={checkInterval}
              onChange={(e) => setCheckInterval(Number(e.target.value))}
              className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-1 px-3 text-sm"
            >
              {checkIntervalOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Manual check button */}
          <button
            onClick={triggerAlertCheck}
            className="px-4 py-1 bg-blue-500 text-white rounded-md text-sm font-medium"
          >
            Check Now
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {/* Active Alerts */}
        <div>
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            Active Alerts
          </h2>
          <AlertsDashboard defaultStatus={AlertStatus.ACTIVE} />
        </div>

        {/* Acknowledged Alerts */}
        <div>
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            Acknowledged Alerts
          </h2>
          <AlertsDashboard defaultStatus={AlertStatus.ACKNOWLEDGED} />
        </div>

        {/* Alert Information */}
        <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
            About Alerts
          </h2>
          <div className="prose dark:prose-invert max-w-none">
            <p>
              The Alerts system automatically monitors API performance, circuit breaker states, and trade operations
              to detect potential issues and notify you when problems occur.
            </p>
            
            <h3>Alert Types</h3>
            <ul>
              <li>
                <strong className="text-red-500">API Failure:</strong> Issues with API calls to exchanges
              </li>
              <li>
                <strong className="text-orange-500">Circuit Breaker:</strong> Circuit breaker state changes
              </li>
              <li>
                <strong className="text-yellow-500">Trade Failure:</strong> Failed trade operations
              </li>
              <li>
                <strong className="text-blue-500">Rate Limit:</strong> Rate limiting issues with exchanges
              </li>
              <li>
                <strong className="text-purple-500">Performance:</strong> Slow response times or performance issues
              </li>
            </ul>

            <h3>Alert Severities</h3>
            <ul>
              <li>
                <strong className="text-blue-500">Info:</strong> Informational alerts that don't require immediate action
              </li>
              <li>
                <strong className="text-yellow-500">Warning:</strong> Issues that may require attention but aren't critical
              </li>
              <li>
                <strong className="text-orange-500">Error:</strong> Serious issues that need attention
              </li>
              <li>
                <strong className="text-red-500">Critical:</strong> Severe issues that require immediate action
              </li>
            </ul>

            <h3>Alert Actions</h3>
            <p>
              You can take the following actions on alerts:
            </p>
            <ul>
              <li><strong>Acknowledge:</strong> Mark an alert as seen, but not yet resolved</li>
              <li><strong>Resolve:</strong> Mark an alert as resolved</li>
              <li><strong>Ignore:</strong> Dismiss an alert without resolving it</li>
            </ul>

            <p>
              The system will automatically check for new alerts based on your configured interval.
              You can also manually trigger a check using the "Check Now" button.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlertsPage;
