import React, { useState } from 'react';
import { useAlerts } from '../../context/AlertsContext';
import { AlertSeverity, AlertStatus } from '../../api/utils/alerts';
import { Link } from 'react-router-dom';
import { IconBell, IconBellRinging } from '@tabler/icons-react';

interface AlertsIndicatorProps {
  showCount?: boolean;
  maxAlerts?: number;
}

const AlertsIndicator: React.FC<AlertsIndicatorProps> = ({
  showCount = true,
  maxAlerts = 5
}) => {
  const { activeAlerts, hasUnacknowledgedAlerts, hasUnacknowledgedCriticalAlerts, acknowledgeAlert } = useAlerts();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  // Get the color based on alert severity
  const getIndicatorColor = () => {
    if (!hasUnacknowledgedAlerts) {
      return 'text-gray-400 dark:text-gray-500';
    }
    
    if (hasUnacknowledgedCriticalAlerts) {
      return 'text-red-500';
    }
    
    return 'text-yellow-500';
  };

  // Get the icon based on alert status
  const getIcon = () => {
    if (hasUnacknowledgedCriticalAlerts) {
      return <IconBellRinging className={`${getIndicatorColor()} w-6 h-6`} />;
    }
    
    return <IconBell className={`${getIndicatorColor()} w-6 h-6`} />;
  };

  // Toggle dropdown
  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  // Close dropdown
  const closeDropdown = () => {
    setIsDropdownOpen(false);
  };

  // Handle acknowledge
  const handleAcknowledge = (alertId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    acknowledgeAlert(alertId);
  };

  // Get severity class for styling
  const getSeverityClass = (severity: AlertSeverity): string => {
    switch (severity) {
      case AlertSeverity.CRITICAL:
        return 'bg-red-100 border-red-500 text-red-800 dark:bg-red-900 dark:border-red-600 dark:text-red-200';
      case AlertSeverity.ERROR:
        return 'bg-orange-100 border-orange-500 text-orange-800 dark:bg-orange-900 dark:border-orange-600 dark:text-orange-200';
      case AlertSeverity.WARNING:
        return 'bg-yellow-100 border-yellow-500 text-yellow-800 dark:bg-yellow-900 dark:border-yellow-600 dark:text-yellow-200';
      case AlertSeverity.INFO:
      default:
        return 'bg-blue-100 border-blue-500 text-blue-800 dark:bg-blue-900 dark:border-blue-600 dark:text-blue-200';
    }
  };

  return (
    <div className="relative">
      <button
        className="relative p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none"
        onClick={toggleDropdown}
        aria-label="View alerts"
      >
        {getIcon()}
        
        {showCount && activeAlerts.length > 0 && (
          <span className="absolute top-0 right-0 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white transform translate-x-1/2 -translate-y-1/2 rounded-full bg-red-500">
            {activeAlerts.length}
          </span>
        )}
      </button>
      
      {isDropdownOpen && (
        <>
          {/* Backdrop to close dropdown when clicking outside */}
          <div
            className="fixed inset-0 z-10"
            onClick={closeDropdown}
          ></div>
          
          {/* Dropdown content */}
          <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-md shadow-lg overflow-hidden z-20 border border-gray-200 dark:border-gray-700">
            <div className="p-3 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
              <h3 className="font-medium text-gray-700 dark:text-gray-300">
                Recent Alerts
              </h3>
              <Link
                to="/alerts"
                className="text-blue-500 text-sm hover:text-blue-700 dark:hover:text-blue-300"
                onClick={closeDropdown}
              >
                View All
              </Link>
            </div>
            
            <div className="max-h-96 overflow-y-auto">
              {activeAlerts.length === 0 ? (
                <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                  No active alerts
                </div>
              ) : (
                <ul className="divide-y divide-gray-200 dark:divide-gray-700">
                  {activeAlerts.slice(0, maxAlerts).map(alert => (
                    <li
                      key={alert.id}
                      className={`p-3 ${getSeverityClass(alert.severity)} border-l-4`}
                    >
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-medium text-sm">{alert.title}</h4>
                          <p className="text-xs mt-1 text-gray-600 dark:text-gray-300">
                            {alert.message}
                          </p>
                          {alert.exchange && (
                            <p className="text-xs mt-1 text-gray-500 dark:text-gray-400">
                              {alert.exchange} {alert.method && `• ${alert.method}`}
                            </p>
                          )}
                        </div>
                        <button
                          onClick={(e) => handleAcknowledge(alert.id, e)}
                          className="ml-2 px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
                        >
                          Ack
                        </button>
                      </div>
                    </li>
                  ))}
                  
                  {activeAlerts.length > maxAlerts && (
                    <li className="p-2 text-center text-sm text-gray-500 dark:text-gray-400">
                      <Link
                        to="/alerts"
                        className="text-blue-500 hover:text-blue-700 dark:hover:text-blue-300"
                        onClick={closeDropdown}
                      >
                        +{activeAlerts.length - maxAlerts} more alerts
                      </Link>
                    </li>
                  )}
                </ul>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default AlertsIndicator;
