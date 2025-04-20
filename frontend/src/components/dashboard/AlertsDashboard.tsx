import React, { useState, useEffect } from 'react';
import { useAlerts } from '../../context/AlertsContext';
import {
  Alert,
  AlertSeverity,
  AlertStatus,
  AlertType,
  AlertFilterOptions
} from '../../api/utils/alerts';
import { formatDistanceToNow } from 'date-fns';

interface AlertsDashboardProps {
  defaultSeverity?: AlertSeverity;
  defaultStatus?: AlertStatus;
  defaultType?: AlertType;
  maxAlerts?: number;
}

const AlertsDashboard: React.FC<AlertsDashboardProps> = ({
  defaultSeverity,
  defaultStatus = AlertStatus.ACTIVE,
  defaultType,
  maxAlerts = 50
}) => {
  const {
    alerts,
    getFilteredAlerts,
    acknowledgeAlert,
    resolveAlert,
    ignoreAlert,
    requestNotifications,
    notificationsEnabled
  } = useAlerts();

  // State for filters
  const [selectedSeverity, setSelectedSeverity] = useState<AlertSeverity | undefined>(
    defaultSeverity
  );
  const [selectedStatus, setSelectedStatus] = useState<AlertStatus>(defaultStatus);
  const [selectedType, setSelectedType] = useState<AlertType | undefined>(defaultType);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [timeRange, setTimeRange] = useState<string>('24h'); // 1h, 6h, 24h, 7d, all
  const [expandedAlertId, setExpandedAlertId] = useState<string | null>(null);
  const [filteredAlerts, setFilteredAlerts] = useState<Alert[]>([]);

  // Apply filters when they change
  useEffect(() => {
    // Calculate start time based on time range
    let startTime: number | undefined;
    const now = Date.now();

    switch (timeRange) {
      case '1h':
        startTime = now - 60 * 60 * 1000;
        break;
      case '6h':
        startTime = now - 6 * 60 * 60 * 1000;
        break;
      case '24h':
        startTime = now - 24 * 60 * 60 * 1000;
        break;
      case '7d':
        startTime = now - 7 * 24 * 60 * 60 * 1000;
        break;
      case 'all':
      default:
        startTime = undefined;
        break;
    }

    // Create filter options
    const filterOptions: AlertFilterOptions = {
      severity: selectedSeverity,
      status: selectedStatus,
      type: selectedType,
      startTime,
      limit: maxAlerts
    };

    // Get filtered alerts
    let filtered = getFilteredAlerts(filterOptions);

    // Apply search query if provided
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        alert =>
          alert.title.toLowerCase().includes(query) ||
          alert.message.toLowerCase().includes(query) ||
          (alert.exchange && alert.exchange.toLowerCase().includes(query)) ||
          (alert.method && alert.method.toLowerCase().includes(query))
      );
    }

    setFilteredAlerts(filtered);
  }, [
    selectedSeverity,
    selectedStatus,
    selectedType,
    timeRange,
    searchQuery,
    alerts,
    getFilteredAlerts,
    maxAlerts
  ]);

  // Toggle expanded alert
  const toggleExpandAlert = (alertId: string) => {
    if (expandedAlertId === alertId) {
      setExpandedAlertId(null);
    } else {
      setExpandedAlertId(alertId);
    }
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

  // Get status badge class
  const getStatusBadgeClass = (status: AlertStatus): string => {
    switch (status) {
      case AlertStatus.ACTIVE:
        return 'bg-red-500 text-white';
      case AlertStatus.ACKNOWLEDGED:
        return 'bg-yellow-500 text-white';
      case AlertStatus.RESOLVED:
        return 'bg-green-500 text-white';
      case AlertStatus.IGNORED:
        return 'bg-gray-500 text-white';
      default:
        return 'bg-blue-500 text-white';
    }
  };

  // Format alert timestamp
  const formatAlertTime = (timestamp: number): string => {
    return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
  };

  // Handle alert actions
  const handleAcknowledge = (alertId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    acknowledgeAlert(alertId);
  };

  const handleResolve = (alertId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    resolveAlert(alertId);
  };

  const handleIgnore = (alertId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    ignoreAlert(alertId);
  };

  // Enable browser notifications
  const handleEnableNotifications = async () => {
    await requestNotifications();
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg overflow-hidden">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white">Alerts</h2>
          
          {!notificationsEnabled && (
            <button
              onClick={handleEnableNotifications}
              className="px-3 py-1 bg-blue-500 text-white rounded-md text-sm font-medium"
            >
              Enable Notifications
            </button>
          )}
        </div>

        {/* Filters */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          {/* Severity Filter */}
          <div>
            <label htmlFor="severityFilter" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Severity
            </label>
            <select
              id="severityFilter"
              value={selectedSeverity || ''}
              onChange={(e) => setSelectedSeverity(e.target.value ? e.target.value as AlertSeverity : undefined)}
              className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3 text-sm"
            >
              <option value="">All Severities</option>
              <option value={AlertSeverity.INFO}>Info</option>
              <option value={AlertSeverity.WARNING}>Warning</option>
              <option value={AlertSeverity.ERROR}>Error</option>
              <option value={AlertSeverity.CRITICAL}>Critical</option>
            </select>
          </div>

          {/* Status Filter */}
          <div>
            <label htmlFor="statusFilter" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Status
            </label>
            <select
              id="statusFilter"
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value as AlertStatus)}
              className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3 text-sm"
            >
              <option value={AlertStatus.ACTIVE}>Active</option>
              <option value={AlertStatus.ACKNOWLEDGED}>Acknowledged</option>
              <option value={AlertStatus.RESOLVED}>Resolved</option>
              <option value={AlertStatus.IGNORED}>Ignored</option>
              <option value="">All Statuses</option>
            </select>
          </div>

          {/* Type Filter */}
          <div>
            <label htmlFor="typeFilter" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Type
            </label>
            <select
              id="typeFilter"
              value={selectedType || ''}
              onChange={(e) => setSelectedType(e.target.value ? e.target.value as AlertType : undefined)}
              className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3 text-sm"
            >
              <option value="">All Types</option>
              <option value={AlertType.API_FAILURE}>API Failure</option>
              <option value={AlertType.CIRCUIT_BREAKER}>Circuit Breaker</option>
              <option value={AlertType.TRADE_FAILURE}>Trade Failure</option>
              <option value={AlertType.RATE_LIMIT}>Rate Limit</option>
              <option value={AlertType.AUTHENTICATION}>Authentication</option>
              <option value={AlertType.NETWORK}>Network</option>
              <option value={AlertType.PERFORMANCE}>Performance</option>
              <option value={AlertType.SYSTEM}>System</option>
            </select>
          </div>

          {/* Time Range Filter */}
          <div>
            <label htmlFor="timeRangeFilter" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Time Range
            </label>
            <select
              id="timeRangeFilter"
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3 text-sm"
            >
              <option value="1h">Last Hour</option>
              <option value="6h">Last 6 Hours</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="all">All Time</option>
            </select>
          </div>
        </div>

        {/* Search */}
        <div className="mb-4">
          <input
            type="text"
            placeholder="Search alerts..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3 text-sm"
          />
        </div>
      </div>

      {/* Alerts List */}
      <div className="overflow-y-auto max-h-[600px]">
        {filteredAlerts.length === 0 ? (
          <div className="p-6 text-center text-gray-500 dark:text-gray-400">
            No alerts match your filters
          </div>
        ) : (
          <ul className="divide-y divide-gray-200 dark:divide-gray-700">
            {filteredAlerts.map((alert) => (
              <li
                key={alert.id}
                className={`p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750 ${
                  getSeverityClass(alert.severity)
                } border-l-4`}
                onClick={() => toggleExpandAlert(alert.id)}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center mb-1">
                      <span
                        className={`inline-block px-2 py-1 rounded-full text-xs font-medium mr-2 ${getStatusBadgeClass(
                          alert.status
                        )}`}
                      >
                        {alert.status}
                      </span>
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {formatAlertTime(alert.timestamp)}
                      </span>
                    </div>
                    <h3 className="font-medium text-gray-900 dark:text-white">{alert.title}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">{alert.message}</p>
                    
                    {alert.exchange && (
                      <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                        <span className="font-medium">Exchange:</span> {alert.exchange}
                        {alert.method && ` • Method: ${alert.method}`}
                      </div>
                    )}
                  </div>
                  
                  <div className="flex space-x-2">
                    {alert.status === AlertStatus.ACTIVE && (
                      <>
                        <button
                          onClick={(e) => handleAcknowledge(alert.id, e)}
                          className="px-2 py-1 bg-yellow-500 text-white rounded-md text-xs"
                          title="Acknowledge"
                        >
                          Ack
                        </button>
                        <button
                          onClick={(e) => handleResolve(alert.id, e)}
                          className="px-2 py-1 bg-green-500 text-white rounded-md text-xs"
                          title="Resolve"
                        >
                          Resolve
                        </button>
                        <button
                          onClick={(e) => handleIgnore(alert.id, e)}
                          className="px-2 py-1 bg-gray-500 text-white rounded-md text-xs"
                          title="Ignore"
                        >
                          Ignore
                        </button>
                      </>
                    )}
                    {alert.status === AlertStatus.ACKNOWLEDGED && (
                      <button
                        onClick={(e) => handleResolve(alert.id, e)}
                        className="px-2 py-1 bg-green-500 text-white rounded-md text-xs"
                        title="Resolve"
                      >
                        Resolve
                      </button>
                    )}
                  </div>
                </div>
                
                {/* Expanded Alert Details */}
                {expandedAlertId === alert.id && (
                  <div className="mt-4 p-3 bg-white dark:bg-gray-700 rounded-md border border-gray-200 dark:border-gray-600">
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Alert Details</h4>
                    
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="font-medium text-gray-700 dark:text-gray-300">ID:</span>{' '}
                        <span className="text-gray-600 dark:text-gray-400">{alert.id}</span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-700 dark:text-gray-300">Type:</span>{' '}
                        <span className="text-gray-600 dark:text-gray-400">{alert.type}</span>
                      </div>
                      <div>
                        <span className="font-medium text-gray-700 dark:text-gray-300">Created:</span>{' '}
                        <span className="text-gray-600 dark:text-gray-400">
                          {new Date(alert.timestamp).toLocaleString()}
                        </span>
                      </div>
                      {alert.acknowledgedAt && (
                        <div>
                          <span className="font-medium text-gray-700 dark:text-gray-300">Acknowledged:</span>{' '}
                          <span className="text-gray-600 dark:text-gray-400">
                            {new Date(alert.acknowledgedAt).toLocaleString()}
                          </span>
                        </div>
                      )}
                      {alert.resolvedAt && (
                        <div>
                          <span className="font-medium text-gray-700 dark:text-gray-300">Resolved:</span>{' '}
                          <span className="text-gray-600 dark:text-gray-400">
                            {new Date(alert.resolvedAt).toLocaleString()}
                          </span>
                        </div>
                      )}
                    </div>
                    
                    {alert.metadata && (
                      <div className="mt-3">
                        <h5 className="font-medium text-gray-900 dark:text-white mb-1">Metadata</h5>
                        <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded overflow-x-auto">
                          {JSON.stringify(alert.metadata, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default AlertsDashboard;
