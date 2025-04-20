import React, { useState, useEffect } from 'react';
import { 
  getLogs, 
  LogEntry, 
  LogLevel, 
  clearLogs 
} from '../../api/utils/enhancedLogging';

// Props for the API Logs Dashboard component
interface ApiLogsDashboardProps {
  refreshInterval?: number; // in milliseconds
  maxLogs?: number;
  defaultExchange?: string;
  defaultMethod?: string;
  defaultLevel?: LogLevel;
}

const ApiLogsDashboard: React.FC<ApiLogsDashboardProps> = ({
  refreshInterval = 5000, // 5 seconds default
  maxLogs = 100,
  defaultExchange,
  defaultMethod,
  defaultLevel = LogLevel.INFO
}) => {
  // State for logs and filters
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedExchange, setSelectedExchange] = useState<string | undefined>(defaultExchange);
  const [selectedMethod, setSelectedMethod] = useState<string | undefined>(defaultMethod);
  const [selectedLevel, setSelectedLevel] = useState<LogLevel>(defaultLevel);
  const [timeRange, setTimeRange] = useState<string>('1h'); // 1h, 6h, 24h, 7d, all
  const [expandedLogIndex, setExpandedLogIndex] = useState<number | null>(null);
  const [exchanges, setExchanges] = useState<string[]>([]);
  const [methods, setMethods] = useState<string[]>([]);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [currentMaxLogs, setCurrentMaxLogs] = useState<number>(maxLogs);

  // Load logs based on filters
  const loadLogs = () => {
    setLoading(true);
    
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
    
    // Get logs with filters
    const filteredLogs = getLogs({
      level: selectedLevel,
      exchange: selectedExchange,
      method: selectedMethod,
      startTime,
      limit: currentMaxLogs
    });
    
    setLogs(filteredLogs);
    
    // Extract unique exchanges and methods for filters
    const uniqueExchanges = Array.from(new Set(filteredLogs.map(log => log.exchange)));
    const uniqueMethods = Array.from(new Set(filteredLogs.map(log => log.method)));
    
    setExchanges(uniqueExchanges);
    setMethods(uniqueMethods);
    setLoading(false);
  };

  // Initialize and set up refresh interval
  useEffect(() => {
    loadLogs();
    
    let interval: NodeJS.Timeout | null = null;
    
    if (autoRefresh) {
      interval = setInterval(loadLogs, refreshInterval);
    }
    
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [
    selectedExchange, 
    selectedMethod, 
    selectedLevel, 
    timeRange, 
    autoRefresh, 
    refreshInterval
  ]);

  // Handle log clearing
  const handleClearLogs = () => {
    if (window.confirm('Are you sure you want to clear all logs? This action cannot be undone.')) {
      clearLogs();
      loadLogs();
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp: number): string => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  // Get color for log level
  const getLevelColor = (level: LogLevel): string => {
    switch (level) {
      case LogLevel.DEBUG:
        return 'text-gray-500';
      case LogLevel.INFO:
        return 'text-blue-500';
      case LogLevel.WARNING:
        return 'text-yellow-500';
      case LogLevel.ERROR:
        return 'text-red-500';
      case LogLevel.CRITICAL:
        return 'text-purple-500';
      default:
        return 'text-gray-700';
    }
  };

  // Get background color for log level
  const getLevelBgColor = (level: LogLevel): string => {
    switch (level) {
      case LogLevel.DEBUG:
        return 'bg-gray-100';
      case LogLevel.INFO:
        return 'bg-blue-50';
      case LogLevel.WARNING:
        return 'bg-yellow-50';
      case LogLevel.ERROR:
        return 'bg-red-50';
      case LogLevel.CRITICAL:
        return 'bg-purple-50';
      default:
        return 'bg-white';
    }
  };

  // Format JSON for display
  const formatJson = (data: any): string => {
    if (!data) return 'null';
    try {
      return JSON.stringify(data, null, 2);
    } catch (error) {
      return String(data);
    }
  };

  // Toggle expanded log
  const toggleExpandLog = (index: number) => {
    if (expandedLogIndex === index) {
      setExpandedLogIndex(null);
    } else {
      setExpandedLogIndex(index);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800 dark:text-white">
          API Logs
        </h2>
        
        <div className="flex space-x-2">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-3 py-1 rounded text-sm font-medium ${
              autoRefresh 
                ? 'bg-green-500 text-white' 
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            {autoRefresh ? 'Auto-refresh On' : 'Auto-refresh Off'}
          </button>
          
          <button
            onClick={loadLogs}
            className="px-3 py-1 bg-blue-500 text-white rounded text-sm font-medium"
          >
            Refresh
          </button>
          
          <button
            onClick={handleClearLogs}
            className="px-3 py-1 bg-red-500 text-white rounded text-sm font-medium"
          >
            Clear Logs
          </button>
        </div>
      </div>
      
      {/* Filters */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
        {/* Log Level Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Log Level
          </label>
          <select
            value={selectedLevel}
            onChange={(e) => setSelectedLevel(e.target.value as LogLevel)}
            className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3 text-sm"
          >
            <option value={LogLevel.DEBUG}>Debug & Above</option>
            <option value={LogLevel.INFO}>Info & Above</option>
            <option value={LogLevel.WARNING}>Warning & Above</option>
            <option value={LogLevel.ERROR}>Error & Above</option>
            <option value={LogLevel.CRITICAL}>Critical Only</option>
          </select>
        </div>
        
        {/* Exchange Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Exchange
          </label>
          <select
            value={selectedExchange || ''}
            onChange={(e) => setSelectedExchange(e.target.value || undefined)}
            className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3 text-sm"
          >
            <option value="">All Exchanges</option>
            {exchanges.map((exchange) => (
              <option key={exchange} value={exchange}>
                {exchange}
              </option>
            ))}
          </select>
        </div>
        
        {/* Method Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Method
          </label>
          <select
            value={selectedMethod || ''}
            onChange={(e) => setSelectedMethod(e.target.value || undefined)}
            className="w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md py-2 px-3 text-sm"
          >
            <option value="">All Methods</option>
            {methods.map((method) => (
              <option key={method} value={method}>
                {method}
              </option>
            ))}
          </select>
        </div>
        
        {/* Time Range Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Time Range
          </label>
          <select
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
      
      {/* Logs Table */}
      {loading ? (
        <div className="flex justify-center items-center h-40">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      ) : logs.length === 0 ? (
        <div className="text-center text-gray-500 dark:text-gray-400 py-8">
          No logs found with the current filters.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Time
                </th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Level
                </th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Exchange
                </th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Method
                </th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Message
                </th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Duration
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {logs.map((log, index) => (
                <React.Fragment key={index}>
                  <tr 
                    className={`hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer ${getLevelBgColor(log.level)}`}
                    onClick={() => toggleExpandLog(index)}
                  >
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {formatTimestamp(log.timestamp)}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm">
                      <span className={`font-medium ${getLevelColor(log.level)}`}>
                        {log.level.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {log.exchange}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {log.method}
                    </td>
                    <td className="px-3 py-2 text-sm text-gray-500 dark:text-gray-400 truncate max-w-xs">
                      {log.message}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {log.duration ? `${log.duration}ms` : '-'}
                    </td>
                  </tr>
                  
                  {/* Expanded Log Details */}
                  {expandedLogIndex === index && (
                    <tr>
                      <td colSpan={6} className="px-3 py-2 text-sm">
                        <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-md">
                          <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Log Details
                          </h4>
                          
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {/* Metadata */}
                            {log.metadata && (
                              <div>
                                <h5 className="font-medium text-gray-600 dark:text-gray-400 mb-1">
                                  Metadata
                                </h5>
                                <pre className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-xs overflow-auto max-h-40">
                                  {formatJson(log.metadata)}
                                </pre>
                              </div>
                            )}
                            
                            {/* Error */}
                            {log.error && (
                              <div>
                                <h5 className="font-medium text-red-600 dark:text-red-400 mb-1">
                                  Error
                                </h5>
                                <pre className="bg-red-50 dark:bg-red-900 dark:bg-opacity-20 p-2 rounded text-xs overflow-auto max-h-40">
                                  {log.error.message}
                                  {log.error.stack && (
                                    <>
                                      <br />
                                      {log.error.stack}
                                    </>
                                  )}
                                </pre>
                              </div>
                            )}
                            
                            {/* Request Data */}
                            {log.requestData && (
                              <div>
                                <h5 className="font-medium text-gray-600 dark:text-gray-400 mb-1">
                                  Request Data
                                </h5>
                                <pre className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-xs overflow-auto max-h-40">
                                  {formatJson(log.requestData)}
                                </pre>
                              </div>
                            )}
                            
                            {/* Response Data */}
                            {log.responseData && (
                              <div>
                                <h5 className="font-medium text-gray-600 dark:text-gray-400 mb-1">
                                  Response Data
                                </h5>
                                <pre className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-xs overflow-auto max-h-40">
                                  {formatJson(log.responseData)}
                                </pre>
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
      
      {/* Pagination or Load More (if needed) */}
      {logs.length >= currentMaxLogs && (
        <div className="mt-4 text-center">
          <button
            onClick={() => setCurrentMaxLogs(currentMaxLogs + 100)}
            className="px-4 py-2 bg-blue-500 text-white rounded-md text-sm font-medium"
          >
            Load More Logs
          </button>
        </div>
      )}
    </div>
  );
};

export default ApiLogsDashboard;
