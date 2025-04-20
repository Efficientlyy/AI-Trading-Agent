import { getEnhancedApiMetrics, EnhancedApiMetrics, getApiHealthDashboard } from './enhancedMonitoring';
import { LogLevel, getLogs, addLogEntry } from './enhancedLogging';

// Define log function to match enhancedLogging.log
const log = (entry: {
  level: LogLevel;
  exchange: string;
  method: string;
  message: string;
  duration?: number;
  requestData?: any;
  responseData?: any;
  error?: Error;
  statusCode?: number;
  metadata?: Record<string, any>;
}) => {
  return addLogEntry({
    timestamp: Date.now(),
    ...entry
  });
};

/**
 * Alert severity levels
 */
export enum AlertSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical'
}

/**
 * Alert types for categorization
 */
export enum AlertType {
  API_FAILURE = 'api_failure',
  CIRCUIT_BREAKER = 'circuit_breaker',
  TRADE_FAILURE = 'trade_failure',
  RATE_LIMIT = 'rate_limit',
  AUTHENTICATION = 'authentication',
  NETWORK = 'network',
  PERFORMANCE = 'performance',
  SYSTEM = 'system'
}

/**
 * Alert status
 */
export enum AlertStatus {
  ACTIVE = 'active',
  ACKNOWLEDGED = 'acknowledged',
  RESOLVED = 'resolved',
  IGNORED = 'ignored'
}

/**
 * Alert interface
 */
export interface Alert {
  id: string;
  timestamp: number;
  severity: AlertSeverity;
  type: AlertType;
  exchange?: string;
  method?: string;
  title: string;
  message: string;
  status: AlertStatus;
  metadata?: any;
  acknowledgedBy?: string;
  acknowledgedAt?: number;
  resolvedBy?: string;
  resolvedAt?: number;
}

/**
 * Alert threshold configuration
 */
export interface AlertThreshold {
  exchange: string;
  method?: string;
  metric: string;
  threshold: number;
  severity: AlertSeverity;
  enabled: boolean;
}

/**
 * Alert filter options
 */
export interface AlertFilterOptions {
  severity?: AlertSeverity;
  type?: AlertType;
  exchange?: string;
  method?: string;
  status?: AlertStatus;
  startTime?: number;
  endTime?: number;
  limit?: number;
}

// In-memory storage for alerts and thresholds
const alerts: Alert[] = [];
const thresholds: AlertThreshold[] = [];

// Default thresholds
const defaultThresholds: AlertThreshold[] = [
  {
    exchange: 'all',
    metric: 'successRate',
    threshold: 0.7, // Alert when success rate drops below 70%
    severity: AlertSeverity.WARNING,
    enabled: true
  },
  {
    exchange: 'all',
    metric: 'successRate',
    threshold: 0.5, // Alert when success rate drops below 50%
    severity: AlertSeverity.ERROR,
    enabled: true
  },
  {
    exchange: 'all',
    metric: 'responseTime',
    threshold: 5000, // Alert when response time exceeds 5 seconds
    severity: AlertSeverity.WARNING,
    enabled: true
  },
  {
    exchange: 'all',
    metric: 'circuitBreakerState',
    threshold: 1, // Alert when circuit breaker opens (1 = open)
    severity: AlertSeverity.ERROR,
    enabled: true
  }
];

// Initialize default thresholds
const initializeDefaultThresholds = () => {
  if (thresholds.length === 0) {
    thresholds.push(...defaultThresholds);
  }
};

// Initialize on module load
initializeDefaultThresholds();

/**
 * Generate a unique ID for alerts
 */
const generateAlertId = (): string => {
  return `alert_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
};

/**
 * Create a new alert
 */
export const createAlert = (
  severity: AlertSeverity,
  type: AlertType,
  title: string,
  message: string,
  metadata?: any,
  exchange?: string,
  method?: string
): Alert => {
  const alert: Alert = {
    id: generateAlertId(),
    timestamp: Date.now(),
    severity,
    type,
    exchange,
    method,
    title,
    message,
    status: AlertStatus.ACTIVE,
    metadata
  };

  // Add to alerts array
  alerts.unshift(alert);

  // Limit the number of stored alerts to prevent memory issues
  if (alerts.length > 1000) {
    alerts.pop();
  }

  // Log the alert
  const logLevel = alertSeverityToLogLevel(severity);
  log({
    level: logLevel,
    exchange: exchange || 'system',
    method: method || 'alerts',
    message: `ALERT: ${title} - ${message}`,
    metadata: {
      alertId: alert.id,
      alertType: type,
      ...metadata
    }
  });

  // Trigger notifications if needed
  triggerAlertNotification(alert);

  return alert;
};

/**
 * Convert alert severity to log level
 */
const alertSeverityToLogLevel = (severity: AlertSeverity): LogLevel => {
  switch (severity) {
    case AlertSeverity.INFO:
      return LogLevel.INFO;
    case AlertSeverity.WARNING:
      return LogLevel.WARNING;
    case AlertSeverity.ERROR:
      return LogLevel.ERROR;
    case AlertSeverity.CRITICAL:
      return LogLevel.CRITICAL;
    default:
      return LogLevel.INFO;
  }
};

/**
 * Trigger notifications for an alert
 */
const triggerAlertNotification = (alert: Alert): void => {
  // Only trigger notifications for WARNING, ERROR, and CRITICAL alerts
  if (
    alert.severity === AlertSeverity.WARNING ||
    alert.severity === AlertSeverity.ERROR ||
    alert.severity === AlertSeverity.CRITICAL
  ) {
    // Check if browser notifications are supported and permission is granted
    if (
      typeof window !== 'undefined' &&
      'Notification' in window &&
      Notification.permission === 'granted'
    ) {
      new Notification(`${alert.severity.toUpperCase()}: ${alert.title}`, {
        body: alert.message,
        icon: '/logo192.png'
      });
    }

    // Trigger sound for CRITICAL alerts
    if (alert.severity === AlertSeverity.CRITICAL && typeof Audio !== 'undefined') {
      const audio = new Audio('/alert-sound.mp3');
      audio.play().catch(e => console.error('Failed to play alert sound:', e));
    }

    // Add visual indicator in the UI (handled by the AlertsProvider)
  }
};

/**
 * Get alerts with optional filtering
 */
export const getAlerts = (options?: AlertFilterOptions): Alert[] => {
  if (!options) {
    return [...alerts];
  }

  let filtered = [...alerts];

  // Apply filters
  if (options.severity) {
    filtered = filtered.filter(alert => alert.severity === options.severity);
  }

  if (options.type) {
    filtered = filtered.filter(alert => alert.type === options.type);
  }

  if (options.exchange) {
    filtered = filtered.filter(alert => alert.exchange === options.exchange);
  }

  if (options.method) {
    filtered = filtered.filter(alert => alert.method === options.method);
  }

  if (options.status) {
    filtered = filtered.filter(alert => alert.status === options.status);
  }

  if (options.startTime !== undefined) {
    filtered = filtered.filter(alert => alert.timestamp >= options.startTime!);
  }

  if (options.endTime !== undefined) {
    filtered = filtered.filter(alert => alert.timestamp <= options.endTime!);
  }

  // Apply limit
  if (options.limit && options.limit > 0) {
    filtered = filtered.slice(0, options.limit);
  }

  return filtered;
};

/**
 * Update alert status
 */
export const updateAlertStatus = (
  alertId: string,
  status: AlertStatus,
  userId?: string
): Alert | null => {
  const alertIndex = alerts.findIndex(alert => alert.id === alertId);
  if (alertIndex === -1) {
    return null;
  }

  const alert = alerts[alertIndex];
  const updatedAlert = { ...alert, status };

  // Add acknowledgement info if status is ACKNOWLEDGED
  if (status === AlertStatus.ACKNOWLEDGED && userId) {
    updatedAlert.acknowledgedBy = userId;
    updatedAlert.acknowledgedAt = Date.now();
  }

  // Add resolution info if status is RESOLVED
  if (status === AlertStatus.RESOLVED && userId) {
    updatedAlert.resolvedBy = userId;
    updatedAlert.resolvedAt = Date.now();
  }

  // Update the alert in the array
  alerts[alertIndex] = updatedAlert;

  return updatedAlert;
};

/**
 * Clear all alerts
 */
export const clearAlerts = (): void => {
  alerts.length = 0;
};

/**
 * Set an alert threshold
 */
export const setAlertThreshold = (
  exchange: string,
  metric: string,
  threshold: number,
  severity: AlertSeverity,
  method?: string
): AlertThreshold => {
  // Check if threshold already exists
  const existingIndex = thresholds.findIndex(
    t => t.exchange === exchange && t.method === method && t.metric === metric
  );

  const thresholdConfig: AlertThreshold = {
    exchange,
    method,
    metric,
    threshold,
    severity,
    enabled: true
  };

  if (existingIndex !== -1) {
    // Update existing threshold
    thresholds[existingIndex] = thresholdConfig;
  } else {
    // Add new threshold
    thresholds.push(thresholdConfig);
  }

  return thresholdConfig;
};

/**
 * Get alert thresholds
 */
export const getAlertThresholds = (
  exchange?: string,
  method?: string
): AlertThreshold[] => {
  if (!exchange) {
    return [...thresholds];
  }

  return thresholds.filter(
    t =>
      (t.exchange === exchange || t.exchange === 'all') &&
      (!method || !t.method || t.method === method)
  );
};

/**
 * Enable or disable an alert threshold
 */
export const setThresholdEnabled = (
  exchange: string,
  metric: string,
  enabled: boolean,
  method?: string
): boolean => {
  const thresholdIndex = thresholds.findIndex(
    t => t.exchange === exchange && t.method === method && t.metric === metric
  );

  if (thresholdIndex === -1) {
    return false;
  }

  thresholds[thresholdIndex].enabled = enabled;
  return true;
};

/**
 * Remove an alert threshold
 */
export const removeAlertThreshold = (
  exchange: string,
  metric: string,
  method?: string
): boolean => {
  const thresholdIndex = thresholds.findIndex(
    t => t.exchange === exchange && t.method === method && t.metric === metric
  );

  if (thresholdIndex === -1) {
    return false;
  }

  thresholds.splice(thresholdIndex, 1);
  return true;
};

/**
 * Check metrics against thresholds and create alerts if needed
 */
export const checkMetricsAgainstThresholds = (): void => {
  try {
    // Get all exchanges with metrics
    const dashboard = getApiHealthDashboard(60); // Use getApiHealthDashboard which returns a summary of all API health
    
    if (!dashboard) {
      console.warn('No API health dashboard data available');
      return;
    }
    
    // Check each exchange's metrics against thresholds
    Object.keys(dashboard).forEach(exchange => {
      const exchangeData = dashboard[exchange];
      
      if (!exchangeData || typeof exchangeData !== 'object') {
        return;
      }
      
      // Check each method's metrics
      Object.keys(exchangeData).forEach(method => {
        const methodSummary = exchangeData[method];
        
        if (!methodSummary) {
          return;
        }
        
        // Get applicable thresholds for this exchange and method
        const applicableThresholds = getAlertThresholds(exchange, method);
        
        // Check each threshold
        applicableThresholds.forEach(threshold => {
          if (!threshold.enabled) {
            return;
          }
          
          // Check different metric types based on the dashboard data structure
          switch (threshold.metric) {
            case 'successRate':
              const successRate = methodSummary.successRate || 0;
              if (successRate < threshold.threshold) {
                createAlert(
                  threshold.severity,
                  AlertType.API_FAILURE,
                  `Low Success Rate for ${exchange} ${method}`,
                  `Success rate (${(successRate * 100).toFixed(1)}%) is below threshold (${(threshold.threshold * 100).toFixed(1)}%)`,
                  { successRate, threshold: threshold.threshold },
                  exchange,
                  method
                );
              }
              break;
              
            case 'circuitBreakerState':
              // The dashboard provides circuitState directly
              const circuitState = methodSummary.circuitState;
              if (circuitState && circuitState === 'open' && threshold.threshold === 1) {
                createAlert(
                  threshold.severity,
                  AlertType.CIRCUIT_BREAKER,
                  `Circuit Breaker Open for ${exchange} ${method}`,
                  `The circuit breaker for ${exchange} ${method} is currently open due to excessive failures`,
                  { circuitState },
                  exchange,
                  method
                );
              }
              break;
              
            // Note: responseTime and errorRate are not directly available in the dashboard
            // We would need to fetch detailed metrics for these
            // For now, we'll skip these checks to prevent errors
          }
        });
        
        // Check health score against a default threshold
        const healthScore = methodSummary.healthScore || 0;
        if (healthScore < 50) { // Default threshold for health score
          createAlert(
            healthScore < 30 ? AlertSeverity.ERROR : AlertSeverity.WARNING,
            AlertType.PERFORMANCE,
            `Low Health Score for ${exchange} ${method}`,
            `API health score (${healthScore.toFixed(0)}) is below acceptable threshold (50)`,
            { healthScore },
            exchange,
            method
          );
        }
      });
    });
  } catch (error) {
    console.error('Error in checkMetricsAgainstThresholds:', error);
  }
};

/**
 * Check logs for trade failures and create alerts
 */
export const checkLogsForTradeFailures = (): void => {
  // Get recent error logs related to trading
  const recentLogs = getLogs({
    level: LogLevel.ERROR,
    startTime: Date.now() - 15 * 60 * 1000, // Last 15 minutes
  });
  
  // Filter for trade-related errors
  const tradeMethods = ['placeOrder', 'cancelOrder', 'modifyOrder'];
  const tradeFailureLogs = recentLogs.filter(log => 
    tradeMethods.some(method => log.method.includes(method))
  );
  
  // Group by exchange and method to avoid duplicate alerts
  const groupedFailures: Record<string, any> = {};
  
  tradeFailureLogs.forEach(log => {
    const key = `${log.exchange}:${log.method}`;
    if (!groupedFailures[key]) {
      groupedFailures[key] = {
        count: 0,
        logs: []
      };
    }
    
    groupedFailures[key].count++;
    groupedFailures[key].logs.push(log);
  });
  
  // Create alerts for trade failures
  Object.keys(groupedFailures).forEach(key => {
    const [exchange, method] = key.split(':');
    const failures = groupedFailures[key];
    
    // Only alert if there are multiple failures (to reduce noise)
    if (failures.count >= 2) {
      createAlert(
        AlertSeverity.ERROR,
        AlertType.TRADE_FAILURE,
        `Multiple ${method} Failures on ${exchange}`,
        `Detected ${failures.count} failures in the last 15 minutes for ${method} on ${exchange}`,
        { failureCount: failures.count, recentLogs: failures.logs.slice(0, 5) },
        exchange,
        method
      );
    }
  });
};

/**
 * Request browser notification permission
 */
export const requestNotificationPermission = async (): Promise<boolean> => {
  if (!('Notification' in window)) {
    console.warn('This browser does not support desktop notifications');
    return false;
  }
  
  if (Notification.permission === 'granted') {
    return true;
  }
  
  if (Notification.permission !== 'denied') {
    const permission = await Notification.requestPermission();
    return permission === 'granted';
  }
  
  return false;
};

/**
 * Start periodic checks for alerts
 * @param intervalMs Interval in milliseconds between checks
 * @returns Function to stop the periodic checks
 */
export const startAlertChecks = (intervalMs: number = 60000): () => void => {
  // Run initial check
  checkMetricsAgainstThresholds();
  checkLogsForTradeFailures();
  
  // Set up interval for periodic checks
  const intervalId = window.setInterval(() => {
    checkMetricsAgainstThresholds();
    checkLogsForTradeFailures();
  }, intervalMs);
  
  // Return function to stop checks
  return () => window.clearInterval(intervalId);
};
