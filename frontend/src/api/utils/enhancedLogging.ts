/**
 * Enhanced logging utilities for API calls
 * Provides structured logging, filtering, and persistence for API interactions
 */

import { ApiError, NetworkError } from './errorHandling';

// Log levels
export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical'
}

// Log entry interface
export interface LogEntry {
  timestamp: number;
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
}

// Log storage options
export interface LogStorageOptions {
  maxEntries?: number;
  persistToLocalStorage?: boolean;
  localStorageKey?: string;
  consoleOutput?: boolean;
  filterLevel?: LogLevel;
  filterExchanges?: string[];
  filterMethods?: string[];
}

// Default storage options
const DEFAULT_STORAGE_OPTIONS: LogStorageOptions = {
  maxEntries: 1000,
  persistToLocalStorage: true,
  localStorageKey: 'api_logs',
  consoleOutput: true,
  filterLevel: LogLevel.INFO
};

// In-memory log storage
let logEntries: LogEntry[] = [];
let storageOptions: LogStorageOptions = DEFAULT_STORAGE_OPTIONS;

/**
 * Initialize the logging system with custom options
 * @param options Log storage options
 */
export const initLogging = (options: LogStorageOptions = {}): void => {
  storageOptions = { ...DEFAULT_STORAGE_OPTIONS, ...options };
  
  // Load existing logs from localStorage if enabled
  if (storageOptions.persistToLocalStorage && typeof window !== 'undefined') {
    try {
      const storedLogs = localStorage.getItem(storageOptions.localStorageKey || 'api_logs');
      if (storedLogs) {
        logEntries = JSON.parse(storedLogs);
      }
    } catch (error) {
      console.error('Failed to load logs from localStorage:', error);
    }
  }
};

/**
 * Add a log entry to the storage
 * @param entry Log entry to add
 */
export const addLogEntry = (entry: LogEntry): void => {
  // Apply filters
  if (shouldLogEntry(entry)) {
    // Add to in-memory storage
    logEntries.unshift(entry);
    
    // Trim to max entries
    if (storageOptions.maxEntries && logEntries.length > storageOptions.maxEntries) {
      logEntries = logEntries.slice(0, storageOptions.maxEntries);
    }
    
    // Persist to localStorage if enabled
    if (storageOptions.persistToLocalStorage && typeof window !== 'undefined') {
      try {
        localStorage.setItem(
          storageOptions.localStorageKey || 'api_logs',
          JSON.stringify(logEntries)
        );
      } catch (error) {
        console.error('Failed to save logs to localStorage:', error);
      }
    }
    
    // Output to console if enabled
    if (storageOptions.consoleOutput) {
      logToConsole(entry);
    }
  }
};

/**
 * Check if a log entry should be stored based on filters
 * @param entry Log entry to check
 * @returns Whether the entry should be logged
 */
const shouldLogEntry = (entry: LogEntry): boolean => {
  // Filter by level
  if (storageOptions.filterLevel) {
    const levels = [
      LogLevel.DEBUG,
      LogLevel.INFO,
      LogLevel.WARNING,
      LogLevel.ERROR,
      LogLevel.CRITICAL
    ];
    const entryLevelIndex = levels.indexOf(entry.level);
    const filterLevelIndex = levels.indexOf(storageOptions.filterLevel);
    
    if (entryLevelIndex < filterLevelIndex) {
      return false;
    }
  }
  
  // Filter by exchange
  if (storageOptions.filterExchanges && storageOptions.filterExchanges.length > 0) {
    if (!storageOptions.filterExchanges.includes(entry.exchange)) {
      return false;
    }
  }
  
  // Filter by method
  if (storageOptions.filterMethods && storageOptions.filterMethods.length > 0) {
    if (!storageOptions.filterMethods.includes(entry.method)) {
      return false;
    }
  }
  
  return true;
};

/**
 * Output a log entry to the console
 * @param entry Log entry to output
 */
const logToConsole = (entry: LogEntry): void => {
  const timestamp = new Date(entry.timestamp).toISOString();
  const prefix = `[${timestamp}] [${entry.level.toUpperCase()}] [${entry.exchange}] [${entry.method}]`;
  
  switch (entry.level) {
    case LogLevel.DEBUG:
      console.debug(`${prefix} ${entry.message}`, entry.metadata || '');
      break;
    case LogLevel.INFO:
      console.info(`${prefix} ${entry.message}`, entry.metadata || '');
      break;
    case LogLevel.WARNING:
      console.warn(`${prefix} ${entry.message}`, entry.metadata || '');
      break;
    case LogLevel.ERROR:
    case LogLevel.CRITICAL:
      console.error(`${prefix} ${entry.message}`, entry.error || entry.metadata || '');
      break;
  }
};

/**
 * Get all log entries, optionally filtered
 * @param options Filter options
 * @returns Filtered log entries
 */
export const getLogs = (options: {
  level?: LogLevel;
  exchange?: string;
  method?: string;
  startTime?: number;
  endTime?: number;
  limit?: number;
} = {}): LogEntry[] => {
  let filtered = [...logEntries];
  
  // Apply filters
  if (options.level) {
    filtered = filtered.filter(entry => entry.level === options.level);
  }
  
  if (options.exchange) {
    filtered = filtered.filter(entry => entry.exchange === options.exchange);
  }
  
  if (options.method) {
    filtered = filtered.filter(entry => entry.method === options.method);
  }
  
  if (options.startTime !== undefined) {
    filtered = filtered.filter(entry => entry.timestamp >= options.startTime!);
  }
  
  if (options.endTime !== undefined) {
    filtered = filtered.filter(entry => entry.timestamp <= options.endTime!);
  }
  
  // Apply limit
  if (options.limit && options.limit > 0) {
    filtered = filtered.slice(0, options.limit);
  }
  
  return filtered;
};

/**
 * Clear all logs
 */
export const clearLogs = (): void => {
  logEntries = [];
  
  // Clear localStorage if enabled
  if (storageOptions.persistToLocalStorage && typeof window !== 'undefined') {
    try {
      localStorage.removeItem(storageOptions.localStorageKey || 'api_logs');
    } catch (error) {
      console.error('Failed to clear logs from localStorage:', error);
    }
  }
};

/**
 * Log an API call attempt
 * @param exchange Exchange name
 * @param method Method name
 * @param requestData Request data
 */
export const logApiCallAttempt = (
  exchange: string,
  method: string,
  requestData?: any
): void => {
  addLogEntry({
    timestamp: Date.now(),
    level: LogLevel.INFO,
    exchange,
    method,
    message: `API call attempt: ${exchange}.${method}`,
    requestData,
    metadata: { status: 'attempt' }
  });
};

/**
 * Log a successful API call
 * @param exchange Exchange name
 * @param method Method name
 * @param duration Call duration in ms
 * @param responseData Response data
 * @param requestData Request data
 */
export const logApiCallSuccess = (
  exchange: string,
  method: string,
  duration: number,
  responseData?: any,
  requestData?: any
): void => {
  addLogEntry({
    timestamp: Date.now(),
    level: LogLevel.INFO,
    exchange,
    method,
    message: `API call success: ${exchange}.${method} (${duration}ms)`,
    duration,
    requestData,
    responseData,
    metadata: { status: 'success' }
  });
};

/**
 * Log a failed API call
 * @param exchange Exchange name
 * @param method Method name
 * @param error Error object
 * @param duration Call duration in ms
 * @param requestData Request data
 */
export const logApiCallFailure = (
  exchange: string,
  method: string,
  error: Error,
  duration: number,
  requestData?: any
): void => {
  // Determine log level based on error type
  let level = LogLevel.ERROR;
  let statusCode: number | undefined;
  
  if (error instanceof ApiError) {
    statusCode = error.status;
    
    // Rate limiting is a warning, not an error
    if (error.status === 429) {
      level = LogLevel.WARNING;
    }
  } else if (error instanceof NetworkError) {
    level = LogLevel.WARNING;
  }
  
  addLogEntry({
    timestamp: Date.now(),
    level,
    exchange,
    method,
    message: `API call failure: ${exchange}.${method} - ${error.message}`,
    error,
    duration,
    requestData,
    statusCode,
    metadata: { status: 'failure' }
  });
};

/**
 * Log a fallback attempt
 * @param exchange Exchange name
 * @param method Method name
 * @param fallbackType Type of fallback
 */
export const logFallbackAttempt = (
  exchange: string,
  method: string,
  fallbackType: 'primary' | 'secondary' | 'cache'
): void => {
  addLogEntry({
    timestamp: Date.now(),
    level: LogLevel.WARNING,
    exchange,
    method,
    message: `Fallback attempt: ${exchange}.${method} (${fallbackType})`,
    metadata: { status: 'fallback_attempt', fallbackType }
  });
};

/**
 * Log a successful fallback
 * @param exchange Exchange name
 * @param method Method name
 * @param fallbackType Type of fallback
 * @param duration Call duration in ms
 */
export const logFallbackSuccess = (
  exchange: string,
  method: string,
  fallbackType: 'primary' | 'secondary' | 'cache',
  duration: number
): void => {
  addLogEntry({
    timestamp: Date.now(),
    level: LogLevel.WARNING,
    exchange,
    method,
    message: `Fallback success: ${exchange}.${method} (${fallbackType}, ${duration}ms)`,
    duration,
    metadata: { status: 'fallback_success', fallbackType }
  });
};

/**
 * Log a failed fallback
 * @param exchange Exchange name
 * @param method Method name
 * @param fallbackType Type of fallback
 * @param error Error object
 */
export const logFallbackFailure = (
  exchange: string,
  method: string,
  fallbackType: 'primary' | 'secondary' | 'cache',
  error: Error
): void => {
  addLogEntry({
    timestamp: Date.now(),
    level: LogLevel.ERROR,
    exchange,
    method,
    message: `Fallback failure: ${exchange}.${method} (${fallbackType}) - ${error.message}`,
    error,
    metadata: { status: 'fallback_failure', fallbackType }
  });
};

/**
 * Log a circuit breaker state change
 * @param exchange Exchange name
 * @param method Method name
 * @param previousState Previous circuit breaker state
 * @param newState New circuit breaker state
 * @param reason Reason for state change
 */
export const logCircuitBreakerStateChange = (
  exchange: string,
  method: string,
  previousState: string,
  newState: string,
  reason: string
): void => {
  const level = newState === 'open' ? LogLevel.ERROR : LogLevel.WARNING;
  
  addLogEntry({
    timestamp: Date.now(),
    level,
    exchange,
    method,
    message: `Circuit breaker state change: ${exchange}.${method} (${previousState} -> ${newState}) - ${reason}`,
    metadata: {
      status: 'circuit_breaker_change',
      previousState,
      newState,
      reason
    }
  });
};

// Initialize logging with default options
initLogging();
