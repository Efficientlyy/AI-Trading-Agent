/**
 * API monitoring and logging utilities
 */

// API call metrics
interface ApiCallMetrics {
  totalCalls: number;
  successCalls: number;
  failedCalls: number;
  totalDuration: number;
  minDuration: number;
  maxDuration: number;
  lastCallTime: number;
  lastError?: Error;
}

// API metrics store
interface ApiMetricsStore {
  [exchange: string]: {
    [method: string]: ApiCallMetrics;
  };
}

// Export ApiCallMetrics type for use in other files
export type { ApiCallMetrics };

// Initialize metrics store
const metricsStore: ApiMetricsStore = {};

// Initialize metrics for a specific API call
const initializeMetrics = (exchange: string, method: string): void => {
  if (!metricsStore[exchange]) {
    metricsStore[exchange] = {};
  }

  if (!metricsStore[exchange][method]) {
    metricsStore[exchange][method] = {
      totalCalls: 0,
      successCalls: 0,
      failedCalls: 0,
      totalDuration: 0,
      minDuration: Infinity,
      maxDuration: 0,
      lastCallTime: 0,
    };
  }
};

// Record API call metrics
export const recordApiCall = (
  exchange: string,
  method: string,
  status: 'attempt' | 'success' | 'failure',
  duration: number = 0,
  error?: Error
): void => {
  initializeMetrics(exchange, method);

  const metrics = metricsStore[exchange][method];

  // Only increment total calls on attempt to avoid double counting
  if (status === 'attempt') {
    metrics.totalCalls += 1;
    metrics.lastCallTime = Date.now();
    return;
  }

  // Update metrics for success or failure
  metrics.totalDuration += duration;
  metrics.minDuration = Math.min(metrics.minDuration, duration);
  metrics.maxDuration = Math.max(metrics.maxDuration, duration);

  if (status === 'success') {
    metrics.successCalls += 1;
  } else if (status === 'failure') {
    metrics.failedCalls += 1;
    metrics.lastError = error;
  }
};

// Get API call metrics
export const getApiCallMetrics = (
  exchange: string,
  method: string
): ApiCallMetrics => {
  if (!metricsStore[exchange] || !metricsStore[exchange][method]) {
    initializeMetrics(exchange, method);
  }

  return metricsStore[exchange][method];
};

// Get all metrics
export const getAllMetrics = (): ApiMetricsStore => {
  return metricsStore;
};

// Calculate success rate for a specific API call
export const getSuccessRate = (
  exchange: string,
  method: string
): number | null => {
  const metrics = getApiCallMetrics(exchange, method);

  if (!metrics || metrics.totalCalls === 0) {
    return null;
  }

  return metrics.successCalls / metrics.totalCalls;
};

// Calculate average duration for a specific API call
export const getAverageDuration = (
  exchange: string,
  method: string
): number | null => {
  const metrics = getApiCallMetrics(exchange, method);

  if (!metrics || metrics.totalCalls === 0) {
    return null;
  }

  return metrics.totalDuration / metrics.totalCalls;
};

// Check if an API is healthy
export const isApiHealthy = (
  exchange: string,
  method?: string,
  thresholdSuccessRate = 0.9,
  maxAllowedFailures = 3
): boolean => {
  if (method) {
    // Check specific method
    const metrics = getApiCallMetrics(exchange, method);

    if (!metrics || metrics.totalCalls === 0) {
      return true; // No data, assume healthy
    }

    const successRate = metrics.successCalls / metrics.totalCalls;
    return (
      successRate >= thresholdSuccessRate ||
      metrics.failedCalls <= maxAllowedFailures
    );
  } else {
    // Check all methods for the exchange
    const exchangeMethods = metricsStore[exchange];

    if (!exchangeMethods) {
      return true; // No data, assume healthy
    }

    // Check if any method is unhealthy
    for (const method in exchangeMethods) {
      if (!isApiHealthy(exchange, method, thresholdSuccessRate, maxAllowedFailures)) {
        return false;
      }
    }

    return true;
  }
};

// Circuit breaker state
interface CircuitBreakerState {
  [key: string]: {
    state: 'closed' | 'open' | 'half-open';
    failureCount: number;
    lastFailureTime: number;
    nextAttemptTime: number;
    halfOpenCallCount: number;  // Track calls in half-open state
    halfOpenSuccessCount: number; // Track successful calls in half-open state
    openCount: number; // Track how many times circuit has opened
    lastStateChangeTime: number; // When the state last changed
  };
}

const circuitBreakers: CircuitBreakerState = {};

// Circuit breaker configuration
export interface CircuitBreakerConfig {
  failureThreshold: number;
  resetTimeoutMs: number;
  halfOpenMaxCalls: number;
  halfOpenSuccessThreshold?: number; // Require multiple successes to close circuit
}

const defaultCircuitBreakerConfig: CircuitBreakerConfig = {
  failureThreshold: 5,
  resetTimeoutMs: 30000, // 30 seconds
  halfOpenMaxCalls: 1,
  halfOpenSuccessThreshold: 1, // Default: one success to close circuit
};

// Initialize circuit breaker
const initializeCircuitBreaker = (key: string): void => {
  if (!circuitBreakers[key]) {
    circuitBreakers[key] = {
      state: 'closed',
      failureCount: 0,
      lastFailureTime: 0,
      nextAttemptTime: 0,
      halfOpenCallCount: 0,
      halfOpenSuccessCount: 0,
      openCount: 0,
      lastStateChangeTime: Date.now(),
    };
  }
};

// Change circuit breaker state with tracking
const changeCircuitBreakerState = (
  key: string,
  newState: 'closed' | 'open' | 'half-open',
  nextAttemptTime: number = 0
): void => {
  const breaker = circuitBreakers[key];

  if (breaker.state !== newState) {
    const now = Date.now();

    // Track state change
    breaker.lastStateChangeTime = now;

    // Reset state-specific counters
    if (newState === 'half-open') {
      breaker.halfOpenCallCount = 0;
      breaker.halfOpenSuccessCount = 0;
    } else if (newState === 'open') {
      breaker.openCount += 1;
    }

    // Set state and next attempt time
    breaker.state = newState;
    breaker.nextAttemptTime = nextAttemptTime;
  }
};

// Check if circuit breaker allows the call
export const canMakeApiCall = (
  exchange: string,
  method: string,
  config: Partial<CircuitBreakerConfig> = {}
): boolean => {
  const key = `${exchange}:${method}`;
  initializeCircuitBreaker(key);

  const breaker = circuitBreakers[key];
  const { halfOpenMaxCalls } = {
    ...defaultCircuitBreakerConfig,
    ...config,
  };

  const now = Date.now();

  // Check circuit breaker state
  switch (breaker.state) {
    case 'closed':
      return true;

    case 'open':
      // Check if it's time to transition to half-open
      if (now >= breaker.nextAttemptTime) {
        changeCircuitBreakerState(key, 'half-open');
        return true;
      }
      return false;

    case 'half-open':
      // Allow limited calls in half-open state
      if (breaker.halfOpenCallCount < halfOpenMaxCalls) {
        breaker.halfOpenCallCount++;
        return true;
      }
      return false;

    default:
      return true;
  }
};

// Record API call result for circuit breaker
export const recordCircuitBreakerResult = (
  exchange: string,
  method: string,
  success: boolean,
  config: Partial<CircuitBreakerConfig> = {}
): void => {
  const key = `${exchange}:${method}`;
  initializeCircuitBreaker(key);

  const breaker = circuitBreakers[key];
  const { failureThreshold, resetTimeoutMs, halfOpenSuccessThreshold } = {
    ...defaultCircuitBreakerConfig,
    ...config,
  };

  const now = Date.now();

  if (success) {
    // Success case
    if (breaker.state === 'half-open') {
      // Increment success count in half-open state
      breaker.halfOpenSuccessCount++;

      // If we have enough successes, close the circuit
      if (breaker.halfOpenSuccessCount >= halfOpenSuccessThreshold!) {
        changeCircuitBreakerState(key, 'closed');
        breaker.failureCount = 0;
      }
    } else if (breaker.state === 'closed') {
      // Reset failure count on success in closed state
      breaker.failureCount = 0;
    }
  } else {
    // Failure case
    breaker.lastFailureTime = now;

    if (breaker.state === 'closed') {
      breaker.failureCount += 1;

      // Check if threshold exceeded
      if (breaker.failureCount >= failureThreshold) {
        changeCircuitBreakerState(key, 'open', now + resetTimeoutMs);
      }
    } else if (breaker.state === 'half-open') {
      // Transition back to open on failure in half-open state
      changeCircuitBreakerState(key, 'open', now + resetTimeoutMs);
    }
  }
};

// Get circuit breaker state with detailed information
export const getCircuitBreakerState = (
  exchange: string,
  method: string
): {
  state: string;
  failureCount: number;
  remainingTimeMs: number;
  halfOpenCallCount?: number;
  halfOpenSuccessCount?: number;
  openCount?: number;
  lastStateChangeTime?: number;
  timeSinceLastChange?: number;
} | null => {
  const key = `${exchange}:${method}`;

  if (!circuitBreakers[key]) {
    return null;
  }

  const breaker = circuitBreakers[key];
  const now = Date.now();

  return {
    state: breaker.state,
    failureCount: breaker.failureCount,
    remainingTimeMs: Math.max(0, breaker.nextAttemptTime - now),
    halfOpenCallCount: breaker.halfOpenCallCount,
    halfOpenSuccessCount: breaker.halfOpenSuccessCount,
    openCount: breaker.openCount,
    lastStateChangeTime: breaker.lastStateChangeTime,
    timeSinceLastChange: now - breaker.lastStateChangeTime
  };
};

// Reset circuit breaker
export const resetCircuitBreaker = (
  exchange: string,
  method: string
): void => {
  const key = `${exchange}:${method}`;

  if (circuitBreakers[key]) {
    changeCircuitBreakerState(key, 'closed');
    circuitBreakers[key].failureCount = 0;
    circuitBreakers[key].openCount = 0;
  }
};

// API health dashboard data
export const getApiHealthDashboard = (): any => {
  const dashboard: any = {
    exchanges: {},
    overallHealth: true,
    totalCalls: 0,
    successRate: 1,
    averageDuration: 0,
  };

  let totalSuccessCalls = 0;
  let totalDuration = 0;

  // Process each exchange
  for (const exchange in metricsStore) {
    dashboard.exchanges[exchange] = {
      methods: {},
      health: true,
      totalCalls: 0,
      successCalls: 0,
      failedCalls: 0,
      successRate: 1,
      averageDuration: 0,
      circuitBreakerStates: {},
    };

    const exchangeData = dashboard.exchanges[exchange];

    // Process each method
    for (const method in metricsStore[exchange]) {
      const metrics = metricsStore[exchange][method];
      const circuitBreakerState = getCircuitBreakerState(exchange, method);

      // Method data
      exchangeData.methods[method] = {
        ...metrics,
        successRate: metrics.totalCalls > 0
          ? metrics.successCalls / metrics.totalCalls
          : 1,
        averageDuration: metrics.successCalls > 0
          ? metrics.totalDuration / metrics.successCalls
          : 0,
        circuitBreaker: circuitBreakerState,
      };

      // Update exchange totals
      exchangeData.totalCalls += metrics.totalCalls;
      exchangeData.successCalls += metrics.successCalls;
      exchangeData.failedCalls += metrics.failedCalls;
      exchangeData.totalDuration = (exchangeData.totalDuration || 0) + metrics.totalDuration;

      // Check method health
      const methodHealth = isApiHealthy(exchange, method);
      if (!methodHealth) {
        exchangeData.health = false;
      }

      // Add circuit breaker state
      if (circuitBreakerState) {
        exchangeData.circuitBreakerStates[method] = circuitBreakerState;
      }
    }

    // Calculate exchange metrics
    if (exchangeData.totalCalls > 0) {
      exchangeData.successRate =
        exchangeData.successCalls / exchangeData.totalCalls;
    }

    if (exchangeData.successCalls > 0) {
      exchangeData.averageDuration = exchangeData.totalDuration / exchangeData.successCalls;
    }

    // Update overall totals
    dashboard.totalCalls += exchangeData.totalCalls;
    totalSuccessCalls += exchangeData.successCalls;
    totalDuration += exchangeData.totalDuration;

    // Update overall health
    if (!exchangeData.health) {
      dashboard.overallHealth = false;
    }
  }

  // Calculate overall metrics
  if (dashboard.totalCalls > 0) {
    dashboard.successRate = totalSuccessCalls / dashboard.totalCalls;
  }

  if (totalSuccessCalls > 0) {
    dashboard.averageDuration = totalDuration / totalSuccessCalls;
  }

  return dashboard;
};
