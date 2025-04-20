import { recordApiCall, getApiCallMetrics, getCircuitBreakerState } from './monitoring';

// Define types for the monitoring system
type ApiCallStatus = 'attempt' | 'success' | 'failure';
type EnhancedApiCallStatus = ApiCallStatus | 'fallback_attempt' | 'fallback_success' | 'fallback_failure' | 'fallback_cache_hit' | 'fallback_internal_cache_hit' | 'fallback_secondary_success';
type CircuitBreakerStateType = 'closed' | 'open' | 'half-open';
type ReliabilityTrend = 'improving' | 'stable' | 'degrading';

/**
 * Enhanced metrics for API calls with additional insights
 */
export interface EnhancedApiMetrics {
  /** Basic metrics from the original monitoring system */
  basicMetrics: {
    totalCalls: number;
    successCalls: number;
    failedCalls: number;
    totalDuration: number;
    minDuration: number;
    maxDuration: number;
    lastCallTime: number;
    lastError?: Error;
  };
  
  /** Enhanced metrics */
  enhancedMetrics: {
    /** Success rate as a percentage */
    successRate: number;
    /** Average response time in milliseconds */
    averageResponseTime: number;
    /** Number of fallback attempts */
    fallbackAttempts: number;
    /** Number of successful fallbacks */
    fallbackSuccesses: number;
    /** Fallback success rate as a percentage */
    fallbackSuccessRate: number;
    /** Number of circuit breaker openings */
    circuitBreakerOpenings: number;
    /** Average time between circuit breaker openings in milliseconds */
    averageTimeBetweenOpenings: number;
    /** Health score from 0-100 based on multiple factors */
    healthScore: number;
    /** Reliability trend (improving, stable, degrading) */
    reliabilityTrend: 'improving' | 'stable' | 'degrading';
    /** Last 10 response times for trend analysis */
    recentResponseTimes: number[];
    /** Error categories and counts */
    errorCategories: Record<string, number>;
  };
  
  /** Circuit breaker state */
  circuitBreakerState: {
    state: 'closed' | 'open' | 'half-open';
    failureCount: number;
    lastFailureTime: number;
    nextAttemptTime: number;
    remainingTimeMs: number;
  };
  
  /** Time window for metrics */
  timeWindow: {
    start: number;
    end: number;
    durationMinutes: number;
  };
}

// Store enhanced metrics history for trend analysis
interface MetricsHistoryEntry {
  timestamp: number;
  successRate: number;
  averageResponseTime: number;
  circuitBreakerState: string;
}

// Store metrics history for each exchange and method
const metricsHistory: Record<string, Record<string, MetricsHistoryEntry[]>> = {};

// Store error categories
const errorCategories: Record<string, Record<string, Record<string, number>>> = {};

// Store circuit breaker state changes
interface CircuitBreakerStateChange {
  timestamp: number;
  fromState: string;
  toState: string;
  reason?: string;
}

const circuitBreakerStateChanges: Record<string, Record<string, CircuitBreakerStateChange[]>> = {};

/**
 * Record a circuit breaker state change
 */
export const recordCircuitBreakerStateChange = (
  exchange: string,
  method: string,
  fromState: string,
  toState: string,
  reason?: string
): void => {
  // Initialize if needed
  if (!circuitBreakerStateChanges[exchange]) {
    circuitBreakerStateChanges[exchange] = {};
  }
  
  if (!circuitBreakerStateChanges[exchange][method]) {
    circuitBreakerStateChanges[exchange][method] = [];
  }
  
  // Record state change
  circuitBreakerStateChanges[exchange][method].push({
    timestamp: Date.now(),
    fromState,
    toState,
    reason
  });
  
  // Log state change
  console.log(`[Circuit Breaker] ${exchange}.${method}: ${fromState} -> ${toState}${reason ? ` (${reason})` : ''}`);
};

/**
 * Categorize an error for better analysis
 */
export const categorizeError = (
  exchange: string,
  method: string,
  error: Error
): string => {
  // Initialize if needed
  if (!errorCategories[exchange]) {
    errorCategories[exchange] = {};
  }
  
  if (!errorCategories[exchange][method]) {
    errorCategories[exchange][method] = {};
  }
  
  // Determine error category
  let category = 'unknown';
  
  if (error.name === 'NetworkError' || error.message.includes('network')) {
    category = 'network';
  } else if (error.message.includes('timeout')) {
    category = 'timeout';
  } else if (error.message.includes('rate limit') || error.message.includes('too many requests')) {
    category = 'rate_limit';
  } else if (error.message.includes('authentication') || error.message.includes('unauthorized')) {
    category = 'authentication';
  } else if (error.message.includes('not found')) {
    category = 'not_found';
  } else if (error.message.includes('server error') || error.message.includes('500')) {
    category = 'server_error';
  } else if (error.message.includes('validation') || error.message.includes('invalid')) {
    category = 'validation';
  }
  
  // Increment error category count
  if (!errorCategories[exchange][method][category]) {
    errorCategories[exchange][method][category] = 0;
  }
  
  errorCategories[exchange][method][category]++;
  
  return category;
};

/**
 * Record enhanced API call metrics
 */
export const recordEnhancedApiCall = (
  exchange: string,
  method: string,
  status: EnhancedApiCallStatus,
  duration: number = 0,
  error?: Error
): void => {
  // Record in base monitoring system
  if (['attempt', 'success', 'failure'].includes(status)) {
    recordApiCall(exchange, method, status as ApiCallStatus, duration, error);
  }
  
  // Initialize metrics history if needed
  if (!metricsHistory[exchange]) {
    metricsHistory[exchange] = {};
  }
  
  if (!metricsHistory[exchange][method]) {
    metricsHistory[exchange][method] = [];
  }
  
  // If it's a success, record response time for trend analysis
  if (status === 'success' && duration > 0) {
    // Get current metrics
    const metrics = getApiCallMetrics(exchange, method);
    if (!metrics) return;
    
    const successRate = metrics.successCalls / Math.max(1, metrics.totalCalls);
    const avgResponseTime = metrics.totalDuration / Math.max(1, metrics.successCalls);
    
    // Get circuit breaker state
    const cbState = getCircuitBreakerState(exchange, method);
    
    // Add to history
    metricsHistory[exchange][method].push({
      timestamp: Date.now(),
      successRate,
      averageResponseTime: avgResponseTime,
      circuitBreakerState: cbState?.state as CircuitBreakerStateType || 'closed'
    });
    
    // Keep only last 100 entries
    if (metricsHistory[exchange][method].length > 100) {
      metricsHistory[exchange][method].shift();
    }
  }
  
  // Categorize errors
  if (status === 'failure' && error) {
    categorizeError(exchange, method, error);
  }
};

/**
 * Calculate a health score from 0-100 based on multiple factors
 */
const calculateHealthScore = (
  successRate: number,
  avgResponseTime: number,
  maxAcceptableResponseTime: number,
  circuitBreakerState: string,
  errorCategoryDistribution: Record<string, number>
): number => {
  // Base score from success rate (0-60 points)
  let score = successRate * 60;
  
  // Response time score (0-20 points)
  // Lower is better, with diminishing returns
  const responseTimeScore = Math.max(0, 20 - (avgResponseTime / maxAcceptableResponseTime) * 20);
  score += responseTimeScore;
  
  // Circuit breaker state penalty
  if (circuitBreakerState === 'open') {
    score -= 30; // Heavy penalty for open circuit
  } else if (circuitBreakerState === 'half-open') {
    score -= 15; // Moderate penalty for half-open
  }
  
  // Error distribution penalty
  // More severe errors get higher penalties
  let errorPenalty = 0;
  const totalErrors = Object.values(errorCategoryDistribution).reduce((sum, count) => sum + count, 0);
  
  if (totalErrors > 0) {
    // Calculate weighted penalty based on error types
    const severityWeights = {
      'network': 0.5,
      'timeout': 0.7,
      'rate_limit': 0.6,
      'authentication': 0.9,
      'not_found': 0.4,
      'server_error': 0.8,
      'validation': 0.3,
      'unknown': 0.5
    };
    
    for (const [category, count] of Object.entries(errorCategoryDistribution)) {
      const weight = severityWeights[category as keyof typeof severityWeights] || 0.5;
      errorPenalty += (count / totalErrors) * weight * 20; // Max 20 points penalty
    }
    
    score -= errorPenalty;
  }
  
  // Ensure score is between 0-100
  return Math.max(0, Math.min(100, score));
};

/**
 * Determine reliability trend based on historical data
 */
const determineReliabilityTrend = (
  history: MetricsHistoryEntry[]
): 'improving' | 'stable' | 'degrading' => {
  if (history.length < 5) return 'stable'; // Not enough data
  
  // Get recent entries for trend analysis
  const recentEntries = history.slice(-5);
  
  // Calculate success rate trend
  const oldSuccessRate = recentEntries[0].successRate;
  const newSuccessRate = recentEntries[recentEntries.length - 1].successRate;
  const successRateDiff = newSuccessRate - oldSuccessRate;
  
  // Calculate response time trend (lower is better)
  const oldResponseTime = recentEntries[0].averageResponseTime;
  const newResponseTime = recentEntries[recentEntries.length - 1].averageResponseTime;
  const responseTimeDiff = oldResponseTime - newResponseTime; // Positive means improving
  
  // Check circuit breaker state changes
  const recentlyOpened = recentEntries.some(entry => entry.circuitBreakerState === 'open');
  
  // Determine overall trend
  if (successRateDiff > 0.1 || responseTimeDiff > 50) {
    return 'improving';
  } else if (successRateDiff < -0.1 || responseTimeDiff < -50 || recentlyOpened) {
    return 'degrading';
  } else {
    return 'stable';
  }
};

/**
 * Get enhanced metrics for an API
 */
export const getEnhancedApiMetrics = (
  exchange: string,
  method: string,
  timeWindowMinutes: number = 60
): EnhancedApiMetrics => {
  // Get basic metrics
  const basicMetrics = getApiCallMetrics(exchange, method) || {
    totalCalls: 0,
    successCalls: 0,
    failedCalls: 0,
    totalDuration: 0,
    minDuration: Infinity,
    maxDuration: 0,
    lastCallTime: 0
  };
  
  // Get circuit breaker state
  const cbState = getCircuitBreakerState(exchange, method) || {
    state: 'closed' as CircuitBreakerStateType,
    failureCount: 0,
    lastFailureTime: 0,
    nextAttemptTime: 0,
    remainingTimeMs: 0
  };
  
  // Calculate time window
  const endTime = Date.now();
  const startTime = endTime - (timeWindowMinutes * 60 * 1000);
  
  // Get metrics history within time window
  const history = (metricsHistory[exchange]?.[method] || [])
    .filter(entry => entry.timestamp >= startTime);
  
  // Calculate enhanced metrics
  
  // Success rate
  const successRate = basicMetrics.successCalls / Math.max(1, basicMetrics.totalCalls);
  
  // Average response time
  const avgResponseTime = basicMetrics.totalDuration / Math.max(1, basicMetrics.successCalls);
  
  // Get fallback metrics
  const fallbackAttempts = 0; // This would come from the fallback tracking
  const fallbackSuccesses = 0; // This would come from the fallback tracking
  const fallbackSuccessRate = fallbackAttempts > 0 ? fallbackSuccesses / fallbackAttempts : 0;
  
  // Get circuit breaker openings
  const cbOpenings = (circuitBreakerStateChanges[exchange]?.[method] || [])
    .filter(change => change.toState === 'open' && change.timestamp >= startTime)
    .length;
  
  // Calculate average time between openings
  let avgTimeBetweenOpenings = 0;
  const openings = (circuitBreakerStateChanges[exchange]?.[method] || [])
    .filter(change => change.toState === 'open');
  
  if (openings.length > 1) {
    let totalTime = 0;
    for (let i = 1; i < openings.length; i++) {
      totalTime += openings[i].timestamp - openings[i-1].timestamp;
    }
    avgTimeBetweenOpenings = totalTime / (openings.length - 1);
  }
  
  // Get recent response times
  const recentResponseTimes = history
    .slice(-10)
    .map(entry => entry.averageResponseTime);
  
  // Get error categories
  const errorCats = errorCategories[exchange]?.[method] || {};
  
  // Calculate health score
  const healthScore = calculateHealthScore(
    successRate,
    avgResponseTime,
    2000, // 2 seconds as max acceptable response time
    cbState.state,
    errorCats
  );
  
  // Determine reliability trend
  const reliabilityTrend = determineReliabilityTrend(history);
  
  return {
    basicMetrics: basicMetrics || {
      totalCalls: 0,
      successCalls: 0,
      failedCalls: 0,
      totalDuration: 0,
      minDuration: 0,
      maxDuration: 0,
      lastCallTime: 0
    },
    enhancedMetrics: {
      successRate,
      averageResponseTime: avgResponseTime,
      fallbackAttempts,
      fallbackSuccesses,
      fallbackSuccessRate,
      circuitBreakerOpenings: cbOpenings,
      averageTimeBetweenOpenings: avgTimeBetweenOpenings,
      healthScore,
      reliabilityTrend,
      recentResponseTimes,
      errorCategories: errorCats
    },
    circuitBreakerState: cbState as any,
    timeWindow: {
      start: startTime,
      end: endTime,
      durationMinutes: timeWindowMinutes
    }
  };
};

/**
 * Get a dashboard summary of all API health
 */
export const getApiHealthDashboard = (
  timeWindowMinutes: number = 60
): Record<string, Record<string, {
  healthScore: number;
  reliability: 'improving' | 'stable' | 'degrading';
  circuitState: string;
  successRate: number;
}>> => {
  const dashboard: Record<string, Record<string, any>> = {};
  
  // Collect all exchanges and methods with metrics
  for (const exchange of Object.keys(metricsHistory)) {
    dashboard[exchange] = {};
    
    for (const method of Object.keys(metricsHistory[exchange])) {
      const metrics = getEnhancedApiMetrics(exchange, method, timeWindowMinutes);
      
      dashboard[exchange][method] = {
        healthScore: metrics.enhancedMetrics.healthScore,
        reliability: metrics.enhancedMetrics.reliabilityTrend,
        circuitState: metrics.circuitBreakerState.state,
        successRate: metrics.enhancedMetrics.successRate
      };
    }
  }
  
  return dashboard;
};

/**
 * Get detailed error analysis for an API
 */
export const getApiErrorAnalysis = (
  exchange: string,
  method: string,
  timeWindowMinutes: number = 60
): {
  categories: Record<string, number>;
  trend: 'improving' | 'stable' | 'worsening';
  mostCommonCategory: string;
  recommendations: string[];
} => {
  // Get error categories
  const categories = errorCategories[exchange]?.[method] || {};
  
  // Determine most common category
  let mostCommonCategory = 'unknown';
  let maxCount = 0;
  
  for (const [category, count] of Object.entries(categories)) {
    if (count > maxCount) {
      maxCount = count;
      mostCommonCategory = category;
    }
  }
  
  // Get circuit breaker state changes to analyze trend
  const stateChanges = (circuitBreakerStateChanges[exchange]?.[method] || [])
    .filter(change => change.timestamp >= Date.now() - (timeWindowMinutes * 60 * 1000));
  
  // Determine error trend
  let trend: 'improving' | 'stable' | 'worsening' = 'stable';
  
  if (stateChanges.length > 3) {
    // More than 3 state changes in the time window is concerning
    trend = 'worsening';
  } else if (stateChanges.length === 0 && Object.values(categories).reduce((sum, count) => sum + count, 0) > 0) {
    // Errors but no circuit breaker openings suggests resilience
    trend = 'improving';
  }
  
  // Generate recommendations based on error categories
  const recommendations: string[] = [];
  
  if (categories['network'] && categories['network'] > 3) {
    recommendations.push('Check network connectivity and DNS resolution');
  }
  
  if (categories['timeout'] && categories['timeout'] > 3) {
    recommendations.push('Consider increasing request timeouts or optimizing API calls');
  }
  
  if (categories['rate_limit'] && categories['rate_limit'] > 1) {
    recommendations.push('Implement rate limiting controls and backoff strategies');
  }
  
  if (categories['authentication'] && categories['authentication'] > 0) {
    recommendations.push('Verify API credentials and token refresh mechanisms');
  }
  
  if (categories['server_error'] && categories['server_error'] > 3) {
    recommendations.push('Contact the exchange support team about potential server issues');
  }
  
  if (Object.keys(categories).length === 0) {
    recommendations.push('No errors recorded in the specified time window');
  }
  
  return {
    categories,
    trend,
    mostCommonCategory,
    recommendations
  };
};
