/**
 * Circuit Breaker Mock for Testing
 * 
 * This module provides mock implementations of the circuit breaker functions
 * that can be used in tests to avoid state interference between tests.
 */

// Mock state storage
type CircuitState = 'closed' | 'open' | 'half-open';

interface CircuitBreakerState {
  state: CircuitState;
  failureCount: number;
  lastFailureTime: number;
  nextAttemptTime: number;
}

interface CircuitBreakerConfig {
  failureThreshold: number;
  resetTimeoutMs: number;
}

// Default configuration
const defaultConfig: CircuitBreakerConfig = {
  failureThreshold: 5,
  resetTimeoutMs: 60000, // 1 minute
};

// Mock storage
const mockCircuitBreakers: Record<string, CircuitBreakerState> = {};

// Mock time function that can be controlled in tests
let mockTimeNow = Date.now();

/**
 * Set the mock time for testing
 */
export const setMockTime = (time: number): void => {
  mockTimeNow = time;
};

/**
 * Advance the mock time by a specified amount
 */
export const advanceMockTime = (milliseconds: number): void => {
  mockTimeNow += milliseconds;
};

/**
 * Reset all mock circuit breakers
 */
export const resetAllCircuitBreakers = (): void => {
  Object.keys(mockCircuitBreakers).forEach(key => {
    mockCircuitBreakers[key] = {
      state: 'closed',
      failureCount: 0,
      lastFailureTime: 0,
      nextAttemptTime: 0
    };
  });
};

/**
 * Check if an API call can be made based on circuit breaker state
 */
export const mockCanMakeApiCall = (
  exchange: string,
  method: string,
  config: Partial<CircuitBreakerConfig> = {}
): boolean => {
  const key = `${exchange}:${method}`;
  
  // Initialize if not exists
  if (!mockCircuitBreakers[key]) {
    mockCircuitBreakers[key] = {
      state: 'closed',
      failureCount: 0,
      lastFailureTime: 0,
      nextAttemptTime: 0
    };
  }
  
  const breaker = mockCircuitBreakers[key];
  
  if (breaker.state === 'open') {
    // Check if timeout has elapsed to transition to half-open
    if (mockTimeNow >= breaker.nextAttemptTime) {
      breaker.state = 'half-open';
      return true;
    }
    return false;
  }
  
  return true;
};

/**
 * Record a circuit breaker result
 */
export const mockRecordCircuitBreakerResult = (
  exchange: string,
  method: string,
  success: boolean,
  config: Partial<CircuitBreakerConfig> = {}
): void => {
  const key = `${exchange}:${method}`;
  const { failureThreshold, resetTimeoutMs } = {
    ...defaultConfig,
    ...config,
  };
  
  // Initialize if not exists
  if (!mockCircuitBreakers[key]) {
    mockCircuitBreakers[key] = {
      state: 'closed',
      failureCount: 0,
      lastFailureTime: 0,
      nextAttemptTime: 0
    };
  }
  
  const breaker = mockCircuitBreakers[key];
  
  if (success) {
    // Success case
    if (breaker.state === 'half-open') {
      // Reset on success in half-open state
      breaker.state = 'closed';
      breaker.failureCount = 0;
    } else if (breaker.state === 'closed') {
      // Reset failure count on success
      breaker.failureCount = 0;
    }
  } else {
    // Failure case
    breaker.lastFailureTime = mockTimeNow;
    
    if (breaker.state === 'closed') {
      breaker.failureCount += 1;
      
      // Check if threshold exceeded
      if (breaker.failureCount >= failureThreshold) {
        breaker.state = 'open';
        breaker.nextAttemptTime = mockTimeNow + resetTimeoutMs;
      }
    } else if (breaker.state === 'half-open') {
      // Transition back to open on failure
      breaker.state = 'open';
      breaker.nextAttemptTime = mockTimeNow + resetTimeoutMs;
    }
  }
};

/**
 * Reset a circuit breaker
 */
export const mockResetCircuitBreaker = (
  exchange: string,
  method: string
): void => {
  const key = `${exchange}:${method}`;
  
  mockCircuitBreakers[key] = {
    state: 'closed',
    failureCount: 0,
    lastFailureTime: 0,
    nextAttemptTime: 0
  };
};

/**
 * Get the state of a circuit breaker
 */
export const mockGetCircuitBreakerState = (
  exchange: string,
  method: string
): CircuitBreakerState | undefined => {
  const key = `${exchange}:${method}`;
  return mockCircuitBreakers[key];
};
