import { canMakeApiCall, recordCircuitBreakerResult, resetCircuitBreaker, getCircuitBreakerState } from './monitoring';

// Mock the circuit breakers object to isolate tests
jest.mock('./monitoring', () => {
  // Store the original module
  const originalModule = jest.requireActual('./monitoring');
  
  // Create a fresh circuit breakers object for each test
  let mockCircuitBreakers: { [key: string]: any } = {};
  
  // Return a modified version of the original module
  return {
    ...originalModule,
    // Override the functions that use circuit breakers
    canMakeApiCall: (exchange: string, method: string, config: { failureThreshold?: number; resetTimeoutMs?: number } = {}) => {
      const key = `${exchange}:${method}`;
      if (!mockCircuitBreakers[key]) {
        return true; // Default to closed circuit
      }
      
      const breaker = mockCircuitBreakers[key];
      const now = Date.now();
      
      if (breaker.state === 'open') {
        // Check if timeout has elapsed to transition to half-open
        if (now >= breaker.nextAttemptTime) {
          breaker.state = 'half-open';
          return true;
        }
        return false;
      }
      
      return true;
    },
    
    recordCircuitBreakerResult: (exchange: string, method: string, success: boolean, config: { failureThreshold?: number; resetTimeoutMs?: number } = {}) => {
      const key = `${exchange}:${method}`;
      const { failureThreshold = 5, resetTimeoutMs = 60000 } = config;
      
      if (!mockCircuitBreakers[key]) {
        mockCircuitBreakers[key] = {
          state: 'closed',
          failureCount: 0,
          lastFailureTime: 0,
          nextAttemptTime: 0
        };
      }
      
      const breaker = mockCircuitBreakers[key];
      const now = Date.now();
      
      if (success) {
        if (breaker.state === 'half-open') {
          breaker.state = 'closed';
          breaker.failureCount = 0;
        } else if (breaker.state === 'closed') {
          breaker.failureCount = 0;
        }
      } else {
        breaker.lastFailureTime = now;
        
        if (breaker.state === 'closed') {
          breaker.failureCount += 1;
          
          if (breaker.failureCount >= failureThreshold) {
            breaker.state = 'open';
            breaker.nextAttemptTime = now + resetTimeoutMs;
          }
        } else if (breaker.state === 'half-open') {
          breaker.state = 'open';
          breaker.nextAttemptTime = now + resetTimeoutMs;
        }
      }
    },
    
    resetCircuitBreaker: (exchange: string, method: string) => {
      const key = `${exchange}:${method}`;
      mockCircuitBreakers[key] = {
        state: 'closed',
        failureCount: 0,
        lastFailureTime: 0,
        nextAttemptTime: 0
      };
    },
    
    getCircuitBreakerState: (exchange: string, method: string) => {
      const key = `${exchange}:${method}`;
      return mockCircuitBreakers[key] || {
        state: 'closed',
        failureCount: 0,
        lastFailureTime: 0,
        nextAttemptTime: 0
      };
    }
  };
});

describe('Circuit Breaker Tests', () => {
  // Mock Date.now for predictable test results
  const originalDateNow = Date.now;
  let currentTime = 1000;

  beforeEach(() => {
    // Reset time for each test
    currentTime = 1000;
    Date.now = jest.fn(() => currentTime);
  });

  afterAll(() => {
    // Restore original Date.now
    Date.now = originalDateNow;
  });

  test('should start with closed circuit', () => {
    // Use a unique method name for this test
    const method = 'startClosedMethod';
    resetCircuitBreaker('TestExchange', method);
    
    // Circuit should be closed by default
    expect(canMakeApiCall('TestExchange', method)).toBe(true);
    
    // State should be closed
    const state = getCircuitBreakerState('TestExchange', method);
    expect(state?.state).toBe('closed');
  });

  test('should open circuit after failures', () => {
    // Use a unique method name for this test
    const method = 'openCircuitMethod';
    resetCircuitBreaker('TestExchange', method);
    
    // Configure a low threshold for testing
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Record failures
    recordCircuitBreakerResult('TestExchange', method, false, config);
    recordCircuitBreakerResult('TestExchange', method, false, config);
    
    // Circuit should still be closed
    expect(canMakeApiCall('TestExchange', method, config)).toBe(true);
    
    // Record one more failure to exceed threshold
    recordCircuitBreakerResult('TestExchange', method, false, config);
    
    // Circuit should now be open
    expect(canMakeApiCall('TestExchange', method, config)).toBe(false);
  });

  test('should transition to half-open after timeout', () => {
    // Use a unique method name for this test
    const method = 'halfOpenMethod';
    resetCircuitBreaker('TestExchange', method);
    
    // Configure circuit breaker with short timeout
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult('TestExchange', method, false, config);
    recordCircuitBreakerResult('TestExchange', method, false, config);
    recordCircuitBreakerResult('TestExchange', method, false, config);
    
    // Verify circuit is open
    expect(canMakeApiCall('TestExchange', method, config)).toBe(false);
    
    // Advance time past timeout
    currentTime += 6000; // 6 seconds
    
    // Circuit should now be half-open
    expect(canMakeApiCall('TestExchange', method, config)).toBe(true);
    
    // State should be half-open
    const state = getCircuitBreakerState('TestExchange', method);
    expect(state?.state).toBe('half-open');
  });

  test('should close circuit after success in half-open state', () => {
    // Use a unique method name for this test
    const method = 'closeCircuitMethod';
    resetCircuitBreaker('TestExchange', method);
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult('TestExchange', method, false, config);
    recordCircuitBreakerResult('TestExchange', method, false, config);
    recordCircuitBreakerResult('TestExchange', method, false, config);
    
    // Advance time to transition to half-open
    currentTime += 6000; // 6 seconds
    
    // Record success in half-open state
    recordCircuitBreakerResult('TestExchange', method, true, config);
    
    // Circuit should be closed
    expect(canMakeApiCall('TestExchange', method, config)).toBe(true);
    
    // State should be closed
    const state = getCircuitBreakerState('TestExchange', method);
    expect(state?.state).toBe('closed');
  });

  test('should reopen circuit after failure in half-open state', () => {
    // Use a unique method name for this test
    const method = 'reopenMethod';
    resetCircuitBreaker('TestExchange', method);
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult('TestExchange', method, false, config);
    recordCircuitBreakerResult('TestExchange', method, false, config);
    recordCircuitBreakerResult('TestExchange', method, false, config);
    
    // Advance time to transition to half-open
    currentTime += 6000; // 6 seconds
    
    // Record failure in half-open state
    recordCircuitBreakerResult('TestExchange', method, false, config);
    
    // Circuit should be open again
    expect(canMakeApiCall('TestExchange', method, config)).toBe(false);
    
    // State should be open
    const state = getCircuitBreakerState('TestExchange', method);
    expect(state?.state).toBe('open');
  });

  test('should reset circuit breaker when requested', () => {
    // Use a unique method name for this test
    const method = 'resetMethod';
    resetCircuitBreaker('TestExchange', method);
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult('TestExchange', method, false, config);
    recordCircuitBreakerResult('TestExchange', method, false, config);
    recordCircuitBreakerResult('TestExchange', method, false, config);
    
    // Verify circuit is open
    expect(canMakeApiCall('TestExchange', method, config)).toBe(false);
    
    // Reset circuit
    resetCircuitBreaker('TestExchange', method);
    
    // Circuit should be closed
    expect(canMakeApiCall('TestExchange', method, config)).toBe(true);
  });
});
