/**
 * Circuit breaker tests for the monitoring utilities
 */
import { 
  canMakeApiCall, 
  recordCircuitBreakerResult, 
  resetCircuitBreaker, 
  getCircuitBreakerState 
} from './monitoring';

describe('Circuit Breaker Tests', () => {
  // Mock Date.now for predictable test results
  const originalDateNow = Date.now;
  let currentTime = 1000;

  beforeEach(() => {
    // Reset time for each test
    currentTime = 1000;
    Date.now = jest.fn(() => currentTime);
    
    // Reset circuit breakers for test exchanges
    resetCircuitBreaker('TestExchange', 'testMethod');
  });

  afterAll(() => {
    // Restore original Date.now
    Date.now = originalDateNow;
  });

  test('should start with closed circuit', () => {
    // Circuit should be closed by default
    expect(canMakeApiCall('TestExchange', 'testMethod')).toBe(true);
    
    // State should be closed
    const state = getCircuitBreakerState('TestExchange', 'testMethod');
    expect(state?.state).toBe('closed');
  });

  test('should open circuit after failures', () => {
    // Configure a low threshold for testing
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Record failures
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    
    // Circuit should still be closed
    expect(canMakeApiCall('TestExchange', 'testMethod', config)).toBe(true);
    
    // Record one more failure to exceed threshold
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    
    // Circuit should now be open
    expect(canMakeApiCall('TestExchange', 'testMethod', config)).toBe(false);
  });

  test('should transition to half-open after timeout', () => {
    // Configure circuit breaker with short timeout
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    
    // Verify circuit is open
    expect(canMakeApiCall('TestExchange', 'testMethod', config)).toBe(false);
    
    // Advance time past timeout
    currentTime += 6000; // 6 seconds
    
    // Circuit should now be half-open
    expect(canMakeApiCall('TestExchange', 'testMethod', config)).toBe(true);
    
    // State should be half-open
    const state = getCircuitBreakerState('TestExchange', 'testMethod');
    expect(state?.state).toBe('half-open');
  });

  test('should close circuit after success in half-open state', () => {
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    
    // Advance time to transition to half-open
    currentTime += 6000; // 6 seconds
    
    // Record success in half-open state
    recordCircuitBreakerResult('TestExchange', 'testMethod', true, config);
    
    // Circuit should be closed
    expect(canMakeApiCall('TestExchange', 'testMethod', config)).toBe(true);
    
    // State should be closed
    const state = getCircuitBreakerState('TestExchange', 'testMethod');
    expect(state?.state).toBe('closed');
  });

  test('should reopen circuit after failure in half-open state', () => {
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    
    // Advance time to transition to half-open
    currentTime += 6000; // 6 seconds
    
    // Record failure in half-open state
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    
    // Circuit should be open again
    expect(canMakeApiCall('TestExchange', 'testMethod', config)).toBe(false);
    
    // State should be open
    const state = getCircuitBreakerState('TestExchange', 'testMethod');
    expect(state?.state).toBe('open');
  });

  test('should reset circuit breaker when requested', () => {
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
    
    // Verify circuit is open
    expect(canMakeApiCall('TestExchange', 'testMethod', config)).toBe(false);
    
    // Reset circuit
    resetCircuitBreaker('TestExchange', 'testMethod');
    
    // Circuit should be closed
    expect(canMakeApiCall('TestExchange', 'testMethod', config)).toBe(true);
  });
});
