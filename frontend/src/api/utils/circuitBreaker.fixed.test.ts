/**
 * Fixed tests for circuit breaker functionality
 */

// Import directly from the monitoring module
import * as monitoring from './monitoring';

describe('Circuit Breaker Tests', () => {
  // Mock Date.now for predictable test results
  const originalDateNow = Date.now;
  let mockTime = 1000;

  beforeEach(() => {
    // Reset time for each test
    mockTime = 1000;
    Date.now = jest.fn(() => mockTime);
    
    // Reset all circuit breakers used in tests
    monitoring.resetCircuitBreaker('TestExchange', 'startClosedMethod');
    monitoring.resetCircuitBreaker('TestExchange', 'openCircuitMethod');
    monitoring.resetCircuitBreaker('TestExchange', 'halfOpenMethod');
    monitoring.resetCircuitBreaker('TestExchange', 'closeCircuitMethod');
    monitoring.resetCircuitBreaker('TestExchange', 'reopenMethod');
    monitoring.resetCircuitBreaker('TestExchange', 'resetMethod');
  });

  afterEach(() => {
    // Clear mocks after each test
    jest.clearAllMocks();
  });
  
  afterAll(() => {
    // Restore original Date.now
    Date.now = originalDateNow;
  });

  test('should start with closed circuit', () => {
    // Use a unique method name for this test
    const exchange = 'TestExchange';
    const method = 'startClosedMethod';
    
    // Circuit should be closed by default
    expect(monitoring.canMakeApiCall(exchange, method)).toBe(true);
    
    // State should be closed
    const state = monitoring.getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });

  test('should open circuit after failures', () => {
    // Use a unique method name for this test
    const exchange = 'TestExchange';
    const method = 'openCircuitMethod';
    
    // Configure a low threshold for testing
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Record failures
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should still be closed
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record one more failure to exceed threshold
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should now be open
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(false);
  });

  test('should transition to half-open after timeout', () => {
    // Use a unique method name for this test
    const exchange = 'TestExchange';
    const method = 'halfOpenMethod';
    
    // Configure circuit breaker with short timeout
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Verify circuit is open
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(false);
    
    // Advance time past timeout
    mockTime += 6000; // 6 seconds
    
    // Circuit should now be half-open
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(true);
    
    // State should be half-open
    const state = monitoring.getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('half-open');
  });

  test('should close circuit after success in half-open state', () => {
    // Use a unique method name for this test
    const exchange = 'TestExchange';
    const method = 'closeCircuitMethod';
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Advance time to transition to half-open
    mockTime += 6000; // 6 seconds
    
    // Record success in half-open state
    monitoring.recordCircuitBreakerResult(exchange, method, true, config);
    
    // Circuit should be closed
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(true);
    
    // State should be closed
    const state = monitoring.getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });

  test('should reopen circuit after failure in half-open state', () => {
    // Use a unique method name for this test
    const exchange = 'TestExchange';
    const method = 'reopenMethod';
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Advance time to transition to half-open
    mockTime += 6000; // 6 seconds
    
    // Record failure in half-open state
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should be open again
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(false);
    
    // State should be open
    const state = monitoring.getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('open');
  });

  test('should reset circuit breaker when requested', () => {
    // Use a unique method name for this test
    const exchange = 'TestExchange';
    const method = 'resetMethod';
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Verify circuit is open
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(false);
    
    // Reset circuit
    monitoring.resetCircuitBreaker(exchange, method);
    
    // Circuit should be closed
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(true);
  });
});
