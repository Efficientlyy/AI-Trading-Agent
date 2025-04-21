/**
 * Working tests for circuit breaker functionality
 */

// Import directly from the monitoring module
import * as monitoring from './monitoring';

describe('Circuit Breaker Tests', () => {
  // Mock Date.now for predictable test results
  const originalDateNow = Date.now;
  let mockTime = 1000;

  beforeAll(() => {
    // Set up Date.now mock
    Date.now = jest.fn(() => mockTime);
  });

  afterAll(() => {
    // Restore original Date.now
    Date.now = originalDateNow;
  });

  // Test the circuit breaker functionality
  test('circuit breaker should open after failures', () => {
    const exchange = 'TestExchange';
    const method = 'testMethod';
    
    // Use a low threshold for testing
    const config = { failureThreshold: 2, resetTimeoutMs: 1000 };
    
    // Reset circuit breaker to ensure clean state
    monitoring.resetCircuitBreaker(exchange, method);
    
    // Initially circuit should be closed
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record failures to open the circuit
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should now be open
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(false);
    
    // Get circuit state
    const state = monitoring.getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('open');
  });

  test('circuit breaker should transition to half-open after timeout', () => {
    const exchange = 'TimeoutExchange';
    const method = 'timeoutMethod';
    
    // Use a short timeout for testing
    const config = { failureThreshold: 2, resetTimeoutMs: 1000 };
    
    // Reset circuit breaker to ensure clean state
    monitoring.resetCircuitBreaker(exchange, method);
    
    // Open the circuit
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Verify circuit is open
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(false);
    
    // Advance time past timeout
    mockTime += 1500; // 1.5 seconds
    
    // Circuit should now be half-open
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Get state
    const state = monitoring.getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('half-open');
  });

  test('circuit breaker should close after success in half-open state', () => {
    const exchange = 'HalfOpenExchange';
    const method = 'halfOpenMethod';
    
    // Use a short timeout for testing
    const config = { failureThreshold: 2, resetTimeoutMs: 1000 };
    
    // Reset circuit breaker to ensure clean state
    monitoring.resetCircuitBreaker(exchange, method);
    
    // Open the circuit
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Advance time to transition to half-open
    mockTime += 1500; // 1.5 seconds
    
    // Verify circuit is half-open by making a call
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record success in half-open state
    monitoring.recordCircuitBreakerResult(exchange, method, true, config);
    
    // Circuit should now be closed
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Get state
    const state = monitoring.getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });

  test('circuit breaker should reopen after failure in half-open state', () => {
    const exchange = 'ReopenExchange';
    const method = 'reopenMethod';
    
    // Use a short timeout for testing
    const config = { failureThreshold: 2, resetTimeoutMs: 1000 };
    
    // Reset circuit breaker to ensure clean state
    monitoring.resetCircuitBreaker(exchange, method);
    
    // Open the circuit
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Advance time to transition to half-open
    mockTime += 1500; // 1.5 seconds
    
    // Verify circuit is half-open by making a call
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record failure in half-open state
    monitoring.recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should be open again
    expect(monitoring.canMakeApiCall(exchange, method, config)).toBe(false);
    
    // Get state
    const state = monitoring.getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('open');
  });
});
