/**
 * Isolated tests for circuit breaker functionality
 */

// Import specific functions from the monitoring module
import {
  canMakeApiCall,
  recordCircuitBreakerResult,
  resetCircuitBreaker,
  getCircuitBreakerState
} from './monitoring';

describe('Isolated Circuit Breaker Tests', () => {
  // Mock Date.now for predictable test results
  const originalDateNow = Date.now;
  let mockTime = 1000;

  beforeEach(() => {
    // Reset time for each test
    mockTime = 1000;
    Date.now = jest.fn(() => mockTime);
    
    // Reset all circuit breakers used in tests to avoid state interference
    resetCircuitBreaker('IsolatedExchange', 'closedMethod');
    resetCircuitBreaker('IsolatedExchange', 'openMethod');
    resetCircuitBreaker('IsolatedExchange', 'halfOpenMethod');
    resetCircuitBreaker('IsolatedExchange', 'successMethod');
    resetCircuitBreaker('IsolatedExchange', 'failureMethod');
    resetCircuitBreaker('IsolatedExchange', 'resetMethod');
  });

  afterEach(() => {
    // Clear mocks after each test
    jest.clearAllMocks();
  });
  
  afterAll(() => {
    // Restore original Date.now
    Date.now = originalDateNow;
  });

  test('circuit should start in closed state', () => {
    const exchange = 'IsolatedExchange';
    const method = 'closedMethod';
    
    // Circuit should be closed by default
    expect(canMakeApiCall(exchange, method)).toBe(true);
    
    // State should be closed
    const state = getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });

  test('circuit should open after threshold failures', () => {
    const exchange = 'IsolatedExchange';
    const method = 'openMethod';
    
    // Configure a low threshold for testing
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Record failures
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should still be closed
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record one more failure to exceed threshold
    recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should now be open
    expect(canMakeApiCall(exchange, method, config)).toBe(false);
    
    // Verify state
    const state = getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('open');
  });

  test('circuit should transition to half-open after timeout', () => {
    const exchange = 'IsolatedExchange';
    const method = 'halfOpenMethod';
    
    // Configure circuit breaker with short timeout
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    
    // Verify circuit is open
    expect(canMakeApiCall(exchange, method, config)).toBe(false);
    
    // Advance time past timeout
    mockTime += 6000; // 6 seconds
    
    // Circuit should now be half-open
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // State should be half-open
    const state = getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('half-open');
  });

  test('circuit should close after success in half-open state', () => {
    const exchange = 'IsolatedExchange';
    const method = 'successMethod';
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    
    // Advance time to transition to half-open
    mockTime += 6000; // 6 seconds
    
    // Verify circuit is half-open
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record success in half-open state
    recordCircuitBreakerResult(exchange, method, true, config);
    
    // Circuit should be closed
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // State should be closed
    const state = getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });

  test('circuit should reopen after failure in half-open state', () => {
    const exchange = 'IsolatedExchange';
    const method = 'failureMethod';
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    
    // Advance time to transition to half-open
    mockTime += 6000; // 6 seconds
    
    // Verify circuit is half-open
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record failure in half-open state
    recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should be open again
    expect(canMakeApiCall(exchange, method, config)).toBe(false);
    
    // State should be open
    const state = getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('open');
  });

  test('circuit should reset when explicitly requested', () => {
    const exchange = 'IsolatedExchange';
    const method = 'resetMethod';
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    
    // Verify circuit is open
    expect(canMakeApiCall(exchange, method, config)).toBe(false);
    
    // Reset circuit
    resetCircuitBreaker(exchange, method);
    
    // Circuit should be closed
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // State should be closed
    const state = getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });
});
