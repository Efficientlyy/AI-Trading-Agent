/**
 * Standalone tests for circuit breaker functionality
 * This test file uses a completely isolated approach to avoid state interference
 */

// Import the functions we need to test
import {
  canMakeApiCall,
  getCircuitBreakerState,
  recordCircuitBreakerResult,
  resetCircuitBreaker
} from './monitoring';

// Create a unique prefix for all test exchanges/methods to avoid collisions
const TEST_PREFIX = `test_${Date.now()}_`;

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

  // Helper to get unique exchange/method names for each test
  const getUniqueKey = (testName: string) => {
    return {
      exchange: `${TEST_PREFIX}${testName}_exchange`,
      method: `${TEST_PREFIX}${testName}_method`
    };
  };

  test('should start with closed circuit', () => {
    const { exchange, method } = getUniqueKey('closed');

    // Reset circuit breaker to ensure clean state
    resetCircuitBreaker(exchange, method);

    // Circuit should be closed by default
    expect(canMakeApiCall(exchange, method)).toBe(true);

    // State should be closed
    const state = getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });

  test('should open circuit after failures', () => {
    const { exchange, method } = getUniqueKey('open');

    // Reset circuit breaker to ensure clean state
    resetCircuitBreaker(exchange, method);

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
  });

  test('should transition to half-open after timeout', () => {
    const { exchange, method } = getUniqueKey('timeout');

    // Reset circuit breaker to ensure clean state
    resetCircuitBreaker(exchange, method);

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

  test('should close circuit after success in half-open state', () => {
    const { exchange, method } = getUniqueKey('success');

    // Reset circuit breaker to ensure clean state
    resetCircuitBreaker(exchange, method);

    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };

    // Open the circuit
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);

    // Advance time to transition to half-open
    mockTime += 6000; // 6 seconds

    // Verify circuit is half-open by making a call
    expect(canMakeApiCall(exchange, method, config)).toBe(true);

    // Record success in half-open state
    recordCircuitBreakerResult(exchange, method, true, config);

    // Circuit should be closed
    expect(canMakeApiCall(exchange, method, config)).toBe(true);

    // State should be closed
    const state = getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });

  test('should reopen circuit after failure in half-open state', () => {
    const { exchange, method } = getUniqueKey('reopen');

    // Reset circuit breaker to ensure clean state
    resetCircuitBreaker(exchange, method);

    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };

    // Open the circuit
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);

    // Advance time to transition to half-open
    mockTime += 6000; // 6 seconds

    // Verify circuit is half-open by making a call
    expect(canMakeApiCall(exchange, method, config)).toBe(true);

    // Record failure in half-open state
    recordCircuitBreakerResult(exchange, method, false, config);

    // Circuit should be open again
    expect(canMakeApiCall(exchange, method, config)).toBe(false);

    // State should be open
    const state = getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('open');
  });

  test('should reset circuit breaker when requested', () => {
    const { exchange, method } = getUniqueKey('reset');

    // Reset circuit breaker to ensure clean state
    resetCircuitBreaker(exchange, method);

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
  });
});
