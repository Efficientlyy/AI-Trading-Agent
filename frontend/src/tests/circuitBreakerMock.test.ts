/**
 * Tests for the circuit breaker mock implementation
 */
import {
  mockCanMakeApiCall,
  mockRecordCircuitBreakerResult,
  mockResetCircuitBreaker,
  mockGetCircuitBreakerState,
  setMockTime,
  advanceMockTime,
  resetAllCircuitBreakers
} from './mocks/circuitBreakerMock';

describe('Circuit Breaker Mock Tests', () => {
  beforeEach(() => {
    // Reset all circuit breakers before each test
    resetAllCircuitBreakers();
    // Set a fixed time for predictable tests
    setMockTime(1000);
  });

  test('should start with closed circuit', () => {
    const exchange = 'TestExchange';
    const method = 'testMethod';
    
    // Circuit should be closed by default
    expect(mockCanMakeApiCall(exchange, method)).toBe(true);
    
    // State should be closed
    const state = mockGetCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });

  test('should open circuit after failures', () => {
    const exchange = 'TestExchange';
    const method = 'openCircuitMethod';
    
    // Configure a low threshold for testing
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Record failures
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should still be closed
    expect(mockCanMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record one more failure to exceed threshold
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should now be open
    expect(mockCanMakeApiCall(exchange, method, config)).toBe(false);
  });

  test('should transition to half-open after timeout', () => {
    const exchange = 'TestExchange';
    const method = 'halfOpenMethod';
    
    // Configure circuit breaker with short timeout
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    
    // Verify circuit is open
    expect(mockCanMakeApiCall(exchange, method, config)).toBe(false);
    
    // Advance time past timeout
    advanceMockTime(6000); // 6 seconds
    
    // Circuit should now be half-open
    expect(mockCanMakeApiCall(exchange, method, config)).toBe(true);
    
    // State should be half-open
    const state = mockGetCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('half-open');
  });

  test('should close circuit after success in half-open state', () => {
    const exchange = 'TestExchange';
    const method = 'closeCircuitMethod';
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    
    // Advance time to transition to half-open
    advanceMockTime(6000); // 6 seconds
    
    // Verify circuit is half-open by making a call
    expect(mockCanMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record success in half-open state
    mockRecordCircuitBreakerResult(exchange, method, true, config);
    
    // Circuit should be closed
    expect(mockCanMakeApiCall(exchange, method, config)).toBe(true);
    
    // State should be closed
    const state = mockGetCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });

  test('should reopen circuit after failure in half-open state', () => {
    const exchange = 'TestExchange';
    const method = 'reopenMethod';
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    
    // Advance time to transition to half-open
    advanceMockTime(6000); // 6 seconds
    
    // Verify circuit is half-open by making a call
    expect(mockCanMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record failure in half-open state
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should be open again
    expect(mockCanMakeApiCall(exchange, method, config)).toBe(false);
    
    // State should be open
    const state = mockGetCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('open');
  });

  test('should reset circuit breaker when requested', () => {
    const exchange = 'TestExchange';
    const method = 'resetMethod';
    
    // Configure circuit breaker
    const config = { failureThreshold: 3, resetTimeoutMs: 5000 };
    
    // Open the circuit
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    mockRecordCircuitBreakerResult(exchange, method, false, config);
    
    // Verify circuit is open
    expect(mockCanMakeApiCall(exchange, method, config)).toBe(false);
    
    // Reset circuit
    mockResetCircuitBreaker(exchange, method);
    
    // Circuit should be closed
    expect(mockCanMakeApiCall(exchange, method, config)).toBe(true);
  });
});
