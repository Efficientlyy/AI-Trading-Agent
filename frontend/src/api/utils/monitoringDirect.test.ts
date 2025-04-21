/**
 * Direct tests for monitoring utilities without mocking
 */

import {
  recordApiCall,
  getApiCallMetrics,
  getSuccessRate,
  canMakeApiCall,
  recordCircuitBreakerResult,
  resetCircuitBreaker,
  getCircuitBreakerState
} from './monitoring';

describe('Monitoring Direct Tests', () => {
  // Use unique exchange and method names for each test to avoid state conflicts
  
  test('should record and retrieve API call metrics', () => {
    const exchange = 'Exchange1';
    const method = 'Method1';
    
    // Record API calls
    recordApiCall(exchange, method, 'attempt');
    recordApiCall(exchange, method, 'success', 100);
    
    // Get metrics
    const metrics = getApiCallMetrics(exchange, method);
    
    // Verify metrics
    expect(metrics.totalCalls).toBe(1);
    expect(metrics.successCalls).toBe(1);
    expect(metrics.failedCalls).toBe(0);
    expect(metrics.totalDuration).toBe(100);
  });
  
  test('should calculate success rate correctly', () => {
    const exchange = 'Exchange2';
    const method = 'Method2';
    
    // Record API calls with mixed results
    recordApiCall(exchange, method, 'attempt');
    recordApiCall(exchange, method, 'attempt');
    recordApiCall(exchange, method, 'success', 100);
    recordApiCall(exchange, method, 'failure', 200, new Error('Test error'));
    
    // Calculate success rate
    const successRate = getSuccessRate(exchange, method);
    
    // Verify success rate (1/2 = 0.5)
    expect(successRate).toBe(0.5);
  });
  
  test('should handle circuit breaker state transitions', () => {
    const exchange = 'Exchange3';
    const method = 'Method3';
    
    // Use a low threshold for testing
    const config = { failureThreshold: 2, resetTimeoutMs: 1000 };
    
    // Initially circuit should be closed
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Record failures to open the circuit
    recordCircuitBreakerResult(exchange, method, false, config);
    recordCircuitBreakerResult(exchange, method, false, config);
    
    // Circuit should now be open
    expect(canMakeApiCall(exchange, method, config)).toBe(false);
    
    // Reset circuit
    resetCircuitBreaker(exchange, method);
    
    // Circuit should be closed again
    expect(canMakeApiCall(exchange, method, config)).toBe(true);
    
    // Get circuit state
    const state = getCircuitBreakerState(exchange, method);
    expect(state?.state).toBe('closed');
  });
});
