/**
 * Simplified tests for monitoring utility functions
 */

// Import the functions directly
import { 
  recordApiCall, 
  getApiCallMetrics,
  canMakeApiCall,
  recordCircuitBreakerResult,
  resetCircuitBreaker
} from './monitoring';

describe('Monitoring Utilities', () => {
  // Use unique exchange and method names for each test to avoid state conflicts
  
  describe('API Call Recording', () => {
    test('should record API call attempts', () => {
      const exchange = 'Exchange1';
      const method = 'Method1';
      
      // Record an API call attempt
      recordApiCall(exchange, method, 'attempt');
      
      // Get metrics
      const metrics = getApiCallMetrics(exchange, method);
      
      // Verify metrics
      expect(metrics.totalCalls).toBe(1);
    });
    
    test('should record API call successes', () => {
      const exchange = 'Exchange2';
      const method = 'Method2';
      
      // Record an API call attempt and success
      recordApiCall(exchange, method, 'attempt');
      recordApiCall(exchange, method, 'success', 100);
      
      // Get metrics
      const metrics = getApiCallMetrics(exchange, method);
      
      // Verify metrics
      expect(metrics.successCalls).toBe(1);
      expect(metrics.totalDuration).toBe(100);
    });
    
    test('should record API call failures', () => {
      const exchange = 'Exchange3';
      const method = 'Method3';
      
      // Record an API call attempt and failure
      recordApiCall(exchange, method, 'attempt');
      recordApiCall(exchange, method, 'failure', 50, new Error('Test error'));
      
      // Get metrics
      const metrics = getApiCallMetrics(exchange, method);
      
      // Verify metrics
      expect(metrics.failedCalls).toBe(1);
      expect(metrics.lastError).toBeTruthy();
    });
  });
  
  describe('Circuit Breaker', () => {
    test('should allow API calls by default', () => {
      const exchange = 'Exchange4';
      const method = 'Method4';
      
      // Check if API call is allowed
      const canCall = canMakeApiCall(exchange, method);
      
      // Verify API call is allowed
      expect(canCall).toBe(true);
    });
    
    test('should track circuit breaker state', () => {
      const exchange = 'Exchange5';
      const method = 'Method5';
      
      // Record a success
      recordCircuitBreakerResult(exchange, method, true);
      
      // Check if API call is allowed
      const canCall = canMakeApiCall(exchange, method);
      
      // Verify API call is allowed
      expect(canCall).toBe(true);
    });
    
    test('should reset circuit breaker', () => {
      const exchange = 'Exchange6';
      const method = 'Method6';
      
      // Reset the circuit breaker
      resetCircuitBreaker(exchange, method);
      
      // Check if API call is allowed
      const canCall = canMakeApiCall(exchange, method);
      
      // Verify API call is allowed
      expect(canCall).toBe(true);
    });
  });
});
