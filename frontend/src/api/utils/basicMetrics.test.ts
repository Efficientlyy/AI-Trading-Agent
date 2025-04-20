/**
 * Basic tests for API metrics functionality
 */

// Import the monitoring module
import { 
  recordApiCall, 
  getApiCallMetrics,
  resetCircuitBreaker,
  recordCircuitBreakerResult,
  canMakeApiCall,
  getCircuitBreakerState
} from './monitoring';

describe('Basic API Metrics Tests', () => {
  const exchange = 'TestExchange';
  const method = 'testMethod';
  
  beforeEach(() => {
    // Reset circuit breaker before each test
    resetCircuitBreaker(exchange, method);
  });
  
  test('should record API call attempts', () => {
    // Record an API call attempt
    recordApiCall(exchange, method, 'attempt');
    
    // Get metrics
    const metrics = getApiCallMetrics(exchange, method);
    
    // Verify metrics
    expect(metrics.totalCalls).toBe(1);
    expect(metrics.successCalls).toBe(0);
    expect(metrics.failedCalls).toBe(0);
  });
  
  test('should record API call successes', () => {
    // Record an API call attempt and success
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
});
