/**
 * Simplified tests for monitoring utilities
 */

// Import directly from the module
import * as monitoring from './monitoring';

describe('Monitoring Simple Tests', () => {
  test('should record API call attempts', () => {
    const exchange = 'SimpleExchange';
    const method = 'simpleMethod';
    
    // Record an API call attempt
    monitoring.recordApiCall(exchange, method, 'attempt');
    
    // Get metrics
    const metrics = monitoring.getApiCallMetrics(exchange, method);
    
    // Verify metrics
    expect(metrics.totalCalls).toBe(1);
    expect(metrics.successCalls).toBe(0);
    expect(metrics.failedCalls).toBe(0);
  });
});
