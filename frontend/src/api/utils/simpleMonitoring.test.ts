/**
 * Simple test for monitoring utilities
 */
import * as monitoring from './monitoring';

describe('Simple Monitoring Test', () => {
  test('should record API call attempts', () => {
    const exchange = 'SimpleExchange';
    const method = 'simpleMethod';
    
    // Record an API call attempt directly using the module
    monitoring.recordApiCall(exchange, method, 'attempt');
    
    // Get metrics directly using the module
    const metrics = monitoring.getApiCallMetrics(exchange, method);
    
    // Verify metrics
    expect(metrics.totalCalls).toBe(1);
  });
});
