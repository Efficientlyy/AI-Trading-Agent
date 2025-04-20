/**
 * Tests for API metrics functionality in the monitoring module
 */
import { recordApiCall, getApiCallMetrics } from './monitoring';

describe('API Metrics', () => {
  // Use unique exchange/method combinations for each test to avoid state interference
  
  test('should record API call attempts', () => {
    const exchange = 'MetricsExchange1';
    const method = 'metricsMethod1';
    
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
    const exchange = 'MetricsExchange2';
    const method = 'metricsMethod2';
    
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
    expect(metrics.minDuration).toBe(100);
    expect(metrics.maxDuration).toBe(100);
  });
  
  test('should record API call failures', () => {
    const exchange = 'MetricsExchange3';
    const method = 'metricsMethod3';
    
    // Record an API call attempt and failure
    const error = new Error('Test error');
    recordApiCall(exchange, method, 'attempt');
    recordApiCall(exchange, method, 'failure', 50, error);
    
    // Get metrics
    const metrics = getApiCallMetrics(exchange, method);
    
    // Verify metrics
    expect(metrics.totalCalls).toBe(1);
    expect(metrics.successCalls).toBe(0);
    expect(metrics.failedCalls).toBe(1);
    expect(metrics.totalDuration).toBe(50);
    expect(metrics.lastError).toBe(error);
  });
  
  test('should track min/max durations correctly', () => {
    const exchange = 'MetricsExchange4';
    const method = 'metricsMethod4';
    
    // Record API calls with different durations
    recordApiCall(exchange, method, 'attempt');
    recordApiCall(exchange, method, 'success', 100);
    recordApiCall(exchange, method, 'success', 50);
    recordApiCall(exchange, method, 'success', 200);
    
    // Get metrics
    const metrics = getApiCallMetrics(exchange, method);
    
    // Verify min/max durations
    expect(metrics.minDuration).toBe(50);
    expect(metrics.maxDuration).toBe(200);
    expect(metrics.totalDuration).toBe(350);
  });
  
  test('should initialize metrics for new exchange/method combinations', () => {
    const exchange = 'MetricsExchange5';
    const method = 'metricsMethod5';
    
    // Get metrics without recording any calls
    const metrics = getApiCallMetrics(exchange, method);
    
    // Verify default metrics
    expect(metrics.totalCalls).toBe(0);
    expect(metrics.successCalls).toBe(0);
    expect(metrics.failedCalls).toBe(0);
    expect(metrics.totalDuration).toBe(0);
    expect(metrics.minDuration).toBe(Infinity);
    expect(metrics.maxDuration).toBe(0);
  });
});
