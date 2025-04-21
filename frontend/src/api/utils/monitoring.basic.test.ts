/**
 * Basic tests for monitoring utilities
 */

import {
  recordApiCall,
  getApiCallMetrics,
  canMakeApiCall,
  recordCircuitBreakerResult,
  getCircuitBreakerState,
  resetCircuitBreaker,
  getSuccessRate,
  getAverageDuration,
  isApiHealthy,
  getApiHealthDashboard
} from './monitoring';

// Mock Date.now for predictable test results
const originalDateNow = Date.now;
let mockTime = 1000;

beforeAll(() => {
  // Set up Date.now mock
  mockTime = 1000;
  Date.now = jest.fn(() => mockTime);
});

afterAll(() => {
  // Restore original Date.now
  Date.now = originalDateNow;
});

describe('Monitoring Utilities', () => {
  // Use unique exchange and method names for each test to avoid state conflicts
  
  describe('API Call Metrics', () => {
    test('should record and retrieve API call metrics', () => {
      const exchange = 'TestExchange1';
      const method = 'testMethod1';
      
      // Record API calls
      recordApiCall(exchange, method, 'attempt');
      recordApiCall(exchange, method, 'success', 100);
      
      // Get metrics
      const metrics = getApiCallMetrics(exchange, method);
      
      // Verify metrics
      expect(metrics.totalCalls).toBe(1);
      expect(metrics.successCalls).toBe(1);
    });
    
    test('should calculate success rate correctly', () => {
      const exchange = 'TestExchange2';
      const method = 'testMethod2';
      
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
    
    test('should calculate average duration correctly', () => {
      const exchange = 'TestExchange3';
      const method = 'testMethod3';
      
      // Record API calls
      recordApiCall(exchange, method, 'attempt');
      recordApiCall(exchange, method, 'attempt');
      recordApiCall(exchange, method, 'success', 100);
      recordApiCall(exchange, method, 'success', 300);
      
      // Calculate average duration
      const avgDuration = getAverageDuration(exchange, method);
      
      // Verify average duration (400/2 = 200)
      expect(avgDuration).toBe(200);
    });
  });
  
  describe('Circuit Breaker', () => {
    test('should open circuit breaker after failures', () => {
      const exchange = 'TestExchange4';
      const method = 'testMethod4';
      
      // Configure a low threshold for testing
      const config = { failureThreshold: 2, resetTimeoutMs: 1000 };
      
      // Initially circuit should be closed
      expect(canMakeApiCall(exchange, method)).toBe(true);
      
      // Record failures
      recordCircuitBreakerResult(exchange, method, false, config);
      recordCircuitBreakerResult(exchange, method, false, config);
      
      // Circuit should now be open
      expect(canMakeApiCall(exchange, method, config)).toBe(false);
      
      // Check state
      const state = getCircuitBreakerState(exchange, method);
      expect(state?.state).toBe('open');
    });
    
    test('should reset circuit breaker', () => {
      const exchange = 'TestExchange5';
      const method = 'testMethod5';
      
      // Open the circuit
      const config = { failureThreshold: 2, resetTimeoutMs: 1000 };
      recordCircuitBreakerResult(exchange, method, false, config);
      recordCircuitBreakerResult(exchange, method, false, config);
      
      // Verify circuit is open
      expect(canMakeApiCall(exchange, method, config)).toBe(false);
      
      // Reset circuit
      resetCircuitBreaker(exchange, method);
      
      // Verify circuit is closed
      expect(canMakeApiCall(exchange, method, config)).toBe(true);
    });
    
    test('should transition to half-open state after timeout', () => {
      const exchange = 'TestExchange6';
      const method = 'testMethod6';
      
      // Configure circuit breaker with short timeout
      const config = { failureThreshold: 2, resetTimeoutMs: 5000 };
      
      // Open the circuit
      recordCircuitBreakerResult(exchange, method, false, config);
      recordCircuitBreakerResult(exchange, method, false, config);
      
      // Verify circuit is open
      expect(canMakeApiCall(exchange, method, config)).toBe(false);
      
      // Advance time past timeout
      mockTime += 6000;
      
      // Circuit should now be half-open
      expect(canMakeApiCall(exchange, method, config)).toBe(true);
      
      // Verify state
      const state = getCircuitBreakerState(exchange, method);
      expect(state?.state).toBe('half-open');
    });
  });
  
  describe('API Health', () => {
    test('should determine API health correctly', () => {
      const healthyExchange = 'HealthyExchange';
      const unhealthyExchange = 'UnhealthyExchange';
      const method = 'testMethod';
      
      // Setup healthy API
      recordApiCall(healthyExchange, method, 'attempt');
      recordApiCall(healthyExchange, method, 'attempt');
      recordApiCall(healthyExchange, method, 'success', 100);
      recordApiCall(healthyExchange, method, 'success', 100);
      
      // Setup unhealthy API
      recordApiCall(unhealthyExchange, method, 'attempt');
      recordApiCall(unhealthyExchange, method, 'attempt');
      recordApiCall(unhealthyExchange, method, 'attempt');
      recordApiCall(unhealthyExchange, method, 'attempt');
      recordApiCall(unhealthyExchange, method, 'success', 100);
      recordApiCall(unhealthyExchange, method, 'failure', 100);
      recordApiCall(unhealthyExchange, method, 'failure', 100);
      recordApiCall(unhealthyExchange, method, 'failure', 100);
      
      // Check health with custom thresholds
      const healthyStatus = isApiHealthy(healthyExchange, method, 0.9, 1);
      const unhealthyStatus = isApiHealthy(unhealthyExchange, method, 0.9, 1);
      
      // Verify health status
      expect(healthyStatus).toBe(true); // 100% success rate
      expect(unhealthyStatus).toBe(false); // 25% success rate and more than 1 failure
    });
    
    test('should generate dashboard data', () => {
      const exchange1 = 'DashboardExchange1';
      const exchange2 = 'DashboardExchange2';
      
      // Record API calls for multiple exchanges
      recordApiCall(exchange1, 'getPortfolio', 'attempt');
      recordApiCall(exchange1, 'getPortfolio', 'success', 100);
      
      recordApiCall(exchange2, 'getPortfolio', 'attempt');
      recordApiCall(exchange2, 'getPortfolio', 'failure', 200, new Error('Test error'));
      
      // Get dashboard
      const dashboard = getApiHealthDashboard();
      
      // Verify dashboard structure
      expect(dashboard.exchanges[exchange1]).toBeTruthy();
      expect(dashboard.exchanges[exchange2]).toBeTruthy();
      
      // Verify metrics
      expect(dashboard.exchanges[exchange1].successRate).toBe(1);
      expect(dashboard.exchanges[exchange2].successRate).toBe(0);
    });
  });
});
