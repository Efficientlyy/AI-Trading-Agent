/**
 * Fixed tests for API monitoring utilities
 */

// Import the monitoring module directly (no mocking)
import * as monitoring from './monitoring';

describe('Monitoring Utilities', () => {
  // Reset all state before each test
  beforeEach(() => {
    // Reset metrics store
    const metricsStore = monitoring.getAllMetrics();
    Object.keys(metricsStore).forEach(key => {
      delete metricsStore[key];
    });
    
    // Reset circuit breakers for test exchanges
    monitoring.resetCircuitBreaker('TestExchange', 'testMethod');
    monitoring.resetCircuitBreaker('Exchange1', 'getPortfolio');
    monitoring.resetCircuitBreaker('Exchange2', 'getPortfolio');
  });
  
  describe('API Call Metrics', () => {
    test('should record and retrieve API call metrics', () => {
      // Record API calls
      monitoring.recordApiCall('TestExchange', 'testMethod', 'attempt');
      monitoring.recordApiCall('TestExchange', 'testMethod', 'success', 100);
      
      // Get metrics
      const metrics = monitoring.getApiCallMetrics('TestExchange', 'testMethod');
      
      // Verify metrics
      expect(metrics.totalCalls).toBe(1);
      expect(metrics.successCalls).toBe(1);
    });
    
    test('should calculate success rate correctly', () => {
      // Record API calls with mixed results
      monitoring.recordApiCall('TestExchange', 'testMethod', 'attempt');
      monitoring.recordApiCall('TestExchange', 'testMethod', 'attempt');
      monitoring.recordApiCall('TestExchange', 'testMethod', 'success', 100);
      monitoring.recordApiCall('TestExchange', 'testMethod', 'failure', 200, new Error('Test error'));
      
      // Calculate success rate
      const successRate = monitoring.getSuccessRate('TestExchange', 'testMethod');
      
      // Verify success rate (1/2 = 0.5)
      expect(successRate).toBe(0.5);
    });
  });
  
  describe('Circuit Breaker', () => {
    test('should open circuit breaker after failures', () => {
      // Configure a low threshold for testing
      const config = { failureThreshold: 2, resetTimeoutMs: 1000 };
      
      // Initially circuit should be closed
      expect(monitoring.canMakeApiCall('TestExchange', 'testMethod')).toBe(true);
      
      // Record failures
      monitoring.recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
      monitoring.recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
      
      // Circuit should now be open
      expect(monitoring.canMakeApiCall('TestExchange', 'testMethod', config)).toBe(false);
      
      // Check state
      const state = monitoring.getCircuitBreakerState('TestExchange', 'testMethod');
      expect(state?.state).toBe('open');
    });
    
    test('should reset circuit breaker', () => {
      // Open the circuit
      const config = { failureThreshold: 2, resetTimeoutMs: 1000 };
      monitoring.recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
      monitoring.recordCircuitBreakerResult('TestExchange', 'testMethod', false, config);
      
      // Verify circuit is open
      expect(monitoring.canMakeApiCall('TestExchange', 'testMethod', config)).toBe(false);
      
      // Reset circuit
      monitoring.resetCircuitBreaker('TestExchange', 'testMethod');
      
      // Verify circuit is closed
      expect(monitoring.canMakeApiCall('TestExchange', 'testMethod', config)).toBe(true);
    });
  });
  
  describe('API Health Dashboard', () => {
    test('should generate dashboard data', () => {
      // Record API calls for multiple exchanges
      monitoring.recordApiCall('Exchange1', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('Exchange1', 'getPortfolio', 'success', 100);
      
      monitoring.recordApiCall('Exchange2', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('Exchange2', 'getPortfolio', 'failure', 200, new Error('Test error'));
      
      // Get dashboard
      const dashboard = monitoring.getApiHealthDashboard();
      
      // Verify dashboard structure
      expect(dashboard.exchanges.Exchange1).toBeTruthy();
      expect(dashboard.exchanges.Exchange2).toBeTruthy();
      
      // Verify metrics
      expect(dashboard.exchanges.Exchange1.successRate).toBe(1);
      expect(dashboard.exchanges.Exchange2.successRate).toBe(0);
    });
  });
});
