/**
 * Simplified tests for API monitoring and circuit breaker utilities
 */

import * as monitoring from './monitoring';

// Reset the metrics store between tests
beforeEach(() => {
  // Access the metrics store directly and reset it
  const metricsStore = monitoring.getAllMetrics();
  Object.keys(metricsStore).forEach(key => {
    delete metricsStore[key];
  });
});

describe('API Monitoring Utilities', () => {
  describe('API Call Metrics', () => {
    it('should record and retrieve API call metrics', () => {
      // Record API call attempts
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      
      // Record one success and one failure
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'success', 100);
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'failure', 200, new Error('Test error'));
      
      // Get metrics
      const metrics = monitoring.getApiCallMetrics('TestExchange', 'getPortfolio');
      
      // Verify metrics
      expect(metrics.totalCalls).toBe(2);
      expect(metrics.successCalls).toBe(1);
      expect(metrics.failedCalls).toBe(1);
      expect(metrics.totalDuration).toBe(300); // 100 + 200
      expect(metrics.minDuration).toBe(100);
      expect(metrics.maxDuration).toBe(200);
      expect(metrics.lastError).toBeInstanceOf(Error);
      expect(metrics.lastError?.message).toBe('Test error');
    });

    it('should calculate success rate correctly', () => {
      // Record API calls
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      
      // Record 3 successes and 1 failure
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'success', 100);
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'success', 150);
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'success', 200);
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'failure', 300, new Error('Test error'));
      
      // Get success rate
      const successRate = monitoring.getSuccessRate('TestExchange', 'getPortfolio');
      
      // Verify success rate (3/4 = 0.75)
      expect(successRate).toBe(0.75);
    });

    it('should calculate average duration correctly', () => {
      // Record API calls
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      
      // Record durations
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'success', 100);
      monitoring.recordApiCall('TestExchange', 'getPortfolio', 'success', 300);
      
      // Get average duration
      const avgDuration = monitoring.getAverageDuration('TestExchange', 'getPortfolio');
      
      // Verify average duration (400/2 = 200)
      expect(avgDuration).toBe(200);
    });

    it('should determine API health correctly', () => {
      // Setup healthy API
      monitoring.recordApiCall('HealthyExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('HealthyExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('HealthyExchange', 'getPortfolio', 'success', 100);
      monitoring.recordApiCall('HealthyExchange', 'getPortfolio', 'success', 100);
      
      // Setup unhealthy API
      monitoring.recordApiCall('UnhealthyExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('UnhealthyExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('UnhealthyExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('UnhealthyExchange', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('UnhealthyExchange', 'getPortfolio', 'success', 100);
      monitoring.recordApiCall('UnhealthyExchange', 'getPortfolio', 'failure', 100);
      monitoring.recordApiCall('UnhealthyExchange', 'getPortfolio', 'failure', 100);
      monitoring.recordApiCall('UnhealthyExchange', 'getPortfolio', 'failure', 100);
      
      // Check health with custom thresholds
      const healthyStatus = monitoring.isApiHealthy('HealthyExchange', 'getPortfolio', 0.9, 1);
      const unhealthyStatus = monitoring.isApiHealthy('UnhealthyExchange', 'getPortfolio', 0.9, 1);
      
      // Verify health status
      expect(healthyStatus).toBe(true); // 100% success rate
      expect(unhealthyStatus).toBe(false); // 25% success rate and more than 1 failure
    });
  });

  describe('Circuit Breaker', () => {
    // Mock Date.now() for predictable test results
    const originalDateNow = Date.now;
    let currentTime = 1000;
    
    beforeEach(() => {
      // Reset time for each test
      currentTime = 1000;
      Date.now = jest.fn(() => currentTime);
      
      // Reset circuit breakers
      monitoring.resetCircuitBreaker('TestExchange', 'getPortfolio');
    });
    
    afterEach(() => {
      // Restore original Date.now
      Date.now = originalDateNow;
    });
    
    it('should allow API calls when circuit breaker is closed', () => {
      // Check if API call is allowed
      const canCall = monitoring.canMakeApiCall('TestExchange', 'getPortfolio');
      
      // Verify API call is allowed
      expect(canCall).toBe(true);
    });

    it('should open circuit breaker after multiple failures', () => {
      // Configure circuit breaker with low threshold for testing
      const testConfig = { failureThreshold: 3, resetTimeoutMs: 100 };
      
      // Record multiple failures
      for (let i = 0; i < 3; i++) {
        monitoring.recordCircuitBreakerResult('TestExchange', 'getPortfolio', false, testConfig);
      }
      
      // Check circuit breaker state
      const state = monitoring.getCircuitBreakerState('TestExchange', 'getPortfolio');
      
      // Verify circuit breaker is open
      expect(state?.state).toBe('open');
    });

    it('should reset circuit breaker', () => {
      // Configure circuit breaker with low threshold for testing
      const testConfig = { failureThreshold: 3, resetTimeoutMs: 100 };
      
      // Record multiple failures to open circuit breaker
      for (let i = 0; i < 3; i++) {
        monitoring.recordCircuitBreakerResult('TestExchange', 'getPortfolio', false, testConfig);
      }
      
      // Reset circuit breaker
      monitoring.resetCircuitBreaker('TestExchange', 'getPortfolio');
      
      // Check circuit breaker state
      const state = monitoring.getCircuitBreakerState('TestExchange', 'getPortfolio');
      
      // Verify circuit breaker is closed
      expect(state?.state).toBe('closed');
    });

    it('should transition from open to half-open after timeout', () => {
      // Configure circuit breaker with short timeout for testing
      const testConfig = { failureThreshold: 3, resetTimeoutMs: 100 };
      
      // Record multiple failures to open circuit breaker
      for (let i = 0; i < 3; i++) {
        monitoring.recordCircuitBreakerResult('TestExchange', 'getPortfolio', false, testConfig);
      }
      
      // Verify circuit breaker is open
      expect(monitoring.canMakeApiCall('TestExchange', 'getPortfolio', testConfig)).toBe(false);
      
      // Advance time past timeout
      currentTime += 150;
      
      // Verify circuit breaker is now half-open
      expect(monitoring.canMakeApiCall('TestExchange', 'getPortfolio', testConfig)).toBe(true);
      
      // Get state
      const state = monitoring.getCircuitBreakerState('TestExchange', 'getPortfolio');
      expect(state?.state).toBe('half-open');
    });

    it('should close circuit breaker after success in half-open state', () => {
      // Configure circuit breaker with short timeout for testing
      const testConfig = { failureThreshold: 3, resetTimeoutMs: 100 };
      
      // Record multiple failures to open circuit breaker
      for (let i = 0; i < 3; i++) {
        monitoring.recordCircuitBreakerResult('TestExchange', 'getPortfolio', false, testConfig);
      }
      
      // Advance time to transition to half-open
      currentTime += 150;
      
      // Record success in half-open state
      monitoring.recordCircuitBreakerResult('TestExchange', 'getPortfolio', true, testConfig);
      
      // Get state
      const state = monitoring.getCircuitBreakerState('TestExchange', 'getPortfolio');
      
      // Verify circuit breaker is closed
      expect(state?.state).toBe('closed');
    });
  });

  describe('Performance Dashboard', () => {
    it('should generate API health dashboard data', () => {
      // Setup test data for multiple exchanges and methods
      // Exchange 1 - Healthy
      monitoring.recordApiCall('Exchange1', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('Exchange1', 'getPortfolio', 'success', 100);
      monitoring.recordApiCall('Exchange1', 'createOrder', 'attempt');
      monitoring.recordApiCall('Exchange1', 'createOrder', 'success', 200);
      
      // Exchange 2 - Unhealthy
      monitoring.recordApiCall('Exchange2', 'getPortfolio', 'attempt');
      monitoring.recordApiCall('Exchange2', 'getPortfolio', 'failure', 300, new Error('Test error'));
      
      // Generate dashboard data
      const dashboard = monitoring.getApiHealthDashboard();
      
      // Verify dashboard structure
      expect(dashboard).toBeTruthy();
      expect(dashboard.exchanges).toBeTruthy();
      expect(dashboard.exchanges.Exchange1).toBeTruthy();
      expect(dashboard.exchanges.Exchange2).toBeTruthy();
      
      // Verify Exchange1 data
      const exchange1 = dashboard.exchanges.Exchange1;
      expect(exchange1.totalCalls).toBe(2);
      expect(exchange1.successCalls).toBe(2);
      expect(exchange1.failedCalls).toBe(0);
      expect(exchange1.successRate).toBe(1); // 100% success
      
      // Verify Exchange2 data
      const exchange2 = dashboard.exchanges.Exchange2;
      expect(exchange2.totalCalls).toBe(1);
      expect(exchange2.successCalls).toBe(0);
      expect(exchange2.failedCalls).toBe(1);
      expect(exchange2.successRate).toBe(0); // 0% success
    });
  });
});
