/**
 * Enhanced tests for API monitoring and circuit breaker utilities
 */

import {
  recordApiCall,
  canMakeApiCall,
  recordCircuitBreakerResult,
  getCircuitBreakerState,
  resetCircuitBreaker,
  getApiCallMetrics,
  getSuccessRate,
  getAverageDuration,
  isApiHealthy,
  getApiHealthDashboard,
  getAllMetrics
} from './monitoring';

// Mock Date.now() for predictable test results
const originalDateNow = Date.now;
let mockTime = 1000;

beforeEach(() => {
  // Reset time for each test
  mockTime = 1000;
  Date.now = jest.fn(() => mockTime);
  
  // Reset metrics store by clearing all exchanges
  const metricsStore = getAllMetrics();
  Object.keys(metricsStore).forEach(exchange => {
    delete metricsStore[exchange];
  });
  
  // Reset all circuit breakers we'll use in tests
  resetCircuitBreaker('TestExchange', 'getPortfolio');
  resetCircuitBreaker('HealthyExchange', 'testMethod');
  resetCircuitBreaker('UnhealthyExchange', 'testMethod');
  resetCircuitBreaker('Exchange1', 'getPortfolio');
  resetCircuitBreaker('Exchange1', 'createOrder');
  resetCircuitBreaker('Exchange2', 'getPortfolio');
});

afterEach(() => {
  // Restore original Date.now
  Date.now = originalDateNow;
});

describe('API Monitoring Utilities', () => {
  describe('API Call Metrics', () => {
    it('should record API call attempts correctly', () => {
      // Record an attempt
      recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      
      // Get metrics
      const metrics = getApiCallMetrics('TestExchange', 'getPortfolio');
      
      // Verify metrics
      expect(metrics).toBeTruthy();
      expect(metrics.totalCalls).toBe(1);
      expect(metrics.successCalls).toBe(0);
      expect(metrics.failedCalls).toBe(0);
      expect(metrics.lastCallTime).toBe(1000);
    });

    it('should record API call successes correctly', () => {
      // Record an attempt and success
      recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      recordApiCall('TestExchange', 'getPortfolio', 'success', 100);
      
      // Get metrics
      const metrics = getApiCallMetrics('TestExchange', 'getPortfolio');
      
      // Verify metrics
      expect(metrics).toBeTruthy();
      expect(metrics.totalCalls).toBe(1);
      expect(metrics.successCalls).toBe(1);
      expect(metrics.failedCalls).toBe(0);
      expect(metrics.totalDuration).toBe(100);
      expect(metrics.minDuration).toBe(100);
      expect(metrics.maxDuration).toBe(100);
    });

    it('should record API call failures correctly', () => {
      // Record an attempt and failure
      const error = new Error('Test error');
      recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      recordApiCall('TestExchange', 'getPortfolio', 'failure', 50, error);
      
      // Get metrics
      const metrics = getApiCallMetrics('TestExchange', 'getPortfolio');
      
      // Verify metrics
      expect(metrics).toBeTruthy();
      expect(metrics.totalCalls).toBe(1);
      expect(metrics.successCalls).toBe(0);
      expect(metrics.failedCalls).toBe(1);
      expect(metrics.totalDuration).toBe(50);
      expect(metrics.lastError).toBe(error);
    });
    
    it('should calculate success rate correctly', () => {
      // Record multiple API calls
      recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      
      // Record 3 successes and 1 failure
      recordApiCall('TestExchange', 'getPortfolio', 'success', 100);
      recordApiCall('TestExchange', 'getPortfolio', 'success', 150);
      recordApiCall('TestExchange', 'getPortfolio', 'success', 200);
      recordApiCall('TestExchange', 'getPortfolio', 'failure', 300, new Error('Test error'));
      
      // Calculate success rate
      const successRate = getSuccessRate('TestExchange', 'getPortfolio');
      
      // Verify success rate (3/4 = 0.75)
      expect(successRate).toBe(0.75);
    });
    
    it('should calculate average duration correctly', () => {
      // Record API calls
      recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      recordApiCall('TestExchange', 'getPortfolio', 'attempt');
      
      // Record durations
      recordApiCall('TestExchange', 'getPortfolio', 'success', 100);
      recordApiCall('TestExchange', 'getPortfolio', 'success', 300);
      
      // Calculate average duration
      const avgDuration = getAverageDuration('TestExchange', 'getPortfolio');
      
      // Verify average duration (400/2 = 200)
      expect(avgDuration).toBe(200);
    });
    
    it('should determine API health correctly', () => {
      // Setup healthy API
      recordApiCall('HealthyExchange', 'testMethod', 'attempt');
      recordApiCall('HealthyExchange', 'testMethod', 'attempt');
      recordApiCall('HealthyExchange', 'testMethod', 'success', 100);
      recordApiCall('HealthyExchange', 'testMethod', 'success', 100);
      
      // Setup unhealthy API
      recordApiCall('UnhealthyExchange', 'testMethod', 'attempt');
      recordApiCall('UnhealthyExchange', 'testMethod', 'attempt');
      recordApiCall('UnhealthyExchange', 'testMethod', 'attempt');
      recordApiCall('UnhealthyExchange', 'testMethod', 'attempt');
      recordApiCall('UnhealthyExchange', 'testMethod', 'success', 100);
      recordApiCall('UnhealthyExchange', 'testMethod', 'failure', 100);
      recordApiCall('UnhealthyExchange', 'testMethod', 'failure', 100);
      recordApiCall('UnhealthyExchange', 'testMethod', 'failure', 100);
      
      // Check health with custom thresholds
      const healthyStatus = isApiHealthy('HealthyExchange', 'testMethod', 0.9, 1);
      const unhealthyStatus = isApiHealthy('UnhealthyExchange', 'testMethod', 0.9, 1);
      
      // Verify health status
      expect(healthyStatus).toBe(true); // 100% success rate
      expect(unhealthyStatus).toBe(false); // 25% success rate and more than 1 failure
    });
  });

  describe('Circuit Breaker', () => {
    it('should allow API calls when circuit is closed', () => {
      // Default state is closed
      const canCall = canMakeApiCall('TestExchange', 'getPortfolio');
      expect(canCall).toBe(true);
    });

    it('should open circuit after failure threshold is reached', () => {
      // Record multiple failures (using a low threshold for testing)
      const testConfig = { failureThreshold: 3, resetTimeoutMs: 30000 };
      
      for (let i = 0; i < 3; i++) {
        recordCircuitBreakerResult('TestExchange', 'getPortfolio', false, testConfig);
      }
      
      // Check if circuit is open
      const canCall = canMakeApiCall('TestExchange', 'getPortfolio', testConfig);
      expect(canCall).toBe(false);
      
      // Check circuit state
      const state = getCircuitBreakerState('TestExchange', 'getPortfolio');
      expect(state).toBeTruthy();
      expect(state?.state).toBe('open');
    });

    it('should transition to half-open state after timeout', () => {
      // Record failures to open circuit
      const testConfig = { failureThreshold: 3, resetTimeoutMs: 5000 };
      
      for (let i = 0; i < 3; i++) {
        recordCircuitBreakerResult('TestExchange', 'getPortfolio', false, testConfig);
      }
      
      // Verify circuit is open
      expect(canMakeApiCall('TestExchange', 'getPortfolio', testConfig)).toBe(false);
      
      // Advance time past reset timeout
      mockTime += 6000; // 6 seconds
      
      // Circuit should now allow a call (transitioning to half-open)
      const canCall = canMakeApiCall('TestExchange', 'getPortfolio', testConfig);
      expect(canCall).toBe(true);
      
      // Verify state is half-open
      const state = getCircuitBreakerState('TestExchange', 'getPortfolio');
      expect(state?.state).toBe('half-open');
    });

    it('should close circuit after successful call in half-open state', () => {
      // Open circuit with consistent config
      const testConfig = { failureThreshold: 3, resetTimeoutMs: 5000 };
      
      for (let i = 0; i < 3; i++) {
        recordCircuitBreakerResult('TestExchange', 'getPortfolio', false, testConfig);
      }
      
      // Transition to half-open
      mockTime += 6000; // 6 seconds
      expect(canMakeApiCall('TestExchange', 'getPortfolio', testConfig)).toBe(true);
      
      // Record success
      recordCircuitBreakerResult('TestExchange', 'getPortfolio', true, testConfig);
      
      // Circuit should be closed
      const state = getCircuitBreakerState('TestExchange', 'getPortfolio');
      expect(state?.state).toBe('closed');
      expect(canMakeApiCall('TestExchange', 'getPortfolio', testConfig)).toBe(true);
    });

    it('should reopen circuit after failure in half-open state', () => {
      // Open circuit with consistent config
      const testConfig = { failureThreshold: 3, resetTimeoutMs: 5000 };
      
      for (let i = 0; i < 3; i++) {
        recordCircuitBreakerResult('TestExchange', 'getPortfolio', false, testConfig);
      }
      
      // Transition to half-open
      mockTime += 6000; // 6 seconds
      expect(canMakeApiCall('TestExchange', 'getPortfolio', testConfig)).toBe(true);
      
      // Record failure
      recordCircuitBreakerResult('TestExchange', 'getPortfolio', false, testConfig);
      
      // Circuit should be open again
      expect(canMakeApiCall('TestExchange', 'getPortfolio', testConfig)).toBe(false);
      
      const state = getCircuitBreakerState('TestExchange', 'getPortfolio');
      expect(state?.state).toBe('open');
    });

    it('should reset circuit breaker when requested', () => {
      // Open circuit with consistent config
      const testConfig = { failureThreshold: 3, resetTimeoutMs: 5000 };
      
      for (let i = 0; i < 3; i++) {
        recordCircuitBreakerResult('TestExchange', 'getPortfolio', false, testConfig);
      }
      
      // Verify circuit is open
      expect(canMakeApiCall('TestExchange', 'getPortfolio', testConfig)).toBe(false);
      
      // Reset circuit
      resetCircuitBreaker('TestExchange', 'getPortfolio');
      
      // Circuit should be closed
      expect(canMakeApiCall('TestExchange', 'getPortfolio', testConfig)).toBe(true);
      
      const state = getCircuitBreakerState('TestExchange', 'getPortfolio');
      expect(state?.state).toBe('closed');
    });
  });
  
  describe('Performance Dashboard', () => {
    it('should generate API health dashboard data', () => {
      // Setup test data for multiple exchanges and methods
      // Exchange 1 - Healthy
      recordApiCall('Exchange1', 'getPortfolio', 'attempt');
      recordApiCall('Exchange1', 'getPortfolio', 'success', 100);
      recordApiCall('Exchange1', 'createOrder', 'attempt');
      recordApiCall('Exchange1', 'createOrder', 'success', 200);
      
      // Exchange 2 - Unhealthy
      recordApiCall('Exchange2', 'getPortfolio', 'attempt');
      recordApiCall('Exchange2', 'getPortfolio', 'failure', 300, new Error('Test error'));
      
      // Generate dashboard data
      const dashboard = getApiHealthDashboard();
      
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
