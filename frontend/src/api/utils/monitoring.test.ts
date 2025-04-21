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
  getAllMetrics,
  type ApiCallMetrics
} from './monitoring';

// Make sure to reset the module between tests
jest.mock('./monitoring', () => {
  const originalModule = jest.requireActual('./monitoring');
  return {
    ...originalModule,
    // Explicitly mock these functions to ensure they're properly reset between tests
    resetCircuitBreaker: jest.fn(originalModule.resetCircuitBreaker),
    recordCircuitBreakerResult: jest.fn(originalModule.recordCircuitBreakerResult),
    canMakeApiCall: jest.fn(originalModule.canMakeApiCall),
    getCircuitBreakerState: jest.fn(originalModule.getCircuitBreakerState),
    recordApiCall: jest.fn(originalModule.recordApiCall),
    getApiCallMetrics: jest.fn(originalModule.getApiCallMetrics),
    getSuccessRate: jest.fn(originalModule.getSuccessRate),
    getAverageDuration: jest.fn(originalModule.getAverageDuration),
    isApiHealthy: jest.fn(originalModule.isApiHealthy),
    getApiHealthDashboard: jest.fn(originalModule.getApiHealthDashboard),
    getAllMetrics: jest.fn(originalModule.getAllMetrics)
  };
});

type CircuitBreakerState = {
  state: 'closed' | 'open' | 'half-open';
  failureCount: number;
  lastFailureTime: number;
  nextAttemptTime: number;
};

// Mock Date.now() for predictable test results
const mockDateNow = jest.spyOn(Date, 'now');

// Helper function to reset all mocks and state
const resetAllMocks = () => {
  jest.clearAllMocks();
  mockDateNow.mockImplementation(() => 1000);
  
  // Reset circuit breakers
  resetCircuitBreaker('TestExchange', 'testMethod');
  resetCircuitBreaker('HealthyExchange', 'testMethod');
  resetCircuitBreaker('UnhealthyExchange', 'testMethod');
  resetCircuitBreaker('Exchange1', 'getPortfolio');
  resetCircuitBreaker('Exchange1', 'createOrder');
  resetCircuitBreaker('Exchange2', 'getPortfolio');
  
  // Reset metrics store by accessing the internal store
  const metricsStore = getAllMetrics();
  Object.keys(metricsStore).forEach(exchange => {
    delete metricsStore[exchange];
  });
};

describe('Monitoring Utilities', () => {
  beforeEach(() => {
    // Reset mocks and set initial time
    resetAllMocks();
  });

  afterEach(() => {
    mockDateNow.mockRestore();
  });

  describe('API Call Metrics', () => {
    it('should record API call attempts correctly', () => {
      // Record an attempt
      recordApiCall('TestExchange', 'testMethod', 'attempt');
      
      // Get metrics
      const metrics = getApiCallMetrics('TestExchange', 'testMethod');
      
      // Verify metrics
      expect(metrics).toBeTruthy();
      expect(metrics.totalCalls).toBe(1);
      expect(metrics.successCalls).toBe(0);
      expect(metrics.failedCalls).toBe(0);
      expect(metrics.lastCallTime).toBe(1000);
    });

    it('should record API call successes correctly', () => {
      // Record an attempt and success
      recordApiCall('TestExchange', 'testMethod', 'attempt');
      recordApiCall('TestExchange', 'testMethod', 'success', 100);
      
      // Get metrics
      const metrics = getApiCallMetrics('TestExchange', 'testMethod');
      
      // Verify metrics
      expect(metrics).toBeTruthy();
      // totalCalls is only incremented on 'attempt' to avoid double counting
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
      recordApiCall('TestExchange', 'testMethod', 'attempt');
      recordApiCall('TestExchange', 'testMethod', 'failure', 50, error);
      
      // Get metrics
      const metrics = getApiCallMetrics('TestExchange', 'testMethod');
      
      // Verify metrics
      expect(metrics).toBeTruthy();
      // totalCalls is only incremented on 'attempt' to avoid double counting
      expect(metrics.totalCalls).toBe(1);
      expect(metrics.successCalls).toBe(0);
      expect(metrics.failedCalls).toBe(1);
      expect(metrics.totalDuration).toBe(50);
      expect(metrics.lastError).toBe(error);
    });
    
    it('should calculate success rate correctly', () => {
      // Record multiple API calls
      recordApiCall('TestExchange', 'testMethod', 'attempt');
      recordApiCall('TestExchange', 'testMethod', 'attempt');
      recordApiCall('TestExchange', 'testMethod', 'attempt');
      recordApiCall('TestExchange', 'testMethod', 'attempt');
      
      // Record 3 successes and 1 failure
      recordApiCall('TestExchange', 'testMethod', 'success', 100);
      recordApiCall('TestExchange', 'testMethod', 'success', 150);
      recordApiCall('TestExchange', 'testMethod', 'success', 200);
      recordApiCall('TestExchange', 'testMethod', 'failure', 300, new Error('Test error'));
      
      // Calculate success rate
      const successRate = getSuccessRate('TestExchange', 'testMethod');
      
      // Verify success rate (3/4 = 0.75)
      expect(successRate).toBe(0.75);
    });
    
    it('should calculate average duration correctly', () => {
      // Record API calls
      recordApiCall('TestExchange', 'testMethod', 'attempt');
      recordApiCall('TestExchange', 'testMethod', 'attempt');
      
      // Record durations
      recordApiCall('TestExchange', 'testMethod', 'success', 100);
      recordApiCall('TestExchange', 'testMethod', 'success', 300);
      
      // Calculate average duration
      const avgDuration = getAverageDuration('TestExchange', 'testMethod');
      
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
    beforeEach(() => {
      // Ensure circuit breaker is reset before each test
      resetAllMocks();
    });
    
    it('should allow API calls when circuit is closed', () => {
      // Default state is closed
      const canCall = canMakeApiCall('TestExchange', 'testMethod');
      expect(canCall).toBe(true);
    });

    it('should open circuit after failure threshold is reached', () => {
      // Record multiple failures (default threshold is 5 in the implementation)
      for (let i = 0; i < 5; i++) {
        recordCircuitBreakerResult('TestExchange', 'testMethod', false, { failureThreshold: 5 });
      }
      
      // Check if circuit is open
      const canCall = canMakeApiCall('TestExchange', 'testMethod');
      expect(canCall).toBe(false);
      
      // Check circuit state
      const state = getCircuitBreakerState('TestExchange', 'testMethod');
      expect(state).toBeTruthy();
      expect(state?.state).toBe('open');
    });

    it('should transition to half-open state after timeout', () => {
      // Record failures to open circuit
      for (let i = 0; i < 5; i++) {
        recordCircuitBreakerResult('TestExchange', 'testMethod', false, { failureThreshold: 5 });
      }
      
      // Verify circuit is open
      expect(canMakeApiCall('TestExchange', 'testMethod', { failureThreshold: 5 })).toBe(false);
      
      // Advance time past reset timeout (default 30 seconds = 30000ms)
      mockDateNow.mockImplementation(() => 40000);
      
      // Circuit should now allow a call (transitioning to half-open)
      const canCall = canMakeApiCall('TestExchange', 'testMethod', { failureThreshold: 5 });
      expect(canCall).toBe(true);
    });

    it('should close circuit after successful call in half-open state', () => {
      // Open circuit with consistent config
      const testConfig = { failureThreshold: 5, resetTimeoutMs: 30000 };
      
      for (let i = 0; i < 5; i++) {
        recordCircuitBreakerResult('TestExchange', 'testMethod', false, testConfig);
      }
      
      // Transition to half-open
      mockDateNow.mockImplementation(() => 40000);
      const canCall = canMakeApiCall('TestExchange', 'testMethod', testConfig);
      expect(canCall).toBe(true);
      
      // Record success
      recordCircuitBreakerResult('TestExchange', 'testMethod', true, testConfig);
      
      // Circuit should be closed
      expect(canMakeApiCall('TestExchange', 'testMethod', testConfig)).toBe(true);
    });

    it('should reopen circuit after failure in half-open state', () => {
      // Open circuit with consistent config
      const testConfig = { failureThreshold: 5, resetTimeoutMs: 30000 };
      
      for (let i = 0; i < 5; i++) {
        recordCircuitBreakerResult('TestExchange', 'testMethod', false, testConfig);
      }
      
      // Transition to half-open
      mockDateNow.mockImplementation(() => 40000);
      const canCall = canMakeApiCall('TestExchange', 'testMethod', testConfig);
      expect(canCall).toBe(true);
      
      // Record failure
      recordCircuitBreakerResult('TestExchange', 'testMethod', false, testConfig);
      
      // Circuit should be open again
      expect(canMakeApiCall('TestExchange', 'testMethod', testConfig)).toBe(false);
    });

    it('should reset circuit breaker when requested', () => {
      // Open circuit with consistent config
      const testConfig = { failureThreshold: 5, resetTimeoutMs: 30000 };
      
      for (let i = 0; i < 5; i++) {
        recordCircuitBreakerResult('TestExchange', 'testMethod', false, testConfig);
      }
      
      // Verify circuit is open
      expect(canMakeApiCall('TestExchange', 'testMethod', testConfig)).toBe(false);
      
      // Reset circuit
      resetCircuitBreaker('TestExchange', 'testMethod');
      
      // Circuit should be closed
      expect(canMakeApiCall('TestExchange', 'testMethod', testConfig)).toBe(true);
    });
  });
  
  describe('Performance Dashboard', () => {
    beforeEach(() => {
      // Reset mocks and set initial time
      resetAllMocks();
    });
    
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
