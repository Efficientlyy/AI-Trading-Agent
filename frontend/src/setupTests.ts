// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';
// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom


// Note: We need to handle react-router-dom v7.x in tests
// The solution is to create a __mocks__ directory with a mock implementation

// Setup Jest mocks for axios and other dependencies
jest.mock('axios', () => {
  const mockAxios = {
    create: jest.fn(() => ({
      get: jest.fn().mockResolvedValue({ data: {} }),
      post: jest.fn().mockResolvedValue({ data: {} }),
      put: jest.fn().mockResolvedValue({ data: {} }),
      delete: jest.fn().mockResolvedValue({ data: {} }),
      interceptors: {
        request: { use: jest.fn(), eject: jest.fn() },
        response: { use: jest.fn(), eject: jest.fn() },
      }
    })),
    get: jest.fn().mockResolvedValue({ data: {} }),
    post: jest.fn().mockResolvedValue({ data: {} }),
    put: jest.fn().mockResolvedValue({ data: {} }),
    delete: jest.fn().mockResolvedValue({ data: {} }),
    isAxiosError: jest.fn().mockImplementation((error) => {
      return error && error.isAxiosError === true;
    }),
  };
  
  // Add mockReturnValue and other Jest mock methods to the create function
  mockAxios.create.mockReturnValue = jest.fn();
  mockAxios.create.mockImplementation = jest.fn();
  mockAxios.create.mockResolvedValue = jest.fn();
  mockAxios.create.mockRejectedValue = jest.fn();
  
  return mockAxios;
});

// Mock monitoring utilities
jest.mock('./api/utils/monitoring', () => {
  const monitoring = {
    canMakeApiCall: jest.fn().mockReturnValue(true),
    recordApiCall: jest.fn(),
    recordCircuitBreakerResult: jest.fn(),
    getEnhancedApiMetrics: jest.fn().mockReturnValue({
      basicMetrics: {
        totalCalls: 0,
        successCalls: 0,
        failedCalls: 0,
        totalDuration: 0,
        minDuration: 0,
        maxDuration: 0,
        lastCallTime: Date.now()
      },
      enhancedMetrics: {
        successRate: 1,
        averageResponseTime: 0,
        healthScore: 100,
        reliabilityTrend: 'stable'
      },
      circuitBreakerState: {
        state: 'closed',
        failureCount: 0,
        lastAttemptTime: Date.now(),
        nextAttemptTime: null
      }
    }),
    resetCircuitBreaker: jest.fn(),
    getApiHealthDashboard: jest.fn().mockReturnValue({
      binance: {
        getMarketPrice: {
          healthScore: 100,
          reliability: 'stable',
          circuitState: 'closed',
          successRate: 1
        }
      }
    })
  };
  
  // Add mockReturnValue and other Jest mock methods
  monitoring.canMakeApiCall.mockReturnValue = jest.fn();
  monitoring.canMakeApiCall.mockImplementation = jest.fn();
  
  return monitoring;
});

// Mock circuit breaker executor
jest.mock('./api/utils/circuitBreakerExecutor', () => {
  const executor = {
    executeWithCircuitBreaker: jest.fn().mockImplementation((exchange, method, fn) => {
      // Properly handle the function call with the correct parameters
      if (typeof fn === 'function') {
        return Promise.resolve(fn());
      } else if (typeof method === 'function') {
        // Handle case where only two parameters are passed (method is actually the function)
        return Promise.resolve(method());
      }
      return Promise.resolve({});
    }),
  };
  
  // Add mockReturnValue and other Jest mock methods
  executor.executeWithCircuitBreaker.mockReturnValue = jest.fn();
  executor.executeWithCircuitBreaker.mockImplementation = jest.fn();
  executor.executeWithCircuitBreaker.mockResolvedValue = jest.fn();
  executor.executeWithCircuitBreaker.mockRejectedValue = jest.fn();
  
  return executor;
});
