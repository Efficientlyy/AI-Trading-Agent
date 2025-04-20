/**
 * Standard mock implementations for commonly used modules
 * This helps ensure consistency across test files
 */

import { jest } from '@jest/globals';

// Define response type for better type safety
type AxiosResponse<T = any> = {
  data: T;
  status?: number;
  statusText?: string;
  headers?: Record<string, string>;
  config?: any;
};

// Standard axios mock
export const createAxiosMock = () => {
  const axiosInstance = {
    get: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    post: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    put: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    delete: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    request: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    getUri: jest.fn(),
    defaults: {
      headers: {
        common: { Accept: 'application/json, text/plain, */*' },
        delete: {},
        get: {},
        head: {},
        post: { 'Content-Type': 'application/x-www-form-urlencoded' },
        put: { 'Content-Type': 'application/x-www-form-urlencoded' },
        patch: { 'Content-Type': 'application/x-www-form-urlencoded' },
      }
    },
    interceptors: {
      request: { 
        use: jest.fn(), 
        eject: jest.fn(),
        clear: jest.fn()
      },
      response: { 
        use: jest.fn(), 
        eject: jest.fn(),
        clear: jest.fn()
      },
    },
    head: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    options: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    patch: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    postForm: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    putForm: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    patchForm: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
  };

  const axiosMock = {
    create: jest.fn(() => axiosInstance),
    get: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    post: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    put: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    delete: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    request: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    isAxiosError: jest.fn().mockImplementation((error: any) => {
      return error && error.isAxiosError === true;
    }),
    getUri: jest.fn(),
    defaults: {
      headers: {
        common: { Accept: 'application/json, text/plain, */*' },
        delete: {},
        get: {},
        head: {},
        post: { 'Content-Type': 'application/x-www-form-urlencoded' },
        put: { 'Content-Type': 'application/x-www-form-urlencoded' },
        patch: { 'Content-Type': 'application/x-www-form-urlencoded' },
      }
    },
    interceptors: {
      request: { 
        use: jest.fn(), 
        eject: jest.fn(),
        clear: jest.fn()
      },
      response: { 
        use: jest.fn(), 
        eject: jest.fn(),
        clear: jest.fn()
      },
    },
    head: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    options: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    patch: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    postForm: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    putForm: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    patchForm: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
  };

  return axiosMock;
};

// Standard monitoring utilities mock
export const createMonitoringMock = () => {
  return {
    recordApiCall: jest.fn(),
    canMakeApiCall: jest.fn().mockReturnValue(true),
    recordCircuitBreakerResult: jest.fn(),
    resetCircuitBreaker: jest.fn(),
    getCircuitBreakerState: jest.fn().mockReturnValue({
      state: 'closed',
      remainingTimeMs: 0
    }),
    getApiCallMetrics: jest.fn().mockReturnValue({
      totalCalls: 0,
      successCalls: 0,
      failedCalls: 0,
      totalDuration: 0,
      minDuration: 0,
      maxDuration: 0,
      lastCallTime: 0
    }),
    getAllMetrics: jest.fn().mockReturnValue({}),
    getSuccessRate: jest.fn().mockReturnValue(1),
    getAverageDuration: jest.fn().mockReturnValue(0),
    isApiHealthy: jest.fn().mockReturnValue(true),
    getApiHealthDashboard: jest.fn().mockReturnValue({
      exchanges: {},
      overallHealth: true,
      totalCalls: 0,
      successRate: 1,
      averageDuration: 0
    })
  };
};

// Standard authenticated client mock
export const createAuthenticatedClientMock = () => {
  return {
    get: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    post: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    put: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    delete: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} })),
    request: jest.fn().mockImplementation(() => Promise.resolve<AxiosResponse>({ data: {} }))
  };
};

// Standard crypto mock for signatures
export const createCryptoMock = () => {
  return {
    createHmac: jest.fn().mockReturnValue({
      update: jest.fn().mockReturnThis(),
      digest: jest.fn().mockReturnValue('mocked-signature'),
    }),
  };
};
