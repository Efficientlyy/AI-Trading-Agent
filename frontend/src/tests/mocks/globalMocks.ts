/**
 * Global mock implementations for commonly used modules
 * This provides consistent mocking across all test files
 */
import { jest } from '@jest/globals';
import { ApiCallMetrics } from '../../api/utils/monitoring';

// Type definitions for axios mock to fix TypeScript errors
type AxiosResponseData = { data: Record<string, any> };

// Create a compatible mock interface that matches Jest's expectations
interface MockContext<TReturn, TArgs extends any[]> {
  calls: TArgs[];
  instances: any[];
  invocationCallOrder: number[];
  results: Array<{type: string; value: TReturn}>;
  lastCall?: TArgs; // Make lastCall optional to match Jest's implementation
}

// Define a compatible mock function type
interface CompatibleMock<TReturn = any, TArgs extends any[] = any[]> {
  (...args: TArgs): TReturn;
  mock: MockContext<TReturn, TArgs>;
  mockClear(): void;
  mockReset(): void;
  mockRestore(): void;
  mockImplementation(fn: (...args: TArgs) => TReturn): CompatibleMock<TReturn, TArgs>;
  mockImplementationOnce(fn: (...args: TArgs) => TReturn): CompatibleMock<TReturn, TArgs>;
  mockReturnValue(value: TReturn): CompatibleMock<TReturn, TArgs>;
  mockReturnValueOnce(value: TReturn): CompatibleMock<TReturn, TArgs>;
}

// Use this type for all Jest mocks
type JestMockFn = CompatibleMock;

interface AxiosMockInstance {
  get: JestMockFn;
  post: JestMockFn;
  put: JestMockFn;
  delete: JestMockFn;
  request: JestMockFn;
  interceptors: {
    request: { use: JestMockFn; eject: JestMockFn; clear: JestMockFn };
    response: { use: JestMockFn; eject: JestMockFn; clear: JestMockFn };
  };
  defaults: {
    headers: {
      common: { Accept: string };
      delete: Record<string, unknown>;
      get: Record<string, unknown>;
      head: Record<string, unknown>;
      post: { 'Content-Type': string };
      put: { 'Content-Type': string };
      patch: { 'Content-Type': string };
    }
  };
}

interface AxiosMock {
  create: JestMockFn;
  get: JestMockFn;
  post: JestMockFn;
  put: JestMockFn;
  delete: JestMockFn;
  request: JestMockFn;
  isAxiosError: JestMockFn;
  interceptors: {
    request: { use: JestMockFn; eject: JestMockFn; clear: JestMockFn };
    response: { use: JestMockFn; eject: JestMockFn; clear: JestMockFn };
  };
  defaults: {
    headers: {
      common: { Accept: string };
      delete: Record<string, unknown>;
      get: Record<string, unknown>;
      head: Record<string, unknown>;
      post: { 'Content-Type': string };
      put: { 'Content-Type': string };
      patch: { 'Content-Type': string };
    }
  };
}

// Mock axios
export const mockAxios = (): AxiosMock => {
  const mockData: AxiosResponseData = { data: {} };
  
  const axiosInstance: AxiosMockInstance = {
    get: jest.fn().mockImplementation(() => Promise.resolve(mockData)),
    post: jest.fn().mockImplementation(() => Promise.resolve(mockData)),
    put: jest.fn().mockImplementation(() => Promise.resolve(mockData)),
    delete: jest.fn().mockImplementation(() => Promise.resolve(mockData)),
    request: jest.fn().mockImplementation(() => Promise.resolve(mockData)),
    interceptors: {
      request: { use: jest.fn(), eject: jest.fn(), clear: jest.fn() },
      response: { use: jest.fn(), eject: jest.fn(), clear: jest.fn() },
    },
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
    }
  };

  return {
    create: jest.fn().mockReturnValue(axiosInstance),
    get: jest.fn().mockImplementation(() => Promise.resolve(mockData)),
    post: jest.fn().mockImplementation(() => Promise.resolve(mockData)),
    put: jest.fn().mockImplementation(() => Promise.resolve(mockData)),
    delete: jest.fn().mockImplementation(() => Promise.resolve(mockData)),
    request: jest.fn().mockImplementation(() => Promise.resolve(mockData)),
    isAxiosError: jest.fn().mockImplementation((error: any) => error?.isAxiosError === true),
    interceptors: {
      request: { use: jest.fn(), eject: jest.fn(), clear: jest.fn() },
      response: { use: jest.fn(), eject: jest.fn(), clear: jest.fn() },
    },
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
    }
  };
};

// Mock monitoring utilities
export const mockMonitoring = () => {
  const defaultMetrics: ApiCallMetrics = {
    totalCalls: 0,
    successCalls: 0,
    failedCalls: 0,
    totalDuration: 0,
    minDuration: 0,
    maxDuration: 0,
    lastCallTime: 0
  };

  return {
    recordApiCall: jest.fn(),
    canMakeApiCall: jest.fn().mockReturnValue(true),
    recordCircuitBreakerResult: jest.fn(),
    resetCircuitBreaker: jest.fn(),
    getApiCallMetrics: jest.fn().mockReturnValue(defaultMetrics),
    getCircuitBreakerState: jest.fn().mockReturnValue({
      state: 'closed',
      remainingTimeMs: 0
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

// Mock crypto for signatures
export const mockCrypto = () => {
  return {
    createHmac: jest.fn().mockReturnValue({
      update: jest.fn().mockReturnThis(),
      digest: jest.fn().mockReturnValue('mocked-signature'),
    }),
  };
};

// Mock localStorage
export const mockLocalStorage = () => {
  const store: Record<string, string> = {};
  
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: jest.fn((key: string) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      Object.keys(store).forEach(key => delete store[key]);
    }),
    key: jest.fn((index: number) => Object.keys(store)[index] || null),
    length: jest.fn(() => Object.keys(store).length),
  };
};

// Mock setTimeout to avoid waiting in tests
export const mockSetTimeout = () => {
  return jest.spyOn(global, 'setTimeout').mockImplementation((callback: any) => {
    if (typeof callback === 'function') {
      callback();
    }
    return 1 as any; // Return a number as a timer ID
  });
};
