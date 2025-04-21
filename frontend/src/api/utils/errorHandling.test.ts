/**
 * Tests for error handling utilities
 */
import { executeApiCall, withRetry, ApiError, NetworkError } from './errorHandling';

// Mock dependencies
jest.mock('./monitoring', () => ({
  recordApiCall: jest.fn(),
  canMakeApiCall: jest.fn().mockReturnValue(true),
  recordCircuitBreakerResult: jest.fn()
}));

// Mock the sleep function to avoid waiting in tests
jest.mock('./utils', () => ({
  sleep: jest.fn().mockImplementation(() => Promise.resolve(undefined))
}));

describe('Error Handling Utilities', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('executeApiCall', () => {
    it('should execute the API call and return the result on success', async () => {
      // Setup a mock function that resolves with 'success'
      const mockFn = jest.fn().mockImplementation(() => Promise.resolve('success'));
      
      // Execute the function
      const result = await executeApiCall(mockFn);
      
      // Assert the results
      expect(result).toBe('success');
      expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('should handle errors and rethrow them', async () => {
      // Setup a mock function that rejects with an error
      const error = new Error('API error');
      const mockFn = jest.fn().mockImplementation(() => Promise.reject(error));
      
      // Execute and assert
      await expect(await executeApiCall(mockFn)).rejects.toThrow('API error');
      expect(mockFn).toHaveBeenCalledTimes(1);
    });
  });

  describe('withRetry', () => {
    it('should not retry when the call succeeds', async () => {
      // Setup a mock function that resolves with 'success'
      const mockFn = jest.fn().mockImplementation(() => Promise.resolve('success'));
      
      // Execute the function
      const result = await withRetry(mockFn);
      
      // Assert the results
      expect(result).toBe('success');
      expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('should retry the specified number of times before giving up', async () => {
      // Setup a mock function that always rejects
      const error = new Error('Persistent error');
      const mockFn = jest.fn().mockImplementation(() => Promise.reject(error));
      
      // Create a retryableErrors function that always returns true
      const retryableErrorsFn = jest.fn().mockImplementation(() => true);
      
      // Execute and assert
      await expect(await withRetry(mockFn, {
        maxRetries: 3,
        initialDelayMs: 10,
        maxDelayMs: 100,
        retryableErrors: retryableErrorsFn,
      })).rejects.toThrow('Persistent error');
      
      // Should be called maxRetries times
      expect(mockFn).toHaveBeenCalledTimes(3);
      // retryableErrors should be called maxRetries - 1 times
      expect(retryableErrorsFn).toHaveBeenCalledTimes(2);
    });

    it('should not retry when error is not retryable', async () => {
      // Setup a mock function that always rejects
      const error = new Error('Non-retryable error');
      const mockFn = jest.fn().mockImplementation(() => Promise.reject(error));
      
      // Create a retryableErrors function that always returns false
      const retryableErrorsFn = jest.fn().mockImplementation(() => false);
      
      // Execute and assert
      await expect(await withRetry(mockFn, {
        maxRetries: 3,
        initialDelayMs: 10,
        maxDelayMs: 100,
        retryableErrors: retryableErrorsFn,
      })).rejects.toThrow('Non-retryable error');
      
      // Should only be called once since error is not retryable
      expect(mockFn).toHaveBeenCalledTimes(1);
      expect(retryableErrorsFn).toHaveBeenCalledTimes(1);
    });
  });

  describe('Error Classes', () => {
    it('should create ApiError with correct properties', () => {
      // Create an ApiError instance
      const error = new ApiError('API error', 429, { message: 'Rate limited' }, true);
      
      // Assert the properties
      expect(error.message).toBe('API error');
      expect(error.status).toBe(429);
      expect(error.data).toEqual({ message: 'Rate limited' });
      expect(error.isRetryable).toBe(true);
      expect(error.name).toBe('ApiError');
    });

    it('should create NetworkError with correct properties', () => {
      // Create a NetworkError instance
      const error = new NetworkError('Network error', true);
      
      // Assert the properties
      expect(error.message).toBe('Network error');
      expect(error.isRetryable).toBe(true);
      expect(error.name).toBe('NetworkError');
    });
  });
});
