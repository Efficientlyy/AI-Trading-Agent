/**
 * Error handling and retry utilities for API calls
 */
import axios, { AxiosError, AxiosResponse } from 'axios';

// Custom error classes
export class ApiError extends Error {
  status: number;
  data: any;
  isRetryable: boolean;

  constructor(message: string, status: number, data?: any, isRetryable = true) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
    this.isRetryable = isRetryable;
  }
}

export class NetworkError extends Error {
  isRetryable: boolean;

  constructor(message: string, isRetryable = true) {
    super(message);
    this.name = 'NetworkError';
    this.isRetryable = isRetryable;
  }
}

export class RateLimitError extends ApiError {
  retryAfter?: number;

  constructor(message: string, status: number, retryAfter?: number, data?: any) {
    super(message, status, data, true);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

// Error classification
export const classifyError = (error: any): Error => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;
    
    // No response (network error)
    if (!axiosError.response) {
      return new NetworkError(
        axiosError.message || 'Network error occurred',
        true
      );
    }
    
    const { status, data } = axiosError.response as AxiosResponse;
    const message = 
      (data && typeof data === 'object' && 'message' in data) 
        ? data.message 
        : axiosError.message || 'API error occurred';
    
    // Rate limit errors
    if (status === 429) {
      const retryAfter = axiosError.response?.headers['retry-after']
        ? parseInt(axiosError.response.headers['retry-after'], 10) * 1000
        : undefined;
      
      return new RateLimitError(
        'Rate limit exceeded',
        status,
        retryAfter,
        data
      );
    }
    
    // Server errors (5xx) are retryable
    if (status >= 500 && status < 600) {
      return new ApiError(
        message,
        status,
        data,
        true // Retryable
      );
    }
    
    // Client errors (4xx) are generally not retryable (except rate limiting)
    if (status >= 400 && status < 500) {
      return new ApiError(
        message,
        status,
        data,
        false // Not retryable
      );
    }
    
    // Other status codes
    return new ApiError(
      message,
      status,
      data,
      true // Default to retryable
    );
  }
  
  // Non-Axios errors
  if (error instanceof Error) {
    return error;
  }
  
  // Unknown errors
  return new Error(
    typeof error === 'string' 
      ? error 
      : 'An unknown error occurred'
  );
};

// Retry configuration
export interface RetryConfig {
  maxRetries: number;
  initialDelayMs: number;
  maxDelayMs: number;
  backoffFactor: number;
  retryableErrors?: (error: Error) => boolean;
}

const defaultRetryConfig: RetryConfig = {
  maxRetries: 3,
  initialDelayMs: 1000,
  maxDelayMs: 10000,
  backoffFactor: 2,
  retryableErrors: (error: Error) => {
    if ('isRetryable' in error) {
      return (error as any).isRetryable;
    }
    
    // Default retryable errors
    return (
      error instanceof NetworkError ||
      error instanceof RateLimitError ||
      (error instanceof ApiError && error.status >= 500)
    );
  }
};

// Calculate backoff delay with jitter
const calculateBackoffDelay = (
  attempt: number, 
  { initialDelayMs, maxDelayMs, backoffFactor }: RetryConfig
): number => {
  // Exponential backoff formula: initialDelay * (backoffFactor ^ attempt)
  const exponentialDelay = initialDelayMs * Math.pow(backoffFactor, attempt);
  
  // Add jitter (random value between 0 and 1) to prevent thundering herd
  const jitter = Math.random();
  const delay = exponentialDelay * (1 + jitter * 0.1);
  
  // Respect the maximum delay
  return Math.min(delay, maxDelayMs);
};

// Sleep utility
const sleep = (ms: number): Promise<void> => 
  new Promise(resolve => setTimeout(resolve, ms));

// Retry function
export const withRetry = async <T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): Promise<T> => {
  // Ensure we have a complete retry config by merging with defaults
  const retryConfig = { ...defaultRetryConfig, ...config };
  let lastError: Error | null = null;
  
  // Make sure retryableErrors is a function
  const checkRetryable = typeof retryConfig.retryableErrors === 'function' 
    ? retryConfig.retryableErrors 
    : defaultRetryConfig.retryableErrors;
  
  for (let attempt = 0; attempt < retryConfig.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = classifyError(error);
      
      // Check if error is retryable using the function we verified above
      const isRetryable = checkRetryable ? checkRetryable(lastError) : false;
      if (!isRetryable) {
        throw lastError;
      }
      
      // If this is the last attempt, don't wait, just throw
      if (attempt === retryConfig.maxRetries - 1) {
        throw lastError;
      }
      
      // Handle rate limit with specific retry-after header
      if (lastError instanceof RateLimitError && lastError.retryAfter) {
        await sleep(lastError.retryAfter);
        continue;
      }
      
      // Calculate backoff delay
      const delay = calculateBackoffDelay(attempt, retryConfig);
      
      // Log retry attempt
      console.warn(
        `API call failed with error: ${lastError.message}. Retrying in ${delay}ms (attempt ${attempt + 1}/${retryConfig.maxRetries})`,
        lastError
      );
      
      // Wait before retrying
      await sleep(delay);
    }
  }
  
  // This should never happen because we throw on the last attempt
  throw lastError || new Error('Maximum retry attempts reached');
};

// Retry decorator for class methods
export function retryable(
  config: Partial<RetryConfig> = {}
): MethodDecorator {
  return function(
    target: Object,
    propertyKey: string | symbol,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;
    
    descriptor.value = function(...args: any[]) {
      return withRetry(() => originalMethod.apply(this, args), config);
    };
    
    return descriptor;
  };
}

// Logging utility
export const logApiCall = (
  exchange: string,
  method: string,
  success: boolean,
  duration: number,
  params?: any,
  error?: Error
): void => {
  const status = success ? 'SUCCESS' : 'FAILURE';
  const message = `[${exchange}] ${method} - ${status} (${duration}ms)`;
  
  if (success) {
    console.log(message, params);
  } else {
    console.error(message, params, error);
  }
};

// Measure execution time
export const measureExecutionTime = async <T>(
  fn: () => Promise<T>
): Promise<[T, number]> => {
  const startTime = performance.now();
  try {
    const result = await fn();
    const endTime = performance.now();
    const duration = Math.round(endTime - startTime);
    return [result, duration];
  } catch (error) {
    const endTime = performance.now();
    const duration = Math.round(endTime - startTime);
    throw { error, duration };
  }
};

// Wrapper for API calls with retry, logging, and timing
export const executeApiCall = async <T>(
  fn: () => Promise<T>,
  retryConfig: Partial<RetryConfig> = {}
): Promise<T> => {
  try {
    const [result, duration] = await measureExecutionTime(
      () => withRetry(fn, retryConfig)
    );
    
    return result;
  } catch (caught) {
    const { error, duration } = caught as { error: Error; duration: number };
    throw error;
  }
};
