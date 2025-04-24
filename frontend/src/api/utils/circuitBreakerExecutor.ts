import {
  logApiCallAttempt,
  logApiCallFailure,
  logApiCallSuccess,
  logCircuitBreakerStateChange,
  logFallbackAttempt,
  logFallbackFailure,
  logFallbackSuccess,
} from './enhancedLogging';
import { getEnhancedApiMetrics, recordCircuitBreakerStateChange, recordEnhancedApiCall } from './enhancedMonitoring';
import { ApiError, executeApiCall, NetworkError } from './errorHandling';
import { cacheForFallback, executeFallback } from './fallback';
import { canMakeApiCall, recordApiCall, recordCircuitBreakerResult } from './monitoring';
import {
  createBatchProcessor,
  getPerformanceMetrics,
  measureExecutionTime,
  memoize
} from './performanceOptimizations';

/**
 * Options for the circuit breaker executor
 */
export interface CircuitBreakerOptions<T> {
  /** Exchange name (e.g., 'Alpaca', 'Binance', 'Coinbase') */
  exchange: string;
  /** Method name (e.g., 'getPortfolio', 'createOrder') */
  method: string;
  /** Maximum number of retries for retryable errors */
  maxRetries?: number;
  /** Initial delay in milliseconds between retries */
  initialDelayMs?: number;
  /** Maximum delay in milliseconds between retries */
  maxDelayMs?: number;
  /** Function to determine if an error is retryable */
  isRetryable?: (error: Error) => boolean;
  /** Primary fallback function */
  primaryFallback?: () => Promise<T>;
  /** Secondary fallback function */
  secondaryFallback?: () => Promise<T>;
  /** Cache retrieval function */
  cacheRetrieval?: () => Promise<T | null>;
  /** Validator function for fallback results */
  validator?: (result: T) => boolean;
  /** Maximum age of cached data in milliseconds */
  maxCacheAgeMs?: number;
  /** Whether this operation is critical */
  isCritical?: boolean;
  /** Whether to enable memoization for this API call */
  enableMemoization?: boolean;
  /** Maximum age of memoized results in milliseconds */
  memoizationMaxAgeMs?: number;
  /** Maximum number of memoized results to store */
  memoizationMaxCacheSize?: number;
}

/**
 * Default options for circuit breaker
 */
const DEFAULT_OPTIONS = {
  maxRetries: 3,
  initialDelayMs: 500,
  maxDelayMs: 5000,
  maxCacheAgeMs: 5 * 60 * 1000, // 5 minutes
  isCritical: false,
  enableMemoization: true,
  memoizationMaxAgeMs: 30 * 1000, // 30 seconds
  memoizationMaxCacheSize: 50,
};

/**
 * Execute a function with circuit breaker pattern
 * 
 * @param apiCall Function to execute
 * @param options Circuit breaker options
 * @returns Result of the function or fallback
 * @throws Error if the function fails and no fallback is available
 */
export const executeWithCircuitBreaker = measureExecutionTime(async <T>(
  apiCall: () => Promise<T>,
  options: CircuitBreakerOptions<T>
): Promise<T> => {
  const {
    exchange,
    method,
    maxRetries = DEFAULT_OPTIONS.maxRetries,
    initialDelayMs = DEFAULT_OPTIONS.initialDelayMs,
    maxDelayMs = DEFAULT_OPTIONS.maxDelayMs,
    isRetryable,
    primaryFallback,
    secondaryFallback,
    cacheRetrieval,
    validator,
    maxCacheAgeMs = DEFAULT_OPTIONS.maxCacheAgeMs,
    isCritical = DEFAULT_OPTIONS.isCritical,
    enableMemoization = DEFAULT_OPTIONS.enableMemoization,
    memoizationMaxAgeMs = DEFAULT_OPTIONS.memoizationMaxAgeMs,
    memoizationMaxCacheSize = DEFAULT_OPTIONS.memoizationMaxCacheSize,
  } = options;

  // Check if circuit breaker is open
  if (!canMakeApiCall(exchange, method)) {
    console.warn(`Circuit breaker open for ${exchange} ${method}, using fallback`);
    recordEnhancedApiCall(exchange, method, 'fallback_attempt');

    // Log circuit breaker open state
    logCircuitBreakerStateChange(
      exchange,
      method,
      'open',
      'open',
      'Circuit breaker is open, attempting fallback'
    );

    // Use fallback mechanism if available
    if (primaryFallback || secondaryFallback || cacheRetrieval) {
      try {
        // Log fallback attempt
        if (primaryFallback) {
          logFallbackAttempt(exchange, method, 'primary');
        } else if (cacheRetrieval) {
          logFallbackAttempt(exchange, method, 'cache');
        } else if (secondaryFallback) {
          logFallbackAttempt(exchange, method, 'secondary');
        }

        const startTime = Date.now();
        const result = await executeFallback({
          primary: primaryFallback,
          secondary: secondaryFallback,
          cache: cacheRetrieval,
          validator,
          maxCacheAgeMs,
          context: {
            exchange,
            method,
            isCritical
          }
        });

        const duration = Date.now() - startTime;

        // Log fallback success
        if (primaryFallback) {
          logFallbackSuccess(exchange, method, 'primary', duration);
        } else if (cacheRetrieval) {
          logFallbackSuccess(exchange, method, 'cache', duration);
        } else if (secondaryFallback) {
          logFallbackSuccess(exchange, method, 'secondary', duration);
        }

        return result;
      } catch (fallbackError) {
        recordEnhancedApiCall(exchange, method, 'fallback_failure');

        // Log fallback failure
        if (primaryFallback) {
          logFallbackFailure(exchange, method, 'primary', fallbackError as Error);
        } else if (cacheRetrieval) {
          logFallbackFailure(exchange, method, 'cache', fallbackError as Error);
        } else if (secondaryFallback) {
          logFallbackFailure(exchange, method, 'secondary', fallbackError as Error);
        }

        console.error(`${exchange} fallback error in ${method}:`, fallbackError);
        throw new Error(`${exchange} API ${method} unavailable and fallback failed: ${(fallbackError as Error).message}`);
      }
    }

    throw new Error(`${exchange} API ${method} unavailable and no fallback provided`);
  }

  const startTime = Date.now();

  try {
    // Record the API call attempt
    recordEnhancedApiCall(exchange, method, 'attempt');
    recordApiCall(exchange, method, 'attempt');

    // Log API call attempt
    logApiCallAttempt(exchange, method);

    // Create a memoized version of the API call if enabled
    const memoizedApiCall = enableMemoization ?
      memoize(apiCall, {
        maxAgeMs: memoizationMaxAgeMs,
        maxCacheSize: memoizationMaxCacheSize,
        keyGenerator: () => `${exchange}:${method}`, // Use exchange and method as the cache key
      }) :
      apiCall;

    // Execute the API call with retry logic
    const result = await executeApiCall<T>(() => memoizedApiCall(), {
      maxRetries,
      initialDelayMs,
      maxDelayMs,
      retryableErrors: isRetryable || ((error: any) => {
        // Default retry logic: retry on network errors and server errors (5xx)
        if (error instanceof NetworkError) return true;
        if (error instanceof ApiError && error.isRetryable) return true;
        return false;
      }),
    });

    // Record successful API call with duration
    const duration = Date.now() - startTime;
    recordEnhancedApiCall(exchange, method, 'success', duration);
    recordApiCall(exchange, method, 'success', duration);
    recordCircuitBreakerResult(exchange, method, true);

    // Log API call success
    logApiCallSuccess(exchange, method, duration, result);

    // Cache successful result for potential future fallbacks
    cacheForFallback(`${exchange}:${method}`, result);

    return result;
  } catch (error) {
    // Record failed API call with duration
    const duration = Date.now() - startTime;
    recordEnhancedApiCall(exchange, method, 'failure', duration, error as Error);
    recordApiCall(exchange, method, 'failure', duration, error as Error);

    // Log API call failure
    logApiCallFailure(exchange, method, error as Error, duration);

    // Record circuit breaker result and potentially change state
    const previousState = getEnhancedApiMetrics(exchange, method).circuitBreakerState.state;
    recordCircuitBreakerResult(exchange, method, false);
    const currentState = getEnhancedApiMetrics(exchange, method).circuitBreakerState.state;

    // Record state change if it occurred
    if (previousState !== currentState) {
      recordCircuitBreakerStateChange(
        exchange,
        method,
        previousState,
        currentState,
        `Error: ${(error as Error).message}`
      );

      // Log circuit breaker state change
      logCircuitBreakerStateChange(
        exchange,
        method,
        previousState,
        currentState,
        `Error: ${(error as Error).message}`
      );
    }

    console.error(`${exchange} API error in ${method}:`, error);

    // Use fallback mechanism if available
    if (primaryFallback || secondaryFallback || cacheRetrieval) {
      try {
        recordEnhancedApiCall(exchange, method, 'fallback_attempt');

        // Log fallback attempt
        if (primaryFallback) {
          logFallbackAttempt(exchange, method, 'primary');
        } else if (cacheRetrieval) {
          logFallbackAttempt(exchange, method, 'cache');
        } else if (secondaryFallback) {
          logFallbackAttempt(exchange, method, 'secondary');
        }

        const fallbackStartTime = Date.now();
        const result = await executeFallback({
          primary: primaryFallback,
          secondary: secondaryFallback,
          cache: cacheRetrieval,
          validator,
          maxCacheAgeMs,
          context: {
            exchange,
            method,
            isCritical,
            error: error as Error
          }
        });

        const fallbackDuration = Date.now() - fallbackStartTime;

        // Log fallback success
        if (primaryFallback) {
          logFallbackSuccess(exchange, method, 'primary', fallbackDuration);
        } else if (cacheRetrieval) {
          logFallbackSuccess(exchange, method, 'cache', fallbackDuration);
        } else if (secondaryFallback) {
          logFallbackSuccess(exchange, method, 'secondary', fallbackDuration);
        }

        return result;
      } catch (fallbackError) {
        recordEnhancedApiCall(exchange, method, 'fallback_failure');

        // Log fallback failure
        if (primaryFallback) {
          logFallbackFailure(exchange, method, 'primary', fallbackError as Error);
        } else if (cacheRetrieval) {
          logFallbackFailure(exchange, method, 'cache', fallbackError as Error);
        } else if (secondaryFallback) {
          logFallbackFailure(exchange, method, 'secondary', fallbackError as Error);
        }

        console.error(`${exchange} fallback error in ${method}:`, fallbackError);
        throw new Error(`${exchange} API ${method} failed and fallback failed: ${(fallbackError as Error).message}`);
      }
    }

    // Re-throw the error if no fallback
    throw error as Error;
  }
}, 'executeWithCircuitBreaker');

/**
 * Create a batch processor for circuit breaker API calls
 * @param processBatchFn Function to process a batch of API calls
 * @param options Batch processor options
 * @returns Batch processor
 */
export const createCircuitBreakerBatchProcessor = <T, R>(
  processBatchFn: (items: T[]) => Promise<R[]>,
  options: {
    exchange: string;
    method: string;
    maxBatchSize?: number;
    maxWaitMs?: number;
    keyGenerator?: (item: T) => string;
  }
) => {
  const {
    exchange,
    method,
    maxBatchSize = 10,
    maxWaitMs = 100,
    keyGenerator,
  } = options;

  return createBatchProcessor<T, R>({
    maxBatchSize,
    maxWaitMs,
    keyGenerator,
    processBatch: async (items: T[]) => {
      // Execute batch with circuit breaker
      return executeWithCircuitBreaker(
        () => processBatchFn(items),
        {
          exchange,
          method: `${method}_batch`,
          maxRetries: DEFAULT_OPTIONS.maxRetries,
          initialDelayMs: DEFAULT_OPTIONS.initialDelayMs,
          maxDelayMs: DEFAULT_OPTIONS.maxDelayMs,
        }
      );
    },
  });
};

/**
 * Get performance metrics for circuit breaker executions
 * @returns Performance metrics for circuit breaker executions
 */
export const getCircuitBreakerPerformanceMetrics = () => {
  return getPerformanceMetrics();
};
