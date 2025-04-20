/**
 * Performance optimization utilities for the AI Trading Agent
 * Provides memoization, caching strategies, and performance monitoring
 */

// Type definitions
type MemoizedFunction<T extends (...args: any[]) => any> = T & {
  cache: Map<string, { result: ReturnType<T>; timestamp: number }>;
  clearCache: () => void;
};

type BatchProcessor<T, R> = {
  add: (item: T) => Promise<R>;
  flush: () => Promise<void>;
  clear: () => void;
  getQueueSize: () => number;
  getPendingItems: () => T[];
};

/**
 * Options for memoization
 */
export interface MemoizationOptions {
  /** Maximum age of cached results in milliseconds */
  maxAgeMs?: number;
  /** Maximum number of cached results */
  maxCacheSize?: number;
  /** Custom key generator function */
  keyGenerator?: (...args: any[]) => string;
  /** Whether to cache rejected promises */
  cacheRejections?: boolean;
}

/**
 * Options for batch processing
 */
export interface BatchProcessorOptions<T, R> {
  /** Maximum batch size */
  maxBatchSize?: number;
  /** Maximum wait time before processing in milliseconds */
  maxWaitMs?: number;
  /** Batch processor function */
  processBatch: (items: T[]) => Promise<R[]>;
  /** Optional key generator to identify items */
  keyGenerator?: (item: T) => string;
}

/**
 * Performance metrics for function calls
 */
export interface PerformanceMetrics {
  /** Function name */
  functionName: string;
  /** Total number of calls */
  callCount: number;
  /** Total execution time in milliseconds */
  totalExecutionTime: number;
  /** Average execution time in milliseconds */
  averageExecutionTime: number;
  /** Minimum execution time in milliseconds */
  minExecutionTime: number;
  /** Maximum execution time in milliseconds */
  maxExecutionTime: number;
  /** Cache hit count (for memoized functions) */
  cacheHitCount?: number;
  /** Cache miss count (for memoized functions) */
  cacheMissCount?: number;
  /** Cache hit ratio (for memoized functions) */
  cacheHitRatio?: number;
}

// Store performance metrics
const performanceMetrics: Record<string, PerformanceMetrics> = {};

/**
 * Memoize a function to cache its results
 * @param fn Function to memoize
 * @param options Memoization options
 * @returns Memoized function
 */
export function memoize<T extends (...args: any[]) => any>(
  fn: T,
  options: MemoizationOptions = {}
): MemoizedFunction<T> {
  const {
    maxAgeMs = 60000, // 1 minute default
    maxCacheSize = 100,
    keyGenerator = (...args) => JSON.stringify(args),
    cacheRejections = false,
  } = options;

  // Create cache
  const cache = new Map<string, { result: ReturnType<T>; timestamp: number }>();

  // Create memoized function
  const memoized = ((...args: Parameters<T>): ReturnType<T> => {
    const startTime = performance.now();
    const key = keyGenerator(...args);
    const functionName = fn.name || 'anonymous';

    // Initialize metrics if needed
    if (!performanceMetrics[functionName]) {
      performanceMetrics[functionName] = {
        functionName,
        callCount: 0,
        totalExecutionTime: 0,
        averageExecutionTime: 0,
        minExecutionTime: Infinity,
        maxExecutionTime: 0,
        cacheHitCount: 0,
        cacheMissCount: 0,
        cacheHitRatio: 0,
      };
    }

    // Update call count
    performanceMetrics[functionName].callCount++;

    // Check cache
    const cached = cache.get(key);
    if (cached && Date.now() - cached.timestamp <= maxAgeMs) {
      // Cache hit
      const endTime = performance.now();
      const executionTime = endTime - startTime;
      
      // Update metrics
      performanceMetrics[functionName].totalExecutionTime += executionTime;
      performanceMetrics[functionName].averageExecutionTime = 
        performanceMetrics[functionName].totalExecutionTime / performanceMetrics[functionName].callCount;
      performanceMetrics[functionName].minExecutionTime = 
        Math.min(performanceMetrics[functionName].minExecutionTime, executionTime);
      performanceMetrics[functionName].maxExecutionTime = 
        Math.max(performanceMetrics[functionName].maxExecutionTime, executionTime);
      performanceMetrics[functionName].cacheHitCount!++;
      performanceMetrics[functionName].cacheHitRatio = 
        performanceMetrics[functionName].cacheHitCount! / performanceMetrics[functionName].callCount;
      
      return cached.result;
    }

    // Cache miss
    performanceMetrics[functionName].cacheMissCount!++;
    performanceMetrics[functionName].cacheHitRatio = 
      (performanceMetrics[functionName].cacheHitCount || 0) / performanceMetrics[functionName].callCount;

    try {
      // Execute function
      const result = fn(...args);

      // Handle promises
      if (result instanceof Promise) {
        return result
          .then((resolvedResult) => {
            // Cache successful promise result
            cache.set(key, { result: resolvedResult as any, timestamp: Date.now() });
            
            // Limit cache size
            if (cache.size > maxCacheSize) {
              // Convert Map entries to array for sorting
              const entries: [string, { result: ReturnType<T>; timestamp: number }][] = [];
              cache.forEach((value, key) => {
                entries.push([key, value]);
              });
              
              // Sort by timestamp and get the oldest key
              const oldestKey = entries
                .sort((a, b) => a[1].timestamp - b[1].timestamp)[0][0];
              cache.delete(oldestKey);
            }

            // Update metrics
            const endTime = performance.now();
            const executionTime = endTime - startTime;
            performanceMetrics[functionName].totalExecutionTime += executionTime;
            performanceMetrics[functionName].averageExecutionTime = 
              performanceMetrics[functionName].totalExecutionTime / performanceMetrics[functionName].callCount;
            performanceMetrics[functionName].minExecutionTime = 
              Math.min(performanceMetrics[functionName].minExecutionTime, executionTime);
            performanceMetrics[functionName].maxExecutionTime = 
              Math.max(performanceMetrics[functionName].maxExecutionTime, executionTime);

            return resolvedResult;
          })
          .catch((error) => {
            // Cache rejected promise if configured
            if (cacheRejections) {
              const rejectedPromise = Promise.reject(error);
              cache.set(key, { result: rejectedPromise as any, timestamp: Date.now() });
            }

            // Update metrics
            const endTime = performance.now();
            const executionTime = endTime - startTime;
            performanceMetrics[functionName].totalExecutionTime += executionTime;
            performanceMetrics[functionName].averageExecutionTime = 
              performanceMetrics[functionName].totalExecutionTime / performanceMetrics[functionName].callCount;
            performanceMetrics[functionName].minExecutionTime = 
              Math.min(performanceMetrics[functionName].minExecutionTime, executionTime);
            performanceMetrics[functionName].maxExecutionTime = 
              Math.max(performanceMetrics[functionName].maxExecutionTime, executionTime);

            throw error;
          }) as ReturnType<T>;
      }

      // Cache synchronous result
      cache.set(key, { result: result as any, timestamp: Date.now() });
      
      // Limit cache size
      if (cache.size > maxCacheSize) {
        // Convert Map entries to array for sorting
        const entries: [string, { result: ReturnType<T>; timestamp: number }][] = [];
        cache.forEach((value, key) => {
          entries.push([key, value]);
        });
        
        // Sort by timestamp and get the oldest key
        const oldestKey = entries
          .sort((a, b) => a[1].timestamp - b[1].timestamp)[0][0];
        cache.delete(oldestKey);
      }

      // Update metrics
      const endTime = performance.now();
      const executionTime = endTime - startTime;
      performanceMetrics[functionName].totalExecutionTime += executionTime;
      performanceMetrics[functionName].averageExecutionTime = 
        performanceMetrics[functionName].totalExecutionTime / performanceMetrics[functionName].callCount;
      performanceMetrics[functionName].minExecutionTime = 
        Math.min(performanceMetrics[functionName].minExecutionTime, executionTime);
      performanceMetrics[functionName].maxExecutionTime = 
        Math.max(performanceMetrics[functionName].maxExecutionTime, executionTime);

      return result;
    } catch (error) {
      // Update metrics for synchronous errors
      const endTime = performance.now();
      const executionTime = endTime - startTime;
      performanceMetrics[functionName].totalExecutionTime += executionTime;
      performanceMetrics[functionName].averageExecutionTime = 
        performanceMetrics[functionName].totalExecutionTime / performanceMetrics[functionName].callCount;
      performanceMetrics[functionName].minExecutionTime = 
        Math.min(performanceMetrics[functionName].minExecutionTime, executionTime);
      performanceMetrics[functionName].maxExecutionTime = 
        Math.max(performanceMetrics[functionName].maxExecutionTime, executionTime);

      throw error;
    }
  }) as MemoizedFunction<T>;

  // Add cache and clearCache method to function
  memoized.cache = cache;
  memoized.clearCache = () => cache.clear();

  return memoized;
}

/**
 * Create a batch processor for API calls
 * @param options Batch processor options
 * @returns Batch processor
 */
export function createBatchProcessor<T, R>(
  options: BatchProcessorOptions<T, R>
): BatchProcessor<T, R> {
  const {
    maxBatchSize = 10,
    maxWaitMs = 100,
    processBatch,
    keyGenerator = (item) => JSON.stringify(item),
  } = options;

  // Queue of items to process
  let queue: T[] = [];
  // Map of promises for each item
  const promises = new Map<string, { resolve: (value: R) => void; reject: (reason: any) => void }>();
  // Timer for batch processing
  let timer: ReturnType<typeof setTimeout> | null = null;

  /**
   * Process the current batch
   */
  const processCurrentBatch = async () => {
    // Clear timer
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }

    // Get items to process
    const itemsToProcess = [...queue];
    queue = [];

    if (itemsToProcess.length === 0) {
      return;
    }

    try {
      // Process batch
      const results = await processBatch(itemsToProcess);

      // Resolve promises
      itemsToProcess.forEach((item, index) => {
        const key = keyGenerator(item);
        const promise = promises.get(key);
        if (promise) {
          promise.resolve(results[index]);
          promises.delete(key);
        }
      });
    } catch (error) {
      // Reject all promises
      itemsToProcess.forEach((item) => {
        const key = keyGenerator(item);
        const promise = promises.get(key);
        if (promise) {
          promise.reject(error);
          promises.delete(key);
        }
      });
    }
  };

  return {
    /**
     * Add an item to the batch
     * @param item Item to add
     * @returns Promise that resolves when the item is processed
     */
    add: (item: T): Promise<R> => {
      return new Promise<R>((resolve, reject) => {
        const key = keyGenerator(item);
        promises.set(key, { resolve, reject });
        queue.push(item);

        // Process immediately if batch size reached
        if (queue.length >= maxBatchSize) {
          processCurrentBatch();
        } else if (!timer) {
          // Set timer for batch processing
          timer = setTimeout(processCurrentBatch, maxWaitMs);
        }
      });
    },

    /**
     * Flush the current batch
     */
    flush: async (): Promise<void> => {
      await processCurrentBatch();
    },

    /**
     * Clear the current batch
     */
    clear: (): void => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }

      // Reject all promises
      queue.forEach((item) => {
        const key = keyGenerator(item);
        const promise = promises.get(key);
        if (promise) {
          promise.reject(new Error('Batch processor cleared'));
          promises.delete(key);
        }
      });

      queue = [];
    },

    /**
     * Get the current queue size
     */
    getQueueSize: (): number => {
      return queue.length;
    },

    /**
     * Get the pending items
     */
    getPendingItems: (): T[] => {
      return [...queue];
    },
  };
}

/**
 * Debounce a function
 * @param fn Function to debounce
 * @param wait Wait time in milliseconds
 * @returns Debounced function
 */
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return function(this: any, ...args: Parameters<T>): void {
    const context = this;

    if (timeout) {
      clearTimeout(timeout);
    }

    timeout = setTimeout(() => {
      timeout = null;
      fn.apply(context, args);
    }, wait);
  };
}

/**
 * Throttle a function
 * @param fn Function to throttle
 * @param limit Limit in milliseconds
 * @returns Throttled function
 */
export function throttle<T extends (...args: any[]) => any>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => ReturnType<T> | undefined {
  let lastCall = 0;
  let lastResult: ReturnType<T> | undefined;

  return function(this: any, ...args: Parameters<T>): ReturnType<T> | undefined {
    const now = Date.now();
    
    if (now - lastCall >= limit) {
      lastCall = now;
      lastResult = fn.apply(this, args);
    }
    
    return lastResult;
  };
}

/**
 * Get performance metrics for all tracked functions
 * @returns Performance metrics
 */
export function getPerformanceMetrics(): Record<string, PerformanceMetrics> {
  return { ...performanceMetrics };
}

/**
 * Clear performance metrics
 */
export function clearPerformanceMetrics(): void {
  Object.keys(performanceMetrics).forEach((key) => {
    delete performanceMetrics[key];
  });
}

/**
 * Measure the execution time of a function
 * @param fn Function to measure
 * @param functionName Optional function name for metrics
 * @returns Function with execution time measurement
 */
export function measureExecutionTime<T extends (...args: any[]) => any>(
  fn: T,
  functionName?: string
): T {
  const name = functionName || fn.name || 'anonymous';

  return function(this: any, ...args: Parameters<T>): ReturnType<T> {
    const startTime = performance.now();
    
    try {
      const result = fn.apply(this, args);
      
      // Handle promises
      if (result instanceof Promise) {
        return result
          .then((resolvedResult) => {
            const endTime = performance.now();
            const executionTime = endTime - startTime;
            
            // Initialize metrics if needed
            if (!performanceMetrics[name]) {
              performanceMetrics[name] = {
                functionName: name,
                callCount: 0,
                totalExecutionTime: 0,
                averageExecutionTime: 0,
                minExecutionTime: Infinity,
                maxExecutionTime: 0,
              };
            }
            
            // Update metrics
            performanceMetrics[name].callCount++;
            performanceMetrics[name].totalExecutionTime += executionTime;
            performanceMetrics[name].averageExecutionTime = 
              performanceMetrics[name].totalExecutionTime / performanceMetrics[name].callCount;
            performanceMetrics[name].minExecutionTime = 
              Math.min(performanceMetrics[name].minExecutionTime, executionTime);
            performanceMetrics[name].maxExecutionTime = 
              Math.max(performanceMetrics[name].maxExecutionTime, executionTime);
            
            return resolvedResult;
          })
          .catch((error) => {
            const endTime = performance.now();
            const executionTime = endTime - startTime;
            
            // Initialize metrics if needed
            if (!performanceMetrics[name]) {
              performanceMetrics[name] = {
                functionName: name,
                callCount: 0,
                totalExecutionTime: 0,
                averageExecutionTime: 0,
                minExecutionTime: Infinity,
                maxExecutionTime: 0,
              };
            }
            
            // Update metrics
            performanceMetrics[name].callCount++;
            performanceMetrics[name].totalExecutionTime += executionTime;
            performanceMetrics[name].averageExecutionTime = 
              performanceMetrics[name].totalExecutionTime / performanceMetrics[name].callCount;
            performanceMetrics[name].minExecutionTime = 
              Math.min(performanceMetrics[name].minExecutionTime, executionTime);
            performanceMetrics[name].maxExecutionTime = 
              Math.max(performanceMetrics[name].maxExecutionTime, executionTime);
            
            throw error;
          }) as ReturnType<T>;
      }
      
      // Handle synchronous results
      const endTime = performance.now();
      const executionTime = endTime - startTime;
      
      // Initialize metrics if needed
      if (!performanceMetrics[name]) {
        performanceMetrics[name] = {
          functionName: name,
          callCount: 0,
          totalExecutionTime: 0,
          averageExecutionTime: 0,
          minExecutionTime: Infinity,
          maxExecutionTime: 0,
        };
      }
      
      // Update metrics
      performanceMetrics[name].callCount++;
      performanceMetrics[name].totalExecutionTime += executionTime;
      performanceMetrics[name].averageExecutionTime = 
        performanceMetrics[name].totalExecutionTime / performanceMetrics[name].callCount;
      performanceMetrics[name].minExecutionTime = 
        Math.min(performanceMetrics[name].minExecutionTime, executionTime);
      performanceMetrics[name].maxExecutionTime = 
        Math.max(performanceMetrics[name].maxExecutionTime, executionTime);
      
      return result;
    } catch (error) {
      const endTime = performance.now();
      const executionTime = endTime - startTime;
      
      // Initialize metrics if needed
      if (!performanceMetrics[name]) {
        performanceMetrics[name] = {
          functionName: name,
          callCount: 0,
          totalExecutionTime: 0,
          averageExecutionTime: 0,
          minExecutionTime: Infinity,
          maxExecutionTime: 0,
        };
      }
      
      // Update metrics
      performanceMetrics[name].callCount++;
      performanceMetrics[name].totalExecutionTime += executionTime;
      performanceMetrics[name].averageExecutionTime = 
        performanceMetrics[name].totalExecutionTime / performanceMetrics[name].callCount;
      performanceMetrics[name].minExecutionTime = 
        Math.min(performanceMetrics[name].minExecutionTime, executionTime);
      performanceMetrics[name].maxExecutionTime = 
        Math.max(performanceMetrics[name].maxExecutionTime, executionTime);
      
      throw error;
    }
  } as T;
}
