/**
 * Performance testing utilities for the AI Trading Agent
 * Provides tools for benchmarking and validating performance optimizations
 */

import { 
  memoize, 
  createBatchProcessor, 
  measureExecutionTime, 
  getPerformanceMetrics, 
  clearPerformanceMetrics 
} from '../api/utils/performanceOptimizations';

/**
 * Test case for performance benchmarking
 */
export interface PerformanceTestCase {
  /** Test name */
  name: string;
  /** Function to test */
  fn: (...args: any[]) => any;
  /** Arguments to pass to the function */
  args: any[];
  /** Number of iterations to run */
  iterations: number;
  /** Optional setup function to run before each test */
  setup?: () => void;
  /** Optional teardown function to run after each test */
  teardown?: () => void;
}

/**
 * Performance test result
 */
export interface PerformanceTestResult {
  /** Test name */
  name: string;
  /** Average execution time in milliseconds */
  averageExecutionTime: number;
  /** Minimum execution time in milliseconds */
  minExecutionTime: number;
  /** Maximum execution time in milliseconds */
  maxExecutionTime: number;
  /** Total execution time in milliseconds */
  totalExecutionTime: number;
  /** Number of iterations run */
  iterations: number;
  /** Operations per second */
  opsPerSecond: number;
  /** Memory usage in bytes (if available) */
  memoryUsage?: number;
  /** Cache hit ratio (if applicable) */
  cacheHitRatio?: number;
}

/**
 * Run a performance benchmark test
 * @param testCase Test case to run
 * @returns Test result
 */
export async function runPerformanceTest(
  testCase: PerformanceTestCase
): Promise<PerformanceTestResult> {
  const { name, fn, args, iterations, setup, teardown } = testCase;
  
  // Initialize result
  const result: PerformanceTestResult = {
    name,
    averageExecutionTime: 0,
    minExecutionTime: Infinity,
    maxExecutionTime: 0,
    totalExecutionTime: 0,
    iterations,
    opsPerSecond: 0,
  };
  
  // Clear existing metrics
  clearPerformanceMetrics();
  
  // Measure function with a unique name for this test
  const testFnName = `${name}_${Date.now()}`;
  const measuredFn = measureExecutionTime(fn, testFnName);
  
  // Run test iterations
  const startTime = performance.now();
  
  for (let i = 0; i < iterations; i++) {
    // Run setup if provided
    if (setup) {
      setup();
    }
    
    // Run the function
    try {
      await measuredFn(...args);
    } catch (error) {
      console.error(`Error in test iteration ${i}:`, error);
    }
    
    // Run teardown if provided
    if (teardown) {
      teardown();
    }
  }
  
  const endTime = performance.now();
  const totalTime = endTime - startTime;
  
  // Get metrics
  const metrics = getPerformanceMetrics()[testFnName];
  
  if (metrics) {
    result.averageExecutionTime = metrics.averageExecutionTime;
    result.minExecutionTime = metrics.minExecutionTime;
    result.maxExecutionTime = metrics.maxExecutionTime;
    result.totalExecutionTime = metrics.totalExecutionTime;
    
    // Calculate operations per second
    result.opsPerSecond = iterations / (totalTime / 1000);
    
    // Add cache metrics if available
    if (metrics.cacheHitRatio !== undefined) {
      result.cacheHitRatio = metrics.cacheHitRatio;
    }
  }
  
  return result;
}

/**
 * Compare performance of two functions
 * @param originalFn Original function
 * @param optimizedFn Optimized function
 * @param args Arguments to pass to both functions
 * @param iterations Number of iterations to run
 * @returns Comparison result with improvement percentage
 */
export async function comparePerformance(
  originalFn: (...args: any[]) => any,
  optimizedFn: (...args: any[]) => any,
  args: any[] = [],
  iterations: number = 100
): Promise<{
  original: PerformanceTestResult;
  optimized: PerformanceTestResult;
  improvement: number;
}> {
  // Run tests
  const originalResult = await runPerformanceTest({
    name: 'original',
    fn: originalFn,
    args,
    iterations,
  });
  
  const optimizedResult = await runPerformanceTest({
    name: 'optimized',
    fn: optimizedFn,
    args,
    iterations,
  });
  
  // Calculate improvement percentage
  const improvement = ((originalResult.averageExecutionTime - optimizedResult.averageExecutionTime) / 
    originalResult.averageExecutionTime) * 100;
  
  return {
    original: originalResult,
    optimized: optimizedResult,
    improvement,
  };
}

/**
 * Test the effectiveness of memoization
 * @param fn Function to test
 * @param args Arguments to pass to the function
 * @param iterations Number of iterations to run
 * @param options Memoization options
 * @returns Test results with cache hit metrics
 */
export async function testMemoization<T extends (...args: any[]) => any>(
  fn: T,
  args: Parameters<T>,
  iterations: number = 100,
  options: {
    maxAgeMs?: number;
    maxCacheSize?: number;
  } = {}
): Promise<{
  withoutMemoization: PerformanceTestResult;
  withMemoization: PerformanceTestResult;
  improvement: number;
  cacheHitRatio: number;
}> {
  // Create memoized version
  const memoizedFn = memoize(fn, options);
  
  // Test without memoization
  const withoutMemoizationResult = await runPerformanceTest({
    name: 'withoutMemoization',
    fn,
    args: args as any[],
    iterations,
  });
  
  // Clear cache before test
  memoizedFn.clearCache();
  
  // Test with memoization
  const withMemoizationResult = await runPerformanceTest({
    name: 'withMemoization',
    fn: memoizedFn,
    args: args as any[],
    iterations,
  });
  
  // Get metrics
  const metrics = getPerformanceMetrics()['withMemoization'];
  const cacheHitRatio = metrics?.cacheHitRatio || 0;
  
  // Calculate improvement
  const improvement = ((withoutMemoizationResult.averageExecutionTime - withMemoizationResult.averageExecutionTime) / 
    withoutMemoizationResult.averageExecutionTime) * 100;
  
  return {
    withoutMemoization: withoutMemoizationResult,
    withMemoization: withMemoizationResult,
    improvement,
    cacheHitRatio,
  };
}

/**
 * Test the effectiveness of batch processing
 * @param singleFn Function that processes a single item
 * @param batchFn Function that processes a batch of items
 * @param items Items to process
 * @param batchSize Batch size to use
 * @returns Test results with batch processing metrics
 */
export async function testBatchProcessing<T, R>(
  singleFn: (item: T) => Promise<R>,
  batchFn: (items: T[]) => Promise<R[]>,
  items: T[],
  batchSize: number = 10
): Promise<{
  withoutBatching: PerformanceTestResult;
  withBatching: PerformanceTestResult;
  improvement: number;
  batchCount: number;
}> {
  // Test without batching
  const startWithoutBatching = performance.now();
  const withoutBatchingResults = await Promise.all(items.map(item => singleFn(item)));
  const endWithoutBatching = performance.now();
  
  // Test with batching
  const startWithBatching = performance.now();
  const batches: T[][] = [];
  
  // Split items into batches
  for (let i = 0; i < items.length; i += batchSize) {
    batches.push(items.slice(i, i + batchSize));
  }
  
  // Process batches
  const batchResults = await Promise.all(batches.map(batch => batchFn(batch)));
  const withBatchingResults = batchResults.flat();
  const endWithBatching = performance.now();
  
  // Calculate metrics
  const withoutBatchingTime = endWithoutBatching - startWithoutBatching;
  const withBatchingTime = endWithBatching - startWithBatching;
  const improvement = ((withoutBatchingTime - withBatchingTime) / withoutBatchingTime) * 100;
  
  return {
    withoutBatching: {
      name: 'withoutBatching',
      averageExecutionTime: withoutBatchingTime / items.length,
      minExecutionTime: 0, // Not measured individually
      maxExecutionTime: 0, // Not measured individually
      totalExecutionTime: withoutBatchingTime,
      iterations: items.length,
      opsPerSecond: (items.length / withoutBatchingTime) * 1000,
    },
    withBatching: {
      name: 'withBatching',
      averageExecutionTime: withBatchingTime / items.length,
      minExecutionTime: 0, // Not measured individually
      maxExecutionTime: 0, // Not measured individually
      totalExecutionTime: withBatchingTime,
      iterations: items.length,
      opsPerSecond: (items.length / withBatchingTime) * 1000,
    },
    improvement,
    batchCount: batches.length,
  };
}

/**
 * Generate a performance report in markdown format
 * @param testResults Test results to include in the report
 * @returns Markdown report
 */
export function generatePerformanceReport(
  testResults: Record<string, PerformanceTestResult | {
    original?: PerformanceTestResult;
    optimized?: PerformanceTestResult;
    improvement?: number;
    cacheHitRatio?: number;
    batchCount?: number;
  }>
): string {
  let report = `# Performance Test Report\n\n`;
  report += `Generated on: ${new Date().toLocaleString()}\n\n`;
  
  // Add summary table
  report += `## Summary\n\n`;
  report += `| Test | Avg. Time | Ops/Sec | Improvement |\n`;
  report += `|------|-----------|---------|-------------|\n`;
  
  for (const [testName, result] of Object.entries(testResults)) {
    if ('original' in result && result.original && 'optimized' in result && result.optimized) {
      // Comparison test
      report += `| ${testName} | ${result.optimized.averageExecutionTime.toFixed(2)}ms | ${result.optimized.opsPerSecond.toFixed(2)} | ${result.improvement?.toFixed(2)}% |\n`;
    } else if ('improvement' in result) {
      // Special test with improvement
      const baseResult = 'withMemoization' in result 
        ? (result as any).withMemoization 
        : 'withBatching' in result 
          ? (result as any).withBatching 
          : null;
          
      if (baseResult) {
        report += `| ${testName} | ${baseResult.averageExecutionTime.toFixed(2)}ms | ${baseResult.opsPerSecond.toFixed(2)} | ${result.improvement?.toFixed(2)}% |\n`;
      }
    } else {
      // Single test
      const singleResult = result as PerformanceTestResult;
      report += `| ${testName} | ${singleResult.averageExecutionTime.toFixed(2)}ms | ${singleResult.opsPerSecond.toFixed(2)} | - |\n`;
    }
  }
  
  // Add detailed results
  report += `\n## Detailed Results\n\n`;
  
  for (const [testName, result] of Object.entries(testResults)) {
    report += `### ${testName}\n\n`;
    
    if ('original' in result && result.original && 'optimized' in result && result.optimized) {
      // Comparison test
      report += `#### Original vs Optimized\n\n`;
      report += `| Metric | Original | Optimized | Difference |\n`;
      report += `|--------|----------|-----------|------------|\n`;
      report += `| Avg. Time | ${result.original.averageExecutionTime.toFixed(2)}ms | ${result.optimized.averageExecutionTime.toFixed(2)}ms | ${(result.original.averageExecutionTime - result.optimized.averageExecutionTime).toFixed(2)}ms |\n`;
      report += `| Min Time | ${result.original.minExecutionTime.toFixed(2)}ms | ${result.optimized.minExecutionTime.toFixed(2)}ms | ${(result.original.minExecutionTime - result.optimized.minExecutionTime).toFixed(2)}ms |\n`;
      report += `| Max Time | ${result.original.maxExecutionTime.toFixed(2)}ms | ${result.optimized.maxExecutionTime.toFixed(2)}ms | ${(result.original.maxExecutionTime - result.optimized.maxExecutionTime).toFixed(2)}ms |\n`;
      report += `| Total Time | ${result.original.totalExecutionTime.toFixed(2)}ms | ${result.optimized.totalExecutionTime.toFixed(2)}ms | ${(result.original.totalExecutionTime - result.optimized.totalExecutionTime).toFixed(2)}ms |\n`;
      report += `| Ops/Sec | ${result.original.opsPerSecond.toFixed(2)} | ${result.optimized.opsPerSecond.toFixed(2)} | ${(result.optimized.opsPerSecond - result.original.opsPerSecond).toFixed(2)} |\n`;
      report += `| Improvement | - | - | ${result.improvement?.toFixed(2)}% |\n`;
    } else if ('withMemoization' in result) {
      // Memoization test
      const memoTest = result as any;
      report += `#### Memoization Test\n\n`;
      report += `| Metric | Without Memoization | With Memoization | Difference |\n`;
      report += `|--------|---------------------|------------------|------------|\n`;
      report += `| Avg. Time | ${memoTest.withoutMemoization.averageExecutionTime.toFixed(2)}ms | ${memoTest.withMemoization.averageExecutionTime.toFixed(2)}ms | ${(memoTest.withoutMemoization.averageExecutionTime - memoTest.withMemoization.averageExecutionTime).toFixed(2)}ms |\n`;
      report += `| Total Time | ${memoTest.withoutMemoization.totalExecutionTime.toFixed(2)}ms | ${memoTest.withMemoization.totalExecutionTime.toFixed(2)}ms | ${(memoTest.withoutMemoization.totalExecutionTime - memoTest.withMemoization.totalExecutionTime).toFixed(2)}ms |\n`;
      report += `| Ops/Sec | ${memoTest.withoutMemoization.opsPerSecond.toFixed(2)} | ${memoTest.withMemoization.opsPerSecond.toFixed(2)} | ${(memoTest.withMemoization.opsPerSecond - memoTest.withoutMemoization.opsPerSecond).toFixed(2)} |\n`;
      report += `| Cache Hit Ratio | - | ${(memoTest.cacheHitRatio * 100).toFixed(2)}% | - |\n`;
      report += `| Improvement | - | - | ${memoTest.improvement.toFixed(2)}% |\n`;
    } else if ('withBatching' in result) {
      // Batch processing test
      const batchTest = result as any;
      report += `#### Batch Processing Test\n\n`;
      report += `| Metric | Without Batching | With Batching | Difference |\n`;
      report += `|--------|------------------|---------------|------------|\n`;
      report += `| Avg. Time | ${batchTest.withoutBatching.averageExecutionTime.toFixed(2)}ms | ${batchTest.withBatching.averageExecutionTime.toFixed(2)}ms | ${(batchTest.withoutBatching.averageExecutionTime - batchTest.withBatching.averageExecutionTime).toFixed(2)}ms |\n`;
      report += `| Total Time | ${batchTest.withoutBatching.totalExecutionTime.toFixed(2)}ms | ${batchTest.withBatching.totalExecutionTime.toFixed(2)}ms | ${(batchTest.withoutBatching.totalExecutionTime - batchTest.withBatching.totalExecutionTime).toFixed(2)}ms |\n`;
      report += `| Ops/Sec | ${batchTest.withoutBatching.opsPerSecond.toFixed(2)} | ${batchTest.withBatching.opsPerSecond.toFixed(2)} | ${(batchTest.withBatching.opsPerSecond - batchTest.withoutBatching.opsPerSecond).toFixed(2)} |\n`;
      report += `| Batch Count | - | ${batchTest.batchCount} | - |\n`;
      report += `| Improvement | - | - | ${batchTest.improvement.toFixed(2)}% |\n`;
    } else {
      // Single test
      const singleResult = result as PerformanceTestResult;
      report += `#### Test Metrics\n\n`;
      report += `| Metric | Value |\n`;
      report += `|--------|-------|\n`;
      report += `| Avg. Time | ${singleResult.averageExecutionTime.toFixed(2)}ms |\n`;
      report += `| Min Time | ${singleResult.minExecutionTime.toFixed(2)}ms |\n`;
      report += `| Max Time | ${singleResult.maxExecutionTime.toFixed(2)}ms |\n`;
      report += `| Total Time | ${singleResult.totalExecutionTime.toFixed(2)}ms |\n`;
      report += `| Iterations | ${singleResult.iterations} |\n`;
      report += `| Ops/Sec | ${singleResult.opsPerSecond.toFixed(2)} |\n`;
      
      if (singleResult.cacheHitRatio !== undefined) {
        report += `| Cache Hit Ratio | ${(singleResult.cacheHitRatio * 100).toFixed(2)}% |\n`;
      }
      
      if (singleResult.memoryUsage !== undefined) {
        report += `| Memory Usage | ${(singleResult.memoryUsage / 1024).toFixed(2)} KB |\n`;
      }
    }
    
    report += `\n`;
  }
  
  return report;
}

/**
 * Run a standard performance test suite on common operations
 * @returns Test results
 */
export async function runStandardPerformanceTestSuite(): Promise<Record<string, any>> {
  const results: Record<string, any> = {};
  
  // Test 1: Simple function memoization
  const fibonacci = (n: number): number => {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
  };
  
  results.fibonacciMemoization = await testMemoization(
    fibonacci,
    [20],
    100
  );
  
  // Test 2: Async function memoization
  const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
  
  const asyncOperation = async (input: string): Promise<string> => {
    await delay(5); // Small delay to simulate API call
    return `Processed: ${input}`;
  };
  
  results.asyncMemoization = await testMemoization(
    asyncOperation,
    ['test-input'],
    50
  );
  
  // Test 3: Batch processing
  const singleProcessor = async (item: number): Promise<number> => {
    await delay(5); // Small delay to simulate API call
    return item * 2;
  };
  
  const batchProcessor = async (items: number[]): Promise<number[]> => {
    await delay(10); // Small delay to simulate API call
    return items.map(item => item * 2);
  };
  
  const testItems = Array.from({ length: 50 }, (_, i) => i);
  
  results.batchProcessing = await testBatchProcessing(
    singleProcessor,
    batchProcessor,
    testItems,
    10
  );
  
  return results;
}
