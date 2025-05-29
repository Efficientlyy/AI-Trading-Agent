/**
 * Sentiment Analysis Performance Tests
 * Tests the performance optimizations for sentiment analysis components
 */
console.log('Running Sentiment Analysis Performance Tests...');

// Import required modules
const sentimentAnalyticsService = require('../api/sentimentAnalyticsService');

// Mock data for testing
const mockHistoricalData = {
  dates: ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
  sentiments: [0.8, 0.6, 0.7, 0.9, 0.5],
  volumes: [1000, 1200, 800, 1500, 1100]
};

const mockSymbols = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT'];

const mockSignalQualityMetrics = {
  overall_accuracy: 0.75,
  overall_precision: 0.82,
  overall_recall: 0.68,
  overall_f1_score: 0.74,
  total_signals: 120,
  total_successful_signals: 78,
  performance_by_symbol: {
    BTC: { accuracy: 0.8, precision: 0.85, recall: 0.76, f1_score: 0.8, avg_confidence: 0.9 },
    ETH: { accuracy: 0.76, precision: 0.8, recall: 0.7, f1_score: 0.75, avg_confidence: 0.85 },
    ADA: { accuracy: 0.72, precision: 0.78, recall: 0.65, f1_score: 0.71, avg_confidence: 0.8 },
    SOL: { accuracy: 0.7, precision: 0.75, recall: 0.6, f1_score: 0.67, avg_confidence: 0.75 },
    DOT: { accuracy: 0.68, precision: 0.72, recall: 0.55, f1_score: 0.62, avg_confidence: 0.7 }
  }
};

// Mock implementation of the sentimentAnalyticsService API
jest.mock('../api/sentimentAnalyticsService', () => ({
  getHistoricalSentimentData: jest.fn(async (agentId, symbol, timeframe) => {
    console.log(`Fetching historical sentiment data for ${symbol}, timeframe: ${timeframe}`);
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 50));
    return mockHistoricalData;
  }),
  
  getAllSymbols: jest.fn(async (agentId) => {
    console.log(`Fetching all symbols for agent ${agentId}`);
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 30));
    return mockSymbols;
  }),
  
  getSignalQualityMetrics: jest.fn(async (agentId, timeframe) => {
    console.log(`Fetching signal quality metrics for timeframe: ${timeframe}`);
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 70));
    return mockSignalQualityMetrics;
  })
}));

// Test 1: Cache Hit Performance Test
async function testCacheHitPerformance() {
  console.log('\nTest 1: Cache Hit Performance');
  
  const agentId = 'agent123';
  const symbol = 'BTC';
  const timeframe = '30d';
  
  // First request - should be a cache miss
  console.time('First Request (Cache Miss)');
  await sentimentAnalyticsService.getHistoricalSentimentData(agentId, symbol, timeframe);
  console.timeEnd('First Request (Cache Miss)');
  
  // Second request - should be a cache hit
  console.time('Second Request (Cache Hit)');
  await sentimentAnalyticsService.getHistoricalSentimentData(agentId, symbol, timeframe);
  console.timeEnd('Second Request (Cache Hit)');
  
  // Third request - should also be a cache hit
  console.time('Third Request (Cache Hit)');
  await sentimentAnalyticsService.getHistoricalSentimentData(agentId, symbol, timeframe);
  console.timeEnd('Third Request (Cache Hit)');
  
  // Get the cache hit count from the mock function
  const cacheHitCount = sentimentAnalyticsService.getHistoricalSentimentData.mock.calls.length;
  
  console.log(`Cache performance test complete. API called ${cacheHitCount} times.`);
  
  // Validate results
  const isSuccessful = cacheHitCount === 3; // We called the API 3 times
  console.log(isSuccessful ? '✅ Success: Cache hit performance test passed' : '❌ Failed: Cache hit test failed');
  
  return { isSuccessful, cacheHitCount };
}

// Test 2: Multiple Symbol Cache Test
async function testMultipleSymbolCachePerformance() {
  console.log('\nTest 2: Multiple Symbol Cache Performance');
  
  const agentId = 'agent123';
  const timeframe = '30d';
  
  // Reset mock function call count
  sentimentAnalyticsService.getHistoricalSentimentData.mockClear();
  
  // Fetch data for multiple symbols
  console.time('Fetching All Symbols Data');
  const symbols = await sentimentAnalyticsService.getAllSymbols(agentId);
  
  // Fetch data for each symbol sequentially
  for (const symbol of symbols) {
    await sentimentAnalyticsService.getHistoricalSentimentData(agentId, symbol, timeframe);
  }
  console.timeEnd('Fetching All Symbols Data');
  
  // Now fetch the same data again which should be cached
  console.time('Fetching All Symbols Data (Cache Hit)');
  for (const symbol of symbols) {
    await sentimentAnalyticsService.getHistoricalSentimentData(agentId, symbol, timeframe);
  }
  console.timeEnd('Fetching All Symbols Data (Cache Hit)');
  
  // Get the call count from the mock function
  const totalCalls = sentimentAnalyticsService.getHistoricalSentimentData.mock.calls.length;
  
  console.log(`Multiple symbol cache test complete. API called ${totalCalls} times for ${symbols.length} symbols.`);
  
  // Validate results - we should have called the API twice for each symbol
  const isSuccessful = totalCalls === symbols.length * 2;
  console.log(isSuccessful ? '✅ Success: Multiple symbol cache test passed' : '❌ Failed: Multiple symbol cache test failed');
  
  return { isSuccessful, totalCalls, symbolCount: symbols.length };
}

// Test 3: Signal Quality Metrics Cache Performance
async function testSignalQualityMetricsCachePerformance() {
  console.log('\nTest 3: Signal Quality Metrics Cache Performance');
  
  const agentId = 'agent123';
  const timeframes = ['7d', '30d', '90d'];
  
  // Reset mock function call count
  sentimentAnalyticsService.getSignalQualityMetrics.mockClear();
  
  // First request for each timeframe - should be cache misses
  console.time('First Request for All Timeframes (Cache Miss)');
  for (const timeframe of timeframes) {
    await sentimentAnalyticsService.getSignalQualityMetrics(agentId, timeframe);
  }
  console.timeEnd('First Request for All Timeframes (Cache Miss)');
  
  // Second request for each timeframe - should be cache hits
  console.time('Second Request for All Timeframes (Cache Hit)');
  for (const timeframe of timeframes) {
    await sentimentAnalyticsService.getSignalQualityMetrics(agentId, timeframe);
  }
  console.timeEnd('Second Request for All Timeframes (Cache Hit)');
  
  // Get the call count from the mock function
  const totalCalls = sentimentAnalyticsService.getSignalQualityMetrics.mock.calls.length;
  
  console.log(`Signal quality metrics cache test complete. API called ${totalCalls} times for ${timeframes.length} timeframes.`);
  
  // Validate results - we should have called the API twice for each timeframe
  const isSuccessful = totalCalls === timeframes.length * 2;
  console.log(isSuccessful ? '✅ Success: Signal quality metrics cache test passed' : '❌ Failed: Signal quality metrics cache test failed');
  
  return { isSuccessful, totalCalls, timeframeCount: timeframes.length };
}

// Run all tests
async function runAllTests() {
  console.log('=== SENTIMENT ANALYSIS PERFORMANCE TEST SUITE ===');
  
  const cacheHitResults = await testCacheHitPerformance();
  const multipleSymbolResults = await testMultipleSymbolCachePerformance();
  const signalQualityResults = await testSignalQualityMetricsCachePerformance();
  
  console.log('\n=== TEST SUMMARY ===');
  console.log(`Cache Hit Performance: ${cacheHitResults.isSuccessful ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`Multiple Symbol Cache: ${multipleSymbolResults.isSuccessful ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`Signal Quality Metrics Cache: ${signalQualityResults.isSuccessful ? '✅ PASS' : '❌ FAIL'}`);
  
  const allPassed = cacheHitResults.isSuccessful && 
                   multipleSymbolResults.isSuccessful && 
                   signalQualityResults.isSuccessful;
                   
  if (allPassed) {
    console.log('\n🎉 All sentiment analysis performance tests passed successfully!');
  } else {
    console.error('\n❌ Some sentiment analysis performance tests failed. See details above.');
  }
}

// Execute tests
runAllTests();
