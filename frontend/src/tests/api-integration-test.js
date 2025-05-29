/**
 * API Integration Test for Sentiment Analytics
 * This test verifies that our API integration is working properly
 */

// Use ES modules
import sentimentAnalyticsService from '../api/sentimentAnalyticsService.js';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Test configuration
const TEST_AGENT_ID = 'test-agent-123';
const TEST_SYMBOL = 'BTC';
const TEST_TIMEFRAME = '30d';

// Helper to display test results
const displayResult = (testName, success, data = null, error = null) => {
  console.log(`\n=== ${testName} ===`);
  console.log(`Status: ${success ? '✅ SUCCESS' : '❌ FAILED'}`);
  
  if (data) {
    console.log('Data Preview:');
    console.log(JSON.stringify(data, null, 2).substring(0, 300) + '...');
  }
  
  if (error) {
    console.log('Error:');
    console.log(error.message || error);
  }
};

// Test functions
async function testHistoricalSentimentData() {
  try {
    console.log(`\nFetching historical sentiment data for ${TEST_SYMBOL}...`);
    const data = await sentimentAnalyticsService.getHistoricalSentimentData(
      TEST_AGENT_ID, 
      TEST_SYMBOL, 
      TEST_TIMEFRAME
    );
    
    const success = data && data.dataPoints && data.dataPoints.length > 0;
    displayResult('Historical Sentiment Data', success, data);
    return { success, data };
  } catch (error) {
    displayResult('Historical Sentiment Data', false, null, error);
    return { success: false, error };
  }
}

async function testAllSymbolsSentimentData() {
  try {
    console.log(`\nFetching sentiment data for all symbols...`);
    const data = await sentimentAnalyticsService.getAllSymbolsSentimentData(
      TEST_AGENT_ID, 
      TEST_TIMEFRAME
    );
    
    const success = data && Object.keys(data).length > 0;
    displayResult('All Symbols Sentiment Data', success, data);
    return { success, data };
  } catch (error) {
    displayResult('All Symbols Sentiment Data', false, null, error);
    return { success: false, error };
  }
}

async function testMonitoredSymbolsWithSentiment() {
  try {
    console.log(`\nFetching monitored symbols with sentiment...`);
    const data = await sentimentAnalyticsService.getMonitoredSymbolsWithSentiment(
      TEST_AGENT_ID
    );
    
    const success = data && data.length > 0;
    displayResult('Monitored Symbols With Sentiment', success, data);
    return { success, data };
  } catch (error) {
    displayResult('Monitored Symbols With Sentiment', false, null, error);
    return { success: false, error };
  }
}

async function testSignalQualityMetrics() {
  try {
    console.log(`\nFetching signal quality metrics...`);
    const data = await sentimentAnalyticsService.getSignalQualityMetrics(
      TEST_AGENT_ID, 
      TEST_TIMEFRAME
    );
    
    const success = data && data.overall_accuracy !== undefined;
    displayResult('Signal Quality Metrics', success, data);
    return { success, data };
  } catch (error) {
    displayResult('Signal Quality Metrics', false, null, error);
    return { success: false, error };
  }
}

async function testCachingMechanism() {
  try {
    console.log(`\nTesting caching mechanism...`);
    console.log('First request (should hit API):');
    const startTime1 = Date.now();
    await sentimentAnalyticsService.getHistoricalSentimentData(
      TEST_AGENT_ID, 
      TEST_SYMBOL, 
      TEST_TIMEFRAME
    );
    const endTime1 = Date.now();
    const firstRequestTime = endTime1 - startTime1;
    
    console.log('Second request (should use cache):');
    const startTime2 = Date.now();
    await sentimentAnalyticsService.getHistoricalSentimentData(
      TEST_AGENT_ID, 
      TEST_SYMBOL, 
      TEST_TIMEFRAME
    );
    const endTime2 = Date.now();
    const secondRequestTime = endTime2 - startTime2;
    
    const cachingWorks = secondRequestTime < firstRequestTime;
    
    displayResult(
      'Caching Mechanism', 
      cachingWorks, 
      {
        firstRequestTime: `${firstRequestTime}ms`,
        secondRequestTime: `${secondRequestTime}ms`,
        improvement: `${Math.round((1 - secondRequestTime / firstRequestTime) * 100)}%`
      }
    );
    
    return { success: cachingWorks };
  } catch (error) {
    displayResult('Caching Mechanism', false, null, error);
    return { success: false, error };
  }
}

// Run all tests
async function runAllTests() {
  console.log('=== SENTIMENT ANALYTICS API INTEGRATION TEST ===');
  console.log(`Test Agent ID: ${TEST_AGENT_ID}`);
  console.log(`Test Symbol: ${TEST_SYMBOL}`);
  console.log(`Test Timeframe: ${TEST_TIMEFRAME}`);
  
  // Run tests sequentially
  const results = {
    historicalData: await testHistoricalSentimentData(),
    allSymbolsData: await testAllSymbolsSentimentData(),
    monitoredSymbols: await testMonitoredSymbolsWithSentiment(),
    signalQualityMetrics: await testSignalQualityMetrics(),
    caching: await testCachingMechanism()
  };
  
  // Overall results
  const totalTests = Object.keys(results).length;
  const passedTests = Object.values(results).filter(r => r.success).length;
  
  console.log('\n=== TEST SUMMARY ===');
  console.log(`${passedTests} of ${totalTests} tests passed`);
  
  if (passedTests === totalTests) {
    console.log('🎉 All tests passed! API integration is working properly.');
  } else {
    console.log('⚠️ Some tests failed. Check error messages above.');
    console.log('\nPossible issues:');
    console.log('1. Backend API is not running');
    console.log('2. API endpoints are not implemented yet');
    console.log('3. API endpoint URLs are incorrect');
    console.log('4. Network connectivity issues');
    console.log('5. Authentication problems');
  }
}

// Execute tests
runAllTests();
