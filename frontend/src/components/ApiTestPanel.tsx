import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Typography, 
  Paper, 
  Divider, 
  CircularProgress,
  Alert,
  Stack
} from '@mui/material';
import sentimentAnalyticsService from '../api/sentimentAnalyticsService';

interface TestResult {
  name: string;
  status: 'success' | 'error' | 'pending';
  data?: any;
  error?: string;
  executionTime?: number;
}

const ApiTestPanel: React.FC = () => {
  const [results, setResults] = useState<TestResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  
  // Test configuration
  const agentId = 'test-agent-123';
  const symbol = 'BTC';
  const timeframe = '30d' as const; // Type assertion for timeframe
  
  const runTest = async (name: string, testFn: () => Promise<any>) => {
    // Add pending test to results
    setResults(prev => [...prev, { name, status: 'pending' }]);
    
    try {
      const startTime = performance.now();
      const data = await testFn();
      const executionTime = Math.round(performance.now() - startTime);
      
      // Update test result with success
      setResults(prev => prev.map(result => 
        result.name === name 
          ? { name, status: 'success', data, executionTime } 
          : result
      ));
      
      return { success: true, data };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      
      // Update test result with error
      setResults(prev => prev.map(result => 
        result.name === name 
          ? { name, status: 'error', error: errorMessage } 
          : result
      ));
      
      return { success: false, error };
    }
  };
  
  const runAllTests = async () => {
    setIsRunning(true);
    setResults([]);
    
    // Test Historical Sentiment Data
    await runTest('Historical Sentiment Data', async () => {
      return await sentimentAnalyticsService.getHistoricalSentimentData(agentId, symbol, timeframe);
    });
    
    // Test All Symbols Sentiment Data
    await runTest('All Symbols Sentiment Data', async () => {
      return await sentimentAnalyticsService.getAllSymbolsSentimentData(agentId, timeframe);
    });
    
    // Test Monitored Symbols
    await runTest('Monitored Symbols', async () => {
      return await sentimentAnalyticsService.getMonitoredSymbolsWithSentiment(agentId);
    });
    
    // Test Signal Quality Metrics
    await runTest('Signal Quality Metrics', async () => {
      return await sentimentAnalyticsService.getSignalQualityMetrics(agentId, timeframe);
    });
    
    // Test Caching
    await runTest('Caching Test', async () => {
      // First call (should hit API)
      const startTime1 = performance.now();
      await sentimentAnalyticsService.getHistoricalSentimentData(agentId, symbol, timeframe);
      const firstCallTime = performance.now() - startTime1;
      
      // Second call (should use cache)
      const startTime2 = performance.now();
      await sentimentAnalyticsService.getHistoricalSentimentData(agentId, symbol, timeframe);
      const secondCallTime = performance.now() - startTime2;
      
      const cachingWorks = secondCallTime < firstCallTime;
      const improvement = Math.round((1 - (secondCallTime / firstCallTime)) * 100);
      
      return {
        cachingWorks,
        firstCallTime: `${Math.round(firstCallTime)}ms`,
        secondCallTime: `${Math.round(secondCallTime)}ms`,
        improvement: `${improvement}%`
      };
    });
    
    setIsRunning(false);
  };
  
  const renderTestResultContent = (result: TestResult) => {
    if (result.status === 'pending') {
      return <CircularProgress size={20} />;
    }
    
    if (result.status === 'error') {
      return (
        <Alert severity="error" sx={{ mt: 1 }}>
          {result.error}
        </Alert>
      );
    }
    
    return (
      <Box>
        <Alert severity="success" sx={{ mt: 1 }}>
          Success! Execution time: {result.executionTime}ms
        </Alert>
        <Box 
          sx={{ 
            mt: 1, 
            p: 1, 
            backgroundColor: 'rgba(0,0,0,0.04)', 
            borderRadius: 1,
            maxHeight: 150,
            overflow: 'auto',
            fontSize: '0.8rem'
          }}
        >
          <pre>{JSON.stringify(result.data, null, 2).substring(0, 500)}{result.data && JSON.stringify(result.data).length > 500 ? '...' : ''}</pre>
        </Box>
      </Box>
    );
  };
  
  return (
    <Paper sx={{ p: 3, mt: 3, maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h5" gutterBottom>
        API Integration Test
      </Typography>
      
      <Typography variant="body2" color="text.secondary" paragraph>
        This panel tests the integration between the frontend and backend API endpoints for sentiment analysis.
        It verifies that all API calls are working correctly and that client-side caching is functioning properly.
      </Typography>
      
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2">Test Configuration:</Typography>
        <Typography variant="body2">Agent ID: {agentId}</Typography>
        <Typography variant="body2">Symbol: {symbol}</Typography>
        <Typography variant="body2">Timeframe: {timeframe}</Typography>
      </Box>
      
      <Button 
        variant="contained" 
        onClick={runAllTests} 
        disabled={isRunning}
        sx={{ mb: 3 }}
      >
        {isRunning ? 'Running Tests...' : 'Run All Tests'}
      </Button>
      
      <Stack spacing={2}>
        {results.map((result, index) => (
          <Box key={result.name}>
            {index > 0 && <Divider sx={{ my: 2 }} />}
            <Typography variant="subtitle1" fontWeight="bold">
              {result.name}
              {result.status === 'success' && ' ✓'}
              {result.status === 'error' && ' ✗'}
            </Typography>
            {renderTestResultContent(result)}
          </Box>
        ))}
      </Stack>
      
      {results.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Divider sx={{ mb: 2 }} />
          <Typography variant="subtitle1">
            Summary: {results.filter(r => r.status === 'success').length} of {results.length} tests passed
          </Typography>
          
          {results.some(r => r.status === 'error') && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              <Typography variant="subtitle2">Troubleshooting Tips:</Typography>
              <ul style={{ marginTop: 5, paddingLeft: 20 }}>
                <li>Check if the backend API server is running</li>
                <li>Verify that the API endpoints are implemented correctly</li>
                <li>Check network connectivity and CORS settings</li>
                <li>Verify authentication if required by the API</li>
              </ul>
            </Alert>
          )}
        </Box>
      )}
    </Paper>
  );
};

export default ApiTestPanel;
