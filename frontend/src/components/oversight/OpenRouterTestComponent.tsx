import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Divider,
  Paper,
  Typography,
  Alert,
} from '@mui/material';
import { CheckCircle, Error as ErrorIcon } from '@mui/icons-material';
import { getOpenRouterCredentials, verifyOpenRouterCredentials } from '../../api/openRouterClient';
import { useLLMOversight } from '../../context/LLMOversightContext';
import { oversightClient } from '../../api/oversightClient';

/**
 * Component for testing OpenRouter connection and LLM Oversight functionality
 */
const OpenRouterTestComponent: React.FC = () => {
  const { status, checkConnection, runHealthCheck } = useLLMOversight();
  const [testing, setTesting] = useState(false);
  const [testResults, setTestResults] = useState<{
    credentialsFound: boolean;
    credentialsValid: boolean;
    modelAvailable: boolean;
    sampleAnalysisSuccess: boolean;
    error?: string;
  } | null>(null);

  const runConnectionTest = async () => {
    setTesting(true);
    setTestResults(null);
    
    try {
      // Step 1: Check if credentials exist
      const credentials = getOpenRouterCredentials();
      const credentialsFound = !!credentials;
      
      let credentialsValid = false;
      let modelAvailable = false;
      let sampleAnalysisSuccess = false;
      let error = '';
      
      // Step 2: Verify credentials with OpenRouter
      if (credentialsFound) {
        try {
          const verifyResult = await verifyOpenRouterCredentials();
          credentialsValid = verifyResult.valid;
          modelAvailable = verifyResult.modelAvailable;
        } catch (e: any) {
          credentialsValid = false;
          modelAvailable = false;
          error = e.message || 'Failed to verify credentials with OpenRouter';
        }
      }
      
      // Step 3: Try a sample analysis
      if (credentialsFound && credentialsValid && modelAvailable) {
        try {
          // Create a sample trading decision for testing
          const sampleDecision = {
            symbol: 'AAPL',
            action: 'BUY',
            price: 175.50,
            quantity: 10,
            strategy: 'Moving Average Crossover',
            reasoning: 'Short-term MA crossed above long-term MA, signaling potential uptrend'
          };
          
          const analysisResult = await oversightClient.analyzeTradingDecision(sampleDecision);
          sampleAnalysisSuccess = !!analysisResult;
          
          // If we reached here, update the oversight context connection status
          await checkConnection();
          await runHealthCheck();
        } catch (e: any) {
          sampleAnalysisSuccess = false;
          error = e.message || 'Failed to run sample analysis';
        }
      }
      
      // Set final test results
      setTestResults({
        credentialsFound,
        credentialsValid,
        modelAvailable,
        sampleAnalysisSuccess,
        error: error || undefined
      });
    } catch (e: any) {
      setTestResults({
        credentialsFound: false,
        credentialsValid: false,
        modelAvailable: false,
        sampleAnalysisSuccess: false,
        error: e.message || 'Unknown error occurred during testing'
      });
    } finally {
      setTesting(false);
    }
  };

  const getStatusColor = (isSuccess: boolean) => {
    return isSuccess ? '#4caf50' : '#f44336';
  };

  return (
    <Card sx={{ mb: 3, bgcolor: '#1e2030', color: '#e0e0e0', borderRadius: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          OpenRouter LLM Integration Test
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
            Current Connection Status
          </Typography>
          <Paper sx={{ p: 2, bgcolor: '#2c3144' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Box 
                sx={{ 
                  width: 12, 
                  height: 12, 
                  borderRadius: '50%', 
                  bgcolor: status.isConnected ? '#4caf50' : '#f44336',
                  mr: 1 
                }} 
              />
              <Typography>
                {status.isConnected ? 'Connected' : 'Disconnected'}
              </Typography>
            </Box>
            
            <Typography variant="body2" color="textSecondary">
              Provider: {status.provider}
            </Typography>
            
            <Typography variant="body2" color="textSecondary">
              Model: {status.model}
            </Typography>
            
            <Typography variant="body2" color="textSecondary">
              Health: {status.health}
            </Typography>
            
            {status.lastChecked && (
              <Typography variant="body2" color="textSecondary">
                Last Checked: {new Date(status.lastChecked).toLocaleString()}
              </Typography>
            )}
          </Paper>
        </Box>
        
        <Button 
          variant="contained" 
          color="primary" 
          onClick={runConnectionTest}
          disabled={testing}
          startIcon={testing ? <CircularProgress size={20} /> : undefined}
          sx={{ mb: 3 }}
        >
          {testing ? 'Testing Connection...' : 'Test OpenRouter Connection'}
        </Button>
        
        {testResults && (
          <Box>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="subtitle1" gutterBottom>
              Test Results
            </Typography>
            
            <Box sx={{ mb: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                {testResults.credentialsFound ? 
                  <CheckCircle fontSize="small" sx={{ color: getStatusColor(true), mr: 1 }} /> : 
                  <ErrorIcon fontSize="small" sx={{ color: getStatusColor(false), mr: 1 }} />
                }
                <Typography>
                  API Credentials: {testResults.credentialsFound ? 'Found' : 'Not Found'}
                </Typography>
              </Box>
            </Box>
            
            {testResults.credentialsFound && (
              <>
                <Box sx={{ mb: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {testResults.credentialsValid ? 
                      <CheckCircle fontSize="small" sx={{ color: getStatusColor(true), mr: 1 }} /> : 
                      <ErrorIcon fontSize="small" sx={{ color: getStatusColor(false), mr: 1 }} />
                    }
                    <Typography>
                      Credentials Validation: {testResults.credentialsValid ? 'Valid' : 'Invalid'}
                    </Typography>
                  </Box>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {testResults.modelAvailable ? 
                      <CheckCircle fontSize="small" sx={{ color: getStatusColor(true), mr: 1 }} /> : 
                      <ErrorIcon fontSize="small" sx={{ color: getStatusColor(false), mr: 1 }} />
                    }
                    <Typography>
                      Model Availability: {testResults.modelAvailable ? 'Available' : 'Unavailable'}
                    </Typography>
                  </Box>
                </Box>
                
                {testResults.credentialsValid && testResults.modelAvailable && (
                  <Box sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {testResults.sampleAnalysisSuccess ? 
                        <CheckCircle fontSize="small" sx={{ color: getStatusColor(true), mr: 1 }} /> : 
                        <ErrorIcon fontSize="small" sx={{ color: getStatusColor(false), mr: 1 }} />
                      }
                      <Typography>
                        Sample Analysis: {testResults.sampleAnalysisSuccess ? 'Successful' : 'Failed'}
                      </Typography>
                    </Box>
                  </Box>
                )}
              </>
            )}
            
            {testResults.error && (
              <Alert severity="error" sx={{ mt: 2, bgcolor: 'rgba(244, 67, 54, 0.1)' }}>
                {testResults.error}
              </Alert>
            )}
            
            {testResults.credentialsFound && 
             testResults.credentialsValid && 
             testResults.modelAvailable && 
             testResults.sampleAnalysisSuccess && (
              <Alert severity="success" sx={{ mt: 2, bgcolor: 'rgba(76, 175, 80, 0.1)' }}>
                Connection test successful! OpenRouter LLM integration is working properly.
              </Alert>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default OpenRouterTestComponent;
