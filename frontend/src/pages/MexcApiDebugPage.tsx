import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, CircularProgress, Button, Alert, Divider, Card, CardContent } from '@mui/material';
import { Link } from 'react-router-dom';
import axios from 'axios';

// Very simple debugging page for MEXC API
const MexcApiDebugPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<any>(null);
  const [cors, setCors] = useState<boolean>(true);

  // Test the MEXC API directly without any processing
  const testDirectApi = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);
    
    try {
      console.log('Testing direct MEXC API connection...');
      const url = 'https://api.mexc.com/api/v3/ticker/24hr?symbol=BTCUSDT';
      
      const response = await axios.get(url, {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });
      
      console.log('MEXC API Response:', response.data);
      setResponse(response.data);
      setError(null);
    } catch (err) {
      console.error('API Error:', err);
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Error: ${errorMessage}`);
      setResponse(null);
    } finally {
      setLoading(false);
    }
  };

  // Test API with CORS proxy
  const testWithCorsProxy = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);
    setCors(true);
    
    try {
      console.log('Testing MEXC API with CORS proxy...');
      // Using a public CORS proxy for testing
      const proxyUrl = 'https://cors-anywhere.herokuapp.com/';
      const targetUrl = 'https://api.mexc.com/api/v3/ticker/24hr?symbol=BTCUSDT';
      
      const response = await axios.get(proxyUrl + targetUrl, {
        headers: {
          'X-Requested-With': 'XMLHttpRequest',
          'Origin': window.location.origin
        }
      });
      
      console.log('MEXC API Response via proxy:', response.data);
      setResponse(response.data);
      setError(null);
    } catch (err) {
      console.error('API Error with proxy:', err);
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Error with proxy: ${errorMessage}`);
      setResponse(null);
    } finally {
      setLoading(false);
    }
  };

  // Check a simple public API to validate internet connectivity
  const testPublicApi = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);
    setCors(false);
    
    try {
      console.log('Testing public API...');
      const response = await axios.get('https://jsonplaceholder.typicode.com/todos/1');
      console.log('Public API Response:', response.data);
      setResponse(response.data);
      setError(null);
    } catch (err) {
      console.error('Public API Error:', err);
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(`Error with public API: ${errorMessage}`);
      setResponse(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: '800px', mx: 'auto' }}>
      <Typography variant="h4" gutterBottom>
        MEXC API Debug
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        This page helps diagnose connectivity issues with the MEXC API. Check the browser console for detailed logs.
      </Alert>
      
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Button 
          variant="contained" 
          onClick={testDirectApi} 
          disabled={loading}
        >
          Test Direct API
        </Button>
        
        <Button 
          variant="contained" 
          onClick={testWithCorsProxy} 
          disabled={loading}
          color="secondary"
        >
          Test With CORS Proxy
        </Button>
        
        <Button 
          variant="contained" 
          onClick={testPublicApi} 
          disabled={loading}
          color="success"
        >
          Test Public API
        </Button>
      </Box>
      
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
          <CircularProgress />
        </Box>
      )}
      
      {error && (
        <Alert severity="error" sx={{ my: 2 }}>
          {error}
        </Alert>
      )}
      
      {response && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              API Response {cors ? '(MEXC API)' : '(Public API)'}:
            </Typography>
            <Box 
              component="pre" 
              sx={{ 
                bgcolor: '#f5f5f5', 
                p: 2, 
                borderRadius: 1,
                overflow: 'auto',
                maxHeight: '300px'
              }}
            >
              {JSON.stringify(response, null, 2)}
            </Box>
          </CardContent>
        </Card>
      )}
      
      <Box sx={{ mt: 4 }}>
        <Divider sx={{ my: 2 }} />
        <Typography variant="h6" gutterBottom>Debugging Tips:</Typography>
        <ul>
          <li>Check if any of the tests succeed</li>
          <li>If only the public API works, there may be CORS issues with MEXC</li>
          <li>Look at the browser console for additional error details</li>
          <li>Try using a different network connection</li>
          <li>Check if a browser extension might be blocking the requests</li>
        </ul>
      </Box>
      
      <Button 
        component={Link} 
        to="/mexc-dashboard" 
        variant="outlined" 
        sx={{ mt: 3, mr: 2 }}
      >
        Back to Mock Dashboard
      </Button>
      
      <Button 
        component={Link} 
        to="/mexc-dashboard-real" 
        variant="outlined" 
        sx={{ mt: 3 }}
      >
        Try Real Dashboard
      </Button>
    </Box>
  );
};

export default MexcApiDebugPage;
