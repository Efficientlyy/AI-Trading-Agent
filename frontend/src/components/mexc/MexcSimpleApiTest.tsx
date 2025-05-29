import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, CircularProgress, Button, Grid, Card, CardContent, Alert } from '@mui/material';
import mexcService from '../../api/mexcService';

// Setting this to true will automatically test the API on component mount
const AUTO_TEST_ON_MOUNT = true;

/**
 * Simple and lightweight component to test MEXC API connectivity
 */
const MexcSimpleApiTest: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tickerData, setTickerData] = useState<any>(null);
  const [symbol, setSymbol] = useState<string>('BTCUSDT');
  
  // Function to test basic ticker data - no WebSockets, just REST API
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('Attempting to fetch data from MEXC API...');
      const data = await mexcService.getSymbolData(symbol);
      setTickerData(data);
      console.log('MEXC API response:', data);
    } catch (err) {
      console.error('MEXC API error:', err);
      let errorMessage = err instanceof Error ? err.message : String(err);
      
      // Add more detailed error information
      if (err instanceof Error && err.stack) {
        console.error('Error stack:', err.stack);
      }
      
      if (errorMessage.includes('Network Error') || errorMessage.includes('timeout')) {
        errorMessage += ' - This may be due to CORS issues or API connectivity problems.';
      }
      
      setError(`Error: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Auto-test on component mount
  useEffect(() => {
    if (AUTO_TEST_ON_MOUNT) {
      fetchData();
    }
  }, []);
  
  const displayValue = (label: string, value: any) => {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="body2" color="text.secondary">{label}:</Typography>
        <Typography variant="body2">{value}</Typography>
      </Box>
    );
  };
  
  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        MEXC API Test (Simple Version)
      </Typography>
      
      <Button 
        variant="contained" 
        color="primary" 
        onClick={fetchData} 
        disabled={loading}
        sx={{ mb: 3 }}
      >
        {loading ? 'Loading...' : 'Test MEXC API'}
      </Button>
      
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 3 }}>
          <CircularProgress />
        </Box>
      )}
      
      {error && (
        <Typography color="error" sx={{ my: 2 }}>
          {error}
        </Typography>
      )}
      
      {tickerData && (
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {tickerData.symbol} Ticker
                </Typography>
                
                {displayValue('Price', tickerData.lastPrice)}
                {displayValue('24h Change', `${tickerData.priceChangePercent}%`)}
                {displayValue('24h High', tickerData.highPrice)}
                {displayValue('24h Low', tickerData.lowPrice)}
                {displayValue('24h Volume', tickerData.volume)}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default MexcSimpleApiTest;
