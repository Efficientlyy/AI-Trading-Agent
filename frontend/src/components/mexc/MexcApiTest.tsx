import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Paper, CircularProgress } from '@mui/material';
import mexcService from '../../api/mexcService';

/**
 * Simple component to test MEXC API connectivity
 */
const MexcApiTest: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tickerData, setTickerData] = useState<any>(null);
  const [orderBookData, setOrderBookData] = useState<any>(null);
  const [tradesData, setTradesData] = useState<any>(null);
  
  const testSymbol = 'BTCUSDT';
  
  // Function to test ticker data
  const testTickerApi = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await mexcService.getSymbolData(testSymbol);
      setTickerData(data);
      console.log('Ticker data:', data);
    } catch (err) {
      setError(`Error fetching ticker data: ${err instanceof Error ? err.message : String(err)}`);
      console.error('Error fetching ticker data:', err);
    } finally {
      setLoading(false);
    }
  };
  
  // Function to test order book
  const testOrderBookApi = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await mexcService.getOrderBook(testSymbol);
      setOrderBookData(data);
      console.log('Order book data:', data);
    } catch (err) {
      setError(`Error fetching order book: ${err instanceof Error ? err.message : String(err)}`);
      console.error('Error fetching order book:', err);
    } finally {
      setLoading(false);
    }
  };
  
  // Function to test recent trades
  const testTradesApi = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await mexcService.getRecentTrades(testSymbol);
      setTradesData(data);
      console.log('Trades data:', data);
    } catch (err) {
      setError(`Error fetching trades: ${err instanceof Error ? err.message : String(err)}`);
      console.error('Error fetching trades:', err);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Paper sx={{ p: 3, m: 2 }}>
      <Typography variant="h5" sx={{ mb: 2 }}>
        MEXC API Test
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Button 
          variant="contained" 
          onClick={testTickerApi} 
          disabled={loading}
        >
          Test Ticker API
        </Button>
        
        <Button 
          variant="contained" 
          onClick={testOrderBookApi} 
          disabled={loading}
        >
          Test Order Book API
        </Button>
        
        <Button 
          variant="contained" 
          onClick={testTradesApi} 
          disabled={loading}
        >
          Test Trades API
        </Button>
      </Box>
      
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
        <Box sx={{ my: 2 }}>
          <Typography variant="h6">Ticker Data Result:</Typography>
          <pre style={{ 
            backgroundColor: '#f5f5f5', 
            padding: '10px', 
            borderRadius: '4px',
            overflow: 'auto',
            maxHeight: '200px'
          }}>
            {JSON.stringify(tickerData, null, 2)}
          </pre>
        </Box>
      )}
      
      {orderBookData && (
        <Box sx={{ my: 2 }}>
          <Typography variant="h6">Order Book Data Result:</Typography>
          <pre style={{ 
            backgroundColor: '#f5f5f5', 
            padding: '10px', 
            borderRadius: '4px',
            overflow: 'auto',
            maxHeight: '200px'
          }}>
            {JSON.stringify(orderBookData, null, 2)}
          </pre>
        </Box>
      )}
      
      {tradesData && (
        <Box sx={{ my: 2 }}>
          <Typography variant="h6">Trades Data Result:</Typography>
          <pre style={{ 
            backgroundColor: '#f5f5f5', 
            padding: '10px', 
            borderRadius: '4px',
            overflow: 'auto',
            maxHeight: '200px'
          }}>
            {JSON.stringify(tradesData, null, 2)}
          </pre>
        </Box>
      )}
    </Paper>
  );
};

export default MexcApiTest;
