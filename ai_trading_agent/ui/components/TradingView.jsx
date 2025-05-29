/**
 * Trading View Component
 * 
 * This component provides a chart view for trading data with support for
 * both mock and real data sources.
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  ButtonGroup,
  Button,
  useTheme
} from '@mui/material';
import axios from 'axios';

// Chart component - using a placeholder for now, in a real app you would 
// integrate with a charting library like TradingView, ApexCharts, etc.
const ChartPlaceholder = ({ isLoading, mockDataActive }) => {
  const theme = useTheme();
  
  if (isLoading) {
    return (
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        height: '100%'
      }}>
        <CircularProgress />
      </Box>
    );
  }
  
  return (
    <Box sx={{ 
      height: '100%', 
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      border: `1px dashed ${theme.palette.divider}`,
      borderRadius: 1,
      p: 2,
      bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)'
    }}>
      <Typography variant="subtitle1" color="textSecondary" gutterBottom>
        {mockDataActive ? 'Mock Chart Data' : 'Real Market Data'}
      </Typography>
      <Typography variant="body2" color="textSecondary" align="center">
        This is a placeholder for the chart component. In a production environment,
        this would be replaced with a real charting library that displays
        {mockDataActive ? ' generated mock data.' : ' real market data.'}
      </Typography>
    </Box>
  );
};

/**
 * Trading view component with symbol and timeframe selection
 */
const TradingView = () => {
  const theme = useTheme();
  const [isLoading, setIsLoading] = useState(true);
  const [symbol, setSymbol] = useState('BTC-USD');
  const [timeframe, setTimeframe] = useState('1d');
  const [mockDataActive, setMockDataActive] = useState(true);
  
  // Available symbols and timeframes
  const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD'];
  const timeframes = [
    { value: '5m', label: '5 min' },
    { value: '15m', label: '15 min' },
    { value: '1h', label: '1 hour' },
    { value: '4h', label: '4 hours' },
    { value: '1d', label: '1 day' },
  ];
  
  // Fetch data source status on component mount
  useEffect(() => {
    const fetchDataSourceStatus = async () => {
      try {
        const response = await axios.get('/api/data-source/status');
        setMockDataActive(response.data.use_mock_data);
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to fetch data source status:', error);
        setIsLoading(false);
      }
    };

    fetchDataSourceStatus();
    
    // Simulate chart loading
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1500);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Handle symbol change
  const handleSymbolChange = (event) => {
    setSymbol(event.target.value);
    setIsLoading(true);
    
    // Simulate chart loading
    setTimeout(() => {
      setIsLoading(false);
    }, 800);
  };
  
  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe);
    setIsLoading(true);
    
    // Simulate chart loading
    setTimeout(() => {
      setIsLoading(false);
    }, 500);
  };
  
  return (
    <Paper 
      elevation={0} 
      sx={{ 
        p: 2, 
        height: '100%',
        border: `1px solid ${theme.palette.divider}`,
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <Box sx={{ mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel id="symbol-select-label">Symbol</InputLabel>
              <Select
                labelId="symbol-select-label"
                id="symbol-select"
                value={symbol}
                label="Symbol"
                onChange={handleSymbolChange}
              >
                {symbols.map((sym) => (
                  <MenuItem key={sym} value={sym}>{sym}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={8}>
            <ButtonGroup size="small" aria-label="timeframe selection">
              {timeframes.map((tf) => (
                <Button 
                  key={tf.value}
                  variant={timeframe === tf.value ? 'contained' : 'outlined'}
                  onClick={() => handleTimeframeChange(tf.value)}
                >
                  {tf.label}
                </Button>
              ))}
            </ButtonGroup>
          </Grid>
        </Grid>
      </Box>
      
      <Box sx={{ flexGrow: 1, minHeight: 0 }}>
        <ChartPlaceholder isLoading={isLoading} mockDataActive={mockDataActive} />
      </Box>
    </Paper>
  );
};

export default TradingView;
