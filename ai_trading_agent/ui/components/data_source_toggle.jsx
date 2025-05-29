/**
 * Data Source Toggle Component
 * 
 * This component provides a UI toggle switch to control whether the system
 * uses mock data or real market data for analysis and trading.
 */

import React, { useState, useEffect } from 'react';
import { Switch, FormControlLabel, Box, Typography, Snackbar, Alert } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useTheme } from '@mui/material/styles';
import axios from 'axios';

const StyledFormControlLabel = styled(FormControlLabel)(({ theme }) => ({
  marginRight: theme.spacing(2),
  '& .MuiFormControlLabel-label': {
    fontWeight: 500,
  },
}));

/**
 * Data Source Toggle component that provides a UI control for switching
 * between mock and real data sources.
 */
const DataSourceToggle = ({ onChange, initialState, disabled = false }) => {
  const theme = useTheme();
  const [isRealData, setIsRealData] = useState(initialState || false);
  const [isLoading, setIsLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  // Load initial state from API if not provided
  useEffect(() => {
    if (initialState === undefined) {
      fetchCurrentState();
    }
  }, []);

  /**
   * Fetch the current data source state from the API
   */
  const fetchCurrentState = async () => {
    try {
      const response = await axios.get('/api/data-source/status');
      setIsRealData(!response.data.use_mock_data);
    } catch (error) {
      console.error('Failed to fetch data source status:', error);
      setSnackbar({
        open: true,
        message: 'Failed to fetch data source status',
        severity: 'error'
      });
    }
  };

  /**
   * Handle toggle switch change
   */
  const handleChange = async (event) => {
    if (disabled || isLoading) return;
    
    setIsLoading(true);
    try {
      // Toggle via API
      const response = await axios.post('/api/data-source/toggle');
      const newState = response.data.current === 'real';
      
      // Update local state
      setIsRealData(newState);
      
      // Call onChange handler if provided
      if (onChange) {
        onChange(newState);
      }
      
      // Show success message
      setSnackbar({
        open: true,
        message: `Data source switched to ${newState ? 'real' : 'mock'} data`,
        severity: 'success'
      });
    } catch (error) {
      console.error('Failed to toggle data source:', error);
      setSnackbar({
        open: true,
        message: 'Failed to toggle data source',
        severity: 'error'
      });
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Close snackbar notification
   */
  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      alignItems: 'center', 
      bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
      borderRadius: 1,
      padding: '4px 12px',
      border: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
    }}>
      <Typography 
        variant="subtitle2" 
        sx={{ 
          marginRight: '8px',
          color: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)'
        }}
      >
        Data:
      </Typography>
      
      <Typography 
        variant="subtitle2" 
        sx={{ 
          marginRight: '4px',
          color: !isRealData ? theme.palette.primary.main : 'inherit',
          fontWeight: !isRealData ? 'bold' : 'normal'
        }}
      >
        Mock
      </Typography>
      
      <Switch
        checked={isRealData}
        onChange={handleChange}
        disabled={disabled || isLoading}
        color="primary"
        size="small"
        inputProps={{ 'aria-label': 'Toggle between mock and real data' }}
      />
      
      <Typography 
        variant="subtitle2" 
        sx={{ 
          marginLeft: '4px',
          color: isRealData ? theme.palette.primary.main : 'inherit',
          fontWeight: isRealData ? 'bold' : 'normal'
        }}
      >
        Real
      </Typography>
      
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={4000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DataSourceToggle;
