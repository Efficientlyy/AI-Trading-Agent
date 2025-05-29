import React from 'react';
import { Box, Typography, Button, Paper, Alert } from '@mui/material';
import { Link } from 'react-router-dom';
import RealMexcDashboard from '../components/mexc/RealMexcDashboard';

/**
 * Page wrapper for the real MEXC dashboard that uses actual API data
 */
const RealMexcDashboardPage: React.FC = () => {
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="h5" component="h1">
          MEXC Dashboard (Real Data)
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Trading dashboard with real MEXC API data - REST API with controlled refresh
        </Typography>
        
        <Alert severity="info" sx={{ mb: 2 }}>
          This version uses the real MEXC API with REST endpoints and controlled refresh rate to prevent browser unresponsiveness.
        </Alert>

        <Button 
          component={Link} 
          to="/mexc-dashboard" 
          variant="outlined" 
          size="small"
          sx={{ mr: 1 }}
        >
          Switch to Mock Data Version
        </Button>
      </Box>
      
      <Box sx={{ flexGrow: 1, overflow: 'hidden' }}>
        <RealMexcDashboard />
      </Box>
    </Box>
  );
};

export default RealMexcDashboardPage;
