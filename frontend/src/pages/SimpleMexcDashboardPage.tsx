import React from 'react';
import SimpleMexcDashboard from '../components/mexc/SimpleMexcDashboard';
import { Box } from '@mui/material';

const SimpleMexcDashboardPage: React.FC = () => {
  return (
    <Box sx={{ height: '100%', width: '100%' }}>
      <SimpleMexcDashboard defaultSymbol="BTC/USDT" defaultTimeframe="1h" />
    </Box>
  );
};

export default SimpleMexcDashboardPage;