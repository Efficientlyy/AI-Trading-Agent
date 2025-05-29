/**
 * Performance Metrics Panel Component
 * 
 * This component displays key performance metrics for the trading system,
 * with awareness of whether data is coming from mock or real sources.
 */

import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Grid,
  Divider,
  LinearProgress,
  useTheme,
  Chip
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import WarningIcon from '@mui/icons-material/Warning';
import axios from 'axios';

// Metric card component
const MetricCard = ({ title, value, change, suffix, warning, isMockData }) => {
  const theme = useTheme();
  const isPositive = change > 0;
  
  return (
    <Box 
      sx={{ 
        p: 2, 
        borderRadius: 1,
        border: `1px solid ${theme.palette.divider}`,
        bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)',
        height: '100%'
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Typography variant="body2" color="text.secondary">
          {title}
        </Typography>
        {isMockData && (
          <Chip 
            label="Mock" 
            size="small" 
            color="warning" 
            variant="outlined" 
            sx={{ height: 20, '& .MuiChip-label': { px: 0.5, py: 0.1 } }} 
          />
        )}
      </Box>
      
      <Typography variant="h5" sx={{ my: 1, fontWeight: 500 }}>
        {value}{suffix}
      </Typography>
      
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        {warning ? (
          <Box sx={{ display: 'flex', alignItems: 'center', color: theme.palette.warning.main }}>
            <WarningIcon fontSize="small" sx={{ mr: 0.5 }} />
            <Typography variant="body2" color="inherit">
              {warning}
            </Typography>
          </Box>
        ) : (
          <Box 
            sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              color: isPositive ? theme.palette.success.main : theme.palette.error.main 
            }}
          >
            {isPositive ? 
              <ArrowUpwardIcon fontSize="small" sx={{ mr: 0.5 }} /> : 
              <ArrowDownwardIcon fontSize="small" sx={{ mr: 0.5 }} />
            }
            <Typography variant="body2" color="inherit">
              {Math.abs(change)}%
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};

/**
 * Performance Metrics Panel Component
 */
const PerformanceMetricsPanel = () => {
  const theme = useTheme();
  const [metrics, setMetrics] = useState({
    winRate: { value: 68.4, change: 2.3, suffix: '%' },
    profitFactor: { value: 1.84, change: -0.12, suffix: '' },
    sharpeRatio: { value: 1.27, change: 0.19, suffix: '' },
    drawdown: { value: 8.2, change: 1.5, suffix: '%', warning: 'Increasing' }
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isMockData, setIsMockData] = useState(true);

  // Fetch metrics on component mount
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        // In a real implementation, we would fetch performance metrics from API
        // For now, we'll just fetch the data source status
        const response = await axios.get('/api/data-source/status');
        setIsMockData(response.data.use_mock_data);
        
        // Simulate metrics loading
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
        setIsLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        p: 2, 
        height: '100%',
        border: `1px solid ${theme.palette.divider}`
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Performance Metrics</Typography>
        
        {isMockData && (
          <Chip 
            label="Mock Data Metrics" 
            size="small" 
            color="warning" 
            icon={<WarningIcon fontSize="small" />}
          />
        )}
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      {isLoading ? (
        <Box sx={{ width: '100%', mt: 2 }}>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 2 }}>
            Loading performance metrics...
          </Typography>
        </Box>
      ) : (
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <MetricCard 
              title="Win Rate" 
              value={metrics.winRate.value} 
              change={metrics.winRate.change} 
              suffix={metrics.winRate.suffix}
              isMockData={isMockData}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <MetricCard 
              title="Profit Factor" 
              value={metrics.profitFactor.value} 
              change={metrics.profitFactor.change} 
              suffix={metrics.profitFactor.suffix}
              isMockData={isMockData}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <MetricCard 
              title="Sharpe Ratio" 
              value={metrics.sharpeRatio.value} 
              change={metrics.sharpeRatio.change} 
              suffix={metrics.sharpeRatio.suffix}
              isMockData={isMockData}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <MetricCard 
              title="Max Drawdown" 
              value={metrics.drawdown.value} 
              change={metrics.drawdown.change} 
              suffix={metrics.drawdown.suffix}
              warning={metrics.drawdown.warning}
              isMockData={isMockData}
            />
          </Grid>
        </Grid>
      )}
      
      {!isLoading && isMockData && (
        <Typography 
          variant="caption" 
          color="text.secondary" 
          sx={{ 
            display: 'block', 
            mt: 2, 
            fontStyle: 'italic',
            textAlign: 'center'
          }}
        >
          These metrics are based on mock data and may not reflect actual market performance.
          Switch to real data for accurate performance metrics.
        </Typography>
      )}
    </Paper>
  );
};

export default PerformanceMetricsPanel;
