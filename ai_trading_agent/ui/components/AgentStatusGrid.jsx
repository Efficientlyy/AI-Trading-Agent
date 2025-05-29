/**
 * Agent Status Grid Component
 * 
 * This component displays the status of various trading agents and systems,
 * including data source status (mock/real).
 */

import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Grid,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  useTheme
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import MemoryIcon from '@mui/icons-material/Memory';
import StorageIcon from '@mui/icons-material/Storage';
import EqualizerIcon from '@mui/icons-material/Equalizer';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import DataObjectIcon from '@mui/icons-material/DataObject';
import axios from 'axios';

/**
 * Agent Status Grid Component
 */
const AgentStatusGrid = () => {
  const theme = useTheme();
  const [agentStatus, setAgentStatus] = useState({
    technicalAnalysis: { status: 'online', lastUpdate: new Date() },
    sentimentAnalysis: { status: 'warning', lastUpdate: new Date() },
    marketRegime: { status: 'online', lastUpdate: new Date() },
    signalValidator: { status: 'online', lastUpdate: new Date() },
    riskManager: { status: 'online', lastUpdate: new Date() },
    dataSource: { status: 'online', type: 'mock', lastUpdate: new Date() }
  });

  // Fetch agent status on component mount
  useEffect(() => {
    const fetchAgentStatus = async () => {
      try {
        // In a real implementation, this would fetch data from API
        // For now, we'll fetch just the data source status
        const response = await axios.get('/api/data-source/status');
        
        setAgentStatus(prevStatus => ({
          ...prevStatus,
          dataSource: { 
            status: 'online', 
            type: response.data.use_mock_data ? 'mock' : 'real', 
            lastUpdate: new Date() 
          }
        }));
      } catch (error) {
        console.error('Failed to fetch agent status:', error);
        
        // Update data source status to error
        setAgentStatus(prevStatus => ({
          ...prevStatus,
          dataSource: { 
            status: 'error', 
            type: prevStatus.dataSource.type, 
            lastUpdate: new Date() 
          }
        }));
      }
    };

    fetchAgentStatus();
    
    // Refresh status every 30 seconds
    const interval = setInterval(fetchAgentStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Format time elapsed since last update
  const formatTimeElapsed = (timestamp) => {
    const now = new Date();
    const seconds = Math.floor((now - timestamp) / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  // Status icon based on status
  const getStatusIcon = (status) => {
    switch (status) {
      case 'online':
        return <CheckCircleIcon fontSize="small" color="success" />;
      case 'warning':
        return <WarningIcon fontSize="small" color="warning" />;
      case 'error':
        return <ErrorIcon fontSize="small" color="error" />;
      default:
        return <CheckCircleIcon fontSize="small" color="success" />;
    }
  };

  // Agent icon based on agent type
  const getAgentIcon = (agentType) => {
    switch (agentType) {
      case 'technicalAnalysis':
        return <ShowChartIcon fontSize="small" />;
      case 'sentimentAnalysis':
        return <EqualizerIcon fontSize="small" />;
      case 'marketRegime':
        return <StorageIcon fontSize="small" />;
      case 'signalValidator':
        return <MemoryIcon fontSize="small" />;
      case 'riskManager':
        return <DataObjectIcon fontSize="small" />;
      case 'dataSource':
        return <StorageIcon fontSize="small" />;
      default:
        return <MemoryIcon fontSize="small" />;
    }
  };

  // Format agent name
  const formatAgentName = (name) => {
    return name
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, (str) => str.toUpperCase());
  };

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
        <Typography variant="h6">Agent Status</Typography>
        
        {/* Data Source Indicator */}
        <Chip 
          label={`${agentStatus.dataSource.type === 'mock' ? 'Mock' : 'Real'} Data`}
          size="small"
          color={agentStatus.dataSource.type === 'mock' ? 'warning' : 'success'}
          variant="outlined"
        />
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      <List>
        {Object.entries(agentStatus).map(([agentType, { status, lastUpdate, type }]) => (
          <ListItem key={agentType} sx={{ py: 0.5 }}>
            <ListItemIcon sx={{ minWidth: 36 }}>
              {getAgentIcon(agentType)}
            </ListItemIcon>
            <ListItemText 
              primary={
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body2">
                    {formatAgentName(agentType)}
                    {agentType === 'dataSource' && type && (
                      <Typography 
                        component="span" 
                        variant="caption" 
                        sx={{ 
                          ml: 1,
                          color: type === 'mock' ? theme.palette.warning.main : theme.palette.success.main
                        }}
                      >
                        ({type})
                      </Typography>
                    )}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {getStatusIcon(status)}
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                      {formatTimeElapsed(lastUpdate)}
                    </Typography>
                  </Box>
                </Box>
              }
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  );
};

export default AgentStatusGrid;
