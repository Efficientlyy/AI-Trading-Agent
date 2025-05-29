/**
 * Activity Feed Component
 * 
 * This component displays a feed of recent trading activity, system events,
 * and data source changes.
 */

import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemAvatar,
  Avatar,
  Divider,
  Chip,
  useTheme
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import NotificationsIcon from '@mui/icons-material/Notifications';
import ErrorIcon from '@mui/icons-material/Error';
import StorageIcon from '@mui/icons-material/Storage';
import SyncIcon from '@mui/icons-material/Sync';
import axios from 'axios';

/**
 * Activity Feed Component
 */
const ActivityFeed = () => {
  const theme = useTheme();
  const [activities, setActivities] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  // Mock activities for demonstration
  const mockActivities = [
    {
      id: 1,
      type: 'signal',
      symbol: 'BTC-USD',
      action: 'buy',
      timestamp: new Date(Date.now() - 5 * 60000), // 5 minutes ago
      details: 'Strong buy signal detected with 85% confidence'
    },
    {
      id: 2,
      type: 'dataSource',
      action: 'toggle',
      from: 'real',
      to: 'mock',
      timestamp: new Date(Date.now() - 15 * 60000), // 15 minutes ago
      details: 'Data source switched to mock for testing'
    },
    {
      id: 3,
      type: 'system',
      action: 'warning',
      timestamp: new Date(Date.now() - 30 * 60000), // 30 minutes ago
      details: 'High volatility detected in market, adjusting parameters'
    },
    {
      id: 4,
      type: 'signal',
      symbol: 'ETH-USD',
      action: 'sell',
      timestamp: new Date(Date.now() - 55 * 60000), // 55 minutes ago
      details: 'Sell signal confirmed by multiple indicators'
    },
    {
      id: 5,
      type: 'system',
      action: 'info',
      timestamp: new Date(Date.now() - 120 * 60000), // 2 hours ago
      details: 'Market regime classified as trending'
    },
    {
      id: 6,
      type: 'dataSource',
      action: 'update',
      timestamp: new Date(Date.now() - 180 * 60000), // 3 hours ago
      details: 'Mock data generator settings updated: increased volatility'
    }
  ];

  // Fetch activities on component mount
  useEffect(() => {
    const fetchActivities = async () => {
      try {
        // In a real implementation, this would fetch data from API
        // For now, we'll use our mock data
        
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 800));
        
        setActivities(mockActivities);
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to fetch activities:', error);
        setIsLoading(false);
      }
    };

    fetchActivities();
  }, []);

  // Format time elapsed since activity
  const formatTimeElapsed = (timestamp) => {
    const now = new Date();
    const seconds = Math.floor((now - timestamp) / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  // Get icon based on activity type and action
  const getActivityIcon = (type, action) => {
    if (type === 'signal') {
      if (action === 'buy') {
        return (
          <Avatar sx={{ bgcolor: theme.palette.success.main, width: 32, height: 32 }}>
            <TrendingUpIcon fontSize="small" />
          </Avatar>
        );
      } else {
        return (
          <Avatar sx={{ bgcolor: theme.palette.error.main, width: 32, height: 32 }}>
            <TrendingDownIcon fontSize="small" />
          </Avatar>
        );
      }
    } else if (type === 'dataSource') {
      return (
        <Avatar sx={{ bgcolor: theme.palette.info.main, width: 32, height: 32 }}>
          <StorageIcon fontSize="small" />
        </Avatar>
      );
    } else if (type === 'system') {
      if (action === 'warning') {
        return (
          <Avatar sx={{ bgcolor: theme.palette.warning.main, width: 32, height: 32 }}>
            <ErrorIcon fontSize="small" />
          </Avatar>
        );
      } else {
        return (
          <Avatar sx={{ bgcolor: theme.palette.primary.main, width: 32, height: 32 }}>
            <NotificationsIcon fontSize="small" />
          </Avatar>
        );
      }
    }
    
    return (
      <Avatar sx={{ bgcolor: theme.palette.primary.main, width: 32, height: 32 }}>
        <SyncIcon fontSize="small" />
      </Avatar>
    );
  };

  // Get activity title
  const getActivityTitle = (activity) => {
    const { type, action, symbol, from, to } = activity;
    
    if (type === 'signal') {
      return `${action.toUpperCase()} signal - ${symbol}`;
    } else if (type === 'dataSource') {
      if (action === 'toggle') {
        return `Data source switched: ${from} → ${to}`;
      }
      return 'Data source settings updated';
    } else if (type === 'system') {
      if (action === 'warning') {
        return 'System Warning';
      }
      return 'System Notification';
    }
    
    return 'Activity';
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
        <Typography variant="h6">Activity Feed</Typography>
        
        <Chip 
          label={`${activities.length} Events`}
          size="small"
          variant="outlined"
          color="primary"
        />
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <Typography variant="body2" color="text.secondary">
            Loading activities...
          </Typography>
        </Box>
      ) : (
        <List sx={{ overflow: 'auto', maxHeight: 350 }}>
          {activities.map((activity, index) => (
            <React.Fragment key={activity.id}>
              <ListItem alignItems="flex-start" sx={{ py: 1 }}>
                <ListItemAvatar sx={{ minWidth: 50 }}>
                  {getActivityIcon(activity.type, activity.action)}
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" fontWeight={500}>
                        {getActivityTitle(activity)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {formatTimeElapsed(activity.timestamp)}
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ display: 'inline', mt: 0.5 }}
                    >
                      {activity.details}
                    </Typography>
                  }
                />
              </ListItem>
              {index < activities.length - 1 && <Divider variant="inset" component="li" />}
            </React.Fragment>
          ))}
        </List>
      )}
    </Paper>
  );
};

export default ActivityFeed;
