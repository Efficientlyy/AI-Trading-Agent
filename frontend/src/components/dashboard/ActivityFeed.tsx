import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  MoreVert as MoreVertIcon,
  Pause as PauseIcon,
  PlayArrow as PlayIcon,
  Refresh as RefreshIcon,
  Stop as StopIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  IconButton,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Tooltip,
  Typography
} from '@mui/material';
import React, { useEffect, useState } from 'react';
import { useSystemControl } from '../../context/SystemControlContext';

// Dark theme constants
const darkBg = '#1E1E1E';
const darkPaperBg = '#2D2D2D';
const darkBorder = '#444444';
const darkText = '#FFFFFF';
const darkSecondaryText = '#AAAAAA';

// Define activity event interface
interface ActivityEvent {
  id: string;
  timestamp: string;
  type: 'system' | 'agent' | 'session' | 'alert';
  action: string;
  status: 'success' | 'error' | 'warning' | 'info';
  message: string;
  details?: {
    agent_id?: string;
    agent_name?: string;
    session_id?: string;
    symbol?: string;
    [key: string]: any;
  };
}

const ActivityFeed: React.FC = () => {
  const { systemStatus, agents, sessions } = useSystemControl();
  const [activities, setActivities] = useState<ActivityEvent[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [maxItems, setMaxItems] = useState<number>(10);

  // Mock function to fetch activities - in a real app, this would come from an API
  const fetchActivities = async () => {
    setLoading(true);

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 500));

      // In a real implementation, this would be an API call
      // For now, we'll generate mock activities based on current system state
      const mockActivities: ActivityEvent[] = [
        {
          id: '1',
          timestamp: new Date().toISOString(),
          type: 'system',
          action: 'status_change',
          status: 'success',
          message: `System ${systemStatus?.status === 'running' ? 'started' : 'stopped'} successfully`
        },
        {
          id: '2',
          timestamp: new Date(Date.now() - 5 * 60000).toISOString(),
          type: 'agent',
          action: 'status_change',
          status: 'info',
          message: 'Trading agent status changed',
          details: {
            agent_id: agents[0]?.agent_id || 'agent-1',
            agent_name: agents[0]?.name || 'BTC Momentum Agent',
            previous_status: 'stopped',
            new_status: 'running'
          }
        },
        {
          id: '3',
          timestamp: new Date(Date.now() - 15 * 60000).toISOString(),
          type: 'session',
          action: 'pause',
          status: 'warning',
          message: 'Trading session paused',
          details: {
            session_id: sessions[0]?.session_id || 'session-1',
            reason: 'User initiated'
          }
        },
        {
          id: '4',
          timestamp: new Date(Date.now() - 30 * 60000).toISOString(),
          type: 'alert',
          action: 'drawdown_warning',
          status: 'warning',
          message: 'Portfolio drawdown warning',
          details: {
            session_id: sessions[0]?.session_id || 'session-1',
            drawdown: '5.2%',
            threshold: '5%'
          }
        },
        {
          id: '5',
          timestamp: new Date(Date.now() - 60 * 60000).toISOString(),
          type: 'agent',
          action: 'trade_signal',
          status: 'success',
          message: 'Buy signal generated',
          details: {
            agent_id: agents[0]?.agent_id || 'agent-1',
            agent_name: agents[0]?.name || 'BTC Momentum Agent',
            symbol: 'BTC/USD',
            signal: 'BUY',
            confidence: '0.85'
          }
        },
        {
          id: '6',
          timestamp: new Date(Date.now() - 90 * 60000).toISOString(),
          type: 'system',
          action: 'health_check',
          status: 'error',
          message: 'System health check failed',
          details: {
            component: 'Data Feed',
            error: 'Connection timeout'
          }
        },
        {
          id: '7',
          timestamp: new Date(Date.now() - 120 * 60000).toISOString(),
          type: 'session',
          action: 'start',
          status: 'success',
          message: 'Trading session started',
          details: {
            session_id: sessions[0]?.session_id || 'session-1',
            initial_capital: '$10,000'
          }
        },
        {
          id: '8',
          timestamp: new Date(Date.now() - 150 * 60000).toISOString(),
          type: 'agent',
          action: 'config_change',
          status: 'info',
          message: 'Agent configuration updated',
          details: {
            agent_id: agents[1]?.agent_id || 'agent-2',
            agent_name: agents[1]?.name || 'ETH Trend Agent',
            changes: 'Risk parameters adjusted'
          }
        },
        {
          id: '9',
          timestamp: new Date(Date.now() - 180 * 60000).toISOString(),
          type: 'alert',
          action: 'volatility_spike',
          status: 'warning',
          message: 'Market volatility spike detected',
          details: {
            symbol: 'BTC/USD',
            volatility: '4.2%',
            threshold: '3%'
          }
        },
        {
          id: '10',
          timestamp: new Date(Date.now() - 210 * 60000).toISOString(),
          type: 'system',
          action: 'startup',
          status: 'success',
          message: 'System initialized successfully',
          details: {
            version: '1.0.0',
            components: 'All components loaded'
          }
        }
      ];

      setActivities(mockActivities);
    } catch (error) {
      console.error('Error fetching activities:', error);
    } finally {
      setLoading(false);
    }
  };

  // Load activities on component mount
  useEffect(() => {
    fetchActivities();
  }, []);

  // Get icon for activity type
  const getActivityIcon = (activity: ActivityEvent) => {
    // First determine by status
    switch (activity.status) {
      case 'success':
        return <CheckCircleIcon color="success" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'info':
        return <InfoIcon color="info" />;
      default:
        // If no status, determine by type and action
        switch (activity.type) {
          case 'system':
            return activity.action === 'startup' || activity.action === 'status_change'
              ? <PlayIcon color="primary" />
              : <InfoIcon color="primary" />;
          case 'agent':
            return activity.action === 'status_change'
              ? (activity.details?.new_status === 'running' ? <PlayIcon color="success" /> : <StopIcon color="error" />)
              : <InfoIcon color="primary" />;
          case 'session':
            return activity.action === 'pause'
              ? <PauseIcon color="warning" />
              : activity.action === 'start'
                ? <PlayIcon color="success" />
                : <InfoIcon color="primary" />;
          case 'alert':
            return <WarningIcon color="warning" />;
          default:
            return <InfoIcon color="primary" />;
        }
    }
  };

  // Format timestamp to readable format
  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.round(diffMs / 60000);

    if (diffMins < 1) {
      return 'Just now';
    } else if (diffMins < 60) {
      return `${diffMins} min${diffMins > 1 ? 's' : ''} ago`;
    } else if (diffMins < 1440) {
      const hours = Math.floor(diffMins / 60);
      return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else {
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
  };

  // Get color for activity type chip
  const getActivityTypeColor = (type: string) => {
    switch (type) {
      case 'system':
        return 'primary';
      case 'agent':
        return 'info';
      case 'session':
        return 'secondary';
      case 'alert':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Card elevation={3} sx={{ mb: 3, bgcolor: '#0f0f0f', color: darkText, borderRadius: 2, border: '1px solid #555' }}> {/* Made bg even darker, added border, added mb:3 like others */}
      <CardContent sx={{ bgcolor: 'transparent' }}> {/* Ensure CardContent is transparent */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5" component="div" sx={{ fontWeight: 'bold', color: darkText }}>
            Activity Feed
          </Typography>

          <Tooltip title="Refresh Activities">
            <IconButton
              onClick={fetchActivities}
              disabled={loading}
              size="small"
              sx={{ color: darkText }}
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : activities.length === 0 ? (
          <Box sx={{ textAlign: 'center', p: 3 }}>
            <Typography variant="body1" sx={{ color: darkSecondaryText }}>
              No activities recorded yet
            </Typography>
          </Box>
        ) : (
          <>
            <List sx={{ width: '100%', bgcolor: darkPaperBg }}>
              {activities.slice(0, maxItems).map((activity, index) => (
                <React.Fragment key={activity.id}>
                  {index > 0 && <Divider component="li" sx={{ borderColor: darkBorder }} />}
                  <ListItem
                    alignItems="flex-start"
                    sx={{ '&:hover': { bgcolor: darkBg } }}
                    secondaryAction={
                      <Tooltip title="More Options">
                        <IconButton edge="end" size="small" sx={{ color: darkText }}>
                          <MoreVertIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    }
                  >
                    <ListItemIcon>
                      {getActivityIcon(activity)}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                          <Typography variant="body1" component="span" sx={{ fontWeight: 'medium', color: darkText }}>
                            {activity.message}
                          </Typography>
                          <Chip
                            label={activity.type.toUpperCase()}
                            size="small"
                            color={getActivityTypeColor(activity.type) as any}
                            sx={{ height: 20, fontSize: '0.7rem' }}
                          />
                        </Box>
                      }
                      secondary={
                        // Wrap everything in a span to avoid p > div nesting issue
                        <span>
                          <Typography
                            component="span"
                            variant="body2"
                            sx={{ color: darkSecondaryText, display: 'block' }}
                          >
                            {formatTimestamp(activity.timestamp)}
                          </Typography>

                          {activity.details && (
                            <span style={{ display: 'block', marginTop: '4px' }}>
                              {Object.entries(activity.details).map(([key, value]) => (
                                <Typography
                                  key={key}
                                  variant="caption"
                                  component="span"
                                  sx={{ display: 'flex', gap: 1, color: darkSecondaryText, marginTop: '2px' }}
                                >
                                  <span style={{ fontWeight: 500 }}>
                                    {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}:
                                  </span>
                                  <span>{value}</span>
                                </Typography>
                              ))}
                            </span>
                          )}
                        </span>
                      }
                    />
                  </ListItem>
                </React.Fragment>
              ))}
            </List>

            {activities.length > maxItems && (
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => setMaxItems(prev => prev + 10)}
                  sx={{ color: darkText, borderColor: darkBorder, '&:hover': { borderColor: darkText } }}
                >
                  Load More
                </Button>
              </Box>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default ActivityFeed;
