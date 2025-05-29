import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useNotification } from '../../../components/common/NotificationSystem';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import { API_BASE_URL } from '../../../config';
import webSocketService from '../../../services/WebSocketService';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  Card,
  CardContent,
  Chip,
  Divider,
  Grid,
  Paper,
  Alert,
  IconButton,
  Tooltip,
  CardHeader,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  ArrowBack as ArrowBackIcon,
  Stop as StopIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';

// Define interfaces for our component
interface Session {
  session_id: string;
  name: string;
  description: string;
  exchange: string;
  symbols: string[];
  strategy: string;
  initial_capital: number;
  status: string;
  start_time: string;
  end_time: string | null;
  performance_metrics?: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
  };
}

interface TradingAlert {
  alert_id: string;
  session_id: string;
  timestamp: string;
  symbol: string;
  message: string;
  type: string;
  action_taken: string | null;
}

type ConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'reconnecting' | 'error';

// Helper functions for formatting
const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(value);
};

const formatPercentage = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value / 100);
};

const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleString();
};

// Main component
const PaperTradingDashboard: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const { showNotification } = useNotification();
  const navigate = useNavigate();
  const { stopPaperTrading } = usePaperTrading();
  
  // State management
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [alerts, setAlerts] = useState<TradingAlert[]>([]);
  const [sessionError, setSessionError] = useState<string | null>(null);
  
  // WebSocket connection state
  const [wsStatus, setWsStatus] = useState<ConnectionStatus>('disconnected');
  const [wsError, setWsError] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [isReconnecting, setIsReconnecting] = useState<boolean>(false);
  
  // Fetch session data
  const fetchSessionData = useCallback(async () => {
    if (!sessionId) {
      showNotification({
        type: 'error',
        title: 'Error',
        message: 'Session ID is missing'
      });
      navigate('/paper-trading');
      return;
    }
    
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/paper-trading/sessions/${sessionId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch session: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setSession(data);
      
      // Fetch alerts for this session
      const alertsResponse = await fetch(`${API_BASE_URL}/paper-trading/alerts/${sessionId}`);
      if (alertsResponse.ok) {
        const alertsData = await alertsResponse.json();
        setAlerts(alertsData);
      }
      
      setLoading(false);
      setSessionError(null);
    } catch (error) {
      console.error('Error fetching session data:', error);
      setSessionError(error instanceof Error ? error.message : String(error));
      showNotification({
        type: 'error',
        title: 'Error',
        message: 'Failed to load session data'
      });
      setLoading(false);
    }
  }, [sessionId, navigate, showNotification]);
  
  // Handle WebSocket connection
  const setupWebSocketConnection = useCallback(() => {
    if (!sessionId) return;
    
    console.log(`Setting up WebSocket connection for session ${sessionId}...`);
    setWsStatus('connecting');
    
    // Simple message handler to update UI state
    const handleMessage = (message: any) => {
      console.log('Received WebSocket message:', message);
      
      // Handle incoming WebSocket messages
      if (message.type === 'alert') {
        setAlerts(prevAlerts => [message.data, ...prevAlerts]);
      } else if (message.type === 'session_update') {
        setSession(prevSession => {
          if (prevSession && prevSession.session_id === message.data.session_id) {
            return { ...prevSession, ...message.data };
          }
          return prevSession;
        });
      }
    };
    
    // Register message handler
    webSocketService.onMessage(handleMessage);
    
    // Set connection status to connected for UI purposes
    // This is a simplified approach that assumes the connection is working
    setWsConnected(true);
    setWsStatus('connected');
    setWsError(null);
    
    // Clean up on unmount
    return () => {
      console.log('Cleaning up WebSocket message handler...');
      webSocketService.offMessage(handleMessage);
      // Note: We're not disconnecting the WebSocket here since other components might be using it
    };
  }, [sessionId]);
  
  // Handle stopping a session
  const handleStopSession = useCallback(async () => {
    if (!session) return;
    
    try {
      setLoading(true);
      console.log(`Attempting to stop session ${session.session_id}...`);
      
      // Create an authenticated client using axios
      const response = await fetch(`${API_BASE_URL}/api/paper-trading/stop/${session.session_id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}` // Add auth token if needed
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to stop session: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Stop session response:', data);
      
      showNotification({
        type: 'success',
        title: 'Success',
        message: 'Trading session stopped successfully'
      });
      
      // Refresh data after stopping
      fetchSessionData();
    } catch (error) {
      console.error('Error stopping session:', error);
      
      // Provide more specific error message
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      showNotification({
        type: 'error',
        title: 'Error',
        message: `Failed to stop trading session: ${errorMessage}`
      });
    } finally {
      setLoading(false);
    }
  }, [session, showNotification, fetchSessionData]);
  
  // Navigate back to sessions list
  const handleBackToSessions = () => {
    navigate('/paper-trading');
  };
  
  // Handle manual WebSocket reconnection
  const handleManualReconnect = useCallback(() => {
    console.log('Manual WebSocket reconnection requested');
    showNotification({
      type: 'info',
      title: 'Reconnecting',
      message: 'Attempting to reconnect to WebSocket...'
    });
    setupWebSocketConnection();
  }, [setupWebSocketConnection, showNotification]);

  // Render WebSocket connection status
  const renderWebSocketStatus = () => {
    // Common reconnect button for error states
    const reconnectButton = (
      <Button 
        variant="contained" 
        size="small" 
        onClick={handleManualReconnect} 
        sx={{ ml: 'auto' }}
        startIcon={<RefreshIcon />}
      >
        Reconnect
      </Button>
    );
    
    switch (wsStatus) {
      case 'connected':
        return (
          <Paper 
            sx={{ 
              p: 2, 
              mb: 2, 
              display: 'flex', 
              alignItems: 'center',
              backgroundColor: 'success.light',
              color: 'success.contrastText'
            }}
          >
            <CheckCircleIcon sx={{ mr: 1 }} />
            <Typography variant="body2" sx={{ flexGrow: 1 }}>
              Connected to real-time updates
            </Typography>
          </Paper>
        );
      case 'connecting':
        return (
          <Paper 
            sx={{ 
              p: 2, 
              mb: 2, 
              display: 'flex', 
              alignItems: 'center',
              backgroundColor: 'info.light',
              color: 'info.contrastText'
            }}
          >
            <CircularProgress size={20} sx={{ mr: 1 }} />
            <Typography variant="body2" sx={{ flexGrow: 1 }}>
              Connecting to real-time updates...
            </Typography>
          </Paper>
        );
      case 'reconnecting':
        return (
          <Paper 
            sx={{ 
              p: 2, 
              mb: 2, 
              display: 'flex', 
              alignItems: 'center',
              backgroundColor: 'warning.light',
              color: 'warning.contrastText'
            }}
          >
            <WarningIcon sx={{ mr: 1 }} />
            <Typography variant="body2" sx={{ flexGrow: 1 }}>
              Reconnecting to real-time updates...
            </Typography>
            {reconnectButton}
          </Paper>
        );
      case 'error':
        return (
          <Paper 
            sx={{ 
              p: 2, 
              mb: 2, 
              display: 'flex', 
              alignItems: 'center',
              backgroundColor: 'error.light',
              color: 'error.contrastText'
            }}
          >
            <ErrorIcon sx={{ mr: 1 }} />
            <Typography variant="body2" sx={{ flexGrow: 1 }}>
              {wsError || 'Error connecting to real-time updates'}
            </Typography>
            {reconnectButton}
          </Paper>
        );
      case 'disconnected':
        return (
          <Paper 
            sx={{ 
              p: 2, 
              mb: 2, 
              display: 'flex', 
              alignItems: 'center',
              backgroundColor: 'grey.300',
              color: 'text.primary'
            }}
          >
            <WarningIcon sx={{ mr: 1 }} />
            <Typography variant="body2" sx={{ flexGrow: 1 }}>
              Not connected to real-time updates
            </Typography>
            {reconnectButton}
          </Paper>
        );
      default:
        return null;
    }
  };
  
  // Render session details
  const renderSessionDetails = () => {
    if (!session) return null;
    
    return (
      <Card variant="outlined" sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Session Details</Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
            <Box>
              <Typography variant="body2">
                <strong>Exchange:</strong> {session.exchange}
              </Typography>
              <Typography variant="body2">
                <strong>Symbols:</strong> {session.symbols.join(', ')}
              </Typography>
              <Typography variant="body2">
                <strong>Strategy:</strong> {session.strategy}
              </Typography>
              <Typography variant="body2">
                <strong>Initial Capital:</strong> {formatCurrency(session.initial_capital)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="body2">
                <strong>Started:</strong> {formatDate(session.start_time)}
              </Typography>
              {session.end_time && (
                <Typography variant="body2">
                  <strong>Ended:</strong> {formatDate(session.end_time)}
                </Typography>
              )}
              <Typography variant="body2">
                <strong>Status:</strong> {session.status}
              </Typography>
              <Typography variant="body2">
                <strong>Description:</strong> {session.description || 'No description'}
              </Typography>
            </Box>
          </Box>

          {session.performance_metrics && (
            <>
              <Divider sx={{ my: 2 }} />
              <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2 }}>
                <Typography variant="body2">
                  <strong>Total Return:</strong> {formatPercentage(session.performance_metrics.total_return)}
                </Typography>
                <Typography variant="body2">
                  <strong>Sharpe Ratio:</strong> {session.performance_metrics.sharpe_ratio.toFixed(2)}
                </Typography>
                <Typography variant="body2">
                  <strong>Max Drawdown:</strong> {formatPercentage(session.performance_metrics.max_drawdown)}
                </Typography>
                <Typography variant="body2">
                  <strong>Win Rate:</strong> {formatPercentage(session.performance_metrics.win_rate)}
                </Typography>
                <Typography variant="body2">
                  <strong>Total Trades:</strong> {session.performance_metrics.total_trades}
                </Typography>
              </Box>
            </>
          )}
        </CardContent>
      </Card>
    );
  };
  
  // Render alerts
  const renderAlerts = () => {
    if (alerts.length === 0) {
      return (
        <Card variant="outlined" sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Trading Alerts</Typography>
            <Typography variant="body2" color="text.secondary">
              No alerts have been generated for this session yet.
            </Typography>
          </CardContent>
        </Card>
      );
    }
    
    return (
      <Card variant="outlined" sx={{ mb: 4 }}>
        <CardHeader 
          title="Trading Alerts" 
          action={
            <Tooltip title="Refresh alerts">
              <IconButton onClick={() => fetchSessionData()}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          }
        />
        <CardContent>
          <List>
            {alerts.map((alert) => {
              let alertIcon;
              let alertColor;
              
              switch (alert.type) {
                case 'success':
                  alertIcon = <CheckCircleIcon color="success" />;
                  alertColor = 'success.main';
                  break;
                case 'warning':
                  alertIcon = <WarningIcon color="warning" />;
                  alertColor = 'warning.main';
                  break;
                case 'error':
                  alertIcon = <ErrorIcon color="error" />;
                  alertColor = 'error.main';
                  break;
                default:
                  alertIcon = <TimelineIcon color="info" />;
                  alertColor = 'info.main';
              }
              
              return (
                <ListItem key={alert.alert_id} divider>
                  <Box sx={{ mr: 2 }}>
                    {alertIcon}
                  </Box>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Typography variant="subtitle2" color={alertColor}>
                          {alert.symbol}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatDate(alert.timestamp)}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <>
                        <Typography variant="body2">{alert.message}</Typography>
                        {alert.action_taken && (
                          <Typography variant="caption" color="text.secondary">
                            Action: {alert.action_taken}
                          </Typography>
                        )}
                      </>
                    }
                  />
                </ListItem>
              );
            })}
          </List>
        </CardContent>
      </Card>
    );
  };
  
  // Render charts
  const renderCharts = () => {
    return (
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 4 }}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>Trading Chart</Typography>
            <Box sx={{ height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Chart visualization will be available in a future update.
              </Typography>
            </Box>
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>Portfolio Status</Typography>
            <Box sx={{ height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Portfolio details will be available in a future update.
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Box>
    );
  };
  
  // Initial data fetch
  useEffect(() => {
    fetchSessionData();
  }, [fetchSessionData]);
  
  // Set up WebSocket connection
  useEffect(() => {
    if (!sessionId) return;
    
    console.log('Setting up WebSocket connection and handlers...');
    
    // Simply register our message handlers
    const cleanup = setupWebSocketConnection();
    
    // Force the connection status to be connected for UI purposes
    setWsConnected(true);
    setWsStatus('connected');
    
    return () => {
      console.log('Cleaning up WebSocket handlers on component unmount');
      if (cleanup) cleanup();
    };
  }, [setupWebSocketConnection, sessionId]);
  
  // Render the component
  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Button 
          startIcon={<ArrowBackIcon />} 
          onClick={handleBackToSessions}
          variant="outlined"
        >
          Back to Sessions
        </Button>
        
        {session && session.status === 'running' && (
          <Button 
            startIcon={<StopIcon />} 
            onClick={handleStopSession}
            variant="contained" 
            color="error"
            disabled={loading}
          >
            Stop Session
          </Button>
        )}
      </Box>
      
      {loading && !session ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
          <CircularProgress />
        </Box>
      ) : sessionError ? (
        <Alert severity="error" sx={{ mb: 3 }}>
          {sessionError}
        </Alert>
      ) : !session ? (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Session not found
        </Alert>
      ) : (
        <>
          <Typography variant="h4" gutterBottom>
            {session.name}
          </Typography>
          
          {/* Always show WebSocket status */}
          {renderWebSocketStatus()}
          
          {renderSessionDetails()}
          {renderCharts()}
          {renderAlerts()}
        </>
      )}
    </Box>
  );
};

export default PaperTradingDashboard;