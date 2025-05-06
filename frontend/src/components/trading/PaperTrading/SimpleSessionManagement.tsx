import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../../../config';
import { 
  Box, 
  Button, 
  Card, 
  CardActions, 
  CardContent, 
  Chip,
  CircularProgress,
  Divider,
  Grid as MuiGrid, 
  Typography
} from '@mui/material';
import { 
  PlayArrow as PlayArrowIcon, 
  Stop as StopIcon, 
  Refresh as RefreshIcon,
  Add as AddIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

// Create a Grid component that works with Material UI v5
const Grid = (props: any) => <MuiGrid {...props} />;

// Define interfaces for session data
interface Session {
  session_id: string;
  name: string;
  description: string;
  exchange: string;
  symbols: string[];
  strategy: string;
  initial_capital: number;
  status: 'active' | 'paused' | 'completed' | 'error';
  start_time: string;
  end_time?: string;
  performance_metrics?: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
  };
}

const SimpleSessionManagement: React.FC = () => {
  // Component state
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(false);
  const [stoppedSessions, setStoppedSessions] = useState<Set<string>>(new Set());
  const navigate = useNavigate();

  // Fetch sessions on component mount
  useEffect(() => {
    fetchSessions();
  }, []);

  // Fetch sessions from API
  const fetchSessions = async () => {
    try {
      setLoading(true);
      const sessionsUrl = `${API_BASE_URL}/paper-trading/sessions`;
      console.log('Fetching sessions from:', sessionsUrl);
      
      const response = await fetch(sessionsUrl);
      console.log('Sessions response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response from server:', errorText);
        throw new Error(`Failed to fetch sessions: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Sessions retrieved:', data);
      setSessions(Array.isArray(data) ? data : []);
    } catch (error: any) {
      console.error('Error fetching sessions:', error);
      alert(`Failed to fetch sessions: ${error.message || 'Unknown error'}`);
      setSessions([]);
    } finally {
      setLoading(false);
    }
  };

  // Handle create session
  const handleCreateSession = () => {
    navigate('/paper-trading/new');
  };

  // Handle refresh
  const handleRefresh = () => {
    fetchSessions();
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  // Handle stop session
  const handleStopSession = async (sessionId: string) => {
    try {
      // If we've already tried to stop this session, don't try again
      if (stoppedSessions.has(sessionId)) {
        console.log(`Session ${sessionId} already being stopped, ignoring duplicate request`);
        return;
      }
      
      // Mark this session as being stopped to prevent multiple attempts
      setStoppedSessions(prev => {
        const newSet = new Set(prev);
        newSet.add(sessionId);
        return newSet;
      });
      
      // Immediately update the UI to show the session as stopped
      setSessions(prevSessions => {
        return prevSessions.map(session => {
          if (session.session_id === sessionId) {
            // Create a copy of the session with status changed to 'completed'
            return {
              ...session,
              status: 'completed',
              end_time: new Date().toISOString()
            };
          }
          return session;
        });
      });
      
      setLoading(true);
      console.log(`Stopping session with ID: ${sessionId}`);
      
      // Use the endpoint that we know works from our previous debugging
      const stopUrl = `${API_BASE_URL}/paper-trading/sessions/${sessionId}/stop`;
      console.log(`Making API call to: ${stopUrl}`);
      
      const response = await fetch(stopUrl, {
        method: 'POST',
      });
      
      console.log('Stop session response:', response);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response from server:', errorText);
        // Don't throw here - we've already updated the UI
        console.error(`Failed to stop session on backend: ${response.status} ${response.statusText}`);
      } else {
        console.log('Session stopped successfully on backend');
      }
    } catch (error: any) {
      console.error('Error stopping session:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mb: 3 
      }}>
        <Typography variant="h4" sx={{ color: '#f8fafc' }}>Paper Trading Sessions</Typography>
        <Box>
          <Button
            variant="outlined"
            sx={{ 
              mr: 2,
              color: '#f8fafc',
              borderColor: '#f8fafc',
              '&:hover': {
                borderColor: '#cbd5e1',
                backgroundColor: 'rgba(248, 250, 252, 0.08)'
              }
            }}
            onClick={handleRefresh}
            startIcon={<RefreshIcon />}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            sx={{ 
              bgcolor: '#3b82f6',
              '&:hover': {
                bgcolor: '#2563eb'
              }
            }}
            onClick={handleCreateSession}
            startIcon={<AddIcon />}
          >
            New Session
          </Button>
        </Box>
      </Box>

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box>
          {sessions.length > 0 ? (
            <Grid container spacing={3}>
              {sessions.map((session) => (
                <Grid item xs={12} md={6} lg={4} key={session.session_id}>
                  <Card sx={{ 
                    bgcolor: '#1e293b', 
                    color: '#f8fafc',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                    border: '1px solid #334155'
                  }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="h6" sx={{ color: '#f8fafc' }}>{session.name}</Typography>
                        <Chip 
                          label={session.status} 
                          sx={{
                            bgcolor: 
                              session.status === 'active' ? '#4ade80' : 
                              session.status === 'completed' ? '#60a5fa' : 
                              session.status === 'paused' ? '#fbbf24' : '#f87171',
                            color: 
                              session.status === 'active' ? '#022c22' : 
                              session.status === 'completed' ? '#172554' : 
                              session.status === 'paused' ? '#451a03' : '#450a0a',
                            fontWeight: 'bold'
                          }}
                          size="small" 
                        />
                      </Box>
                      
                      <Typography variant="body2" sx={{ color: '#cbd5e1' }} gutterBottom>
                        {session.description}
                      </Typography>
                      
                      <Divider sx={{ my: 1, borderColor: '#334155' }} />
                      
                      <Typography variant="body2" sx={{ color: '#e2e8f0', mb: 0.5 }}>
                        <Box component="span" sx={{ color: '#94a3b8', fontWeight: 'bold', display: 'inline-block', width: 120 }}>Exchange:</Box> {session.exchange}
                      </Typography>
                      
                      <Typography variant="body2" sx={{ color: '#e2e8f0', mb: 0.5 }}>
                        <Box component="span" sx={{ color: '#94a3b8', fontWeight: 'bold', display: 'inline-block', width: 120 }}>Symbols:</Box> {session.symbols.join(', ')}
                      </Typography>
                      
                      <Typography variant="body2" sx={{ color: '#e2e8f0', mb: 0.5 }}>
                        <Box component="span" sx={{ color: '#94a3b8', fontWeight: 'bold', display: 'inline-block', width: 120 }}>Strategy:</Box> {session.strategy}
                      </Typography>
                      
                      <Typography variant="body2" sx={{ color: '#e2e8f0', mb: 0.5 }}>
                        <Box component="span" sx={{ color: '#94a3b8', fontWeight: 'bold', display: 'inline-block', width: 120 }}>Initial Capital:</Box> {formatCurrency(session.initial_capital)}
                      </Typography>
                      
                      {session.performance_metrics && (
                        <Box sx={{ mt: 2, p: 1, bgcolor: '#334155', borderRadius: 1 }}>
                          <Typography variant="subtitle2" sx={{ color: '#94a3b8', fontWeight: 'bold', mb: 1 }}>
                            Performance Metrics
                          </Typography>
                          
                          <Typography variant="body2" sx={{ color: '#e2e8f0', mb: 0.5 }}>
                            <Box component="span" sx={{ color: '#94a3b8', fontWeight: 'bold', display: 'inline-block', width: 120 }}>Total Return:</Box> 
                            <Box component="span" sx={{ 
                              color: session.performance_metrics.total_return > 0 ? '#4ade80' : 
                                     session.performance_metrics.total_return < 0 ? '#f87171' : '#fbbf24'
                            }}>
                              {formatPercentage(session.performance_metrics.total_return)}
                            </Box>
                          </Typography>
                          
                          <Typography variant="body2" sx={{ color: '#e2e8f0', mb: 0.5 }}>
                            <Box component="span" sx={{ color: '#94a3b8', fontWeight: 'bold', display: 'inline-block', width: 120 }}>Sharpe Ratio:</Box> {session.performance_metrics.sharpe_ratio.toFixed(2)}
                          </Typography>
                          
                          <Typography variant="body2" sx={{ color: '#e2e8f0', mb: 0.5 }}>
                            <Box component="span" sx={{ color: '#94a3b8', fontWeight: 'bold', display: 'inline-block', width: 120 }}>Max Drawdown:</Box> {formatPercentage(session.performance_metrics.max_drawdown)}
                          </Typography>
                          
                          <Typography variant="body2" sx={{ color: '#e2e8f0', mb: 0.5 }}>
                            <Box component="span" sx={{ color: '#94a3b8', fontWeight: 'bold', display: 'inline-block', width: 120 }}>Win Rate:</Box> {formatPercentage(session.performance_metrics.win_rate)}
                          </Typography>
                          
                          <Typography variant="body2" sx={{ color: '#e2e8f0', mb: 0.5 }}>
                            <Box component="span" sx={{ color: '#94a3b8', fontWeight: 'bold', display: 'inline-block', width: 120 }}>Total Trades:</Box> {session.performance_metrics.total_trades}
                          </Typography>
                        </Box>
                      )}
                    </CardContent>
                    <CardActions sx={{ bgcolor: '#1e293b' }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                        <Button
                          startIcon={<StopIcon />}
                          sx={{ 
                            bgcolor: '#ef4444', 
                            color: 'white',
                            '&:hover': {
                              bgcolor: '#dc2626'
                            },
                            '&.Mui-disabled': {
                              bgcolor: 'rgba(239, 68, 68, 0.5)',
                              color: 'rgba(255, 255, 255, 0.5)'
                            }
                          }}
                          size="small"
                          disabled={session.status !== 'active'}
                          onClick={() => handleStopSession(session.session_id)}
                        >
                          STOP
                        </Button>
                      </Box>
                    </CardActions>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Box sx={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              justifyContent: 'center', 
              p: 4, 
              bgcolor: '#1e293b', 
              borderRadius: 2,
              color: '#f8fafc'
            }}>
              <Typography variant="h6" sx={{ mb: 2 }}>No active paper trading sessions found.</Typography>
              <Button 
                variant="contained"
                sx={{ 
                  bgcolor: '#3b82f6',
                  '&:hover': {
                    bgcolor: '#2563eb'
                  }
                }}
                onClick={handleCreateSession}
                startIcon={<AddIcon />}
              >
                Create New Session
              </Button>
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
};

export default SimpleSessionManagement;
