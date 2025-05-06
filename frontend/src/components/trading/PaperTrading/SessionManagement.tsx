import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Chip,
  CircularProgress,
  Divider,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Grid
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import StopIcon from '@mui/icons-material/Stop';
import AddIcon from '@mui/icons-material/Add';
import InfoIcon from '@mui/icons-material/Info';
import { API_BASE_URL } from '../../../config';

// Session interface
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

const SessionManagement: React.FC = () => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [stoppedSessions, setStoppedSessions] = useState<Set<string>>(new Set());
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [detailsOpen, setDetailsOpen] = useState<boolean>(false);
  const navigate = useNavigate();

  // Fetch sessions on component mount
  useEffect(() => {
    fetchSessions();
  }, []);

  // Fetch sessions from API
  const fetchSessions = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/paper-trading/sessions`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch sessions: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Fetched sessions:', data);
      
      // Check if data has a sessions property (which is what our backend returns)
      if (data && data.sessions && Array.isArray(data.sessions)) {
        setSessions(data.sessions);
      } else {
        console.error('Invalid response format:', data);
        setSessions([]);
      }
      
      // Clear stopped sessions set
      setStoppedSessions(new Set());
    } catch (error: any) {
      console.error('Error fetching sessions:', error);
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

  // Format date
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  // Handle stop session
  const handleStopSession = async (sessionId: string) => {
    try {
      // If we've already tried to stop this session, don't try again
      if (stoppedSessions.has(sessionId)) {
        console.log(`Session ${sessionId} already stopped`);
        return;
      }

      setLoading(true);
      
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
      
      // Add this session to the set of stopped sessions
      setStoppedSessions(prev => {
        const newSet = new Set(prev);
        newSet.add(sessionId);
        return newSet;
      });
      
      // Now make the API call to actually stop the session on the backend
      const response = await fetch(`${API_BASE_URL}/paper-trading/stop/${sessionId}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        console.error(`Backend error stopping session: ${response.status} ${response.statusText}`);
        // We don't throw here because we've already updated the UI
      } else {
        console.log(`Successfully stopped session ${sessionId} on the backend`);
      }
    } catch (error: any) {
      console.error('Error stopping session:', error);
    } finally {
      setLoading(false);
    }
  };

  // Render session card
  const renderSessionCard = (session: Session) => (
    <Box key={session.session_id} sx={{ width: { xs: '100%', md: '45%', lg: '30%' } }}>
      <Card variant="outlined" sx={{ height: '100%', bgcolor: 'background.paper' }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <Typography variant="h6" component="h3" gutterBottom>
              {session.name}
            </Typography>
            <Chip 
              label={session.status} 
              color={session.status === 'active' ? 'success' : 'default'}
              size="small"
            />
          </Box>
          
          <Typography variant="body2" color="text.secondary" gutterBottom>
            {session.description || 'No description'}
          </Typography>
          
          <Box sx={{ mt: 2 }}>
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
            <Typography variant="body2">
              <strong>Started:</strong> {new Date(session.start_time).toLocaleString()}
            </Typography>
            {session.end_time && (
              <Typography variant="body2">
                <strong>Ended:</strong> {new Date(session.end_time).toLocaleString()}
              </Typography>
            )}
          </Box>
          
          {session.performance_metrics && (
            <>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" gutterBottom>
                Performance Metrics
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
                <Box sx={{ width: '50%', mb: 1 }}>
                  <Typography variant="body2">
                    <strong>Total Return:</strong> {formatPercentage(session.performance_metrics.total_return)}
                  </Typography>
                </Box>
                <Box sx={{ width: '50%', mb: 1 }}>
                  <Typography variant="body2">
                    <strong>Sharpe Ratio:</strong> {session.performance_metrics.sharpe_ratio.toFixed(2)}
                  </Typography>
                </Box>
                <Box sx={{ width: '50%', mb: 1 }}>
                  <Typography variant="body2">
                    <strong>Max Drawdown:</strong> {formatPercentage(session.performance_metrics.max_drawdown)}
                  </Typography>
                </Box>
                <Box sx={{ width: '50%', mb: 1 }}>
                  <Typography variant="body2">
                    <strong>Win Rate:</strong> {formatPercentage(session.performance_metrics.win_rate)}
                  </Typography>
                </Box>
                <Box sx={{ width: '100%' }}>
                  <Typography variant="body2">
                    <strong>Total Trades:</strong> {session.performance_metrics.total_trades}
                  </Typography>
                </Box>
              </Box>
            </>
          )}
        </CardContent>
        
        <CardActions>
          {(session.status === 'running' || session.status === 'active') && (
            <Button 
              startIcon={<StopIcon />} 
              color="error" 
              onClick={() => handleStopSession(session.session_id)}
              disabled={loading || stoppedSessions.has(session.session_id)}
            >
              Stop
            </Button>
          )}
          <Button 
            startIcon={<InfoIcon />}
            onClick={() => {
              setSelectedSession(session);
              setDetailsOpen(true);
            }}
            disabled={loading}
          >
            View Details
          </Button>
        </CardActions>
      </Card>
    </Box>
  );

  return (
    <Box className="session-management">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          Paper Trading Sessions
        </Typography>
        <Box>
          <Tooltip title="Refresh">
            <IconButton onClick={handleRefresh} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          <Button 
            variant="contained" 
            color="primary" 
            startIcon={<AddIcon />} 
            onClick={handleCreateSession}
            sx={{ ml: 1 }}
          >
            New Session
          </Button>
        </Box>
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {!loading && sessions.length === 0 && (
        <Card variant="outlined" sx={{ bgcolor: 'background.paper' }}>
          <CardContent>
            <Typography variant="body1" align="center">
              No paper trading sessions found. Create a new session to get started.
            </Typography>
          </CardContent>
        </Card>
      )}

      <Grid container spacing={3}>
        {sessions.map(session => renderSessionCard(session))}
      </Grid>

      {/* Session Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        {selectedSession && (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6">{selectedSession.name}</Typography>
                <Chip 
                  label={selectedSession.status} 
                  color={(selectedSession.status === 'active' || selectedSession.status === 'running') ? 'success' : 'default'}
                  size="small"
                />
              </Box>
            </DialogTitle>
            <DialogContent>
              <TableContainer component={Paper} sx={{ mb: 2 }}>
                <Table>
                  <TableBody>
                    <TableRow>
                      <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                        Session ID
                      </TableCell>
                      <TableCell>{selectedSession.session_id}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                        Exchange
                      </TableCell>
                      <TableCell>{selectedSession.exchange}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                        Symbols
                      </TableCell>
                      <TableCell>{selectedSession.symbols?.join(', ')}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                        Initial Capital
                      </TableCell>
                      <TableCell>{formatCurrency(selectedSession.initial_capital)}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                        Start Time
                      </TableCell>
                      <TableCell>{formatDate(selectedSession.start_time)}</TableCell>
                    </TableRow>
                    {selectedSession.end_time && (
                      <TableRow>
                        <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                          End Time
                        </TableCell>
                        <TableCell>{formatDate(selectedSession.end_time)}</TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>

              {selectedSession.performance_metrics && (
                <>
                  <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
                  <TableContainer component={Paper}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Total Return</TableCell>
                          <TableCell>Sharpe Ratio</TableCell>
                          <TableCell>Max Drawdown</TableCell>
                          <TableCell>Win Rate</TableCell>
                          <TableCell>Total Trades</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>{(selectedSession.performance_metrics.total_return * 100).toFixed(2)}%</TableCell>
                          <TableCell>{selectedSession.performance_metrics.sharpe_ratio.toFixed(2)}</TableCell>
                          <TableCell>{(selectedSession.performance_metrics.max_drawdown * 100).toFixed(2)}%</TableCell>
                          <TableCell>{(selectedSession.performance_metrics.win_rate * 100).toFixed(2)}%</TableCell>
                          <TableCell>{selectedSession.performance_metrics.total_trades}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </>
              )}
            </DialogContent>
            <DialogActions>
              {(selectedSession.status === 'running' || selectedSession.status === 'active') && (
                <Button 
                  startIcon={<StopIcon />} 
                  color="error" 
                  onClick={() => {
                    handleStopSession(selectedSession.session_id);
                    setDetailsOpen(false);
                  }}
                  disabled={loading || stoppedSessions.has(selectedSession.session_id)}
                >
                  Stop Session
                </Button>
              )}
              <Button onClick={() => setDetailsOpen(false)}>Close</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default SessionManagement;
