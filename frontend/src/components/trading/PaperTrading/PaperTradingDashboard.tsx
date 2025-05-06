import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useNotification } from '../../../components/common/NotificationSystem';
import { API_BASE_URL } from '../../../config';
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
  Paper
} from '@mui/material';

// Define the Session interface
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

const PaperTradingDashboard: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const { showNotification } = useNotification();
  const navigate = useNavigate();
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch the session details when the component mounts
  useEffect(() => {
    const fetchSession = async () => {
      if (!sessionId) return;
      
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE_URL}/paper-trading/sessions/${sessionId}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch session: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        setSession(data);
      } catch (error: any) {
        console.error('Error fetching session:', error);
        setError(error.message || 'Failed to fetch session details');
      } finally {
        setLoading(false);
      }
    };
    
    fetchSession();
  }, [sessionId]);

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
  const handleStopSession = async () => {
    if (!sessionId) return;

    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/paper-trading/sessions/${sessionId}/stop`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Failed to stop session: ${response.status} ${response.statusText}`);
      }
      
      // Update the session status locally
      if (session) {
        setSession({
          ...session,
          status: 'completed',
          end_time: new Date().toISOString()
        });
      }
      
      showNotification({
        title: 'Session Stopped',
        message: 'Paper trading session has been stopped',
        type: 'info'
      });
    } catch (error: any) {
      console.error('Error stopping session:', error);
      showNotification({
        title: 'Error',
        message: `Failed to stop paper trading session: ${error.message}`,
        type: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle back to sessions
  const handleBackToSessions = () => {
    navigate('/paper-trading');
  };

  if (loading) {
    return (
      <Box sx={{ p: 4, display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: 2 }}>
        <Typography variant="h5">Paper Trading Dashboard</Typography>
        <CircularProgress />
        <Typography variant="body1">Loading session details...</Typography>
      </Box>
    );
  }

  if (error || !session) {
    return (
      <Box sx={{ p: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h5">Paper Trading Dashboard</Typography>
          <Button variant="outlined" onClick={handleBackToSessions}>
            Back to Sessions
          </Button>
        </Box>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" color="error" gutterBottom>
              {error || 'No session selected or session not found.'}
            </Typography>
            <Button variant="contained" onClick={handleBackToSessions}>
              View All Sessions
            </Button>
          </CardContent>
        </Card>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" gutterBottom>Paper Trading Dashboard</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="subtitle1">
              Session: {session.name} ({session.session_id.substring(0, 8)}...)
            </Typography>
            <Chip 
              label={session.status} 
              color={session.status === 'active' ? 'success' : 'default'}
              size="small"
            />
          </Box>
        </Box>
        <Box>
          {session.status === 'active' && (
            <Button 
              variant="contained" 
              color="error" 
              onClick={handleStopSession}
              disabled={loading}
              sx={{ mr: 2 }}
            >
              Stop Session
            </Button>
          )}
          <Button variant="outlined" onClick={handleBackToSessions}>
            Back to Sessions
          </Button>
        </Box>
      </Box>

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
                <strong>Started:</strong> {new Date(session.start_time).toLocaleString()}
              </Typography>
              {session.end_time && (
                <Typography variant="body2">
                  <strong>Ended:</strong> {new Date(session.end_time).toLocaleString()}
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
    </Box>
  );
};

export default PaperTradingDashboard;
