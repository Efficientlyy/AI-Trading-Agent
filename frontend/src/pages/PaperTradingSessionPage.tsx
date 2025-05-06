import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Divider,
  CircularProgress,
  Chip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import StopIcon from '@mui/icons-material/Stop';
import RefreshIcon from '@mui/icons-material/Refresh';
import { API_BASE_URL } from '../config';

// Session interface
interface Session {
  session_id: string;
  name: string;
  description?: string;
  exchange: string;
  symbols: string[];
  strategy: string;
  initial_capital: number;
  status: string;
  start_time: string;
  end_time?: string;
  uptime_seconds?: number;
  performance_metrics?: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
  };
  trades?: any[];
  positions?: any[];
}

// Position interface
interface Position {
  symbol: string;
  side: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
}

// Trade interface
interface Trade {
  id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  timestamp: string;
  profit?: number;
  fee?: number;
}

const PaperTradingSessionPage: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  // Fetch session details
  const fetchSessionDetails = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // First try to get the session details
      const response = await fetch(`${API_BASE_URL}/paper-trading/sessions/${sessionId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch session details: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Session details:', data);
      setSession(data);
      
      // Then try to get the session status for more up-to-date info
      const statusResponse = await fetch(`${API_BASE_URL}/paper-trading/status?session_id=${sessionId}`);
      
      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        console.log('Session status:', statusData);
        
        // Merge the status data with the session data
        setSession(prevSession => {
          if (!prevSession) return null;
          
          return {
            ...prevSession,
            status: statusData.status || prevSession.status,
            performance_metrics: statusData.performance_metrics || prevSession.performance_metrics,
            trades: statusData.recent_trades || prevSession.trades,
            positions: statusData.current_portfolio?.positions || prevSession.positions
          };
        });
      }
    } catch (error: any) {
      console.error('Error fetching session details:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Stop the session
  const handleStopSession = async () => {
    try {
      setLoading(true);
      
      const response = await fetch(`${API_BASE_URL}/paper-trading/stop/${sessionId}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Failed to stop session: ${response.status} ${response.statusText}`);
      }
      
      // Refresh the session details
      await fetchSessionDetails();
    } catch (error: any) {
      console.error('Error stopping session:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  // Format date
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  // Format uptime
  const formatUptime = (seconds?: number) => {
    if (!seconds) return 'N/A';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;
    
    return `${hours}h ${minutes}m ${remainingSeconds}s`;
  };

  // Load session details on component mount
  useEffect(() => {
    if (sessionId) {
      fetchSessionDetails();
    }
  }, [sessionId]);

  // Handle refresh
  const handleRefresh = () => {
    fetchSessionDetails();
  };

  // Handle back
  const handleBack = () => {
    navigate('/paper-trading');
  };

  return (
    <Box className="paper-trading-session-page">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Button 
          startIcon={<ArrowBackIcon />} 
          onClick={handleBack}
        >
          Back to Sessions
        </Button>
        
        <Box>
          <Button 
            startIcon={<RefreshIcon />} 
            onClick={handleRefresh}
            disabled={loading}
            sx={{ mr: 1 }}
          >
            Refresh
          </Button>
          
          {session && (session.status === 'active' || session.status === 'running') && (
            <Button 
              startIcon={<StopIcon />} 
              color="error" 
              onClick={handleStopSession}
              disabled={loading}
            >
              Stop Session
            </Button>
          )}
        </Box>
      </Box>

      {loading && !session && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Card variant="outlined" sx={{ bgcolor: 'error.light', mb: 3 }}>
          <CardContent>
            <Typography variant="h6" color="error">
              Error
            </Typography>
            <Typography variant="body1">
              {error}
            </Typography>
          </CardContent>
        </Card>
      )}

      {session && (
        <>
          <Card variant="outlined" sx={{ mb: 3 }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Typography variant="h5" component="h1" gutterBottom>
                  {session.name}
                </Typography>
                <Chip 
                  label={session.status} 
                  color={(session.status === 'active' || session.status === 'running') ? 'success' : 'default'}
                />
              </Box>
              
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 3, mt: 1 }}>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Session ID
                  </Typography>
                  <Typography variant="body1">
                    {session.session_id.substring(0, 8)}...
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Exchange
                  </Typography>
                  <Typography variant="body1">
                    {session.exchange}
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Strategy
                  </Typography>
                  <Typography variant="body1">
                    {session.strategy}
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Initial Capital
                  </Typography>
                  <Typography variant="body1">
                    {formatCurrency(session.initial_capital)}
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Started
                  </Typography>
                  <Typography variant="body1">
                    {formatDate(session.start_time)}
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Uptime
                  </Typography>
                  <Typography variant="body1">
                    {formatUptime(session.uptime_seconds)}
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Symbols
                  </Typography>
                  <Typography variant="body1">
                    {session.symbols.join(', ')}
                  </Typography>
                </Box>
                
                {session.end_time && (
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Ended
                    </Typography>
                    <Typography variant="body1">
                      {formatDate(session.end_time)}
                    </Typography>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>

          {session.performance_metrics && (
            <Card variant="outlined" sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Metrics
                </Typography>
                
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', sm: 'repeat(3, 1fr)', md: 'repeat(6, 1fr)' }, gap: 3 }}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Total Return
                    </Typography>
                    <Typography variant="body1" color={session.performance_metrics.total_return >= 0 ? 'success.main' : 'error.main'}>
                      {formatPercentage(session.performance_metrics.total_return)}
                    </Typography>
                  </Box>
                  
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Sharpe Ratio
                    </Typography>
                    <Typography variant="body1">
                      {session.performance_metrics.sharpe_ratio.toFixed(2)}
                    </Typography>
                  </Box>
                  
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Max Drawdown
                    </Typography>
                    <Typography variant="body1" color="error.main">
                      {formatPercentage(session.performance_metrics.max_drawdown)}
                    </Typography>
                  </Box>
                  
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Win Rate
                    </Typography>
                    <Typography variant="body1">
                      {formatPercentage(session.performance_metrics.win_rate)}
                    </Typography>
                  </Box>
                  
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Total Trades
                    </Typography>
                    <Typography variant="body1">
                      {session.performance_metrics.total_trades}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}

          {session.positions && session.positions.length > 0 && (
            <Card variant="outlined" sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Current Positions
                </Typography>
                
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Side</TableCell>
                        <TableCell>Quantity</TableCell>
                        <TableCell>Entry Price</TableCell>
                        <TableCell>Current Price</TableCell>
                        <TableCell>Unrealized P&L</TableCell>
                        <TableCell>Realized P&L</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {session.positions.map((position: Position) => (
                        <TableRow key={position.symbol}>
                          <TableCell>{position.symbol}</TableCell>
                          <TableCell>{position.side}</TableCell>
                          <TableCell>{position.quantity}</TableCell>
                          <TableCell>{formatCurrency(position.entry_price)}</TableCell>
                          <TableCell>{formatCurrency(position.current_price)}</TableCell>
                          <TableCell 
                            sx={{ 
                              color: position.unrealized_pnl >= 0 ? 'success.main' : 'error.main' 
                            }}
                          >
                            {formatCurrency(position.unrealized_pnl)}
                          </TableCell>
                          <TableCell 
                            sx={{ 
                              color: position.realized_pnl >= 0 ? 'success.main' : 'error.main' 
                            }}
                          >
                            {formatCurrency(position.realized_pnl)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}

          {session.trades && session.trades.length > 0 && (
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recent Trades
                </Typography>
                
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>ID</TableCell>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Side</TableCell>
                        <TableCell>Quantity</TableCell>
                        <TableCell>Price</TableCell>
                        <TableCell>Timestamp</TableCell>
                        <TableCell>Profit</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {session.trades.map((trade: Trade) => (
                        <TableRow key={trade.id}>
                          <TableCell>{trade.id.substring(0, 8)}...</TableCell>
                          <TableCell>{trade.symbol}</TableCell>
                          <TableCell>{trade.side}</TableCell>
                          <TableCell>{trade.quantity}</TableCell>
                          <TableCell>{formatCurrency(trade.price)}</TableCell>
                          <TableCell>{formatDate(trade.timestamp)}</TableCell>
                          <TableCell 
                            sx={{ 
                              color: (trade.profit || 0) >= 0 ? 'success.main' : 'error.main' 
                            }}
                          >
                            {trade.profit !== undefined ? formatCurrency(trade.profit) : 'N/A'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </Box>
  );
};

export default PaperTradingSessionPage;
