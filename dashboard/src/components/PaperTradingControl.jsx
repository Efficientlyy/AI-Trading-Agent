import React from 'react';
import SystemDashboard from './SystemDashboard';

const PaperTradingControl = () => {
  // State
  const [configPath, setConfigPath] = useState('config/paper_trading_config.yaml');
  const [duration, setDuration] = useState(60);
  const [interval, setInterval] = useState(1);
  const [status, setStatus] = useState('idle');
  const [loading, setLoading] = useState(false);
  const [statusData, setStatusData] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [refreshTimer, setRefreshTimer] = useState(null);
  const [availableConfigs, setAvailableConfigs] = useState([]);

  // Fetch available config files on component mount
  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/config/list`);
        if (response.data && Array.isArray(response.data.configs)) {
          setAvailableConfigs(response.data.configs.filter(c => c.includes('paper_trading')));
        }
      } catch (error) {
        console.error('Error fetching config files:', error);
        // Fallback to default config
        setAvailableConfigs(['config/paper_trading_config.yaml']);
      }
    };

    fetchConfigs();
  }, []);

  // Fetch status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/paper-trading/status`);
        setStatus(response.data.status);
        setStatusData(response.data);
      } catch (error) {
        console.error('Error fetching status:', error);
      }
    };

    // Initial fetch
    fetchStatus();

    // Set up periodic refresh
    const timer = setInterval(fetchStatus, refreshInterval);
    setRefreshTimer(timer);

    // Clean up on unmount
    return () => {
      if (refreshTimer) {
        clearInterval(refreshTimer);
      }
    };
  }, [refreshInterval]);

  // Handle start button click
  const handleStart = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/paper-trading/start`, {
        config_path: configPath,
        duration_minutes: duration,
        interval_minutes: interval
      });

      setSuccessMessage('Paper trading started successfully');
      setStatus('running');
    } catch (error) {
      console.error('Error starting paper trading:', error);
      setErrorMessage(error.response?.data?.detail || 'Error starting paper trading');
    } finally {
      setLoading(false);
    }
  };

  // Handle stop button click
  const handleStop = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/paper-trading/stop`);
      setSuccessMessage('Paper trading stopping...');
      setStatus('stopping');
    } catch (error) {
      console.error('Error stopping paper trading:', error);
      setErrorMessage(error.response?.data?.detail || 'Error stopping paper trading');
    } finally {
      setLoading(false);
    }
  };

  // Handle refresh click
  const handleRefresh = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/paper-trading/status`);
      setStatus(response.data.status);
      setStatusData(response.data);
      setSuccessMessage('Status refreshed');
    } catch (error) {
      console.error('Error refreshing status:', error);
      setErrorMessage('Error refreshing status');
    }
  };

  // Handle snackbar close
  const handleCloseSnackbar = () => {
    setErrorMessage('');
    setSuccessMessage('');
  };

  // Format portfolio data for display
  const formatPortfolio = (portfolio) => {
    if (!portfolio) return 'No portfolio data available';

    return (
      <Box>
        <Typography variant="subtitle2">Total Value: ${portfolio.total_value?.toFixed(2) || 0}</Typography>
        <Typography variant="subtitle2">Cash: ${portfolio.cash?.toFixed(2) || 0}</Typography>
        <Divider sx={{ my: 1 }} />
        <Typography variant="subtitle2">Positions:</Typography>
        {portfolio.positions ? (
          Object.entries(portfolio.positions).map(([symbol, position]) => (
            <Typography key={symbol} variant="body2">
              {symbol}: {position.quantity} @ ${position.avg_price?.toFixed(2)} 
              (${(position.quantity * position.current_price)?.toFixed(2)})
            </Typography>
          ))
        ) : (
          <Typography variant="body2">No positions</Typography>
        )}
      </Box>
    );
  };

  // Format recent trades for display
  const formatTrades = (trades) => {
    if (!trades || trades.length === 0) return 'No recent trades';

    return (
      <Box>
        {trades.map((trade, index) => (
          <Box key={index} sx={{ mb: 1 }}>
            <Typography variant="body2">
              {trade.timestamp && new Date(trade.timestamp).toLocaleString()}: {trade.action} {trade.symbol} 
              {trade.quantity} @ ${trade.price?.toFixed(2)}
            </Typography>
          </Box>
        ))}
      </Box>
    );
  };

  return (
    <Card>
      <CardHeader 
        title="Paper Trading Control" 
        subheader="Control and monitor paper trading sessions"
      />
      <Divider />
      <CardContent>
        <Grid container spacing={3}>
          {/* Configuration Section */}
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>Configuration</Typography>
                <FormControl fullWidth margin="normal">
                  <InputLabel id="config-select-label">Config File</InputLabel>
                  <Select
                    labelId="config-select-label"
                    value={configPath}
                    onChange={(e) => setConfigPath(e.target.value)}
                    disabled={status === 'running' || status === 'stopping'}
                  >
                    {availableConfigs.map((config) => (
                      <MenuItem key={config} value={config}>{config}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <TextField
                  fullWidth
                  margin="normal"
                  label="Duration (minutes)"
                  type="number"
                  value={duration}
                  onChange={(e) => setDuration(parseInt(e.target.value))}
                  disabled={status === 'running' || status === 'stopping'}
                />
                <TextField
                  fullWidth
                  margin="normal"
                  label="Update Interval (minutes)"
                  type="number"
                  value={interval}
                  onChange={(e) => setInterval(parseInt(e.target.value))}
                  disabled={status === 'running' || status === 'stopping'}
                />
                <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<PlayArrowIcon />}
                    onClick={handleStart}
                    disabled={status === 'running' || status === 'stopping' || loading}
                  >
                    {loading ? <CircularProgress size={24} /> : 'Start Trading'}
                  </Button>
                  <Button
                    variant="contained"
                    color="secondary"
                    startIcon={<StopIcon />}
                    onClick={handleStop}
                    disabled={status !== 'running' || loading}
                  >
                    {loading ? <CircularProgress size={24} /> : 'Stop Trading'}
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Status Section */}
          <Grid item xs={12} md={6}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Status</Typography>
                  <Button
                    startIcon={<RefreshIcon />}
                    onClick={handleRefresh}
                    size="small"
                  >
                    Refresh
                  </Button>
                </Box>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1">
                    Current Status: 
                    <Box component="span" sx={{ 
                      ml: 1, 
                      fontWeight: 'bold',
                      color: status === 'running' ? 'success.main' : 
                             status === 'stopping' ? 'warning.main' :
                             status === 'error' ? 'error.main' : 'text.primary'
                    }}>
                      {status.toUpperCase()}
                    </Box>
                  </Typography>
                  {statusData?.uptime_seconds && (
                    <Typography variant="body2">
                      Uptime: {Math.floor(statusData.uptime_seconds / 60)} min {statusData.uptime_seconds % 60} sec
                    </Typography>
                  )}
                </Box>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1" gutterBottom>Current Portfolio</Typography>
                {statusData?.current_portfolio ? (
                  formatPortfolio(statusData.current_portfolio)
                ) : (
                  <Typography variant="body2">No portfolio data available</Typography>
                )}
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1" gutterBottom>Recent Trades</Typography>
                {statusData?.recent_trades ? (
                  formatTrades(statusData.recent_trades)
                ) : (
                  <Typography variant="body2">No recent trades</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </CardContent>

      {/* Snackbars for success/error messages */}
      <Snackbar
        open={!!errorMessage}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
      >
        <Alert onClose={handleCloseSnackbar} severity="error" sx={{ width: '100%' }}>
          {errorMessage}
        </Alert>
      </Snackbar>
      <Snackbar
        open={!!successMessage}
        autoHideDuration={3000}
        onClose={handleCloseSnackbar}
      >
        <Alert onClose={handleCloseSnackbar} severity="success" sx={{ width: '100%' }}>
          {successMessage}
        </Alert>
      </Snackbar>
    </Card>
  );
};

export default PaperTradingControl;
