import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Button,
  Divider,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  Switch,
  FormControlLabel,
} from '@mui/material';
import { 
  Refresh as RefreshIcon,
  PlayArrow as StartIcon, 
  Stop as StopIcon,
  Settings as SettingsIcon,
  BarChart as MetricsIcon,
  Timeline as TimelineIcon,
  Dashboard as DashboardIcon
} from '@mui/icons-material';
import { DataGrid } from '@mui/x-data-grid';
import { Line } from 'react-chartjs-2';
import moment from 'moment';

// API base URL
const API_BASE_URL = '/api/technical-analysis';

const TechnicalAnalysisAdmin = () => {
  // State for health status
  const [healthStatus, setHealthStatus] = useState({
    status: 'loading',
    message: 'Loading health status...',
    timestamp: new Date().toISOString(),
  });
  
  // State for monitoring metrics
  const [monitoringMetrics, setMonitoringMetrics] = useState(null);
  
  // State for orchestrator status
  const [orchestratorStatus, setOrchestratorStatus] = useState({
    status: 'unknown',
    agent_id: null,
    last_update: null,
    symbols_monitored: [],
    timeframes_monitored: [],
    data_source: 'unknown',
    signal_count: 0,
    error_count: 0,
    queue_sizes: {},
  });
  
  // State for loading indicators
  const [isLoading, setIsLoading] = useState({
    health: true,
    metrics: true,
    orchestrator: true,
  });
  
  // State for error messages
  const [errors, setErrors] = useState({
    health: null,
    metrics: null,
    orchestrator: null,
  });
  
  // State for orchestration settings dialog
  const [orchestrationDialog, setOrchestrationDialog] = useState({
    open: false,
    symbols: ['BTC/USD', 'ETH/USD', 'XRP/USD'],
    timeframes: ['1h', '4h', '1d'],
    selectedSymbols: ['BTC/USD'],
    selectedTimeframes: ['1h'],
  });
  
  // State for refresh intervals
  const [refreshIntervals, setRefreshIntervals] = useState({
    health: 30000, // 30 seconds
    metrics: 60000, // 1 minute
    orchestrator: 15000, // 15 seconds
  });
  
  // State for auto-refresh
  const [autoRefresh, setAutoRefresh] = useState({
    health: true,
    metrics: true,
    orchestrator: true,
  });
  
  // Handle status color
  const getStatusColor = (status) => {
    switch(status.toLowerCase()) {
      case 'healthy':
      case 'running':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'unhealthy':
      case 'stopped':
        return 'error';
      default:
        return 'default';
    }
  };
  
  // Format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    return moment(timestamp).format('YYYY-MM-DD HH:mm:ss');
  };
  
  // Load health status
  const loadHealthStatus = async () => {
    try {
      setIsLoading(prev => ({ ...prev, health: true }));
      const response = await axios.get(`${API_BASE_URL}/monitoring/health`);
      setHealthStatus(response.data);
      setErrors(prev => ({ ...prev, health: null }));
    } catch (error) {
      console.error('Error loading health status:', error);
      setErrors(prev => ({ ...prev, health: error.message }));
    } finally {
      setIsLoading(prev => ({ ...prev, health: false }));
    }
  };
  
  // Load monitoring metrics
  const loadMonitoringMetrics = async () => {
    try {
      setIsLoading(prev => ({ ...prev, metrics: true }));
      const response = await axios.get(`${API_BASE_URL}/monitoring/metrics`);
      setMonitoringMetrics(response.data);
      setErrors(prev => ({ ...prev, metrics: null }));
    } catch (error) {
      console.error('Error loading monitoring metrics:', error);
      setErrors(prev => ({ ...prev, metrics: error.message }));
    } finally {
      setIsLoading(prev => ({ ...prev, metrics: false }));
    }
  };
  
  // Load orchestrator status
  const loadOrchestratorStatus = async () => {
    try {
      setIsLoading(prev => ({ ...prev, orchestrator: true }));
      const response = await axios.get(`${API_BASE_URL}/orchestrator/status`);
      setOrchestratorStatus(response.data);
      setErrors(prev => ({ ...prev, orchestrator: null }));
    } catch (error) {
      console.error('Error loading orchestrator status:', error);
      setErrors(prev => ({ ...prev, orchestrator: error.message }));
    } finally {
      setIsLoading(prev => ({ ...prev, orchestrator: false }));
    }
  };
  
  // Start orchestration
  const startOrchestration = async () => {
    try {
      const { selectedSymbols, selectedTimeframes } = orchestrationDialog;
      
      if (selectedSymbols.length === 0 || selectedTimeframes.length === 0) {
        alert('Please select at least one symbol and timeframe');
        return;
      }
      
      setIsLoading(prev => ({ ...prev, orchestrator: true }));
      
      const response = await axios.post(`${API_BASE_URL}/orchestrator/start`, null, {
        params: {
          symbols: selectedSymbols,
          timeframes: selectedTimeframes,
        }
      });
      
      // Close dialog and refresh status
      setOrchestrationDialog(prev => ({ ...prev, open: false }));
      loadOrchestratorStatus();
      
    } catch (error) {
      console.error('Error starting orchestration:', error);
      alert(`Failed to start orchestration: ${error.message}`);
    } finally {
      setIsLoading(prev => ({ ...prev, orchestrator: false }));
    }
  };
  
  // Stop orchestration
  const stopOrchestration = async () => {
    try {
      setIsLoading(prev => ({ ...prev, orchestrator: true }));
      
      const response = await axios.post(`${API_BASE_URL}/orchestrator/stop`);
      
      // Refresh status
      loadOrchestratorStatus();
      
    } catch (error) {
      console.error('Error stopping orchestration:', error);
      alert(`Failed to stop orchestration: ${error.message}`);
    } finally {
      setIsLoading(prev => ({ ...prev, orchestrator: false }));
    }
  };
  
  // Handle orchestration dialog open
  const handleOpenOrchestrationDialog = () => {
    setOrchestrationDialog(prev => ({ ...prev, open: true }));
  };
  
  // Handle orchestration dialog close
  const handleCloseOrchestrationDialog = () => {
    setOrchestrationDialog(prev => ({ ...prev, open: false }));
  };
  
  // Handle symbol selection change
  const handleSymbolsChange = (event) => {
    setOrchestrationDialog(prev => ({
      ...prev,
      selectedSymbols: event.target.value,
    }));
  };
  
  // Handle timeframe selection change
  const handleTimeframesChange = (event) => {
    setOrchestrationDialog(prev => ({
      ...prev,
      selectedTimeframes: event.target.value,
    }));
  };
  
  // Handle auto-refresh toggle
  const handleAutoRefreshToggle = (key) => {
    setAutoRefresh(prev => ({
      ...prev,
      [key]: !prev[key],
    }));
  };
  
  // Setup auto-refresh intervals
  useEffect(() => {
    // Setup intervals if auto-refresh is enabled
    const intervals = {};
    
    if (autoRefresh.health) {
      intervals.health = setInterval(loadHealthStatus, refreshIntervals.health);
    }
    
    if (autoRefresh.metrics) {
      intervals.metrics = setInterval(loadMonitoringMetrics, refreshIntervals.metrics);
    }
    
    if (autoRefresh.orchestrator) {
      intervals.orchestrator = setInterval(loadOrchestratorStatus, refreshIntervals.orchestrator);
    }
    
    // Cleanup intervals on unmount or when settings change
    return () => {
      Object.values(intervals).forEach(interval => clearInterval(interval));
    };
  }, [
    autoRefresh.health, autoRefresh.metrics, autoRefresh.orchestrator,
    refreshIntervals.health, refreshIntervals.metrics, refreshIntervals.orchestrator,
  ]);
  
  // Initial data load
  useEffect(() => {
    loadHealthStatus();
    loadMonitoringMetrics();
    loadOrchestratorStatus();
  }, []);
  
  // Render health status card
  const renderHealthStatusCard = () => (
    <Card>
      <CardHeader 
        title="System Health" 
        action={
          <Box display="flex" alignItems="center">
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh.health}
                  onChange={() => handleAutoRefreshToggle('health')}
                  color="primary"
                  size="small"
                />
              }
              label="Auto-refresh"
            />
            <IconButton 
              onClick={loadHealthStatus} 
              disabled={isLoading.health}
            >
              <RefreshIcon />
            </IconButton>
          </Box>
        }
      />
      <Divider />
      <CardContent>
        {isLoading.health ? (
          <Box display="flex" justifyContent="center" p={2}>
            <CircularProgress size={40} />
          </Box>
        ) : errors.health ? (
          <Alert severity="error">{errors.health}</Alert>
        ) : (
          <Box>
            <Box display="flex" alignItems="center" mb={2}>
              <Chip 
                label={healthStatus.status.toUpperCase()} 
                color={getStatusColor(healthStatus.status)} 
                sx={{ mr: 2 }}
              />
              <Typography variant="body1">{healthStatus.message}</Typography>
            </Box>
            <Typography variant="caption" color="textSecondary">
              Last updated: {formatTimestamp(healthStatus.timestamp)}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
  
  // Render orchestrator status card
  const renderOrchestratorStatusCard = () => (
    <Card>
      <CardHeader 
        title="Technical Analysis Orchestration" 
        action={
          <Box display="flex" alignItems="center">
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh.orchestrator}
                  onChange={() => handleAutoRefreshToggle('orchestrator')}
                  color="primary"
                  size="small"
                />
              }
              label="Auto-refresh"
            />
            <IconButton 
              onClick={loadOrchestratorStatus}
              disabled={isLoading.orchestrator}
            >
              <RefreshIcon />
            </IconButton>
          </Box>
        }
      />
      <Divider />
      <CardContent>
        {isLoading.orchestrator ? (
          <Box display="flex" justifyContent="center" p={2}>
            <CircularProgress size={40} />
          </Box>
        ) : errors.orchestrator ? (
          <Alert severity="error">{errors.orchestrator}</Alert>
        ) : (
          <Box>
            <Box display="flex" alignItems="center" mb={3} justifyContent="space-between">
              <Box display="flex" alignItems="center">
                <Chip 
                  label={orchestratorStatus.status.toUpperCase()} 
                  color={getStatusColor(orchestratorStatus.status)} 
                  sx={{ mr: 2 }}
                />
                <Typography variant="body1">
                  Agent ID: {orchestratorStatus.agent_id ? 
                    orchestratorStatus.agent_id.toString().substring(0, 8) : 'N/A'}
                </Typography>
              </Box>
              <Box>
                {orchestratorStatus.status === 'running' ? (
                  <Button
                    variant="outlined"
                    color="error"
                    startIcon={<StopIcon />}
                    onClick={stopOrchestration}
                    disabled={isLoading.orchestrator}
                  >
                    Stop
                  </Button>
                ) : (
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<StartIcon />}
                    onClick={handleOpenOrchestrationDialog}
                    disabled={isLoading.orchestrator}
                  >
                    Start
                  </Button>
                )}
              </Box>
            </Box>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
                  <Typography variant="subtitle2" gutterBottom>Monitoring</Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="Symbols" 
                        secondary={orchestratorStatus.symbols_monitored.join(', ') || 'None'} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Timeframes" 
                        secondary={orchestratorStatus.timeframes_monitored.join(', ') || 'None'} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Data Source" 
                        secondary={orchestratorStatus.data_source} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Last Update" 
                        secondary={formatTimestamp(orchestratorStatus.last_update)} 
                      />
                    </ListItem>
                  </List>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
                  <Typography variant="subtitle2" gutterBottom>Statistics</Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="Signal Count" 
                        secondary={orchestratorStatus.signal_count} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Error Count" 
                        secondary={orchestratorStatus.error_count} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Decision Queue" 
                        secondary={orchestratorStatus.queue_sizes?.decision || 0} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Visualization Queue" 
                        secondary={orchestratorStatus.queue_sizes?.visualization || 0} 
                      />
                    </ListItem>
                  </List>
                </Paper>
              </Grid>
            </Grid>
          </Box>
        )}
      </CardContent>
    </Card>
  );
  
  // Render monitoring metrics
  const renderMonitoringMetricsCard = () => (
    <Card>
      <CardHeader 
        title="Performance Metrics" 
        action={
          <Box display="flex" alignItems="center">
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh.metrics}
                  onChange={() => handleAutoRefreshToggle('metrics')}
                  color="primary"
                  size="small"
                />
              }
              label="Auto-refresh"
            />
            <IconButton 
              onClick={loadMonitoringMetrics}
              disabled={isLoading.metrics}
            >
              <RefreshIcon />
            </IconButton>
          </Box>
        }
      />
      <Divider />
      <CardContent>
        {isLoading.metrics ? (
          <Box display="flex" justifyContent="center" p={2}>
            <CircularProgress size={40} />
          </Box>
        ) : errors.metrics ? (
          <Alert severity="error">{errors.metrics}</Alert>
        ) : !monitoringMetrics ? (
          <Typography variant="body2" color="textSecondary">No metrics data available</Typography>
        ) : (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              System Metrics
            </Typography>
            
            <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    <TableCell>Latest Value</TableCell>
                    <TableCell>Average</TableCell>
                    <TableCell>Unit</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {monitoringMetrics.metrics && Object.entries(monitoringMetrics.metrics).map(([key, metric]) => (
                    <TableRow key={key}>
                      <TableCell>{metric.name}</TableCell>
                      <TableCell>{metric.latest !== null ? metric.latest.toFixed(2) : 'N/A'}</TableCell>
                      <TableCell>{metric.average !== null ? metric.average.toFixed(2) : 'N/A'}</TableCell>
                      <TableCell>{metric.unit}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            
            <Typography variant="subtitle2" gutterBottom>
              Component Health
            </Typography>
            
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Component</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Message</TableCell>
                    <TableCell>Last Check</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {monitoringMetrics.health && Object.entries(monitoringMetrics.health).map(([key, health]) => (
                    <TableRow key={key}>
                      <TableCell>{health.component}</TableCell>
                      <TableCell>
                        <Chip 
                          label={health.status} 
                          color={getStatusColor(health.status)} 
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{health.message}</TableCell>
                      <TableCell>{formatTimestamp(health.last_check)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            
            {monitoringMetrics.alerts && monitoringMetrics.alerts.length > 0 && (
              <Box mt={3}>
                <Typography variant="subtitle2" gutterBottom>
                  Active Alerts
                </Typography>
                
                {monitoringMetrics.alerts.map((alert, index) => (
                  <Alert 
                    key={index} 
                    severity={alert.severity === 'high' ? 'error' : 'warning'}
                    sx={{ mb: 1 }}
                  >
                    {alert.message}
                  </Alert>
                ))}
              </Box>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
  
  // Render orchestration settings dialog
  const renderOrchestrationDialog = () => (
    <Dialog
      open={orchestrationDialog.open}
      onClose={handleCloseOrchestrationDialog}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>Start Technical Analysis Orchestration</DialogTitle>
      <DialogContent>
        <Box mt={2}>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Symbols</InputLabel>
            <Select
              multiple
              value={orchestrationDialog.selectedSymbols}
              onChange={handleSymbolsChange}
              renderValue={(selected) => selected.join(', ')}
            >
              {orchestrationDialog.symbols.map((symbol) => (
                <MenuItem key={symbol} value={symbol}>
                  {symbol}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormControl fullWidth>
            <InputLabel>Timeframes</InputLabel>
            <Select
              multiple
              value={orchestrationDialog.selectedTimeframes}
              onChange={handleTimeframesChange}
              renderValue={(selected) => selected.join(', ')}
            >
              {orchestrationDialog.timeframes.map((timeframe) => (
                <MenuItem key={timeframe} value={timeframe}>
                  {timeframe}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleCloseOrchestrationDialog}>Cancel</Button>
        <Button 
          onClick={startOrchestration} 
          variant="contained" 
          color="primary"
          disabled={
            orchestrationDialog.selectedSymbols.length === 0 || 
            orchestrationDialog.selectedTimeframes.length === 0
          }
        >
          Start Monitoring
        </Button>
      </DialogActions>
    </Dialog>
  );
  
  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>
        Technical Analysis Admin
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        Monitor and control the Technical Analysis system. View health status, performance metrics, and manage the analysis orchestration.
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          {renderHealthStatusCard()}
        </Grid>
        
        <Grid item xs={12} md={8}>
          {renderOrchestratorStatusCard()}
        </Grid>
        
        <Grid item xs={12}>
          {renderMonitoringMetricsCard()}
        </Grid>
      </Grid>
      
      {renderOrchestrationDialog()}
    </Box>
  );
};

export default TechnicalAnalysisAdmin;
