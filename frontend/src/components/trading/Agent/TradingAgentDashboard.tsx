import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  Grid as MuiGrid,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Divider,
  CircularProgress,
  SelectChangeEvent,
  Paper,
  Switch,
  FormControlLabel
} from '@mui/material';
import axios from 'axios';
import { API_BASE_URL } from '../../../config';

// Create a Grid component that works with Material UI v5
const Grid = (props: any) => <MuiGrid {...props} />;

// Define types for the trading agent
interface TradingAgentConfig {
  name: string;
  description: string;
  strategies: {
    market_regime: {
      enabled: boolean;
      weight: number;
    };
    ml_strategy: {
      enabled: boolean;
      weight: number;
    };
    portfolio_strategy: {
      enabled: boolean;
      weight: number;
    };
  };
  risk_management: {
    max_position_size: number;
    stop_loss_pct: number;
    take_profit_pct: number;
  };
  symbols: string[];
  initial_capital: number;
}

interface TradingAgentStatus {
  status: 'idle' | 'running' | 'stopped' | 'error';
  error_message?: string;
  uptime_seconds: number;
  total_trades: number;
  profitable_trades: number;
  current_portfolio_value: number;
  pnl_pct: number;
  active_positions: {
    symbol: string;
    quantity: number;
    entry_price: number;
    current_price: number;
    unrealized_pnl: number;
    unrealized_pnl_pct: number;
  }[];
}

const defaultConfig: TradingAgentConfig = {
  name: "Default Trading Agent",
  description: "A trading agent that uses multiple strategies to make trading decisions",
  strategies: {
    market_regime: {
      enabled: true,
      weight: 0.33
    },
    ml_strategy: {
      enabled: true,
      weight: 0.33
    },
    portfolio_strategy: {
      enabled: true,
      weight: 0.34
    }
  },
  risk_management: {
    max_position_size: 0.1,
    stop_loss_pct: 0.05,
    take_profit_pct: 0.1
  },
  symbols: ["BTC/USD", "ETH/USD", "XRP/USD"],
  initial_capital: 10000
};

const TradingAgentDashboard: React.FC = () => {
  const [config, setConfig] = useState<TradingAgentConfig>(defaultConfig);
  const [status, setStatus] = useState<TradingAgentStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isConfigEditing, setIsConfigEditing] = useState(false);
  const [newSymbol, setNewSymbol] = useState("");

  // Fetch agent status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/agent/status`);
        setStatus(response.data);
        setError(null);
      } catch (err) {
        console.error('Error fetching agent status:', err);
        setError('Failed to fetch agent status');
      }
    };

    // Fetch status immediately
    fetchStatus();

    // Set up interval to fetch status every 5 seconds
    const intervalId = setInterval(fetchStatus, 5000);

    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  // Fetch agent configuration
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/agent/config`);
        setConfig(response.data);
        setError(null);
      } catch (err) {
        console.error('Error fetching agent config:', err);
        setError('Failed to fetch agent configuration');
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  // Start the trading agent
  const startAgent = async () => {
    try {
      setLoading(true);
      await axios.post(`${API_BASE_URL}/agent/start`, config);
      // Fetch updated status
      const response = await axios.get(`${API_BASE_URL}/agent/status`);
      setStatus(response.data);
      setError(null);
    } catch (err) {
      console.error('Error starting agent:', err);
      setError('Failed to start trading agent');
    } finally {
      setLoading(false);
    }
  };

  // Stop the trading agent
  const stopAgent = async () => {
    try {
      setLoading(true);
      await axios.post(`${API_BASE_URL}/agent/stop`);
      // Fetch updated status
      const response = await axios.get(`${API_BASE_URL}/agent/status`);
      setStatus(response.data);
      setError(null);
    } catch (err) {
      console.error('Error stopping agent:', err);
      setError('Failed to stop trading agent');
    } finally {
      setLoading(false);
    }
  };

  // Save agent configuration
  const saveConfig = async () => {
    try {
      setLoading(true);
      await axios.post(`${API_BASE_URL}/agent/config`, config);
      setIsConfigEditing(false);
      setError(null);
    } catch (err) {
      console.error('Error saving config:', err);
      setError('Failed to save agent configuration');
    } finally {
      setLoading(false);
    }
  };

  // Handle strategy weight change
  const handleStrategyWeightChange = (strategy: keyof typeof config.strategies, value: number) => {
    setConfig(prev => ({
      ...prev,
      strategies: {
        ...prev.strategies,
        [strategy]: {
          ...prev.strategies[strategy],
          weight: value
        }
      }
    }));
  };

  // Handle strategy enabled/disabled
  const handleStrategyEnabledChange = (strategy: keyof typeof config.strategies, enabled: boolean) => {
    setConfig(prev => ({
      ...prev,
      strategies: {
        ...prev.strategies,
        [strategy]: {
          ...prev.strategies[strategy],
          enabled
        }
      }
    }));
  };

  // Handle risk management change
  const handleRiskManagementChange = (field: keyof typeof config.risk_management, value: number) => {
    setConfig(prev => ({
      ...prev,
      risk_management: {
        ...prev.risk_management,
        [field]: value
      }
    }));
  };

  // Add a new symbol
  const addSymbol = () => {
    if (newSymbol && !config.symbols.includes(newSymbol)) {
      setConfig(prev => ({
        ...prev,
        symbols: [...prev.symbols, newSymbol]
      }));
      setNewSymbol("");
    }
  };

  // Remove a symbol
  const removeSymbol = (symbol: string) => {
    setConfig(prev => ({
      ...prev,
      symbols: prev.symbols.filter(s => s !== symbol)
    }));
  };

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value / 100);
  };

  return (
    <Box className="trading-agent-dashboard">
      <Typography variant="h4" gutterBottom>
        Trading Agent Dashboard
      </Typography>

      {/* Agent Status Card */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Agent Status
          </Typography>
          
          {loading && !status ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress />
            </Box>
          ) : status ? (
            <Grid container spacing={2}>
              <Grid xs={12} md={4}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default' }}>
                  <Typography variant="body2" color="text.secondary">
                    Status
                  </Typography>
                  <Typography variant="h6">
                    <Chip 
                      label={status.status.toUpperCase()} 
                      color={
                        status.status === 'running' ? 'success' :
                        status.status === 'error' ? 'error' :
                        status.status === 'stopped' ? 'warning' : 'default'
                      } 
                    />
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid xs={12} md={4}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default' }}>
                  <Typography variant="body2" color="text.secondary">
                    Portfolio Value
                  </Typography>
                  <Typography variant="h6">
                    {formatCurrency(status.current_portfolio_value)}
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid xs={12} md={4}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default' }}>
                  <Typography variant="body2" color="text.secondary">
                    PnL
                  </Typography>
                  <Typography variant="h6" color={status.pnl_pct >= 0 ? 'success.main' : 'error.main'}>
                    {formatPercentage(status.pnl_pct)}
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid xs={12} md={4}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default' }}>
                  <Typography variant="body2" color="text.secondary">
                    Total Trades
                  </Typography>
                  <Typography variant="h6">
                    {status.total_trades}
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid xs={12} md={4}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default' }}>
                  <Typography variant="body2" color="text.secondary">
                    Profitable Trades
                  </Typography>
                  <Typography variant="h6">
                    {status.profitable_trades} ({status.total_trades > 0 ? 
                      `${((status.profitable_trades / status.total_trades) * 100).toFixed(1)}%` : 
                      '0%'})
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid xs={12} md={4}>
                <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default' }}>
                  <Typography variant="body2" color="text.secondary">
                    Uptime
                  </Typography>
                  <Typography variant="h6">
                    {Math.floor(status.uptime_seconds / 3600)}h {Math.floor((status.uptime_seconds % 3600) / 60)}m {status.uptime_seconds % 60}s
                  </Typography>
                </Paper>
              </Grid>
              
              {status.error_message && (
                <Grid xs={12}>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'error.light' }}>
                    <Typography variant="body2" color="error.main">
                      Error: {status.error_message}
                    </Typography>
                  </Paper>
                </Grid>
              )}
            </Grid>
          ) : (
            <Typography variant="body1" color="text.secondary">
              No status information available
            </Typography>
          )}
          
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            {status?.status === 'running' ? (
              <Button 
                variant="contained" 
                color="error" 
                onClick={stopAgent}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Stop Agent'}
              </Button>
            ) : (
              <Button 
                variant="contained" 
                color="success" 
                onClick={startAgent}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Start Agent'}
              </Button>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Active Positions */}
      {status?.active_positions && status.active_positions.length > 0 && (
        <Card variant="outlined" sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Active Positions
            </Typography>
            
            <Grid container spacing={2}>
              {status.active_positions.map(position => (
                <Grid xs={12} md={6} lg={4} key={position.symbol}>
                  <Paper elevation={1} sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      {position.symbol}
                    </Typography>
                    
                    <Grid container spacing={1}>
                      <Grid xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Quantity
                        </Typography>
                        <Typography variant="body1">
                          {position.quantity}
                        </Typography>
                      </Grid>
                      
                      <Grid xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Entry Price
                        </Typography>
                        <Typography variant="body1">
                          {formatCurrency(position.entry_price)}
                        </Typography>
                      </Grid>
                      
                      <Grid xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Current Price
                        </Typography>
                        <Typography variant="body1">
                          {formatCurrency(position.current_price)}
                        </Typography>
                      </Grid>
                      
                      <Grid xs={6}>
                        <Typography variant="body2" color="text.secondary">
                          Unrealized PnL
                        </Typography>
                        <Typography 
                          variant="body1" 
                          color={position.unrealized_pnl >= 0 ? 'success.main' : 'error.main'}
                        >
                          {formatCurrency(position.unrealized_pnl)} ({formatPercentage(position.unrealized_pnl_pct)})
                        </Typography>
                      </Grid>
                    </Grid>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Agent Configuration */}
      <Card variant="outlined">
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Agent Configuration
            </Typography>
            
            {isConfigEditing ? (
              <Box>
                <Button 
                  variant="outlined" 
                  color="secondary" 
                  onClick={() => setIsConfigEditing(false)}
                  sx={{ mr: 1 }}
                  disabled={loading}
                >
                  Cancel
                </Button>
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={saveConfig}
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Save'}
                </Button>
              </Box>
            ) : (
              <Button 
                variant="outlined" 
                color="primary" 
                onClick={() => setIsConfigEditing(true)}
                disabled={loading || status?.status === 'running'}
              >
                Edit Configuration
              </Button>
            )}
          </Box>
          
          {isConfigEditing ? (
            <Grid container spacing={3}>
              <Grid xs={12} md={6}>
                <TextField
                  label="Agent Name"
                  fullWidth
                  value={config.name}
                  onChange={(e) => setConfig(prev => ({ ...prev, name: e.target.value }))}
                  margin="normal"
                />
                
                <TextField
                  label="Description"
                  fullWidth
                  multiline
                  rows={2}
                  value={config.description}
                  onChange={(e) => setConfig(prev => ({ ...prev, description: e.target.value }))}
                  margin="normal"
                />
                
                <TextField
                  label="Initial Capital"
                  fullWidth
                  type="number"
                  value={config.initial_capital}
                  onChange={(e) => setConfig(prev => ({ ...prev, initial_capital: parseFloat(e.target.value) }))}
                  margin="normal"
                  InputProps={{
                    startAdornment: <Typography variant="body1">$</Typography>
                  }}
                />
              </Grid>
              
              <Grid xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Trading Symbols
                </Typography>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap', mb: 2 }}>
                  {config.symbols.map(symbol => (
                    <Chip
                      key={symbol}
                      label={symbol}
                      onDelete={() => removeSymbol(symbol)}
                      sx={{ m: 0.5 }}
                    />
                  ))}
                </Box>
                
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <TextField
                    label="Add Symbol"
                    value={newSymbol}
                    onChange={(e) => setNewSymbol(e.target.value)}
                    sx={{ mr: 1, flexGrow: 1 }}
                  />
                  <Button 
                    variant="contained" 
                    onClick={addSymbol}
                    disabled={!newSymbol}
                  >
                    Add
                  </Button>
                </Box>
              </Grid>
              
              <Grid xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1" gutterBottom>
                  Strategy Weights
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid xs={12} md={4}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={config.strategies.market_regime.enabled}
                          onChange={(e) => handleStrategyEnabledChange('market_regime', e.target.checked)}
                        />
                      }
                      label="Market Regime Strategy"
                    />
                    <TextField
                      label="Weight"
                      type="number"
                      fullWidth
                      value={config.strategies.market_regime.weight}
                      onChange={(e) => handleStrategyWeightChange('market_regime', parseFloat(e.target.value))}
                      disabled={!config.strategies.market_regime.enabled}
                      inputProps={{ min: 0, max: 1, step: 0.01 }}
                    />
                  </Grid>
                  
                  <Grid xs={12} md={4}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={config.strategies.ml_strategy.enabled}
                          onChange={(e) => handleStrategyEnabledChange('ml_strategy', e.target.checked)}
                        />
                      }
                      label="ML Strategy"
                    />
                    <TextField
                      label="Weight"
                      type="number"
                      fullWidth
                      value={config.strategies.ml_strategy.weight}
                      onChange={(e) => handleStrategyWeightChange('ml_strategy', parseFloat(e.target.value))}
                      disabled={!config.strategies.ml_strategy.enabled}
                      inputProps={{ min: 0, max: 1, step: 0.01 }}
                    />
                  </Grid>
                  
                  <Grid xs={12} md={4}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={config.strategies.portfolio_strategy.enabled}
                          onChange={(e) => handleStrategyEnabledChange('portfolio_strategy', e.target.checked)}
                        />
                      }
                      label="Portfolio Strategy"
                    />
                    <TextField
                      label="Weight"
                      type="number"
                      fullWidth
                      value={config.strategies.portfolio_strategy.weight}
                      onChange={(e) => handleStrategyWeightChange('portfolio_strategy', parseFloat(e.target.value))}
                      disabled={!config.strategies.portfolio_strategy.enabled}
                      inputProps={{ min: 0, max: 1, step: 0.01 }}
                    />
                  </Grid>
                </Grid>
              </Grid>
              
              <Grid xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1" gutterBottom>
                  Risk Management
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid xs={12} md={4}>
                    <TextField
                      label="Max Position Size"
                      type="number"
                      fullWidth
                      value={config.risk_management.max_position_size}
                      onChange={(e) => handleRiskManagementChange('max_position_size', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <Typography variant="body2">of portfolio</Typography>
                      }}
                      inputProps={{ min: 0, max: 1, step: 0.01 }}
                    />
                  </Grid>
                  
                  <Grid xs={12} md={4}>
                    <TextField
                      label="Stop Loss"
                      type="number"
                      fullWidth
                      value={config.risk_management.stop_loss_pct}
                      onChange={(e) => handleRiskManagementChange('stop_loss_pct', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <Typography variant="body2">%</Typography>
                      }}
                      inputProps={{ min: 0, max: 100, step: 0.1 }}
                    />
                  </Grid>
                  
                  <Grid xs={12} md={4}>
                    <TextField
                      label="Take Profit"
                      type="number"
                      fullWidth
                      value={config.risk_management.take_profit_pct}
                      onChange={(e) => handleRiskManagementChange('take_profit_pct', parseFloat(e.target.value))}
                      InputProps={{
                        endAdornment: <Typography variant="body2">%</Typography>
                      }}
                      inputProps={{ min: 0, max: 100, step: 0.1 }}
                    />
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          ) : (
            <Grid container spacing={3}>
              <Grid xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Basic Information
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  Name: {config.name}
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  Description: {config.description}
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  Initial Capital: {formatCurrency(config.initial_capital)}
                </Typography>
                
                <Typography variant="subtitle1" sx={{ mt: 2 }} gutterBottom>
                  Trading Symbols
                </Typography>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
                  {config.symbols.map(symbol => (
                    <Chip
                      key={symbol}
                      label={symbol}
                      sx={{ m: 0.5 }}
                    />
                  ))}
                </Box>
              </Grid>
              
              <Grid xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Strategy Weights
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  Market Regime Strategy: {config.strategies.market_regime.enabled ? 
                    `${(config.strategies.market_regime.weight * 100).toFixed(0)}%` : 
                    'Disabled'}
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  ML Strategy: {config.strategies.ml_strategy.enabled ? 
                    `${(config.strategies.ml_strategy.weight * 100).toFixed(0)}%` : 
                    'Disabled'}
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  Portfolio Strategy: {config.strategies.portfolio_strategy.enabled ? 
                    `${(config.strategies.portfolio_strategy.weight * 100).toFixed(0)}%` : 
                    'Disabled'}
                </Typography>
                
                <Typography variant="subtitle1" sx={{ mt: 2 }} gutterBottom>
                  Risk Management
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  Max Position Size: {(config.risk_management.max_position_size * 100).toFixed(0)}% of portfolio
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  Stop Loss: {config.risk_management.stop_loss_pct.toFixed(1)}%
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  Take Profit: {config.risk_management.take_profit_pct.toFixed(1)}%
                </Typography>
              </Grid>
            </Grid>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default TradingAgentDashboard;
