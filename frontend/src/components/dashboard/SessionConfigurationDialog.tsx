import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Grid,
  Typography,
  Divider,
  Chip,
  Box,
  Autocomplete,
  Switch,
  FormControlLabel,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip,
  InputAdornment
} from '@mui/material';
import {
  Close as CloseIcon,
  Add as AddIcon,
  Info as InfoIcon,
  AttachMoney as MoneyIcon
} from '@mui/icons-material';
import { useSystemControl } from '../../context/SystemControlContext';

// Define session configuration interface
interface SessionConfig {
  name: string;
  description: string;
  initial_capital: number;
  symbols: string[];
  agents: string[];
  risk_settings: {
    max_drawdown_percentage: number;
    use_stop_loss: boolean;
    stop_loss_percentage: number;
    max_open_positions: number;
    position_sizing_method: string;
    max_position_size_percentage: number;
  };
  advanced_settings: {
    auto_rebalance: boolean;
    rebalance_frequency: string;
    use_hedging: boolean;
    hedging_instruments: string[];
    log_level: string;
    backtest_before_live: boolean;
  };
}

interface SessionConfigurationDialogProps {
  open: boolean;
  onClose: () => void;
  editSession?: any; // Optional session to edit
}

const SessionConfigurationDialog: React.FC<SessionConfigurationDialogProps> = ({
  open,
  onClose,
  editSession
}) => {
  const { agents } = useSystemControl();
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);
  
  // Available symbols (would come from API in real implementation)
  const availableSymbols = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD',
    'DOGE/USD', 'DOT/USD', 'AVAX/USD', 'MATIC/USD', 'LINK/USD',
    'UNI/USD', 'AAVE/USD', 'ATOM/USD', 'ALGO/USD', 'FIL/USD'
  ];
  
  // Default session configuration
  const defaultSessionConfig: SessionConfig = {
    name: '',
    description: '',
    initial_capital: 10000,
    symbols: [],
    agents: [],
    risk_settings: {
      max_drawdown_percentage: 10,
      use_stop_loss: true,
      stop_loss_percentage: 5,
      max_open_positions: 5,
      position_sizing_method: 'fixed_percentage',
      max_position_size_percentage: 20
    },
    advanced_settings: {
      auto_rebalance: false,
      rebalance_frequency: 'daily',
      use_hedging: false,
      hedging_instruments: [],
      log_level: 'info',
      backtest_before_live: true
    }
  };
  
  // Session configuration state
  const [sessionConfig, setSessionConfig] = useState<SessionConfig>(defaultSessionConfig);
  
  // Form validation state
  const [formErrors, setFormErrors] = useState<{[key: string]: string}>({});
  
  // Initialize form with session data if editing
  useEffect(() => {
    if (editSession) {
      setSessionConfig({
        ...defaultSessionConfig,
        ...editSession,
        // Ensure all required fields are present
        risk_settings: {
          ...defaultSessionConfig.risk_settings,
          ...(editSession.risk_settings || {})
        },
        advanced_settings: {
          ...defaultSessionConfig.advanced_settings,
          ...(editSession.advanced_settings || {})
        }
      });
    } else {
      setSessionConfig(defaultSessionConfig);
    }
    
    // Reset states
    setError(null);
    setSuccess(false);
    setFormErrors({});
  }, [editSession, open]);
  
  // Handle form field changes
  const handleChange = (field: string, value: any) => {
    // Clear error for this field if it exists
    if (formErrors[field]) {
      setFormErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[field];
        return newErrors;
      });
    }
    
    // Update the field
    setSessionConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // Handle risk setting changes
  const handleRiskSettingChange = (setting: string, value: any) => {
    // Clear error for this setting if it exists
    if (formErrors[`risk_settings.${setting}`]) {
      setFormErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[`risk_settings.${setting}`];
        return newErrors;
      });
    }
    
    // Update the setting
    setSessionConfig(prev => ({
      ...prev,
      risk_settings: {
        ...prev.risk_settings,
        [setting]: value
      }
    }));
  };
  
  // Handle advanced setting changes
  const handleAdvancedSettingChange = (setting: string, value: any) => {
    // Clear error for this setting if it exists
    if (formErrors[`advanced_settings.${setting}`]) {
      setFormErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[`advanced_settings.${setting}`];
        return newErrors;
      });
    }
    
    // Update the setting
    setSessionConfig(prev => ({
      ...prev,
      advanced_settings: {
        ...prev.advanced_settings,
        [setting]: value
      }
    }));
  };
  
  // Validate form
  const validateForm = (): boolean => {
    const errors: {[key: string]: string} = {};
    
    // Basic validation
    if (!sessionConfig.name.trim()) {
      errors.name = 'Session name is required';
    }
    
    if (sessionConfig.initial_capital <= 0) {
      errors.initial_capital = 'Initial capital must be greater than 0';
    }
    
    if (sessionConfig.symbols.length === 0) {
      errors.symbols = 'At least one symbol is required';
    }
    
    // Risk settings validation
    if (sessionConfig.risk_settings.max_drawdown_percentage <= 0) {
      errors['risk_settings.max_drawdown_percentage'] = 'Must be greater than 0';
    }
    
    if (sessionConfig.risk_settings.use_stop_loss && sessionConfig.risk_settings.stop_loss_percentage <= 0) {
      errors['risk_settings.stop_loss_percentage'] = 'Must be greater than 0';
    }
    
    if (sessionConfig.risk_settings.max_open_positions <= 0) {
      errors['risk_settings.max_open_positions'] = 'Must be greater than 0';
    }
    
    if (sessionConfig.risk_settings.max_position_size_percentage <= 0 || 
        sessionConfig.risk_settings.max_position_size_percentage > 100) {
      errors['risk_settings.max_position_size_percentage'] = 'Must be between 0 and 100';
    }
    
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };
  
  // Handle form submission
  const handleSubmit = async () => {
    // Validate form
    if (!validateForm()) {
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // In a real implementation, this would be an API call to create/update the session
      console.log('Session configuration:', sessionConfig);
      
      // Show success message
      setSuccess(true);
      
      // Close dialog after a delay
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err) {
      console.error('Error creating/updating session:', err);
      setError('Failed to create/update session. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="md" 
      fullWidth
      PaperProps={{
        sx: { borderRadius: 2 }
      }}
    >
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6">
          {editSession ? 'Edit Trading Session' : 'Create New Trading Session'}
        </Typography>
        <IconButton onClick={onClose} size="small">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      
      <DialogContent dividers>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            Session {editSession ? 'updated' : 'created'} successfully!
          </Alert>
        )}
        
        <Grid container spacing={3}>
          {/* Basic Information */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" fontWeight="medium" gutterBottom>
              Basic Information
            </Typography>
            <Divider sx={{ mb: 2 }} />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Session Name"
              fullWidth
              value={sessionConfig.name}
              onChange={(e) => handleChange('name', e.target.value)}
              error={!!formErrors.name}
              helperText={formErrors.name}
              required
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Description"
              fullWidth
              value={sessionConfig.description}
              onChange={(e) => handleChange('description', e.target.value)}
              error={!!formErrors.description}
              helperText={formErrors.description}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Initial Capital"
              type="number"
              fullWidth
              value={sessionConfig.initial_capital}
              onChange={(e) => handleChange('initial_capital', Number(e.target.value))}
              error={!!formErrors.initial_capital}
              helperText={formErrors.initial_capital}
              required
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <MoneyIcon />
                  </InputAdornment>
                ),
                inputProps: { min: 1 }
              }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Autocomplete
              multiple
              id="symbols-autocomplete"
              options={availableSymbols}
              value={sessionConfig.symbols}
              onChange={(_, newValue) => handleChange('symbols', newValue)}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Trading Symbols"
                  error={!!formErrors.symbols}
                  helperText={formErrors.symbols}
                  required
                />
              )}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    label={option}
                    size="small"
                    {...getTagProps({ index })}
                  />
                ))
              }
            />
          </Grid>
          
          <Grid item xs={12}>
            <Autocomplete
              multiple
              id="agents-autocomplete"
              options={agents.map(agent => ({ id: agent.agent_id, name: agent.name }))}
              getOptionLabel={(option) => option.name}
              value={sessionConfig.agents.map(agentId => {
                const agent = agents.find(a => a.agent_id === agentId);
                return agent ? { id: agent.agent_id, name: agent.name } : { id: agentId, name: agentId };
              })}
              onChange={(_, newValue) => handleChange('agents', newValue.map(v => v.id))}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Assign Trading Agents"
                  helperText="Select agents to use in this session"
                />
              )}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    label={option.name}
                    size="small"
                    {...getTagProps({ index })}
                  />
                ))
              }
            />
          </Grid>
          
          {/* Risk Settings */}
          <Grid item xs={12} sx={{ mt: 2 }}>
            <Typography variant="subtitle1" fontWeight="medium" gutterBottom>
              Risk Settings
            </Typography>
            <Divider sx={{ mb: 2 }} />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Max Drawdown (%)"
              type="number"
              fullWidth
              value={sessionConfig.risk_settings.max_drawdown_percentage}
              onChange={(e) => handleRiskSettingChange('max_drawdown_percentage', Number(e.target.value))}
              error={!!formErrors['risk_settings.max_drawdown_percentage']}
              helperText={formErrors['risk_settings.max_drawdown_percentage'] || 'Session will pause if drawdown exceeds this percentage'}
              InputProps={{ inputProps: { min: 1, max: 100 } }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Max Open Positions"
              type="number"
              fullWidth
              value={sessionConfig.risk_settings.max_open_positions}
              onChange={(e) => handleRiskSettingChange('max_open_positions', Number(e.target.value))}
              error={!!formErrors['risk_settings.max_open_positions']}
              helperText={formErrors['risk_settings.max_open_positions'] || 'Maximum number of concurrent positions'}
              InputProps={{ inputProps: { min: 1, max: 50 } }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel id="position-sizing-method-label">Position Sizing Method</InputLabel>
              <Select
                labelId="position-sizing-method-label"
                value={sessionConfig.risk_settings.position_sizing_method}
                label="Position Sizing Method"
                onChange={(e) => handleRiskSettingChange('position_sizing_method', e.target.value)}
              >
                <MenuItem value="fixed_percentage">Fixed Percentage</MenuItem>
                <MenuItem value="risk_parity">Risk Parity</MenuItem>
                <MenuItem value="kelly_criterion">Kelly Criterion</MenuItem>
                <MenuItem value="volatility_adjusted">Volatility Adjusted</MenuItem>
              </Select>
              <FormHelperText>
                Method used to determine position sizes
              </FormHelperText>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Max Position Size (%)"
              type="number"
              fullWidth
              value={sessionConfig.risk_settings.max_position_size_percentage}
              onChange={(e) => handleRiskSettingChange('max_position_size_percentage', Number(e.target.value))}
              error={!!formErrors['risk_settings.max_position_size_percentage']}
              helperText={formErrors['risk_settings.max_position_size_percentage'] || 'Maximum percentage of portfolio per position'}
              InputProps={{ inputProps: { min: 1, max: 100 } }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={sessionConfig.risk_settings.use_stop_loss}
                    onChange={(e) => handleRiskSettingChange('use_stop_loss', e.target.checked)}
                  />
                }
                label="Use Stop Loss"
              />
              <Tooltip title="Automatically exit positions when losses reach a certain threshold">
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            
            {sessionConfig.risk_settings.use_stop_loss && (
              <TextField
                label="Stop Loss (%)"
                type="number"
                fullWidth
                value={sessionConfig.risk_settings.stop_loss_percentage}
                onChange={(e) => handleRiskSettingChange('stop_loss_percentage', Number(e.target.value))}
                error={!!formErrors['risk_settings.stop_loss_percentage']}
                helperText={formErrors['risk_settings.stop_loss_percentage']}
                InputProps={{ inputProps: { min: 0.1, max: 50, step: 0.1 } }}
                size="small"
                sx={{ mt: 1 }}
              />
            )}
          </Grid>
          
          {/* Advanced Settings */}
          <Grid item xs={12} sx={{ mt: 2 }}>
            <Typography variant="subtitle1" fontWeight="medium" gutterBottom>
              Advanced Settings
            </Typography>
            <Divider sx={{ mb: 2 }} />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={sessionConfig.advanced_settings.auto_rebalance}
                    onChange={(e) => handleAdvancedSettingChange('auto_rebalance', e.target.checked)}
                  />
                }
                label="Auto Rebalance Portfolio"
              />
              <Tooltip title="Automatically rebalance portfolio to maintain target allocations">
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            
            {sessionConfig.advanced_settings.auto_rebalance && (
              <FormControl fullWidth size="small" sx={{ mt: 1 }}>
                <InputLabel id="rebalance-frequency-label">Rebalance Frequency</InputLabel>
                <Select
                  labelId="rebalance-frequency-label"
                  value={sessionConfig.advanced_settings.rebalance_frequency}
                  label="Rebalance Frequency"
                  onChange={(e) => handleAdvancedSettingChange('rebalance_frequency', e.target.value)}
                >
                  <MenuItem value="hourly">Hourly</MenuItem>
                  <MenuItem value="daily">Daily</MenuItem>
                  <MenuItem value="weekly">Weekly</MenuItem>
                  <MenuItem value="monthly">Monthly</MenuItem>
                </Select>
              </FormControl>
            )}
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={sessionConfig.advanced_settings.use_hedging}
                    onChange={(e) => handleAdvancedSettingChange('use_hedging', e.target.checked)}
                  />
                }
                label="Use Hedging"
              />
              <Tooltip title="Use hedging instruments to reduce portfolio risk">
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            
            {sessionConfig.advanced_settings.use_hedging && (
              <Autocomplete
                multiple
                id="hedging-instruments-autocomplete"
                options={['BTC-PERP', 'ETH-PERP', 'SOL-PERP', 'XRP-PERP', 'ADA-PERP']}
                value={sessionConfig.advanced_settings.hedging_instruments}
                onChange={(_, newValue) => handleAdvancedSettingChange('hedging_instruments', newValue)}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Hedging Instruments"
                    size="small"
                    sx={{ mt: 1 }}
                  />
                )}
                size="small"
              />
            )}
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel id="log-level-label">Log Level</InputLabel>
              <Select
                labelId="log-level-label"
                value={sessionConfig.advanced_settings.log_level}
                label="Log Level"
                onChange={(e) => handleAdvancedSettingChange('log_level', e.target.value)}
              >
                <MenuItem value="debug">Debug</MenuItem>
                <MenuItem value="info">Info</MenuItem>
                <MenuItem value="warning">Warning</MenuItem>
                <MenuItem value="error">Error</MenuItem>
              </Select>
              <FormHelperText>
                Determines the verbosity of session logs
              </FormHelperText>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={sessionConfig.advanced_settings.backtest_before_live}
                  onChange={(e) => handleAdvancedSettingChange('backtest_before_live', e.target.checked)}
                />
              }
              label="Run Backtest Before Live Trading"
            />
            <FormHelperText>
              Automatically run a backtest with the selected configuration before starting live trading
            </FormHelperText>
          </Grid>
        </Grid>
      </DialogContent>
      
      <DialogActions sx={{ px: 3, py: 2 }}>
        <Button onClick={onClose} variant="outlined" disabled={loading}>
          Cancel
        </Button>
        <Button 
          onClick={handleSubmit} 
          variant="contained" 
          color="primary"
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : <AddIcon />}
        >
          {editSession ? 'Update Session' : 'Create Session'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default SessionConfigurationDialog;
