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
  Tooltip
} from '@mui/material';
import {
  Close as CloseIcon,
  Add as AddIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useSystemControl } from '../../context/SystemControlContext';

// Dark theme constants
const darkBg = '#1E1E1E';
const darkPaperBg = '#2D2D2D';
const darkBorder = '#444444';
const darkText = '#FFFFFF';
const darkSecondaryText = '#AAAAAA';

// Define agent configuration interface
interface AgentConfig {
  name: string;
  type: string;
  strategy: string;
  symbols: string[];
  parameters: {
    risk_level: string;
    max_position_size: number;
    use_stop_loss: boolean;
    stop_loss_percentage: number;
    use_take_profit: boolean;
    take_profit_percentage: number;
    max_open_positions: number;
    use_trailing_stop: boolean;
    trailing_stop_percentage: number;
    [key: string]: any;
  };
  advanced_settings: {
    use_sentiment: boolean;
    sentiment_threshold: number;
    use_market_regime_filter: boolean;
    use_volatility_filter: boolean;
    max_drawdown_threshold: number;
    [key: string]: any;
  };
}

interface AgentConfigurationDialogProps {
  open: boolean;
  onClose: () => void;
  editAgent?: any; // Optional agent to edit
}

const AgentConfigurationDialog: React.FC<AgentConfigurationDialogProps> = ({
  open,
  onClose,
  editAgent
}) => {
  const { startAgent } = useSystemControl();
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);
  
  // Available strategies and symbols (would come from API in real implementation)
  const availableStrategies = [
    { id: 'momentum', name: 'Momentum Strategy' },
    { id: 'mean_reversion', name: 'Mean Reversion Strategy' },
    { id: 'trend_following', name: 'Trend Following Strategy' },
    { id: 'breakout', name: 'Breakout Strategy' },
    { id: 'rsi_strategy', name: 'RSI Strategy' },
    { id: 'macd_strategy', name: 'MACD Strategy' },
    { id: 'dual_ma_strategy', name: 'Dual Moving Average Strategy' },
    { id: 'bollinger_bands', name: 'Bollinger Bands Strategy' }
  ];
  
  const availableSymbols = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD',
    'DOGE/USD', 'DOT/USD', 'AVAX/USD', 'MATIC/USD', 'LINK/USD',
    'UNI/USD', 'AAVE/USD', 'ATOM/USD', 'ALGO/USD', 'FIL/USD'
  ];
  
  const agentTypes = [
    { id: 'crypto', name: 'Cryptocurrency Agent' },
    { id: 'forex', name: 'Forex Agent' },
    { id: 'stock', name: 'Stock Agent' },
    { id: 'multi_asset', name: 'Multi-Asset Agent' }
  ];
  
  // Default agent configuration
  const defaultAgentConfig: AgentConfig = {
    name: '',
    type: 'crypto',
    strategy: '',
    symbols: [],
    parameters: {
      risk_level: 'medium',
      max_position_size: 10,
      use_stop_loss: true,
      stop_loss_percentage: 5,
      use_take_profit: true,
      take_profit_percentage: 10,
      max_open_positions: 3,
      use_trailing_stop: false,
      trailing_stop_percentage: 2
    },
    advanced_settings: {
      use_sentiment: false,
      sentiment_threshold: 0.6,
      use_market_regime_filter: false,
      use_volatility_filter: true,
      max_drawdown_threshold: 15
    }
  };
  
  // Agent configuration state
  const [agentConfig, setAgentConfig] = useState<AgentConfig>(defaultAgentConfig);
  
  // Form validation state
  const [formErrors, setFormErrors] = useState<{[key: string]: string}>({});
  
  // Initialize form with agent data if editing
  useEffect(() => {
    if (editAgent) {
      setAgentConfig({
        ...defaultAgentConfig,
        ...editAgent,
        // Ensure all required fields are present
        parameters: {
          ...defaultAgentConfig.parameters,
          ...(editAgent.parameters || {})
        },
        advanced_settings: {
          ...defaultAgentConfig.advanced_settings,
          ...(editAgent.advanced_settings || {})
        }
      });
    } else {
      setAgentConfig(defaultAgentConfig);
    }
    
    // Reset states
    setError(null);
    setSuccess(false);
    setFormErrors({});
  }, [editAgent, open]);
  
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
    setAgentConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // Handle parameter changes
  const handleParameterChange = (parameter: string, value: any) => {
    // Clear error for this parameter if it exists
    if (formErrors[`parameters.${parameter}`]) {
      setFormErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[`parameters.${parameter}`];
        return newErrors;
      });
    }
    
    // Update the parameter
    setAgentConfig(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [parameter]: value
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
    setAgentConfig(prev => ({
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
    if (!agentConfig.name.trim()) {
      errors.name = 'Agent name is required';
    }
    
    if (!agentConfig.strategy) {
      errors.strategy = 'Strategy is required';
    }
    
    if (agentConfig.symbols.length === 0) {
      errors.symbols = 'At least one symbol is required';
    }
    
    // Parameter validation
    if (agentConfig.parameters.max_position_size <= 0) {
      errors['parameters.max_position_size'] = 'Must be greater than 0';
    }
    
    if (agentConfig.parameters.use_stop_loss && agentConfig.parameters.stop_loss_percentage <= 0) {
      errors['parameters.stop_loss_percentage'] = 'Must be greater than 0';
    }
    
    if (agentConfig.parameters.use_take_profit && agentConfig.parameters.take_profit_percentage <= 0) {
      errors['parameters.take_profit_percentage'] = 'Must be greater than 0';
    }
    
    if (agentConfig.parameters.max_open_positions <= 0) {
      errors['parameters.max_open_positions'] = 'Must be greater than 0';
    }
    
    if (agentConfig.parameters.use_trailing_stop && agentConfig.parameters.trailing_stop_percentage <= 0) {
      errors['parameters.trailing_stop_percentage'] = 'Must be greater than 0';
    }
    
    // Advanced settings validation
    if (agentConfig.advanced_settings.use_sentiment && 
        (agentConfig.advanced_settings.sentiment_threshold < 0 || 
         agentConfig.advanced_settings.sentiment_threshold > 1)) {
      errors['advanced_settings.sentiment_threshold'] = 'Must be between 0 and 1';
    }
    
    if (agentConfig.advanced_settings.max_drawdown_threshold <= 0) {
      errors['advanced_settings.max_drawdown_threshold'] = 'Must be greater than 0';
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
      
      // In a real implementation, this would be an API call to create/update the agent
      console.log('Agent configuration:', agentConfig);
      
      // Show success message
      setSuccess(true);
      
      // Close dialog after a delay
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (err) {
      console.error('Error creating/updating agent:', err);
      setError('Failed to create/update agent. Please try again.');
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
        sx: { borderRadius: 2, bgcolor: darkPaperBg, color: darkText }
      }}
    >
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: darkBg, color: darkText, borderBottom: `1px solid ${darkBorder}` }}>
        <Typography variant="h6" sx={{ color: darkText, fontWeight: 'bold' }}>
          {editAgent ? 'Edit Trading Agent' : 'Create New Trading Agent'}
        </Typography>
        <IconButton onClick={onClose} size="small" sx={{ color: darkText }}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      
      <DialogContent dividers sx={{ bgcolor: darkPaperBg, color: darkText, borderColor: darkBorder }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            Agent {editAgent ? 'updated' : 'created'} successfully!
          </Alert>
        )}
        
        <Grid container spacing={3}>
          {/* Basic Information */}
          <Grid item xs={12}>
            <Typography variant="subtitle1" sx={{ color: darkText, fontWeight: 'bold' }} gutterBottom>
              Basic Information
            </Typography>
            <Divider sx={{ mb: 2, borderColor: darkBorder }} />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Agent Name"
              fullWidth
              value={agentConfig.name}
              onChange={(e) => handleChange('name', e.target.value)}
              error={!!formErrors.name}
              helperText={formErrors.name}
              required
              InputLabelProps={{ style: { color: darkSecondaryText } }}
              InputProps={{ style: { color: darkText } }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth error={!!formErrors.type}>
              <InputLabel id="type-label" sx={{ color: darkSecondaryText }}>Agent Type</InputLabel>
              <Select
                labelId="type-label"
                value={agentConfig.type}
                label="Agent Type"
                onChange={(e) => handleChange('type', e.target.value)}
                required
                sx={{ color: darkText }}
              >
                <MenuItem value="algorithmic">Algorithmic</MenuItem>
                <MenuItem value="ai">AI-Powered</MenuItem>
                <MenuItem value="hybrid">Hybrid</MenuItem>
              </Select>
              {formErrors.type && <FormHelperText sx={{ color: '#f44336' }}>{formErrors.type}</FormHelperText>}
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth error={!!formErrors.strategy}>
              <InputLabel id="strategy-label" sx={{ color: darkSecondaryText }}>Trading Strategy</InputLabel>
              <Select
                labelId="strategy-label"
                value={agentConfig.strategy}
                label="Trading Strategy"
                onChange={(e) => handleChange('strategy', e.target.value)}
                required
                sx={{ color: darkText }}
              >
                <MenuItem value="momentum">Momentum</MenuItem>
                <MenuItem value="mean_reversion">Mean Reversion</MenuItem>
                <MenuItem value="trend_following">Trend Following</MenuItem>
                <MenuItem value="breakout">Breakout</MenuItem>
                <MenuItem value="custom">Custom</MenuItem>
              </Select>
              {formErrors.strategy && <FormHelperText sx={{ color: '#f44336' }}>{formErrors.strategy}</FormHelperText>}
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Autocomplete
              multiple
              id="symbols-autocomplete"
              options={availableSymbols}
              value={agentConfig.symbols}
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
          
          {/* Trading Parameters */}
          <Grid item xs={12} sx={{ mt: 2 }}>
            <Typography variant="subtitle1" fontWeight="medium" gutterBottom>
              Trading Parameters
            </Typography>
            <Divider sx={{ mb: 2 }} />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel id="risk-level-label">Risk Level</InputLabel>
              <Select
                labelId="risk-level-label"
                value={agentConfig.parameters.risk_level}
                label="Risk Level"
                onChange={(e) => handleParameterChange('risk_level', e.target.value)}
              >
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="high">High</MenuItem>
              </Select>
              <FormHelperText>
                Determines position sizing and risk management settings
              </FormHelperText>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Max Position Size (%)"
              type="number"
              fullWidth
              value={agentConfig.parameters.max_position_size}
              onChange={(e) => handleParameterChange('max_position_size', Number(e.target.value))}
              error={!!formErrors['parameters.max_position_size']}
              helperText={formErrors['parameters.max_position_size'] || 'Maximum percentage of portfolio per position'}
              InputProps={{ inputProps: { min: 1, max: 100 } }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Max Open Positions"
              type="number"
              fullWidth
              value={agentConfig.parameters.max_open_positions}
              onChange={(e) => handleParameterChange('max_open_positions', Number(e.target.value))}
              error={!!formErrors['parameters.max_open_positions']}
              helperText={formErrors['parameters.max_open_positions'] || 'Maximum number of concurrent positions'}
              InputProps={{ inputProps: { min: 1, max: 20 } }}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={agentConfig.parameters.use_stop_loss}
                    onChange={(e) => handleParameterChange('use_stop_loss', e.target.checked)}
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
            
            {agentConfig.parameters.use_stop_loss && (
              <TextField
                label="Stop Loss (%)"
                type="number"
                fullWidth
                value={agentConfig.parameters.stop_loss_percentage}
                onChange={(e) => handleParameterChange('stop_loss_percentage', Number(e.target.value))}
                error={!!formErrors['parameters.stop_loss_percentage']}
                helperText={formErrors['parameters.stop_loss_percentage']}
                InputProps={{ inputProps: { min: 0.1, max: 50, step: 0.1 } }}
                size="small"
                sx={{ mt: 1 }}
              />
            )}
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={agentConfig.parameters.use_take_profit}
                    onChange={(e) => handleParameterChange('use_take_profit', e.target.checked)}
                  />
                }
                label="Use Take Profit"
              />
              <Tooltip title="Automatically exit positions when profits reach a certain threshold">
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            
            {agentConfig.parameters.use_take_profit && (
              <TextField
                label="Take Profit (%)"
                type="number"
                fullWidth
                value={agentConfig.parameters.take_profit_percentage}
                onChange={(e) => handleParameterChange('take_profit_percentage', Number(e.target.value))}
                error={!!formErrors['parameters.take_profit_percentage']}
                helperText={formErrors['parameters.take_profit_percentage']}
                InputProps={{ inputProps: { min: 0.1, max: 100, step: 0.1 } }}
                size="small"
                sx={{ mt: 1 }}
              />
            )}
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={agentConfig.parameters.use_trailing_stop}
                    onChange={(e) => handleParameterChange('use_trailing_stop', e.target.checked)}
                  />
                }
                label="Use Trailing Stop"
              />
              <Tooltip title="Dynamic stop loss that follows price movement to lock in profits">
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            
            {agentConfig.parameters.use_trailing_stop && (
              <TextField
                label="Trailing Stop (%)"
                type="number"
                fullWidth
                value={agentConfig.parameters.trailing_stop_percentage}
                onChange={(e) => handleParameterChange('trailing_stop_percentage', Number(e.target.value))}
                error={!!formErrors['parameters.trailing_stop_percentage']}
                helperText={formErrors['parameters.trailing_stop_percentage']}
                InputProps={{ inputProps: { min: 0.1, max: 20, step: 0.1 } }}
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
                    checked={agentConfig.advanced_settings.use_sentiment}
                    onChange={(e) => handleAdvancedSettingChange('use_sentiment', e.target.checked)}
                  />
                }
                label="Use Sentiment Analysis"
              />
              <Tooltip title="Incorporate market sentiment data into trading decisions">
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            
            {agentConfig.advanced_settings.use_sentiment && (
              <TextField
                label="Sentiment Threshold"
                type="number"
                fullWidth
                value={agentConfig.advanced_settings.sentiment_threshold}
                onChange={(e) => handleAdvancedSettingChange('sentiment_threshold', Number(e.target.value))}
                error={!!formErrors['advanced_settings.sentiment_threshold']}
                helperText={formErrors['advanced_settings.sentiment_threshold'] || 'Value between 0 and 1'}
                InputProps={{ inputProps: { min: 0, max: 1, step: 0.05 } }}
                size="small"
                sx={{ mt: 1 }}
              />
            )}
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={agentConfig.advanced_settings.use_market_regime_filter}
                    onChange={(e) => handleAdvancedSettingChange('use_market_regime_filter', e.target.checked)}
                  />
                }
                label="Market Regime Filter"
              />
              <Tooltip title="Adapt strategy based on current market conditions (trending, ranging, volatile)">
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={agentConfig.advanced_settings.use_volatility_filter}
                    onChange={(e) => handleAdvancedSettingChange('use_volatility_filter', e.target.checked)}
                  />
                }
                label="Volatility Filter"
              />
              <Tooltip title="Adjust position sizing based on current market volatility">
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              label="Max Drawdown Threshold (%)"
              type="number"
              fullWidth
              value={agentConfig.advanced_settings.max_drawdown_threshold}
              onChange={(e) => handleAdvancedSettingChange('max_drawdown_threshold', Number(e.target.value))}
              error={!!formErrors['advanced_settings.max_drawdown_threshold']}
              helperText={formErrors['advanced_settings.max_drawdown_threshold'] || 'Agent will pause trading if drawdown exceeds this percentage'}
              InputLabelProps={{ style: { color: darkSecondaryText } }}
              InputProps={{ 
                inputProps: { min: 1, max: 50, step: 0.5 },
                style: { color: darkText } 
              }}
              sx={{ '& .MuiOutlinedInput-root': { '& fieldset': { borderColor: darkBorder } } }}
            />
          </Grid>
        </Grid>
      </DialogContent>
      
      <DialogActions sx={{ px: 3, py: 2, bgcolor: darkPaperBg, borderTop: `1px solid ${darkBorder}` }}>
        <Button onClick={onClose} variant="outlined" disabled={loading} sx={{ color: darkText, borderColor: darkBorder }}>
          Cancel
        </Button>
        <Button 
          onClick={handleSubmit} 
          variant="contained" 
          color="primary"
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : <AddIcon />}
        >
          {editAgent ? 'Update Agent' : 'Create Agent'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AgentConfigurationDialog;
