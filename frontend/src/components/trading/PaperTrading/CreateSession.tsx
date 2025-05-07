import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  OutlinedInput,
  CircularProgress,
  Paper,
  Stack,
  SelectChangeEvent
} from '@mui/material';
import { API_BASE_URL } from '../../../config';

// Define available exchanges and symbols
const exchanges = ['binance', 'coinbase', 'kraken'];
const availableSymbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'XRP/USDT', 'DOT/USDT'];
const strategies = ['default', 'moving_average', 'rsi', 'macd', 'bollinger', 'autonomous'];

interface FormData {
  name: string;
  description: string;
  exchange: string;
  symbols: string[];
  strategy: string;
  initial_capital: number;
}

const CreateSession: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState<FormData>({
    name: '',
    description: '',
    exchange: 'binance',
    symbols: ['BTC/USDT'],
    strategy: 'autonomous',
    initial_capital: 10000
  });

  // Handle form input changes
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement> | SelectChangeEvent) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name as string]: value
    });
  };

  // Handle symbol selection
  const handleSymbolChange = (event: SelectChangeEvent<string[]>) => {
    const {
      target: { value },
    } = event;
    setFormData({
      ...formData,
      symbols: typeof value === 'string' ? value.split(',') : value,
    });
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/paper-trading/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        throw new Error(`Failed to create session: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Session created:', data);
      
      // Redirect to sessions list
      navigate('/paper-trading');
    } catch (error) {
      console.error('Error creating session:', error);
      alert('Failed to create paper trading session. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: 'auto' }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h5" component="h1" gutterBottom>
          Create Paper Trading Session
        </Typography>
        
        <form onSubmit={handleSubmit}>
          <Stack spacing={3}>
            <TextField
              fullWidth
              label="Session Name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
              variant="outlined"
              placeholder="My Trading Session"
            />
            
            <TextField
              fullWidth
              label="Description"
              name="description"
              value={formData.description}
              onChange={handleChange}
              variant="outlined"
              multiline
              rows={2}
              placeholder="Optional description for this session"
            />
            
            <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 2 }}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Exchange</InputLabel>
                <Select
                  name="exchange"
                  value={formData.exchange}
                  onChange={handleChange}
                  label="Exchange"
                >
                  {exchanges.map(exchange => (
                    <MenuItem key={exchange} value={exchange}>
                      {exchange.charAt(0).toUpperCase() + exchange.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <FormControl fullWidth variant="outlined">
                <InputLabel>Strategy</InputLabel>
                <Select
                  name="strategy"
                  value={formData.strategy}
                  onChange={handleChange}
                  label="Strategy"
                >
                  {strategies.map(strategy => (
                    <MenuItem key={strategy} value={strategy}>
                      {strategy.replace('_', ' ').charAt(0).toUpperCase() + strategy.replace('_', ' ').slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
            
            <FormControl fullWidth variant="outlined">
              <InputLabel>Trading Pairs</InputLabel>
              <Select
                multiple
                name="symbols"
                value={formData.symbols}
                onChange={handleSymbolChange}
                input={<OutlinedInput label="Trading Pairs" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip key={value} label={value} />
                    ))}
                  </Box>
                )}
              >
                {availableSymbols.map(symbol => (
                  <MenuItem key={symbol} value={symbol}>
                    {symbol}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <Box sx={{ maxWidth: '50%' }}>
              <TextField
                fullWidth
                label="Initial Capital (USD)"
                name="initial_capital"
                type="number"
                value={formData.initial_capital}
                onChange={handleChange}
                required
                variant="outlined"
                inputProps={{ min: 1000 }}
              />
            </Box>
            
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
              <Button 
                variant="outlined" 
                onClick={() => navigate('/paper-trading')}
                disabled={loading}
              >
                Cancel
              </Button>
              <Button 
                type="submit" 
                variant="contained" 
                color="primary"
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : null}
              >
                {loading ? 'Creating...' : 'Create Session'}
              </Button>
            </Box>
          </Stack>
        </form>
      </Paper>
    </Box>
  );
};

export default CreateSession;
