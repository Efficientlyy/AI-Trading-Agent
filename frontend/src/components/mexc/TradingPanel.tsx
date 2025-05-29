import React, { useState } from 'react';
import { Box, Typography, Button, TextField, Slider, ToggleButtonGroup, ToggleButton, Grid, Divider, FormControl, InputLabel, Select, MenuItem, SelectChangeEvent } from '@mui/material';

interface TradingPanelProps {
  symbol: string;
  lastPrice: number;
  connectionStatus: 'connected' | 'connecting' | 'disconnected' | 'reconnecting';
}

const TradingPanel: React.FC<TradingPanelProps> = ({ symbol, lastPrice, connectionStatus }) => {
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [amount, setAmount] = useState<string>('');
  const [price, setPrice] = useState<string>(lastPrice.toFixed(2));
  const [percentOfBalance, setPercentOfBalance] = useState<number>(0);

  // Split the symbol into base and quote currencies
  const [baseCurrency, quoteCurrency] = symbol.split('/');
  
  // Mock balance data - in a real app this would come from the API
  const mockBalance = {
    [baseCurrency]: 0.1,
    [quoteCurrency]: 10000
  };

  const handleSideChange = (event: React.MouseEvent<HTMLElement>, newSide: 'buy' | 'sell') => {
    if (newSide !== null) {
      setSide(newSide);
    }
  };

  const handleOrderTypeChange = (event: React.MouseEvent<HTMLElement>, newType: 'market' | 'limit') => {
    if (newType !== null) {
      setOrderType(newType);
    }
  };

  const handlePercentChange = (event: Event, newValue: number | number[]) => {
    const percent = newValue as number;
    setPercentOfBalance(percent);
    
    const maxAmount = side === 'buy' 
      ? mockBalance[quoteCurrency] / lastPrice 
      : mockBalance[baseCurrency];
    
    setAmount((maxAmount * percent / 100).toFixed(4));
  };

  const handleSubmit = () => {
    // This would be a call to your backend API to place the order
    alert(`Order placed: ${side} ${amount} ${baseCurrency} at ${orderType === 'market' ? 'market price' : price} ${quoteCurrency}`);
  };

  return (
    <Box>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>Place Order</Typography>
            
            <Box sx={{ mb: 2 }}>
              <ToggleButtonGroup
                value={side}
                exclusive
                onChange={handleSideChange}
                aria-label="order side"
                fullWidth
                sx={{ mb: 2 }}
              >
                <ToggleButton value="buy" sx={{ 
                  py: 1.5,
                  '&.Mui-selected': { 
                    bgcolor: 'success.main', 
                    color: 'white',
                    '&:hover': { bgcolor: 'success.dark' }
                  } 
                }}>
                  Buy {baseCurrency}
                </ToggleButton>
                <ToggleButton value="sell" sx={{ 
                  py: 1.5,
                  '&.Mui-selected': { 
                    bgcolor: 'error.main', 
                    color: 'white',
                    '&:hover': { bgcolor: 'error.dark' }
                  } 
                }}>
                  Sell {baseCurrency}
                </ToggleButton>
              </ToggleButtonGroup>

              <ToggleButtonGroup
                value={orderType}
                exclusive
                onChange={handleOrderTypeChange}
                aria-label="order type"
                fullWidth
                sx={{ mb: 2 }}
              >
                <ToggleButton value="market" sx={{ py: 1 }}>Market</ToggleButton>
                <ToggleButton value="limit" sx={{ py: 1 }}>Limit</ToggleButton>
              </ToggleButtonGroup>
            </Box>

            {orderType === 'limit' && (
              <TextField
                fullWidth
                label="Price"
                variant="outlined"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                InputProps={{ endAdornment: <Typography variant="body2">{quoteCurrency}</Typography> }}
                sx={{ mb: 2 }}
              />
            )}

            <TextField
              fullWidth
              label="Amount"
              variant="outlined"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              InputProps={{ endAdornment: <Typography variant="body2">{baseCurrency}</Typography> }}
              sx={{ mb: 1 }}
            />

            <Box sx={{ px: 1 }}>
              <Slider
                value={percentOfBalance}
                onChange={handlePercentChange}
                step={25}
                marks={[
                  { value: 0, label: '0%' },
                  { value: 25, label: '25%' },
                  { value: 50, label: '50%' },
                  { value: 75, label: '75%' },
                  { value: 100, label: '100%' },
                ]}
              />
            </Box>

            <Button 
              fullWidth 
              variant="contained" 
              color={side === 'buy' ? 'success' : 'error'}
              size="large"
              onClick={handleSubmit}
              disabled={connectionStatus !== 'connected' || !amount || parseFloat(amount) <= 0}
              sx={{ mt: 2, py: 1.5 }}
            >
              {side === 'buy' ? 'Buy' : 'Sell'} {baseCurrency}
            </Button>
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
          <Typography variant="h6" gutterBottom>Account Balance</Typography>
          <Box sx={{ mb: 2, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
            <Typography variant="body1" gutterBottom>
              Available {baseCurrency}: <strong>{mockBalance[baseCurrency].toFixed(8)}</strong>
            </Typography>
            <Typography variant="body1">
              Available {quoteCurrency}: <strong>{mockBalance[quoteCurrency].toFixed(2)}</strong>
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom>Order Summary</Typography>
          <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
            <Typography variant="body2" gutterBottom>
              Order Type: <strong>{orderType.charAt(0).toUpperCase() + orderType.slice(1)}</strong>
            </Typography>
            <Typography variant="body2" gutterBottom>
              Price: <strong>{orderType === 'market' ? 'Market Price' : `${price} ${quoteCurrency}`}</strong>
            </Typography>
            <Typography variant="body2" gutterBottom>
              Amount: <strong>{amount || '0.0000'} {baseCurrency}</strong>
            </Typography>
            <Typography variant="body2">
              Total: <strong>{(parseFloat(amount || '0') * (orderType === 'market' ? lastPrice : parseFloat(price || '0'))).toFixed(2)} {quoteCurrency}</strong>
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TradingPanel;