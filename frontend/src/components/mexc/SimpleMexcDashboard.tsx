import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Grid, 
  Paper, 
  Typography, 
  Divider, 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel, 
  Stack, 
  Chip, 
  useTheme, 
  useMediaQuery, 
  IconButton, 
  Container, 
  SelectChangeEvent, 
  Alert, 
  Tabs, 
  Tab, 
  CircularProgress, 
  Button
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import TradingViewChart from './TradingViewChart';
import OrderBook from './OrderBook';
import MarketTrades from './MarketTrades';
import TradingPanel from './TradingPanel';
import PriceTickerBar from './PriceTickerBar';
import MexcApiTest from './MexcApiTest';
import mexcService from '../../api/mexcService';
import { useMexcData } from '../../hooks/useMexcData';

// Setting this for backward compatibility
const USE_MOCK_DATA = true;

interface SimpleMexcDashboardProps {
  defaultSymbol?: string;
  defaultTimeframe?: string;
}

const SimpleMexcDashboard: React.FC<SimpleMexcDashboardProps> = ({
  defaultSymbol = 'BTC/USDT',
  defaultTimeframe = '1h'
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [symbol, setSymbol] = useState(defaultSymbol);
  const [timeframe, setTimeframe] = useState(defaultTimeframe);
  const [tabValue, setTabValue] = useState(0);
  const [showApiTest, setShowApiTest] = useState(false);
  
  // Fetch data from MEXC API
  const { 
    ticker, 
    orderBook, 
    trades, 
    klineData, 
    connectionStatus, 
    isLoading, 
    error, 
    refreshData 
  } = useMexcData(symbol, timeframe);
  
  // Derived states
  const lastPrice = ticker ? parseFloat(ticker.lastPrice) : 0;
  const priceChange = ticker ? parseFloat(ticker.priceChangePercent) : 0;
  
  // When using mock data, we should suppress network errors
  // Don't display errors when using mock data
  const displayError = error && typeof error !== 'undefined' && !USE_MOCK_DATA;
  
  // Available trading pairs
  const availableSymbols = [
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'ADA/USDT',
    'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'SHIB/USDT', 'MATIC/USDT',
    'LTC/USDT', 'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT'
  ];
  
  // Available timeframes
  const availableTimeframes = [
    '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
  ];

  // Format trades data for our component
  const formattedTrades = React.useMemo(() => {
    if (!trades || trades.length === 0) {
      return [];
    }
    
    return trades.map(trade => ({
      id: trade.id.toString(),
      price: parseFloat(trade.price),
      amount: parseFloat(trade.qty),
      time: trade.time,
      isBuyer: !trade.isBuyerMaker
    }));
  }, [trades]);
  
  // Format order book data for our component with performance optimization
  const formattedOrderBook = React.useMemo(() => {
    if (!orderBook || !orderBook.bids || !orderBook.asks) {
      return { bids: [], asks: [] };
    }
    
    // Limit the number of entries to reduce rendering load
    const maxEntries = 10;
    
    return {
      bids: orderBook.bids.slice(0, maxEntries).map(([price, quantity]) => [parseFloat(price), parseFloat(quantity)]),
      asks: orderBook.asks.slice(0, maxEntries).map(([price, quantity]) => [parseFloat(price), parseFloat(quantity)])
    };
  }, [orderBook]);
  
  // Use React.memo for child components to prevent unnecessary re-renders
  const MemoizedOrderBook = React.useMemo(() => {
    return <OrderBook 
      bids={formattedOrderBook.bids} 
      asks={formattedOrderBook.asks} 
      lastPrice={lastPrice} 
      loading={isLoading} 
    />;
  }, [formattedOrderBook.bids, formattedOrderBook.asks, lastPrice, isLoading]);
  
  const MemoizedMarketTrades = React.useMemo(() => {
    return <MarketTrades 
      trades={formattedTrades}
      loading={isLoading}
    />;
  }, [formattedTrades, isLoading]);
  
  const MemoizedTradingPanel = React.useMemo(() => {
    return <TradingPanel 
      symbol={symbol} 
      lastPrice={lastPrice} 
      connectionStatus={connectionStatus} 
    />;
  }, [symbol, lastPrice, connectionStatus]);
  
  const MemoizedTradingViewChart = React.useMemo(() => {
    return <TradingViewChart 
      symbol={symbol} 
      timeframe={timeframe} 
      data={klineData} 
    />;
  }, [symbol, timeframe, klineData && klineData.length]);

  // Handle symbol change
  const handleSymbolChange = (event: SelectChangeEvent<string>) => {
    setSymbol(event.target.value as string);
  };
  
  // Handle timeframe change
  const handleTimeframeChange = (event: SelectChangeEvent<string>) => {
    setTimeframe(event.target.value as string);
  };
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  return (
    <Box 
      className="SimpleMexcDashboard"
      sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        height: '100%', 
        width: '100%', 
        overflow: 'hidden', 
        bgcolor: 'background.default',
        color: 'text.primary'
    }}>
      {/* Top section - Symbol, Timeframe selectors and Ticker */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        p: 1, 
        borderBottom: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.paper'
      }}>
        <Stack direction="row" spacing={2} alignItems="center">
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel id="symbol-select-label">Symbol</InputLabel>
            <Select
              labelId="symbol-select-label"
              value={symbol}
              label="Symbol"
              onChange={handleSymbolChange}
            >
              {availableSymbols.map(s => (
                <MenuItem key={s} value={s}>{s}</MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <FormControl size="small" sx={{ minWidth: 80 }}>
            <InputLabel id="timeframe-select-label">Timeframe</InputLabel>
            <Select
              labelId="timeframe-select-label"
              value={timeframe}
              label="Timeframe"
              onChange={handleTimeframeChange}
            >
              {availableTimeframes.map(tf => (
                <MenuItem key={tf} value={tf}>{tf}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Stack>
        
        <Stack direction="row" spacing={2} alignItems="center">
          <Chip 
            label={connectionStatus} 
            color={connectionStatus === 'connected' ? 'success' : connectionStatus === 'connecting' ? 'warning' : 'error'}
            size="small"
          />
          {displayError && (
            <Chip
              label={typeof error === 'string' ? error : 'Network Error'}
              color="error"
              size="small"
            />
          )}
          <IconButton size="small" onClick={refreshData} disabled={isLoading}>
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Stack>
      </Box>
      
      {/* Main content */}
      <Grid container sx={{ flexGrow: 1, minHeight: 0 }}>
        {/* Left column - TradingView chart */}
        <Grid item xs={12} md={8} sx={{ height: '100%', overflow: 'hidden' }}>
          <Paper elevation={0} sx={{ height: '100%', overflow: 'hidden', borderRadius: 0, bgcolor: 'background.paper' }}>
            {MemoizedTradingViewChart}
          </Paper>
        </Grid>
        
        {/* Right column - OrderBook, Market Trades, Trading Panel */}
        <Grid item xs={12} md={4} sx={{ height: '100%', display: 'flex', flexDirection: 'column', borderLeft: '1px solid', borderColor: 'divider', bgcolor: 'background.paper' }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab label="Order Book" />
            <Tab label="Trades" />
            <Tab label="Trading" />
          </Tabs>
          
          <Box sx={{ flexGrow: 1, overflow: 'auto', p: 1, bgcolor: 'background.paper' }}>
            {tabValue === 0 && MemoizedOrderBook}
            {tabValue === 1 && MemoizedMarketTrades}
            {tabValue === 2 && MemoizedTradingPanel}
          </Box>
        </Grid>
      </Grid>

      {/* API Test Toggle Button */}
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 1, borderTop: '1px solid', borderColor: 'divider' }}>
        <Button 
          variant="outlined" 
          size="small" 
          onClick={() => setShowApiTest(!showApiTest)}
          sx={{ mb: 1 }}
        >
          {showApiTest ? 'Hide API Test' : 'Show API Test'}
        </Button>
      </Box>

      {/* API Test Component */}
      {showApiTest && <MexcApiTest />}
      
      {/* Bottom ticker bar */}
      <Box sx={{ borderTop: '1px solid', borderColor: 'divider', bgcolor: 'background.paper' }}>
        <PriceTickerBar />
      </Box>
    </Box>
  );
};

export default SimpleMexcDashboard;