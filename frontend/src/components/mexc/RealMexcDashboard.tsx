import React, { useState, useEffect, useCallback } from 'react';
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
  Button,
  Container,
  SelectChangeEvent,
  Alert,
  Tabs,
  Tab,
  CircularProgress
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import TradingViewChart from './TradingViewChart';
import OrderBook from './OrderBook';
import MarketTrades from './MarketTrades';
import TradingPanel from './TradingPanel';
import PriceTickerBar from './PriceTickerBar';
import mexcService from '../../api/mexcService';
import { ConnectionStatus } from '../../api/mexcWebSocketService';

// Real MEXC API implementation with safe performance characteristics

// Types
interface MexcTicker {
  symbol: string;
  lastPrice: string;
  priceChange: string;
  priceChangePercent: string;
  volume: string;
  quoteVolume: string;
  high: string;
  low: string;
}

interface MexcOrderBook {
  symbol: string;
  bids: [string, string][];
  asks: [string, string][];
  timestamp: number;
}

interface MexcTrade {
  id: number;
  price: string;
  qty: string;
  time: number;
  isBuyerMaker: boolean;
}

interface RealMexcDashboardProps {
  defaultSymbol?: string;
  defaultTimeframe?: string;
}

// For our component's internal use only - we'll map this to the expected ConnectionStatus when passing to TradingPanel
type DashboardStatus = ConnectionStatus | 'error';

const RealMexcDashboard: React.FC<RealMexcDashboardProps> = ({
  defaultSymbol = 'BTC/USDT',
  defaultTimeframe = '1h'
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [symbol, setSymbol] = useState(defaultSymbol);
  const [timeframe, setTimeframe] = useState(defaultTimeframe);
  const [tabValue, setTabValue] = useState(0);
  const [showApiTest, setShowApiTest] = useState(false);
  
  // States for API data
  const [ticker, setTicker] = useState<MexcTicker | null>(null);
  const [orderBook, setOrderBook] = useState<MexcOrderBook | null>(null);
  const [trades, setTrades] = useState<MexcTrade[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<DashboardStatus>('disconnected');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  
  // Auto-refresh interval (in milliseconds)
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const refreshInterval = 10000; // 10 seconds
  
  // Available symbols and timeframes
  const availableSymbols = [
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'ADA/USDT',
    'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'SHIB/USDT', 'MATIC/USDT',
    'LTC/USDT', 'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT'
  ];
  
  const availableTimeframes = [
    '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
  ];
  
  // Fetch all data
  const fetchAllData = useCallback(async () => {
    if (isLoading) return; // Prevent multiple simultaneous fetches
    
    setIsLoading(true);
    setError(null);
    setConnectionStatus('connecting');
    
    try {
      // Format the symbol correctly (MEXC doesn't use /)
      const formattedSymbol = symbol.replace('/', '');
      
      // Fetch ticker data
      const tickerData = await mexcService.getSymbolData(formattedSymbol);
      setTicker(tickerData);
      
      // Fetch order book
      const orderBookData = await mexcService.getOrderBook(formattedSymbol);
      setOrderBook(orderBookData);
      
      // Fetch recent trades
      const tradesData = await mexcService.getRecentTrades(formattedSymbol);
      setTrades(tradesData);
      
      setConnectionStatus('connected');
      setLastUpdated(new Date());
    } catch (err) {
      console.error('Error fetching MEXC data:', err);
      setError(err instanceof Error ? err.message : String(err));
      setConnectionStatus('disconnected'); // Use 'disconnected' instead of 'error' to be compatible with TradingPanel
    } finally {
      setIsLoading(false);
    }
  }, [symbol, isLoading]);
  
  // Initial data fetch
  useEffect(() => {
    fetchAllData();
  }, [fetchAllData]);
  
  // Set up auto-refresh
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null;
    
    if (autoRefresh) {
      intervalId = setInterval(() => {
        fetchAllData();
      }, refreshInterval);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [fetchAllData, autoRefresh, refreshInterval]);
  
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
  
  // Format order book data
  const formattedOrderBook = React.useMemo(() => {
    if (!orderBook || !orderBook.bids || !orderBook.asks) {
      return { bids: [], asks: [] };
    }
    
    // Limit the number of entries to reduce rendering load
    const maxEntries = 10;
    
    return {
      bids: orderBook.bids.slice(0, maxEntries).map(([price, quantity]) => 
        [parseFloat(price), parseFloat(quantity)] as [number, number]
      ),
      asks: orderBook.asks.slice(0, maxEntries).map(([price, quantity]) => 
        [parseFloat(price), parseFloat(quantity)] as [number, number]
      )
    };
  }, [orderBook]);
  
  // Get last price and price change
  const lastPrice = ticker ? parseFloat(ticker.lastPrice) : 0;
  const priceChange = ticker ? parseFloat(ticker.priceChangePercent) : 0;
  
  // Handle symbol change
  const handleSymbolChange = (event: SelectChangeEvent<string>) => {
    setSymbol(event.target.value);
  };
  
  // Handle timeframe change
  const handleTimeframeChange = (event: SelectChangeEvent<string>) => {
    setTimeframe(event.target.value);
  };
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Toggle auto-refresh
  const toggleAutoRefresh = () => {
    setAutoRefresh(prev => !prev);
  };
  
  return (
    <Box 
      className="RealMexcDashboard"
      sx={{ 
        display: 'flex', 
        flexDirection: 'column',
        height: '100%', 
        width: '100%', 
        bgcolor: 'background.default',
        color: 'text.primary'
      }}
    >
      {/* Top section - Symbol, Timeframe selectors and controls */}
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
            color={connectionStatus === 'connected' ? 'success' : 
                  connectionStatus === 'connecting' ? 'warning' : 'error'}
            size="small"
          />
          
          <Chip
            label={autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
            color={autoRefresh ? 'success' : 'default'}
            size="small"
            onClick={toggleAutoRefresh}
          />
          
          {error && (
            <Chip
              label={error.length > 20 ? `${error.substring(0, 20)}...` : error}
              color="error"
              size="small"
            />
          )}
          
          <IconButton 
            size="small" 
            onClick={fetchAllData} 
            disabled={isLoading}
          >
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Stack>
      </Box>
      
      {/* Main content */}
      <Grid container sx={{ flexGrow: 1, overflow: 'auto' }}>
        {/* Left column - TradingView chart */}
        <Grid item xs={12} md={8} sx={{ height: '100%' }}>
          <Paper 
            elevation={0} 
            sx={{ 
              height: '100%', 
              borderRadius: 0, 
              bgcolor: 'background.paper' 
            }}
          >
            <TradingViewChart 
              symbol={symbol} 
              timeframe={timeframe} 
              data={[]} // No need for data with TradingView widget
            />
          </Paper>
        </Grid>
        
        {/* Right column - OrderBook, Market Trades, Trading Panel */}
        <Grid item xs={12} md={4} sx={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column', 
          borderLeft: '1px solid', 
          borderColor: 'divider',
          bgcolor: 'background.paper'
        }}>
          <Tabs 
            value={tabValue} 
            onChange={handleTabChange} 
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab label="Order Book" />
            <Tab label="Trades" />
            <Tab label="Trading" />
          </Tabs>
          
          <Box sx={{ flexGrow: 1, overflow: 'auto', p: 1 }}>
            {isLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <CircularProgress />
              </Box>
            ) : (
              <>
                {tabValue === 0 && (
                  <OrderBook 
                    bids={formattedOrderBook.bids} 
                    asks={formattedOrderBook.asks} 
                    loading={isLoading}
                  />
                )}
                
                {tabValue === 1 && (
                  <MarketTrades 
                    trades={formattedTrades}
                    loading={isLoading}
                  />
                )}
                
                {tabValue === 2 && (
                  <TradingPanel 
                    symbol={symbol} 
                    lastPrice={lastPrice} 
                    connectionStatus={connectionStatus === 'error' ? 'disconnected' : connectionStatus}
                  />
                )}
              </>
            )}
          </Box>
          
          {lastUpdated && (
            <Box sx={{ 
              p: 0.5, 
              display: 'flex', 
              justifyContent: 'flex-end',
              borderTop: '1px solid',
              borderColor: 'divider'
            }}>
              <Typography variant="caption" sx={{ opacity: 0.7 }}>
                Last updated: {lastUpdated.toLocaleTimeString()}
              </Typography>
            </Box>
          )}
        </Grid>
      </Grid>

      {/* Bottom ticker bar */}
      <Box sx={{ 
        borderTop: '1px solid', 
        borderColor: 'divider', 
        bgcolor: 'background.paper' 
      }}>
        <PriceTickerBar />
      </Box>
    </Box>
  );
};

export default RealMexcDashboard;
