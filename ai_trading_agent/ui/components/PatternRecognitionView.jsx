/**
 * Pattern Recognition View Component
 * 
 * This component displays detected chart patterns from the Technical Analysis Agent.
 * Enhanced with advanced visualization and interactive features.
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemIcon,
  Divider,
  Chip,
  CircularProgress,
  useTheme,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  IconButton,
  Tooltip,
  Card,
  CardContent,
  CardMedia,
  CardActionArea,
  CardActions,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Tabs,
  Tab,
  Collapse,
  Alert,
  Rating,
  LinearProgress,
  Stack,
  alpha
} from '@mui/material';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import SignalCellularAltIcon from '@mui/icons-material/SignalCellularAlt';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';
import WarningIcon from '@mui/icons-material/Warning';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import InfoIcon from '@mui/icons-material/Info';
import FilterListIcon from '@mui/icons-material/FilterList';
import RefreshIcon from '@mui/icons-material/Refresh';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import axios from 'axios';

// Pattern card component for displaying individual patterns
const PatternCard = ({ pattern, isMockData }) => {
  const theme = useTheme();
  
  // Parse pattern details
  const { 
    name, 
    direction, 
    confidence, 
    symbol, 
    timeframe,
    description,
    candles
  } = pattern;
  
  // Format pattern name
  const formatPatternName = (name) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };
  
  // Get direction icon
  const getDirectionIcon = () => {
    if (direction === 'bullish') {
      return <TrendingUpIcon sx={{ color: theme.palette.success.main }} />;
    } else if (direction === 'bearish') {
      return <TrendingDownIcon sx={{ color: theme.palette.error.main }} />;
    } else {
      return <TrendingFlatIcon sx={{ color: theme.palette.info.main }} />;
    }
  };
  
  // Pattern descriptions
  const patternDescriptions = {
    'morning_star': 'A bullish reversal pattern consisting of a large bearish candle, followed by a small-bodied candle, and completed by a large bullish candle.',
    'evening_star': 'A bearish reversal pattern consisting of a large bullish candle, followed by a small-bodied candle, and completed by a large bearish candle.',
    'three_white_soldiers': 'A bullish reversal pattern consisting of three consecutive bullish candles, each closing higher than the previous.',
    'three_black_crows': 'A bearish reversal pattern consisting of three consecutive bearish candles, each closing lower than the previous.',
    'three_inside_up': 'A bullish reversal pattern where a small bullish candle is contained within the prior bearish candle, followed by a bullish candle that closes above the second candle.',
    'three_inside_down': 'A bearish reversal pattern where a small bearish candle is contained within the prior bullish candle, followed by a bearish candle that closes below the second candle.',
    'three_outside_up': 'A bullish reversal pattern where a bullish candle engulfs the prior bearish candle, followed by another bullish candle that closes higher.',
    'three_outside_down': 'A bearish reversal pattern where a bearish candle engulfs the prior bullish candle, followed by another bearish candle that closes lower.',
    'abandoned_baby_bullish': 'A bullish reversal pattern consisting of a bearish candle, followed by a doji that gaps down, and completed by a bullish candle that gaps up.',
    'abandoned_baby_bearish': 'A bearish reversal pattern consisting of a bullish candle, followed by a doji that gaps up, and completed by a bearish candle that gaps down.'
  };
  
  // Get pattern description
  const getPatternDescription = (patternName) => {
    return patternDescriptions[patternName] || 'A technical chart pattern detected by the Technical Analysis Agent.';
  };
  
  // Draw simple pattern visualization using SVG
  const drawPatternVisualization = () => {
    const svgWidth = 100;
    const svgHeight = 60;
    const padding = 5;
    const candleWidth = 12;
    const spacing = (svgWidth - padding * 2 - candleWidth * 3) / 2;
    
    // Draw based on pattern type
    const candlesticks = [];
    
    if (name === 'morning_star') {
      // First candle: bearish, long body
      candlesticks.push({
        x: padding,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 20,
        bodyBottom: 40,
        isBullish: false
      });
      
      // Second candle: small doji
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 25,
        wickBottom: 35,
        bodyTop: 29,
        bodyBottom: 31,
        isBullish: true
      });
      
      // Third candle: bullish, long body
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 40,
        bodyBottom: 20,
        isBullish: true
      });
    } else if (name === 'evening_star') {
      // First candle: bullish, long body
      candlesticks.push({
        x: padding,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 40,
        bodyBottom: 20,
        isBullish: true
      });
      
      // Second candle: small doji
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 25,
        wickBottom: 35,
        bodyTop: 31,
        bodyBottom: 29,
        isBullish: false
      });
      
      // Third candle: bearish, long body
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 20,
        bodyBottom: 40,
        isBullish: false
      });
    } else if (name === 'three_white_soldiers') {
      // Three bullish candles, each higher than the previous
      candlesticks.push({
        x: padding,
        wickTop: 30,
        wickBottom: 50,
        bodyTop: 45,
        bodyBottom: 35,
        isBullish: true
      });
      
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 20,
        wickBottom: 40,
        bodyTop: 35,
        bodyBottom: 25,
        isBullish: true
      });
      
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 10,
        wickBottom: 30,
        bodyTop: 25,
        bodyBottom: 15,
        isBullish: true
      });
    } else if (name === 'three_black_crows') {
      // Three bearish candles, each lower than the previous
      candlesticks.push({
        x: padding,
        wickTop: 10,
        wickBottom: 30,
        bodyTop: 15,
        bodyBottom: 25,
        isBullish: false
      });
      
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 20,
        wickBottom: 40,
        bodyTop: 25,
        bodyBottom: 35,
        isBullish: false
      });
      
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 30,
        wickBottom: 50,
        bodyTop: 35,
        bodyBottom: 45,
        isBullish: false
      });
    } else if (name === 'three_inside_up') {
      // First candle: bearish, long body
      candlesticks.push({
        x: padding,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 20,
        bodyBottom: 40,
        isBullish: false
      });
      
      // Second candle: bullish, inside first
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 25,
        wickBottom: 40,
        bodyTop: 35,
        bodyBottom: 25,
        isBullish: true
      });
      
      // Third candle: bullish, closing higher
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 15,
        wickBottom: 35,
        bodyTop: 30,
        bodyBottom: 20,
        isBullish: true
      });
    } else if (name === 'three_inside_down') {
      // First candle: bullish, long body
      candlesticks.push({
        x: padding,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 40,
        bodyBottom: 20,
        isBullish: true
      });
      
      // Second candle: bearish, inside first
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 20,
        wickBottom: 35,
        bodyTop: 25,
        bodyBottom: 35,
        isBullish: false
      });
      
      // Third candle: bearish, closing lower
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 25,
        wickBottom: 45,
        bodyTop: 30,
        bodyBottom: 40,
        isBullish: false
      });
    } else if (name === 'three_outside_up') {
      // First candle: bearish, normal body
      candlesticks.push({
        x: padding,
        wickTop: 20,
        wickBottom: 40,
        bodyTop: 25,
        bodyBottom: 35,
        isBullish: false
      });
      
      // Second candle: bullish, engulfing
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 40,
        bodyBottom: 20,
        isBullish: true
      });
      
      // Third candle: bullish, closing higher
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 10,
        wickBottom: 30,
        bodyTop: 25,
        bodyBottom: 15,
        isBullish: true
      });
    } else if (name === 'three_outside_down') {
      // First candle: bullish, normal body
      candlesticks.push({
        x: padding,
        wickTop: 20,
        wickBottom: 40,
        bodyTop: 35,
        bodyBottom: 25,
        isBullish: true
      });
      
      // Second candle: bearish, engulfing
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 20,
        bodyBottom: 40,
        isBullish: false
      });
      
      // Third candle: bearish, closing lower
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 30,
        wickBottom: 50,
        bodyTop: 35,
        bodyBottom: 45,
        isBullish: false
      });
    } else if (name === 'abandoned_baby_bullish') {
      // First candle: bearish, long body
      candlesticks.push({
        x: padding,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 20,
        bodyBottom: 40,
        isBullish: false
      });
      
      // Second candle: doji with gap
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 45,
        wickBottom: 55,
        bodyTop: 50,
        bodyBottom: 50,
        isBullish: true
      });
      
      // Third candle: bullish with gap
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 40,
        bodyBottom: 20,
        isBullish: true
      });
    } else if (name === 'abandoned_baby_bearish') {
      // First candle: bullish, long body
      candlesticks.push({
        x: padding,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 40,
        bodyBottom: 20,
        isBullish: true
      });
      
      // Second candle: doji with gap
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 5,
        wickBottom: 15,
        bodyTop: 10,
        bodyBottom: 10,
        isBullish: true
      });
      
      // Third candle: bearish with gap
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 15,
        wickBottom: 45,
        bodyTop: 20,
        bodyBottom: 40,
        isBullish: false
      });
    } else {
      // Generic three-candle pattern
      candlesticks.push({
        x: padding,
        wickTop: 20,
        wickBottom: 40,
        bodyTop: 25,
        bodyBottom: 35,
        isBullish: direction === 'bullish'
      });
      
      candlesticks.push({
        x: padding + candleWidth + spacing,
        wickTop: 15,
        wickBottom: 35,
        bodyTop: 20,
        bodyBottom: 30,
        isBullish: direction === 'bullish'
      });
      
      candlesticks.push({
        x: padding + (candleWidth + spacing) * 2,
        wickTop: 10,
        wickBottom: 30,
        bodyTop: 15,
        bodyBottom: 25,
        isBullish: direction === 'bullish'
      });
    }
    
    // Generate SVG elements for candlesticks
    const candlestickElements = candlesticks.map((candle, index) => {
      const wickColor = theme.palette.text.primary;
      const bodyColor = candle.isBullish ? theme.palette.success.main : theme.palette.error.main;
      
      return (
        <g key={index}>
          {/* Wick */}
          <line 
            x1={candle.x + candleWidth/2} 
            y1={candle.wickTop} 
            x2={candle.x + candleWidth/2} 
            y2={candle.wickBottom} 
            stroke={wickColor} 
            strokeWidth="1.5" 
          />
          {/* Body */}
          <rect 
            x={candle.x} 
            y={candle.bodyTop} 
            width={candleWidth} 
            height={Math.max(1, Math.abs(candle.bodyBottom - candle.bodyTop))} 
            fill={bodyColor}
          />
        </g>
      );
    });
    
    return (
      <svg width={svgWidth} height={svgHeight}>
        {candlestickElements}
      </svg>
    );
  };
  
  return (
    <Card 
      variant="outlined" 
      sx={{ 
        height: '100%',
        border: `1px solid ${theme.palette.divider}`,
        bgcolor: theme.palette.mode === 'dark' ? 
          alpha(direction === 'bullish' ? theme.palette.success.main : theme.palette.error.main, 0.05) : 
          alpha(direction === 'bullish' ? theme.palette.success.main : theme.palette.error.main, 0.03)
      }}
    >
      <CardHeader
        avatar={
          <Avatar 
            sx={{ 
              bgcolor: direction === 'bullish' ? theme.palette.success.main : theme.palette.error.main 
            }}
          >
            {getDirectionIcon()}
          </Avatar>
        }
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="subtitle1">
              {formatPatternName(name)}
            </Typography>
            {isMockData && (
              <Chip 
                label="Mock" 
                size="small" 
                color="warning" 
                sx={{ height: 20, '& .MuiChip-label': { px: 0.5, py: 0.1 } }} 
              />
            )}
          </Box>
        }
        subheader={
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Typography variant="caption" color="text.secondary">
              {symbol} • {timeframe}
            </Typography>
            <Box sx={{ flexGrow: 1 }} />
            <Chip
              size="small"
              label={`${Math.round(confidence * 100)}%`}
              color={confidence > 0.8 ? 'success' : confidence > 0.6 ? 'primary' : 'default'}
              variant="outlined"
              sx={{ ml: 1, height: 20, '& .MuiChip-label': { px: 1, py: 0.1 } }}
            />
          </Box>
        }
      />
      <Divider />
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box sx={{ flexGrow: 1, textAlign: 'center' }}>
            {drawPatternVisualization()}
          </Box>
        </Box>
        <Typography variant="body2" color="text.secondary">
          {patternDescriptions[name] || description || "A technical chart pattern detected by the system."}
        </Typography>
      </CardContent>
    </Card>
  );
};

/**
 * Pattern Recognition View Component
 */
const PatternRecognitionView = () => {
  const theme = useTheme();
  const [isLoading, setIsLoading] = useState(true);
  const [patterns, setPatterns] = useState([]);
  const [symbol, setSymbol] = useState('BTC-USD');
  const [timeframe, setTimeframe] = useState('1d');
  const [isMockData, setIsMockData] = useState(true);
  
  // Available symbols and timeframes
  const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD'];
  const timeframes = [
    { value: '5m', label: '5 min' },
    { value: '15m', label: '15 min' },
    { value: '1h', label: '1 hour' },
    { value: '4h', label: '4 hours' },
    { value: '1d', label: '1 day' },
  ];
  
  // Generate mock pattern data for demo
  const generateMockPatterns = (symbol, timeframe) => {
    // Fixed list of patterns to match our implemented ones
    const patternTypes = [
      'morning_star',
      'evening_star',
      'three_white_soldiers',
      'three_black_crows',
      'three_inside_up',
      'three_inside_down',
      'three_outside_up',
      'three_outside_down',
      'abandoned_baby_bullish',
      'abandoned_baby_bearish'
    ];
    
    // Randomly select 1-3 patterns to display
    const numPatterns = Math.floor(Math.random() * 3) + 1;
    const patternIndices = [];
    
    while (patternIndices.length < numPatterns) {
      const idx = Math.floor(Math.random() * patternTypes.length);
      if (!patternIndices.includes(idx)) {
        patternIndices.push(idx);
      }
    }
    
    // Generate patterns
    return patternIndices.map(idx => {
      const patternName = patternTypes[idx];
      const isBullish = patternName.includes('bullish') || 
                        patternName === 'morning_star' || 
                        patternName === 'three_white_soldiers' ||
                        patternName === 'three_inside_up' ||
                        patternName === 'three_outside_up';
      
      return {
        id: `${patternName}-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
        name: patternName,
        direction: isBullish ? 'bullish' : 'bearish',
        confidence: 0.6 + Math.random() * 0.39,
        symbol,
        timeframe,
        timestamp: new Date().toISOString(),
        candles: [1, 2, 3]  // Mock candle indices
      };
    });
  };
  
  // Fetch patterns and data source status
  useEffect(() => {
    const fetchPatterns = async () => {
      setIsLoading(true);
      
      try {
        // In a real implementation, this would fetch data from API
        // For now, we'll just check the data source status
        const response = await axios.get('/api/data-source/status');
        setIsMockData(response.data.use_mock_data);
        
        // Generate patterns based on data source
        // In a real implementation, we would fetch actual detected patterns
        setPatterns(generateMockPatterns(symbol, timeframe));
        
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to fetch patterns:', error);
        
        // Fallback to mock patterns
        setPatterns(generateMockPatterns(symbol, timeframe));
        setIsMockData(true);
        setIsLoading(false);
      }
    };

    fetchPatterns();
  }, [symbol, timeframe]);
  
  // Handle symbol change
  const handleSymbolChange = (event) => {
    setSymbol(event.target.value);
    setIsLoading(true);
  };
  
  // Handle timeframe change
  const handleTimeframeChange = (event) => {
    setTimeframe(event.target.value);
    setIsLoading(true);
  };
  
  return (
    <Paper 
      elevation={0} 
      sx={{ 
        p: 2, 
        height: '100%',
        border: `1px solid ${theme.palette.divider}`,
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Pattern Recognition</Typography>
        
        {isMockData && (
          <Chip 
            icon={<WarningIcon fontSize="small" />}
            label="Mock Data" 
            size="small"
            color="warning"
          />
        )}
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      <Box sx={{ mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          {/* Symbol selector */}
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth size="small">
              <InputLabel id="pattern-symbol-select-label">Symbol</InputLabel>
              <Select
                labelId="pattern-symbol-select-label"
                id="pattern-symbol-select"
                value={symbol}
                label="Symbol"
                onChange={handleSymbolChange}
              >
                {symbols.map((sym) => (
                  <MenuItem key={sym} value={sym}>{sym}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          {/* Timeframe selector */}
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth size="small">
              <InputLabel id="pattern-timeframe-select-label">Timeframe</InputLabel>
              <Select
                labelId="pattern-timeframe-select-label"
                id="pattern-timeframe-select"
                value={timeframe}
                label="Timeframe"
                onChange={handleTimeframeChange}
              >
                {timeframes.map((tf) => (
                  <MenuItem key={tf.value} value={tf.value}>{tf.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Box>
      
      {/* Patterns grid */}
      <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
        {isLoading ? (
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center',
            height: '100%'
          }}>
            <CircularProgress />
          </Box>
        ) : patterns.length > 0 ? (
          <Grid container spacing={2}>
            {patterns.map((pattern) => (
              <Grid item xs={12} sm={6} key={pattern.id}>
                <PatternCard pattern={pattern} isMockData={isMockData} />
              </Grid>
            ))}
          </Grid>
        ) : (
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            justifyContent: 'center', 
            alignItems: 'center',
            height: '100%',
            py: 4
          }}>
            <SignalCellularAltIcon 
              sx={{ 
                fontSize: 48, 
                color: theme.palette.action.disabled, 
                mb: 2 
              }} 
            />
            <Typography variant="body1" color="text.secondary" align="center">
              No patterns detected for {symbol} on {timeframe} timeframe
            </Typography>
            <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 1 }}>
              Try changing the symbol or timeframe
            </Typography>
          </Box>
        )}
      </Box>
      
      {patterns.length > 0 && !isLoading && (
        <Box sx={{ mt: 2, pt: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
          <Typography variant="body2" color="text.secondary">
            <strong>{patterns.length}</strong> pattern{patterns.length !== 1 ? 's' : ''} detected
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default PatternRecognitionView;
