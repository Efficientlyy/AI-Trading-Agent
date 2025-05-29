/**
 * Technical Chart Viewer Component
 * 
 * This component displays market data with technical indicators and overlays.
 * Enhanced with advanced visualization features.
 */

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Grid,
  Divider,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  CircularProgress,
  useTheme,
  Tooltip,
  Fade,
  Slider,
  Popper,
  ClickAwayListener,
  IconButton,
  Menu,
  Tabs,
  Tab
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import TimelineIcon from '@mui/icons-material/Timeline';
import BarChartIcon from '@mui/icons-material/BarChart';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import CandlestickChartIcon from '@mui/icons-material/CandlestickChart';
import SettingsIcon from '@mui/icons-material/Settings';
import RefreshIcon from '@mui/icons-material/Refresh';
import SaveIcon from '@mui/icons-material/Save';
import DownloadIcon from '@mui/icons-material/Download';
import ZoomInIcon from '@mui/icons-material/ZoomIn';
import ZoomOutIcon from '@mui/icons-material/ZoomOut';
import PanToolIcon from '@mui/icons-material/PanTool';
import TimelapseIcon from '@mui/icons-material/Timelapse';
import FormatPaintIcon from '@mui/icons-material/FormatPaint';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import axios from 'axios';

// Mock library imports - in a real implementation, you would use a charting library
// like TradingView, ApexCharts, Lightweight Charts, etc.
const MockChart = ({ 
  data, 
  chartType, 
  indicators, 
  symbol, 
  timeframe, 
  height, 
  width,
  isMockData
}) => {
  const theme = useTheme();
  const canvasRef = useRef(null);
  
  // Mock chart rendering using HTML5 Canvas
  useEffect(() => {
    if (!canvasRef.current || !data || data.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    // Set canvas size
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw background
    ctx.fillStyle = theme.palette.mode === 'dark' ? '#1e1e1e' : '#f5f5f5';
    ctx.fillRect(0, 0, width, height);
    
    // If no data, show message
    if (!data || data.length === 0) {
      ctx.font = '14px Arial';
      ctx.fillStyle = theme.palette.text.secondary;
      ctx.textAlign = 'center';
      ctx.fillText('No data available', width / 2, height / 2);
      return;
    }
    
    // Draw mock chart
    const padding = { top: 20, right: 50, bottom: 30, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    // Find min/max values for scaling
    let minPrice = Math.min(...data.map(d => d.low));
    let maxPrice = Math.max(...data.map(d => d.high));
    const priceRange = maxPrice - minPrice;
    minPrice = minPrice - priceRange * 0.05;
    maxPrice = maxPrice + priceRange * 0.05;
    
    // Draw time axis
    ctx.strokeStyle = theme.palette.divider;
    ctx.beginPath();
    ctx.moveTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();
    
    // Draw price axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.stroke();
    
    // Draw price labels
    ctx.font = '10px Arial';
    ctx.fillStyle = theme.palette.text.secondary;
    ctx.textAlign = 'right';
    
    const priceSteps = 5;
    for (let i = 0; i <= priceSteps; i++) {
      const price = minPrice + (maxPrice - minPrice) * (i / priceSteps);
      const y = padding.top + chartHeight - (chartHeight * (price - minPrice) / (maxPrice - minPrice));
      
      ctx.fillText(price.toFixed(2), padding.left - 5, y + 4);
      
      // Draw horizontal grid line
      ctx.strokeStyle = alpha(theme.palette.divider, 0.5);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }
    
    // Draw candles or line based on chart type
    const barWidth = Math.max(chartWidth / data.length * 0.8, 1);
    const spacing = (chartWidth / data.length) - barWidth;
    
    if (chartType === 'candlestick') {
      // Draw candlesticks
      data.forEach((candle, i) => {
        const x = padding.left + i * (barWidth + spacing) + spacing/2;
        const open = padding.top + chartHeight - (chartHeight * (candle.open - minPrice) / (maxPrice - minPrice));
        const close = padding.top + chartHeight - (chartHeight * (candle.close - minPrice) / (maxPrice - minPrice));
        const high = padding.top + chartHeight - (chartHeight * (candle.high - minPrice) / (maxPrice - minPrice));
        const low = padding.top + chartHeight - (chartHeight * (candle.low - minPrice) / (maxPrice - minPrice));
        
        // Draw wick
        ctx.strokeStyle = theme.palette.text.primary;
        ctx.beginPath();
        ctx.moveTo(x + barWidth/2, high);
        ctx.lineTo(x + barWidth/2, low);
        ctx.stroke();
        
        // Draw body
        const isBullish = candle.close >= candle.open;
        ctx.fillStyle = isBullish 
          ? theme.palette.success.main 
          : theme.palette.error.main;
          
        const bodyTop = isBullish ? open : close;
        const bodyHeight = Math.abs(close - open);
        
        ctx.fillRect(x, bodyTop, barWidth, bodyHeight);
      });
    } else {
      // Draw line chart
      ctx.strokeStyle = theme.palette.primary.main;
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      data.forEach((point, i) => {
        const x = padding.left + i * (barWidth + spacing) + spacing/2 + barWidth/2;
        const y = padding.top + chartHeight - (chartHeight * (point.close - minPrice) / (maxPrice - minPrice));
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    }
    
    // Draw indicators if enabled
    if (indicators.includes('sma')) {
      // Calculate simple moving average (20-period)
      const period = 20;
      const smaData = [];
      
      for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
          smaData.push(null);
        } else {
          let sum = 0;
          for (let j = 0; j < period; j++) {
            sum += data[i - j].close;
          }
          smaData.push(sum / period);
        }
      }
      
      // Draw SMA line
      ctx.strokeStyle = theme.palette.info.main;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      
      let hasStarted = false;
      
      smaData.forEach((sma, i) => {
        if (sma !== null) {
          const x = padding.left + i * (barWidth + spacing) + spacing/2 + barWidth/2;
          const y = padding.top + chartHeight - (chartHeight * (sma - minPrice) / (maxPrice - minPrice));
          
          if (!hasStarted) {
            ctx.moveTo(x, y);
            hasStarted = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
      });
      
      ctx.stroke();
    }
    
    if (indicators.includes('bollinger')) {
      // Simplified Bollinger Bands calculation (20-period, 2 standard deviations)
      const period = 20;
      const stdDevMultiplier = 2;
      const bollingerData = [];
      
      for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
          bollingerData.push({ middle: null, upper: null, lower: null });
        } else {
          let sum = 0;
          let prices = [];
          
          for (let j = 0; j < period; j++) {
            sum += data[i - j].close;
            prices.push(data[i - j].close);
          }
          
          const sma = sum / period;
          
          // Calculate standard deviation
          let sumSquaredDiff = 0;
          for (const price of prices) {
            sumSquaredDiff += Math.pow(price - sma, 2);
          }
          const stdDev = Math.sqrt(sumSquaredDiff / period);
          
          bollingerData.push({
            middle: sma,
            upper: sma + (stdDev * stdDevMultiplier),
            lower: sma - (stdDev * stdDevMultiplier)
          });
        }
      }
      
      // Draw middle band (SMA)
      ctx.strokeStyle = theme.palette.warning.main;
      ctx.lineWidth = 1;
      ctx.beginPath();
      
      let hasStarted = false;
      
      bollingerData.forEach((band, i) => {
        if (band.middle !== null) {
          const x = padding.left + i * (barWidth + spacing) + spacing/2 + barWidth/2;
          const y = padding.top + chartHeight - (chartHeight * (band.middle - minPrice) / (maxPrice - minPrice));
          
          if (!hasStarted) {
            ctx.moveTo(x, y);
            hasStarted = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
      });
      
      ctx.stroke();
      
      // Draw upper band
      ctx.strokeStyle = alpha(theme.palette.warning.main, 0.7);
      ctx.lineWidth = 1;
      ctx.beginPath();
      
      hasStarted = false;
      
      bollingerData.forEach((band, i) => {
        if (band.upper !== null) {
          const x = padding.left + i * (barWidth + spacing) + spacing/2 + barWidth/2;
          const y = padding.top + chartHeight - (chartHeight * (band.upper - minPrice) / (maxPrice - minPrice));
          
          if (!hasStarted) {
            ctx.moveTo(x, y);
            hasStarted = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
      });
      
      ctx.stroke();
      
      // Draw lower band
      ctx.beginPath();
      
      hasStarted = false;
      
      bollingerData.forEach((band, i) => {
        if (band.lower !== null) {
          const x = padding.left + i * (barWidth + spacing) + spacing/2 + barWidth/2;
          const y = padding.top + chartHeight - (chartHeight * (band.lower - minPrice) / (maxPrice - minPrice));
          
          if (!hasStarted) {
            ctx.moveTo(x, y);
            hasStarted = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
      });
      
      ctx.stroke();
    }
    
    // Draw symbol and timeframe info
    ctx.font = 'bold 12px Arial';
    ctx.fillStyle = theme.palette.text.primary;
    ctx.textAlign = 'left';
    ctx.fillText(`${symbol} - ${timeframe}`, padding.left, padding.top - 5);
    
    // Draw mock data indicator if using mock data
    if (isMockData) {
      ctx.font = 'bold 12px Arial';
      ctx.fillStyle = theme.palette.warning.main;
      ctx.textAlign = 'right';
      ctx.fillText('MOCK DATA', width - padding.right, padding.top - 5);
    }
    
  }, [data, chartType, indicators, theme, width, height, isMockData, symbol, timeframe]);
  
  return (
    <canvas 
      ref={canvasRef} 
      style={{ width: '100%', height: '100%' }}
    />
  );
};

/**
 * Technical Chart Viewer Component
 */
const TechnicalChartViewer = () => {
  const theme = useTheme();
  const [isLoading, setIsLoading] = useState(true);
  const [chartData, setChartData] = useState([]);
  const [symbol, setSymbol] = useState('BTC-USD');
  const [timeframe, setTimeframe] = useState('1d');
  const [chartType, setChartType] = useState('candlestick');
  const [isMockData, setIsMockData] = useState(true);
  const [indicators, setIndicators] = useState(['sma']);
  
  // Available symbols and timeframes
  const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD'];
  const timeframes = [
    { value: '5m', label: '5 min' },
    { value: '15m', label: '15 min' },
    { value: '1h', label: '1 hour' },
    { value: '4h', label: '4 hours' },
    { value: '1d', label: '1 day' },
  ];
  
  // Available indicators
  const availableIndicators = [
    { value: 'sma', label: 'SMA (20)' },
    { value: 'ema', label: 'EMA (20)' },
    { value: 'bollinger', label: 'Bollinger Bands' },
    { value: 'rsi', label: 'RSI (14)' },
    { value: 'macd', label: 'MACD' }
  ];
  
  // Generate mock chart data
  const generateMockData = (symbol, timeframe, dataPoints = 100) => {
    const data = [];
    let basePrice = symbol === 'BTC-USD' ? 30000 : 
                    symbol === 'ETH-USD' ? 2000 :
                    symbol === 'SOL-USD' ? 80 :
                    symbol === 'ADA-USD' ? 0.5 : 0.6;
    
    let volatility = 0.02;
    let trendStrength = 0.3;
    
    // Generate random seed based on symbol and timeframe
    const seed = symbol.charCodeAt(0) + timeframe.charCodeAt(0);
    const random = (min, max) => {
      const x = Math.sin(seed + data.length) * 10000;
      const r = x - Math.floor(x);
      return min + r * (max - min);
    };
    
    // Generate data with some trend and volatility
    let trend = random(-1, 1) > 0 ? 1 : -1;
    let price = basePrice;
    
    for (let i = 0; i < dataPoints; i++) {
      // Switch trend occasionally
      if (random(0, 1) < 0.05) {
        trend = -trend;
      }
      
      // Calculate price change with trend and volatility
      const change = price * (random(-volatility, volatility) + trend * trendStrength * 0.01);
      price += change;
      
      // Generate OHLC data
      const open = price;
      const close = price + price * random(-volatility/2, volatility/2);
      const high = Math.max(open, close) + price * random(0, volatility);
      const low = Math.min(open, close) - price * random(0, volatility);
      
      data.push({
        time: new Date(Date.now() - (dataPoints - i) * getTimeframeMinutes(timeframe) * 60 * 1000),
        open,
        high,
        close,
        low,
        volume: basePrice * 10 * random(0.5, 1.5)
      });
    }
    
    return data;
  };
  
  // Helper to convert timeframe to minutes
  const getTimeframeMinutes = (tf) => {
    switch (tf) {
      case '5m': return 5;
      case '15m': return 15;
      case '1h': return 60;
      case '4h': return 240;
      case '1d': return 1440;
      default: return 60;
    }
  };
  
  // Fetch chart data and data source status
  useEffect(() => {
    const fetchChartData = async () => {
      setIsLoading(true);
      
      try {
        // In a real implementation, this would fetch data from API
        // For now, we'll just check the data source status
        const response = await axios.get('/api/data-source/status');
        setIsMockData(response.data.use_mock_data);
        
        // Generate or fetch data based on data source
        if (response.data.use_mock_data) {
          // Generate mock data
          setChartData(generateMockData(symbol, timeframe));
        } else {
          // In a real implementation, we would fetch real market data
          // For now, just use mock data but with different parameters
          setChartData(generateMockData(symbol, timeframe, 120));
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to fetch chart data:', error);
        
        // Fallback to mock data
        setChartData(generateMockData(symbol, timeframe));
        setIsMockData(true);
        setIsLoading(false);
      }
    };

    fetchChartData();
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
  
  // Handle chart type change
  const handleChartTypeChange = (event, newType) => {
    if (newType !== null) {
      setChartType(newType);
    }
  };
  
  // Handle indicator toggle
  const handleIndicatorChange = (event) => {
    const indicator = event.target.value;
    
    if (indicators.includes(indicator)) {
      setIndicators(indicators.filter(i => i !== indicator));
    } else {
      setIndicators([...indicators, indicator]);
    }
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
        <Typography variant="h6">Technical Analysis</Typography>
        
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
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center">
          {/* Symbol selector */}
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel id="symbol-select-label">Symbol</InputLabel>
            <Select
              labelId="symbol-select-label"
              id="symbol-select"
              value={symbol}
              label="Symbol"
              onChange={handleSymbolChange}
            >
              {symbols.map((sym) => (
                <MenuItem key={sym} value={sym}>{sym}</MenuItem>
              ))}
            </Select>
          </FormControl>
          
          {/* Timeframe selector */}
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel id="timeframe-select-label">Timeframe</InputLabel>
            <Select
              labelId="timeframe-select-label"
              id="timeframe-select"
              value={timeframe}
              label="Timeframe"
              onChange={handleTimeframeChange}
            >
              {timeframes.map((tf) => (
                <MenuItem key={tf.value} value={tf.value}>{tf.label}</MenuItem>
              ))}
            </Select>
          </FormControl>
          
          {/* Chart type toggle */}
          <ToggleButtonGroup
            value={chartType}
            exclusive
            onChange={handleChartTypeChange}
            size="small"
          >
            <ToggleButton value="candlestick">
              <CandlestickChartIcon fontSize="small" />
            </ToggleButton>
            <ToggleButton value="line">
              <TimelineIcon fontSize="small" />
            </ToggleButton>
          </ToggleButtonGroup>
        </Stack>
      </Box>
      
      {/* Indicators selector */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Indicators
        </Typography>
        <Stack direction="row" spacing={2} flexWrap="wrap">
          {availableIndicators.map((indicator) => (
            <FormControlLabel
              key={indicator.value}
              control={
                <Checkbox 
                  size="small"
                  checked={indicators.includes(indicator.value)} 
                  onChange={handleIndicatorChange}
                  value={indicator.value}
                />
              }
              label={indicator.label}
            />
          ))}
        </Stack>
      </Box>
      
      {/* Chart container */}
      <Box sx={{ flexGrow: 1, minHeight: 0, position: 'relative' }}>
        {isLoading ? (
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center',
            height: '100%'
          }}>
            <CircularProgress />
          </Box>
        ) : (
          <MockChart
            data={chartData}
            chartType={chartType}
            indicators={indicators}
            symbol={symbol}
            timeframe={timeframe}
            height={400}
            width={800}
            isMockData={isMockData}
          />
        )}
      </Box>
    </Paper>
  );
};

export default TechnicalChartViewer;
