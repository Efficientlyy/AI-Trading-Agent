/**
 * Enhanced Technical Chart Viewer Component
 * 
 * This component provides advanced technical chart visualization with interactive
 * features, multiple chart types, and customizable indicator overlays.
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
  Tab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  Stack,
  alpha,
  Collapse
} from '@mui/material';

// Icons
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
import WarningIcon from '@mui/icons-material/Warning';
import CloseIcon from '@mui/icons-material/Close';
import InfoIcon from '@mui/icons-material/Info';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';

// Utility imports
import axios from 'axios';
import debounce from 'lodash/debounce';

// Define available indicators with configuration options
const AVAILABLE_INDICATORS = [
  { 
    id: 'rsi', 
    name: 'RSI', 
    category: 'momentum',
    color: '#5D69B1',
    params: [
      { name: 'period', label: 'Period', type: 'number', default: 14, min: 2, max: 50 },
      { name: 'overbought', label: 'Overbought', type: 'number', default: 70, min: 50, max: 90 },
      { name: 'oversold', label: 'Oversold', type: 'number', default: 30, min: 10, max: 50 }
    ]
  },
  { 
    id: 'macd', 
    name: 'MACD', 
    category: 'trend',
    color: '#52BCA3',
    params: [
      { name: 'fastPeriod', label: 'Fast Period', type: 'number', default: 12, min: 2, max: 50 },
      { name: 'slowPeriod', label: 'Slow Period', type: 'number', default: 26, min: 5, max: 100 },
      { name: 'signalPeriod', label: 'Signal Period', type: 'number', default: 9, min: 2, max: 50 }
    ]
  },
  { 
    id: 'bollinger_bands', 
    name: 'Bollinger Bands', 
    category: 'volatility',
    color: '#E5AE38',
    params: [
      { name: 'period', label: 'Period', type: 'number', default: 20, min: 5, max: 100 },
      { name: 'stdDev', label: 'Standard Deviations', type: 'number', default: 2, min: 1, max: 5, step: 0.5 }
    ]
  },
  { 
    id: 'moving_average', 
    name: 'Moving Average', 
    category: 'trend',
    color: '#6689C6',
    params: [
      { name: 'period', label: 'Period', type: 'number', default: 20, min: 2, max: 200 },
      { name: 'type', label: 'Type', type: 'select', default: 'sma', options: [
        { value: 'sma', label: 'Simple' },
        { value: 'ema', label: 'Exponential' },
        { value: 'wma', label: 'Weighted' }
      ]}
    ]
  },
  { 
    id: 'stochastic', 
    name: 'Stochastic', 
    category: 'momentum',
    color: '#D75487',
    params: [
      { name: 'kPeriod', label: 'K Period', type: 'number', default: 14, min: 2, max: 50 },
      { name: 'dPeriod', label: 'D Period', type: 'number', default: 3, min: 1, max: 20 },
      { name: 'slowing', label: 'Slowing', type: 'number', default: 3, min: 1, max: 20 }
    ]
  }
];

// Available timeframes
const TIMEFRAMES = [
  { value: '1m', label: '1 Minute' },
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '30m', label: '30 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: '1 Day' },
  { value: '1w', label: '1 Week' }
];

// Available chart types
const CHART_TYPES = [
  { value: 'candle', label: 'Candlestick', icon: <CandlestickChartIcon /> },
  { value: 'line', label: 'Line', icon: <ShowChartIcon /> },
  { value: 'bar', label: 'Bar', icon: <BarChartIcon /> },
  { value: 'area', label: 'Area', icon: <TimelineIcon /> }
];

// Chart tools
const CHART_TOOLS = [
  { value: 'pan', label: 'Pan', icon: <PanToolIcon /> },
  { value: 'zoom', label: 'Zoom', icon: <ZoomInIcon /> },
  { value: 'draw', label: 'Draw', icon: <FormatPaintIcon /> },
  { value: 'measure', label: 'Measure', icon: <TimelapseIcon /> }
];

/**
 * Enhanced Technical Chart Viewer Component
 */
const EnhancedTechnicalChartViewer = ({ fullscreen = false, onToggleFullscreen = null }) => {
  const theme = useTheme();
  const chartRef = useRef(null);
  
  // State
  const [symbol, setSymbol] = useState('BTC/USD');
  const [timeframe, setTimeframe] = useState('1h');
  const [chartType, setChartType] = useState('candle');
  const [activeTool, setActiveTool] = useState('pan');
  const [indicators, setIndicators] = useState([
    { id: 'rsi', params: { period: 14, overbought: 70, oversold: 30 } },
    { id: 'bollinger_bands', params: { period: 20, stdDev: 2 } }
  ]);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [indicatorDialogOpen, setIndicatorDialogOpen] = useState(false);
  const [selectedIndicator, setSelectedIndicator] = useState(null);
  const [isMockData, setIsMockData] = useState(false);
  const [settingsMenuAnchor, setSettingsMenuAnchor] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [chartTheme, setChartTheme] = useState(theme.palette.mode);
  const [zoomLevel, setZoomLevel] = useState(1);

  // Helper for getting indicator details
  const getIndicatorDetails = (id) => {
    return AVAILABLE_INDICATORS.find(indicator => indicator.id === id);
  };
  
  // Fetch chart data when symbol or timeframe changes
  useEffect(() => {
    const fetchChartData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Get data source status
        const dsResponse = await axios.get('/api/data-source/status');
        setIsMockData(dsResponse.data.use_mock_data);
        
        // Get technical analysis data
        const taResponse = await axios.get('/api/technical-analysis/analysis', {
          params: {
            symbol,
            timeframe,
            indicators: indicators.map(ind => ind.id)
          }
        });
        
        setChartData(taResponse.data);
      } catch (err) {
        console.error('Error fetching chart data:', err);
        setError('Failed to load chart data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchChartData();
    
    // Set up polling for real-time updates (every 30 seconds)
    const interval = setInterval(() => {
      fetchChartData();
    }, 30000);
    
    return () => clearInterval(interval);
  }, [symbol, timeframe, indicators]);
  
  // Handle indicator dialog open
  const handleOpenIndicatorDialog = () => {
    setSelectedIndicator(null);
    setIndicatorDialogOpen(true);
  };
  
  // Handle indicator selection
  const handleSelectIndicator = (indicatorId) => {
    const indicatorInfo = getIndicatorDetails(indicatorId);
    
    // Create default params
    const defaultParams = {};
    indicatorInfo.params.forEach(param => {
      defaultParams[param.name] = param.default;
    });
    
    setSelectedIndicator({
      id: indicatorId,
      params: defaultParams
    });
  };
  
  // Handle indicator parameter change
  const handleParamChange = (paramName, value) => {
    if (!selectedIndicator) return;
    
    setSelectedIndicator({
      ...selectedIndicator,
      params: {
        ...selectedIndicator.params,
        [paramName]: value
      }
    });
  };
  
  // Add selected indicator
  const handleAddIndicator = () => {
    if (!selectedIndicator) return;
    
    setIndicators([...indicators, selectedIndicator]);
    setIndicatorDialogOpen(false);
  };
  
  // Remove indicator
  const handleRemoveIndicator = (indicatorId) => {
    setIndicators(indicators.filter(ind => ind.id !== indicatorId));
  };
  
  // Handle chart type change
  const handleChartTypeChange = (event, newType) => {
    if (newType !== null) {
      setChartType(newType);
    }
  };
  
  // Handle active tool change
  const handleToolChange = (event, newTool) => {
    if (newTool !== null) {
      setActiveTool(newTool);
    }
  };
  
  // Handle zoom in/out
  const handleZoom = (direction) => {
    setZoomLevel(prev => {
      const newZoom = direction === 'in' ? prev * 1.2 : prev / 1.2;
      // Clamp between 0.5 and 3
      return Math.max(0.5, Math.min(3, newZoom));
    });
  };
  
  // Handle settings menu
  const handleOpenSettings = (event) => {
    setSettingsMenuAnchor(event.currentTarget);
  };
  
  const handleCloseSettings = () => {
    setSettingsMenuAnchor(null);
  };
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Handle theme change
  const handleThemeChange = (event) => {
    setChartTheme(event.target.value);
  };
  
  // Handle refresh
  const handleRefresh = () => {
    // Refetch data
    const fetchChartData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await axios.get('/api/technical-analysis/analysis', {
          params: {
            symbol,
            timeframe,
            indicators: indicators.map(ind => ind.id)
          }
        });
        
        setChartData(response.data);
      } catch (err) {
        console.error('Error refreshing chart data:', err);
        setError('Failed to refresh chart data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchChartData();
  };

  // Handle export
  const handleExport = () => {
    if (!chartData) return;
    
    // Create data for export
    const exportData = {
      metadata: {
        symbol,
        timeframe,
        timestamp: new Date().toISOString(),
        data_source: isMockData ? 'mock' : 'real'
      },
      data: chartData
    };
    
    // Create blob and download
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `${symbol.replace('/', '_')}_${timeframe}_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  // Render indicator parameters
  const renderIndicatorParams = () => {
    if (!selectedIndicator) return null;
    
    const indicatorInfo = getIndicatorDetails(selectedIndicator.id);
    
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Parameters for {indicatorInfo.name}
        </Typography>
        
        <Grid container spacing={2}>
          {indicatorInfo.params.map(param => (
            <Grid item xs={12} sm={6} key={param.name}>
              {param.type === 'select' ? (
                <FormControl fullWidth size="small">
                  <InputLabel>{param.label}</InputLabel>
                  <Select
                    value={selectedIndicator.params[param.name] || param.default}
                    onChange={(e) => handleParamChange(param.name, e.target.value)}
                    label={param.label}
                  >
                    {param.options.map(option => (
                      <MenuItem value={option.value} key={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              ) : (
                <TextField
                  fullWidth
                  label={param.label}
                  type="number"
                  size="small"
                  value={selectedIndicator.params[param.name] || param.default}
                  onChange={(e) => handleParamChange(param.name, Number(e.target.value))}
                  inputProps={{
                    min: param.min,
                    max: param.max,
                    step: param.step || 1
                  }}
                />
              )}
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };
  
  // Render mock data placeholder chart
  const renderMockChartPlaceholder = () => {
    return (
      <Box 
        sx={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          textAlign: 'center',
          bgcolor: alpha(theme.palette.background.paper, 0.7),
          borderRadius: 1,
          p: 3
        }}
      >
        <Typography variant="h6" color="text.secondary" gutterBottom>
          {loading ? 'Loading Chart Data...' : 'Chart Visualization'}
        </Typography>
        
        {loading ? (
          <CircularProgress size={40} />
        ) : (
          <>
            <Box 
              sx={{ 
                width: '100%', 
                height: 200, 
                mt: 2, 
                mb: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: `1px dashed ${theme.palette.divider}`,
                borderRadius: 1
              }}
            >
              <Typography variant="body2" color="text.secondary">
                {isMockData ? 
                  'Using simulated market data for visualization' : 
                  'Chart visualization will appear here'}
              </Typography>
            </Box>
            
            <Typography variant="body2" color="text.secondary">
              {error || 'In a production environment, this would display an interactive chart using a library like TradingView or Highcharts.'}
            </Typography>
            
            {isMockData && (
              <Alert 
                severity="warning" 
                icon={<WarningIcon />}
                sx={{ mt: 2, width: '100%' }}
              >
                Currently using mock data. Toggle to real data in the header for actual market analysis.
              </Alert>
            )}
          </>
        )}
      </Box>
    );
  };

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        p: 2, 
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${theme.palette.divider}`
      }}
    >
      {/* Chart Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <CandlestickChartIcon sx={{ mr: 1, color: theme.palette.primary.main }} />
          <Typography variant="h6">Technical Chart</Typography>
          
          {isMockData && (
            <Chip 
              icon={<WarningIcon fontSize="small" />}
              label="Mock Data" 
              size="small"
              color="warning"
              sx={{ ml: 2 }}
            />
          )}
        </Box>
        
        <Box>
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Chart Settings">
            <IconButton onClick={handleOpenSettings}>
              <SettingsIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Export Data">
            <IconButton onClick={handleExport} disabled={!chartData}>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
          
          {onToggleFullscreen && (
            <Tooltip title={fullscreen ? "Exit Fullscreen" : "Fullscreen"}>
              <IconButton onClick={onToggleFullscreen}>
                {fullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
              </IconButton>
            </Tooltip>
          )}
        </Box>
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      {/* Chart Controls */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Symbol</InputLabel>
            <Select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              label="Symbol"
            >
              <MenuItem value="BTC/USD">BTC/USD</MenuItem>
              <MenuItem value="ETH/USD">ETH/USD</MenuItem>
              <MenuItem value="AAPL">AAPL</MenuItem>
              <MenuItem value="MSFT">MSFT</MenuItem>
              <MenuItem value="GOOG">GOOG</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={6} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              label="Timeframe"
            >
              {TIMEFRAMES.map(tf => (
                <MenuItem value={tf.value} key={tf.value}>
                  {tf.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={6} sm={4}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <ToggleButtonGroup
              value={chartType}
              exclusive
              onChange={handleChartTypeChange}
              size="small"
              aria-label="chart type"
            >
              {CHART_TYPES.map(type => (
                <ToggleButton value={type.value} key={type.value} aria-label={type.label}>
                  <Tooltip title={type.label}>
                    <Box>{type.icon}</Box>
                  </Tooltip>
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
            
            <Box sx={{ ml: 1 }}>
              <Tooltip title="Add Indicator">
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  size="small"
                  onClick={handleOpenIndicatorDialog}
                >
                  Indicator
                </Button>
              </Tooltip>
            </Box>
          </Box>
        </Grid>
      </Grid>
      
      {/* Active Indicators */}
      {indicators.length > 0 && (
        <Box sx={{ mb: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {indicators.map(indicator => {
            const details = getIndicatorDetails(indicator.id);
            return (
              <Chip
                key={indicator.id}
                label={details.name}
                onDelete={() => handleRemoveIndicator(indicator.id)}
                size="small"
                sx={{ 
                  borderLeft: `3px solid ${details.color}`,
                  pl: 0.5
                }}
              />
            );
          })}
        </Box>
      )}
      
      {/* Chart Drawing Tools */}
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <ToggleButtonGroup
          value={activeTool}
          exclusive
          onChange={handleToolChange}
          size="small"
          aria-label="chart tools"
        >
          {CHART_TOOLS.map(tool => (
            <ToggleButton value={tool.value} key={tool.value} aria-label={tool.label}>
              <Tooltip title={tool.label}>
                <Box>{tool.icon}</Box>
              </Tooltip>
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Tooltip title="Zoom Out">
            <IconButton size="small" onClick={() => handleZoom('out')}>
              <ZoomOutIcon />
            </IconButton>
          </Tooltip>
          
          <Typography variant="body2" sx={{ mx: 1 }}>
            {Math.round(zoomLevel * 100)}%
          </Typography>
          
          <Tooltip title="Zoom In">
            <IconButton size="small" onClick={() => handleZoom('in')}>
              <ZoomInIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      {/* Chart Area */}
      <Box 
        ref={chartRef}
        sx={{ 
          flexGrow: 1,
          border: `1px solid ${theme.palette.divider}`,
          borderRadius: 1,
          overflow: 'hidden',
          position: 'relative'
        }}
      >
        {renderMockChartPlaceholder()}
      </Box>
      
      {/* Indicator Dialog */}
      <Dialog
        open={indicatorDialogOpen}
        onClose={() => setIndicatorDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Add Technical Indicator
          <IconButton
            onClick={() => setIndicatorDialogOpen(false)}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        
        <DialogContent dividers>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="Trend" />
            <Tab label="Momentum" />
            <Tab label="Volatility" />
            <Tab label="Volume" />
          </Tabs>
          
          <Box sx={{ mt: 2 }}>
            {/* Filter indicators by category based on active tab */}
            <Grid container spacing={1}>
              {AVAILABLE_INDICATORS
                .filter(ind => {
                  const categories = ['trend', 'momentum', 'volatility', 'volume'];
                  return ind.category === categories[activeTab];
                })
                .map(indicator => (
                  <Grid item xs={6} key={indicator.id}>
                    <Button
                      variant={selectedIndicator?.id === indicator.id ? "contained" : "outlined"}
                      onClick={() => handleSelectIndicator(indicator.id)}
                      fullWidth
                      sx={{ justifyContent: 'flex-start', mb: 1 }}
                    >
                      {indicator.name}
                    </Button>
                  </Grid>
                ))}
            </Grid>
            
            {/* Render parameters for selected indicator */}
            {renderIndicatorParams()}
          </Box>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setIndicatorDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleAddIndicator}
            variant="contained" 
            disabled={!selectedIndicator}
          >
            Add Indicator
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Settings Menu */}
      <Menu
        anchorEl={settingsMenuAnchor}
        open={Boolean(settingsMenuAnchor)}
        onClose={handleCloseSettings}
      >
        <MenuItem>
          <FormControl fullWidth size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Chart Theme</InputLabel>
            <Select
              value={chartTheme}
              onChange={handleThemeChange}
              label="Chart Theme"
            >
              <MenuItem value="light">Light</MenuItem>
              <MenuItem value="dark">Dark</MenuItem>
              <MenuItem value="custom">Custom</MenuItem>
            </Select>
          </FormControl>
        </MenuItem>
        
        <MenuItem onClick={handleCloseSettings}>
          Show Volume
        </MenuItem>
        
        <MenuItem onClick={handleCloseSettings}>
          Show Grid Lines
        </MenuItem>
        
        <MenuItem onClick={handleCloseSettings}>
          Auto-scale Chart
        </MenuItem>
      </Menu>
    </Paper>
  );
};

export default EnhancedTechnicalChartViewer;
