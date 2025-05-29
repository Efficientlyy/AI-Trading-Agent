/**
 * Enhanced Pattern Recognition View
 * 
 * Advanced component for visualizing and interacting with detected chart patterns.
 * Features include pattern filtering, detailed analysis, and visual representations.
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
  DialogActions,
  Rating,
  Stack,
  alpha,
  Alert
} from '@mui/material';

// Icons
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
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
import CloseIcon from '@mui/icons-material/Close';

import axios from 'axios';

// Pattern information database (would be loaded from API in production)
const PATTERN_INFO = {
  'morning_star': {
    name: 'Morning Star',
    description: 'A bullish three-candle pattern that signals a potential reversal from a downtrend to an uptrend.',
    reliability: 4,
    image: 'morning_star.png',
    direction: 'bullish',
    characteristics: [
      'First candle: Large bearish candle',
      'Second candle: Small bodied candle (star)',
      'Third candle: Large bullish candle'
    ],
    tradingStrategy: 'Consider long positions when confirmed by increased volume and support levels.'
  },
  'evening_star': {
    name: 'Evening Star',
    description: 'A bearish three-candle pattern that signals a potential reversal from an uptrend to a downtrend.',
    reliability: 4,
    image: 'evening_star.png',
    direction: 'bearish',
    characteristics: [
      'First candle: Large bullish candle',
      'Second candle: Small bodied candle (star)',
      'Third candle: Large bearish candle'
    ],
    tradingStrategy: 'Consider short positions when confirmed by increased volume and resistance levels.'
  },
  'three_white_soldiers': {
    name: 'Three White Soldiers',
    description: 'A bullish reversal pattern consisting of three consecutive bullish candles, each closing higher than the previous.',
    reliability: 4.5,
    image: 'three_white_soldiers.png',
    direction: 'bullish',
    characteristics: [
      'Three consecutive bullish candles',
      'Each candle opens within the previous candle\'s body',
      'Each candle closes higher than the previous candle'
    ],
    tradingStrategy: 'Strong bullish signal, especially after a downtrend. Consider long positions.'
  },
  'three_black_crows': {
    name: 'Three Black Crows',
    description: 'A bearish reversal pattern consisting of three consecutive bearish candles, each closing lower than the previous.',
    reliability: 4.5,
    image: 'three_black_crows.png',
    direction: 'bearish',
    characteristics: [
      'Three consecutive bearish candles',
      'Each candle opens within the previous candle\'s body',
      'Each candle closes lower than the previous candle'
    ],
    tradingStrategy: 'Strong bearish signal, especially after an uptrend. Consider short positions.'
  }
};

/**
 * Pattern Card Component
 */
const PatternCard = ({ pattern, onViewDetails }) => {
  const theme = useTheme();
  
  // Get pattern info
  const patternInfo = PATTERN_INFO[pattern.pattern] || {
    name: pattern.pattern.replace(/_/g, ' '),
    description: 'No detailed information available for this pattern.',
    reliability: 3,
    direction: pattern.direction,
    characteristics: [],
    tradingStrategy: 'No specific trading strategy available.'
  };
  
  // Convert confidence to rating (0-5 scale)
  const confidenceRating = Math.round(pattern.confidence * 5);
  
  // Determine icon based on direction
  const DirectionIcon = patternInfo.direction === 'bullish' ? TrendingUpIcon : TrendingDownIcon;
  
  // Determine color based on direction
  const directionColor = patternInfo.direction === 'bullish' ? theme.palette.success.main : theme.palette.error.main;
  
  return (
    <Card 
      sx={{ 
        mb: 2,
        borderLeft: `4px solid ${directionColor}`,
        transition: 'transform 0.2s',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: theme.shadows[4]
        }
      }}
    >
      <CardActionArea onClick={() => onViewDetails(pattern)}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Box 
                sx={{ 
                  color: directionColor,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mr: 1
                }}
              >
                <DirectionIcon />
              </Box>
              <Typography variant="h6" component="div">
                {patternInfo.name}
              </Typography>
            </Box>
            
            <Chip 
              label={pattern.direction.charAt(0).toUpperCase() + pattern.direction.slice(1)}
              size="small"
              sx={{ 
                bgcolor: alpha(directionColor, 0.1),
                color: directionColor,
                fontWeight: 'bold'
              }}
            />
          </Box>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {patternInfo.description.substring(0, 100)}
            {patternInfo.description.length > 100 ? '...' : ''}
          </Typography>
          
          <Divider sx={{ my: 1 }} />
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Confidence
              </Typography>
              <Rating 
                value={confidenceRating} 
                readOnly 
                size="small"
                sx={{ display: 'block' }}
              />
            </Box>
            
            <Box>
              <Typography variant="caption" color="text.secondary">
                Detected at position
              </Typography>
              <Typography variant="body2">
                {pattern.position}
              </Typography>
            </Box>
            
            <Box>
              <Typography variant="caption" color="text.secondary">
                Pattern Reliability
              </Typography>
              <Rating 
                value={patternInfo.reliability} 
                readOnly 
                size="small"
                sx={{ display: 'block' }}
              />
            </Box>
          </Box>
        </CardContent>
      </CardActionArea>
    </Card>
  );
};

/**
 * Enhanced Pattern Recognition View Component
 */
const EnhancedPatternRecognitionView = ({ fullscreen = false, onToggleFullscreen = null }) => {
  const theme = useTheme();
  
  // State
  const [symbol, setSymbol] = useState('BTC/USD');
  const [timeframe, setTimeframe] = useState('1h');
  const [patterns, setPatterns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isMockData, setIsMockData] = useState(true);
  const [selectedPattern, setSelectedPattern] = useState(null);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [directionFilter, setDirectionFilter] = useState('all');
  
  // Fetch patterns when symbol or timeframe changes
  useEffect(() => {
    const fetchPatterns = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Get data source status
        const dsResponse = await axios.get('/api/data-source/status');
        setIsMockData(dsResponse.data.use_mock_data);
        
        // Get pattern data
        const response = await axios.get('/api/technical-analysis/patterns', {
          params: {
            symbol,
            timeframe
          }
        });
        
        setPatterns(response.data);
      } catch (err) {
        console.error('Error fetching patterns:', err);
        setError('Failed to load pattern data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchPatterns();
    
    // Set up polling for real-time updates
    const interval = setInterval(() => {
      fetchPatterns();
    }, 60000); // Check every minute
    
    return () => clearInterval(interval);
  }, [symbol, timeframe]);
  
  // Handle view pattern details
  const handleViewDetails = (pattern) => {
    setSelectedPattern(pattern);
    setDetailsDialogOpen(true);
  };
  
  // Handle refresh
  const handleRefresh = () => {
    const fetchPatterns = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await axios.get('/api/technical-analysis/patterns', {
          params: {
            symbol,
            timeframe
          }
        });
        
        setPatterns(response.data);
      } catch (err) {
        console.error('Error refreshing patterns:', err);
        setError('Failed to refresh pattern data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchPatterns();
  };
  
  // Filter patterns by direction
  const filteredPatterns = patterns.filter(pattern => {
    if (directionFilter === 'all') return true;
    return pattern.direction === directionFilter;
  });
  
  // Render pattern details dialog
  const renderPatternDetailsDialog = () => {
    if (!selectedPattern) return null;
    
    const patternInfo = PATTERN_INFO[selectedPattern.pattern] || {
      name: selectedPattern.pattern.replace(/_/g, ' '),
      description: 'No detailed information available for this pattern.',
      reliability: 3,
      direction: selectedPattern.direction,
      characteristics: [],
      tradingStrategy: 'No specific trading strategy available.'
    };
    
    return (
      <Dialog 
        open={detailsDialogOpen} 
        onClose={() => setDetailsDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            {patternInfo.direction === 'bullish' ? 
              <TrendingUpIcon sx={{ mr: 1, color: theme.palette.success.main }} /> : 
              <TrendingDownIcon sx={{ mr: 1, color: theme.palette.error.main }} />
            }
            {patternInfo.name}
          </Box>
          <IconButton onClick={() => setDetailsDialogOpen(false)}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        
        <DialogContent dividers>
          <Grid container spacing={3}>
            <Grid item xs={12} md={7}>
              <Typography variant="subtitle1" gutterBottom>
                Description
              </Typography>
              <Typography variant="body2" paragraph>
                {patternInfo.description}
              </Typography>
              
              <Typography variant="subtitle1" gutterBottom>
                Characteristics
              </Typography>
              <List dense>
                {patternInfo.characteristics.map((char, index) => (
                  <ListItem key={index}>
                    <ListItemIcon sx={{ minWidth: 36 }}>
                      <InfoIcon fontSize="small" color="primary" />
                    </ListItemIcon>
                    <ListItemText primary={char} />
                  </ListItem>
                ))}
              </List>
              
              <Typography variant="subtitle1" gutterBottom>
                Trading Strategy
              </Typography>
              <Typography variant="body2" paragraph>
                {patternInfo.tradingStrategy}
              </Typography>
              
              <Typography variant="subtitle1" gutterBottom>
                Pattern Details
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    Symbol
                  </Typography>
                  <Typography variant="body2">
                    {symbol}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    Timeframe
                  </Typography>
                  <Typography variant="body2">
                    {timeframe}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    Position
                  </Typography>
                  <Typography variant="body2">
                    {selectedPattern.position}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    Confidence
                  </Typography>
                  <Typography variant="body2">
                    {(selectedPattern.confidence * 100).toFixed(1)}%
                  </Typography>
                </Grid>
              </Grid>
            </Grid>
            
            <Grid item xs={12} md={5}>
              <Paper 
                elevation={0} 
                sx={{ 
                  p: 2, 
                  bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  border: `1px dashed ${theme.palette.divider}`
                }}
              >
                <Typography variant="subtitle2" color="text.secondary" align="center" gutterBottom>
                  Pattern Visualization
                </Typography>
                
                <Box sx={{ width: '100%', height: 200, mb: 2, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    Pattern chart visualization would appear here in a production environment.
                  </Typography>
                </Box>
                
                <Box sx={{ width: '100%' }}>
                  <Typography variant="caption" color="text.secondary">
                    Pattern Reliability
                  </Typography>
                  <Rating 
                    value={patternInfo.reliability} 
                    readOnly 
                    sx={{ display: 'block' }}
                  />
                  
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                    Detection Confidence
                  </Typography>
                  <Rating 
                    value={selectedPattern.confidence * 5} 
                    readOnly 
                    sx={{ display: 'block' }}
                  />
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setDetailsDialogOpen(false)}>Close</Button>
          <Button 
            variant="contained" 
            startIcon={<AnalyticsIcon />}
            onClick={() => setDetailsDialogOpen(false)}
          >
            Analyze Trade
          </Button>
        </DialogActions>
      </Dialog>
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
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <AnalyticsIcon sx={{ mr: 1, color: theme.palette.primary.main }} />
          <Typography variant="h6">Pattern Recognition</Typography>
          
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
          <Tooltip title="Refresh Patterns">
            <IconButton onClick={handleRefresh} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Pattern Information">
            <IconButton>
              <HelpOutlineIcon />
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
      
      {/* Controls */}
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
              <MenuItem value="1h">1 Hour</MenuItem>
              <MenuItem value="4h">4 Hours</MenuItem>
              <MenuItem value="1d">1 Day</MenuItem>
              <MenuItem value="1w">1 Week</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={6} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Direction</InputLabel>
            <Select
              value={directionFilter}
              onChange={(e) => setDirectionFilter(e.target.value)}
              label="Direction"
              startAdornment={<FilterListIcon sx={{ ml: 1, mr: -0.5, color: 'action.active' }} />}
            >
              <MenuItem value="all">All Patterns</MenuItem>
              <MenuItem value="bullish">Bullish Only</MenuItem>
              <MenuItem value="bearish">Bearish Only</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>
      
      {/* Content */}
      <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        ) : filteredPatterns.length === 0 ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100%', textAlign: 'center' }}>
            <Typography variant="subtitle1" color="text.secondary" gutterBottom>
              No patterns detected
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {directionFilter !== 'all' 
                ? `No ${directionFilter} patterns found for ${symbol} on ${timeframe} timeframe.` 
                : `No patterns found for ${symbol} on ${timeframe} timeframe.`}
            </Typography>
            <Button 
              startIcon={<RefreshIcon />} 
              onClick={handleRefresh} 
              sx={{ mt: 2 }}
            >
              Refresh
            </Button>
          </Box>
        ) : (
          <Box>
            {/* Summary */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Found {filteredPatterns.length} pattern{filteredPatterns.length !== 1 ? 's' : ''} 
                {directionFilter !== 'all' ? ` (${directionFilter})` : ''} for {symbol} on {timeframe} timeframe.
              </Typography>
            </Box>
            
            {/* Pattern Cards */}
            {filteredPatterns.map((pattern, index) => (
              <PatternCard 
                key={`${pattern.pattern}-${index}`} 
                pattern={pattern} 
                onViewDetails={handleViewDetails}
              />
            ))}
          </Box>
        )}
      </Box>
      
      {/* Pattern Details Dialog */}
      {renderPatternDetailsDialog()}
      
      {/* Mock Data Warning */}
      {isMockData && (
        <Box sx={{ 
          mt: 2, 
          p: 1, 
          borderRadius: 1,
          bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 152, 0, 0.1)' : 'rgba(255, 152, 0, 0.05)',
          border: `1px solid ${theme.palette.warning.light}`
        }}>
          <Typography variant="caption" color="warning.main">
            <WarningIcon sx={{ fontSize: 16, verticalAlign: 'text-bottom', mr: 0.5 }} />
            Currently viewing mock data. Toggle to real data in the header for actual market analysis.
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default EnhancedPatternRecognitionView;
