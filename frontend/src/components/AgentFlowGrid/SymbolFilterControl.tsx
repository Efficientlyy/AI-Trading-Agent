import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Box,
  Chip,
  Typography,
  Autocomplete,
  TextField,
  Skeleton,
  Tooltip,
  Paper,
  IconButton,
  useTheme,
  useMediaQuery
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';
import FilterListIcon from '@mui/icons-material/FilterList';
import ClearIcon from '@mui/icons-material/Clear';
import sentimentAnalyticsService from '../../api/sentimentAnalyticsService';

interface SymbolSentimentInfo {
  symbol: string;
  latest_sentiment: number;
  sentiment_change: number;
  confidence: number;
}

interface SymbolFilterControlProps {
  agentId: string;
  selectedSymbol: string | null;
  onSymbolChange: (symbol: string | null) => void;
}

const SymbolFilterControl: React.FC<SymbolFilterControlProps> = ({
  agentId,
  selectedSymbol,
  onSymbolChange
}) => {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  const [isLoading, setIsLoading] = useState(false);
  const [symbols, setSymbols] = useState<SymbolSentimentInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Memoized fetch function to avoid unnecessary recreations
  const fetchSymbols = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await sentimentAnalyticsService.getMonitoredSymbolsWithSentiment(agentId);
      setSymbols(data);
    } catch (err) {
      console.error('Error fetching symbols:', err);
      setError('Failed to load symbols');
    } finally {
      setIsLoading(false);
    }
  }, [agentId]);
  
  // Fetch symbols only when dependencies change
  useEffect(() => {
    fetchSymbols();
  }, [fetchSymbols]);
  
  // Clear the selected symbol
  const handleClearSelection = useCallback(() => {
    onSymbolChange(null);
  }, [onSymbolChange]);

  // Helper function to get color based on sentiment value
  const getSentimentColor = useCallback((sentiment: number) => {
    if (sentiment > 0.2) return '#4caf50';
    if (sentiment < -0.2) return '#f44336';
    if (sentiment > 0) return '#8bc34a';
    if (sentiment < 0) return '#ff9800';
    return '#9e9e9e';
  }, []);

  // Helper function to get icon based on sentiment change
  const getSentimentChangeIcon = useCallback((change: number) => {
    if (change > 0.05) {
      return <TrendingUpIcon fontSize="small" sx={{ color: '#4caf50' }} />;
    } else if (change < -0.05) {
      return <TrendingDownIcon fontSize="small" sx={{ color: '#f44336' }} />;
    } else {
      return <TrendingFlatIcon fontSize="small" sx={{ color: '#ff9800' }} />;
    }
  }, []);

  // Memoized symbol options for better performance
  const symbolOptions = useMemo(() => {
    return symbols.map(s => s.symbol);
  }, [symbols]);

  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        mb: 3,
        borderRadius: 2,
        bgcolor: isDarkMode ? 'rgba(30, 30, 30, 0.7)' : '#fff'
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <FilterListIcon color="primary" fontSize={isMobile ? "small" : "medium"} />
          <Typography variant={isMobile ? "subtitle2" : "subtitle1"} fontWeight={500}>
            Symbol Filter
          </Typography>
        </Box>
        
        {selectedSymbol && (
          <Tooltip title="Clear filter">
            <IconButton 
              size="small" 
              onClick={handleClearSelection}
              sx={{ 
                bgcolor: isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                '&:hover': {
                  bgcolor: isDarkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'
                }
              }}
            >
              <ClearIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
      </Box>
      
      <Box sx={{ mb: 2 }}>
        <Autocomplete
          id="symbol-autocomplete"
          value={selectedSymbol}
          options={symbolOptions}
          onChange={(_, newValue) => onSymbolChange(newValue)}
          loading={isLoading}
          loadingText="Loading symbols..."
          noOptionsText="No symbols available"
          sx={{
            '& .MuiOutlinedInput-root': {
              '& fieldset': {
                borderColor: isDarkMode ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)',
              },
              '&:hover fieldset': {
                borderColor: 'primary.main',
              },
            },
            '& .MuiInputLabel-root': {
              color: isDarkMode ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.6)'
            }
          }}
          renderInput={(params) => (
            <TextField 
              {...params} 
              label="Filter by Symbol" 
              variant="outlined" 
              size={isMobile ? "small" : "medium"}
              placeholder="Select symbol..."
              fullWidth
              error={!!error}
              helperText={error}
            />
          )}
        />
      </Box>
      
      <Typography variant="body2" sx={{ mb: 1, color: isDarkMode ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.7)' }}>
        Monitored Assets
      </Typography>
      
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
        {isLoading ? (
          Array.from(new Array(6)).map((_, index) => (
            <Chip
              key={index}
              label={`...`}
              sx={{ 
                bgcolor: isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                opacity: 0.5
              }}
            />
          ))
        ) : error ? (
          <Typography variant="body2" color="error" sx={{ fontSize: '0.8rem' }}>
            {error}
          </Typography>
        ) : (
          symbols.map((symbol) => (
            <Tooltip 
              key={symbol.symbol}
              title={`Sentiment: ${symbol.latest_sentiment.toFixed(2)} (${symbol.sentiment_change > 0 ? '+' : ''}${symbol.sentiment_change.toFixed(2)}) | Confidence: ${(symbol.confidence * 100).toFixed(0)}%`}
              arrow
            >
              <Chip
                label={symbol.symbol}
                onClick={() => onSymbolChange(symbol.symbol)}
                variant={selectedSymbol === symbol.symbol ? 'filled' : 'outlined'}
                color={selectedSymbol === symbol.symbol ? 'primary' : 'default'}
                icon={getSentimentChangeIcon(symbol.sentiment_change)}
                sx={{ 
                  color: selectedSymbol === symbol.symbol 
                    ? 'white' 
                    : getSentimentColor(symbol.latest_sentiment),
                  borderColor: getSentimentColor(symbol.latest_sentiment),
                  '&:hover': {
                    bgcolor: isDarkMode 
                      ? 'rgba(255,255,255,0.1)' 
                      : 'rgba(0,0,0,0.05)'
                  }
                }}
              />
            </Tooltip>
          ))
        )}
      </Box>
    </Paper>
  );
};

// Wrap with React.memo to prevent unnecessary re-renders
export default React.memo(SymbolFilterControl);