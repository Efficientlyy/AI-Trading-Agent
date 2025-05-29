import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Box,
  Chip,
  Typography,
  Autocomplete,
  TextField,
  Skeleton,
  useTheme,
  useMediaQuery
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';
import sentimentAnalyticsService from '../../api/sentimentAnalyticsService';

interface SymbolSentimentInfo {
  symbol: string;
  sentiment: number;
  change: number;
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
      
      const mappedData = data.map(item => ({
        symbol: item.symbol,
        sentiment: item.latest_sentiment,
        change: item.sentiment_change,
        confidence: item.confidence
      }));
      
      setSymbols(mappedData);
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

  // Get trend icon for sentiment visualization
  const getTrendIcon = useCallback((change: number) => {
    if (change > 0.02) return <TrendingUpIcon sx={{ color: '#4caf50' }} />;
    if (change < -0.02) return <TrendingDownIcon sx={{ color: '#f44336' }} />;
    return <TrendingFlatIcon sx={{ color: '#ff9800' }} />;
  }, []);

  // Get color based on sentiment value
  const getSentimentColor = useCallback((sentiment: number) => {
    if (sentiment > 0.2) return '#4caf50';
    if (sentiment < -0.2) return '#f44336';
    if (sentiment > 0) return '#8bc34a';
    if (sentiment < 0) return '#ff9800';
    return '#9e9e9e';
  }, []);

  // Memoized symbol options for the Autocomplete
  const symbolOptions = useMemo(() => 
    symbols.map(s => s.symbol),
    [symbols]
  );

  return (
    <Box sx={{ mb: 3 }}>
      <Typography variant="subtitle1" gutterBottom>
        Filter by Symbol
      </Typography>
      
      <Box sx={{ mb: 2 }}>
        <Autocomplete
          id="symbol-autocomplete"
          value={selectedSymbol}
          options={symbolOptions}
          onChange={(_, newValue) => onSymbolChange(newValue)}
          loading={isLoading}
          sx={{ width: '100%' }}
          renderInput={(params) => (
            <TextField 
              {...params} 
              label="Symbol" 
              variant="outlined" 
              size="small"
              error={!!error}
              helperText={error}
              InputProps={{
                ...params.InputProps,
                endAdornment: (
                  <React.Fragment>
                    {isLoading ? <Skeleton width={20} height={20} /> : null}
                    {params.InputProps.endAdornment}
                  </React.Fragment>
                ),
              }}
            />
          )}
        />
      </Box>
      
      <Typography variant="caption" sx={{ display: 'block', mb: 1, color: isDarkMode ? '#aaa' : '#666' }}>
        Monitored Symbols & Sentiment
      </Typography>
      
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
        {isLoading ? (
          Array.from(new Array(6)).map((_, index) => (
            <Chip
              key={index}
              label={<Skeleton width={40} />}
              size="small"
              sx={{ opacity: 0.5 }}
            />
          ))
        ) : error ? (
          <Typography variant="caption" color="error">
            {error}
          </Typography>
        ) : (
          symbols.map((item) => (
            <Chip
              key={item.symbol}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <span>{item.symbol}</span>
                  {getTrendIcon(item.change)}
                </Box>
              }
              onClick={() => onSymbolChange(item.symbol)}
              onDelete={selectedSymbol === item.symbol ? handleClearSelection : undefined}
              color={selectedSymbol === item.symbol ? 'primary' : 'default'}
              variant={selectedSymbol === item.symbol ? 'filled' : 'outlined'}
              size="small"
              sx={{
                borderColor: getSentimentColor(item.sentiment),
                '& .MuiChip-icon': {
                  color: getSentimentColor(item.sentiment)
                }
              }}
            />
          ))
        )}
      </Box>
    </Box>
  );
};

// Wrap with React.memo to prevent unnecessary re-renders
export default React.memo(SymbolFilterControl);
