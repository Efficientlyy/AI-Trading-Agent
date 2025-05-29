import React from 'react';
import { Box, Typography, Grid, Chip, Skeleton, Alert } from '@mui/material';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import { useMexcTickers, TickerItem } from '../../hooks';

// This constant should match the one in useMexcData.ts
const USE_MOCK_DATA = true;

const PriceTickerBar: React.FC = () => {
  // Use our custom hook to fetch real ticker data
  const { tickers, loading, error } = useMexcTickers();

  const formatVolume = (volume: number): string => {
    if (volume >= 1000000000) {
      return `$${(volume / 1000000000).toFixed(1)}B`;
    } else if (volume >= 1000000) {
      return `$${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `$${(volume / 1000).toFixed(1)}K`;
    }
    return `$${volume.toFixed(0)}`;
  };

  return (
    <Box sx={{ overflowX: 'auto', mb: 2, py: 1, borderBottom: 1, borderColor: 'divider' }}>
      {error && !USE_MOCK_DATA && (
        <Alert severity="error" sx={{ mb: 1 }}>
          Error loading ticker data: {error.message}
        </Alert>
      )}
      <Grid container spacing={2} wrap="nowrap" sx={{ minWidth: 800 }}>
        {loading ? (
          // Show skeletons while loading
          Array.from(new Array(8)).map((_, index) => (
            <Grid item key={index}>
              <Skeleton variant="rectangular" width={120} height={40} />
            </Grid>
          ))
        ) : (
          // Show actual ticker data
          tickers.map((ticker: TickerItem) => (
            <Grid item key={ticker.symbol}>
              <Box sx={{ px: 1, py: 0.5, borderRadius: 1, '&:hover': { bgcolor: 'action.hover' } }}>
                <Typography variant="body2" fontWeight="bold">{ticker.symbol}</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                  <Typography variant="body2" sx={{ mr: 1 }}>
                    ${ticker.price.toFixed(ticker.price < 1 ? 4 : 2)}
                  </Typography>
                  <Chip
                    size="small"
                    label={`${ticker.change > 0 ? '+' : ''}${ticker.change.toFixed(1)}%`}
                    color={ticker.change >= 0 ? 'success' : 'error'}
                    icon={ticker.change >= 0 ? <ArrowDropUpIcon /> : <ArrowDropDownIcon />}
                    sx={{ height: 20, '& .MuiChip-label': { px: 0.5 } }}
                  />
                </Box>
                <Typography variant="caption" color="text.secondary">
                  Vol: {formatVolume(ticker.volume)}
                </Typography>
              </Box>
            </Grid>
          ))
        )}
      </Grid>
    </Box>
  );
};

export default PriceTickerBar;