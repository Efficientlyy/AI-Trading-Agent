import React from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Box, Typography, CircularProgress, useTheme } from '@mui/material';

interface OrderBookProps {
  bids: number[][]; // [price, amount]
  asks: number[][]; // [price, amount]
  lastPrice?: number;
  loading?: boolean;
}

const OrderBook: React.FC<OrderBookProps> = ({ bids = [], asks = [], lastPrice = 0, loading = false }) => {
  const theme = useTheme();
  const textColor = theme.palette.mode === 'dark' ? '#f8fafc' : theme.palette.text.primary;
  const sellColor = theme.palette.mode === 'dark' ? '#f87171' : theme.palette.error.main;
  const buyColor = theme.palette.mode === 'dark' ? '#4ade80' : theme.palette.success.main;
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  // Use provided bids and asks, or empty arrays if none provided
  const bidData = bids.length > 0 ? bids : [];
  const askData = asks.length > 0 ? asks : [];

  // Sort asks in descending order (highest price first)
  const sortedAsks = [...askData].sort((a, b) => b[0] - a[0]);
  // Sort bids in descending order (highest price first)
  const sortedBids = [...bidData].sort((a, b) => b[0] - a[0]);

  // Take only top entries to fit in the UI
  const displayAsks = sortedAsks.slice(0, 10);
  const displayBids = sortedBids.slice(0, 10);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Asks (Sell orders) */}
      <TableContainer sx={{ flex: 1, maxHeight: '130px', bgcolor: 'background.paper' }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell sx={{ color: textColor }}>Price</TableCell>
              <TableCell align="right" sx={{ color: textColor }}>Amount</TableCell>
              <TableCell align="right" sx={{ color: textColor }}>Total</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {displayAsks.map((ask, index) => (
              <TableRow key={`ask-${index}`} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                <TableCell sx={{ color: sellColor }}>{ask[0].toFixed(2)}</TableCell>
                <TableCell align="right" sx={{ color: textColor }}>{ask[1].toFixed(4)}</TableCell>
                <TableCell align="right" sx={{ color: textColor }}>{(ask[0] * ask[1]).toFixed(2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Spread Display */}
      <Box sx={{ py: 1, display: 'flex', justifyContent: 'center', borderTop: '1px dashed', borderBottom: '1px dashed', borderColor: 'divider', bgcolor: 'background.paper' }}>
        {displayAsks.length > 0 && displayBids.length > 0 && (
          <Typography variant="body2" sx={{ color: textColor }}>
            Spread: {(displayAsks[displayAsks.length - 1][0] - displayBids[0][0]).toFixed(2)} ({((displayAsks[displayAsks.length - 1][0] - displayBids[0][0]) / displayBids[0][0] * 100).toFixed(2)}%)
          </Typography>
        )}
      </Box>

      {/* Bids (Buy orders) */}
      <TableContainer sx={{ flex: 1, maxHeight: '130px', bgcolor: 'background.paper' }}>
        <Table size="small">
          <TableBody>
            {displayBids.map((bid, index) => (
              <TableRow key={`bid-${index}`} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                <TableCell sx={{ color: buyColor }}>{bid[0].toFixed(2)}</TableCell>
                <TableCell align="right" sx={{ color: textColor }}>{bid[1].toFixed(4)}</TableCell>
                <TableCell align="right" sx={{ color: textColor }}>{(bid[0] * bid[1]).toFixed(2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

// Generate realistic mock data if real data is not available
function generateMockOrderbookData(isBids: boolean, symbol: string): [number, number][] {
  const basePrice = symbol.startsWith('BTC') ? 60000 : 
                    symbol.startsWith('ETH') ? 3000 : 
                    symbol.startsWith('SOL') ? 150 : 100;
  
  const result: [number, number][] = [];
  const count = 10;
  
  for (let i = 0; i < count; i++) {
    const priceDelta = (Math.random() * 10) * (i + 1) * (isBids ? -1 : 1);
    const price = basePrice + priceDelta;
    const amount = Math.random() * 2;
    result.push([price, amount]);
  }
  
  return result;
}

export default OrderBook;