import React from 'react';
import { Table, TableBody, TableCell, TableContainer, TableRow, Box, Typography, CircularProgress } from '@mui/material';
import { format } from 'date-fns';

interface Trade {
  id: string;
  price: number;
  amount: number;
  time: number; // timestamp
  isBuyer: boolean; // true for buy, false for sell
}

interface MarketTradesProps {
  trades?: Trade[];
  loading?: boolean;
}

const MarketTrades: React.FC<MarketTradesProps> = ({ trades = [], loading = false }) => {
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  // Use provided trades or empty array
  const tradeData = trades.length > 0 ? trades : [];

  return (
    <TableContainer sx={{ height: '100%' }}>
      <Table size="small" stickyHeader>
        <TableBody>
          {tradeData.map((trade) => (
            <TableRow key={trade.id} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
              <TableCell sx={{ color: trade.isBuyer ? 'success.main' : 'error.main', py: 0.5 }}>
                {trade.price.toFixed(2)}
              </TableCell>
              <TableCell align="right" sx={{ py: 0.5 }}>{trade.amount.toFixed(4)}</TableCell>
              <TableCell align="right" sx={{ py: 0.5, color: 'text.secondary', fontSize: '0.75rem' }}>
                {format(new Date(trade.time), 'HH:mm:ss')}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

// Generate realistic mock data if real data is not available
function generateMockTradeData(symbol: string): Trade[] {
  const basePrice = symbol.startsWith('BTC') ? 60000 : 
                    symbol.startsWith('ETH') ? 3000 : 
                    symbol.startsWith('SOL') ? 150 : 100;
  
  const result: Trade[] = [];
  const count = 20;
  const now = Date.now();
  
  for (let i = 0; i < count; i++) {
    const priceDelta = (Math.random() * 20) - 10;
    const price = basePrice + priceDelta;
    const amount = Math.random() * 2;
    const isBuyer = Math.random() > 0.5;
    const time = now - (i * 10000); // Each trade 10 seconds apart
    
    result.push({
      id: `mock-trade-${i}`,
      price,
      amount,
      time,
      isBuyer
    });
  }
  
  return result;
}

export default MarketTrades;