import React, { useEffect, useState } from 'react';
import { backtestApi } from '../../api/backtest';
import { BacktestResult, Order } from '../../types';

function calculateTradeStats(trades: Order[]) {
  if (!trades || trades.length === 0) return null;
  let wins = 0, losses = 0, totalPnL = 0, grossProfit = 0, grossLoss = 0, profitFactor = 0;
  let totalTrades = trades.length;
  trades.forEach(trade => {
    // Use nullish coalescing to handle undefined realized_pnl
    const pnl = trade.realized_pnl ?? 0;
    totalPnL += pnl;
    if (pnl > 0) {
      wins++;
      grossProfit += pnl;
    } else if (pnl < 0) {
      losses++;
      grossLoss += Math.abs(pnl);
    }
  });
  profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;
  return {
    totalTrades,
    wins,
    losses,
    winRate: totalTrades > 0 ? wins / totalTrades : 0,
    profitFactor,
    grossProfit,
    grossLoss,
    netPnL: totalPnL
  };
}

const TradeStatistics: React.FC = () => {
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [trades, setTrades] = useState<Order[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    backtestApi.getAllBacktests().then(res => setResults(res.backtests));
  }, []);

  useEffect(() => {
    if (selectedId) {
      setLoading(true);
      setError(null);
      backtestApi.getBacktestResults(selectedId)
        .then(res => {
          setTrades(res.trades);
          setStats(calculateTradeStats(res.trades));
        })
        .catch(() => setError('Failed to load trades'))
        .finally(() => setLoading(false));
    } else {
      setTrades([]);
      setStats(null);
    }
  }, [selectedId]);

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-3xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Trade Statistics</h2>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Select Backtest Run</label>
        <select
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={selectedId}
          onChange={e => setSelectedId(e.target.value)}
        >
          <option value="">-- Choose a backtest --</option>
          {results.map(r => (
            <option key={r.id} value={r.id}>
              {r.params?.strategy_name || 'Unnamed Strategy'} 
              ({r.created_at ? new Date(r.created_at).toLocaleString() : 'Unknown Date'})
            </option>
          ))}
        </select>
      </div>
      {loading && <div className="text-gray-500 dark:text-gray-400">Loading trade statistics...</div>}
      {error && <div className="text-red-600 font-medium">{error}</div>}
      {stats && (
        <div className="mt-4">
          <div className="mb-2">Total Trades: <b>{stats.totalTrades}</b></div>
          <div className="mb-2">Winning Trades: <b>{stats.wins}</b></div>
          <div className="mb-2">Losing Trades: <b>{stats.losses}</b></div>
          <div className="mb-2">Win Rate: <b>{(stats.winRate * 100).toFixed(2)}%</b></div>
          <div className="mb-2">Profit Factor: <b>{stats.profitFactor.toFixed(4)}</b></div>
          <div className="mb-2">Gross Profit: <b>${stats.grossProfit.toFixed(2)}</b></div>
          <div className="mb-2">Gross Loss: <b>${stats.grossLoss.toFixed(2)}</b></div>
          <div className="mb-2">Net P&L: <b>${stats.netPnL.toFixed(2)}</b></div>
        </div>
      )}
      {!loading && !error && !stats && (
        <div className="text-gray-500 dark:text-gray-400 mt-4">No trade statistics available.</div>
      )}
    </div>
  );
};

export default TradeStatistics;
