import React, { useEffect, useState } from 'react';
import { backtestApi } from '../../api/backtest';
import { BacktestResult, PerformanceMetrics } from '../../types';

const PerformanceMetricsDisplay: React.FC = () => {
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
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
        .then(res => setMetrics(res.metrics))
        .catch(() => setError('Failed to load metrics'))
        .finally(() => setLoading(false));
    } else {
      setMetrics(null);
    }
  }, [selectedId]);

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Performance Metrics</h2>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Select Backtest Run</label>
        <select
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={selectedId}
          onChange={e => setSelectedId(e.target.value)}
        >
          <option value="">-- Choose a backtest --</option>
          {results.map(r => (
            <option key={r.id} value={r.id}>{r.params.strategy_name} ({new Date(r.created_at).toLocaleString()})</option>
          ))}
        </select>
      </div>
      {loading && <div className="text-gray-500 dark:text-gray-400">Loading metrics...</div>}
      {error && <div className="text-red-600 font-medium">{error}</div>}
      {metrics && (
        <div className="mt-4">
          <div className="mb-2">Total Return: <b>{(metrics.total_return * 100).toFixed(2)}%</b></div>
          <div className="mb-2">Sharpe Ratio: <b>{metrics.sharpe_ratio}</b></div>
          <div className="mb-2">Max Drawdown: <b>{(metrics.max_drawdown * 100).toFixed(2)}%</b></div>
          {metrics.win_rate !== undefined && <div className="mb-2">Win Rate: <b>{(metrics.win_rate * 100).toFixed(2)}%</b></div>}
          {metrics.profit_factor !== undefined && <div className="mb-2">Profit Factor: <b>{metrics.profit_factor}</b></div>}
          {metrics.volatility !== undefined && <div className="mb-2">Volatility: <b>{metrics.volatility}</b></div>}
          {metrics.beta !== undefined && <div className="mb-2">Beta: <b>{metrics.beta}</b></div>}
          {metrics.alpha !== undefined && <div className="mb-2">Alpha: <b>{metrics.alpha}</b></div>}
        </div>
      )}
    </div>
  );
};

export default PerformanceMetricsDisplay;
