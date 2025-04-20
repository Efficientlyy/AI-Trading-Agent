import React, { useEffect, useState } from 'react';
import { backtestApi } from '../../api/backtest';
import { BacktestResult, PerformanceMetrics } from '../../types';
import {
  Radar,
  Bar
} from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  BarElement,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(RadialLinearScale, BarElement, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

const METRIC_KEYS = [
  'total_return',
  'sharpe_ratio',
  'max_drawdown',
  'win_rate',
  'profit_factor',
  'volatility',
  'beta',
  'alpha'
];

const METRIC_LABELS: Record<string, string> = {
  total_return: 'Total Return',
  sharpe_ratio: 'Sharpe Ratio',
  max_drawdown: 'Max Drawdown',
  win_rate: 'Win Rate',
  profit_factor: 'Profit Factor',
  volatility: 'Volatility',
  beta: 'Beta',
  alpha: 'Alpha'
};

const ComparativeAnalysis: React.FC = () => {
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Record<string, PerformanceMetrics>>({});

  useEffect(() => {
    backtestApi.getAllBacktests().then(res => setResults(res.backtests));
  }, []);

  useEffect(() => {
    if (selectedIds.length > 0) {
      setLoading(true);
      setError(null);
      Promise.all(selectedIds.map(id => backtestApi.getBacktestResults(id)))
        .then(resArr => {
          const m: Record<string, PerformanceMetrics> = {};
          resArr.forEach((r, i) => {
            m[selectedIds[i]] = r.metrics;
          });
          setMetrics(m);
        })
        .catch(() => setError('Failed to load metrics'))
        .finally(() => setLoading(false));
    } else {
      setMetrics({});
    }
  }, [selectedIds]);

  // Prepare data for radar and bar charts
  const radarData = {
    labels: METRIC_KEYS.map(k => METRIC_LABELS[k]),
    datasets: selectedIds.map((id, idx) => {
      const m = metrics[id] || {};
      return {
        label: results.find(r => r.id === id)?.params.strategy_name || `Run ${idx + 1}`,
        data: METRIC_KEYS.map(k => m[k] ?? 0),
        fill: true,
        backgroundColor: `rgba(${60 + idx * 60}, 120, 220, 0.2)`,
        borderColor: `rgba(${60 + idx * 60}, 120, 220, 1)`,
        pointBackgroundColor: `rgba(${60 + idx * 60}, 120, 220, 1)`
      };
    })
  };

  const barData = {
    labels: METRIC_KEYS.map(k => METRIC_LABELS[k]),
    datasets: selectedIds.map((id, idx) => {
      const m = metrics[id] || {};
      return {
        label: results.find(r => r.id === id)?.params.strategy_name || `Run ${idx + 1}`,
        data: METRIC_KEYS.map(k => m[k] ?? 0),
        backgroundColor: `rgba(${60 + idx * 60}, 120, 220, 0.5)`
      };
    })
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-4xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Comparative Analysis</h2>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Select Backtests to Compare</label>
        <select
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          multiple
          size={Math.min(8, results.length)}
          value={selectedIds}
          onChange={e => {
            const options = Array.from(e.target.selectedOptions).map(opt => opt.value);
            setSelectedIds(options);
          }}
        >
          {results.map(r => (
            <option key={r.id} value={r.id}>{r.params.strategy_name} ({new Date(r.created_at).toLocaleString()})</option>
          ))}
        </select>
        <div className="text-xs text-gray-500 mt-1">Hold Ctrl/Cmd to select multiple.</div>
      </div>
      {loading && <div className="text-gray-500 dark:text-gray-400">Loading metrics...</div>}
      {error && <div className="text-red-600 font-medium">{error}</div>}
      {selectedIds.length > 0 && Object.keys(metrics).length === selectedIds.length && (
        <>
          <div className="my-8">
            <h3 className="text-lg font-semibold mb-2">Radar Chart</h3>
            <Radar data={radarData} />
          </div>
          <div className="my-8">
            <h3 className="text-lg font-semibold mb-2">Bar Chart</h3>
            <Bar data={barData} />
          </div>
          <div className="overflow-x-auto mt-8">
            <table className="min-w-full border border-gray-300 dark:border-gray-700">
              <thead>
                <tr>
                  <th className="px-4 py-2 border-b">Metric</th>
                  {selectedIds.map(id => (
                    <th key={id} className="px-4 py-2 border-b">{results.find(r => r.id === id)?.params.strategy_name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRIC_KEYS.map(k => (
                  <tr key={k}>
                    <td className="px-4 py-2 border-b font-medium">{METRIC_LABELS[k]}</td>
                    {selectedIds.map(id => (
                      <td key={id} className="px-4 py-2 border-b text-center">{metrics[id] && metrics[id][k] !== undefined ? metrics[id][k].toFixed(4) : '-'}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {selectedIds.length === 0 && <div className="text-gray-500 dark:text-gray-400 mt-4">Select one or more backtests to compare.</div>}
    </div>
  );
};

export default ComparativeAnalysis;
