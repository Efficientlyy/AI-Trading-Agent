import React, { useEffect, useState } from 'react';
import { backtestApi } from '../../api/backtest';
import { BacktestResult, PerformanceMetrics } from '../../types';
import { Bar, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  RadialLinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  RadialLinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Define the metric keys as a type-safe array
const METRIC_KEYS = [
  'total_return',
  'sharpe_ratio',
  'sortino_ratio',
  'max_drawdown',
  'win_rate',
  'profit_factor'
] as const;

type MetricKey = typeof METRIC_KEYS[number];

// Type-safe labels
const METRIC_LABELS: Record<MetricKey, string> = {
  total_return: 'Total Return',
  sharpe_ratio: 'Sharpe Ratio',
  sortino_ratio: 'Sortino Ratio',
  max_drawdown: 'Max Drawdown',
  win_rate: 'Win Rate',
  profit_factor: 'Profit Factor'
};

const ComparativeAnalysis: React.FC = () => {
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [metrics, setMetrics] = useState<Record<string, PerformanceMetrics>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
            // Initialize with empty object if metrics is undefined
            m[selectedIds[i]] = r.metrics || {} as PerformanceMetrics;
          });
          setMetrics(m);
        })
        .catch(() => setError('Failed to load comparative data'))
        .finally(() => setLoading(false));
    } else {
      setMetrics({});
    }
  }, [selectedIds]);

  const handleSelectionChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const options = Array.from(e.target.selectedOptions);
    setSelectedIds(options.map(option => option.value));
  };

  const radarData = {
    labels: METRIC_KEYS.map(k => METRIC_LABELS[k]),
    datasets: selectedIds.map((id, idx) => {
      const m = metrics[id] || {} as PerformanceMetrics;
      return {
        label: results.find(r => r.id === id)?.params?.strategy_name || `Run ${idx + 1}`,
        data: METRIC_KEYS.map(k => {
          // Type assertion to access properties by string key
          const value = m[k as keyof PerformanceMetrics];
          return typeof value === 'number' ? value : 0;
        }),
        fill: true,
        backgroundColor: `rgba(${60 + idx * 60}, 120, 220, 0.2)`,
        borderColor: `rgba(${60 + idx * 60}, 120, 220, 1)`,
        pointBackgroundColor: `rgba(${60 + idx * 60}, 120, 220, 1)`,
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: `rgba(${60 + idx * 60}, 120, 220, 1)`,
      };
    })
  };

  const barData = {
    labels: METRIC_KEYS.map(k => METRIC_LABELS[k]),
    datasets: selectedIds.map((id, idx) => {
      const m = metrics[id] || {} as PerformanceMetrics;
      return {
        label: results.find(r => r.id === id)?.params?.strategy_name || `Run ${idx + 1}`,
        data: METRIC_KEYS.map(k => {
          // Type assertion to access properties by string key
          const value = m[k as keyof PerformanceMetrics];
          return typeof value === 'number' ? value : 0;
        }),
        backgroundColor: `rgba(${60 + idx * 60}, 120, 220, 0.5)`
      };
    })
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-3xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Comparative Analysis</h2>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Select Backtest Runs to Compare</label>
        <select
          multiple
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={selectedIds}
          onChange={handleSelectionChange}
        >
          {results.map(r => (
            <option key={r.id} value={r.id}>
              {r.params?.strategy_name || 'Unnamed Strategy'} 
              ({r.created_at ? new Date(r.created_at).toLocaleString() : 'Unknown Date'})
            </option>
          ))}
        </select>
        <div className="text-xs text-gray-500 mt-1">Hold Ctrl/Cmd to select multiple.</div>
      </div>
      {loading && <div className="text-gray-500 dark:text-gray-400">Loading comparative data...</div>}
      {error && <div className="text-red-600 font-medium">{error}</div>}
      {selectedIds.length === 0 && (
        <div className="text-gray-500 dark:text-gray-400 mt-4">Select at least one backtest run to compare.</div>
      )}
      {!loading && !error && selectedIds.length > 0 && (
        <div className="mt-4 space-y-8">
          <div>
            <h3 className="text-lg font-semibold mb-4">Comparative Metrics</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full border-collapse border border-gray-300 dark:border-gray-700">
                <thead>
                  <tr className="bg-gray-100 dark:bg-gray-800">
                    <th className="px-4 py-2 border-b">Metric</th>
                    {selectedIds.map(id => (
                      <th key={id} className="px-4 py-2 border-b">{results.find(r => r.id === id)?.params?.strategy_name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {METRIC_KEYS.map(k => (
                    <tr key={k} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                      <td className="px-4 py-2 border-b font-medium">{METRIC_LABELS[k]}</td>
                      {selectedIds.map(id => {
                        // Type assertion to access properties by string key
                        const value = metrics[id]?.[k as keyof PerformanceMetrics];
                        return (
                          <td key={id} className="px-4 py-2 border-b text-center">
                            {typeof value === 'number' ? value.toFixed(4) : '-'}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Radar Chart Comparison</h3>
            <Radar data={radarData} options={{
              responsive: true,
              plugins: {
                legend: { position: 'top' as const },
                title: { display: false },
              },
            }} />
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Bar Chart Comparison</h3>
            <Bar data={barData} options={{
              responsive: true,
              plugins: {
                legend: { position: 'top' as const },
                title: { display: false },
              },
            }} />
          </div>
        </div>
      )}
    </div>
  );
};

export default ComparativeAnalysis;
