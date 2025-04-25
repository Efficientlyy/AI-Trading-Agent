import React, { useEffect, useState } from 'react';
import { backtestApi } from '../../api/backtest';
import { BacktestResult } from '../../types';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend);

const EquityCurveChart: React.FC = () => {
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [curve, setCurve] = useState<{ timestamp: string; equity: number }[]>([]);
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
        .then(res => setCurve(res.equity_curve))
        .catch(() => setError('Failed to load equity curve'))
        .finally(() => setLoading(false));
    } else {
      setCurve([]);
    }
  }, [selectedId]);

  const chartData = {
    labels: curve.map(point => new Date(point.timestamp).toLocaleString()),
    datasets: [
      {
        label: 'Equity',
        data: curve.map(point => point.equity),
        fill: false,
        borderColor: 'rgb(34,197,94)',
        backgroundColor: 'rgba(34,197,94,0.2)',
        tension: 0.1,
        pointRadius: 0
      }
    ]
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-3xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Equity Curve Chart</h2>
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
      {loading && <div className="text-gray-500 dark:text-gray-400">Loading equity curve...</div>}
      {error && <div className="text-red-600 font-medium">{error}</div>}
      {!loading && !error && curve.length > 0 && (
        <div className="mt-8">
          <Line data={chartData} options={{
            responsive: true,
            plugins: {
              legend: { display: false },
              tooltip: { mode: 'index', intersect: false }
            },
            scales: {
              x: { display: true, title: { display: true, text: 'Time' } },
              y: { display: true, title: { display: true, text: 'Equity ($)' } }
            }
          }} />
        </div>
      )}
      {!loading && !error && curve.length === 0 && (
        <div className="text-gray-500 dark:text-gray-400 mt-4">No equity curve data available.</div>
      )}
    </div>
  );
};

export default EquityCurveChart;
