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

function calculateDrawdown(equityCurve: { equity: number }[]) {
  let maxEquity = -Infinity;
  let maxDrawdown = 0;
  let drawdowns: number[] = [];
  equityCurve.forEach(point => {
    if (point.equity > maxEquity) maxEquity = point.equity;
    const dd = maxEquity > 0 ? (point.equity - maxEquity) / maxEquity : 0;
    drawdowns.push(dd);
    if (dd < maxDrawdown) maxDrawdown = dd;
  });
  return { maxDrawdown, drawdowns };
}

const PerformanceAnalysis: React.FC = () => {
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [metrics, setMetrics] = useState<any>(null);
  const [drawdowns, setDrawdowns] = useState<number[]>([]);
  const [maxDrawdown, setMaxDrawdown] = useState<number>(0);
  const [labels, setLabels] = useState<string[]>([]);
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
          setMetrics(res.metrics);
          const { maxDrawdown, drawdowns } = calculateDrawdown(res.equity_curve);
          setMaxDrawdown(maxDrawdown);
          setDrawdowns(drawdowns);
          setLabels(res.equity_curve.map(point => new Date(point.timestamp).toLocaleString()));
        })
        .catch(() => setError('Failed to load performance data'))
        .finally(() => setLoading(false));
    } else {
      setMetrics(null);
      setDrawdowns([]);
      setLabels([]);
    }
  }, [selectedId]);

  const drawdownChartData = {
    labels,
    datasets: [
      {
        label: 'Drawdown',
        data: drawdowns.map(dd => dd * 100),
        borderColor: 'rgb(239,68,68)',
        backgroundColor: 'rgba(239,68,68,0.2)',
        fill: true,
        tension: 0.1,
        pointRadius: 0
      }
    ]
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-3xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Performance Analysis</h2>
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
      {loading && <div className="text-gray-500 dark:text-gray-400">Loading performance data...</div>}
      {error && <div className="text-red-600 font-medium">{error}</div>}
      {metrics && (
        <div className="mt-4">
          <div className="mb-2">Sharpe Ratio: <b>{metrics.sharpe_ratio}</b></div>
          <div className="mb-2">Volatility: <b>{metrics.volatility}</b></div>
          <div className="mb-2">Alpha: <b>{metrics.alpha}</b></div>
          <div className="mb-2">Beta: <b>{metrics.beta}</b></div>
          <div className="mb-2">Max Drawdown: <b>{(maxDrawdown * 100).toFixed(2)}%</b></div>
        </div>
      )}
      {drawdowns.length > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-2">Drawdown Over Time</h3>
          <Line data={drawdownChartData} options={{
            responsive: true,
            plugins: {
              legend: { display: false },
              tooltip: { mode: 'index', intersect: false }
            },
            scales: {
              x: { display: true, title: { display: true, text: 'Time' } },
              y: { display: true, title: { display: true, text: 'Drawdown (%)' } }
            }
          }} />
        </div>
      )}
      {!loading && !error && !metrics && (
        <div className="text-gray-500 dark:text-gray-400 mt-4">No performance data available.</div>
      )}
    </div>
  );
};

export default PerformanceAnalysis;
