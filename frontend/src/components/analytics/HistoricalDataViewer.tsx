import React, { useState } from 'react';
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
import { OHLCV } from '../../types';
import { marketApi } from '../../api/market';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend);

const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'];

const HistoricalDataViewer: React.FC = () => {
  const [symbol, setSymbol] = useState('BTC/USD');
  const [timeframe, setTimeframe] = useState('1d');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [data, setData] = useState<OHLCV[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Our enhanced marketApi.getHistoricalData now handles string timeframes directly
      const res = await marketApi.getHistoricalData({ symbol, start, end, timeframe });
      setData(res.data);
    } catch (e: any) {
      setError('Failed to fetch historical data');
      console.error('Error fetching historical data:', e);
    } finally {
      setLoading(false);
    }
  };

  const chartData = {
    labels: data.map(d => new Date(d.timestamp).toLocaleDateString()),
    datasets: [
      {
        label: 'Close Price',
        data: data.map(d => d.close),
        borderColor: 'rgb(59,130,246)',
        backgroundColor: 'rgba(59,130,246,0.2)',
        tension: 0.1,
        pointRadius: 0
      }
    ]
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-3xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Historical Data Viewer</h2>
      <div className="mb-4 grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Symbol</label>
          <input
            className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
            value={symbol}
            onChange={e => setSymbol(e.target.value)}
            placeholder="e.g. BTC/USD"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Timeframe</label>
          <select
            className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
            value={timeframe}
            onChange={e => setTimeframe(e.target.value)}
          >
            {timeframes.map(tf => <option key={tf} value={tf}>{tf}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Start Date</label>
          <input
            type="date"
            className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
            value={start}
            onChange={e => setStart(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">End Date</label>
          <input
            type="date"
            className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
            value={end}
            onChange={e => setEnd(e.target.value)}
          />
        </div>
      </div>
      <button
        className="bg-primary text-white px-6 py-2 rounded font-semibold hover:bg-primary-dark disabled:opacity-50"
        onClick={fetchData}
        disabled={loading || !symbol || !start || !end}
      >
        {loading ? 'Loading...' : 'Fetch Data'}
      </button>
      {error && <div className="mt-4 text-red-600 font-medium">{error}</div>}
      {!loading && data.length > 0 && (
        <div className="mt-8">
          <Line data={chartData} options={{
            responsive: true,
            plugins: {
              legend: { display: false },
              tooltip: { mode: 'index', intersect: false }
            },
            scales: {
              x: { display: true, title: { display: true, text: 'Date' } },
              y: { display: true, title: { display: true, text: 'Close Price' } }
            }
          }} />
        </div>
      )}
      {!loading && data.length === 0 && (
        <div className="text-gray-500 dark:text-gray-400 mt-4">No data available.</div>
      )}
    </div>
  );
};

export default HistoricalDataViewer;
