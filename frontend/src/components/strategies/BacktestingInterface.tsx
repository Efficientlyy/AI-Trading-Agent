import React, { useState, useEffect } from 'react';
import { strategiesApi } from '../../api/strategies';
import { backtestApi } from '../../api/backtest';
import { Strategy, BacktestResult } from '../../types';

const BacktestingInterface: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState<number>(10000);
  const [symbols, setSymbols] = useState<string>('BTC/USD');
  const [running, setRunning] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');
  const [progress, setProgress] = useState<number>(0);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    strategiesApi.getStrategies().then(res => setStrategies(res));
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (jobId && running) {
      interval = setInterval(async () => {
        const res = await backtestApi.getBacktestStatus(jobId);
        setStatus(res.status);
        setProgress(res.progress);
        if (res.status === 'completed') {
          clearInterval(interval);
          const data = await backtestApi.getBacktestResults(jobId);
          setResult(data);
          setRunning(false);
        } else if (res.status === 'failed') {
          clearInterval(interval);
          setError(res.message);
          setRunning(false);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, running]);

  const handleStart = async () => {
    setError(null);
    setResult(null);
    setStatus('');
    setProgress(0);
    setRunning(true);
    try {
      const strategy = strategies.find(s => s.id === selectedId);
      if (!strategy) throw new Error('Select a strategy');
      const params = {
        strategy_name: strategy.name,
        parameters: strategy.parameters,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        symbols: symbols.split(',').map(s => s.trim()).filter(Boolean),
      };
      const res = await backtestApi.startBacktest(params);
      setJobId(res.job_id);
    } catch (e: any) {
      setError(e.message || 'Failed to start backtest');
      setRunning(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Backtesting Interface</h2>
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Select Strategy</label>
        <select
          className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white"
          value={selectedId}
          onChange={e => setSelectedId(e.target.value)}
        >
          <option value="">-- Choose a strategy --</option>
          {strategies.map(s => (
            <option key={s.id} value={s.id}>{s.name}</option>
          ))}
        </select>
      </div>
      <div className="mb-4 flex gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium mb-1">Start Date</label>
          <input type="date" className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white" value={startDate} onChange={e => setStartDate(e.target.value)} />
        </div>
        <div className="flex-1">
          <label className="block text-sm font-medium mb-1">End Date</label>
          <input type="date" className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white" value={endDate} onChange={e => setEndDate(e.target.value)} />
        </div>
      </div>
      <div className="mb-4 flex gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium mb-1">Initial Capital</label>
          <input type="number" min="0" className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white" value={initialCapital} onChange={e => setInitialCapital(Number(e.target.value))} />
        </div>
        <div className="flex-1">
          <label className="block text-sm font-medium mb-1">Symbols (comma separated)</label>
          <input type="text" className="w-full px-3 py-2 rounded border border-gray-300 dark:bg-gray-800 dark:text-white" value={symbols} onChange={e => setSymbols(e.target.value)} />
        </div>
      </div>
      <button
        className="bg-primary text-white px-6 py-2 rounded font-semibold hover:bg-primary-dark disabled:opacity-50"
        onClick={handleStart}
        disabled={!selectedId || !startDate || !endDate || running}
      >
        {running ? 'Running...' : 'Start Backtest'}
      </button>
      {error && <div className="mt-4 text-red-600 font-medium">{error}</div>}
      {status && (
        <div className="mt-4">
          <div className="text-sm text-gray-700 dark:text-gray-300">Backtest Status: <b>{status}</b></div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded h-2 mt-2">
            <div className="bg-primary h-2 rounded" style={{ width: `${progress}%` }}></div>
          </div>
        </div>
      )}
      {result && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold mb-2">Backtest Result</h3>
          <div className="mb-2">Total Return: <b>{result.metrics?.total_return !== undefined ? result.metrics.total_return.toFixed(4) : '-'}%</b></div>
          <div className="mb-2">Sharpe Ratio: <b>{result.metrics?.sharpe_ratio !== undefined ? result.metrics.sharpe_ratio : '-'}</b></div>
          <div className="mb-2">Max Drawdown: <b>{result.metrics?.max_drawdown !== undefined ? (result.metrics.max_drawdown * 100).toFixed(2) : '-'}%</b></div>
          <div className="mb-2">Win Rate: <b>{result.metrics?.win_rate !== undefined ? ((result.metrics.win_rate ?? 0) * 100).toFixed(2) : '-'}%</b></div>
          <div className="mb-2">Profit Factor: <b>{result.metrics?.profit_factor !== undefined ? result.metrics.profit_factor : '-'}</b></div>
          <div className="mb-2">Avg Trade: <b>{result.metrics?.avg_trade !== undefined ? result.metrics.avg_trade : '-'}</b></div>
          {/* Add more metrics or charts as needed */}
        </div>
      )}
    </div>
  );
};

export default BacktestingInterface;
