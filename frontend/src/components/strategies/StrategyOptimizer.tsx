import React, { useState, useEffect } from 'react';
import { strategiesApi } from '../../api/strategies';
import { backtestApi } from '../../api/backtest';
import { Strategy, BacktestResult, PerformanceMetrics } from '../../types';

interface OptimizationRun {
  id: string;
  strategyId: string;
  startedAt: string;
  completedAt?: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  bestParams?: Record<string, any>;
  metrics?: PerformanceMetrics;
}

const StrategyOptimizer: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [runs, setRuns] = useState<OptimizationRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showParams, setShowParams] = useState(false);

  // Fetch strategies
  useEffect(() => {
    strategiesApi.getStrategies().then(res => setStrategies(res.strategies));
  }, []);

  // Simulate backend optimization call
  const handleOptimize = async () => {
    setLoading(true);
    setError(null);
    try {
      // In a real implementation, replace this with an optimizer API call
      setTimeout(() => {
        const fakeRun: OptimizationRun = {
          id: Math.random().toString(36).slice(2),
          strategyId: selectedId,
          startedAt: new Date().toISOString(),
          completedAt: new Date(Date.now() + 2000).toISOString(),
          status: 'completed',
          bestParams: { 'Lookback Period': 21, 'Threshold': 0.75 },
          metrics: {
            total_return: 0.23,
            sharpe_ratio: 1.44,
            max_drawdown: 0.08,
            annualized_return: 0.18,
            win_rate: 0.62,
            profit_factor: 1.9,
            avg_trade: 0.012,
            volatility: 0.15,
            beta: 0.9,
            alpha: 0.02
          }
        };
        setRuns(runs => [fakeRun, ...runs]);
        setLoading(false);
      }, 2000);
    } catch (e: any) {
      setError('Optimization failed.');
      setLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-6 max-w-xl mx-auto mt-8">
      <h2 className="text-2xl font-bold mb-4">Strategy Optimizer</h2>
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
      <button
        className="bg-primary text-white px-6 py-2 rounded font-semibold hover:bg-primary-dark disabled:opacity-50"
        onClick={handleOptimize}
        disabled={!selectedId || loading}
      >
        {loading ? 'Optimizing...' : 'Optimize Strategy'}
      </button>
      {error && (
        <div className="mt-4 text-red-600 font-medium">{error}</div>
      )}
      <div className="mt-8">
        <h3 className="text-lg font-semibold mb-2">Optimization Runs</h3>
        {runs.length === 0 && <div className="text-gray-500 dark:text-gray-400">No optimization runs yet.</div>}
        <ul className="space-y-4">
          {runs.map(run => (
            <li key={run.id} className="bg-gray-100 dark:bg-gray-800 rounded p-4">
              <div className="flex justify-between items-center">
                <div>
                  <span className="font-bold">{strategies.find(s => s.id === run.strategyId)?.name || 'Unknown'}</span>
                  <span className="ml-2 text-xs text-gray-500">{new Date(run.startedAt).toLocaleString()}</span>
                </div>
                <span className={`px-2 py-1 rounded text-xs font-semibold ${run.status === 'completed' ? 'bg-green-200 text-green-800' : run.status === 'failed' ? 'bg-red-200 text-red-800' : 'bg-yellow-200 text-yellow-800'}`}>{run.status}</span>
              </div>
              {run.completedAt && run.status === 'completed' && (
                <div className="mt-2">
                  <button className="text-primary underline" onClick={() => setShowParams(show => !show)}>
                    {showParams ? 'Hide' : 'Show'} Best Parameters & Metrics
                  </button>
                  {showParams && (
                    <div className="mt-2">
                      <div className="mb-1 font-medium">Best Parameters:</div>
                      <ul className="ml-4 list-disc">
                        {run.bestParams && Object.entries(run.bestParams).map(([k, v]) => (
                          <li key={k}><span className="font-mono">{k}</span>: <b>{String(v)}</b></li>
                        ))}
                      </ul>
                      <div className="mt-2 mb-1 font-medium">Performance Metrics:</div>
                      <ul className="ml-4 list-disc">
                        {run.metrics && Object.entries(run.metrics).map(([k, v]) => (
                          <li key={k}><span className="font-mono">{k}</span>: <b>{String(v)}</b></li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default StrategyOptimizer;
