import React, { useState } from 'react';
import StrategyBuilder from '../components/strategies/StrategyBuilder';
import StrategyOptimizer from '../components/strategies/StrategyOptimizer';
import BacktestingInterface from '../components/strategies/BacktestingInterface';
import PerformanceMetricsDisplay from '../components/strategies/PerformanceMetrics';
import ComparativeAnalysis from '../components/strategies/ComparativeAnalysis';

const Strategies: React.FC = () => {
  const [tab, setTab] = useState<'builder' | 'optimizer' | 'backtest' | 'metrics' | 'compare'>('builder');

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 py-10">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center">Strategies</h1>
        <div className="flex justify-center gap-4 mb-8">
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'builder' ? 'bg-primary text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('builder')}
          >
            Strategy Builder
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'optimizer' ? 'bg-primary text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('optimizer')}
          >
            Strategy Optimizer
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'backtest' ? 'bg-primary text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('backtest')}
          >
            Backtesting Interface
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'metrics' ? 'bg-primary text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('metrics')}
          >
            Performance Metrics
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'compare' ? 'bg-primary text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('compare')}
          >
            Comparative Analysis
          </button>
        </div>
        {tab === 'builder' && <StrategyBuilder />}
        {tab === 'optimizer' && <StrategyOptimizer />}
        {tab === 'backtest' && <BacktestingInterface />}
        {tab === 'metrics' && <PerformanceMetricsDisplay />}
        {tab === 'compare' && <ComparativeAnalysis />}
      </div>
    </div>
  );
};

export default Strategies;
