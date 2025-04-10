import React, { useState } from 'react';

export interface TradingStrategyProps {
  symbol: string;
  onStrategyChange?: (strategy: StrategyConfig) => void;
}

export interface StrategyConfig {
  id: string;
  name: string;
  type: 'technical' | 'sentiment' | 'combined';
  parameters: Record<string, any>;
  description: string;
}

const AVAILABLE_STRATEGIES: StrategyConfig[] = [
  {
    id: 'macd_crossover',
    name: 'MACD Crossover',
    type: 'technical',
    parameters: {
      fastPeriod: 12,
      slowPeriod: 26,
      signalPeriod: 9,
      positionSize: 10,
    },
    description: 'Generates buy/sell signals based on MACD line crossing the signal line.',
  },
  {
    id: 'bollinger_breakout',
    name: 'Bollinger Breakout',
    type: 'technical',
    parameters: {
      period: 20,
      stdDev: 2,
      positionSize: 10,
    },
    description: 'Generates signals when price breaks out of Bollinger Bands.',
  },
  {
    id: 'sentiment_momentum',
    name: 'Sentiment Momentum',
    type: 'sentiment',
    parameters: {
      lookbackPeriod: 3,
      threshold: 0.6,
      positionSize: 10,
    },
    description: 'Uses sentiment analysis to identify momentum in market sentiment.',
  },
  {
    id: 'tech_sentiment_combined',
    name: 'Technical + Sentiment',
    type: 'combined',
    parameters: {
      technicalWeight: 0.6,
      sentimentWeight: 0.4,
      rsiPeriod: 14,
      sentimentThreshold: 0.5,
      positionSize: 10,
    },
    description: 'Combines technical indicators with sentiment analysis for signal generation.',
  },
];

const TradingStrategy: React.FC<TradingStrategyProps> = ({ symbol, onStrategyChange }) => {
  const [selectedStrategy, setSelectedStrategy] = useState<string>(AVAILABLE_STRATEGIES[0].id);
  const [parameters, setParameters] = useState<Record<string, any>>(AVAILABLE_STRATEGIES[0].parameters);
  
  // Get the currently selected strategy configuration
  const currentStrategy = AVAILABLE_STRATEGIES.find(s => s.id === selectedStrategy) || AVAILABLE_STRATEGIES[0];
  
  // Handle strategy selection change
  const handleStrategyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const strategyId = e.target.value;
    const strategy = AVAILABLE_STRATEGIES.find(s => s.id === strategyId);
    
    if (strategy) {
      setSelectedStrategy(strategyId);
      setParameters(strategy.parameters);
      
      if (onStrategyChange) {
        onStrategyChange({
          ...strategy,
          parameters: { ...strategy.parameters },
        });
      }
    }
  };
  
  // Handle parameter change
  const handleParameterChange = (paramName: string, value: any) => {
    const newParameters = {
      ...parameters,
      [paramName]: typeof parameters[paramName] === 'number' ? parseFloat(value) : value,
    };
    
    setParameters(newParameters);
    
    if (onStrategyChange) {
      onStrategyChange({
        ...currentStrategy,
        parameters: newParameters,
      });
    }
  };
  
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Trading Strategy</h2>
      
      <div className="mb-4">
        <label htmlFor="strategy-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Strategy for {symbol}
        </label>
        <select
          id="strategy-select"
          value={selectedStrategy}
          onChange={handleStrategyChange}
          className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
        >
          {AVAILABLE_STRATEGIES.map(strategy => (
            <option key={strategy.id} value={strategy.id}>
              {strategy.name} ({strategy.type})
            </option>
          ))}
        </select>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          {currentStrategy.description}
        </p>
      </div>
      
      <div className="mb-4">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Parameters</h3>
        <div className="space-y-3">
          {Object.entries(parameters).map(([paramName, paramValue]) => (
            <div key={paramName} className="grid grid-cols-2 gap-2 items-center">
              <label htmlFor={`param-${paramName}`} className="text-sm text-gray-600 dark:text-gray-400">
                {paramName.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
              </label>
              <input
                id={`param-${paramName}`}
                type={typeof paramValue === 'number' ? 'number' : 'text'}
                value={paramValue}
                onChange={(e) => handleParameterChange(paramName, e.target.value)}
                className="px-2 py-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              />
            </div>
          ))}
        </div>
      </div>
      
      <div className="mt-4 flex justify-between">
        <button
          type="button"
          className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          Backtest Strategy
        </button>
        <button
          type="button"
          className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
        >
          Apply Strategy
        </button>
      </div>
    </div>
  );
};

export default TradingStrategy;
