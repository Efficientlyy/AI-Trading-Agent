import { Strategy } from '../../types';

export const mockStrategies: Strategy[] = [
  {
    id: '1',
    name: 'Moving Average Crossover',
    description: 'A strategy that uses two moving averages to generate buy and sell signals.',
    parameters: {
      fastPeriod: 10,
      slowPeriod: 50,
      signalPeriod: 9
    },
    asset_class: 'crypto',
    timeframe: '1h',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  },
  {
    id: '2',
    name: 'RSI Oscillator',
    description: 'A strategy that trades based on Relative Strength Index oversold and overbought conditions.',
    parameters: {
      period: 14,
      overbought: 70,
      oversold: 30
    },
    asset_class: 'crypto',
    timeframe: '1h',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  },
  {
    id: '3',
    name: 'MACD Divergence',
    description: 'A strategy that trades based on MACD histogram divergence with price.',
    parameters: {
      fastPeriod: 12,
      slowPeriod: 26,
      signalPeriod: 9
    },
    asset_class: 'crypto', 
    timeframe: '1h',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  },
  {
    id: '4',
    name: 'Bollinger Breakout',
    description: 'A strategy that trades breakouts from Bollinger Bands.',
    parameters: {
      period: 20,
      standardDeviations: 2
    },
    asset_class: 'crypto',
    timeframe: '1h',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  },
  {
    id: '5',
    name: 'Sentiment-Based',
    description: 'A strategy that uses market sentiment data to generate trading signals.',
    parameters: {
      sentimentThreshold: 0.6,
      useMarketSentiment: true,
      useSocialSentiment: true
    },
    asset_class: 'crypto',
    timeframe: '1h',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }
];