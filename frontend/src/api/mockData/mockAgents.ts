import { Agent } from '../../context/SystemControlContext';

export const mockAgents: Agent[] = [
  // LLM Oversight Agent - Will show our enhanced UI features
  {
    agent_id: 'llm_oversight_agent',
    name: 'LLM Oversight Agent',
    status: 'running',
    type: 'LLM Oversight',
    last_updated: new Date().toISOString(),
    metrics: {
      win_rate: 0.87,
      profit_factor: 2.43,
      avg_profit_loss: 0.12,
      max_drawdown: -0.09
    },
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'BTC-USD'],
    strategy: 'LLM Decision Validation',
    agent_role: 'oversight',
    inputs_from: ['market_data_agent', 'sentiment_agent'],
    outputs_to: ['decision_aggregator'],
    config_details: {
      model: 'gpt-4',
      provider: 'OpenRouter',
      confidence_threshold: 0.75
    }
  },
  // Standard Trading Agent
  {
    agent_id: 'macd_trading_agent',
    name: 'MACD Trading Strategy',
    status: 'running',
    type: 'Technical',
    last_updated: new Date().toISOString(),
    metrics: {
      win_rate: 0.63,
      profit_factor: 1.82,
      avg_profit_loss: 0.08,
      max_drawdown: -0.15
    },
    symbols: ['AAPL', 'MSFT'],
    strategy: 'MACD Crossover',
    agent_role: 'signal_generator',
    outputs_to: ['decision_aggregator']
  },
  // Sentiment Analysis Agent
  {
    agent_id: 'sentiment_agent',
    name: 'Sentiment Analyzer (AlphaV)',
    status: 'running',
    type: 'Sentiment',
    last_updated: new Date().toISOString(),
    metrics: {
      win_rate: 0.59,
      profit_factor: 1.4,
      avg_profit_loss: 0.05,
      max_drawdown: -0.12
    },
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    strategy: 'News Sentiment Analysis',
    agent_role: 'data_provider',
    outputs_to: ['llm_oversight_agent', 'decision_aggregator']
  },
  // Decision Aggregator
  {
    agent_id: 'decision_aggregator',
    name: 'Main Decision Aggregator',
    status: 'running', 
    type: 'Aggregator',
    last_updated: new Date().toISOString(),
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    agent_role: 'decision_maker',
    inputs_from: ['macd_trading_agent', 'sentiment_agent', 'llm_oversight_agent'],
    outputs_to: ['execution_handler']
  },
  // Execution Handler
  {
    agent_id: 'execution_handler',
    name: 'Execution Handler (Alpaca)',
    status: 'stopped',
    type: 'Execution',
    last_updated: new Date().toISOString(),
    symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    agent_role: 'executor',
    inputs_from: ['decision_aggregator']
  }
];

export const getMockAgents = (): Promise<Agent[]> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve(mockAgents), 500); // Simulate API delay
  });
};
