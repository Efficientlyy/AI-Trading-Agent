/**
 * Frontend configuration
 */

// API base URL - hardcoded for local development
export const API_BASE_URL = `http://localhost:8000/api`;

// For debugging
console.log('Using API base URL:', API_BASE_URL);

// Uncomment this for production
// export const API_BASE_URL = `${window.location.protocol}//${window.location.host}/api`;

// WebSocket URL - hardcoded for local development
export const WS_BASE_URL = `ws://localhost:8000/ws`;

// Uncomment this for production
// export const WS_BASE_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`;

// Trading mode enum
export enum TradingMode {
  LIVE = 'live',
  PAPER = 'paper',
  BACKTEST = 'backtest',
  MOCK = 'mock'
}

// Exchange names enum
export enum ExchangeName {
  BINANCE = 'binance',
  COINBASE = 'coinbase',
  ALPACA = 'alpaca',
  MOCK = 'mock'
}

// Exchange configuration interface
export interface ExchangeConfig {
  apiKey: string;
  apiSecret: string;
  baseUrl: string;
  sandboxMode: boolean;
}

// Get current trading mode
export const getTradingMode = (): TradingMode => {
  // Default to paper trading for safety
  return TradingMode.PAPER;
};

// Get exchange configuration
export const getExchangeConfig = (exchange: ExchangeName): ExchangeConfig => {
  // Default configuration
  return {
    apiKey: '',
    apiSecret: '',
    baseUrl: '',
    sandboxMode: true
  };
};

// Default paper trading configuration
export const DEFAULT_PAPER_TRADING_CONFIG = {
  duration_minutes: 60, // 1 hour
  interval_minutes: 1,  // 1 minute intervals
  initial_capital: 10000,
  symbols: ['BTC/USD', 'ETH/USD', 'XRP/USD'],
  risk_level: 'medium'
};

// Chart update interval in milliseconds
export const CHART_UPDATE_INTERVAL = 1000; // 1 second

// Maximum number of trades to display
export const MAX_TRADES_DISPLAY = 50;

// Maximum number of portfolio history points to keep
export const MAX_PORTFOLIO_HISTORY = 100;

// Default alert settings
export const DEFAULT_ALERT_SETTINGS = {
  drawdown_threshold: 5, // 5% drawdown
  gain_threshold: 5,     // 5% gain
  large_trade_threshold: 10, // 10% of portfolio
  consecutive_losses: 3  // Alert after 3 consecutive losses
};

// Default export for modules that import the entire config
export default {
  API_BASE_URL,
  WS_BASE_URL,
  getTradingMode,
  getExchangeConfig,
  DEFAULT_PAPER_TRADING_CONFIG,
  CHART_UPDATE_INTERVAL,
  MAX_TRADES_DISPLAY,
  MAX_PORTFOLIO_HISTORY,
  DEFAULT_ALERT_SETTINGS
};
