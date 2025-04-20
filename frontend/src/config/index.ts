/**
 * Configuration service for managing environment variables and application settings
 */

// Trading modes
export type TradingMode = 'mock' | 'paper' | 'live';
export type ExchangeName = 'binance' | 'coinbase' | 'alpaca' | 'custom';

interface Config {
  // API URLs
  apiUrl: string;
  websocketUrl: string;
  
  // Trading configuration
  tradingMode: TradingMode;
  defaultExchange: ExchangeName;
  
  // Exchange API keys
  binance: {
    apiKey: string;
    apiSecret: string;
  };
  
  coinbase: {
    apiKey: string;
    apiSecret: string;
    passphrase: string;
  };
  
  alpaca: {
    apiKey: string;
    apiSecret: string;
    paperTrading: boolean;
  };
  
  // Feature flags
  features: {
    enableNotifications: boolean;
    enableRealTimeUpdates: boolean;
    enableAdvancedCharting: boolean;
  };
  
  // Monitoring and analytics
  monitoring: {
    sentryDsn: string;
    googleAnalyticsId: string;
  };
}

// Default configuration with environment variables
const config: Config = {
  apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  websocketUrl: process.env.REACT_APP_WEBSOCKET_URL || 'ws://localhost:8001',
  
  tradingMode: (process.env.REACT_APP_TRADING_MODE as TradingMode) || 'mock',
  defaultExchange: (process.env.REACT_APP_DEFAULT_EXCHANGE as ExchangeName) || 'binance',
  
  binance: {
    apiKey: process.env.REACT_APP_BINANCE_API_KEY || '',
    apiSecret: process.env.REACT_APP_BINANCE_API_SECRET || '',
  },
  
  coinbase: {
    apiKey: process.env.REACT_APP_COINBASE_API_KEY || '',
    apiSecret: process.env.REACT_APP_COINBASE_API_SECRET || '',
    passphrase: process.env.REACT_APP_COINBASE_PASSPHRASE || '',
  },
  
  alpaca: {
    apiKey: process.env.REACT_APP_ALPACA_API_KEY || '',
    apiSecret: process.env.REACT_APP_ALPACA_API_SECRET || '',
    paperTrading: process.env.REACT_APP_ALPACA_PAPER_TRADING === 'true',
  },
  
  features: {
    enableNotifications: process.env.REACT_APP_ENABLE_NOTIFICATIONS !== 'false',
    enableRealTimeUpdates: process.env.REACT_APP_ENABLE_REAL_TIME_UPDATES !== 'false',
    enableAdvancedCharting: process.env.REACT_APP_ENABLE_ADVANCED_CHARTING !== 'false',
  },
  
  monitoring: {
    sentryDsn: process.env.REACT_APP_SENTRY_DSN || '',
    googleAnalyticsId: process.env.REACT_APP_GOOGLE_ANALYTICS_ID || '',
  },
};

// Utility functions to access configuration
export const getConfig = (): Config => config;

export const getTradingMode = (): TradingMode => config.tradingMode;

export const isLiveTrading = (): boolean => config.tradingMode === 'live';

export const isPaperTrading = (): boolean => config.tradingMode === 'paper';

export const isMockTrading = (): boolean => config.tradingMode === 'mock';

export const getApiUrl = (): string => config.apiUrl;

export const getWebsocketUrl = (): string => config.websocketUrl;

export const getExchangeConfig = (exchange: ExchangeName = config.defaultExchange) => {
  switch (exchange) {
    case 'binance':
      return config.binance;
    case 'coinbase':
      return config.coinbase;
    case 'alpaca':
      return config.alpaca;
    default:
      return null;
  }
};

export const isFeatureEnabled = (featureName: keyof Config['features']): boolean => {
  return config.features[featureName];
};

export default config;
