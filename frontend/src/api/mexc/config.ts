// MEXC API Configuration

export const MEXC_API_CONFIG = {
  // Base URLs
  REST_BASE_URL: 'https://api.mexc.com',
  WS_BASE_URL: 'wss://wbs.mexc.com/ws',
  
  // API Endpoints
  ENDPOINTS: {
    TICKER_24HR: '/api/v3/ticker/24hr',
    KLINES: '/api/v3/klines',
    DEPTH: '/api/v3/depth',
    TRADES: '/api/v3/trades',
  },
  
  // Default Parameters
  DEFAULTS: {
    DEPTH_LIMIT: 20,
    TRADES_LIMIT: 20,
    KLINES_LIMIT: 100,
  },
  
  // WebSocket settings
  WS: {
    PING_INTERVAL: 30000, // 30 seconds
    RECONNECT_ATTEMPTS: 5,
    RECONNECT_DELAY_BASE: 1000, // Base delay in ms
  },
  
  // Supported Trading Pairs
  SUPPORTED_PAIRS: [
    'BTC/USDT',
    'ETH/USDT',
    'SOL/USDT',
    'XRP/USDT',
    'ADA/USDT',
    'DOGE/USDT',
    'DOT/USDT',
    'AVAX/USDT'
  ],
  
  // Supported Timeframes
  SUPPORTED_TIMEFRAMES: [
    '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
  ],
};

export default MEXC_API_CONFIG;