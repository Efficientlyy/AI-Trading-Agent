// Exchange configuration types and defaults

// Alpaca configuration
export interface AlpacaConfig {
  apiKey: string;
  apiSecret: string;
  paperTrading: boolean;
}

// Binance configuration
export interface BinanceConfig {
  apiKey: string;
  apiSecret: string;
  testnet: boolean;
}

// Coinbase configuration
export interface CoinbaseConfig {
  apiKey: string;
  apiSecret: string;
  passphrase: string;
  sandbox: boolean;
}

// Default configurations for testing
export const DEFAULT_ALPACA_CONFIG: AlpacaConfig = {
  apiKey: 'test-alpaca-key',
  apiSecret: 'test-alpaca-secret',
  paperTrading: true,
};

export const DEFAULT_BINANCE_CONFIG: BinanceConfig = {
  apiKey: 'test-binance-key',
  apiSecret: 'test-binance-secret',
  testnet: true,
};

export const DEFAULT_COINBASE_CONFIG: CoinbaseConfig = {
  apiKey: 'test-coinbase-key',
  apiSecret: 'test-coinbase-secret',
  passphrase: 'test-passphrase',
  sandbox: true,
};
