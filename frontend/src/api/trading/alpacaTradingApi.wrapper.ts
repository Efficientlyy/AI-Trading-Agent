import { TradingMode } from '../../config';
import { createAlpacaTradingApi } from './alpacaTradingApi';

// This is a wrapper file to provide the expected export name for tests
export interface AlpacaConfig {
  apiKey: string;
  apiSecret: string;
  paperTrading: boolean;
}

// Export the API creator function with the expected name for tests
export const alpacaTradingApi = (tradingMode: TradingMode, config: AlpacaConfig) => {
  return createAlpacaTradingApi(config);
};
