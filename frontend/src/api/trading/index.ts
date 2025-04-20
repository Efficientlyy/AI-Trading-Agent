import { getTradingMode, getExchangeConfig, ExchangeName } from '../../config';
import { OrderRequest, Order, Portfolio, Position } from '../../types';
import { mockTradingApi } from './mockTradingApi';
import { binanceTradingApi } from './binanceTradingApi';
import { coinbaseTradingApi } from './coinbaseTradingApi';
import { alpacaTradingApi } from './alpacaTradingApi.wrapper';
import { createPaperTradingApi } from './paperTradingApi';

// Define the trading API interface
export interface TradingApi {
  // Account and portfolio methods
  getPortfolio(): Promise<Portfolio>;
  getPositions(): Promise<Record<string, Position>>;
  getBalance(asset?: string): Promise<number>;
  
  // Order management methods
  createOrder(orderRequest: OrderRequest): Promise<Order>;
  cancelOrder(orderId: string): Promise<boolean>;
  getOrders(status?: string): Promise<Order[]>;
  getOrder(orderId: string): Promise<Order | null>;
  
  // Market data methods
  getMarketPrice(symbol: string): Promise<number>;
  getOrderBook(symbol: string, limit?: number): Promise<{ bids: any[], asks: any[] }>;
  getTicker(symbol: string): Promise<{ price: number, volume: number, change: number }>;
  
  // Exchange info methods
  getExchangeInfo(): Promise<any>;
  getSymbols(): Promise<string[]>;
  getAssetInfo(symbol: string): Promise<any>;
}

// Factory function to create the appropriate trading API based on configuration
export const createTradingApi = (exchange?: ExchangeName): TradingApi => {
  const tradingMode = getTradingMode();
  const selectedExchange = exchange || 'binance';
  
  // For mock mode, always return the mock API regardless of exchange
  if (tradingMode === 'mock') {
    return mockTradingApi;
  }
  
  // For paper or live mode, use the appropriate exchange API
  const exchangeConfig = getExchangeConfig(selectedExchange);
  
  if (!exchangeConfig) {
    throw new Error(`No configuration found for exchange: ${selectedExchange}`);
  }
  
  // Create the base API for the selected exchange
  let baseApi: TradingApi;
  
  switch (selectedExchange) {
    case 'binance':
      baseApi = binanceTradingApi(tradingMode, exchangeConfig);
      break;
    case 'coinbase':
      baseApi = coinbaseTradingApi(tradingMode, exchangeConfig);
      break;
    case 'alpaca':
      baseApi = alpacaTradingApi(tradingMode, exchangeConfig);
      break;
    default:
      throw new Error(`Unsupported exchange: ${selectedExchange}`);
  }
  
  // For paper trading mode, wrap the real API with paper trading simulation
  if (tradingMode === 'paper') {
    return createPaperTradingApi(baseApi);
  }
  
  // For live trading, return the real API
  return baseApi;
};

// Create a default trading API instance
export const tradingApi = createTradingApi();

// Export individual exchange APIs for direct usage if needed
export { mockTradingApi } from './mockTradingApi';
export { binanceTradingApi } from './binanceTradingApi';
export { coinbaseTradingApi } from './coinbaseTradingApi';
export { alpacaTradingApi } from './alpacaTradingApi.wrapper';
export { createPaperTradingApi } from './paperTradingApi';
