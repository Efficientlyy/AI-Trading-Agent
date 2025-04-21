import { alpacaTradingApi } from './alpacaTradingApi.wrapper';
import { binanceTradingApi } from './binanceTradingApi';
import { coinbaseTradingApi } from './coinbaseTradingApi';
import axios from 'axios';
import { ApiError } from '../utils/errorHandling';
import { canMakeApiCall, recordApiCall, recordCircuitBreakerResult } from '../utils/monitoring';

import { mockAxios, mockMonitoring } from '../../tests/mocks/globalMocks';

// Mock axios
jest.mock('axios', () => mockAxios());
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock monitoring utilities
jest.mock('../utils/monitoring', () => mockMonitoring());

// Mock authenticated client
jest.mock('../client', () => ({
  createAuthenticatedClient: jest.fn().mockImplementation(() => ({
    get: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    post: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    delete: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
  })),
}));

describe('Trading API Circuit Breaker Integration', () => {
  // Mock configs
  const alpacaConfig = {
    apiKey: 'alpaca-api-key',
    apiSecret: 'alpaca-api-secret',
    paperTrading: true,
  };

  const binanceConfig = {
    apiKey: 'binance-api-key',
    apiSecret: 'binance-api-secret',
    testnet: true, // Add this if required by BinanceConfig
  };

  const coinbaseConfig = {
    apiKey: 'coinbase-api-key',
    apiSecret: 'coinbase-api-secret',
    passphrase: 'coinbase-passphrase',
    sandbox: true, // Add this if required by CoinbaseConfig
  };


  beforeEach(() => {
    jest.clearAllMocks();
    
    // Setup default axios mock responses
    (mockedAxios.create as jest.Mock).mockReturnValue({
      get: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      post: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      delete: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      interceptors: {
        request: { use: jest.fn() },
        response: { use: jest.fn() },
      },
    } as any);
    
    // Reset canMakeApiCall mock to allow calls by default
    (canMakeApiCall as jest.Mock).mockReturnValue(true);
  });

  describe('Alpaca Trading API Circuit Breaker', () => {
    it('should use fallback when circuit breaker is open', async () => {
      // Mock circuit breaker to be open
      (canMakeApiCall as jest.Mock).mockReturnValue(false);
      
      // Setup backend fallback
      const mockBackendClient = {
        get: jest.fn().mockImplementation(() => Promise.resolve({ 
          data: { 
            portfolio: { 
              total_value: 10000, 
              cash: 5000, 
              positions: {},
              daily_pnl: 100,
              margin_multiplier: 1
            } } 
        })),
      };
      require('../client').createAuthenticatedClient.mockReturnValue(mockBackendClient);
      
      // Create API and call method
      const api = alpacaTradingApi('paper', alpacaConfig);
      const portfolio = await api.getPortfolio();
      
      // Assertions
      expect(canMakeApiCall).toHaveBeenCalledWith('Alpaca', 'getPortfolio');
      expect(mockBackendClient.get).toHaveBeenCalledWith('/portfolio');
      expect(portfolio.total_value).toBe(10000);
      expect(portfolio.cash).toBe(5000);
    });
  });

  describe('Binance Trading API Circuit Breaker', () => {
    it('should use fallback when circuit breaker is open', async () => {
      // Mock circuit breaker to be open
      (canMakeApiCall as jest.Mock).mockReturnValue(false);
      
      // Setup backend fallback
      const mockBackendClient = {
        get: jest.fn().mockImplementation(() => Promise.resolve({ 
          data: { 
            portfolio: { 
              total_value: 20000, 
              cash: 10000, 
              positions: {},
              daily_pnl: 200,
              margin_multiplier: 2
            } } 
        })),
      };
      require('../client').createAuthenticatedClient.mockReturnValue(mockBackendClient);
      
      // Create API and call method
      const api = binanceTradingApi('paper', binanceConfig);
      const portfolio = await api.getPortfolio();
      
      // Assertions
      expect(canMakeApiCall).toHaveBeenCalledWith('Binance', 'getPortfolio');
      expect(mockBackendClient.get).toHaveBeenCalledWith('/portfolio');
      expect(portfolio.total_value).toBe(20000);
      expect(portfolio.cash).toBe(10000);
    });
  });

  describe('Coinbase Trading API Circuit Breaker', () => {
    it('should use fallback when circuit breaker is open', async () => {
      // Mock circuit breaker to be open
      (canMakeApiCall as jest.Mock).mockReturnValue(false);
      
      // Setup backend fallback
      const mockBackendClient = {
        get: jest.fn().mockImplementation(() => Promise.resolve({ 
          data: { 
            portfolio: { 
              total_value: 30000, 
              cash: 15000, 
              positions: {},
              daily_pnl: 300,
              margin_multiplier: 3
            } } 
        })),
      };
      require('../client').createAuthenticatedClient.mockReturnValue(mockBackendClient);
      
      // Create API and call method
      const api = coinbaseTradingApi('paper', coinbaseConfig);
      const portfolio = await api.getPortfolio();
      
      // Assertions
      expect(canMakeApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio');
      expect(mockBackendClient.get).toHaveBeenCalledWith('/portfolio');
      expect(portfolio.total_value).toBe(30000);
      expect(portfolio.cash).toBe(15000);
    });

    it('should record API call failure and use fallback', async () => {
      // Mock API failure
      const mockClient = {
        get: jest.fn().mockImplementation(() => Promise.reject(new ApiError('Authentication failed', 401, {}, false))),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() },
        },
      };
      mockedAxios.create.mockReturnValue(mockClient as any);
      
      // Setup backend fallback
      const mockBackendClient = {
        get: jest.fn().mockImplementation(() => Promise.resolve({ 
          data: { 
            portfolio: { 
              total_value: 30000, 
              cash: 15000, 
              positions: {},
              daily_pnl: 300,
              margin_multiplier: 3
            } } 
        })),
      };
      require('../client').createAuthenticatedClient.mockReturnValue(mockBackendClient);
      
      // Create API and call method
      const api = coinbaseTradingApi('paper', coinbaseConfig);
      const portfolio = await api.getPortfolio();
      
      // Assertions
      expect(mockClient.get).toHaveBeenCalledTimes(1); // No retry for non-retryable errors
      expect(mockBackendClient.get).toHaveBeenCalledWith('/portfolio');
      expect(recordApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio', 'attempt');
      expect(recordApiCall).toHaveBeenCalledWith('Coinbase', 'getPortfolio', 'failure');
      expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Coinbase', 'getPortfolio', false);
      expect(portfolio.total_value).toBe(30000);
      expect(portfolio.cash).toBe(15000);
    });
  });
});
