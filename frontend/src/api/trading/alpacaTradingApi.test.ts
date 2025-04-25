import { alpacaTradingApi } from './alpacaTradingApi.wrapper';
import { OrderSide, OrderType } from '../../types';
// Define AlpacaConfig interface if it's not imported correctly
interface AlpacaConfig {
  apiKey: string;
  apiSecret: string;
  paperTrading: boolean;
}
import axios from 'axios';
import { canMakeApiCall, recordApiCall, recordCircuitBreakerResult } from '../utils/monitoring';

// Mock axios
jest.mock('axios', () => {
  return {
    create: jest.fn(() => ({
      get: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      post: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      put: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      delete: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      request: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      getUri: jest.fn(),
      defaults: {},
      interceptors: {
        request: { 
          use: jest.fn(), 
          eject: jest.fn(),
          clear: jest.fn()
        },
        response: { 
          use: jest.fn(), 
          eject: jest.fn(),
          clear: jest.fn()
        },
      },
      head: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      options: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      patch: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    })),
    get: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    post: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    put: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    delete: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    request: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    isAxiosError: jest.fn().mockImplementation((error) => {
      return error && error.isAxiosError === true;
    }),
    getUri: jest.fn(),
    defaults: {},
    head: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    options: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    patch: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
  };
});
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock monitoring utilities
jest.mock('../utils/monitoring', () => {
  const mockCanMakeApiCall = jest.fn().mockReturnValue(true);
  // Add Jest mock methods to the function
  mockCanMakeApiCall.mockReturnValue = jest.fn().mockReturnValue(true);
  mockCanMakeApiCall.mockImplementation = jest.fn();
  
  return {
    recordApiCall: jest.fn(),
    canMakeApiCall: mockCanMakeApiCall,
    recordCircuitBreakerResult: jest.fn(),
    resetCircuitBreaker: jest.fn(),
  };
});

// Mock authenticated client
jest.mock('../client', () => ({
  createAuthenticatedClient: jest.fn().mockReturnValue({
    get: jest.fn(),
    post: jest.fn(),
    delete: jest.fn(),
  }),
}));

describe('Alpaca Trading API', () => {
  const mockConfig = {
    apiKey: 'test-api-key',
    apiSecret: 'test-api-secret',
    paperTrading: true,
  };
  
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Reset canMakeApiCall mock to allow calls by default
    (canMakeApiCall as jest.Mock).mockReturnValue(true);
  });
  
  describe('getPortfolio', () => {
    it('should fetch account information and format portfolio data correctly', async () => {
      // Mock responses
      const mockAccount = {
        cash: '10000',
        portfolio_value: '15000',
        equity: '15000',
        last_equity: '14500',
        multiplier: '1',
      };
      
      const mockPositions = [
        {
          symbol: 'AAPL',
          qty: '10',
          avg_entry_price: '150',
          current_price: '160',
          market_value: '1600',
          unrealized_pl: '100',
        },
      ];
      
      // Setup mocks
      const mockClient = {
        get: jest.fn()
          .mockResolvedValueOnce({ data: mockAccount })
          .mockResolvedValueOnce({ data: mockPositions }),
        post: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        put: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        delete: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        request: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        getUri: jest.fn(),
        defaults: {
          headers: {
            common: { Accept: 'application/json, text/plain, */*' },
            delete: {},
            get: {},
            head: {},
            post: { 'Content-Type': 'application/x-www-form-urlencoded' },
            put: { 'Content-Type': 'application/x-www-form-urlencoded' },
            patch: { 'Content-Type': 'application/x-www-form-urlencoded' },
          } },
        interceptors: {
          request: { 
            use: jest.fn(), 
            eject: jest.fn(),
            clear: jest.fn()
          },
          response: { 
            use: jest.fn(), 
            eject: jest.fn(),
            clear: jest.fn()
          },
        },
        head: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        options: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        patch: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        postForm: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        putForm: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        patchForm: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      };
      mockedAxios.create.mockReturnValue(mockClient as any);
      
      // Create API and call method
      const api = alpacaTradingApi('paper', mockConfig);
      const portfolio = await api.getPortfolio();
      
      // Assertions
      expect(mockClient.get).toHaveBeenCalledWith('/v2/account');
      expect(mockClient.get).toHaveBeenCalledWith('/v2/positions');
      
      // Check portfolio structure
      expect(portfolio).toHaveProperty('cash');
      expect(portfolio).toHaveProperty('total_value');
      expect(portfolio).toHaveProperty('positions');
      
      // Check portfolio values
      expect(portfolio.cash).toBe(10000);
      expect(portfolio.total_value).toBe(15000);
      expect(portfolio.positions).toHaveProperty('AAPL');
      
      // Check position details
      expect(portfolio.positions['AAPL'].quantity).toBe(10);
      expect(portfolio.positions['AAPL'].current_price).toBe(160);
      expect(portfolio.positions['AAPL'].market_value).toBe(1600);
      
      // Check circuit breaker and monitoring
      expect(canMakeApiCall).toHaveBeenCalledWith('Alpaca', 'getPortfolio');
      expect(recordApiCall).toHaveBeenCalledWith('Alpaca', 'getPortfolio', 'attempt');
      expect(recordApiCall).toHaveBeenCalledWith('Alpaca', 'getPortfolio', 'success');
      expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Alpaca', 'getPortfolio', true);
    });
    
    it('should use fallback when circuit breaker is open', async () => {
      // Mock circuit breaker to be open
      (canMakeApiCall as jest.Mock).mockReturnValue(false);
      
      // Mock backend fallback
      const mockBackendPortfolio = {
        cash: 5000,
        total_value: 10000,
        positions: {
          'AAPL': {
            symbol: 'AAPL',
            quantity: 10,
            entry_price: 150,
            current_price: 160,
            market_value: 1600,
            unrealized_pnl: 100,
          },
        },
      };
      
      const mockBackendClient = {
        get: jest.fn().mockImplementation(() => Promise.resolve({ data: { portfolio: mockBackendPortfolio } })),
      };
      require('../client').createAuthenticatedClient.mockReturnValue(mockBackendClient);
      
      // Create API and call method
      const api = alpacaTradingApi('paper', mockConfig);
      const portfolio = await api.getPortfolio();
      
      // Assertions
      expect(canMakeApiCall).toHaveBeenCalledWith('Alpaca', 'getPortfolio');
      expect(mockBackendClient.get).toHaveBeenCalledWith('/portfolio');
      expect(portfolio).toBe(mockBackendPortfolio);
    });
    
    it('should handle API errors and use fallback', async () => {
      // Mock API failure
      const mockClient = {
        get: jest.fn().mockImplementation(() => Promise.reject(new Error('API error'))),
        post: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        put: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        delete: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        request: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        getUri: jest.fn(),
        defaults: {
          headers: {
            common: { Accept: 'application/json, text/plain, */*' },
            delete: {},
            get: {},
            head: {},
            post: { 'Content-Type': 'application/x-www-form-urlencoded' },
            put: { 'Content-Type': 'application/x-www-form-urlencoded' },
            patch: { 'Content-Type': 'application/x-www-form-urlencoded' },
          } },
        interceptors: {
          request: { 
            use: jest.fn(), 
            eject: jest.fn(),
            clear: jest.fn()
          },
          response: { 
            use: jest.fn(), 
            eject: jest.fn(),
            clear: jest.fn()
          },
        },
        head: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        options: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        patch: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        postForm: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        putForm: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        patchForm: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      };
      mockedAxios.create.mockReturnValue(mockClient as any);
      
      // Mock backend fallback
      const mockBackendPortfolio = {
        cash: 5000,
        total_value: 10000,
        positions: {
          'AAPL': {
            symbol: 'AAPL',
            quantity: 10,
            entry_price: 150,
            current_price: 160,
            market_value: 1600,
            unrealized_pnl: 100,
          },
        },
      };
      
      const mockBackendClient = {
        get: jest.fn().mockImplementation(() => Promise.resolve({ data: { portfolio: mockBackendPortfolio } })),
      };
      require('../client').createAuthenticatedClient.mockReturnValue(mockBackendClient);
      
      // Create API and call method
      const api = alpacaTradingApi('paper', mockConfig);
      const portfolio = await api.getPortfolio();
      
      // Assertions
      expect(canMakeApiCall).toHaveBeenCalledWith('Alpaca', 'getPortfolio');
      expect(recordApiCall).toHaveBeenCalledWith('Alpaca', 'getPortfolio', 'attempt');
      expect(recordApiCall).toHaveBeenCalledWith('Alpaca', 'getPortfolio', 'failure');
      expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Alpaca', 'getPortfolio', false);
      expect(mockBackendClient.get).toHaveBeenCalledWith('/portfolio');
      expect(portfolio).toBe(mockBackendPortfolio);
    });
  });
  
  describe('createOrder', () => {
    it('should create a market order correctly', async () => {
      // Mock response
      const mockOrderResponse = {
        id: '12345',
        client_order_id: 'test-client-order-id',
        symbol: 'AAPL',
        side: 'buy',
        type: 'market',
        qty: '10',
        filled_qty: '10',
        status: 'filled',
        created_at: '2023-01-01T12:00:00Z',
        updated_at: '2023-01-01T12:01:00Z',
      };
      
      // Setup mock
      const mockClient = {
        post: jest.fn().mockImplementation(() => Promise.resolve({ data: mockOrderResponse })),
        get: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        put: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        delete: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        request: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        getUri: jest.fn(),
        defaults: {
          headers: {
            common: { Accept: 'application/json, text/plain, */*' },
            delete: {},
            get: {},
            head: {},
            post: { 'Content-Type': 'application/x-www-form-urlencoded' },
            put: { 'Content-Type': 'application/x-www-form-urlencoded' },
            patch: { 'Content-Type': 'application/x-www-form-urlencoded' },
          } },
        interceptors: {
          request: { 
            use: jest.fn(), 
            eject: jest.fn(),
            clear: jest.fn()
          },
          response: { 
            use: jest.fn(), 
            eject: jest.fn(),
            clear: jest.fn()
          },
        },
        head: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        options: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        patch: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        postForm: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        putForm: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
        patchForm: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      };
      mockedAxios.create.mockReturnValue(mockClient as any);
      
      // Create API and call method
      const api = alpacaTradingApi('paper', mockConfig);
      const orderRequest = {
        symbol: 'BTC/USDT',
        side: 'buy' as const,
        order_type: 'market',
        type: OrderType.MARKET, // Add the required 'type' property
        quantity: 1
      };
      
      const order = await api.createOrder(orderRequest);
      
      // Assertions
      expect(mockClient.post).toHaveBeenCalledWith('/v2/orders', {
        symbol: 'BTC/USDT',
        side: 'buy',
        type: 'market',
        qty: 1,
        time_in_force: 'day',
      });
      
      // Check order structure
      expect(order).toHaveProperty('id');
      expect(order).toHaveProperty('symbol');
      expect(order).toHaveProperty('type');
      expect(order).toHaveProperty('side');
      expect(order).toHaveProperty('quantity');
      expect(order).toHaveProperty('status');
      
      // Check circuit breaker and monitoring
      expect(canMakeApiCall).toHaveBeenCalledWith('Alpaca', 'createOrder');
      expect(recordApiCall).toHaveBeenCalledWith('Alpaca', 'createOrder', 'attempt');
      expect(recordApiCall).toHaveBeenCalledWith('Alpaca', 'createOrder', 'success');
      expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Alpaca', 'createOrder', true);
    });
  });
});
