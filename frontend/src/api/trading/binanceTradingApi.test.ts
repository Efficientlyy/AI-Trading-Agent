import { binanceTradingApi } from './binanceTradingApi';
import axios from 'axios';
import { canMakeApiCall, recordApiCall, recordCircuitBreakerResult } from '../utils/monitoring';
import { OrderSide, OrderType } from '../../types';
// Mock Date for consistent test results
const mockDate = new Date('2023-01-01T00:00:00Z');
jest.spyOn(global, 'Date').mockImplementation(() => mockDate);

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
});
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock monitoring utilities
jest.mock('../utils/monitoring', () => {
  return {
    recordApiCall: jest.fn(),
    canMakeApiCall: jest.fn().mockReturnValue(true),
    recordCircuitBreakerResult: jest.fn(),
    resetCircuitBreaker: jest.fn(),
    getCircuitBreakerState: jest.fn().mockReturnValue({
      state: 'closed',
      remainingTimeMs: 0
    }),
    getApiCallMetrics: jest.fn().mockReturnValue({
      totalCalls: 0,
      successCalls: 0,
      failedCalls: 0,
      totalDuration: 0,
      minDuration: 0,
      maxDuration: 0,
      lastCallTime: 0
    })
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

// Mock data for Binance API tests
const mockAccountInfo = {
  balances: [
    { asset: 'BTC', free: '1.0', locked: '0.0' },
    { asset: 'ETH', free: '10.0', locked: '0.0' },
    { asset: 'USDT', free: '1000.0', locked: '0.0' }
  ]
};

describe('Binance Trading API', () => {
  const mockConfig = {
    apiKey: 'test-api-key',
    apiSecret: 'test-api-secret',
  };
  
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Setup default axios mock responses
    (mockedAxios.create as jest.Mock).mockReturnValue({
      get: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
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
    });
    
    // Reset canMakeApiCall mock to allow calls by default
    (canMakeApiCall as jest.Mock).mockReturnValue(true);
  });
  
  describe('getPortfolio', () => {
    it('should fetch account information and format portfolio data correctly', async () => {
      // Mock responses
      const mockAccount = {
        makerCommission: 10,
        takerCommission: 10,
        buyerCommission: 0,
        sellerCommission: 0,
        canTrade: true,
        canWithdraw: true,
        canDeposit: true,
        updateTime: 1617979287394,
        accountType: 'SPOT',
        balances: [
          { asset: 'BTC', free: '0.1', locked: '0.0' },
          { asset: 'ETH', free: '2.0', locked: '0.5' },
          { asset: 'USDT', free: '5000.0', locked: '0.0' },
        ],
      };
      
      const mockPrices = [
        { symbol: 'BTCUSDT', price: '50000.00' },
        { symbol: 'ETHUSDT', price: '3000.00' },
      ];
      
      // Setup mocks
      const mockClient = {
        get: jest.fn()
          .mockResolvedValueOnce({ data: mockAccount })
          .mockResolvedValueOnce({ data: mockPrices }),
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
      };
      (mockedAxios.create as jest.Mock).mockReturnValue(mockClient);
      
      // Create API and call method
      const api = binanceTradingApi('paper', mockConfig);
      const portfolio = await api.getPortfolio();
      
      // Assertions
      expect(mockClient.get).toHaveBeenCalledWith('/api/v3/account');
      expect(mockClient.get).toHaveBeenCalledWith('/api/v3/ticker/price');
      
      // Check portfolio structure
      expect(portfolio).toHaveProperty('cash');
      expect(portfolio).toHaveProperty('total_value');
      expect(portfolio).toHaveProperty('positions');
      
      // Check portfolio values
      expect(portfolio.cash).toBe(5000);
      expect(portfolio.positions).toHaveProperty('BTC/USDT');
      expect(portfolio.positions).toHaveProperty('ETH/USDT');
      
      // Check position details
      expect(portfolio.positions['BTC/USDT'].quantity).toBe(0.1);
      expect(portfolio.positions['BTC/USDT'].current_price).toBe(50000);
      expect(portfolio.positions['BTC/USDT'].market_value).toBe(5000);
      
      expect(portfolio.positions['ETH/USDT'].quantity).toBe(2.5); // free + locked
      expect(portfolio.positions['ETH/USDT'].current_price).toBe(3000);
      expect(portfolio.positions['ETH/USDT'].market_value).toBe(7500);
      
      // Check total value (cash + positions)
      expect(portfolio.total_value).toBe(17500); // 5000 (cash) + 5000 (BTC) + 7500 (ETH)
      
      // Check circuit breaker and monitoring
      expect(canMakeApiCall).toHaveBeenCalledWith('Binance', 'getPortfolio');
      expect(recordApiCall).toHaveBeenCalledWith('Binance', 'getPortfolio', 'attempt');
      expect(recordApiCall).toHaveBeenCalledWith('Binance', 'getPortfolio', 'success');
      expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Binance', 'getPortfolio', true);
    });
    
    it('should use fallback when circuit breaker is open', async () => {
      // Mock circuit breaker to be open
      (canMakeApiCall as jest.Mock).mockReturnValue(false);
      
      // Mock backend fallback
      const mockBackendPortfolio = {
        cash: 10000,
        total_value: 25000,
        positions: {
          'BTC/USDT': {
            symbol: 'BTC/USDT',
            quantity: 0.2,
            entry_price: 45000,
            current_price: 50000,
            market_value: 10000,
            unrealized_pnl: 1000,
          },
          'ETH/USDT': {
            symbol: 'ETH/USDT',
            quantity: 2.0,
            entry_price: 2500,
            current_price: 3000,
            market_value: 6000,
            unrealized_pnl: 1000,
          },
        },
      };
      
      const mockBackendClient = {
        get: jest.fn().mockImplementation(() => Promise.resolve({ data: { portfolio: mockBackendPortfolio } })),
      };
      require('../client').createAuthenticatedClient.mockReturnValue(mockBackendClient);
      
      // Create API and call method
      const api = binanceTradingApi('paper', mockConfig);
      const portfolio = await api.getPortfolio();
      
      // Assertions
      expect(canMakeApiCall).toHaveBeenCalledWith('Binance', 'getPortfolio');
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
      };
      (mockedAxios.create as jest.Mock).mockReturnValue(mockClient);
      
      // Mock backend fallback
      const mockBackendPortfolio = {
        cash: 10000,
        total_value: 25000,
        positions: {
          'BTC/USDT': {
            symbol: 'BTC/USDT',
            quantity: 0.2,
            entry_price: 45000,
            current_price: 50000,
            market_value: 10000,
            unrealized_pnl: 1000,
          },
        },
      };
      
      const mockBackendClient = {
        get: jest.fn().mockImplementation(() => Promise.resolve({ data: { portfolio: mockBackendPortfolio } })),
      };
      require('../client').createAuthenticatedClient.mockReturnValue(mockBackendClient);
      
      // Create API and call method
      const api = binanceTradingApi('paper', mockConfig);
      const portfolio = await api.getPortfolio();
      
      // Assertions
      expect(canMakeApiCall).toHaveBeenCalledWith('Binance', 'getPortfolio');
      expect(recordApiCall).toHaveBeenCalledWith('Binance', 'getPortfolio', 'attempt');
      expect(recordApiCall).toHaveBeenCalledWith('Binance', 'getPortfolio', 'failure');
      expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Binance', 'getPortfolio', false);
      expect(mockBackendClient.get).toHaveBeenCalledWith('/portfolio');
      expect(portfolio).toBe(mockBackendPortfolio);
    });
  });
  
  describe('createOrder', () => {
    it('should create a market order correctly', async () => {
      // Mock response
      const mockOrderResponse = {
        symbol: 'BTCUSDT',
        orderId: 12345,
        orderListId: -1,
        clientOrderId: 'test-client-order-id',
        transactTime: 1617979287394,
        price: '0.00000000',
        origQty: '0.10000000',
        executedQty: '0.10000000',
        status: 'FILLED',
        timeInForce: 'GTC',
        type: 'MARKET',
        side: 'BUY',
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
      };
      (mockedAxios.create as jest.Mock).mockReturnValue(mockClient);
      
      // Create API and call method
      const api = binanceTradingApi('paper', mockConfig);
      const orderRequest = {
        symbol: 'BTC/USDT',
        side: 'buy' as const,
        order_type: 'market',
        type: OrderType.MARKET, // Add required 'type' property
        quantity: 1
      };
      
      const order = await api.createOrder(orderRequest);
      
      // Assertions
      expect(mockClient.post).toHaveBeenCalledWith('/api/v3/order', {
        symbol: 'BTCUSDT',
        side: 'BUY',
        type: 'MARKET',
        quantity: 0.1,
      });
      
      // Check order structure
      expect(order).toHaveProperty('id');
      expect(order).toHaveProperty('symbol');
      expect(order).toHaveProperty('type');
      expect(order).toHaveProperty('side');
      expect(order).toHaveProperty('quantity');
      expect(order).toHaveProperty('status');
      
      // Check order values
      expect(order.id).toBe('12345');
      expect(order.symbol).toBe('BTC/USDT');
      expect(order.type).toBe(OrderType.MARKET);
      expect(order.side).toBe(OrderSide.BUY);
      expect(order.quantity).toBe(0.1);
      expect(order.status).toBe('FILLED');
      
      // Check circuit breaker and monitoring
      expect(canMakeApiCall).toHaveBeenCalledWith('Binance', 'createOrder');
      expect(recordApiCall).toHaveBeenCalledWith('Binance', 'createOrder', 'attempt');
      expect(recordApiCall).toHaveBeenCalledWith('Binance', 'createOrder', 'success');
      expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Binance', 'createOrder', true);
    });
    
    it('should handle API errors and use fallback', async () => {
      // Mock API failure
      const mockClient = {
        post: jest.fn().mockImplementation(() => Promise.reject(new Error('API error'))),
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
      };
      (mockedAxios.create as jest.Mock).mockReturnValue(mockClient);
      
      // Mock backend fallback
      const mockBackendOrder = {
        id: '54321',
        symbol: 'BTC/USDT',
        type: OrderType.MARKET,
        side: OrderSide.BUY,
        quantity: 0.1,
        status: 'NEW',
        createdAt: new Date(),
        updatedAt: new Date(),
        clientOrderId: 'backend-client-order-id',
      };
      
      const mockBackendClient = {
        post: jest.fn().mockImplementation(() => Promise.resolve({ data: { order: mockBackendOrder } })),
      };
      require('../client').createAuthenticatedClient.mockReturnValue(mockBackendClient);
      
      // Create API and call method
      const api = binanceTradingApi('paper', mockConfig);
      const orderRequest = {
        symbol: 'BTC/USDT',
        side: 'buy' as 'buy', // Type assertion to match OrderRequest type
        order_type: 'market' as 'market', // Type assertion to match OrderRequest type
        type: OrderType.MARKET, // Add required 'type' property
        quantity: 0.1,
      };
      
      const order = await api.createOrder(orderRequest);
      
      // Assertions
      expect(canMakeApiCall).toHaveBeenCalledWith('Binance', 'createOrder');
      expect(recordApiCall).toHaveBeenCalledWith('Binance', 'createOrder', 'attempt');
      expect(recordApiCall).toHaveBeenCalledWith('Binance', 'createOrder', 'failure');
      expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Binance', 'createOrder', false);
      expect(mockBackendClient.post).toHaveBeenCalledWith('/orders', orderRequest);
      expect(order).toBe(mockBackendOrder);
    });
  });
  
  describe('getMarketPrice', () => {
    it('should fetch the current market price correctly', async () => {
      // Mock response
      const mockPriceResponse = {
        symbol: 'BTCUSDT',
        price: '50000.00',
      };
      
      // Setup mock
      const mockClient = {
        get: jest.fn().mockImplementation(() => Promise.resolve({ data: mockPriceResponse })),
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
      };
      (mockedAxios.create as jest.Mock).mockReturnValue(mockClient);
      
      // Create API and call method
      const api = binanceTradingApi('paper', mockConfig);
      const price = await api.getMarketPrice('BTC/USDT');
      
      // Assertions
      expect(mockClient.get).toHaveBeenCalledWith('/api/v3/ticker/price', {
        params: { symbol: 'BTCUSDT' },
      });
      expect(price).toBe(50000);
      
      // Check circuit breaker and monitoring
      expect(canMakeApiCall).toHaveBeenCalledWith('Binance', 'getMarketPrice');
      expect(recordApiCall).toHaveBeenCalledWith('Binance', 'getMarketPrice', 'attempt');
      expect(recordApiCall).toHaveBeenCalledWith('Binance', 'getMarketPrice', 'success');
      expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Binance', 'getMarketPrice', true);
    });
  });
});
