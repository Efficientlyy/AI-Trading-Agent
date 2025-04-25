import { createPaperTradingApi } from './paperTradingApi';
import { OrderSide, OrderType, OrderStatus } from '../../types';

// Mock base trading API
const mockBaseApi = {
  getPortfolio: jest.fn(),
  getPositions: jest.fn(),
  getBalance: jest.fn(),
  createOrder: jest.fn(),
  cancelOrder: jest.fn(),
  getOrders: jest.fn(),
  getOrder: jest.fn(),
  getMarketPrice: jest.fn(),
  getOrderBook: jest.fn(),
  getTicker: jest.fn(),
  getExchangeInfo: jest.fn(),
  getSymbols: jest.fn(),
  getAssetInfo: jest.fn(),
};

// Mock authenticated client
jest.mock('../client', () => ({
  createAuthenticatedClient: jest.fn().mockReturnValue({
    get: jest.fn(),
    post: jest.fn(),
    delete: jest.fn(),
  }),
}));

// Mock localStorage
const mockLocalStorage = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => {
      store[key] = value;
    }),
    clear: jest.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

// Mock setInterval and clearInterval
jest.useFakeTimers();

describe('Paper Trading API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.clear();
    
    // Reset base API mock implementations
    Object.keys(mockBaseApi).forEach(key => {
      (mockBaseApi as any)[key].mockReset();
    });
    
    // Default implementations for commonly used methods
    mockBaseApi.getMarketPrice.mockImplementation(() => Promise.resolve(50000));
  });
  
  describe('getPortfolio', () => {
    it('should return the initial portfolio with default values', async () => {
      const paperApi = createPaperTradingApi(mockBaseApi);
      const portfolio = await paperApi.getPortfolio();
      
      expect(portfolio).toHaveProperty('cash');
      expect(portfolio).toHaveProperty('total_value');
      expect(portfolio).toHaveProperty('positions');
      
      expect(portfolio.cash).toBe(100000);
      expect(portfolio.total_value).toBe(100000);
      expect(Object.keys(portfolio.positions).length).toBe(0);
    });
    
    it('should load saved portfolio state from localStorage', async () => {
      // Set up saved state
      const savedState = {
        portfolio: {
          cash: 90000,
          total_value: 110000,
          positions: {
            'BTC/USDT': {
              symbol: 'BTC/USDT',
              quantity: 0.4,
              entry_price: 50000,
              current_price: 50000,
              market_value: 20000,
              unrealized_pnl: 0,
              realized_pnl: 0,
            },
          },
          daily_pnl: 1000,
        },
        orders: [],
        orderIdMap: [],
      };
      
      mockLocalStorage.setItem('paperTradingState', JSON.stringify(savedState));
      
      const paperApi = createPaperTradingApi(mockBaseApi);
      const portfolio = await paperApi.getPortfolio();
      
      expect(portfolio.cash).toBe(90000);
      expect(portfolio.total_value).toBe(110000);
      expect(portfolio.daily_pnl).toBe(1000);
      expect(Object.keys(portfolio.positions).length).toBe(1);
      expect(portfolio.positions['BTC/USDT']).toBeDefined();
      expect(portfolio.positions['BTC/USDT'].quantity).toBe(0.4);
      expect(portfolio.positions['BTC/USDT'].market_value).toBe(20000);
    });
  });
  
  describe('createOrder', () => {
    it('should create a market buy order and update portfolio correctly', async () => {
      // Setup
      mockBaseApi.getMarketPrice.mockImplementation(() => Promise.resolve(50000));
      
      const paperApi = createPaperTradingApi(mockBaseApi);
      
      // Create a market buy order
      const orderRequest = {
        symbol: 'BTC/USDT',
        side: 'buy' as const,
        order_type: 'market',
        type: OrderType.MARKET, // Add the required 'type' property
        quantity: 0.2,
      };
      
      // Execute order
      const order = await paperApi.createOrder(orderRequest);
      
      // Verify order properties
      expect(order).toHaveProperty('id');
      expect(order).toHaveProperty('symbol');
      expect(order).toHaveProperty('type');
      expect(order).toHaveProperty('side');
      expect(order).toHaveProperty('quantity');
      expect(order).toHaveProperty('status');
      
      expect(order.symbol).toBe('BTC/USDT');
      expect(order.type).toBe(OrderType.MARKET);
      expect(order.side).toBe(OrderSide.BUY);
      expect(order.quantity).toBe(0.2);
      expect(order.status).toBe(OrderStatus.FILLED); // Market orders should be filled immediately
      
      // Check portfolio update
      const portfolio = await paperApi.getPortfolio();
      expect(portfolio.cash).toBe(90000); // 100000 - (0.2 * 50000)
      expect(portfolio.positions['BTC/USDT']).toBeDefined();
      expect(portfolio.positions['BTC/USDT'].quantity).toBe(0.2);
      expect(portfolio.positions['BTC/USDT'].entry_price).toBe(50000);
      expect(portfolio.positions['BTC/USDT'].market_value).toBe(10000);
    });
    
    it('should create a market sell order and update portfolio correctly', async () => {
      // Setup initial portfolio with BTC position
      const savedState = {
        portfolio: {
          cash: 90000,
          total_value: 110000,
          positions: {
            'BTC/USDT': {
              symbol: 'BTC/USDT',
              quantity: 0.4,
              entry_price: 50000,
              current_price: 50000,
              market_value: 20000,
              unrealized_pnl: 0,
              realized_pnl: 0,
            },
          },
        },
        orders: [],
        orderIdMap: [],
      };
      
      mockLocalStorage.setItem('paperTradingState', JSON.stringify(savedState));
      mockBaseApi.getMarketPrice.mockImplementation(() => Promise.resolve(50000));
      
      const paperApi = createPaperTradingApi(mockBaseApi);
      
      // Create a market sell order
      const orderRequest = {
        symbol: 'BTC/USDT',
        side: 'sell' as const,
        order_type: 'market',
        type: OrderType.MARKET, // Add the required 'type' property
        quantity: 0.2,
      };
      
      // Execute order
      const order = await paperApi.createOrder(orderRequest);
      
      // Verify order
      expect(order.symbol).toBe('BTC/USDT');
      expect(order.type).toBe(OrderType.MARKET);
      expect(order.side).toBe(OrderSide.SELL);
      expect(order.quantity).toBe(0.2);
      expect(order.status).toBe(OrderStatus.FILLED);
      
      // Check portfolio update
      const portfolio = await paperApi.getPortfolio();
      expect(portfolio.cash).toBe(100000); // 90000 + (0.2 * 50000)
      expect(portfolio.positions['BTC/USDT']).toBeDefined();
      expect(portfolio.positions['BTC/USDT'].quantity).toBe(0.2); // 0.4 - 0.2
      expect(portfolio.positions['BTC/USDT'].market_value).toBe(10000);
    });
    
    it('should create a limit buy order that remains open until price conditions are met', async () => {
      // Setup
      let currentPrice = 51000; // Above limit price
      mockBaseApi.getMarketPrice.mockImplementation(() => Promise.resolve(currentPrice));
      
      const paperApi = createPaperTradingApi(mockBaseApi);
      
      // Create a limit buy order
      const orderRequest = {
        symbol: 'BTC/USDT',
        side: 'buy' as const,
        order_type: 'limit',
        type: OrderType.LIMIT, // Add the required 'type' property
        quantity: 0.2,
        price: 50000, // Limit price
      };
      
      // Execute order
      const order = await paperApi.createOrder(orderRequest);
      
      // Verify order is open
      expect(order.status).toBe(OrderStatus.NEW);
      
      // Check portfolio - cash should not be reduced yet
      let portfolio = await paperApi.getPortfolio();
      expect(portfolio.cash).toBe(100000);
      expect(Object.keys(portfolio.positions).length).toBe(0);
      
      // Simulate price dropping below limit price
      currentPrice = 49000;
      
      // Advance timers to trigger order processing
      jest.advanceTimersByTime(5000);
      
      // Check that order is now filled
      const orders = await paperApi.getOrders();
      expect(orders[0].status).toBe(OrderStatus.FILLED);
      
      // Check portfolio update
      portfolio = await paperApi.getPortfolio();
      expect(portfolio.cash).toBe(90000); // 100000 - (0.2 * 50000) - limit price is used, not market price
      expect(portfolio.positions['BTC/USDT']).toBeDefined();
      expect(portfolio.positions['BTC/USDT'].quantity).toBe(0.2);
    });
    
    it('should reject orders if insufficient funds or assets are available', async () => {
      mockBaseApi.getMarketPrice.mockImplementation(() => Promise.resolve(50000));
      
      const paperApi = createPaperTradingApi(mockBaseApi);
      
      // Create a market buy order with insufficient funds
      const largeOrderRequest = {
        symbol: 'BTC/USDT',
        side: 'buy' as const,
        order_type: 'market',
        type: OrderType.MARKET, // Add the required 'type' property
        quantity: 3, // 3 BTC at $50,000 = $150,000 (exceeds $100,000 cash)
      };
      
      // Execute order
      const order = await paperApi.createOrder(largeOrderRequest);
      
      // Verify order is partially filled
      expect(order.status).toBe(OrderStatus.PARTIALLY_FILLED);
      expect(order.filledQuantity).toBeLessThan(3);
      
      // Check portfolio update
      const portfolio = await paperApi.getPortfolio();
      expect(portfolio.cash).toBe(0); // All cash used
      expect(portfolio.positions['BTC/USDT']).toBeDefined();
      expect(portfolio.positions['BTC/USDT'].quantity).toBe(2); // Only 2 BTC could be bought with $100,000
    });
  });
  
  describe('cancelOrder', () => {
    it('should cancel an open order', async () => {
      // Setup
      mockBaseApi.getMarketPrice.mockImplementation(() => Promise.resolve(51000)); // Above limit price
      
      const paperApi = createPaperTradingApi(mockBaseApi);
      
      // Create a limit buy order that won't fill immediately
      const orderRequest = {
        symbol: 'BTC/USDT',
        side: 'buy' as const,
        order_type: 'limit',
        type: OrderType.LIMIT, // Add the required 'type' property
        quantity: 0.2,
        price: 50000,
      };
      
      // Execute order
      const order = await paperApi.createOrder(orderRequest);
      
      // Cancel the order
      const result = await paperApi.cancelOrder(order.id);
      
      // Verify cancellation
      expect(result).toBe(true);
      
      // Check order status
      const cancelledOrder = await paperApi.getOrder(order.id);
      expect(cancelledOrder?.status).toBe(OrderStatus.CANCELED);
      
      // Check that portfolio remains unchanged
      const portfolio = await paperApi.getPortfolio();
      expect(portfolio.cash).toBe(100000);
      expect(Object.keys(portfolio.positions).length).toBe(0);
    });
    
    it('should not cancel filled orders', async () => {
      // Setup
      mockBaseApi.getMarketPrice.mockImplementation(() => Promise.resolve(50000));
      
      const paperApi = createPaperTradingApi(mockBaseApi);
      
      // Create a market order that fills immediately
      const orderRequest = {
        symbol: 'BTC/USDT',
        side: 'buy' as const,
        order_type: 'market',
        type: OrderType.MARKET, // Add the required 'type' property
        quantity: 0.2,
      };
      
      // Execute order
      const order = await paperApi.createOrder(orderRequest);
      
      // Try to cancel the order
      const result = await paperApi.cancelOrder(order.id);
      
      // Verify cancellation failed
      expect(result).toBe(false);
      
      // Check order status remains filled
      const filledOrder = await paperApi.getOrder(order.id);
      expect(filledOrder?.status).toBe(OrderStatus.FILLED);
    });
  });
  
  describe('getMarketPrice', () => {
    it('should delegate to the base API', async () => {
      mockBaseApi.getMarketPrice.mockImplementation(() => Promise.resolve(50000));
      
      const paperApi = createPaperTradingApi(mockBaseApi);
      const price = await paperApi.getMarketPrice('BTC/USDT');
      
      expect(mockBaseApi.getMarketPrice).toHaveBeenCalledWith('BTC/USDT');
      expect(price).toBe(50000);
    });
  });
  
  describe('getExchangeInfo', () => {
    it('should delegate to the base API and add paper trading indicator', async () => {
      mockBaseApi.getExchangeInfo.mockImplementation(() => Promise.resolve({
        name: 'Binance',
        symbols: ['BTC/USDT', 'ETH/USDT'],
        tradingFees: 0.001,
      }));
      
      const paperApi = createPaperTradingApi(mockBaseApi);
      const info = await paperApi.getExchangeInfo();
      
      expect(mockBaseApi.getExchangeInfo).toHaveBeenCalled();
      expect(info.name).toBe('Binance (Paper)');
      expect(info.symbols).toBe(['BTC/USDT', 'ETH/USDT']);
    });
  });
});
