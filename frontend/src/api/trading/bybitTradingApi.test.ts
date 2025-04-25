import { bybitTradingApi } from './bybitTradingApi';
import axios from 'axios';
import { canMakeApiCall, recordApiCall, recordCircuitBreakerResult } from '../utils/monitoring';
import { OrderSide, OrderType } from '../../types';

// Mock axios
jest.mock('axios', () => {
  return {
    create: jest.fn(() => ({
      get: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      post: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
    }))
  };
});
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock monitoring utilities
jest.mock('../utils/monitoring', () => {
  return {
    recordApiCall: jest.fn(),
    canMakeApiCall: jest.fn().mockReturnValue(true),
    recordCircuitBreakerResult: jest.fn(),
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

describe('Bybit Trading API', () => {
  const mockConfig = {
    apiKey: 'test-api-key',
    apiSecret: 'test-api-secret',
  };
  
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Reset canMakeApiCall mock to allow calls by default
    (canMakeApiCall as jest.Mock).mockReturnValue(true);
  });
  
  describe('createOrder', () => {
    it('should create a market order correctly', async () => {
      // Mock response
      const mockOrderResponse = {
        ret_code: 0,
        ret_msg: 'OK',
        result: {
          order_id: '12345',
          symbol: 'BTCUSDT',
          side: 'Buy',
          order_type: 'Market',
          price: 0,
          qty: 0.1,
          time_in_force: 'GoodTillCancel',
          create_time: '2023-01-01T12:00:00Z',
          update_time: '2023-01-01T12:00:00Z',
          order_status: 'Filled',
        }
      };
      
      // Setup mock
      const mockClient = {
        post: jest.fn().mockImplementation(() => Promise.resolve({ data: mockOrderResponse })),
        get: jest.fn().mockImplementation(() => Promise.resolve({ data: {} })),
      };
      (mockedAxios.create as jest.Mock).mockReturnValue(mockClient);
      
      // Create API and call method
      const api = bybitTradingApi('paper', mockConfig);
      const orderRequest = {
        symbol: 'BTC/USDT',
        side: 'buy' as const,
        order_type: 'market',
        type: OrderType.MARKET, // Add the required 'type' property
        quantity: 0.1
      };
      
      const order = await api.createOrder(orderRequest);
      
      // Assertions
      expect(mockClient.post).toHaveBeenCalled();
      
      // Check order structure
      expect(order).toHaveProperty('id');
      expect(order).toHaveProperty('symbol');
      expect(order).toHaveProperty('type');
      expect(order).toHaveProperty('side');
      expect(order).toHaveProperty('quantity');
      expect(order).toHaveProperty('status');
      
      // Check circuit breaker and monitoring
      expect(canMakeApiCall).toHaveBeenCalledWith('Bybit', 'createOrder');
      expect(recordApiCall).toHaveBeenCalledWith('Bybit', 'createOrder', 'attempt');
      expect(recordApiCall).toHaveBeenCalledWith('Bybit', 'createOrder', 'success');
      expect(recordCircuitBreakerResult).toHaveBeenCalledWith('Bybit', 'createOrder', true);
    });
  });
});