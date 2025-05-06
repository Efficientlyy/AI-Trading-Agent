import { TradingApi } from './index';
import { OrderRequest, Order, Position, Portfolio } from '../../types';

interface MarketData {
  price: number;
  volume: number;
  change: number;
}

/**
 * Circuit breaker for trading API
 * This wrapper adds safety mechanisms for paper trading:
 * - Rate limiting
 * - Error handling
 * - Logging
 */

// Extended Trading API interface for circuit breaker
interface CircuitBreakerTradingApi extends TradingApi {
  // Use createOrder instead of placeOrder to match TradingApi interface
  getOpenOrders(): Promise<Order[]>;
  getOrderHistory(): Promise<Order[]>;
  getMarketData(symbol: string): Promise<MarketData>;
}

export const tradingApiCircuitBreaker = (api: CircuitBreakerTradingApi): CircuitBreakerTradingApi => {
  // Track API calls
  const apiCalls: Record<string, { count: number, lastCall: number }> = {};
  
  // Rate limit configuration
  const rateLimits: Record<string, number> = {
    getPortfolio: 1000, // 1 call per second
    placeOrder: 2000,   // 1 call per 2 seconds
    cancelOrder: 2000,   // 1 call per 2 seconds
    getOpenOrders: 1000, // 1 call per second
    getOrderHistory: 1000, // 1 call per second
    getPositions: 1000,  // 1 call per second
    getMarketData: 500,  // 2 calls per second
    getMarketPrice: 500  // 2 calls per second
  };
  
  // Check rate limit
  const checkRateLimit = (method: string): boolean => {
    const now = Date.now();
    
    // Initialize tracking if not exists
    if (!apiCalls[method]) {
      apiCalls[method] = { count: 0, lastCall: 0 };
    }
    
    // Get rate limit
    const limit = rateLimits[method] || 1000;
    
    // Check if we're within rate limit
    if (now - apiCalls[method].lastCall < limit) {
      console.warn(`Rate limit exceeded for ${method}`);
      return false;
    }
    
    // Update tracking
    apiCalls[method].count++;
    apiCalls[method].lastCall = now;
    
    return true;
  };
  
  // Log API call
  const logApiCall = (method: string, ...args: any[]): void => {
    console.log(`[Trading API] ${method}`, ...args);
  };
  
  // Wrap API with circuit breaker
  const circuitBreakerApi: CircuitBreakerTradingApi = {
    // Get account portfolio
    getPortfolio: async (): Promise<Portfolio> => {
      logApiCall('getPortfolio');
      
      // Check rate limit
      if (!checkRateLimit('getPortfolio')) {
        throw new Error('Rate limit exceeded for getPortfolio');
      }
      
      try {
        // Call API
        return await api.getPortfolio();
      } catch (error) {
        console.error('Error in getPortfolio:', error);
        throw error;
      }
    },
    
    // Create order (previously placeOrder)
    createOrder: async (orderRequest: OrderRequest): Promise<Order> => {
      logApiCall('createOrder', orderRequest);
      
      // Check rate limit
      if (!checkRateLimit('createOrder')) {
        throw new Error('Rate limit exceeded for createOrder');
      }
      
      try {
        // Call API
        return await api.createOrder(orderRequest);
      } catch (error) {
        console.error('Error in createOrder:', error);
        throw error;
      }
    },
    
    // Cancel order
    cancelOrder: async (orderId: string): Promise<boolean> => {
      logApiCall('cancelOrder', orderId);
      
      // Check rate limit
      if (!checkRateLimit('cancelOrder')) {
        throw new Error('Rate limit exceeded for cancelOrder');
      }
      
      try {
        // Call API
        return await api.cancelOrder(orderId);
      } catch (error) {
        console.error('Error in cancelOrder:', error);
        throw error;
      }
    },
    
    // Get open orders
    getOpenOrders: async (): Promise<Order[]> => {
      logApiCall('getOpenOrders');
      
      // Check rate limit
      if (!checkRateLimit('getOpenOrders')) {
        throw new Error('Rate limit exceeded for getOpenOrders');
      }
      
      try {
        // Call API
        return await api.getOpenOrders();
      } catch (error) {
        console.error('Error in getOpenOrders:', error);
        throw error;
      }
    },
    
    // Get order history
    getOrderHistory: async (): Promise<Order[]> => {
      logApiCall('getOrderHistory');
      
      // Check rate limit
      if (!checkRateLimit('getOrderHistory')) {
        throw new Error('Rate limit exceeded for getOrderHistory');
      }
      
      try {
        // Call API
        return await api.getOrderHistory();
      } catch (error) {
        console.error('Error in getOrderHistory:', error);
        throw error;
      }
    },
    
    // Get positions
    getPositions: async (): Promise<Record<string, Position>> => {
      logApiCall('getPositions');
      
      // Check rate limit
      if (!checkRateLimit('getPositions')) {
        throw new Error('Rate limit exceeded for getPositions');
      }
      
      try {
        // Call API
        const positions = await api.getPositions();
        // Convert array to record if needed
        if (Array.isArray(positions)) {
          const positionsRecord: Record<string, Position> = {};
          positions.forEach((position: Position) => {
            if (position.symbol) {
              positionsRecord[position.symbol] = position;
            }
          });
          return positionsRecord;
        }
        return positions;
      } catch (error) {
        console.error('Error in getPositions:', error);
        throw error;
      }
    },
    
    // Get market data
    getMarketData: async (symbol: string): Promise<MarketData> => {
      logApiCall('getMarketData', symbol);
      
      // Check rate limit
      if (!checkRateLimit('getMarketData')) {
        throw new Error('Rate limit exceeded for getMarketData');
      }
      
      try {
        // Call API
        return await api.getMarketData(symbol);
      } catch (error) {
        console.error('Error in getMarketData:', error);
        throw error;
      }
    },
    
    // Get market price
    getMarketPrice: async (symbol: string): Promise<number> => {
      logApiCall('getMarketPrice', symbol);
      
      // Check rate limit
      if (!checkRateLimit('getMarketPrice')) {
        throw new Error('Rate limit exceeded for getMarketPrice');
      }
      
      try {
        // Call API
        return await api.getMarketPrice(symbol);
      } catch (error) {
        console.error('Error in getMarketPrice:', error);
        throw error;
      }
    },

    // Forward other methods from the original API
    getOrders: api.getOrders,
    getOrder: api.getOrder,
    getBalance: api.getBalance,
    getExchangeInfo: api.getExchangeInfo,
    getSymbols: api.getSymbols,
    getAssetInfo: api.getAssetInfo,
    getOrderBook: api.getOrderBook,
    getTicker: api.getTicker
  };

  return circuitBreakerApi;
};
