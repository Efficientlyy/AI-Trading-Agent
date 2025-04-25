/**
 * Trading Service
 * 
 * This service provides a simplified interface for trading operations
 * by abstracting the direct API calls and providing business logic.
 */
import { OrderType, OrderSide } from '../types';
import { binanceTradingApi } from '../api/trading/binanceTradingApi';
import { bybitTradingApi } from '../api/trading/bybitTradingApi';
// Create a dummy implementation for paper trading api
const paperTradingApi = (mode: 'live' | 'paper', config: any) => bybitTradingApi(mode, config);

// Import configuration from config file
import config from '../config';

// Define Config interface to match the expected structure
interface Config {
  trading?: {
    defaultExchange: string;
    mode: string;
    exchanges: {
      binance: {
        apiKey: string;
        apiSecret: string;
      };
      bybit: {
        apiKey: string;
        apiSecret: string;
      };
    };
  };
}

// Create a configuration object with typed config
const configuration = {
  trading: (config as Config).trading || {
    defaultExchange: 'binance',
    mode: 'paper',
    exchanges: {
      binance: {
        apiKey: process.env.REACT_APP_BINANCE_API_KEY || '',
        apiSecret: process.env.REACT_APP_BINANCE_API_SECRET || '',
      },
      bybit: {
        apiKey: process.env.REACT_APP_BYBIT_API_KEY || '',
        apiSecret: process.env.REACT_APP_BYBIT_API_SECRET || '',
      }
    }
  }
};

// Define executeWithCircuitBreaker with proper types
const executeWithCircuitBreaker = async <T>(
  serviceName: string,
  methodName: string,
  fn: () => Promise<T>
): Promise<T> => {
  try {
    return await fn();
  } catch (error) {
    console.error(`Circuit breaker caught error in ${serviceName}.${methodName}:`, error);
    throw error;
  }
};

// Get trading config from the configuration
const tradingConfig = configuration.trading;

// Initialize API clients for each exchange with proper types
const apiClients: Record<string, any> = {
  binance: binanceTradingApi(tradingConfig.mode as 'live' | 'paper', tradingConfig.exchanges.binance),
  bybit: bybitTradingApi(tradingConfig.mode as 'live' | 'paper', tradingConfig.exchanges.bybit),
  paper: paperTradingApi(tradingConfig.mode as 'live' | 'paper', tradingConfig.exchanges.binance),
};

// Get the currently active API client
const getActiveClient = () => {
  // If we're in paper trading mode, always use paper client
  if (tradingConfig.mode === 'paper') {
    return apiClients.paper;
  }
  
  // Otherwise use the configured exchange
  const exchange = tradingConfig.defaultExchange.toLowerCase();
  return apiClients[exchange as string] || apiClients.binance;
};

/**
 * Trading Service Interface
 */
export const tradingService = {
  /**
   * Place a trading order
   * 
   * @param orderDetails Order details including symbol, type, side, quantity, and price (for limit orders)
   * @returns Promise with the created order
   */
  placeOrder: async (orderDetails: {
    symbol: string;
    type: string;
    side: string;
    quantity: number;
    price?: number;
  }) => {
    return executeWithCircuitBreaker('TradingService', 'placeOrder', async () => {
      const api = getActiveClient();
      
      // Convert order type 
      const orderRequest = {
        symbol: orderDetails.symbol,
        order_type: orderDetails.type,
        side: orderDetails.side as 'buy' | 'sell',
        quantity: orderDetails.quantity,
        price: orderDetails.price,
        // Add required type property for TypeScript
        type: orderDetails.type === 'market' ? OrderType.MARKET : OrderType.LIMIT
      };
      
      // Place the order
      const order = await api.createOrder(orderRequest);
      
      // Return simplified order data
      return {
        id: order.id,
        symbol: order.symbol,
        type: order.type,
        side: order.side,
        quantity: order.quantity,
        price: order.price,
        status: order.status,
        timestamp: order.created_at,
      };
    });
  },
  
  /**
   * Cancel an existing order
   * 
   * @param orderId Order ID to cancel
   * @returns Promise with success status
   */
  cancelOrder: async (orderId: string) => {
    return executeWithCircuitBreaker('TradingService', 'cancelOrder', async () => {
      const api = getActiveClient();
      return api.cancelOrder(orderId);
    });
  },
  
  /**
   * Get available tradable assets
   * 
   * @returns Promise with list of available assets
   */
  getAvailableAssets: async () => {
    return executeWithCircuitBreaker('TradingService', 'getAvailableAssets', async () => {
      const api = getActiveClient();
      
      // Get exchange info to get available symbols
      const symbols = await api.getSymbols();
      
      // Convert to asset objects with additional data
      const assets = await Promise.all(symbols.slice(0, 20).map(async (symbol: string) => {
        try {
          // Get current price for the symbol
          const price = await api.getMarketPrice(symbol);
          
          // Extract asset name from symbol
          const name = symbol.split('/')[0];
          
          return {
            symbol,
            name,
            price,
          };
        } catch (error) {
          console.error(`Error getting data for ${symbol}:`, error);
          return {
            symbol,
            name: symbol.split('/')[0],
            price: 0,
          };
        }
      }));
      
      return assets;
    });
  },
  
  /**
   * Get market data for a symbol
   * 
   * @param symbol Trading pair symbol (e.g., "BTC/USDT")
   * @returns Promise with market data
   */
  getMarketData: async (symbol: string) => {
    return executeWithCircuitBreaker('TradingService', 'getMarketData', async () => {
      const api = getActiveClient();
      
      // Get ticker data
      const ticker = await api.getTicker(symbol);
      
      // Get 24h high/low
      const price = ticker.price;
      const change = ticker.change;
      
      // Calculate high/low based on change %
      const high = price * (1 + Math.abs(change) / 100);
      const low = price * (1 - Math.abs(change) / 100);
      
      return {
        price,
        change,
        high,
        low,
        volume: ticker.volume,
      };
    });
  },
  
  /**
   * Get order history for the authenticated user
   * 
   * @param symbol Optional symbol to filter orders
   * @param limit Maximum number of orders to return
   * @returns Promise with order history
   */
  getOrderHistory: async (symbol?: string, limit: number = 50) => {
    return executeWithCircuitBreaker('TradingService', 'getOrderHistory', async () => {
      const api = getActiveClient();
      
      // Get all orders
      let orders = await api.getOrders();
      
      // Filter by symbol if provided
      if (symbol) {
        orders = orders.filter((order: any) => order.symbol === symbol);
      }
      
      // Sort by timestamp descending and limit
      orders = orders
        .sort((a: any, b: any) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        .slice(0, limit);
      
      return orders.map((order: any) => ({
        id: order.id,
        symbol: order.symbol,
        type: order.type,
        side: order.side,
        quantity: order.quantity,
        price: order.price || 0,
        status: order.status,
        timestamp: order.created_at,
        filled: order.filledQuantity || 0,
      }));
    });
  },
  
  /**
   * Get portfolio summary
   * 
   * @returns Promise with portfolio data
   */
  getPortfolio: async () => {
    return executeWithCircuitBreaker('TradingService', 'getPortfolio', async () => {
      const api = getActiveClient();
      return api.getPortfolio();
    });
  },
  
  /**
   * Execute a test/demo trade (for simulation)
   * 
   * @param symbol Symbol to trade
   * @returns Promise with simulated trade result
   */
  executeDemoTrade: async (symbol: string) => {
    // Generate a random buy/sell decision
    const side = Math.random() > 0.5 ? 'buy' : 'sell';
    
    // Get current price
    const api = getActiveClient();
    const price = await api.getMarketPrice(symbol);
    
    // Generate a random quantity
    const quantity = parseFloat((Math.random() * 0.1).toFixed(6));
    
    // Return simulated trade
    return {
      symbol,
      side,
      price,
      quantity,
      timestamp: new Date().toISOString(),
      id: `demo-${Date.now()}`,
      success: true,
    };
  },
};