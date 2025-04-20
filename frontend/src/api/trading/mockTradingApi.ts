import { TradingApi } from './index';
import { OrderRequest, Order, Portfolio, Position, OrderType, OrderSide, OrderStatus } from '../../types';
import { getPortfolioData, getPositionsData, getOrdersData, getRandomPriceMovement } from '../mockData/mockDataProvider';

// Mock data for market prices
const mockMarketPrices: Record<string, number> = {
  'BTC/USD': 48000,
  'ETH/USD': 3500,
  'SOL/USD': 110,
  'ADA/USD': 0.55,
  'XRP/USD': 0.75,
  'DOT/USD': 15.20,
  'DOGE/USD': 0.12,
  'AVAX/USD': 28.50,
  'MATIC/USD': 1.20,
  'LINK/USD': 18.75,
};

// Mock order book data
const generateMockOrderBook = (symbol: string, limit: number = 10) => {
  const basePrice = mockMarketPrices[symbol] || 100;
  const bids = Array.from({ length: limit }, (_, i) => ({
    price: basePrice * (1 - (i + 1) * 0.001),
    size: Math.random() * 10,
  }));
  
  const asks = Array.from({ length: limit }, (_, i) => ({
    price: basePrice * (1 + (i + 1) * 0.001),
    size: Math.random() * 10,
  }));
  
  return { bids, asks };
};

// Mock trading API implementation
export const mockTradingApi: TradingApi = {
  // Account and portfolio methods
  async getPortfolio(): Promise<Portfolio> {
    // Use synchronous data provider to avoid network errors
    return getPortfolioData();
  },
  
  async getPositions(): Promise<Record<string, Position>> {
    // Use synchronous data provider to avoid network errors
    return getPositionsData();
  },
  
  async getBalance(asset?: string): Promise<number> {
    const portfolio = getPortfolioData();
    if (asset) {
      const position = portfolio.positions[asset];
      return position ? position.market_value : 0;
    }
    return portfolio.cash;
  },
  
  // Order management methods
  async createOrder(orderRequest: OrderRequest): Promise<Order> {
    // Generate a mock order from the request
    const now = new Date();
    const mockOrder: Order = {
      id: `mock-order-${Date.now()}`,
      symbol: orderRequest.symbol,
      type: orderRequest.order_type === 'market' ? OrderType.MARKET : OrderType.LIMIT,
      side: orderRequest.side === 'buy' ? OrderSide.BUY : OrderSide.SELL,
      quantity: orderRequest.quantity,
      price: orderRequest.price || mockMarketPrices[orderRequest.symbol] || 0,
      status: OrderStatus.NEW,
      createdAt: now,
      updatedAt: now,
      clientOrderId: `client-${Date.now()}`,
      timeInForce: 'GTC',
    };
    
    // Simulate a delay for API call
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return mockOrder;
  },
  
  async cancelOrder(orderId: string): Promise<boolean> {
    // Simulate a delay for API call
    await new Promise(resolve => setTimeout(resolve, 300));
    
    // Always succeed in mock mode
    return true;
  },
  
  async getOrders(status?: string): Promise<Order[]> {
    const orders = getOrdersData();
    
    if (status) {
      return orders.filter(order => order.status === status);
    }
    
    return orders;
  },
  
  async getOrder(orderId: string): Promise<Order | null> {
    const orders = getOrdersData();
    const order = orders.find(order => order.id === orderId);
    
    return order || null;
  },
  
  // Market data methods
  async getMarketPrice(symbol: string): Promise<number> {
    // Add some random price movement using the helper function
    const basePrice = mockMarketPrices[symbol] || 100;
    return getRandomPriceMovement(basePrice, 0.002);
  },
  
  async getOrderBook(symbol: string, limit: number = 10): Promise<{ bids: any[], asks: any[] }> {
    return generateMockOrderBook(symbol, limit);
  },
  
  async getTicker(symbol: string): Promise<{ price: number, volume: number, change: number }> {
    const basePrice = mockMarketPrices[symbol] || 100;
    const randomChange = (Math.random() * 4 - 2) / 100; // ±2% random change
    
    return {
      price: basePrice,
      volume: Math.random() * 1000,
      change: randomChange,
    };
  },
  
  // Exchange info methods
  async getExchangeInfo(): Promise<any> {
    return {
      name: 'Mock Exchange',
      symbols: Object.keys(mockMarketPrices),
      tradingFees: 0.001,
      withdrawalFees: {},
      serverTime: Date.now(),
    };
  },
  
  async getSymbols(): Promise<string[]> {
    return Object.keys(mockMarketPrices);
  },
  
  async getAssetInfo(symbol: string): Promise<any> {
    return {
      symbol,
      baseAsset: symbol.split('/')[0],
      quoteAsset: symbol.split('/')[1],
      minQuantity: 0.0001,
      maxQuantity: 1000,
      quantityPrecision: 4,
      pricePrecision: 2,
      minNotional: 10,
    };
  },
};
