/**
 * Mock Data Provider
 * 
 * This file provides reliable mock data for development without requiring API calls
 * that might fail and cause network errors.
 */
import { Portfolio, Position, Order, OrderType, OrderSide, OrderStatus } from '../../types';

// Mock portfolio data
const mockPortfolio: Portfolio = {
  total_value: 100000,
  cash: 25000,
  daily_pnl: 350,
  margin_multiplier: 2,
  positions: {
    'BTC/USD': {
      symbol: 'BTC/USD',
      quantity: 0.5,
      market_value: 24000,
      entry_price: 45000,
      current_price: 48000,
      unrealized_pnl: 1500,
      realized_pnl: 200
    },
    'ETH/USD': {
      symbol: 'ETH/USD',
      quantity: 3,
      market_value: 10500,
      entry_price: 3200,
      current_price: 3500,
      unrealized_pnl: 900,
      realized_pnl: 150
    },
    'SOL/USD': {
      symbol: 'SOL/USD',
      quantity: 20,
      market_value: 2200,
      entry_price: 100,
      current_price: 110,
      unrealized_pnl: 200,
      realized_pnl: 50
    },
  },
};

// Mock orders data
const mockOrders: Order[] = [
  {
    id: 'order-1',
    symbol: 'BTC/USD',
    type: OrderType.MARKET,
    side: OrderSide.BUY,
    quantity: 0.1,
    price: 47500,
    status: OrderStatus.FILLED,
    created_at: new Date(Date.now() - 86400000).toISOString(), // Added required field
    createdAt: new Date(Date.now() - 86400000), // 1 day ago
    updatedAt: new Date(Date.now() - 86390000),
    clientOrderId: 'client-1',
    timeInForce: 'GTC',
    filledQuantity: 0.1,
  },
  {
    id: 'order-2',
    symbol: 'ETH/USD',
    type: OrderType.LIMIT,
    side: OrderSide.BUY,
    quantity: 1,
    price: 3300,
    status: OrderStatus.NEW,
    created_at: new Date(Date.now() - 3600000).toISOString(), // Added required field
    createdAt: new Date(Date.now() - 3600000), // 1 hour ago
    updatedAt: new Date(Date.now() - 3600000),
    clientOrderId: 'client-2',
    timeInForce: 'GTC',
    filledQuantity: 0,
  },
];

// Reliable mock data providers that don't rely on async calls that could fail
export const getPortfolioData = (): Portfolio => {
  return { ...mockPortfolio };
};

export const getPositionsData = (): Record<string, Position> => {
  return { ...mockPortfolio.positions };
};

export const getOrdersData = (): Order[] => {
  return [...mockOrders];
};

// Helper to get a random price movement
export const getRandomPriceMovement = (basePrice: number, percentRange = 0.005): number => {
  const randomFactor = 1 + (Math.random() * percentRange * 2 - percentRange);
  return basePrice * randomFactor;
};
