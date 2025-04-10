import { Order, OrderStatus, OrderType, OrderSide } from '../../types';

// Generate a random order ID
const generateOrderId = (): string => {
  return Math.random().toString(36).substring(2, 15);
};

// Generate a timestamp within the last 24 hours
const generateTimestamp = (): Date => {
  const now = new Date();
  const hoursAgo = Math.floor(Math.random() * 24);
  const minutesAgo = Math.floor(Math.random() * 60);
  const secondsAgo = Math.floor(Math.random() * 60);
  
  now.setHours(now.getHours() - hoursAgo);
  now.setMinutes(now.getMinutes() - minutesAgo);
  now.setSeconds(now.getSeconds() - secondsAgo);
  
  return now;
};

// Mock active orders data
export const mockActiveOrders: Order[] = [
  {
    id: generateOrderId(),
    symbol: 'BTC',
    type: OrderType.LIMIT,
    side: OrderSide.BUY,
    quantity: 0.5,
    price: 58750.25,
    status: OrderStatus.NEW,
    createdAt: generateTimestamp(),
    updatedAt: generateTimestamp(),
    clientOrderId: 'client-order-1',
    timeInForce: 'GTC'
  },
  {
    id: generateOrderId(),
    symbol: 'ETH',
    type: OrderType.MARKET,
    side: OrderSide.SELL,
    quantity: 2.5,
    status: OrderStatus.PARTIALLY_FILLED,
    filledQuantity: 1.2,
    createdAt: generateTimestamp(),
    updatedAt: generateTimestamp(),
    clientOrderId: 'client-order-2',
    timeInForce: 'GTC'
  },
  {
    id: generateOrderId(),
    symbol: 'AAPL',
    type: OrderType.STOP,
    side: OrderSide.SELL,
    quantity: 10,
    stopPrice: 175.50,
    status: OrderStatus.NEW,
    createdAt: generateTimestamp(),
    updatedAt: generateTimestamp(),
    clientOrderId: 'client-order-3',
    timeInForce: 'GTC'
  },
  {
    id: generateOrderId(),
    symbol: 'MSFT',
    type: OrderType.STOP_LIMIT,
    side: OrderSide.BUY,
    quantity: 5,
    stopPrice: 380.00,
    limitPrice: 382.50,
    status: OrderStatus.NEW,
    createdAt: generateTimestamp(),
    updatedAt: generateTimestamp(),
    clientOrderId: 'client-order-4',
    timeInForce: 'GTC'
  }
];

// Function to get active orders for a specific symbol
export const getMockActiveOrders = (symbol?: string): Order[] => {
  if (!symbol) {
    return mockActiveOrders;
  }
  
  return mockActiveOrders.filter(order => order.symbol === symbol);
};

// Function to create a new mock order
export const createMockOrder = (orderData: Partial<Order>): Order => {
  const newOrder: Order = {
    id: generateOrderId(),
    symbol: orderData.symbol || 'BTC',
    type: orderData.type || OrderType.MARKET,
    side: orderData.side || OrderSide.BUY,
    quantity: orderData.quantity || 1,
    price: orderData.price,
    stopPrice: orderData.stopPrice,
    limitPrice: orderData.limitPrice,
    status: OrderStatus.NEW,
    createdAt: new Date(),
    updatedAt: new Date(),
    clientOrderId: `client-order-${Math.floor(Math.random() * 1000)}`,
    timeInForce: orderData.timeInForce || 'GTC'
  };
  
  mockActiveOrders.push(newOrder);
  return newOrder;
};

// Function to cancel a mock order
export const cancelMockOrder = (orderId: string): boolean => {
  const orderIndex = mockActiveOrders.findIndex(order => order.id === orderId);
  
  if (orderIndex !== -1) {
    mockActiveOrders[orderIndex].status = OrderStatus.CANCELED;
    mockActiveOrders[orderIndex].updatedAt = new Date();
    return true;
  }
  
  return false;
};
