import { Order, OrderStatus, OrderType, OrderSide } from '../../types';

// Helper function to generate mock data
function generateOrderId(): string {
  return `order-${Math.floor(Math.random() * 10000)}`;
}

function generateTimestamp(): Date {
  const now = Date.now();
  const randomOffset = Math.floor(Math.random() * 1000 * 60 * 60 * 24); // Random time in the last 24 hours
  return new Date(now - randomOffset);
}

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
    created_at: generateTimestamp().toISOString(), // Added required field
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
    created_at: generateTimestamp().toISOString(), // Added required field
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
    created_at: generateTimestamp().toISOString(), // Added required field
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
    created_at: generateTimestamp().toISOString(), // Added required field
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
    created_at: new Date().toISOString(), // Added required field
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
