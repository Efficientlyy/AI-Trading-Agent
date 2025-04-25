// mockOrders.ts
import { Order, OrderStatus, OrderType, OrderSide } from '../../types';

let mockOrders: Order[] = [
  {
    id: 'order-1',
    symbol: 'AAPL',
    type: OrderType.MARKET,
    side: OrderSide.BUY,
    quantity: 10,
    price: 180.5,
    status: OrderStatus.NEW,
    created_at: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
    createdAt: new Date(Date.now() - 1000 * 60 * 5),
    updatedAt: new Date(Date.now() - 1000 * 60 * 4),
    clientOrderId: 'mock-client-1',
    timeInForce: 'GTC',
  },
  {
    id: 'order-2',
    symbol: 'MSFT',
    type: OrderType.LIMIT,
    side: OrderSide.SELL,
    quantity: 5,
    price: 320,
    status: OrderStatus.PARTIALLY_FILLED,
    created_at: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
    createdAt: new Date(Date.now() - 1000 * 60 * 10),
    updatedAt: new Date(Date.now() - 1000 * 60 * 9),
    clientOrderId: 'mock-client-2',
    timeInForce: 'GTC',
  },
  {
    id: 'order-3',
    symbol: 'TSLA',
    type: OrderType.STOP,
    side: OrderSide.BUY,
    quantity: 3,
    price: 700,
    status: OrderStatus.FILLED,
    created_at: new Date(Date.now() - 1000 * 60 * 20).toISOString(),
    createdAt: new Date(Date.now() - 1000 * 60 * 20),
    updatedAt: new Date(Date.now() - 1000 * 60 * 18),
    clientOrderId: 'mock-client-3',
    timeInForce: 'GTC',
  },
];

export const getMockActiveOrders = async (): Promise<{ orders: Order[] }> => {
  return { orders: mockOrders };
};

export const getMockOrders = async (): Promise<{ orders: Order[] }> => {
  return { orders: mockOrders };
};

export const createMockOrder = async (order: Partial<Order>): Promise<{ order: Order }> => {
  const newOrder: Order = {
    ...order,
    id: `order-${mockOrders.length + 1}`,
    status: OrderStatus.NEW,
    createdAt: new Date(),
  } as Order;
  mockOrders = [newOrder, ...mockOrders];
  return { order: newOrder };
};

export const cancelMockOrder = async (orderId: string): Promise<{ success: boolean }> => {
  mockOrders = mockOrders.map(order =>
    order.id === orderId ? { ...order, status: OrderStatus.CANCELED } : order
  );
  return { success: true };
};
