import { Order } from '../types';

export const ordersApi = {
  getActiveOrders: async (): Promise<{ orders: Order[] }> => {
    // Always use mock data for active orders
    const { getMockActiveOrders } = await import('./mockData/mockOrders');
    return getMockActiveOrders();
  },
  createOrder: async (orderRequest: Partial<Order>): Promise<{ order: Order }> => {
    // Always use mock data for creating orders
    const { createMockOrder } = await import('./mockData/mockOrders');
    return createMockOrder(orderRequest);
  },
  cancelOrder: async (orderId: string): Promise<{ success: boolean }> => {
    // Always use mock data for canceling orders
    const { cancelMockOrder } = await import('./mockData/mockOrders');
    await cancelMockOrder(orderId);
    return { success: true };
  },
};
