import { createAuthenticatedClient } from './client';
import { Order } from '../types';

export const ordersApi = {
  getActiveOrders: async (): Promise<{ orders: Order[] }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ orders: Order[] }>('/orders/active');
    return response.data;
  },
  createOrder: async (orderRequest: Partial<Order>): Promise<{ order: Order }> => {
    const client = createAuthenticatedClient();
    const response = await client.post<{ order: Order }>('/orders', orderRequest);
    return response.data;
  },
  cancelOrder: async (orderId: string): Promise<{ success: boolean }> => {
    const client = createAuthenticatedClient();
    const response = await client.delete<{ success: boolean }>(`/orders/${orderId}`);
    return response.data;
  },
};
