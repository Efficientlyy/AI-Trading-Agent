import { createAuthenticatedClient } from './client';
import { Portfolio, Order, OrderRequest } from '../types';

export const portfolioApi = {
  getPortfolio: async (): Promise<{ portfolio: Portfolio }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ portfolio: Portfolio }>('/portfolio');
    return response.data;
  },
  
  getOrders: async (): Promise<{ orders: Order[] }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ orders: Order[] }>('/orders');
    return response.data;
  },
  
  createOrder: async (orderRequest: OrderRequest): Promise<{ order: Order }> => {
    const client = createAuthenticatedClient();
    const response = await client.post<{ order: Order }>('/orders', orderRequest);
    return response.data;
  },
  
  cancelOrder: async (orderId: string): Promise<{ success: boolean, message: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.delete<{ success: boolean, message: string }>(`/orders/${orderId}`);
    return response.data;
  },
};
