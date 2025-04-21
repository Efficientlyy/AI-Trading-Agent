import { Portfolio, Order, OrderRequest, OrderSide, OrderType } from '../types';
import { getMockPortfolio } from './mockData/mockPortfolio';

export const portfolioApi = {
  getPortfolio: async (): Promise<{ portfolio: Portfolio }> => {
    // Always use mock data for portfolio
    return getMockPortfolio();
  },
  
  getOrders: async (): Promise<{ orders: Order[] }> => {
    // Always use mock data for orders
    const { getMockOrders } = await import('./mockData/mockOrders');
    return getMockOrders();
  },
  
  createOrder: async (orderRequest: OrderRequest): Promise<{ order: Order }> => {
    // Always use mock data for creating orders
    const { createMockOrder } = await import('./mockData/mockOrders');
    // Map OrderRequest to Partial<Order> for mock compatibility
    const order: Partial<Order> = {
      symbol: orderRequest.symbol,
      side: orderRequest.side === 'buy' ? OrderSide.BUY : OrderSide.SELL,
      type: orderRequest.order_type
        ? OrderType[orderRequest.order_type.toUpperCase() as keyof typeof OrderType]
        : OrderType.MARKET,
      quantity: orderRequest.quantity,
      price: orderRequest.price,
      stopPrice: orderRequest.stop_price,
      timeInForce: 'GTC',
    };
    return createMockOrder(order);
  },
  
  cancelOrder: async (orderId: string): Promise<{ success: boolean, message: string }> => {
    // Always use mock data for canceling orders
    const { cancelMockOrder } = await import('./mockData/mockOrders');
    await cancelMockOrder(orderId);
    return { success: true, message: 'Order cancelled (mock)' };
  },
};
