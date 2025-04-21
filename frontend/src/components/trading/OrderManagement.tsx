import React, { useEffect, useState, useCallback } from 'react';
import { Order, OrderStatus, OrderType, OrderSide } from '../../types';
import { useDataSource } from '../../context/DataSourceContext';
import { getMockActiveOrders, createMockOrder, cancelMockOrder } from '../../api/mockData/mockOrders';
import { ordersApi } from '../../api/orders';
import OrderEntryForm from './OrderEntryForm';

interface OrderManagementProps {
  symbol: string;
  currentPrice?: number;
  portfolio?: any; // Accept portfolio as a prop (should be Portfolio type)
}

const OrderManagement: React.FC<OrderManagementProps> = ({
  symbol,
  currentPrice = 0,
  portfolio = null
}) => {
  const { dataSource } = useDataSource();
  const [activeOrders, setActiveOrders] = useState<Order[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  

  // Remove local newOrder state and use OrderEntryForm instead

  // Fetch active orders
  const fetchOrders = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = dataSource === 'mock'
        ? await getMockActiveOrders()
        : await ordersApi.getActiveOrders();
      setActiveOrders(data.orders);
    } catch (e: any) {
      console.log('Failed to fetch orders');
    } finally {
      setIsLoading(false);
    }
  }, [dataSource]);

  useEffect(() => {
    fetchOrders();
  }, [fetchOrders]);

  // Removed legacy setNewOrder effect (handled by OrderEntryForm)

  // Removed legacy handleInputChange (no longer needed with OrderEntryForm)




  // Handle order submission from OrderEntryForm
  const handleSubmitOrder = async (order: any) => {
    setIsLoading(true);
    try {
      const data = dataSource === 'mock'
        ? await createMockOrder(order)
        : await ordersApi.createOrder(order);
      setActiveOrders(prev => [data.order, ...prev]);
      fetchOrders(); // Refresh order list
    } catch (e: any) {
      console.log('Failed to create order');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle order cancellation
  const handleCancelOrder = async (orderId: string) => {
    setIsLoading(true);
    try {
      if (dataSource === 'mock') {
        await cancelMockOrder(orderId);
      } else {
        await ordersApi.cancelOrder(orderId);
      }
      setActiveOrders(prev => prev.map(order =>
        order.id === orderId ? { ...order, status: OrderStatus.CANCELED } : order
      ));
    } catch (e: any) {
      console.log('Failed to cancel order');
    } finally {
      setIsLoading(false);
    }
  };

  // Removed legacy price input logic (handled by OrderEntryForm)

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-4">Order Management</h2>
      
      {/* New Order Form - replaced with OrderEntryForm */}
      <div className="mb-6">
        <OrderEntryForm
          portfolio={null} // TODO: pass real portfolio if available
          availableSymbols={[symbol]}
          selectedSymbol={symbol}
          onSubmitOrder={handleSubmitOrder}
        />
      </div>
      
      {/* Active Orders */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Active Orders</h3>
        
        {isLoading ? (
          <div className="animate-pulse space-y-2">
            <div className="h-10 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
            <div className="h-10 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
          </div>
        ) : activeOrders.length === 0 ? (
          <div className="text-gray-500 dark:text-gray-400 text-center py-8 text-base font-medium">
            No active orders
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Type
                  </th>
                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Side
                  </th>
                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Qty
                  </th>
                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Price
                  </th>
                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Status
                  </th>
                  <th scope="col" className="px-3 py-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                {activeOrders.map((order) => (
                  <tr key={order.id}>
                    <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                      {order.symbol}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {order.type}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        order.side === OrderSide.BUY
                          ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                          : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                      }`}>
                        {order.side}
                      </span>
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {order.quantity}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {order.price ? `$${order.price.toFixed(2)}` : 'Market'}
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-sm">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        order.status === OrderStatus.FILLED
                          ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                          : order.status === OrderStatus.PARTIALLY_FILLED
                            ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                            : order.status === OrderStatus.CANCELED
                              ? 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
                              : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                      }`}>
                        {order.status}
                      </span>
                    </td>
                    <td className="px-3 py-2 whitespace-nowrap text-right text-sm font-medium">
                      {order.status === OrderStatus.NEW || order.status === OrderStatus.PARTIALLY_FILLED ? (
                        <button
                          onClick={() => handleCancelOrder(order.id)}
                          className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                        >
                          Cancel
                        </button>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default OrderManagement;
