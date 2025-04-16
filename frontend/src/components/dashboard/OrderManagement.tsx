import React, { useEffect, useState } from 'react';
import { Order, OrderStatus, OrderType, OrderSide } from '../../types';
import { useDataSource } from '../../context/DataSourceContext';
import { getMockActiveOrders, createMockOrder, cancelMockOrder } from '../../api/mockData/mockOrders';
import { ordersApi } from '../../api/orders';

interface OrderManagementProps {
  symbol: string;
  currentPrice?: number;
}

const OrderManagement: React.FC<OrderManagementProps> = ({
  symbol,
  currentPrice = 0
}) => {
  const { dataSource } = useDataSource();
  const [activeOrders, setActiveOrders] = useState<Order[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  

  // State for new order form
  const [newOrder, setNewOrder] = useState<Partial<Order>>({
    symbol,
    type: OrderType.MARKET,
    side: OrderSide.BUY,
    quantity: 1,
    price: currentPrice,
    stopPrice: currentPrice * 0.95,
    limitPrice: currentPrice * 1.05,
    timeInForce: 'GTC'
  });

  // Fetch active orders
  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);

    const fetchOrders = async () => {
      try {
        const data = dataSource === 'mock'
          ? await getMockActiveOrders()
          : await ordersApi.getActiveOrders();
        if (isMounted) setActiveOrders(data.orders);
      } catch (e: any) {
        if (isMounted) console.log('Failed to fetch orders');
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };
    fetchOrders();
    return () => { isMounted = false; };
  }, [dataSource]);

  // Update new order when symbol or price changes
  React.useEffect(() => {
    if (symbol || currentPrice) {
      setNewOrder(prev => ({
        ...prev,
        symbol,
        price: currentPrice,
        stopPrice: currentPrice * 0.95,
        limitPrice: currentPrice * 1.05
      }));
    }
  }, [symbol, currentPrice]);

  // Handle input changes for new order
  const handleInputChange = (field: keyof Partial<Order>, value: any) => {
    setNewOrder(prev => ({
      ...prev,
      [field]: value
    }));
  };




  // Handle order submission
  const handleSubmitOrder = async () => {
    setIsLoading(true);
    try {
      const data = dataSource === 'mock'
        ? await createMockOrder(newOrder)
        : await ordersApi.createOrder(newOrder);
      setActiveOrders(prev => [data.order, ...prev]);
      setNewOrder(prev => ({ ...prev, quantity: 1 }));
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

  // Determine if price inputs should be shown based on order type
  const showPriceInput = newOrder.type === OrderType.LIMIT || newOrder.type === OrderType.STOP_LIMIT;
  const showStopPriceInput = newOrder.type === OrderType.STOP || newOrder.type === OrderType.STOP_LIMIT;
  const showLimitPriceInput = newOrder.type === OrderType.STOP_LIMIT;

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-4">Order Management</h2>
      
      {/* New Order Form */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Create New Order</h3>
        
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div>
            <label htmlFor="order-symbol" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
              Symbol
            </label>
            <input
              id="order-symbol"
              type="text"
              value={symbol}
              disabled
              className="block w-full px-3 py-2 bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            />
          </div>
          
          <div>
            <label htmlFor="order-type" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
              Order Type
            </label>
            <select
              id="order-type"
              value={newOrder.type}
              onChange={(e) => handleInputChange('type', e.target.value as OrderType)}
              className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            >
              <option value={OrderType.MARKET}>Market</option>
              <option value={OrderType.LIMIT}>Limit</option>
              <option value={OrderType.STOP}>Stop</option>
              <option value={OrderType.STOP_LIMIT}>Stop Limit</option>
            </select>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div>
            <label htmlFor="order-side" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
              Side
            </label>
            <div className="flex">
              <button
                type="button"
                onClick={() => handleInputChange('side', OrderSide.BUY)}
                className={`flex-1 py-2 text-sm font-medium rounded-l-md focus:outline-none ${
                  newOrder.side === OrderSide.BUY
                    ? 'bg-green-600 text-white'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700'
                }`}
              >
                Buy
              </button>
              <button
                type="button"
                onClick={() => handleInputChange('side', OrderSide.SELL)}
                className={`flex-1 py-2 text-sm font-medium rounded-r-md focus:outline-none ${
                  newOrder.side === OrderSide.SELL
                    ? 'bg-red-600 text-white'
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700'
                }`}
              >
                Sell
              </button>
            </div>
          </div>
          
          <div>
            <label htmlFor="order-quantity" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
              Quantity
            </label>
            <input
              id="order-quantity"
              type="number"
              min="0.00000001"
              step="0.00000001"
              value={newOrder.quantity}
              onChange={(e) => handleInputChange('quantity', parseFloat(e.target.value))}
              className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            />
          </div>
        </div>
        
        {showPriceInput && (
          <div className="mb-3">
            <label htmlFor="order-price" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
              Price
            </label>
            <input
              id="order-price"
              type="number"
              min="0.00000001"
              step="0.00000001"
              value={newOrder.price}
              onChange={(e) => handleInputChange('price', parseFloat(e.target.value))}
              className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            />
          </div>
        )}
        
        {showStopPriceInput && (
          <div className="mb-3">
            <label htmlFor="order-stop-price" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
              Stop Price
            </label>
            <input
              id="order-stop-price"
              type="number"
              min="0.00000001"
              step="0.00000001"
              value={newOrder.stopPrice}
              onChange={(e) => handleInputChange('stopPrice', parseFloat(e.target.value))}
              className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            />
          </div>
        )}
        
        {showLimitPriceInput && (
          <div className="mb-3">
            <label htmlFor="order-limit-price" className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
              Limit Price
            </label>
            <input
              id="order-limit-price"
              type="number"
              min="0.00000001"
              step="0.00000001"
              value={newOrder.limitPrice}
              onChange={(e) => handleInputChange('limitPrice', parseFloat(e.target.value))}
              className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
            />
          </div>
        )}
        
        <div className="mt-4">
          <button
            type="button"
            onClick={handleSubmitOrder}
            className={`w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
              newOrder.side === OrderSide.BUY
                ? 'bg-green-600 hover:bg-green-700 focus:ring-green-500'
                : 'bg-red-600 hover:bg-red-700 focus:ring-red-500'
            } focus:outline-none focus:ring-2 focus:ring-offset-2`}
          >
            {newOrder.side === OrderSide.BUY ? 'Buy' : 'Sell'} {symbol}
          </button>
        </div>
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
