import React from 'react';

interface OrderBookLevel {
  price: number;
  size: number;
}

interface OrderBookData {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
}

interface OrderBookProps {
  symbol: string;
  orderBook: OrderBookData | null;
  isLoading?: boolean;
  error?: string | null;
}

const OrderBook: React.FC<OrderBookProps> = ({ symbol, orderBook, isLoading, error }) => {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Order Book ({symbol})</h2>
      {isLoading ? (
        <div className="text-gray-500 dark:text-gray-400">Loading order book...</div>
      ) : error ? (
        <div className="text-red-500">{error}</div>
      ) : orderBook ? (
        <div className="flex gap-4">
          <div className="w-1/2">
            <div className="font-medium text-green-700 dark:text-green-300 mb-1">Bids</div>
            <table className="w-full text-xs">
              <thead>
                <tr><th className="text-left">Price</th><th className="text-left">Size</th></tr>
              </thead>
              <tbody>
                {orderBook.bids.map((level, i) => (
                  <tr key={i}>
                    <td className="text-green-600">${level.price.toFixed(2)}</td>
                    <td>{level.size}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="w-1/2">
            <div className="font-medium text-red-700 dark:text-red-300 mb-1">Asks</div>
            <table className="w-full text-xs">
              <thead>
                <tr><th className="text-left">Price</th><th className="text-left">Size</th></tr>
              </thead>
              <tbody>
                {orderBook.asks.map((level, i) => (
                  <tr key={i}>
                    <td className="text-red-600">${level.price.toFixed(2)}</td>
                    <td>{level.size}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="text-gray-500 dark:text-gray-400">No order book data.</div>
      )}
    </div>
  );
};

export default OrderBook;
