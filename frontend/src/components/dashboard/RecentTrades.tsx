import React from 'react';
import { Trade } from '../../types';

export interface RecentTradesProps {
  trades: Trade[] | null;
  isLoading?: boolean;
  onSymbolSelect?: (symbol: string) => void;
  selectedSymbol?: string;
}

const RecentTrades: React.FC<RecentTradesProps> = ({ trades, isLoading, onSymbolSelect, selectedSymbol }) => {
  if (isLoading) {
    return (
      <div className="dashboard-widget col-span-2">
        <h2 className="text-lg font-semibold mb-3">Recent Trades</h2>
        <div className="animate-pulse space-y-2">
          {[...Array(5)].map((_, index) => (
            <div key={index} className="bg-gray-200 dark:bg-gray-700 h-12 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (!trades || trades.length === 0) {
    return (
      <div className="dashboard-widget col-span-2">
        <h2 className="text-lg font-semibold mb-3">Recent Trades</h2>
        <div className="text-gray-500 dark:text-gray-400 text-center py-10">
          No recent trades
        </div>
      </div>
    );
  }

  const handleSymbolClick = (symbol: string) => {
    if (onSymbolSelect) {
      // Extract the base symbol from pairs like "BTC/USD"
      const baseSymbol = symbol.split('/')[0];
      onSymbolSelect(baseSymbol);
    }
  };

  // Format timestamp to readable date/time
  const formatTimestamp = (timestamp: string | number): string => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    });
  };

  return (
    <div className="dashboard-widget col-span-2">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-lg font-semibold">Recent Trades</h2>
        <div className="text-xs text-gray-500 dark:text-gray-400">
          Showing {Math.min(trades.length, 5)} of {trades.length} trades
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Time
              </th>
              <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Symbol
              </th>
              <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Side
              </th>
              <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Quantity
              </th>
              <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Price
              </th>
              <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Value
              </th>
              <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
            {trades.slice(0, 5).map((trade, index) => (
              <tr key={trade.id || index} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {formatTimestamp(trade.timestamp)}
                </td>
                <td className="px-3 py-2 whitespace-nowrap text-sm">
                  <button 
                    className="font-medium text-blue-600 dark:text-blue-400 hover:underline"
                    onClick={() => handleSymbolClick(trade.symbol)}
                  >
                    {trade.symbol}
                  </button>
                </td>
                <td className={`px-3 py-2 whitespace-nowrap text-sm ${
                  trade.side === 'buy' 
                    ? 'text-green-600 dark:text-green-400' 
                    : 'text-red-600 dark:text-red-400'
                }`}>
                  {trade.side.toUpperCase()}
                </td>
                <td className="px-3 py-2 whitespace-nowrap text-sm">
                  {trade.quantity.toLocaleString('en-US', { 
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 8
                  })}
                </td>
                <td className="px-3 py-2 whitespace-nowrap text-sm">
                  ${trade.price.toLocaleString('en-US', { 
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 8
                  })}
                </td>
                <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  ${(trade.quantity * trade.price).toLocaleString('en-US', { 
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                  })}
                </td>
                <td className="px-3 py-2 whitespace-nowrap text-sm">
                  <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                    trade.status === 'filled' 
                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                      : trade.status === 'partial' 
                        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                        : trade.status === 'pending'
                          ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                          : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}>
                    {trade.status.charAt(0).toUpperCase() + trade.status.slice(1)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="mt-4 flex justify-end">
        <button className="text-sm text-primary hover:text-primary-dark">View All Trades →</button>
      </div>
    </div>
  );
};

export default RecentTrades;
