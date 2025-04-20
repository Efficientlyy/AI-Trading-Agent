import React from 'react';
import { Trade } from '../../types/index';

interface RecentTradesProps {
  trades: Trade[];
  symbol?: string;
  maxCount?: number;
  onTradeSymbolSelect?: (symbol: string) => void;
}

const RecentTrades: React.FC<RecentTradesProps> = ({ trades, symbol, maxCount = 10, onTradeSymbolSelect }) => {
  const filtered = symbol ? trades.filter(t => t.symbol === symbol) : trades;
  const shown = filtered.slice(0, maxCount);

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Recent Trades{symbol ? ` (${symbol})` : ''}</h2>
      {shown.length === 0 ? (
        <div className="text-gray-500 dark:text-gray-400">No recent trades.</div>
      ) : (
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left">Time</th>
              <th className="text-left">Side</th>
              <th className="text-left">Price</th>
              <th className="text-left">Qty</th>
              <th className="text-left">Status</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((trade, i) => (
              <tr 
                key={trade.id || i} 
                className={`${onTradeSymbolSelect ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800' : ''} ${symbol === trade.symbol ? 'bg-blue-50 dark:bg-blue-900/30' : ''}`}
                onClick={() => onTradeSymbolSelect && onTradeSymbolSelect(trade.symbol)}
              >
                <td>{typeof trade.timestamp === 'number' ? new Date(trade.timestamp).toLocaleTimeString() : trade.timestamp}</td>
                <td className={trade.side === 'buy' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>{trade.side.toUpperCase()}</td>
                <td>${typeof trade.price === 'number' ? trade.price.toFixed(2) : trade.price}</td>
                <td>{trade.quantity}</td>
                <td>
                  <span className={`px-2 py-1 rounded-full text-xs ${trade.status === 'completed' || trade.status === 'filled' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'}`}>
                    {trade.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default RecentTrades;
