import React from 'react';
import { OrderSide } from '../../types';

interface TradeHistoryProps {
  trades: Array<{
    id: string;
    symbol: string;
    side: 'buy' | 'sell' | OrderSide;
    quantity: number;
    price: number;
    timestamp: number;
    status: string;
  }>;
}

const TradeHistory: React.FC<TradeHistoryProps> = ({ trades }) => {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Trade History</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr>
              <th className="px-2 py-1">Time</th>
              <th className="px-2 py-1">Symbol</th>
              <th className="px-2 py-1">Side</th>
              <th className="px-2 py-1">Qty</th>
              <th className="px-2 py-1">Price</th>
              <th className="px-2 py-1">Status</th>
            </tr>
          </thead>
          <tbody>
            {trades.length === 0 ? (
              <tr><td colSpan={6} className="text-center text-gray-400">No trades found</td></tr>
            ) : (
              trades.map(trade => (
                <tr key={trade.id}>
                  <td className="px-2 py-1">{new Date(trade.timestamp).toLocaleString()}</td>
                  <td className="px-2 py-1">{trade.symbol}</td>
                  <td className={`px-2 py-1 font-semibold ${
                    trade.side === OrderSide.BUY || trade.side === 'buy' ? 
                    'text-green-600' : 'text-red-600'
                  }`}>{typeof trade.side === 'string' ? trade.side.toUpperCase() : trade.side}</td>
                  <td className="px-2 py-1">{trade.quantity}</td>
                  <td className="px-2 py-1">{trade.price}</td>
                  <td className="px-2 py-1">{trade.status}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default TradeHistory;
