import React from 'react';
import { Position } from '../../types';

interface PositionDetailsProps {
  position: Position | null;
  symbol: string;
}

const PositionDetails: React.FC<PositionDetailsProps> = ({ position, symbol }) => {
  if (!position) {
    return (
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
        <h2 className="text-lg font-semibold mb-3">Position Details ({symbol})</h2>
        <div className="text-gray-500 dark:text-gray-400">No open position for this symbol.</div>
      </div>
    );
  }
  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-3">Position Details ({symbol})</h2>
      <table className="w-full text-sm">
        <tbody>
          <tr>
            <td className="font-medium text-gray-700 dark:text-gray-300 pr-2">Quantity:</td>
            <td>{position.quantity}</td>
          </tr>
          <tr>
            <td className="font-medium text-gray-700 dark:text-gray-300 pr-2">Entry Price:</td>
            <td>${position.entry_price.toFixed(2)}</td>
          </tr>
          <tr>
            <td className="font-medium text-gray-700 dark:text-gray-300 pr-2">Current Price:</td>
            <td>${position.current_price.toFixed(2)}</td>
          </tr>
          <tr>
            <td className="font-medium text-gray-700 dark:text-gray-300 pr-2">Market Value:</td>
            <td>${position.market_value.toFixed(2)}</td>
          </tr>
          <tr>
            <td className="font-medium text-gray-700 dark:text-gray-300 pr-2">Unrealized PnL:</td>
            <td className={position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
              ${position.unrealized_pnl.toFixed(2)}
            </td>
          </tr>
          <tr>
            <td className="font-medium text-gray-700 dark:text-gray-300 pr-2">Realized PnL:</td>
            <td className={position.realized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
              ${position.realized_pnl.toFixed(2)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default PositionDetails;
