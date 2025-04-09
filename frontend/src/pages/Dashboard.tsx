import React from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

const Dashboard: React.FC = () => {
  // Subscribe to real-time updates for portfolio and sentiment data
  const { data, status } = useWebSocket(['portfolio', 'sentiment']);
  
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Trading Dashboard</h1>
      
      {/* WebSocket connection status */}
      <div className="mb-4">
        <span className="text-sm font-medium mr-2">Real-time data:</span>
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
          status === 'connected' 
            ? 'bg-green-100 text-green-800' 
            : status === 'connecting' 
              ? 'bg-yellow-100 text-yellow-800'
              : 'bg-red-100 text-red-800'
        }`}>
          {status === 'connected' ? 'Connected' : status === 'connecting' ? 'Connecting...' : 'Disconnected'}
        </span>
      </div>
      
      {/* Dashboard grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Portfolio Summary Widget */}
        <div className="dashboard-widget col-span-1">
          <h2 className="text-lg font-semibold mb-3">Portfolio Summary</h2>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Total Value</span>
              <span className="font-medium">${data.portfolio?.total_value?.toFixed(2) || '0.00'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Cash</span>
              <span className="font-medium">${data.portfolio?.cash?.toFixed(2) || '0.00'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Positions</span>
              <span className="font-medium">{data.portfolio?.positions ? Object.keys(data.portfolio.positions).length : 0}</span>
            </div>
          </div>
          <div className="mt-4">
            <button className="text-sm text-primary hover:text-primary-dark">View Details →</button>
          </div>
        </div>
        
        {/* Performance Metrics Widget */}
        <div className="dashboard-widget col-span-1">
          <h2 className="text-lg font-semibold mb-3">Performance Metrics</h2>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Total Return</span>
              <span className="font-medium">+12.5%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Sharpe Ratio</span>
              <span className="font-medium">1.8</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Max Drawdown</span>
              <span className="font-medium">-5.2%</span>
            </div>
          </div>
          <div className="mt-4">
            <button className="text-sm text-primary hover:text-primary-dark">View Analytics →</button>
          </div>
        </div>
        
        {/* Sentiment Signal Widget */}
        <div className="dashboard-widget col-span-1">
          <h2 className="text-lg font-semibold mb-3">Sentiment Signals</h2>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-gray-600 dark:text-gray-400">BTC/USD</span>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Buy
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 dark:text-gray-400">ETH/USD</span>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                Hold
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600 dark:text-gray-400">SOL/USD</span>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                Sell
              </span>
            </div>
          </div>
          <div className="mt-4">
            <button className="text-sm text-primary hover:text-primary-dark">View All Signals →</button>
          </div>
        </div>
        
        {/* Equity Curve Widget */}
        <div className="dashboard-widget col-span-1 md:col-span-2">
          <h2 className="text-lg font-semibold mb-3">Equity Curve</h2>
          <div className="bg-gray-100 dark:bg-gray-700 rounded-lg h-64 flex items-center justify-center">
            <p className="text-gray-500 dark:text-gray-400">Chart will be displayed here</p>
          </div>
        </div>
        
        {/* Recent Orders Widget */}
        <div className="dashboard-widget col-span-1">
          <h2 className="text-lg font-semibold mb-3">Recent Orders</h2>
          <div className="space-y-3">
            <div className="flex justify-between items-start">
              <div>
                <p className="font-medium">BTC/USD</p>
                <p className="text-xs text-gray-500">Market Buy</p>
              </div>
              <div className="text-right">
                <p className="font-medium text-green-600">Filled</p>
                <p className="text-xs text-gray-500">0.5 BTC @ $45,230</p>
              </div>
            </div>
            <div className="flex justify-between items-start">
              <div>
                <p className="font-medium">ETH/USD</p>
                <p className="text-xs text-gray-500">Limit Sell</p>
              </div>
              <div className="text-right">
                <p className="font-medium text-yellow-600">Pending</p>
                <p className="text-xs text-gray-500">2.0 ETH @ $3,150</p>
              </div>
            </div>
          </div>
          <div className="mt-4">
            <button className="text-sm text-primary hover:text-primary-dark">View All Orders →</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
