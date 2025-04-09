import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import PortfolioSummary from '../components/dashboard/PortfolioSummary';
import PerformanceMetrics from '../components/dashboard/PerformanceMetrics';
import SentimentSummary from '../components/dashboard/SentimentSummary';
import RecentTrades from '../components/dashboard/RecentTrades';
import EquityCurveChart from '../components/dashboard/EquityCurveChart';
import AssetAllocationChart from '../components/dashboard/AssetAllocationChart';
import OrderEntryForm from '../components/dashboard/OrderEntryForm';
import { useSuccessNotification, useErrorNotification } from '../components/common/NotificationSystem';
import { Trade, Portfolio, SentimentSignal, TopicType } from '../types';

const Dashboard: React.FC = () => {
  // Subscribe to real-time updates for portfolio, sentiment, and performance data
  const { data, status, error } = useWebSocket(['portfolio', 'sentiment_signal', 'performance'] as TopicType[]);
  
  // State for mock data (for development and demonstration)
  const [recentTrades, setRecentTrades] = useState<Trade[]>([]);
  const [equityCurveData, setEquityCurveData] = useState<any[]>([]);
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [timeframe, setTimeframe] = useState<'day' | 'week' | 'month' | 'year' | 'all'>('month');
  
  // State for data with proper null handling
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [performanceData, setPerformanceData] = useState<{
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate?: number;
    profit_factor?: number;
    avg_trade?: number;
  } | null>(null);
  const [sentimentData, setSentimentData] = useState<Record<string, SentimentSignal> | null>(null);
  
  // Notification hooks
  const successNotification = useSuccessNotification();
  const errorNotification = useErrorNotification();
  
  // Update state with WebSocket data
  useEffect(() => {
    // Only update state if the data has actually changed
    if (data.portfolio && JSON.stringify(data.portfolio) !== JSON.stringify(portfolio)) {
      setPortfolio(data.portfolio);
    }
    
    if (data.performance && JSON.stringify(data.performance) !== JSON.stringify(performanceData)) {
      setPerformanceData(data.performance);
    }
    
    if (data.sentiment_signal && JSON.stringify(data.sentiment_signal) !== JSON.stringify(sentimentData)) {
      setSentimentData(data.sentiment_signal);
    }
  }, [data, portfolio, performanceData, sentimentData]);
  
  // Generate mock data for development
  useEffect(() => {
    // Mock recent trades
    const mockTrades: Trade[] = [
      { id: '1', symbol: 'BTC/USD', side: 'buy', quantity: 0.5, price: 45230, timestamp: Date.now() - 3600000, status: 'filled' },
      { id: '2', symbol: 'ETH/USD', side: 'sell', quantity: 2.0, price: 3150, timestamp: Date.now() - 7200000, status: 'filled' },
      { id: '3', symbol: 'SOL/USD', side: 'buy', quantity: 10.0, price: 120.5, timestamp: Date.now() - 10800000, status: 'partial' },
      { id: '4', symbol: 'ADA/USD', side: 'sell', quantity: 500, price: 1.25, timestamp: Date.now() - 14400000, status: 'cancelled' },
      { id: '5', symbol: 'DOT/USD', side: 'buy', quantity: 25, price: 18.75, timestamp: Date.now() - 18000000, status: 'filled' },
    ];
    setRecentTrades(mockTrades);
    
    // Mock equity curve data
    const now = new Date();
    const mockEquityCurveData = [];
    let value = 10000;
    let benchmarkValue = 10000;
    
    // Generate data points for the last 30 days
    for (let i = 30; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      
      // Add some randomness to the values
      const change = (Math.random() * 2 - 0.5) * (value * 0.02); // -0.5% to 1.5% daily change
      const benchmarkChange = (Math.random() * 1.5 - 0.5) * (benchmarkValue * 0.015); // -0.5% to 1% daily change
      
      value += change;
      benchmarkValue += benchmarkChange;
      
      mockEquityCurveData.push({
        timestamp: date.getTime(),
        value,
        benchmark: benchmarkValue
      });
    }
    setEquityCurveData(mockEquityCurveData);
    
    // Mock available symbols
    setAvailableSymbols(['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD', 'XRP/USD', 'DOGE/USD', 'AVAX/USD']);
  }, []);
  
  // Show error notification if WebSocket connection fails
  useEffect(() => {
    if (error) {
      errorNotification('Failed to connect to real-time data service', 'Connection Error', { autoClose: false });
    }
  }, [error, errorNotification]);
  
  // Handle order submission
  const handleSubmitOrder = (order: any) => {
    console.log('Order submitted:', order);
    
    // In a real application, this would send the order to the backend
    // For now, we'll just show a success notification
    successNotification(
      `${order.side.toUpperCase()} ${order.quantity} ${order.symbol} at ${order.type === 'market' ? 'market price' : `$${order.price}`}`,
      'Order Submitted'
    );
    
    // Add the order to recent trades (in a real app, this would come from the backend)
    const newTrade: Trade = {
      id: Math.random().toString(36).substring(2, 9),
      symbol: order.symbol,
      side: order.side,
      quantity: order.quantity,
      price: order.price || (portfolio?.positions?.[order.symbol]?.current_price || 0),
      timestamp: Date.now(),
      status: 'pending'
    };
    
    setRecentTrades(prevTrades => [newTrade, ...prevTrades]);
  };
  
  // Handle timeframe change for equity curve
  const handleTimeframeChange = (newTimeframe: 'day' | 'week' | 'month' | 'year' | 'all') => {
    setTimeframe(newTimeframe);
  };
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Trading Dashboard</h1>
        
        {/* WebSocket connection status */}
        <div>
          <span className="text-sm font-medium mr-2">Real-time data:</span>
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
            status === 'connected' 
              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
              : status === 'connecting' 
                ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
          }`}>
            {status === 'connected' ? 'Connected' : status === 'connecting' ? 'Connecting...' : 'Disconnected'}
          </span>
        </div>
      </div>
      
      {/* Dashboard grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Row 1: Summary Widgets */}
        <PortfolioSummary portfolio={portfolio} isLoading={status === 'connecting'} />
        <PerformanceMetrics performance={performanceData} isLoading={status === 'connecting'} />
        <SentimentSummary sentimentData={sentimentData} isLoading={status === 'connecting'} />
        
        {/* Row 2: Charts */}
        <EquityCurveChart 
          data={equityCurveData} 
          isLoading={status === 'connecting'} 
          timeframe={timeframe}
          showBenchmark={true}
          onTimeframeChange={handleTimeframeChange}
        />
        <AssetAllocationChart portfolio={portfolio} isLoading={status === 'connecting'} />
        
        {/* Row 3: Trading Interface and Recent Trades */}
        <OrderEntryForm 
          portfolio={portfolio} 
          availableSymbols={availableSymbols}
          onSubmitOrder={handleSubmitOrder}
        />
        <RecentTrades trades={recentTrades} isLoading={status === 'connecting'} />
      </div>
    </div>
  );
};

export default Dashboard;
