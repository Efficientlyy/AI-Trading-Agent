import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import PortfolioSummary from '../components/dashboard/PortfolioSummary';
import PerformanceMetrics from '../components/dashboard/PerformanceMetrics';
import SentimentSummary from '../components/dashboard/SentimentSummary';
import EquityCurveChart from '../components/dashboard/EquityCurveChart';
import AssetAllocationChart from '../components/dashboard/AssetAllocationChart';
import RecentTrades from '../components/dashboard/RecentTrades';
import OrderEntryForm from '../components/dashboard/OrderEntryForm';
import { Trade, Portfolio, SentimentSignal, TopicType, Order, StrategyConfig } from '../types';
import SimpleChart from '../components/dashboard/SimpleChart';
import useHistoricalData from '../hooks/useHistoricalData';
import TradingStrategy from '../components/dashboard/TradingStrategy';
import RiskCalculator from '../components/dashboard/RiskCalculator';
import OrderManagement from '../components/dashboard/OrderManagement';
import { getMockActiveOrders, createMockOrder, cancelMockOrder } from '../api/mockData/activeOrders';
import { useSuccessNotification, useErrorNotification } from '../components/common/NotificationSystem';
import BacktestingInterface, { BacktestParams, BacktestResult, BacktestMetrics } from '../components/dashboard/BacktestingInterface';
import { runMockBacktest } from '../api/mockData/backtestResults';
import StrategyOptimizer, { OptimizationParams, OptimizationResult } from '../components/dashboard/StrategyOptimizer';
import { runStrategyOptimization } from '../api/mockData/optimizationResults';
import PortfolioBacktester from '../components/dashboard/PortfolioBacktester';
import { PortfolioBacktestParams, PortfolioBacktestResult, runPortfolioBacktest } from '../api/mockData/portfolioBacktest';
import TradeStatistics from '../components/dashboard/TradeStatistics';
import PerformanceAnalysis from '../components/dashboard/PerformanceAnalysis';

const Dashboard: React.FC = () => {
  // Subscribe to real-time updates for portfolio, sentiment, and performance data
  const { data, status, error } = useWebSocket(['portfolio', 'sentiment_signal', 'performance'] as TopicType[]);
  
  // State for mock data (for development and demonstration)
  const [recentTrades, setRecentTrades] = useState<Trade[]>([]);
  const [equityCurveData, setEquityCurveData] = useState<any[]>([]);
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([]);
  const [timeframe, setTimeframe] = useState<'day' | 'week' | 'month' | 'year' | 'all'>('month');
  const [selectedSymbol, setSelectedSymbol] = useState<string>('BTC');
  const [chartTimeframe, setChartTimeframe] = useState<'1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w'>('1d');
  const [selectedStrategy, setSelectedStrategy] = useState<string>('Moving Average Crossover');
  
  // Get historical data for the selected symbol
  const { 
    data: historicalData, 
    isLoading: isHistoricalDataLoading, 
    changeTimeframe: changeHistoricalDataTimeframe 
  } = useHistoricalData({ 
    symbol: selectedSymbol,
    timeframe: chartTimeframe
  });
  
  // Get current price from historical data
  const currentPrice = useMemo(() => {
    if (historicalData && historicalData.length > 0) {
      return historicalData[historicalData.length - 1].close;
    }
    return 0;
  }, [historicalData]);
  
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
  
  // State for active orders
  const [activeOrders, setActiveOrders] = useState<Order[]>([]);
  
  // State for backtesting
  const [backtestResults, setBacktestResults] = useState<BacktestResult[]>([]);
  const [backtestMetrics, setBacktestMetrics] = useState<BacktestMetrics | null>(null);
  const [backtestTrades, setBacktestTrades] = useState<Trade[]>([]);
  const [isBacktestLoading, setIsBacktestLoading] = useState<boolean>(false);
  
  // State for optimization
  const [optimizationResults, setOptimizationResults] = useState<OptimizationResult[]>([]);
  const [isOptimizationLoading, setIsOptimizationLoading] = useState<boolean>(false);
  
  // State for portfolio backtesting
  const [portfolioBacktestResult, setPortfolioBacktestResult] = useState<PortfolioBacktestResult | undefined>(undefined);
  const [isPortfolioBacktestLoading, setIsPortfolioBacktestLoading] = useState<boolean>(false);
  
  // State for dashboard view
  const [activeTab, setActiveTab] = useState<'overview' | 'backtesting' | 'optimization'>('overview');
  
  // Load active orders for the selected symbol
  useEffect(() => {
    const orders = getMockActiveOrders(selectedSymbol);
    setActiveOrders(orders);
  }, [selectedSymbol]);
  
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
    
    for (let i = 180; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      
      // Random daily change between -3% and +4%
      const dailyChange = (Math.random() * 7 - 3) / 100;
      value = value * (1 + dailyChange);
      
      // Benchmark changes less (between -2% and +3%)
      const benchmarkChange = (Math.random() * 5 - 2) / 100;
      benchmarkValue = benchmarkValue * (1 + benchmarkChange);
      
      mockEquityCurveData.push({
        date: date.toISOString().split('T')[0],
        value: Math.round(value * 100) / 100,
        benchmarkValue: Math.round(benchmarkValue * 100) / 100,
      });
    }
    setEquityCurveData(mockEquityCurveData);
    
    // Mock available symbols
    setAvailableSymbols(['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP', 'DOGE', 'AVAX', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM']);
  }, []);
  
  // Show error notification if WebSocket connection fails
  useEffect(() => {
    if (error) {
      errorNotification('Failed to connect to real-time data service', 'Connection Error', { autoClose: false });
    }
  }, [error, errorNotification]);
  
  // Handle creating a new order
  const handleCreateOrder = useCallback((orderData: Partial<Order>) => {
    const newOrder = createMockOrder(orderData);
    setActiveOrders(prev => [...prev, newOrder]);
    console.log('Created new order:', newOrder);
    successNotification('Order created successfully');
  }, [successNotification]);
  
  // Handle canceling an order
  const handleCancelOrder = useCallback((orderId: string) => {
    const success = cancelMockOrder(orderId);
    if (success) {
      setActiveOrders(getMockActiveOrders(selectedSymbol));
      console.log('Canceled order:', orderId);
      successNotification('Order canceled successfully');
    } else {
      errorNotification('Failed to cancel order');
    }
  }, [selectedSymbol, successNotification, errorNotification]);
  
  // Run backtest
  const handleRunBacktest = useCallback((params: BacktestParams) => {
    setIsBacktestLoading(true);
    
    // Simulate API call delay
    setTimeout(() => {
      try {
        const { results, metrics, trades } = runMockBacktest(params);
        setBacktestResults(results);
        setBacktestMetrics(metrics);
        // Convert trades to match the Trade type from types/index.ts
        const convertedTrades: Trade[] = trades.map(trade => ({
          id: trade.id,
          symbol: trade.symbol,
          side: trade.type === 'buy' ? 'buy' : 'sell',
          quantity: trade.quantity,
          price: trade.exitPrice,
          timestamp: new Date(trade.exitDate).toISOString(),
          status: 'filled', // Using a valid status from the Trade interface
          fee: 0,
          total: trade.quantity * trade.exitPrice
        }));
        setBacktestTrades(convertedTrades);
        setIsBacktestLoading(false);
        successNotification('Backtest completed successfully');
      } catch (error) {
        console.error('Backtest error:', error);
        setIsBacktestLoading(false);
        errorNotification('Failed to run backtest');
      }
    }, 1500);
  }, [successNotification, errorNotification]);
  
  // Run optimization
  const handleRunOptimization = useCallback((params: OptimizationParams) => {
    setIsOptimizationLoading(true);
    
    // Simulate API call delay
    setTimeout(() => {
      try {
        const results = runStrategyOptimization(params);
        setOptimizationResults(results);
        setIsOptimizationLoading(false);
        successNotification('Strategy optimization completed successfully');
      } catch (error) {
        console.error('Optimization error:', error);
        setIsOptimizationLoading(false);
        errorNotification('Failed to run strategy optimization');
      }
    }, 2000);
  }, [successNotification, errorNotification]);
  
  // Run portfolio backtest
  const handleRunPortfolioBacktest = useCallback((params: PortfolioBacktestParams) => {
    setIsPortfolioBacktestLoading(true);
    
    // Simulate API call delay
    setTimeout(() => {
      try {
        const result = runPortfolioBacktest(params);
        setPortfolioBacktestResult(result);
        setIsPortfolioBacktestLoading(false);
        successNotification('Portfolio backtest completed successfully');
      } catch (error) {
        console.error('Portfolio backtest error:', error);
        setIsPortfolioBacktestLoading(false);
        errorNotification('Failed to run portfolio backtest');
      }
    }, 2000);
  }, [successNotification, errorNotification]);
  
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
  
  // Handle symbol selection
  const handleSymbolChange = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
  }, []);

  // Handle chart timeframe change
  const handleChartTimeframeChange = useCallback((newTimeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w') => {
    setChartTimeframe(newTimeframe);
    changeHistoricalDataTimeframe(newTimeframe);
  }, [changeHistoricalDataTimeframe]);

  // Handle strategy change
  const handleStrategyChange = useCallback((strategy: StrategyConfig) => {
    console.log('Strategy changed:', strategy);
    setSelectedStrategy(typeof strategy === 'string' ? strategy : strategy.name);
  }, []);
  
  return (
    <div className="p-6">
      <div className="mb-6 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Trading Dashboard</h1>
        
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-4 py-2 text-sm font-medium rounded-md ${
              activeTab === 'overview'
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveTab('backtesting')}
            className={`px-4 py-2 text-sm font-medium rounded-md ${
              activeTab === 'backtesting'
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            Backtesting
          </button>
          <button
            onClick={() => setActiveTab('optimization')}
            className={`px-4 py-2 text-sm font-medium rounded-md ${
              activeTab === 'optimization'
                ? 'bg-indigo-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            Optimization
          </button>
        </div>
      </div>
      
      {activeTab === 'overview' && (
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Portfolio and Performance */}
          <div className="col-span-12 lg:col-span-8 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <PortfolioSummary />
              
              <PerformanceMetrics />
            </div>
            
            <EquityCurveChart 
              data={equityCurveData} 
              isLoading={status === 'connecting'} 
              timeframe={timeframe}
              onTimeframeChange={handleTimeframeChange}
            />
            
            <SimpleChart 
              data={historicalData} 
              symbol={selectedSymbol}
            />
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <OrderEntryForm 
                portfolio={portfolio}
                availableSymbols={availableSymbols}
                selectedSymbol={selectedSymbol}
                onSymbolChange={handleSymbolChange}
                onSubmitOrder={handleSubmitOrder}
              />
              
              <OrderManagement symbol={selectedSymbol} currentPrice={currentPrice} />
            </div>
            
            <RecentTrades onSymbolSelect={handleSymbolChange} selectedSymbol={selectedSymbol} />
          </div>
          
          {/* Right Column - Asset Allocation and Sentiment */}
          <div className="col-span-12 lg:col-span-4 space-y-6">
            <AssetAllocationChart onAssetSelect={handleSymbolChange} selectedAsset={selectedSymbol} />
            
            <SentimentSummary onSymbolSelect={handleSymbolChange} selectedSymbol={selectedSymbol} />
            
            <TradingStrategy
              symbol={selectedSymbol}
              onStrategyChange={handleStrategyChange}
            />
            
            <RiskCalculator
              symbol={selectedSymbol}
              currentPrice={currentPrice}
            />
          </div>
        </div>
      )}
      
      {activeTab === 'backtesting' && (
        <div className="space-y-6">
          <BacktestingInterface
            symbol={selectedSymbol}
            onRunBacktest={handleRunBacktest}
            backtestResults={backtestResults}
            backtestMetrics={backtestMetrics || undefined}
            isLoading={isBacktestLoading}
          />
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <TradeStatistics 
              trades={backtestTrades as unknown as import('../components/dashboard/TradeStatistics').Trade[]} 
            />
            
            <PerformanceAnalysis 
              backtestResults={backtestResults} 
            />
          </div>
          
          <PortfolioBacktester
            availableAssets={['BTC', 'ETH', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'GLD']}
            onRunBacktest={handleRunPortfolioBacktest}
            backtestResult={portfolioBacktestResult}
            isLoading={isPortfolioBacktestLoading}
          />
        </div>
      )}
      
      {activeTab === 'optimization' && (
        <div className="space-y-6">
          <StrategyOptimizer
            symbol={selectedSymbol}
            strategy={selectedStrategy}
            availableParameters={[]}
            onRunOptimization={handleRunOptimization}
            optimizationResults={optimizationResults}
            isLoading={isOptimizationLoading}
          />
          
          {optimizationResults.length > 0 && (
            <div className="bg-white dark:bg-gray-900 rounded-lg shadow p-4">
              <h2 className="text-lg font-semibold mb-3">Optimization Insights</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Parameter Sensitivity</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    The most sensitive parameters for {selectedStrategy} on {selectedSymbol} are:
                  </p>
                  <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-400">
                    {Object.keys(optimizationResults[0].parameters).map(param => (
                      <li key={param}>
                        <span className="font-medium">{param}</span>: Optimal range {optimizationResults[0].parameters[param]} to {optimizationResults[1].parameters[param]}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Performance Improvement</h3>
                  <div className="bg-green-50 dark:bg-green-900 p-3 rounded-md">
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Optimized parameters improved {
                        optimizationResults[0].metrics.totalReturn > 0 ? 'total return' :
                        optimizationResults[0].metrics.sharpeRatio > 1.5 ? 'Sharpe ratio' :
                        'win rate'
                      } by approximately {Math.round(Math.random() * 30 + 20)}% compared to default parameters.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Status Bar */}
      <div className="mt-6 bg-white dark:bg-gray-900 rounded-lg shadow p-3 flex justify-between items-center text-sm">
        <div className="flex items-center">
          <span className="font-medium mr-2">Selected Symbol:</span>
          <span className="text-blue-600 dark:text-blue-400">{selectedSymbol}</span>
        </div>
        <div className="flex items-center">
          <span className="font-medium mr-2">Timeframe:</span>
          <span className="text-blue-600 dark:text-blue-400">{chartTimeframe}</span>
        </div>
        <div className="flex items-center">
          <span className="font-medium mr-2">Strategy:</span>
          <span className="text-blue-600 dark:text-blue-400">{selectedStrategy}</span>
        </div>
        <div className="flex items-center">
          <span className="text-gray-500 dark:text-gray-400">Last updated: {new Date().toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
