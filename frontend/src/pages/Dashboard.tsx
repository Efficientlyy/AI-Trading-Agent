import React, { useEffect, useCallback, useMemo, memo } from 'react';
import { useSelectedAsset } from '../context/SelectedAssetContext';
import { Link } from 'react-router-dom';
import SimpleNotificationSystem from '../components/common/SimpleNotificationSystem';
import PortfolioSummary from '../components/dashboard/PortfolioSummary';
import PerformanceMetrics from '../components/dashboard/PerformanceMetrics';
import SentimentSummary from '../components/dashboard/SentimentSummary';
import EquityCurveChart from '../components/dashboard/EquityCurveChart';
import AssetAllocationChart from '../components/dashboard/AssetAllocationChart';
import RecentTrades from '../components/dashboard/RecentTrades';
import TechnicalAnalysisChart from '../components/dashboard/TechnicalAnalysisChart';
import useHistoricalData from '../hooks/useHistoricalData';
import TradingStrategy from '../components/dashboard/TradingStrategy';
import RiskCalculator from '../components/dashboard/RiskCalculator';
import BacktestingInterface from '../components/dashboard/BacktestingInterface';
import StrategyOptimizer from '../components/dashboard/StrategyOptimizer';
import PortfolioBacktester from '../components/dashboard/PortfolioBacktester';
import TradeStatistics from '../components/dashboard/TradeStatistics';
import PerformanceAnalysis from '../components/dashboard/PerformanceAnalysis';
import { getMockTrades } from '../api/mockData/mockTrades';

const MOCK_SYMBOL = 'AAPL';
const MOCK_STRATEGY = 'Moving Average Crossover';
const MOCK_PARAMETERS = [
  { name: 'fastPeriod', min: 5, max: 50, step: 1, currentValue: 10 },
  { name: 'slowPeriod', min: 20, max: 200, step: 5, currentValue: 50 }
];
const MOCK_ASSETS = ['AAPL', 'MSFT', 'BTC', 'ETH'];

const Dashboard: React.FC = () => {
  const { symbol: selectedSymbol, setSymbol: setSelectedSymbol } = useSelectedAsset();
  const [mockTrades, setMockTrades] = React.useState<any[]>([]);

  useEffect(() => {
    getMockTrades().then(({ trades }) => setMockTrades(trades));
  }, []);

  const { data: historicalData } = useHistoricalData({
    symbol: selectedSymbol || 'BTC',
    timeframe: '1d',
    limit: 100
  });

  const handleSymbolSelect = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
  }, [setSelectedSymbol]);

  // Memoize the navigation links to prevent unnecessary re-renders
  const navigationLinks = useMemo(() => (
    <div className="flex flex-wrap gap-4 p-6">
      <Link to="/trade">
        <button className="px-4 py-2 rounded bg-blue-600 text-white font-semibold shadow hover:bg-blue-700 transition">Trade</button>
      </Link>
      <Link to="/portfolio">
        <button className="px-4 py-2 rounded bg-green-600 text-white font-semibold shadow hover:bg-green-700 transition">Portfolio</button>
      </Link>
      <Link to="/analytics">
        <button className="px-4 py-2 rounded bg-amber-600 text-white font-semibold shadow hover:bg-amber-700 transition">Analytics</button>
      </Link>
      <Link to="/strategies">
        <button className="px-4 py-2 rounded bg-purple-600 text-white font-semibold shadow hover:bg-purple-700 transition">Strategies</button>
      </Link>
      <Link to="/settings">
        <button className="px-4 py-2 rounded bg-gray-600 text-white font-semibold shadow hover:bg-gray-700 transition">Settings</button>
      </Link>
    </div>
  ), []);

  // Memoize the first column components
  const columnOne = useMemo(() => (
    <div className="space-y-6 col-span-1">
      <PortfolioSummary />
      <PerformanceMetrics />
      <SentimentSummary onSymbolSelect={handleSymbolSelect} selectedSymbol={selectedSymbol} />
    </div>
  ), [handleSymbolSelect, selectedSymbol]);

  // Memoize the second column components
  const columnTwo = useMemo(() => (
    <div className="space-y-6 col-span-1">
      <EquityCurveChart data={[]} isLoading={false} />
      <AssetAllocationChart onAssetSelect={handleSymbolSelect} selectedAsset={selectedSymbol} />
      <RecentTrades trades={mockTrades} symbol={selectedSymbol} />
    </div>
  ), [handleSymbolSelect, selectedSymbol, mockTrades]);

  // Memoize the third column components
  const columnThree = useMemo(() => (
    <div className="space-y-6 col-span-1">
      <TechnicalAnalysisChart 
        symbol={selectedSymbol || 'BTC'}
        data={historicalData}
        isLoading={false}
      />
      <TradingStrategy symbol={selectedSymbol || 'BTC'} />
      <RiskCalculator symbol={selectedSymbol || 'BTC'} />
      <BacktestingInterface 
        symbol={selectedSymbol || MOCK_SYMBOL} 
        availableStrategies={[MOCK_STRATEGY]} 
        backtestResults={[]} 
        backtestMetrics={undefined} 
        backtestTrades={mockTrades} 
        isLoading={false} 
      />
      <StrategyOptimizer 
        symbol={selectedSymbol || MOCK_SYMBOL} 
        strategy={MOCK_STRATEGY} 
        availableParameters={MOCK_PARAMETERS} 
        optimizationResults={[]} 
        isLoading={false} 
      />
      <PortfolioBacktester availableAssets={MOCK_ASSETS} isLoading={false} />
      <TradeStatistics trades={mockTrades} />
      <PerformanceAnalysis />
    </div>
  ), [selectedSymbol, historicalData, mockTrades]);

  return (
    <>
      <SimpleNotificationSystem />
      {/* Quick Links Section */}
      {navigationLinks}
      <div className="dashboard-main grid grid-cols-1 xl:grid-cols-3 gap-6 p-6" data-testid="dashboard-grid">
        {/* Column 1: Portfolio, Performance, Sentiment */}
        {columnOne}
        {/* Column 2: Charts, Allocation, Recent Trades */}
        {columnTwo}
        {/* Column 3: Symbol Chart, Strategy, Risk, Backtesting, Stats */}
        {columnThree}
      </div>
    </>
  );
};

// Wrap the Dashboard component with React.memo to prevent unnecessary re-renders
export default memo(Dashboard);
