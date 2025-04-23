import React, { useEffect, useCallback, useMemo, memo } from 'react';
import { useRenderLogger } from '../hooks/useRenderLogger';
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
import AgentStatus from '../components/dashboard/AgentStatus';
import AgentControls from '../components/dashboard/AgentControls';
import AgentAutonomyBanner from '../components/dashboard/AgentAutonomyBanner';
import { useWebSocket } from '../hooks/useWebSocket';
import { getMockTrades } from '../api/mockData/mockTrades';
import { useNotification } from '../components/common/NotificationSystem';

const MOCK_SYMBOL = 'AAPL';
const MOCK_STRATEGY = 'Moving Average Crossover';
const MOCK_PARAMETERS = [
  { name: 'fastPeriod', min: 5, max: 50, step: 1, currentValue: 10 },
  { name: 'slowPeriod', min: 20, max: 200, step: 5, currentValue: 50 }
];
const MOCK_ASSETS = ['AAPL', 'MSFT', 'BTC', 'ETH'];

const Dashboard: React.FC = () => {
  // ...existing code...
  const { data: wsData } = useWebSocket(['agent_status', 'portfolio', 'recent_trades']);
  useRenderLogger('Dashboard');
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
      <PortfolioSummary {...(wsData.portfolio ? {
        totalValue: wsData.portfolio.total_value,
        availableCash: wsData.portfolio.cash
      } : {})} />
      <PerformanceMetrics />
      <SentimentSummary onSymbolSelect={handleSymbolSelect} selectedSymbol={selectedSymbol} />
    </div>
  ), [wsData.portfolio, handleSymbolSelect, selectedSymbol]);

  // Memoize the second column components
  const columnTwo = useMemo(() => (
    <div className="space-y-6 col-span-1">
      <EquityCurveChart data={[]} isLoading={false} />
      <AssetAllocationChart onAssetSelect={handleSymbolSelect} selectedAsset={selectedSymbol} />
      <RecentTrades trades={wsData.recent_trades || mockTrades} symbol={selectedSymbol} />
    </div>
  ), [handleSymbolSelect, selectedSymbol, wsData.recent_trades, mockTrades]);

  // Memoize the third column components
  const [agentLoading, setAgentLoading] = React.useState(false);
const prevStatusRef = React.useRef(wsData.agent_status?.status);
const { addNotification } = useNotification();

// When agent_status changes, disable loading and show notification
React.useEffect(() => {
  if (agentLoading && wsData.agent_status?.status !== prevStatusRef.current) {
    setAgentLoading(false);
    if (wsData.agent_status?.status === 'running') {
      addNotification({
        type: 'success',
        title: 'Agent Started',
        message: 'Trading agent is now running.'
      });
    } else if (wsData.agent_status?.status === 'stopped') {
      addNotification({
        type: 'info',
        title: 'Agent Stopped',
        message: 'Trading agent has been stopped.'
      });
    } else if (wsData.agent_status?.status === 'error') {
      addNotification({
        type: 'error',
        title: 'Agent Error',
        message: wsData.agent_status?.reasoning || 'An error occurred with the trading agent.'
      });
    }
    prevStatusRef.current = wsData.agent_status?.status;
  }
}, [wsData.agent_status?.status, agentLoading, addNotification]);

const { status: wsStatus } = useWebSocket([]); // get wsRef
const wsRef = React.useRef<WebSocket | null>(null);

// Patch: get wsRef from useWebSocket if exposed, else fallback
// (If wsRef is not exposed, consider patching useWebSocket to provide it)

const sendAgentAction = (action: 'start_agent' | 'stop_agent') => {
  setAgentLoading(true);
  try {
    const ws = (window as any).wsRef || wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action }));
      addNotification({
        type: 'info',
        title: action === 'start_agent' ? 'Starting Agent...' : 'Stopping Agent...',
        message: action === 'start_agent' ? 'Attempting to start the trading agent.' : 'Attempting to stop the trading agent.'
      });
    } else {
      setAgentLoading(false);
      addNotification({
        type: 'error',
        title: 'WebSocket Error',
        message: 'WebSocket not connected. Please refresh and try again.'
      });
    }
  } catch (e) {
    setAgentLoading(false);
    addNotification({
      type: 'error',
      title: 'Action Failed',
      message: 'Failed to send command: ' + (e as any).message
    });
  }
};

const handleStartAgent = () => sendAgentAction('start_agent');
const handleStopAgent = () => sendAgentAction('stop_agent');

const columnThree = useMemo(() => (
    <div className="space-y-6 col-span-1">
      <AgentControls
        status={wsData.agent_status?.status}
        onStart={handleStartAgent}
        onStop={handleStopAgent}
        isLoading={agentLoading}
      />
      <AgentStatus
        status={wsData.agent_status?.status}
        reasoning={wsData.agent_status?.reasoning}
        lastUpdated={wsData.agent_status?.timestamp ? new Date(wsData.agent_status.timestamp).toLocaleTimeString() : undefined}
      />
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
      {/* Autonomy Banner */}
      <div className="px-6">
        <AgentAutonomyBanner
          status={wsData.agent_status?.status}
          lastUpdated={wsData.agent_status?.timestamp ? new Date(wsData.agent_status.timestamp).toLocaleTimeString() : undefined}
          lastTrade={wsData.recent_trades && wsData.recent_trades.length > 0 ? {
            symbol: wsData.recent_trades[wsData.recent_trades.length-1].symbol,
            side: wsData.recent_trades[wsData.recent_trades.length-1].side,
            price: wsData.recent_trades[wsData.recent_trades.length-1].price,
            timestamp: wsData.recent_trades[wsData.recent_trades.length-1].timestamp.toString()
          } : undefined}
          currentStrategy={wsData.agent_status?.reasoning && wsData.agent_status?.reasoning.match(/strategy: ([^\.;]+)/i)?.[1]}
          reasoning={wsData.agent_status?.reasoning}
          activityFeed={(() => {
            const feed: Array<{time: string, message: string}> = [];
            if (wsData.recent_trades) {
              wsData.recent_trades.slice(-5).forEach(trade => {
                feed.push({
                  time: typeof trade.timestamp === 'number' ? new Date(trade.timestamp).toLocaleTimeString() : trade.timestamp,
                  message: `${trade.side.toUpperCase()} ${trade.symbol} @ $${trade.price}`
                });
              });
            }
            if (wsData.agent_status) {
              feed.push({
                time: wsData.agent_status.timestamp ? new Date(wsData.agent_status.timestamp).toLocaleTimeString() : '',
                message: `Agent status: ${wsData.agent_status.status}`
              });
            }
            return feed.reverse();
          })()}
        />
      </div>
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
