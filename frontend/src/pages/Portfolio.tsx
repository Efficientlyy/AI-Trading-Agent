import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useSelectedAsset } from '../context/SelectedAssetContext';
import AssetAllocationChart from '../components/dashboard/AssetAllocationChart';
import PortfolioSummary from '../components/dashboard/PortfolioSummary';
import PositionsList from '../components/dashboard/PositionsList';
import RecentTrades from '../components/dashboard/RecentTrades';

// Define types for portfolio data
interface Position {
  symbol: string;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercentage: number;
}

interface PortfolioData {
  totalValue: number;
  availableCash: number;
  totalPnl: number;
  totalPnlPercentage: number;
  positions: Position[];
}

const Portfolio: React.FC = () => {
  const { symbol: selectedSymbol, setSymbol: setSelectedSymbol } = useSelectedAsset();
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [recentTrades, setRecentTrades] = useState<any[]>([]);
  const [assetAllocation, setAssetAllocation] = useState<any[]>([]);
  const [timeframe, setTimeframe] = useState<'1d' | '1w' | '1m' | 'all'>('1m');

  // Handle asset selection from charts or tables
  const handleAssetSelect = (symbol: string) => {
    setSelectedSymbol(symbol);
  };

  // Fetch portfolio data
  useEffect(() => {
    const fetchPortfolioData = async () => {
      setIsLoading(true);
      try {
        // In a real app, this would be an API call
        // For now, we'll create mock data
        const mockPositions: Position[] = [
          { symbol: 'BTC/USD', quantity: 0.5, entryPrice: 45000, currentPrice: 48000, pnl: 1500, pnlPercentage: 6.67 },
          { symbol: 'ETH/USD', quantity: 5, entryPrice: 3200, currentPrice: 3500, pnl: 1500, pnlPercentage: 9.38 },
          { symbol: 'SOL/USD', quantity: 20, entryPrice: 120, currentPrice: 110, pnl: -200, pnlPercentage: -8.33 },
          { symbol: 'ADA/USD', quantity: 1000, entryPrice: 0.5, currentPrice: 0.55, pnl: 50, pnlPercentage: 10.0 },
        ];
        
        const mockPortfolio: PortfolioData = {
          totalValue: 50000,
          availableCash: 25000,
          totalPnl: 2850,
          totalPnlPercentage: 6.05,
          positions: mockPositions
        };
        
        // Create mock trades
        const mockTrades = Array.from({ length: 15 }, (_, i) => {
          const randomSymbol = mockPositions[Math.floor(Math.random() * mockPositions.length)].symbol;
          const isBuy = Math.random() > 0.5;
          const price = mockPositions.find(p => p.symbol === randomSymbol)?.currentPrice || 0;
          const randomQuantity = Math.random() * 5;
          
          return {
            id: `trade-${i}`,
            symbol: randomSymbol,
            side: isBuy ? 'buy' : 'sell',
            price: price * (0.98 + Math.random() * 0.04), // +/- 2% from current price
            quantity: randomQuantity,
            value: price * randomQuantity,
            timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
            status: 'completed'
          };
        });
        
        // Create asset allocation data
        const mockAllocation = mockPositions.map(position => ({
          asset: position.symbol,
          value: position.quantity * position.currentPrice,
          percentage: (position.quantity * position.currentPrice / mockPortfolio.totalValue) * 100
        }));
        
        // Add cash to allocation
        mockAllocation.push({
          asset: 'Cash',
          value: mockPortfolio.availableCash,
          percentage: (mockPortfolio.availableCash / mockPortfolio.totalValue) * 100
        });
        
        setPortfolioData(mockPortfolio);
        setRecentTrades(mockTrades);
        setAssetAllocation(mockAllocation);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchPortfolioData();
  }, []);

  // Handle timeframe change for portfolio performance
  const handleTimeframeChange = (newTimeframe: typeof timeframe) => {
    setTimeframe(newTimeframe);
    // In a real app, this would trigger a re-fetch of data for the new timeframe
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 py-10">
        <div className="max-w-6xl mx-auto">
          <div className="text-center py-20">
            <p className="text-xl">Loading portfolio data...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 py-10">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">Portfolio</h1>
          <div className="flex gap-4">
            <Link to="/trade" className="px-4 py-2 rounded bg-blue-600 text-white font-semibold shadow hover:bg-blue-700 transition">
              Trade
            </Link>
            <Link to="/" className="px-4 py-2 rounded bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 font-semibold shadow hover:bg-gray-300 dark:hover:bg-gray-600 transition">
              Dashboard
            </Link>
          </div>
        </div>
        
        {/* Portfolio Summary */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2">
            <PortfolioSummary 
              totalValue={portfolioData?.totalValue || 0}
              availableCash={portfolioData?.availableCash || 0}
              totalPnl={portfolioData?.totalPnl || 0}
              totalPnlPercentage={portfolioData?.totalPnlPercentage || 0}
              timeframe={timeframe}
              onTimeframeChange={handleTimeframeChange}
            />
          </div>
          <div>
            <AssetAllocationChart 
              data={assetAllocation} 
              onAssetSelect={handleAssetSelect}
            />
          </div>
        </div>
        
        {/* Positions List */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Positions</h2>
          <PositionsList 
            positions={portfolioData?.positions || []} 
            onPositionSelect={handleAssetSelect}
            selectedSymbol={selectedSymbol}
          />
        </div>
        
        {/* Recent Trades */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Trades</h2>
          <RecentTrades 
            trades={recentTrades} 
            onTradeSymbolSelect={handleAssetSelect}
          />
        </div>
      </div>
    </div>
  );
};

export default Portfolio;
