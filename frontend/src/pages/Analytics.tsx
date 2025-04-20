import React, { useState, useEffect } from 'react';
import EquityCurveChart from '../components/dashboard/EquityCurveChart';
import TradeStatistics from '../components/dashboard/TradeStatistics';
import PerformanceAnalysis from '../components/dashboard/PerformanceAnalysis';
import TechnicalAnalysisChart from '../components/dashboard/TechnicalAnalysisChart';
import { useSelectedAsset } from '../context/SelectedAssetContext';
import { Link } from 'react-router-dom';

const Analytics: React.FC = () => {
  const [tab, setTab] = useState<'equity' | 'stats' | 'performance' | 'technical'>('equity');
  const { symbol: selectedSymbol } = useSelectedAsset();
  const [mockTrades, setMockTrades] = useState<any[]>([]);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch mock data for analytics
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        // In a real app, this would fetch from an API
        // For now, we'll create some mock data
        const mockEquityData = Array.from({ length: 30 }, (_, i) => ({
          date: new Date(Date.now() - (30 - i) * 24 * 60 * 60 * 1000).toISOString(),
          equity: 10000 + Math.random() * 5000 * Math.sin(i / 5)
        }));
        
        const mockTradeData = Array.from({ length: 20 }, (_, i) => ({
          id: `trade-${i}`,
          symbol: selectedSymbol || 'BTC/USD',
          side: Math.random() > 0.5 ? 'buy' : 'sell',
          price: 50000 + Math.random() * 10000,
          quantity: Math.random() * 2,
          timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
          pnl: Math.random() * 2000 - 1000,
          status: 'completed'
        }));
        
        setHistoricalData(mockEquityData);
        setMockTrades(mockTradeData);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, [selectedSymbol]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 py-10">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">Analytics</h1>
          <Link to="/" className="px-4 py-2 rounded bg-blue-600 text-white font-semibold shadow hover:bg-blue-700 transition">
            Back to Dashboard
          </Link>
        </div>
        
        <div className="flex justify-center gap-4 mb-8">
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'equity' ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('equity')}
          >
            Equity Curve Chart
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'stats' ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('stats')}
          >
            Trade Statistics
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'performance' ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('performance')}
          >
            Performance Analysis
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold ${tab === 'technical' ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200'}`}
            onClick={() => setTab('technical')}
          >
            Technical Analysis
          </button>
        </div>
        
        {isLoading ? (
          <div className="text-center py-10">
            <p className="text-lg">Loading analytics data...</p>
          </div>
        ) : (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            {tab === 'equity' && <EquityCurveChart data={historicalData} isLoading={false} />}
            {tab === 'stats' && <TradeStatistics trades={mockTrades} />}
            {tab === 'performance' && <PerformanceAnalysis />}
            {tab === 'technical' && (
              <TechnicalAnalysisChart 
                symbol={selectedSymbol || 'BTC/USD'}
                data={[]} 
                isLoading={false}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Analytics;
