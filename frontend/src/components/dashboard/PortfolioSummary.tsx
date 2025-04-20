import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Portfolio } from '../../types';
import { useDataSource } from '../../context/DataSourceContext';
import { portfolioApi } from '../../api/portfolio';
import { getMockPortfolio } from '../../api/mockData/mockPortfolio';

interface PortfolioSummaryProps {
  totalValue?: number;
  availableCash?: number;
  totalPnl?: number;
  totalPnlPercentage?: number;
  timeframe?: '1d' | '1w' | '1m' | 'all';
  onTimeframeChange?: (timeframe: '1d' | '1w' | '1m' | 'all') => void;
}

const PortfolioSummary: React.FC<PortfolioSummaryProps> = ({
  totalValue: propsTotalValue,
  availableCash: propsAvailableCash,
  totalPnl: propsTotalPnl,
  totalPnlPercentage: propsTotalPnlPercentage,
  timeframe = '1m',
  onTimeframeChange
}) => {
  const { dataSource } = useDataSource();
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // If props are provided, don't fetch from API
    if (propsTotalValue !== undefined) {
      setIsLoading(false);
      return;
    }
    
    let isMounted = true;
    setIsLoading(true);
    const fetchPortfolio = async () => {
      try {
        const data = dataSource === 'mock'
          ? await getMockPortfolio()
          : await portfolioApi.getPortfolio();
        if (isMounted) setPortfolio(data.portfolio);
      } catch (e) {
        if (isMounted) setPortfolio(null);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };
    fetchPortfolio();
    return () => { isMounted = false; };
  }, [dataSource, propsTotalValue]);
  // Determine whether to use props or API data
  const usePropsData = propsTotalValue !== undefined;
  
  // Loading state
  if (isLoading) {
    return (
      <div className="dashboard-widget col-span-1">
        <h2 className="text-lg font-semibold mb-3">Portfolio Summary</h2>
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-4"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-4"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3 mb-4"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-4"></div>
        </div>
      </div>
    );
  }

  // No data state
  if (!portfolio && !usePropsData) {
    return (
      <div className="dashboard-widget col-span-1">
        <h2 className="text-lg font-semibold mb-3">Portfolio Summary</h2>
        <div className="text-gray-500 dark:text-gray-400 text-center py-8 text-base font-medium">
          No portfolio data available
        </div>
      </div>
    );
  }

  // Get values either from props or from API data
  const totalValue = usePropsData 
    ? propsTotalValue! 
    : portfolio!.total_value;
    
  const availableCash = usePropsData
    ? propsAvailableCash!
    : portfolio!.cash;
    
  const totalPnl = usePropsData
    ? propsTotalPnl!
    : (portfolio!.daily_pnl || 0);
    
  const totalPnlPercentage = usePropsData
    ? propsTotalPnlPercentage!
    : (totalValue > 0 ? (totalPnl / (totalValue - totalPnl)) * 100 : 0);
  
  // Calculate positions value (only for API data)
  const positionsValue = !usePropsData && portfolio!.positions 
    ? Object.values(portfolio!.positions).reduce((sum, position) => sum + position.market_value, 0) 
    : (totalValue - availableCash);

  return (
    <div className="dashboard-widget col-span-1 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold">Portfolio Summary</h2>
        
        {/* Timeframe selector */}
        {onTimeframeChange && (
          <div className="flex text-sm rounded-md shadow-sm overflow-hidden">
            {(['1d', '1w', '1m', 'all'] as const).map((tf) => (
              <button
                key={tf}
                onClick={() => onTimeframeChange(tf)}
                className={`px-3 py-1 ${timeframe === tf 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'}`}
              >
                {tf === '1d' ? '1D' : tf === '1w' ? '1W' : tf === '1m' ? '1M' : 'All'}
              </button>
            ))}
          </div>
        )}
      </div>
      
      {/* Total Value */}
      <div className="mb-6">
        <h3 className="text-2xl font-bold">${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</h3>
        <div className={`text-sm ${totalPnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
          {totalPnl >= 0 ? '▲' : '▼'} ${Math.abs(totalPnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} 
          ({totalPnlPercentage.toFixed(2)}%) {timeframe === '1d' ? 'Today' : timeframe === '1w' ? 'This Week' : timeframe === '1m' ? 'This Month' : 'All Time'}
        </div>
      </div>
      
      {/* Asset Allocation */}
      <div className="space-y-3 mb-6">
        <div className="flex justify-between">
          <span className="text-gray-600 dark:text-gray-400">Cash</span>
          <div className="flex flex-col items-end">
            <span className="font-medium">${availableCash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            <span className="text-xs text-gray-500 dark:text-gray-400">{((availableCash / totalValue) * 100).toFixed(1)}%</span>
          </div>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-600 dark:text-gray-400">Positions</span>
          <div className="flex flex-col items-end">
            <span className="font-medium">${positionsValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            <span className="text-xs text-gray-500 dark:text-gray-400">{((positionsValue / totalValue) * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
      
      {/* Position Count */}
      <div className="flex justify-between mb-4">
        <span className="text-gray-600 dark:text-gray-400">Active Positions</span>
        <span className="font-medium">
          {!usePropsData && portfolio?.positions 
            ? Object.keys(portfolio.positions).length 
            : '—'}
        </span>
      </div>
      
      {/* Buying Power */}
      <div className="flex justify-between">
        <span className="text-gray-600 dark:text-gray-400">Buying Power</span>
        <span className="font-medium">
          ${(!usePropsData && portfolio 
            ? (portfolio.cash * (portfolio.margin_multiplier || 1)) 
            : availableCash * 2).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
      </div>
      
      <div className="mt-6 text-center">
        <Link to="/portfolio" className="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 font-medium">
          View Details →
        </Link>
      </div>
    </div>
  );
};

export default PortfolioSummary;
