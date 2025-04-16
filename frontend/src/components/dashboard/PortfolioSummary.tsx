import React, { useEffect, useState } from 'react';
import { Portfolio } from '../../types';
import { useDataSource } from '../../context/DataSourceContext';
import { portfolioApi } from '../../api/portfolio';
import { getMockPortfolio } from '../../api/mockData/mockPortfolio';

const PortfolioSummary: React.FC = () => {
  const { dataSource } = useDataSource();
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
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
  }, [dataSource]);
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

  if (!portfolio) {
    return (
      <div className="dashboard-widget col-span-1">
        <h2 className="text-lg font-semibold mb-3">Portfolio Summary</h2>
        <div className="text-gray-500 dark:text-gray-400 text-center py-8 text-base font-medium">
          No portfolio data available
        </div>
      </div>
    );
  }

  // Calculate total positions value
  const positionsValue = portfolio.positions 
    ? Object.values(portfolio.positions).reduce((sum, position) => sum + position.market_value, 0) 
    : 0;

  // Calculate daily change
  const dailyChangeValue = portfolio.daily_pnl || 0;
  const dailyChangePercent = portfolio.total_value > 0 
    ? (dailyChangeValue / (portfolio.total_value - dailyChangeValue)) * 100 
    : 0;

  return (
    <div className="dashboard-widget col-span-1">
      <h2 className="text-lg font-semibold mb-3">Portfolio Summary</h2>
      
      {/* Total Value */}
      <div className="mb-4">
        <h3 className="text-2xl font-bold">${portfolio.total_value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</h3>
        <div className={`text-sm ${dailyChangeValue >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
          {dailyChangeValue >= 0 ? '▲' : '▼'} ${Math.abs(dailyChangeValue).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} 
          ({dailyChangePercent.toFixed(2)}%) Today
        </div>
      </div>
      
      {/* Asset Allocation */}
      <div className="space-y-2 mb-4">
        <div className="flex justify-between">
          <span className="text-gray-600 dark:text-gray-400">Cash</span>
          <div className="flex flex-col items-end">
            <span className="font-medium">${portfolio.cash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            <span className="text-xs text-gray-500 dark:text-gray-400">{((portfolio.cash / portfolio.total_value) * 100).toFixed(1)}%</span>
          </div>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-600 dark:text-gray-400">Positions</span>
          <div className="flex flex-col items-end">
            <span className="font-medium">${positionsValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            <span className="text-xs text-gray-500 dark:text-gray-400">{((positionsValue / portfolio.total_value) * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
      
      {/* Position Count */}
      <div className="flex justify-between mb-4">
        <span className="text-gray-600 dark:text-gray-400">Active Positions</span>
        <span className="font-medium">{portfolio.positions ? Object.keys(portfolio.positions).length : 0}</span>
      </div>
      
      {/* Buying Power */}
      <div className="flex justify-between">
        <span className="text-gray-600 dark:text-gray-400">Buying Power</span>
        <span className="font-medium">${(portfolio.cash * (portfolio.margin_multiplier || 1)).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
      </div>
      
      <div className="mt-4">
        <button className="text-sm text-primary hover:text-primary-dark">View Details →</button>
      </div>
    </div>
  );
};

export default PortfolioSummary;
