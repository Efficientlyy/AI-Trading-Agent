import React, { useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { Portfolio } from '../../types';

interface AssetAllocationChartProps {
  portfolio: Portfolio | null;
  isLoading: boolean;
}

const AssetAllocationChart: React.FC<AssetAllocationChartProps> = ({ portfolio, isLoading }) => {
  // Generate chart data from portfolio positions
  const { chartData, totalValue, cashPercentage } = useMemo(() => {
    if (!portfolio) {
      return { chartData: [], totalValue: 0, cashPercentage: 0 };
    }

    const positions = Object.values(portfolio.positions || {});
    const positionValues = positions.map(position => ({
      name: position.symbol,
      value: position.market_value,
      quantity: position.quantity,
      price: position.current_price,
      pnl: position.unrealized_pnl,
      pnlPercentage: (position.unrealized_pnl / position.market_value) * 100
    }));

    // Sort by market value (descending)
    positionValues.sort((a, b) => b.value - a.value);

    // Add cash as a position
    const cashValue = portfolio.cash;
    const totalPortfolioValue = portfolio.total_value;
    const cashPercentage = (cashValue / totalPortfolioValue) * 100;

    // Only add cash if there's any
    const chartData = [...positionValues];
    if (cashValue > 0) {
      chartData.push({
        name: 'Cash',
        value: cashValue,
        quantity: cashValue,
        price: 1,
        pnl: 0,
        pnlPercentage: 0
      });
    }

    return { 
      chartData, 
      totalValue: totalPortfolioValue,
      cashPercentage
    };
  }, [portfolio]);

  // Pie chart colors
  const COLORS = [
    '#3b82f6', // blue
    '#10b981', // green
    '#8b5cf6', // purple
    '#f59e0b', // amber
    '#ef4444', // red
    '#ec4899', // pink
    '#6366f1', // indigo
    '#14b8a6', // teal
    '#f97316', // orange
    '#84cc16', // lime
    '#06b6d4', // cyan
    '#a855f7', // violet
  ];

  // Cash color is always gray
  const getCellColor = (entry: any, index: number) => {
    return entry.name === 'Cash' ? '#9ca3af' : COLORS[index % COLORS.length];
  };

  // Custom tooltip for the pie chart
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white dark:bg-gray-800 p-3 border border-gray-200 dark:border-gray-700 shadow-md rounded">
          <p className="font-medium">{data.name}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Value: ${data.value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Allocation: {((data.value / totalValue) * 100).toFixed(2)}%
          </p>
          {data.name !== 'Cash' && (
            <>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Quantity: {data.quantity.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 8 })}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Price: ${data.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 8 })}
              </p>
              <p className={`text-sm ${data.pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                P&L: ${data.pnl.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} 
                ({data.pnlPercentage >= 0 ? '+' : ''}{data.pnlPercentage.toFixed(2)}%)
              </p>
            </>
          )}
        </div>
      );
    }
    return null;
  };

  // Custom legend that includes percentage allocation
  const renderCustomizedLegend = (props: any) => {
    const { payload } = props;
    
    return (
      <ul className="text-xs space-y-1 mt-2">
        {payload.map((entry: any, index: number) => {
          const percentage = ((entry.payload.value / totalValue) * 100).toFixed(1);
          return (
            <li key={`item-${index}`} className="flex items-center justify-between">
              <div className="flex items-center">
                <div
                  className="w-3 h-3 mr-1"
                  style={{ backgroundColor: entry.color }}
                />
                <span className="truncate max-w-[100px]">{entry.value}</span>
              </div>
              <span>{percentage}%</span>
            </li>
          );
        })}
      </ul>
    );
  };

  if (isLoading) {
    return (
      <div className="dashboard-widget col-span-1">
        <h2 className="text-lg font-semibold mb-3">Asset Allocation</h2>
        <div className="animate-pulse h-64 bg-gray-200 dark:bg-gray-700 rounded"></div>
      </div>
    );
  }

  if (!portfolio || !chartData || chartData.length === 0) {
    return (
      <div className="dashboard-widget col-span-1">
        <h2 className="text-lg font-semibold mb-3">Asset Allocation</h2>
        <div className="text-gray-500 dark:text-gray-400 text-center py-24">
          No assets in portfolio
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-widget col-span-1">
      <h2 className="text-lg font-semibold mb-3">Asset Allocation</h2>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={80}
              innerRadius={40}
              paddingAngle={2}
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getCellColor(entry, index)} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend content={renderCustomizedLegend} />
          </PieChart>
        </ResponsiveContainer>
      </div>
      
      {/* Summary statistics */}
      <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
        <div className="bg-gray-50 dark:bg-gray-800 p-2 rounded">
          <div className="text-gray-500 dark:text-gray-400">Total Value</div>
          <div className="font-medium">${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-800 p-2 rounded">
          <div className="text-gray-500 dark:text-gray-400">Cash Allocation</div>
          <div className="font-medium">{cashPercentage.toFixed(1)}%</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-800 p-2 rounded">
          <div className="text-gray-500 dark:text-gray-400">Assets</div>
          <div className="font-medium">{chartData.length - (cashPercentage > 0 ? 1 : 0)}</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-800 p-2 rounded">
          <div className="text-gray-500 dark:text-gray-400">Diversification</div>
          <div className="font-medium">
            {chartData.length <= 1 ? 'None' : 
             chartData.length <= 3 ? 'Low' : 
             chartData.length <= 6 ? 'Medium' : 'High'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AssetAllocationChart;
