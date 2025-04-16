import React, { useEffect, useMemo, useState } from 'react';
import { Portfolio } from '../../types';
import { useDataSource } from '../../context/DataSourceContext';
import { portfolioApi } from '../../api/portfolio';
import { getMockAssetAllocation } from '../../api/mockData/mockAssetAllocation';

// Check if @nivo/pie is installed
let ResponsivePie: any;
try {
  ResponsivePie = require('@nivo/pie').ResponsivePie;
} catch (e) {
  console.warn('Missing @nivo/pie dependency. Using fallback chart.');
}

interface AssetAllocationChartProps {
  onAssetSelect?: (symbol: string) => void;
  selectedAsset?: string;
}

const AssetAllocationChart: React.FC<AssetAllocationChartProps> = ({ onAssetSelect, selectedAsset }) => {
  const { dataSource } = useDataSource();
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);
    const fetchPortfolio = async () => {
      try {
        const data = dataSource === 'mock'
          ? await getMockAssetAllocation()
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
  // Generate chart data from portfolio positions
  const { chartData, totalValue, cashPercentage } = useMemo(() => {
    if (!portfolio) {
      return { chartData: [], totalValue: 0, cashPercentage: 0 };
    }
    
    const positions = portfolio.positions || {};
    let calculatedTotalValue = portfolio.total_value || 0;
    const cashValue = portfolio.cash || 0;
    const cashPct = (cashValue / calculatedTotalValue) * 100;
    
    const data = Object.entries(positions).map(([symbol, position]) => {
      const value = position.current_price * position.quantity;
      const percentage = (value / calculatedTotalValue) * 100;
      
      return {
        id: symbol,
        label: symbol,
        value: parseFloat(percentage.toFixed(2)),
        rawValue: value,
        color: getColorForAsset(symbol),
      };
    });
    
    // Add cash position if it exists
    if (cashValue > 0) {
      data.push({
        id: 'CASH',
        label: 'Cash',
        value: parseFloat(cashPct.toFixed(2)),
        rawValue: cashValue,
        color: '#A3A3A3', // Gray color for cash
      });
    }
    
    return { 
      chartData: data, 
      totalValue: calculatedTotalValue,
      cashPercentage: cashPct
    };
  }, [portfolio]);

  // Handle clicking on a chart slice
  const handlePieClick = (data: any) => {
    if (onAssetSelect && data.id !== 'CASH') {
      onAssetSelect(data.id);
    }
  };
  
  if (isLoading) {
    return (
      <div>
        <h2>Asset Allocation</h2>
        <div className="h-[300px] flex items-center justify-center">
          <div className="h-full w-full bg-gray-200 animate-pulse" />
        </div>
      </div>
    );
  }
  
  if (!portfolio || chartData.length === 0) {
    return (
      <div>
        <h2>Asset Allocation</h2>
        <div className="h-[300px] flex items-center justify-center">
          <p className="text-muted-foreground">No portfolio data available</p>
        </div>
      </div>
    );
  }
  
  // If @nivo/pie is not available, use a simple fallback
  if (!ResponsivePie) {
    return (
      <div>
        <h2 className="text-lg font-semibold mb-4">Asset Allocation</h2>
        <div className="h-[300px] flex flex-col items-center justify-center">
          <div className="grid grid-cols-2 gap-2 w-full">
            {chartData.map((asset) => (
              <div 
                key={asset.id}
                className="flex items-center p-2 rounded border cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800"
                onClick={() => asset.id !== 'CASH' && onAssetSelect && onAssetSelect(asset.id)}
              >
                <div 
                  className="w-4 h-4 mr-2 rounded-full" 
                  style={{ backgroundColor: asset.color }}
                />
                <div className="flex-1">
                  <div className="font-medium">{asset.label}</div>
                  <div className="text-gray-500 dark:text-gray-400 text-center py-8 text-base font-medium">{asset.value}%</div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="mt-4 text-center">
          <p className="text-sm text-muted-foreground">
            Total Portfolio Value: ${totalValue.toLocaleString(undefined, {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}
          </p>
          <p className="text-xs text-muted-foreground">
            Cash: {cashPercentage.toFixed(2)}%
          </p>
          {onAssetSelect && (
            <p className="text-xs text-muted-foreground mt-2">
              Click on an asset to select it for analysis
            </p>
          )}
        </div>
      </div>
    );
  }
  
  return (
    <div>
      <h2>Asset Allocation</h2>
      <div className="h-[300px]">
        <ResponsivePie
          data={chartData}
          margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
          innerRadius={0.5}
          padAngle={0.7}
          cornerRadius={3}
          activeOuterRadiusOffset={8}
          colors={{ datum: 'data.color' }}
          borderWidth={1}
          borderColor={{ from: 'color', modifiers: [['darker', 0.2]] }}
          arcLinkLabelsSkipAngle={10}
          arcLinkLabelsTextColor="#888888"
          arcLinkLabelsThickness={2}
          arcLinkLabelsColor={{ from: 'color' }}
          arcLabelsSkipAngle={10}
          arcLabelsTextColor={{ from: 'color', modifiers: [['darker', 2]] }}
          onClick={handlePieClick}
          tooltip={({ datum }: { datum: any }) => (
            <div
              style={{
                background: 'white',
                padding: '9px 12px',
                border: '1px solid #ccc',
                borderRadius: '4px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <span
                  style={{
                    display: 'block',
                    width: '12px',
                    height: '12px',
                    background: datum.color,
                    marginRight: '8px',
                  }}
                />
                <strong>{datum.id}</strong>
              </div>
              <div style={{ marginTop: '4px' }}>
                <div>{`${datum.value}% of portfolio`}</div>
                <div>{`$${(datum.data as any).rawValue.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}`}</div>
              </div>
            </div>
          )}
        />
      </div>
      <div className="mt-4 text-center">
        <p className="text-sm text-muted-foreground">
          Total Portfolio Value: ${totalValue.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          })}
        </p>
        <p className="text-xs text-muted-foreground">
          Cash: {cashPercentage.toFixed(2)}%
        </p>
        {onAssetSelect && (
          <p className="text-xs text-muted-foreground mt-2">
            Click on an asset to select it for analysis
          </p>
        )}
      </div>
    </div>
  );
};

// Helper function to get a consistent color for each asset
const getColorForAsset = (symbol: string): string => {
  const colors = [
    '#2196F3', // Blue
    '#4CAF50', // Green
    '#FFC107', // Amber
    '#9C27B0', // Purple
    '#F44336', // Red
    '#00BCD4', // Cyan
    '#FF9800', // Orange
    '#795548', // Brown
    '#607D8B', // Blue Grey
    '#E91E63', // Pink
    '#3F51B5', // Indigo
    '#CDDC39', // Lime
  ];
  
  // Simple hash function to get a consistent index for each symbol
  let hash = 0;
  for (let i = 0; i < symbol.length; i++) {
    hash = symbol.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  const index = Math.abs(hash) % colors.length;
  return colors[index];
};

export default AssetAllocationChart;
