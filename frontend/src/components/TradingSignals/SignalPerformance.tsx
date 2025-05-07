import React, { useState, useEffect } from 'react';
import { tradingSignalsApi, PerformanceMetrics } from '../../api/tradingSignals';
import { Card, CardHeader, CardBody } from '../Card';
import { Select, Spinner } from '../Form';
import { Alert } from '../Alert';
import { formatPercent } from '../../utils/formatters';

interface SignalPerformanceProps {
  className?: string;
}

/**
 * Signal Performance Component
 * 
 * Displays performance metrics for trading signals from different strategies
 */
const SignalPerformance: React.FC<SignalPerformanceProps> = ({ className = '' }) => {
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [strategyType, setStrategyType] = useState<string>('enhanced');
  const [daysBack, setDaysBack] = useState<number>(30);
  
  useEffect(() => {
    fetchPerformance();
  }, [strategyType, daysBack]);
  
  const fetchPerformance = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await tradingSignalsApi.getSignalPerformance(strategyType, daysBack);
      setPerformance(data);
    } catch (err) {
      setError('Failed to fetch performance metrics. Please try again.');
      console.error('Error fetching performance:', err);
    } finally {
      setLoading(false);
    }
  };
  
  // Function to render a metric card
  const renderMetricCard = (label: string, value: number | string, formatter?: (val: number) => string) => {
    const formattedValue = typeof value === 'number' && formatter ? formatter(value) : value;
    
    return (
      <div className="bg-white rounded-lg shadow p-4 text-center">
        <div className="text-sm text-gray-500 mb-1">{label}</div>
        <div className="text-xl font-bold">{formattedValue}</div>
      </div>
    );
  };
  
  return (
    <Card className={className}>
      <CardHeader>
        <h2 className="text-xl font-semibold">Signal Performance</h2>
        <div className="flex space-x-4">
          <div>
            <Select
              value={strategyType}
              onChange={(e) => setStrategyType(e.target.value)}
              className="text-sm"
            >
              <option value="enhanced">Enhanced Sentiment</option>
              <option value="trend">Sentiment Trend</option>
              <option value="divergence">Sentiment Divergence</option>
              <option value="shock">Sentiment Shock</option>
            </Select>
          </div>
          <div>
            <Select
              value={daysBack.toString()}
              onChange={(e) => setDaysBack(parseInt(e.target.value))}
              className="text-sm"
            >
              <option value="7">Last 7 days</option>
              <option value="30">Last 30 days</option>
              <option value="90">Last 90 days</option>
              <option value="180">Last 180 days</option>
              <option value="365">Last year</option>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardBody>
        {loading ? (
          <div className="flex justify-center py-8">
            <Spinner size="lg" />
          </div>
        ) : error ? (
          <Alert type="error">{error}</Alert>
        ) : !performance ? (
          <div className="text-center py-8 text-gray-500">
            No performance data available.
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {renderMetricCard('Win Rate', performance.win_rate, formatPercent)}
            {renderMetricCard('Profit Factor', performance.profit_factor)}
            {renderMetricCard('Sharpe Ratio', performance.sharpe_ratio)}
            {renderMetricCard('Max Drawdown', performance.max_drawdown, formatPercent)}
            {renderMetricCard('Total Return', performance.total_return, formatPercent)}
            {renderMetricCard('Avg Return/Trade', performance.avg_return_per_trade, formatPercent)}
            {renderMetricCard('Number of Trades', performance.num_trades)}
            
            {/* Render any additional metrics that might be in the response */}
            {Object.entries(performance)
              .filter(([key]) => !['win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown', 
                               'total_return', 'avg_return_per_trade', 'num_trades'].includes(key))
              .map(([key, value]) => {
                // Format the key for display
                const formattedKey = key
                  .split('_')
                  .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                  .join(' ');
                
                // Determine if the value is a percentage
                const isPercent = key.includes('rate') || 
                                 key.includes('return') || 
                                 key.includes('drawdown');
                
                return renderMetricCard(
                  formattedKey, 
                  value as number, 
                  isPercent ? formatPercent : undefined
                );
              })}
          </div>
        )}
      </CardBody>
    </Card>
  );
};

export default SignalPerformance;
