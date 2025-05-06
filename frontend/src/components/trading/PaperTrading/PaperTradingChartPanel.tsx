import React, { useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Card, CardHeader, CardTitle, CardContent } from '../../../components/ui/Card';
import { Spinner } from '../../../components/common/Spinner';
import { paperTradingApi } from '../../../api/paperTrading';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const PaperTradingChartPanel: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  
  // Fetch results if session is completed
  const {
    data: resultsData,
    isLoading: resultsLoading,
    error: resultsError
  } = useQuery({
    queryKey: ['paperTradingResults', sessionId],
    queryFn: () => sessionId ? paperTradingApi.getResults(sessionId) : null,
    enabled: !!sessionId
  });

  // Prepare chart data
  const portfolioChartData = useMemo(() => {
    if (!resultsData?.portfolio_history?.length) {
      return null;
    }

    const labels = resultsData.portfolio_history.map((item: any) => {
      return new Date(item.timestamp).toLocaleTimeString();
    });

    const portfolioValues = resultsData.portfolio_history.map((item: any) => {
      return item.total_value;
    });

    return {
      labels,
      datasets: [
        {
          label: 'Portfolio Value',
          data: portfolioValues,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          tension: 0.1,
        }
      ]
    };
  }, [resultsData]);

  // Chart options
  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Portfolio Value Over Time',
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      }
    }
  };

  // Prepare trade distribution data
  const tradeDistributionData = useMemo(() => {
    if (!resultsData?.trades?.length) {
      return null;
    }

    // Count trades by symbol
    const tradesBySymbol: Record<string, { buy: number, sell: number }> = {};
    
    resultsData.trades.forEach((trade: any) => {
      if (!tradesBySymbol[trade.symbol]) {
        tradesBySymbol[trade.symbol] = { buy: 0, sell: 0 };
      }
      
      if (trade.side === 'buy') {
        tradesBySymbol[trade.symbol].buy += 1;
      } else {
        tradesBySymbol[trade.symbol].sell += 1;
      }
    });

    const labels = Object.keys(tradesBySymbol);
    const buyData = labels.map(symbol => tradesBySymbol[symbol].buy);
    const sellData = labels.map(symbol => tradesBySymbol[symbol].sell);

    return {
      labels,
      datasets: [
        {
          label: 'Buy Trades',
          data: buyData,
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
        },
        {
          label: 'Sell Trades',
          data: sellData,
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
        }
      ]
    };
  }, [resultsData]);

  if (!sessionId) {
    return (
      <Card>
        <CardContent className="p-6 text-center">
          <p className="text-gray-500">No session selected</p>
        </CardContent>
      </Card>
    );
  }

  if (resultsLoading) {
    return (
      <Card>
        <CardContent className="p-6 flex justify-center">
          <Spinner />
        </CardContent>
      </Card>
    );
  }

  if (resultsError) {
    return (
      <Card>
        <CardContent className="p-6 text-center">
          <p className="text-red-500">Error loading results data</p>
        </CardContent>
      </Card>
    );
  }

  if (!resultsData || !resultsData.portfolio_history?.length) {
    return (
      <Card>
        <CardContent className="p-6 text-center">
          <p className="text-gray-500">No chart data available for this session</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Performance Charts</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Portfolio Value Chart */}
          <div className="h-64">
            {portfolioChartData ? (
              <Line data={portfolioChartData} options={chartOptions} />
            ) : (
              <div className="flex items-center justify-center h-full">
                <p className="text-gray-500">No portfolio data available</p>
              </div>
            )}
          </div>
          
          {/* Additional performance metrics */}
          {resultsData.performance_metrics && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-md">
                <div className="text-sm text-gray-500 dark:text-gray-400">Total Return</div>
                <div className={`text-lg font-semibold ${
                  resultsData.performance_metrics.total_return >= 0 
                    ? 'text-green-600' 
                    : 'text-red-600'
                }`}>
                  {(resultsData.performance_metrics.total_return * 100).toFixed(2)}%
                </div>
              </div>
              
              <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-md">
                <div className="text-sm text-gray-500 dark:text-gray-400">Sharpe Ratio</div>
                <div className="text-lg font-semibold">
                  {resultsData.performance_metrics.sharpe_ratio?.toFixed(2) || 'N/A'}
                </div>
              </div>
              
              <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-md">
                <div className="text-sm text-gray-500 dark:text-gray-400">Max Drawdown</div>
                <div className="text-lg font-semibold text-red-600">
                  {(resultsData.performance_metrics.max_drawdown * 100).toFixed(2)}%
                </div>
              </div>
              
              <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-md">
                <div className="text-sm text-gray-500 dark:text-gray-400">Win Rate</div>
                <div className="text-lg font-semibold">
                  {(resultsData.performance_metrics.win_rate * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default PaperTradingChartPanel;