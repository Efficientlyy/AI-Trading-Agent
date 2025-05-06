import React, { useEffect, useRef, useState } from 'react';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import { Chart, ChartConfiguration } from 'chart.js/auto';
import { format } from 'date-fns';
import { WebSocketTopic } from '../../../services/WebSocketService';

// Define types for portfolio data
interface Position {
  symbol: string;
  quantity: number;
  average_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
}

interface PortfolioData {
  cash: number;
  positions: Record<string, number>;
  position_values: Record<string, number>;
  total_value: number;
}

interface PortfolioHistoryPoint {
  timestamp: string;
  totalValue: number;
  cash: number;
}

const PaperTradingPortfolioMonitor: React.FC = () => {
  const { state, webSocketService } = usePaperTrading();
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  const [portfolioHistory, setPortfolioHistory] = useState<PortfolioHistoryPoint[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [currentPortfolio, setCurrentPortfolio] = useState<PortfolioData | null>(null);

  // WebSocket connection and event handlers
  useEffect(() => {
    if (state.activeSessions.length > 0 && webSocketService) {
      // Connect to WebSocket
      webSocketService.connect();
      setIsConnected(true);

      const handlePortfolioUpdate = (data: any) => {
        if (data.portfolio) {
          setCurrentPortfolio(data.portfolio);

          // Add new portfolio data point
          const newDataPoint: PortfolioHistoryPoint = {
            timestamp: new Date().toISOString(),
            totalValue: data.portfolio.total_value,
            cash: data.portfolio.cash
          };

          setPortfolioHistory(prev => {
            // Keep only the last 100 data points to avoid performance issues
            const newHistory = [...prev, newDataPoint];
            if (newHistory.length > 100) {
              return newHistory.slice(-100);
            }
            return newHistory;
          });
        }
      };

      const handleStatusUpdate = (data: any) => {
        if (data.current_portfolio) {
          setCurrentPortfolio(data.current_portfolio);
        }
      };

      // Register event handlers
      webSocketService.on(WebSocketTopic.PORTFOLIO, handlePortfolioUpdate);
      webSocketService.on(WebSocketTopic.STATUS, handleStatusUpdate);

      // Use context data if available
      if (state.wsData?.portfolio) {
        handlePortfolioUpdate({ portfolio: state.wsData.portfolio });
      }

      // Cleanup function
      return () => {
        webSocketService.off(WebSocketTopic.PORTFOLIO, handlePortfolioUpdate);
        webSocketService.off(WebSocketTopic.STATUS, handleStatusUpdate);
        
        // Disconnect from WebSocket
        webSocketService.disconnect();
        setIsConnected(false);
      };
    }
  }, [state.activeSessions.length, webSocketService, state.wsData]);

  // Initialize and update chart
  useEffect(() => {
    if (!chartRef.current) return;

    // Destroy existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    // Prepare chart data
    const labels = portfolioHistory.map((data) => {
      const date = new Date(data.timestamp);
      return format(date, 'HH:mm:ss');
    });

    const totalValues = portfolioHistory.map((data) => data.totalValue);
    const cashValues = portfolioHistory.map((data) => data.cash);

    // Create chart configuration
    const chartConfig: ChartConfiguration = {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Total Portfolio Value',
            data: totalValues,
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: true,
            tension: 0.1
          },
          {
            label: 'Cash',
            data: cashValues,
            borderColor: 'rgb(153, 102, 255)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            fill: true,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: {
            title: {
              display: true,
              text: 'Time'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Value'
            },
            beginAtZero: false
          }
        },
        animation: {
          duration: 0 // Disable animation for better performance
        }
      }
    };

    // Create new chart
    chartInstance.current = new Chart(chartRef.current, chartConfig);

    // Cleanup function
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [portfolioHistory]);

  // Calculate performance metrics
  const initialValue = portfolioHistory.length > 0 ? portfolioHistory[0].totalValue : 0;
  const currentValue = portfolioHistory.length > 0 ? portfolioHistory[portfolioHistory.length - 1].totalValue : 0;
  const absoluteChange = currentValue - initialValue;
  const percentChange = initialValue > 0 ? (absoluteChange / initialValue) * 100 : 0;

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  return (
    <div className="paper-trading-portfolio-monitor">
      <div className="monitor-header">
        <h3>Portfolio Monitor</h3>
        <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>
      
      <div className="portfolio-summary">
        <div className="summary-item">
          <div className="label">Total Value</div>
          <div className="value">{currentPortfolio ? formatCurrency(currentPortfolio.total_value) : 'N/A'}</div>
        </div>
        <div className="summary-item">
          <div className="label">Cash</div>
          <div className="value">{currentPortfolio ? formatCurrency(currentPortfolio.cash) : 'N/A'}</div>
        </div>
        <div className="summary-item">
          <div className="label">Change</div>
          <div className={`value ${absoluteChange >= 0 ? 'positive' : 'negative'}`}>
            {formatCurrency(absoluteChange)} ({percentChange.toFixed(2)}%)
          </div>
        </div>
      </div>
      
      <div className="positions-table">
        <h4>Current Positions</h4>
        {currentPortfolio && Object.keys(currentPortfolio.positions).length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Quantity</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(currentPortfolio.positions).map(([symbol, quantity]) => (
                <tr key={symbol}>
                  <td>{symbol}</td>
                  <td>{quantity}</td>
                  <td>{formatCurrency(currentPortfolio.position_values[symbol] || 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="no-data">No positions</div>
        )}
      </div>
      
      <div className="chart-container">
        <h4>Portfolio Value History</h4>
        {portfolioHistory.length === 0 ? (
          <div className="no-data">Waiting for portfolio data...</div>
        ) : (
          <canvas ref={chartRef} />
        )}
      </div>
    </div>
  );
};

export default PaperTradingPortfolioMonitor;
