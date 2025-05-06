import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, Card, CardContent, Grid, CircularProgress, Tooltip, Alert } from '@mui/material';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import { Chart, ChartConfiguration } from 'chart.js/auto';
import DrawdownMonitor from './DrawdownMonitor';
import TradeStatistics from './TradeStatistics';
import { usePerformanceData } from '../../../hooks/usePerformanceData';
import '../.././../styles/drawdownMonitor.css';
import '../.././../styles/tradeStatistics.css';

interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor?: number;
  average_win?: number;
  average_loss?: number;
  risk_reward_ratio?: number;
  recovery_factor?: number;
  volatility?: number;
  beta?: number;
  alpha?: number;
  sortino_ratio?: number;
  calmar_ratio?: number;
  max_consecutive_wins?: number;
  max_consecutive_losses?: number;
  profit_per_day?: number;
  current_drawdown?: number;
  drawdown_duration?: number;
  [key: string]: number | undefined;
}

interface PerformanceUpdate {
  timestamp: string;
  metrics: PerformanceMetrics;
  drawdown_data?: DrawdownData;
  trades?: Trade[];
  error?: string;
}

interface DrawdownData {
  timestamps: string[];
  equity: number[];
  drawdown: number[];
  underwater_periods: {
    start: number;
    end: number;
    depth: number;
    duration: number;
  }[];
}

interface Trade {
  id: string;
  symbol: string;
  entry_time: string;
  exit_time: string;
  entry_price: number;
  exit_price: number;
  quantity: number;
  pnl: number;
  pnl_percent: number;
  duration: number; // in minutes
  side: 'buy' | 'sell';
  status: 'open' | 'closed';
}

const PaperTradingPerformanceMetrics: React.FC = () => {
  const { state } = usePaperTrading();
  const radarChartRef = useRef<HTMLCanvasElement>(null);
  const radarChartInstance = useRef<Chart | null>(null);
  const [activeTab, setActiveTab] = useState<'metrics' | 'drawdown' | 'trades'>('metrics');
  
  // Use our custom hook to get performance data
  const {
    metrics: performanceMetrics,
    drawdownData,
    trades,
    tradeStatistics,
    isLoading,
    error
  } = usePerformanceData();

  // Initialize and update radar chart when metrics change
  useEffect(() => {
    if (!performanceMetrics) return;
    updateRadarChart();
  }, [performanceMetrics]);

  // Function to update the radar chart
  const updateRadarChart = () => {
    if (!radarChartRef.current || !performanceMetrics) return;

    // Destroy existing chart
    if (radarChartInstance.current) {
      radarChartInstance.current.destroy();
    }

    // Normalize metrics for radar chart (0-1 scale)
    const normalizeValue = (value: number, min: number, max: number) => {
      return Math.max(0, Math.min(1, (value - min) / (max - min)));
    };

    // Define metrics to show in radar chart with their normalization ranges
    const metricsConfig = [
      { key: 'total_return', label: 'Total Return', min: -0.1, max: 0.3 },
      { key: 'sharpe_ratio', label: 'Sharpe Ratio', min: -1, max: 3 },
      { key: 'win_rate', label: 'Win Rate', min: 0, max: 1 },
      { key: 'profit_factor', label: 'Profit Factor', min: 0, max: 3 },
      { key: 'max_drawdown', label: 'Max Drawdown (inv)', min: 0, max: 0.5 }
    ];

    // Prepare radar chart data
    const radarData = metricsConfig.map(config => {
      let value = 0;
      
      if (!performanceMetrics) return value;
      
      // Type-safe way to access metrics
      const getMetricValue = (key: string): number | undefined => {
        return (performanceMetrics as unknown as Record<string, number | undefined>)[key];
      };
      
      const metricValue = getMetricValue(config.key);
      
      if (config.key === 'max_drawdown' && metricValue !== undefined) {
        // Invert max drawdown (lower is better, but radar chart higher is better)
        value = normalizeValue(1 - Math.abs(metricValue), config.min, config.max);
      } else if (metricValue !== undefined) {
        value = normalizeValue(metricValue, config.min, config.max);
      }
      
      return value;
    });

    // Create radar chart configuration
    const radarChartConfig: ChartConfiguration = {
      type: 'radar',
      data: {
        labels: metricsConfig.map(config => config.label),
        datasets: [
          {
            label: 'Performance',
            data: radarData,
            fill: true,
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgb(75, 192, 192)',
            pointBackgroundColor: 'rgb(75, 192, 192)',
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: 'rgb(75, 192, 192)'
          }
        ]
      },
      options: {
        scales: {
          r: {
            angleLines: {
              display: true
            },
            suggestedMin: 0,
            suggestedMax: 1
          }
        }
      }
    };

    // Create new radar chart
    radarChartInstance.current = new Chart(radarChartRef.current, radarChartConfig);

    // Cleanup function
    return () => {
      if (radarChartInstance.current) {
        radarChartInstance.current.destroy();
        radarChartInstance.current = null;
      }
    };
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  // Format number
  const formatNumber = (value: number) => {
    return value.toFixed(4);
  };

  return (
    <div className="performance-metrics-container">
      <div className="metrics-tabs">
        <button 
          className={`tab-button ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          Metrics
        </button>
        <button 
          className={`tab-button ${activeTab === 'drawdown' ? 'active' : ''}`}
          onClick={() => setActiveTab('drawdown')}
        >
          Drawdown
        </button>
        <button 
          className={`tab-button ${activeTab === 'trades' ? 'active' : ''}`}
          onClick={() => setActiveTab('trades')}
        >
          Trades
        </button>
      </div>

      {isLoading ? (
        <div className="loading-container">
          <p>Loading performance data...</p>
        </div>
      ) : error ? (
        <div className="error-container">
          {error.includes('timeout') ? (
            <div>
              <p className="error-message">{error}</p>
              <p className="info-message">Showing demo data for preview purposes.</p>
            </div>
          ) : (
            <div>
              <p className="error-message">Error: Failed to connect to WebSocket</p>
              <p className="info-message">Showing demo data for preview purposes.</p>
            </div>
          )}
        </div>
      ) : activeTab === 'metrics' && performanceMetrics ? (
        <div className="metrics-content">
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Total Return</div>
              <div className={`metric-value ${performanceMetrics.total_return >= 0 ? 'positive' : 'negative'}`}>
                {formatPercentage(performanceMetrics.total_return)}
              </div>
            </div>
            
            <div className="metric-card">
              <div className="metric-label">Annualized Return</div>
              <div className={`metric-value ${performanceMetrics.annualized_return >= 0 ? 'positive' : 'negative'}`}>
                {formatPercentage(performanceMetrics.annualized_return)}
              </div>
            </div>
            
            <div className="metric-card">
              <div className="metric-label">Sharpe Ratio</div>
              <div className={`metric-value ${performanceMetrics.sharpe_ratio >= 1 ? 'positive' : 'neutral'}`}>
                {formatNumber(performanceMetrics.sharpe_ratio)}
              </div>
            </div>
            
            <div className="metric-card">
              <div className="metric-label">Max Drawdown</div>
              <div className="metric-value negative">
                {formatPercentage(performanceMetrics.max_drawdown)}
              </div>
            </div>
            
            <div className="metric-card">
              <div className="metric-label">Win Rate</div>
              <div className={`metric-value ${performanceMetrics.win_rate >= 0.5 ? 'positive' : 'neutral'}`}>
                {formatPercentage(performanceMetrics.win_rate)}
              </div>
            </div>
            
            {performanceMetrics.profit_factor !== undefined && (
              <div className="metric-card">
                <div className="metric-label">Profit Factor</div>
                <div className={`metric-value ${performanceMetrics.profit_factor >= 1 ? 'positive' : 'negative'}`}>
                  {formatNumber(performanceMetrics.profit_factor)}
                </div>
              </div>
            )}
            
            {performanceMetrics.sortino_ratio !== undefined && (
              <div className="metric-card">
                <div className="metric-label">Sortino Ratio</div>
                <div className={`metric-value ${performanceMetrics.sortino_ratio >= 1 ? 'positive' : 'neutral'}`}>
                  {formatNumber(performanceMetrics.sortino_ratio)}
                </div>
              </div>
            )}
            
            {performanceMetrics.calmar_ratio !== undefined && (
              <div className="metric-card">
                <div className="metric-label">Calmar Ratio</div>
                <div className={`metric-value ${performanceMetrics.calmar_ratio >= 1 ? 'positive' : 'neutral'}`}>
                  {formatNumber(performanceMetrics.calmar_ratio)}
                </div>
              </div>
            )}
          </div>
          
          <div className="radar-chart-container">
            <h4>Performance Radar</h4>
            <canvas ref={radarChartRef} />
          </div>
          
          <div className="additional-metrics">
            <h4>Additional Metrics</h4>
            <table>
              <tbody>
                {performanceMetrics.average_win !== undefined && (
                  <tr>
                    <td>Average Win</td>
                    <td>{formatPercentage(performanceMetrics.average_win)}</td>
                  </tr>
                )}
                
                {performanceMetrics.average_loss !== undefined && (
                  <tr>
                    <td>Average Loss</td>
                    <td>{formatPercentage(performanceMetrics.average_loss)}</td>
                  </tr>
                )}
                
                {performanceMetrics.risk_reward_ratio !== undefined && (
                  <tr>
                    <td>Risk/Reward Ratio</td>
                    <td>{formatNumber(performanceMetrics.risk_reward_ratio)}</td>
                  </tr>
                )}
                
                {performanceMetrics.recovery_factor !== undefined && (
                  <tr>
                    <td>Recovery Factor</td>
                    <td>{formatNumber(performanceMetrics.recovery_factor)}</td>
                  </tr>
                )}
                
                {error ? (
                  <Alert severity="info" sx={{ mt: 2, mb: 2 }}>
                    <strong>Demo Mode Active</strong> - Using simulated trading data for visualization. 
                    <br/>
                    The system will automatically connect to the backend when available.
                    <br/>
                    <small>{error}</small>
                  </Alert>
                ) : null}
                
                {performanceMetrics.volatility !== undefined && (
                  <tr>
                    <td>Volatility</td>
                    <td>{formatPercentage(performanceMetrics.volatility)}</td>
                  </tr>
                )}
                
                {performanceMetrics.max_consecutive_wins !== undefined && (
                  <tr>
                    <td>Max Consecutive Wins</td>
                    <td>{performanceMetrics.max_consecutive_wins}</td>
                  </tr>
                )}
                
                {performanceMetrics.max_consecutive_losses !== undefined && (
                  <tr>
                    <td>Max Consecutive Losses</td>
                    <td>{performanceMetrics.max_consecutive_losses}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      ) : activeTab === 'drawdown' && drawdownData ? (
        <DrawdownMonitor 
          drawdownData={drawdownData}
          maxDrawdown={performanceMetrics?.max_drawdown || 0}
        />
      ) : activeTab === 'trades' && trades ? (
        <TradeStatistics trades={trades} />
      ) : (
        <div className="no-data-container">
          <p>No data available for the selected tab.</p>
        </div>
      )}
    </div>
  );
};

export default PaperTradingPerformanceMetrics;
