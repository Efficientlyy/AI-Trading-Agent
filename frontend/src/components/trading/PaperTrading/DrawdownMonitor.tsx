import React, { useEffect, useRef, useState } from 'react';
import { Chart, ChartConfiguration } from 'chart.js/auto';

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

interface DrawdownMonitorProps {
  drawdownData: DrawdownData | null;
  maxDrawdown: number;
}

const DrawdownMonitor: React.FC<DrawdownMonitorProps> = ({ drawdownData, maxDrawdown }) => {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<number | null>(null);

  // Initialize and update chart
  useEffect(() => {
    if (!chartRef.current || !drawdownData) return;

    // Destroy existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    // Format dates for display
    const formattedDates = drawdownData.timestamps.map(ts => {
      const date = new Date(ts);
      return date.toLocaleDateString();
    });

    // Create gradient for drawdown
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(255, 99, 132, 0.1)');
    gradient.addColorStop(1, 'rgba(255, 99, 132, 0.4)');

    // Create chart
    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: formattedDates,
        datasets: [
          {
            label: 'Drawdown',
            data: drawdownData.drawdown.map(d => d * 100), // Convert to percentage
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: gradient,
            fill: true,
            tension: 0.4,
            yAxisID: 'y'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              label: (context) => {
                return `Drawdown: ${context.raw}%`;
              }
            }
          },
          annotation: {
            annotations: {
              maxDrawdownLine: {
                type: 'line',
                yMin: maxDrawdown * 100,
                yMax: maxDrawdown * 100,
                borderColor: 'rgba(255, 0, 0, 0.7)',
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                  content: `Max Drawdown: ${(maxDrawdown * 100).toFixed(2)}%`,
                  enabled: true,
                  position: 'start'
                }
              },
              ...drawdownData.underwater_periods.map((period, index) => ({
                [`period${index}`]: {
                  type: 'box',
                  xMin: period.start,
                  xMax: period.end,
                  backgroundColor: 'rgba(255, 0, 0, 0.1)',
                  borderColor: 'rgba(255, 0, 0, 0.3)',
                  borderWidth: 1,
                  click: () => setSelectedPeriod(index)
                }
              }))
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Date'
            },
            ticks: {
              maxTicksLimit: 10
            }
          },
          y: {
            title: {
              display: true,
              text: 'Drawdown (%)'
            },
            suggestedMin: maxDrawdown * 100 * 1.2, // 20% below max drawdown
            suggestedMax: 0,
            ticks: {
              callback: (value) => `${value}%`
            }
          }
        }
      }
    } as ChartConfiguration);

  }, [drawdownData, maxDrawdown]);

  // Render underwater periods table
  const renderUnderwaterPeriodsTable = () => {
    if (!drawdownData || !drawdownData.underwater_periods.length) {
      return (
        <div className="no-data">
          No significant drawdown periods detected
        </div>
      );
    }

    return (
      <div className="underwater-periods">
        <h4>Significant Drawdown Periods</h4>
        <table>
          <thead>
            <tr>
              <th>Start</th>
              <th>End</th>
              <th>Depth</th>
              <th>Duration</th>
            </tr>
          </thead>
          <tbody>
            {drawdownData.underwater_periods.map((period, index) => (
              <tr 
                key={index} 
                className={selectedPeriod === index ? 'selected' : ''}
                onClick={() => setSelectedPeriod(index)}
              >
                <td>{new Date(drawdownData.timestamps[period.start]).toLocaleDateString()}</td>
                <td>
                  {period.end < drawdownData.timestamps.length 
                    ? new Date(drawdownData.timestamps[period.end]).toLocaleDateString() 
                    : 'Ongoing'}
                </td>
                <td className="negative">{(period.depth * 100).toFixed(2)}%</td>
                <td>{period.duration} days</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Render drawdown statistics
  const renderDrawdownStats = () => {
    if (!drawdownData) return null;

    // Calculate recovery time (average days to recover from drawdown)
    const recoveryTimes = drawdownData.underwater_periods
      .filter(p => p.end < drawdownData.timestamps.length) // Only completed periods
      .map(p => p.duration);
    
    const avgRecoveryTime = recoveryTimes.length 
      ? recoveryTimes.reduce((sum, time) => sum + time, 0) / recoveryTimes.length 
      : 0;

    // Calculate frequency (average days between drawdown periods)
    let totalDays = 0;
    if (drawdownData.underwater_periods.length >= 2) {
      for (let i = 1; i < drawdownData.underwater_periods.length; i++) {
        const prevEnd = drawdownData.underwater_periods[i-1].end;
        const currStart = drawdownData.underwater_periods[i].start;
        totalDays += currStart - prevEnd;
      }
    }
    
    const avgFrequency = drawdownData.underwater_periods.length >= 2
      ? totalDays / (drawdownData.underwater_periods.length - 1)
      : 0;

    return (
      <div className="drawdown-stats">
        <h4>Drawdown Statistics</h4>
        <div className="stats-grid">
          <div className="stat-item">
            <div className="stat-label">Max Drawdown</div>
            <div className="stat-value negative">{(maxDrawdown * 100).toFixed(2)}%</div>
          </div>
          <div className="stat-item">
            <div className="stat-label">Avg Recovery Time</div>
            <div className="stat-value">{avgRecoveryTime.toFixed(1)} days</div>
          </div>
          <div className="stat-item">
            <div className="stat-label">Drawdown Frequency</div>
            <div className="stat-value">
              {avgFrequency > 0 ? `Every ${avgFrequency.toFixed(1)} days` : 'N/A'}
            </div>
          </div>
          <div className="stat-item">
            <div className="stat-label">Current Drawdown</div>
            <div className="stat-value negative">
              {drawdownData.drawdown.length > 0 
                ? `${(drawdownData.drawdown[drawdownData.drawdown.length - 1] * 100).toFixed(2)}%` 
                : '0.00%'}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="drawdown-monitor">
      <div className="drawdown-header">
        <h3>Drawdown Monitor</h3>
      </div>
      
      {!drawdownData ? (
        <div className="loading-message">
          Waiting for drawdown data...
        </div>
      ) : (
        <div className="drawdown-content">
          {renderDrawdownStats()}
          
          <div className="drawdown-chart-container">
            <canvas ref={chartRef} />
          </div>
          
          {renderUnderwaterPeriodsTable()}
        </div>
      )}
    </div>
  );
};

export default DrawdownMonitor;
