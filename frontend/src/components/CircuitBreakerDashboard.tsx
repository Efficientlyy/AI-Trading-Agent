import React, { useEffect, useState } from 'react';
import { getApiHealthDashboard } from '../api/utils/monitoring';

/**
 * Circuit Breaker Dashboard Component
 * Displays the current state of all circuit breakers in the system
 */
const CircuitBreakerDashboard: React.FC = () => {
  const [dashboard, setDashboard] = useState<any>({
    exchanges: {},
    overallHealth: true,
    totalCalls: 0,
    successRate: 1,
    averageDuration: 0
  });
  const [refreshInterval, setRefreshInterval] = useState<number>(5000);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  // Function to reload dashboard data
  const reloadDashboard = () => {
    const data = getApiHealthDashboard();
    setDashboard(data);
    setLastUpdated(new Date());
  };

  // Reload data on component mount and when refresh interval changes
  useEffect(() => {
    reloadDashboard();

    const intervalId = setInterval(() => {
      reloadDashboard();
    }, refreshInterval);

    return () => clearInterval(intervalId);
  }, [refreshInterval]);

  // Get status color based on circuit breaker state
  const getStatusColor = (state: string): string => {
    switch (state) {
      case 'closed':
        return 'green';
      case 'half-open':
        return 'orange';
      case 'open':
        return 'red';
      default:
        return 'gray';
    }
  };

  // Format percentage
  const formatPercentage = (value: number): string => {
    return (value * 100).toFixed(2) + '%';
  };

  // Format time
  const formatTime = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  // Format date
  const formatDate = (date: Date): string => {
    return date.toLocaleTimeString();
  };

  return (
    <div className="circuit-breaker-dashboard">
      <div className="dashboard-header">
        <h2>API Circuit Breaker Dashboard</h2>
        <div className="dashboard-controls">
          <label>
            Refresh every:
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
            >
              <option value={1000}>1 second</option>
              <option value={5000}>5 seconds</option>
              <option value={10000}>10 seconds</option>
              <option value={30000}>30 seconds</option>
            </select>
          </label>
          <button onClick={reloadDashboard}>Refresh Now</button>
          <span>Last updated: {formatDate(lastUpdated)}</span>
        </div>
      </div>

      <div className="dashboard-overview">
        <div className="overview-card">
          <h3>Overall Health</h3>
          <div className={`health-indicator ${dashboard.overallHealth ? 'healthy' : 'unhealthy'}`}>
            {dashboard.overallHealth ? 'Healthy' : 'Unhealthy'}
          </div>
        </div>

        <div className="overview-card">
          <h3>Total API Calls</h3>
          <div className="metric">{dashboard.totalCalls}</div>
        </div>

        <div className="overview-card">
          <h3>Success Rate</h3>
          <div className="metric">{formatPercentage(dashboard.successRate)}</div>
        </div>

        <div className="overview-card">
          <h3>Average Response Time</h3>
          <div className="metric">{formatTime(dashboard.averageDuration)}</div>
        </div>
      </div>

      <div className="exchanges-container">
        {Object.keys(dashboard.exchanges).map((exchangeName) => {
          const exchange = dashboard.exchanges[exchangeName];
          return (
            <div key={exchangeName} className="exchange-card">
              <h3>{exchangeName}</h3>
              <div className="exchange-metrics">
                <div className="exchange-metric">
                  <span>Success Rate:</span>
                  <span>{formatPercentage(exchange.successRate)}</span>
                </div>
                <div className="exchange-metric">
                  <span>Total Calls:</span>
                  <span>{exchange.totalCalls}</span>
                </div>
                <div className="exchange-metric">
                  <span>Average Response Time:</span>
                  <span>{formatTime(exchange.averageDuration || 0)}</span>
                </div>
              </div>

              <h4>Circuit Breakers</h4>
              <div className="circuit-breakers-table">
                <table>
                  <thead>
                    <tr>
                      <th>Method</th>
                      <th>State</th>
                      <th>Failure Count</th>
                      <th>Remaining Time</th>
                      <th>Success Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {exchange.circuitBreakerStates &&
                      Object.keys(exchange.circuitBreakerStates).map((method) => {
                        const cbState = exchange.circuitBreakerStates[method];
                        const metrics = exchange.methods && exchange.methods[method] ?
                          exchange.methods[method] : { successRate: 1 };

                        return (
                          <tr key={method}>
                            <td>{method}</td>
                            <td>
                              <span
                                className="circuit-state"
                                style={{ backgroundColor: getStatusColor(cbState.state) }}
                              >
                                {cbState.state}
                              </span>
                            </td>
                            <td>{cbState.failureCount}</td>
                            <td>{formatTime(cbState.remainingTimeMs)}</td>
                            <td>{formatPercentage(metrics.successRate)}</td>
                          </tr>
                        );
                      })
                    }
                    {(!exchange.circuitBreakerStates ||
                      Object.keys(exchange.circuitBreakerStates).length === 0) && (
                        <tr>
                          <td colSpan={5}>No active circuit breakers</td>
                        </tr>
                      )}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })}

        {Object.keys(dashboard.exchanges).length === 0 && (
          <div className="no-data">No exchange data available</div>
        )}
      </div>

      {/* Convert inline CSS to styled-jsx compatible format */}
      <style dangerouslySetInnerHTML={{
        __html: `
        .circuit-breaker-dashboard {
          font-family: system-ui, -apple-system, sans-serif;
          padding: 20px;
          background-color: #f5f7fa;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          border-bottom: 1px solid #e1e4e8;
          padding-bottom: 15px;
        }
        
        .dashboard-controls {
          display: flex;
          gap: 15px;
          align-items: center;
        }
        
        .dashboard-controls select, .dashboard-controls button {
          padding: 6px 12px;
          border-radius: 4px;
          border: 1px solid #d1d5da;
          background-color: white;
          cursor: pointer;
        }
        
        .dashboard-controls button {
          background-color: #0366d6;
          color: white;
          border: none;
        }
        
        .dashboard-overview {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 15px;
          margin-bottom: 25px;
        }
        
        .overview-card {
          background-color: white;
          border-radius: 6px;
          padding: 15px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
          text-align: center;
        }
        
        .overview-card h3 {
          margin: 0 0 10px;
          color: #24292e;
          font-size: 14px;
        }
        
        .health-indicator {
          font-size: 18px;
          font-weight: bold;
          padding: 6px;
          border-radius: 4px;
        }
        
        .health-indicator.healthy {
          color: green;
          background-color: rgba(46, 204, 113, 0.1);
        }
        
        .health-indicator.unhealthy {
          color: red;
          background-color: rgba(231, 76, 60, 0.1);
        }
        
        .metric {
          font-size: 20px;
          font-weight: bold;
          color: #0366d6;
        }
        
        .exchanges-container {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
          gap: 20px;
        }
        
        .exchange-card {
          background-color: white;
          border-radius: 6px;
          padding: 20px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .exchange-card h3 {
          margin: 0 0 15px;
          color: #24292e;
          border-bottom: 1px solid #e1e4e8;
          padding-bottom: 10px;
        }
        
        .exchange-metrics {
          display: flex;
          justify-content: space-between;
          margin-bottom: 20px;
          flex-wrap: wrap;
        }
        
        .exchange-metric {
          margin-bottom: 5px;
          flex-basis: 30%;
        }
        
        .exchange-metric span:first-child {
          font-weight: 500;
          margin-right: 5px;
          color: #586069;
        }
        
        .exchange-metric span:last-child {
          font-weight: bold;
          color: #24292e;
        }
        
        .circuit-breakers-table {
          overflow-x: auto;
        }
        
        table {
          width: 100%;
          border-collapse: collapse;
        }
        
        th, td {
          padding: 8px 12px;
          text-align: left;
          border-bottom: 1px solid #e1e4e8;
        }
        
        th {
          font-weight: 600;
          color: #586069;
          background-color: #f6f8fa;
        }
        
        .circuit-state {
          display: inline-block;
          padding: 4px 8px;
          border-radius: 12px;
          color: white;
          text-transform: uppercase;
          font-size: 12px;
          font-weight: bold;
        }
        
        .no-data {
          text-align: center;
          padding: 30px;
          background-color: white;
          border-radius: 6px;
          color: #586069;
        }
      `}} />
    </div>
  );
};

export default CircuitBreakerDashboard;