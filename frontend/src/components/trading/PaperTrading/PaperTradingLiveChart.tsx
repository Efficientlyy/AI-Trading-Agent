import React, { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import webSocketService, { WebSocketTopic } from '../../../services/WebSocketService';
import { Chart, ChartConfiguration, ChartData, ChartOptions } from 'chart.js/auto';
import { format } from 'date-fns';

interface PriceData {
  timestamp: string;
  price: number;
  signal?: 'buy' | 'sell' | 'hold';
}

interface MarketDataUpdate {
  timestamp: string;
  symbols: string[];
  prices: Record<string, number>;
}

interface SignalUpdate {
  timestamp: string;
  signals: Record<string, { direction: 'buy' | 'sell' | 'hold', strength: number }>;
}

const PaperTradingLiveChart: React.FC = () => {
  const { state } = usePaperTrading();
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [priceHistory, setPriceHistory] = useState<PriceData[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  // Get available symbols
  const symbols = state.currentStatus?.symbols || [];

  // Set initial selected symbol when symbols are loaded
  useEffect(() => {
    if (symbols.length > 0 && !selectedSymbol) {
      setSelectedSymbol(symbols[0]);
    }
  }, [symbols, selectedSymbol]);

  // Connect to WebSocket for real-time updates
  useEffect(() => {
    if (state.activeSessions.length > 0) {
      // Connect to WebSocket
      webSocketService.connect([WebSocketTopic.STATUS, WebSocketTopic.PORTFOLIO])
        .then(() => {
          setIsConnected(true);
          console.log('Connected to WebSocket for live chart data');
        })
        .catch(error => {
          console.error('Failed to connect to WebSocket for live chart data:', error);
        });

      // Set up event handlers
      const handleMarketDataUpdate = (data: MarketDataUpdate) => {
        if (!selectedSymbol || !data.prices[selectedSymbol]) return;

        // Add new price data
        const newPriceData: PriceData = {
          timestamp: data.timestamp,
          price: data.prices[selectedSymbol]
        };

        setPriceHistory(prev => {
          // Keep only the last 100 data points to avoid performance issues
          const newHistory = [...prev, newPriceData];
          if (newHistory.length > 100) {
            return newHistory.slice(newHistory.length - 100);
          }
          return newHistory;
        });
      };

      const handleSignalUpdate = (data: SignalUpdate) => {
        if (!selectedSymbol || !data.signals[selectedSymbol]) return;

        // Update the latest price data with signal information
        setPriceHistory(prev => {
          if (prev.length === 0) return prev;

          const newHistory = [...prev];
          const lastItem = newHistory[newHistory.length - 1];
          
          // Update the signal for the last data point
          newHistory[newHistory.length - 1] = {
            ...lastItem,
            signal: data.signals[selectedSymbol].direction
          };
          
          return newHistory;
        });
      };

      // Register event handlers
      webSocketService.on(WebSocketTopic.STATUS, handleMarketDataUpdate);
      webSocketService.on(WebSocketTopic.PORTFOLIO, handleSignalUpdate);

      // Cleanup function
      return () => {
        webSocketService.off(WebSocketTopic.STATUS, handleMarketDataUpdate);
        webSocketService.off(WebSocketTopic.PORTFOLIO, handleSignalUpdate);
        
        // Disconnect from WebSocket
        webSocketService.disconnect();
        setIsConnected(false);
      };
    }
  }, [state.activeSessions.length, selectedSymbol]);

  // Initialize and update chart
  useEffect(() => {
    if (!chartRef.current || !selectedSymbol) return;

    // Destroy existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    // Prepare chart data
    const labels = priceHistory.map(data => {
      const date = new Date(data.timestamp);
      return format(date, 'HH:mm:ss');
    });

    const prices = priceHistory.map(data => data.price);

    // Prepare buy/sell signals
    const buySignals = priceHistory.map(data => 
      data.signal === 'buy' ? data.price : null
    );

    const sellSignals = priceHistory.map(data => 
      data.signal === 'sell' ? data.price : null
    );

    // Create chart configuration
    const chartConfig: ChartConfiguration = {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: `${selectedSymbol} Price`,
            data: prices,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1,
            fill: false
          },
          {
            label: 'Buy Signals',
            data: buySignals,
            pointBackgroundColor: 'green',
            pointBorderColor: 'green',
            pointRadius: 5,
            showLine: false
          },
          {
            label: 'Sell Signals',
            data: sellSignals,
            pointBackgroundColor: 'red',
            pointBorderColor: 'red',
            pointRadius: 5,
            showLine: false
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
              text: 'Price'
            }
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
  }, [selectedSymbol, priceHistory]);

  // Handle symbol change
  const handleSymbolChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedSymbol(e.target.value);
    // Reset price history when symbol changes
    setPriceHistory([]);
  };

  return (
    <div className="paper-trading-live-chart">
      <div className="chart-header">
        <h3>Live Price Chart</h3>
        <div className="chart-controls">
          <select
            value={selectedSymbol}
            onChange={handleSymbolChange}
            disabled={symbols.length === 0}
          >
            {symbols.length === 0 ? (
              <option value="">No symbols available</option>
            ) : (
              symbols.map((symbol: string) => (
                <option key={symbol} value={symbol}>
                  {symbol}
                </option>
              ))
            )}
          </select>
          <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </div>
      
      <div className="chart-container">
        {symbols.length === 0 ? (
          <div className="no-data">No trading symbols available</div>
        ) : priceHistory.length === 0 ? (
          <div className="no-data">Waiting for price data...</div>
        ) : (
          <canvas ref={chartRef} />
        )}
      </div>
    </div>
  );
};

export default PaperTradingLiveChart;
