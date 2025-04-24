import React, { useEffect, useRef, useState } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { OHLCV } from '../../types';

// Define line style constants for v3.8.0
const LineStyle = {
  Solid: 0,
  Dotted: 1,
  Dashed: 2,
  LargeDashed: 3,
  SparseDotted: 4,
};

interface TechnicalIndicator {
  name: string;
  visible: boolean;
  color: string;
}

interface TechnicalAnalysisChartProps {
  symbol: string;
  data: OHLCV[] | null;
  isLoading: boolean;
  error?: Error | string | null;
  availableSymbols?: string[];
  onSymbolChange?: (symbol: string) => void;
  onTimeframeChange?: (timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w') => void;
  timeframe?: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w';
  className?: string;
}

function isNumber(val: any): val is number {
  return typeof val === 'number';
}

const TechnicalAnalysisChart: React.FC<TechnicalAnalysisChartProps> = ({
  symbol,
  data,
  isLoading,
  error,
  availableSymbols = [],
  onSymbolChange,
  onTimeframeChange,
  timeframe = '1m',
  className = '',
}) => {
  // Live OHLCV subscription
  const { data: wsData, getOHLCVStream, status: wsStatus } = useWebSocket([]);
  const [liveOHLCV, setLiveOHLCV] = useState<OHLCV[] | null>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<any>(null);
  const candleSeries = useRef<any>(null);
  const volumeSeries = useRef<any>(null);
  const [lastUpdateTime, setLastUpdateTime] = useState<string>("");

  // Subscribe to live OHLCV on symbol/timeframe change
  useEffect(() => {
    if (symbol && timeframe) {
      console.log(`Subscribing to live OHLCV for ${symbol} (${timeframe})`);
      getOHLCVStream(symbol, timeframe);
    }

    // Reset live data when symbol or timeframe changes
    setLiveOHLCV(null);

    return () => {
      // Cleanup
      if (symbol && timeframe) {
        // Unsubscribe logic could be added here if needed
      }
    };
  }, [symbol, timeframe, getOHLCVStream]);

  // Update chart data when new OHLCV arrives via WebSocket
  useEffect(() => {
    if (wsData.ohlcv && wsData.ohlcv.symbol === symbol && wsData.ohlcv.timeframe === timeframe) {
      // Process the incoming data
      const incomingData = wsData.ohlcv.data;

      if (Array.isArray(incomingData)) {
        // Handle array data (full snapshot)
        const formattedData = incomingData.map((candle: any) => ({
          timestamp: new Date(candle.timestamp).toISOString(),
          time: new Date(candle.timestamp).getTime() / 1000, // Keep time for chart library
          open: Number(candle.open),
          high: Number(candle.high),
          low: Number(candle.low),
          close: Number(candle.close),
          volume: Number(candle.volume)
        }));

        setLiveOHLCV(formattedData as unknown as OHLCV[]);
        setLastUpdateTime(new Date().toLocaleTimeString());
      }
      else if (typeof incomingData === 'object' && incomingData !== null) {
        // Handle single candle update
        const newCandle = {
          timestamp: new Date(incomingData.timestamp).toISOString(),
          time: new Date(incomingData.timestamp).getTime() / 1000, // Keep time for chart library
          open: Number(incomingData.open),
          high: Number(incomingData.high),
          low: Number(incomingData.low),
          close: Number(incomingData.close),
          volume: Number(incomingData.volume)
        };

        setLiveOHLCV(prev => {
          if (!prev || prev.length === 0) return [newCandle as unknown as OHLCV];

          // Check if we're updating an existing candle or adding a new one
          const lastCandle = prev[prev.length - 1];
          // Use the time property for comparison (for chart library compatibility)
          if (lastCandle && (lastCandle as any).time === newCandle.time) {
            // Update existing candle
            return [...prev.slice(0, -1), newCandle as unknown as OHLCV];
          }

          // Add new candle
          const updatedData = [...prev, newCandle as unknown as OHLCV];

          // Keep a reasonable number of candles in memory
          const maxCandles = 1000;
          if (updatedData.length > maxCandles) {
            return updatedData.slice(updatedData.length - maxCandles);
          }

          return updatedData;
        });

        setLastUpdateTime(new Date().toLocaleTimeString());
      }
    }
  }, [wsData, symbol, timeframe]);

  // ...rest of your component code (unchanged)...

  return (
    <div className={className}>
      {/* ... rest of your JSX ... */}
    </div>
  );
};

export default TechnicalAnalysisChart;
