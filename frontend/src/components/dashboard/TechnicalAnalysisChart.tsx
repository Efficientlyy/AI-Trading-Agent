import React, { useEffect, useRef, useState, useMemo } from 'react';
import { useRenderLogger } from '../../hooks/useRenderLogger';
import { createChart } from 'lightweight-charts';
import { OHLCV } from '../../types';
import { useWebSocket } from '../../hooks/useWebSocket';

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

  // Subscribe to live OHLCV on symbol/timeframe change
  useEffect(() => {
    if (symbol && timeframe) {
      getOHLCVStream(symbol, timeframe);
    }
  }, [symbol, timeframe, getOHLCVStream]);

  // Update chart data when new OHLCV arrives
  useEffect(() => {
    if (wsData.ohlcv && wsData.ohlcv.symbol === symbol && wsData.ohlcv.timeframe === timeframe) {
      // wsData.ohlcv.data can be array (snapshot) or single candle (update)
      if (Array.isArray(wsData.ohlcv.data)) {
        setLiveOHLCV(wsData.ohlcv.data as OHLCV[]);
      } else if (typeof wsData.ohlcv.data === 'object' && wsData.ohlcv.data !== null) {
        setLiveOHLCV(prev => {
          const newCandle = wsData.ohlcv!.data as OHLCV;
          if (!prev) return [newCandle];
          const last = prev[prev.length - 1];
          if (last && last.timestamp === newCandle.timestamp) {
            return [...prev.slice(0, -1), newCandle];
          }
          return [...prev, newCandle];
        });
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
