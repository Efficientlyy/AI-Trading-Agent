import React, { useEffect, useRef, useState, useMemo } from 'react';
import { useRenderLogger } from '../../hooks/useRenderLogger';
import { createChart } from 'lightweight-charts';
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
  timeframe,
  className = '',
}) => {
  useRenderLogger('TechnicalAnalysisChart', { symbol, data, isLoading });
  // Chart references
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const indicatorSeriesRefs = useRef<Map<string, any>>(new Map());
  
  // State for selected indicators and timeframe
  const [selectedIndicators, setSelectedIndicators] = useState<TechnicalIndicator[]>([
    { name: 'SMA20', visible: true, color: '#2962FF' },
    { name: 'SMA50', visible: false, color: '#FF6D00' },
    { name: 'SMA200', visible: false, color: '#F44336' },
    { name: 'EMA20', visible: false, color: '#00BCD4' },
    { name: 'RSI', visible: false, color: '#9C27B0' },
    { name: 'MACD', visible: false, color: '#4CAF50' },
    { name: 'BollingerBands', visible: false, color: '#FFC107' },
    { name: 'Stochastic', visible: false, color: '#795548' },
  ]);
  
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w'>('1d');
  
  // Format data for the chart
  const formattedData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    return data.map((item, idx) => {
      if (
        !item ||
        typeof item.open !== 'number' ||
        typeof item.high !== 'number' ||
        typeof item.low !== 'number' ||
        typeof item.close !== 'number' ||
        typeof item.volume !== 'number' ||
        !item.timestamp
      ) {
        console.warn('Skipping invalid OHLCV data at idx', idx, item);
        return null;
      }
      return {
        time: (new Date(item.timestamp).getTime() / 1000),
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
        volume: item.volume,
      };
    }).filter(Boolean);
  }, [data]);

  // Calculate indicators
  const indicators = useMemo(() => {
    if (!formattedData || formattedData.length === 0) return {};

    const calculateSMA = (period: number) => {
      const result: Array<{ time: number | null; value: number | null }> = [];
      for (let i = 0; i < formattedData.length; i++) {
        if (i < period - 1) {
          const dataPoint = formattedData[i];
          if (dataPoint) {
            result.push({ time: dataPoint.time, value: null });
          } else {
            result.push({ time: null, value: null });
          }
          continue;
        }
        // Defensive: skip if any close is missing
        let sum = 0;
        let valid = true;
        for (let j = 0; j < period; j++) {
          const dataPoint = formattedData[i - j];
          if (!dataPoint || typeof dataPoint.close !== 'number') {
            valid = false;
            break;
          }
          sum += dataPoint.close;
        }
        const dataPointOuter = formattedData[i];
        result.push({ time: dataPointOuter ? dataPointOuter.time : null, value: valid ? sum / period : null });
      }
      return result;
    };

    const calculateEMA = (period: number) => {
      const result: Array<{ time: number | null; value: number | null }> = [];
      const k = 2 / (period + 1);
      // First EMA is SMA
      let ema = 0;
      for (let i = 0; i < period; i++) {
        const dataPoint = formattedData[i];
        if (!dataPoint || typeof dataPoint.close !== 'number') {
          result.push({ time: null, value: null });
          return result;
        }
        ema += dataPoint.close;
      }
      ema /= period;
      let dataPoint = formattedData[period - 1];
      result.push({ time: dataPoint ? dataPoint.time : null, value: ema });
      for (let i = period; i < formattedData.length; i++) {
        dataPoint = formattedData[i];
        if (!dataPoint || typeof dataPoint.close !== 'number') {
          result.push({ time: null, value: null });
          continue;
        }
        ema = dataPoint.close * k + ema * (1 - k);
        result.push({ time: dataPoint.time, value: ema });
      }
      return result;
    };

    const calculateRSI = (period: number) => {
      const result: Array<{ time: number | null; value: number | null }> = [];
      let gains = 0;
      let losses = 0;
      for (let i = 1; i <= period; i++) {
        const dataPoint = formattedData[i];
        const prevDataPoint = formattedData[i - 1];
        if (!dataPoint || !prevDataPoint || typeof dataPoint.close !== 'number' || typeof prevDataPoint.close !== 'number') {
          result.push({ time: dataPoint ? dataPoint.time : null, value: null });
          continue;
        }
        const priceChange = (dataPoint.close - prevDataPoint.close);
        if (priceChange > 0) gains += priceChange;
        else losses -= priceChange;
        result.push({ time: dataPoint.time, value: null });
      }
      let avgGain = gains / period;
      let avgLoss = losses / period;
      for (let i = period + 1; i < formattedData.length; i++) {
        const dataPoint = formattedData[i];
        const prevDataPoint = formattedData[i - 1];
        if (!dataPoint || !prevDataPoint || typeof dataPoint.close !== 'number' || typeof prevDataPoint.close !== 'number') {
          result.push({ time: dataPoint ? dataPoint.time : null, value: null });
          continue;
        }
        const priceChange = (dataPoint.close - prevDataPoint.close);
        let gain = 0;
        let loss = 0;
        if (priceChange > 0) gain = priceChange;
        else loss = -priceChange;
        avgGain = (avgGain * (period - 1) + gain) / period;
        avgLoss = (avgLoss * (period - 1) + loss) / period;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        const rsi = 100 - 100 / (1 + rs);
        result.push({ time: dataPoint.time, value: rsi });
      }
      return result;
    };

    const calculateMACD = (fastPeriod: number, slowPeriod: number, signalPeriod: number) => {
      const fastEMA = calculateEMA(fastPeriod);
      const slowEMA = calculateEMA(slowPeriod);
      const macdLine: Array<{ time: number | null; value: number | null }> = [];
      for (let i = 0; i < formattedData.length; i++) {
        if (!fastEMA[i] || !slowEMA[i] || fastEMA[i].value === null || slowEMA[i].value === null) {
          if (formattedData[i]) {
            const macdDataPoint = formattedData[i];
            macdLine.push({ time: macdDataPoint ? macdDataPoint.time : null, value: null });
          } else {
            macdLine.push({ time: null, value: null });
          }
          continue;
        }
        macdLine.push({ time: fastEMA[i].time, value: (fastEMA[i].value as number) - (slowEMA[i].value as number) });
      }
      const signalLine: Array<{ time: number | null; value: number | null }> = calculateEMA(signalPeriod);
      const histogram: Array<{ time: number | null; value: number | null }> = [];
      for (let i = 0; i < macdLine.length; i++) {
        if (!macdLine[i] || !signalLine[i] || macdLine[i].value === null || signalLine[i].value === null) {
          if (macdLine[i]) {
            histogram.push({ time: macdLine[i].time, value: null });
          } else {
            histogram.push({ time: null, value: null });
          }
          continue;
        }
        histogram.push({ time: macdLine[i].time, value: (macdLine[i].value as number) - (signalLine[i].value as number) });
      }
      return { macdLine, signalLine, histogram };
    };

    const calculateBollingerBands = (period: number, multiplier: number) => {
      const sma = calculateSMA(period);
      const upperBand: Array<{ time: number | null; value: number | null }> = [];
      const lowerBand: Array<{ time: number | null; value: number | null }> = [];
      for (let i = 0; i < formattedData.length; i++) {
        if (i < period - 1 || !sma[i] || sma[i].value === null) {
          const time = formattedData[i]?.time ?? null;
          upperBand.push({ time, value: null });
          lowerBand.push({ time, value: null });
          continue;
        }
        // Calculate standard deviation
        let sum = 0;
        let count = 0;
        for (let j = 0; j < period; j++) {
          const dataPoint = formattedData[i - j];
          if (!dataPoint || typeof dataPoint.close !== 'number') continue;
          sum += dataPoint.close;
          count++;
        }
        const mean = sum / count;
        let variance = 0;
        for (let j = 0; j < period; j++) {
          const dataPoint = formattedData[i - j];
          if (!dataPoint || typeof dataPoint.close !== 'number') continue;
          variance += Math.pow(dataPoint.close - mean, 2);
        }
        variance = variance / count;
        const stdDev = Math.sqrt(variance);
        upperBand.push({
          time: formattedData[i]?.time ?? null,
          value: (sma[i].value as number) + (multiplier * stdDev)
        });
        lowerBand.push({
          time: formattedData[i]?.time ?? null,
          value: (sma[i].value as number) - (multiplier * stdDev)
        });
      }
      return {
        middle: sma,
        upper: upperBand,
        lower: lowerBand
      };
    };

    // Calculate Stochastic Oscillator
    const calculateStochastic = (kPeriod: number = 14, dPeriod: number = 3) => {
      const kLine = [];
      // Calculate %K line
      for (let i = 0; i < formattedData.length; i++) {
        if (i < kPeriod - 1) {
          kLine.push({ time: formattedData[i]?.time ?? null, value: null });
          continue;
        }
        // Find highest high and lowest low in the period
        let highestHigh: number | null = formattedData[i - kPeriod + 1]?.high ?? null;
        let lowestLow: number | null = formattedData[i - kPeriod + 1]?.low ?? null;
        for (let j = 0; j < kPeriod; j++) {
          const dataPoint = formattedData[i - j];
          if (!dataPoint || typeof dataPoint.high !== 'number' || typeof dataPoint.low !== 'number') continue;
          const currentHigh = dataPoint.high;
          const currentLow = dataPoint.low;
          if (highestHigh === null || currentHigh > highestHigh) highestHigh = currentHigh;
          if (lowestLow === null || currentLow < lowestLow) lowestLow = currentLow;
        }
        // Calculate %K
        const dataPoint = formattedData[i];
if (
  !isNumber(highestHigh) ||
  !isNumber(lowestLow) ||
  !dataPoint ||
  typeof dataPoint.close !== 'number'
) {
  kLine.push({ time: dataPoint?.time ?? null, value: null });
  continue;
}
const range = highestHigh - lowestLow;
const k = range === 0 ? 50 : ((dataPoint.close - lowestLow) / range) * 100;
kLine.push({ time: dataPoint.time, value: k });
      }
      // Calculate %D line (SMA of %K)
      const dLine = [];
      for (let i = 0; i < kLine.length; i++) {
        if (i < kPeriod - 1 + dPeriod - 1) {
          dLine.push({ time: kLine[i].time, value: null });
          continue;
        }
        let sum = 0;
        for (let j = 0; j < dPeriod; j++) {
          sum += kLine[i - j].value as number;
        }
        dLine.push({ time: kLine[i].time, value: sum / dPeriod });
      }
      return {
        k: kLine,
        d: dLine
      };
    };

    return {
      SMA50: calculateSMA(50),
      SMA200: calculateSMA(200),
      EMA20: calculateEMA(20),
      RSI: calculateRSI(14),
      MACD: calculateMACD(12, 26, 9),
      BollingerBands: calculateBollingerBands(20, 2),
      Stochastic: calculateStochastic(14, 3),
    };
  }, [formattedData]);

  // ... rest of your component code ...

  return (
    <div className={className}>
      {/* ... rest of your JSX ... */}
    </div>
  );
};

export default TechnicalAnalysisChart;
