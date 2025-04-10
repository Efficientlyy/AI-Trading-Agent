import React, { useEffect, useRef, useState, useMemo } from 'react';
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
    
    return data.map((item) => ({
      time: (new Date(item.timestamp).getTime() / 1000),
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
      volume: item.volume,
    }));
  }, [data]);

  // Calculate indicators
  const indicators = useMemo(() => {
    if (!formattedData || formattedData.length === 0) return {};

    const calculateSMA = (period: number) => {
      const result = [];
      for (let i = 0; i < formattedData.length; i++) {
        if (i < period - 1) {
          result.push({ time: formattedData[i].time, value: null });
          continue;
        }
        
        let sum = 0;
        for (let j = 0; j < period; j++) {
          sum += formattedData[i - j].close;
        }
        result.push({ time: formattedData[i].time, value: sum / period });
      }
      return result;
    };

    const calculateEMA = (period: number) => {
      const result = [];
      const k = 2 / (period + 1);
      
      // First EMA is SMA
      let ema = 0;
      for (let i = 0; i < period; i++) {
        ema += formattedData[i].close;
      }
      ema /= period;
      
      for (let i = 0; i < formattedData.length; i++) {
        if (i < period - 1) {
          result.push({ time: formattedData[i].time, value: null });
          continue;
        }
        
        if (i === period - 1) {
          result.push({ time: formattedData[i].time, value: ema });
          continue;
        }
        
        ema = (formattedData[i].close - ema) * k + ema;
        result.push({ time: formattedData[i].time, value: ema });
      }
      
      return result;
    };

    const calculateRSI = (period: number = 14) => {
      const result = [];
      let gains = 0;
      let losses = 0;
      
      // First RSI needs period + 1 data points
      for (let i = 0; i < formattedData.length; i++) {
        if (i < 1) {
          result.push({ time: formattedData[i].time, value: null });
          continue;
        }
        
        const priceChange = formattedData[i].close - formattedData[i - 1].close;
        
        if (i <= period) {
          gains += priceChange > 0 ? priceChange : 0;
          losses += priceChange < 0 ? -priceChange : 0;
          
          if (i < period) {
            result.push({ time: formattedData[i].time, value: null });
            continue;
          }
          
          // First RSI
          const avgGain = gains / period;
          const avgLoss = losses / period;
          const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
          const rsi = 100 - (100 / (1 + rs));
          
          result.push({ time: formattedData[i].time, value: rsi });
          continue;
        }
        
        // Rest of RSIs use smoothed averages
        const priceChangeNext = formattedData[i].close - formattedData[i - 1].close;
        const gain = priceChangeNext > 0 ? priceChangeNext : 0;
        const loss = priceChangeNext < 0 ? -priceChangeNext : 0;
        
        gains = (gains * (period - 1) + gain) / period;
        losses = (losses * (period - 1) + loss) / period;
        
        const rs = losses === 0 ? 100 : gains / losses;
        const rsi = 100 - (100 / (1 + rs));
        
        result.push({ time: formattedData[i].time, value: rsi });
      }
      
      return result;
    };

    const calculateMACD = (fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9) => {
      const fastEMA = calculateEMA(fastPeriod);
      const slowEMA = calculateEMA(slowPeriod);
      const macdLine = [];
      
      // Calculate MACD line (fast EMA - slow EMA)
      for (let i = 0; i < formattedData.length; i++) {
        if (i < slowPeriod - 1 || fastEMA[i].value === null || slowEMA[i].value === null) {
          macdLine.push({ time: formattedData[i].time, value: null });
          continue;
        }
        
        const fastValue = fastEMA[i].value as number;
        const slowValue = slowEMA[i].value as number;
        
        macdLine.push({
          time: formattedData[i].time,
          value: fastValue - slowValue,
        });
      }
      
      // Calculate signal line (EMA of MACD line)
      const signalLine = [];
      let ema = 0;
      let validPoints = 0;
      
      for (let i = 0; i < macdLine.length; i++) {
        if (macdLine[i].value === null) {
          signalLine.push({ time: macdLine[i].time, value: null });
          continue;
        }
        
        if (validPoints < signalPeriod) {
          const macdValue = macdLine[i].value as number;
          ema += macdValue;
          validPoints++;
          
          if (validPoints < signalPeriod) {
            signalLine.push({ time: macdLine[i].time, value: null });
            continue;
          }
          
          ema /= signalPeriod;
          signalLine.push({ time: macdLine[i].time, value: ema });
          continue;
        }
        
        const macdValue = macdLine[i].value as number;
        ema = (macdValue - ema) * (2 / (signalPeriod + 1)) + ema;
        signalLine.push({ time: macdLine[i].time, value: ema });
      }
      
      // Calculate histogram (MACD line - signal line)
      const histogram = [];
      
      for (let i = 0; i < macdLine.length; i++) {
        if (macdLine[i].value === null || signalLine[i].value === null) {
          histogram.push({ time: macdLine[i].time, value: null });
          continue;
        }
        
        const macdValue = macdLine[i].value as number;
        const signalValue = signalLine[i].value as number;
        
        histogram.push({
          time: macdLine[i].time,
          value: macdValue - signalValue,
        });
      }
      
      return {
        macdLine,
        signalLine,
        histogram,
      };
    };

    // Calculate Bollinger Bands (SMA + standard deviation)
    const calculateBollingerBands = (period: number = 20, multiplier: number = 2) => {
      const sma = calculateSMA(period);
      const upperBand = [];
      const lowerBand = [];
      
      for (let i = 0; i < formattedData.length; i++) {
        if (i < period - 1) {
          upperBand.push({ time: formattedData[i].time, value: null });
          lowerBand.push({ time: formattedData[i].time, value: null });
          continue;
        }
        
        // Calculate standard deviation
        let sum = 0;
        for (let j = 0; j < period; j++) {
          const deviation = formattedData[i - j].close - (sma[i].value as number);
          sum += deviation * deviation;
        }
        const stdDev = Math.sqrt(sum / period);
        
        // Calculate upper and lower bands
        upperBand.push({
          time: formattedData[i].time,
          value: (sma[i].value as number) + (multiplier * stdDev)
        });
        
        lowerBand.push({
          time: formattedData[i].time,
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
          kLine.push({ time: formattedData[i].time, value: null });
          continue;
        }
        
        // Find highest high and lowest low in the period
        let highestHigh = formattedData[i - kPeriod + 1].high;
        let lowestLow = formattedData[i - kPeriod + 1].low;
        
        for (let j = 0; j < kPeriod; j++) {
          const currentHigh = formattedData[i - j].high;
          const currentLow = formattedData[i - j].low;
          
          if (currentHigh > highestHigh) highestHigh = currentHigh;
          if (currentLow < lowestLow) lowestLow = currentLow;
        }
        
        // Calculate %K
        const range = highestHigh - lowestLow;
        const k = range === 0 ? 50 : ((formattedData[i].close - lowestLow) / range) * 100;
        
        kLine.push({ time: formattedData[i].time, value: k });
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
      SMA20: calculateSMA(20),
      SMA50: calculateSMA(50),
      SMA200: calculateSMA(200),
      EMA20: calculateEMA(20),
      RSI: calculateRSI(14),
      MACD: calculateMACD(12, 26, 9),
      BollingerBands: calculateBollingerBands(20, 2),
      Stochastic: calculateStochastic(14, 3),
    };
  }, [formattedData]);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        backgroundColor: '#1E1E1E',
        textColor: '#D9D9D9',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 500,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Create candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26A69A',
      downColor: '#EF5350',
      borderVisible: false,
      wickUpColor: '#26A69A',
      wickDownColor: '#EF5350',
    });

    // Create volume series
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // Set references
    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;
    volumeSeriesRef.current = volumeSeries;

    // Handle resize
    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        // Clear the chart container
        if (chartContainerRef.current) {
          chartContainerRef.current.innerHTML = '';
        }
        chartRef.current = null;
      }
    };
  }, []);

  // Update data
  useEffect(() => {
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current || formattedData.length === 0) return;

    try {
      // Set candlestick data
      candlestickSeriesRef.current.setData(formattedData);

      // Set volume data
      const volumeData = formattedData.map((d) => ({
        time: d.time,
        value: d.volume,
        color: d.close >= d.open ? '#26A69A' : '#EF5350',
      }));
      volumeSeriesRef.current.setData(volumeData);

      // Fit content
      if (chartRef.current) {
        chartRef.current.timeScale().fitContent();
      }
    } catch (error) {
      console.error('Error updating chart data:', error);
    }
  }, [formattedData]);

  // Update indicators
  useEffect(() => {
    if (!chartRef.current || !indicators) return;

    try {
      // Clear old indicator series
      indicatorSeriesRefs.current.forEach((series) => {
        if (chartRef.current) {
          chartRef.current.removeSeries(series);
        }
      });
      indicatorSeriesRefs.current.clear();

      // Add visible indicators
      selectedIndicators.forEach((indicator) => {
        if (!indicator.visible || !chartRef.current) return;

        // Handle different indicator types
        if (indicator.name === 'BollingerBands') {
          const bollingerBands = indicators.BollingerBands;
          if (!bollingerBands) return;

          // Middle band (SMA)
          const middleSeries = chartRef.current.addLineSeries({
            color: indicator.color,
            lineWidth: 2,
            lineStyle: LineStyle.Solid,
            title: 'BB Middle',
          });
          middleSeries.setData(bollingerBands.middle);
          indicatorSeriesRefs.current.set('BB Middle', middleSeries);

          // Upper band
          const upperSeries = chartRef.current.addLineSeries({
            color: indicator.color,
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            title: 'BB Upper',
          });
          upperSeries.setData(bollingerBands.upper);
          indicatorSeriesRefs.current.set('BB Upper', upperSeries);

          // Lower band
          const lowerSeries = chartRef.current.addLineSeries({
            color: indicator.color,
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            title: 'BB Lower',
          });
          lowerSeries.setData(bollingerBands.lower);
          indicatorSeriesRefs.current.set('BB Lower', lowerSeries);
        } else if (indicator.name === 'Stochastic') {
          const stochastic = indicators.Stochastic;
          if (!stochastic) return;

          // %K line
          const kSeries = chartRef.current.addLineSeries({
            color: indicator.color,
            lineWidth: 2,
            title: 'Stochastic %K',
            priceScaleId: 'stochastic',
            scaleMargins: {
              top: 0.8,
              bottom: 0,
            },
          });
          kSeries.setData(stochastic.k);
          indicatorSeriesRefs.current.set('Stochastic %K', kSeries);

          // %D line
          const dSeries = chartRef.current.addLineSeries({
            color: '#FF6D00',
            lineWidth: 2,
            title: 'Stochastic %D',
            priceScaleId: 'stochastic',
            scaleMargins: {
              top: 0.8,
              bottom: 0,
            },
          });
          dSeries.setData(stochastic.d);
          indicatorSeriesRefs.current.set('Stochastic %D', dSeries);

          // Add horizontal lines at 80 and 20 (overbought/oversold)
          const horizontalLines = [
            { value: 80, color: '#FF6D00', lineWidth: 1, title: 'Overbought (80)', lineStyle: LineStyle.Dashed },
            { value: 20, color: '#2962FF', lineWidth: 1, title: 'Oversold (20)', lineStyle: LineStyle.Dashed },
          ];

          horizontalLines.forEach(line => {
            if (chartRef.current) {
              const horizontalSeries = chartRef.current.addLineSeries({
                color: line.color,
                lineWidth: line.lineWidth,
                lineStyle: line.lineStyle,
                title: `Stochastic ${line.title}`,
                priceScaleId: 'stochastic',
              });

              const horizontalData = formattedData.map(d => ({
                time: d.time,
                value: line.value,
              }));

              horizontalSeries.setData(horizontalData);
              indicatorSeriesRefs.current.set(`Stochastic ${line.title}`, horizontalSeries);
            }
          });
        } else if (indicator.name === 'MACD') {
          // MACD is a special case with multiple lines
          const macd = indicators.MACD;
          if (!macd) return;

          if (chartRef.current) {
            // MACD line
            const macdSeries = chartRef.current.addLineSeries({
              color: indicator.color,
              lineWidth: 2,
              title: 'MACD Line',
              priceScaleId: 'macd',
              scaleMargins: {
                top: 0.8,
                bottom: 0,
              },
            });
            macdSeries.setData(macd.macdLine);
            indicatorSeriesRefs.current.set('MACD Line', macdSeries);

            // Signal line
            const signalSeries = chartRef.current.addLineSeries({
              color: '#FF6D00',
              lineWidth: 2,
              title: 'Signal Line',
              priceScaleId: 'macd',
              scaleMargins: {
                top: 0.8,
                bottom: 0,
              },
            });
            signalSeries.setData(macd.signalLine);
            indicatorSeriesRefs.current.set('Signal Line', signalSeries);

            // Histogram
            const histogramSeries = chartRef.current.addHistogramSeries({
              color: '#26A69A',
              title: 'MACD Histogram',
              priceScaleId: 'macd',
              scaleMargins: {
                top: 0.8,
                bottom: 0,
              },
            });
            
            // Set histogram colors based on value
            const histogramData = macd.histogram.map((item: any) => ({
              ...item,
              color: item.value >= 0 ? '#26A69A' : '#EF5350',
            }));
            
            histogramSeries.setData(histogramData);
            indicatorSeriesRefs.current.set('MACD Histogram', histogramSeries);
          }
        } else {
          // Standard line indicators (SMA, EMA, RSI)
          let data = indicators[indicator.name as keyof typeof indicators];
          if (!data) return;

          // Special case for RSI - add to separate scale
          const options: any = {
            color: indicator.color,
            lineWidth: 2,
            title: indicator.name,
          };

          if (indicator.name === 'RSI') {
            if (!chartRef.current) return;
            
            const rsiOptions = {
              ...options,
              priceScaleId: 'rsi',
              scaleMargins: {
                top: 0.1,
                bottom: 0,
              }
            };

            const rsiSeries = chartRef.current.addLineSeries(rsiOptions);
            rsiSeries.setData(data as any[]);
            indicatorSeriesRefs.current.set(indicator.name, rsiSeries);

            // Add horizontal lines at 70 and 30 (overbought/oversold)
            const horizontalLines = [
              { value: 70, color: '#FF6D00', lineWidth: 1, title: 'Overbought (70)', lineStyle: LineStyle.Dashed },
              { value: 30, color: '#2962FF', lineWidth: 1, title: 'Oversold (30)', lineStyle: LineStyle.Dashed },
            ];

            horizontalLines.forEach(line => {
              if (chartRef.current) {
                const horizontalSeries = chartRef.current.addLineSeries({
                  color: line.color,
                  lineWidth: line.lineWidth,
                  lineStyle: line.lineStyle,
                  title: `RSI ${line.title}`,
                  priceScaleId: 'rsi',
                  scaleMargins: {
                    top: 0.1,
                    bottom: 0,
                  },
                });

                const horizontalData = formattedData.map(d => ({
                  time: d.time,
                  value: line.value,
                }));

                horizontalSeries.setData(horizontalData);
                indicatorSeriesRefs.current.set(`RSI ${line.title}`, horizontalSeries);
              }
            });
          } else {
            if (!chartRef.current) return;
            
            const series = chartRef.current.addLineSeries(options);
            series.setData(data as any[]);
            indicatorSeriesRefs.current.set(indicator.name, series);
          }
        }
      });
    } catch (error) {
      console.error('Error updating indicators:', error);
    }
  }, [selectedIndicators, indicators, formattedData]);

  // Toggle indicator visibility
  const toggleIndicator = (name: string) => {
    setSelectedIndicators((prevIndicators) =>
      prevIndicators.map((indicator) =>
        indicator.name === name
          ? { ...indicator, visible: !indicator.visible }
          : indicator
      )
    );
  };
  
  // Handle timeframe change
  const handleTimeframeChange = (timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w') => {
    setSelectedTimeframe(timeframe);
    if (onTimeframeChange) {
      onTimeframeChange(timeframe);
    }
  };
  
  // Handle symbol change
  const handleSymbolChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    if (onSymbolChange) {
      onSymbolChange(event.target.value);
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-4 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {symbol} Chart
          </h3>
          {isLoading && (
            <div className="ml-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
            </div>
          )}
          {error && (
            <div className="ml-2 text-red-500">
              Error: {typeof error === 'string' ? error : error.message}
            </div>
          )}
        </div>
        
        <div className="flex space-x-2">
          {/* Symbol selector */}
          {availableSymbols.length > 0 && onSymbolChange && (
            <select
              className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              value={symbol}
              onChange={handleSymbolChange}
            >
              {availableSymbols.map((sym) => (
                <option key={sym} value={sym}>
                  {sym}
                </option>
              ))}
            </select>
          )}
          
          {/* Timeframe selector */}
          <select
            className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            value={selectedTimeframe}
            onChange={(e) => handleTimeframeChange(e.target.value as '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w')}
          >
            <option value="1m">1 Minute</option>
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="30m">30 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="4h">4 Hours</option>
            <option value="1d">1 Day</option>
            <option value="1w">1 Week</option>
          </select>
        </div>
      </div>
      
      {/* Chart container */}
      <div ref={chartContainerRef} className="w-full h-[400px]" />
      
      {/* Indicator toggles */}
      <div className="flex flex-wrap gap-2 mt-4">
        {selectedIndicators.map((indicator) => (
          <button
            key={indicator.name}
            onClick={() => toggleIndicator(indicator.name)}
            className={`px-3 py-1 text-xs font-medium rounded-full ${
              indicator.visible
                ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
            }`}
            style={{ borderLeft: `3px solid ${indicator.color}` }}
          >
            {indicator.name}
          </button>
        ))}
      </div>
      
      {/* No data message */}
      {!isLoading && !error && (!data || data.length === 0) && (
        <div className="flex justify-center items-center h-[400px] absolute top-0 left-0 w-full">
          <p className="text-gray-500 dark:text-gray-400">No data available</p>
        </div>
      )}
    </div>
  );
};

export default TechnicalAnalysisChart;
