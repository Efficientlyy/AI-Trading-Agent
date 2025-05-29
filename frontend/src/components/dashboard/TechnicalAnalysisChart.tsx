import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { OHLCV } from '../../types';
import { PatternDetectionResult, PatternType, PATTERN_COLORS, PATTERN_LINE_STYLES } from '../../types/patterns';
import { v4 as uuidv4 } from 'uuid';
import { createChart } from 'lightweight-charts';
import { IconType } from 'react-icons';
import { FiTrendingUp, FiMenu, FiSettings, FiDownload, FiCamera, FiBarChart2, FiActivity, FiWifi, FiWifiOff } from 'react-icons/fi';

// Import styles
import '../../styles/components/TechnicalAnalysisChart.css';

// Create icon components that properly wrap the icons to make TypeScript happy
const IconWrapper = ({ icon: Icon }: { icon: IconType }) => React.createElement(Icon as React.ElementType);

// Data sources available in the system
export enum DataSource {
  MOCK = 'mock',
  TWELVE_DATA = 'twelvedata',
  MEXC = 'mexc'
}

// TypeScript interfaces for lightweight-charts
interface ChartOptions {
  width?: number;
  height?: number;
  layout?: {
    backgroundColor?: string;
    textColor?: string;
    fontSize?: number;
    fontFamily?: string;
  };
  grid?: {
    vertLines?: { color?: string; style?: number; visible?: boolean };
    horzLines?: { color?: string; style?: number; visible?: boolean };
  };
  timeScale?: any;
  crosshair?: any;
  localization?: any;
}

interface SeriesOptions {
  priceFormat?: {
    type?: string;
    precision?: number;
    minMove?: number;
  };
  color?: string;
  lineWidth?: number;
  crosshairMarkerVisible?: boolean;
  priceLineVisible?: boolean;
  lineType?: number;
  lineStyle?: number;
  [key: string]: any;
}

interface Time {
  day?: number;
  month?: number;
  year?: number;
  time?: number;
}

interface MouseEventParams {
  time?: number | Time;
  point?: { x: number; y: number };
  seriesPrices?: Map<any, number>;
  hoveredSeries?: any;
  hoveredMarkerId?: any;
}

// Define line style constants for v3.8.0
const LineStyle = {
  Solid: 0,
  Dotted: 1,
  Dashed: 2,
  LargeDashed: 3,
  SparseDotted: 4,
};

// Drawing tool types
export enum DrawingToolType {
  NONE = 'none',
  TREND_LINE = 'trend_line',
  HORIZONTAL_LINE = 'horizontal_line',
  RECTANGLE = 'rectangle',
  FIBONACCI = 'fibonacci',
  TEXT = 'text'
}

interface TechnicalIndicator {
  id: string;
  name: string;
  fullName: string;
  visible: boolean;
  color: string;
  series?: any;
  settings?: {
    period?: number;
    source?: 'close' | 'open' | 'high' | 'low' | 'hl2' | 'hlc3' | 'ohlc4';
    [key: string]: any;
  };
}

interface DrawingObject {
  id: string;
  type: DrawingToolType;
  points: Array<{x: number, y: number}>;
  color: string;
  lineWidth?: number;
  text?: string;
  levels?: number[];
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
  patterns?: PatternDetectionResult[];
  showPatterns?: boolean;
  theme?: 'light' | 'dark';
  defaultIndicators?: string[];
  onSaveDrawing?: (drawing: DrawingObject) => void;
  savedDrawings?: DrawingObject[];
  dataSource?: DataSource;
  onDataSourceChange?: (dataSource: DataSource) => void;
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
  patterns = [],
  showPatterns = true,
  theme = 'dark',
  defaultIndicators = [],
  onSaveDrawing,
  savedDrawings = [],
  dataSource = DataSource.MOCK,
  onDataSourceChange
}) => {
  // Live OHLCV subscription
  const { data: wsData, getOHLCVStream, status: wsStatus } = useWebSocket([]);
  const [liveOHLCV, setLiveOHLCV] = useState<OHLCV[] | null>(null);
  const [selectedDataSource, setSelectedDataSource] = useState<DataSource>(dataSource);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [connectionStatus, setConnectionStatus] = useState<{connected: boolean, error?: string}>({connected: false});
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<any>(null);
  const candleSeries = useRef<any>(null);
  const volumeSeries = useRef<any>(null);
  const [lastUpdateTime, setLastUpdateTime] = useState<string>("");
  const webSocketRef = useRef<WebSocket | null>(null);
  
  // Chart UI state
  const [chartType, setChartType] = useState<'candles' | 'bars' | 'line' | 'area'>('candles');
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [showToolbar, setShowToolbar] = useState<boolean>(true);
  
  // Indicators state
  const [availableIndicators] = useState<TechnicalIndicator[]>([
    { id: 'sma20', name: 'SMA', fullName: 'Simple Moving Average (20)', visible: false, color: '#FF6384', settings: { period: 20 } },
    { id: 'sma50', name: 'SMA', fullName: 'Simple Moving Average (50)', visible: false, color: '#36A2EB', settings: { period: 50 } },
    { id: 'sma200', name: 'SMA', fullName: 'Simple Moving Average (200)', visible: false, color: '#FFCE56', settings: { period: 200 } },
    { id: 'ema20', name: 'EMA', fullName: 'Exponential Moving Average (20)', visible: false, color: '#4BC0C0', settings: { period: 20 } },
    { id: 'ema50', name: 'EMA', fullName: 'Exponential Moving Average (50)', visible: false, color: '#9966FF', settings: { period: 50 } },
    { id: 'bb', name: 'BB', fullName: 'Bollinger Bands (20, 2)', visible: false, color: '#FF9F40', settings: { period: 20, stdDev: 2 } },
    { id: 'rsi', name: 'RSI', fullName: 'Relative Strength Index (14)', visible: false, color: '#C9CBCF', settings: { period: 14 } },
  ]);
  const [activeIndicators, setActiveIndicators] = useState<TechnicalIndicator[]>([]);
  
  // Drawing tools state
  const [drawingMode, setDrawingMode] = useState<DrawingToolType>(DrawingToolType.NONE);
  const [drawingPoints, setDrawingPoints] = useState<{x: number, y: number}[]>([]);
  const [drawings, setDrawings] = useState<DrawingObject[]>(savedDrawings || []);
  
  // Pattern visualization state
  const [visualizedPatterns, setVisualizedPatterns] = useState<Record<string, any>>({});
  const [displayedPatterns, setDisplayedPatterns] = useState<PatternType[]>([]);
  const [patternVisibility, setPatternVisibility] = useState<Record<PatternType, boolean>>(
    Object.values(PatternType).reduce((acc, type) => ({ ...acc, [type]: true }), {} as Record<PatternType, boolean>)
  );

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

  // Handle data source change
  const handleDataSourceChange = (newDataSource: DataSource) => {
    setSelectedDataSource(newDataSource);
    if (onDataSourceChange) {
      onDataSourceChange(newDataSource);
    }
    
    // Disconnect existing WebSocket if any
    if (webSocketRef.current) {
      webSocketRef.current.close();
      webSocketRef.current = null;
    }
    
    // Connect to the appropriate data source
    connectToDataSource(newDataSource, symbol, timeframe);
  };
  
  // Connect to the selected data source
  const connectToDataSource = (source: DataSource, symbolParam: string, timeframeParam: string) => {
    // Clean up previous connection
    if (webSocketRef.current) {
      webSocketRef.current.close();
      webSocketRef.current = null;
    }
    
    setIsConnected(false);
    setConnectionStatus({ connected: false });
    
    if (source === DataSource.MEXC) {
      // Connect to MEXC WebSocket
      const wsUrl = `ws://localhost:8000/ws/mexc/${symbolParam.replace('/', '-')}/${timeframeParam}`;
      console.log(`Attempting to connect to MEXC WebSocket: ${wsUrl}`);
      
      try {
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          console.log('Connected to MEXC WebSocket');
          setIsConnected(true);
          setConnectionStatus({ connected: true });
        };
        
        ws.onclose = () => {
          console.log('Disconnected from MEXC WebSocket');
          setIsConnected(false);
          setConnectionStatus({ connected: false });
        };
        
        ws.onerror = (error) => {
          console.error('MEXC WebSocket error:', error);
          setConnectionStatus({
            connected: false,
            error: 'Failed to connect to MEXC API. Check if backend server is running.'
          });
          setIsConnected(false);
        };
        
        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            
            if (message.type === 'kline' && message.data) {
              // Format MEXC kline data to OHLCV format
              const klineData = message.data;
              const ohlcv: OHLCV = {
                timestamp: new Date(klineData.timestamp || klineData.time || Date.now()).toISOString(),
                time: Math.floor(new Date(klineData.timestamp || klineData.time || Date.now()).getTime() / 1000),
                open: typeof klineData.open === 'string' ? parseFloat(klineData.open) : klineData.open,
                high: typeof klineData.high === 'string' ? parseFloat(klineData.high) : klineData.high,
                low: typeof klineData.low === 'string' ? parseFloat(klineData.low) : klineData.low,
                close: typeof klineData.close === 'string' ? parseFloat(klineData.close) : klineData.close,
                volume: typeof klineData.volume === 'string' ? parseFloat(klineData.volume) : klineData.volume
              };
              
              // Update chart with new data
              updateChartWithLiveData(ohlcv);
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
        
        webSocketRef.current = ws;
      } catch (error) {
        console.error('Error connecting to MEXC WebSocket:', error);
        setConnectionStatus({
          connected: false,
          error: 'Failed to connect to MEXC API. Check if backend server is running.'
        });
        setIsConnected(false);
      }
    } else if (source === DataSource.TWELVE_DATA) {
      // Connect to Twelve Data WebSocket (using existing hook)
      getOHLCVStream(symbolParam, timeframeParam);
    }
    // Mock data source doesn't need a WebSocket connection
  };
  
  // Update chart with live data
  const updateChartWithLiveData = (ohlcv: OHLCV) => {
    if (!candleSeries.current || !volumeSeries.current) return;
    
    // Format data for chart
    const formattedData = {
      time: ohlcv.time || Math.floor(new Date(ohlcv.timestamp).getTime() / 1000),
      open: ohlcv.open,
      high: ohlcv.high,
      low: ohlcv.low,
      close: ohlcv.close,
      volume: ohlcv.volume
    };
    
    // Update candle series
    candleSeries.current.update(formattedData);
    volumeSeries.current.update(formattedData);
    
    // Update last update time
    setLastUpdateTime(new Date(ohlcv.timestamp).toLocaleString());
  };
  
  // Chart manipulation functions
  const changeChartType = (type: 'candles' | 'bars' | 'line' | 'area') => {
    setChartType(type);
    
    // Re-initialize chart with the new type
    if (data && data.length > 0 && chartInstance.current && candleSeries.current) {
      // Store the visible range to restore it after changing the chart type
      const visibleRange = chartInstance.current.timeScale().getVisibleRange();
      
      // Determine how to update the chart type
      if (type === 'candles') {
        candleSeries.current.applyOptions({
          type: 'candlestick',
        });
      } else if (type === 'bars') {
        candleSeries.current.applyOptions({
          type: 'bar',
        });
      } else if (type === 'line') {
        candleSeries.current.applyOptions({
          type: 'line',
          lineWidth: 2,
        });
      } else if (type === 'area') {
        candleSeries.current.applyOptions({
          type: 'area',
          lineWidth: 2,
        });
      }
      
      // Restore the visible range
      if (visibleRange) {
        chartInstance.current.timeScale().setVisibleRange(visibleRange);
      }
    }
  };
  
  // Export chart data as CSV
  const exportChartData = () => {
    if (!data) return;
    
    // Create CSV content
    const csvContent = [
      'timestamp,open,high,low,close,volume',
      ...data.map(d => {
        const timestamp = typeof d.timestamp === 'number' ? 
          new Date(d.timestamp).toISOString() : 
          new Date(d.timestamp).toISOString();
        return `${timestamp},${d.open},${d.high},${d.low},${d.close},${d.volume}`;
      })
    ].join('\n');
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${symbol}_${timeframe}_data.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  // Take screenshot of chart
  const takeScreenshot = () => {
    if (chartInstance.current) {
      const canvas = chartInstance.current.takeScreenshot();
      const link = document.createElement('a');
      link.download = `${symbol}_chart_${new Date().toISOString().split('T')[0]}.png`;
      link.href = canvas;
      link.click();
    }
  };
  
  // Toggle indicator visibility
  const toggleIndicator = (indicatorId: string) => {
    setActiveIndicators(prevIndicators => {
      const indicator = availableIndicators.find(ind => ind.id === indicatorId);
      if (!indicator) return prevIndicators;
      
      const isActive = prevIndicators.some(ind => ind.id === indicatorId);
      if (isActive) {
        // Remove indicator
        return prevIndicators.filter(ind => ind.id !== indicatorId);
      } else {
        // Add indicator
        return [...prevIndicators, {...indicator, visible: true}];
      }
    });
  };
  
  // Connect to the data source when component mounts or when parameters change
  useEffect(() => {
    if (selectedDataSource) {
      connectToDataSource(selectedDataSource, symbol, timeframe);
    }
    
    return () => {
      // Clean up WebSocket connection on unmount
      if (webSocketRef.current) {
        webSocketRef.current.close();
        webSocketRef.current = null;
      }
    };
  }, [selectedDataSource, symbol, timeframe, getOHLCVStream]);
  
  // Update chart data when new OHLCV arrives via WebSocket (for Twelve Data)
  useEffect(() => {
    if (selectedDataSource === DataSource.TWELVE_DATA && 
        wsData && wsData.ohlcv && wsData.ohlcv.symbol === symbol && 
        wsData.ohlcv.timeframe === timeframe && wsData.ohlcv.data) {
      // Process the WebSocket data based on its format
      let ohlcvData: OHLCV;
      
      if (Array.isArray(wsData.ohlcv.data)) {
        // Handle bulk data update
        const lastCandle = wsData.ohlcv.data[wsData.ohlcv.data.length - 1];
        ohlcvData = {
          timestamp: new Date(lastCandle.timestamp).toISOString(),
          time: Math.floor(new Date(lastCandle.timestamp).getTime() / 1000),
          open: parseFloat(lastCandle.open),
          high: parseFloat(lastCandle.high),
          low: parseFloat(lastCandle.low),
          close: parseFloat(lastCandle.close),
          volume: parseFloat(lastCandle.volume)
        };
      } else {
        // Handle single candle update
        const singleCandle = wsData.ohlcv.data;
        ohlcvData = {
          timestamp: new Date(singleCandle.timestamp).toISOString(),
          time: Math.floor(new Date(singleCandle.timestamp).getTime() / 1000),
          open: parseFloat(singleCandle.open),
          high: parseFloat(singleCandle.high),
          low: parseFloat(singleCandle.low),
          close: parseFloat(singleCandle.close),
          volume: parseFloat(singleCandle.volume)
        };
      }
      
      // Update chart with new data
      updateChartWithLiveData(ohlcvData);
    }
  }, [wsData, symbol, timeframe, selectedDataSource]);

  // Effect to handle initial historical data loading
  useEffect(() => {
    if (data && data.length > 0) {
      const formattedData = data.map(candle => ({
        time: new Date(candle.timestamp).getTime() / 1000,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume
      }));
      
      if (candleSeries.current) {
        candleSeries.current.setData(formattedData);
      }
      
      if (volumeSeries.current) {
        volumeSeries.current.setData(formattedData);
      }
      
      // Update last update time
      const lastCandle = data[data.length - 1];
      setLastUpdateTime(new Date(lastCandle.timestamp).toLocaleString());
    }
  }, [data]);

  return (
    <div className={`${className} technical-analysis-chart-container`}>
      <div className="chart-header">
        <div className="symbol-selector">
          <select 
            value={symbol} 
            onChange={(e) => onSymbolChange?.(e.target.value)}
            disabled={availableSymbols.length === 0}
          >
            {availableSymbols.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
        
        <div className="timeframe-selector">
          <select 
            value={timeframe} 
            onChange={(e) => onTimeframeChange?.(e.target.value as any)}
          >
            <option value="1m">1m</option>
            <option value="5m">5m</option>
            <option value="15m">15m</option>
            <option value="30m">30m</option>
            <option value="1h">1h</option>
            <option value="4h">4h</option>
            <option value="1d">1d</option>
            <option value="1w">1w</option>
          </select>
        </div>
        
        <div className="data-source-selector">
          <select
            value={selectedDataSource}
            onChange={(e) => handleDataSourceChange(e.target.value as DataSource)}
            className="chart-data-source-selector"
          >
            <option value={DataSource.MOCK}>Mock Data</option>
            <option value={DataSource.TWELVE_DATA}>Twelve Data</option>
            <option value={DataSource.MEXC}>MEXC Exchange</option>
          </select>
          
          {/* Connection Status Indicator */}
          <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`} title={isConnected ? 'Connected to data source' : 'Disconnected from data source'}>
            {isConnected ? <IconWrapper icon={FiWifi} /> : <IconWrapper icon={FiWifiOff} />}
          </div>
          {connectionStatus.error && (
            <div className="connection-error">{connectionStatus.error}</div>
          )}
        </div>
      </div>
      
      {showToolbar && (
        <div className="chart-toolbar">
          <div className="toolbar-group">
            <button 
              onClick={() => changeChartType('candles')} 
              className={chartType === 'candles' ? 'active' : ''}
              title="Candlestick Chart"
            >
              <IconWrapper icon={FiBarChart2} />
            </button>
            <button 
              onClick={() => changeChartType('bars')} 
              className={chartType === 'bars' ? 'active' : ''}
              title="Bar Chart"
            >
              <IconWrapper icon={FiBarChart2} />
            </button>
            <button 
              onClick={() => changeChartType('line')} 
              className={chartType === 'line' ? 'active' : ''}
              title="Line Chart"
            >
              <IconWrapper icon={FiActivity} />
            </button>
            <button 
              onClick={() => changeChartType('area')} 
              className={chartType === 'area' ? 'active' : ''}
              title="Area Chart"
            >
              <IconWrapper icon={FiActivity} />
            </button>
          </div>
          
          <div className="toolbar-group">
            <button 
              onClick={() => setDrawingMode(DrawingToolType.TREND_LINE)}
              className={drawingMode === DrawingToolType.TREND_LINE ? 'active' : ''}
              title="Trend Line"
            >
              <IconWrapper icon={FiTrendingUp} />
            </button>
            <button 
              onClick={() => setDrawingMode(DrawingToolType.HORIZONTAL_LINE)}
              className={drawingMode === DrawingToolType.HORIZONTAL_LINE ? 'active' : ''}
              title="Horizontal Line"
            >
              —
            </button>
            <button 
              onClick={() => setDrawingMode(DrawingToolType.RECTANGLE)}
              className={drawingMode === DrawingToolType.RECTANGLE ? 'active' : ''}
              title="Rectangle"
            >
              □
            </button>
            <button 
              onClick={() => setDrawingMode(DrawingToolType.FIBONACCI)}
              className={drawingMode === DrawingToolType.FIBONACCI ? 'active' : ''}
              title="Fibonacci Retracement"
            >
              Fib
            </button>
          </div>
          
          <div className="toolbar-group">
            <button onClick={() => setShowSettings(!showSettings)} title="Indicators & Settings">
              <IconWrapper icon={FiSettings} />
            </button>
            <button onClick={exportChartData} title="Export Data">
              <IconWrapper icon={FiDownload} />
            </button>
            <button onClick={takeScreenshot} title="Take Screenshot">
              <IconWrapper icon={FiCamera} />
            </button>
          </div>
        </div>
      )}
      
      {showSettings && (
        <div className="chart-settings-panel">
          <div className="indicators-section">
            <h4>Indicators</h4>
            <div className="indicator-list">
              {availableIndicators.map(indicator => (
                <div key={indicator.id} className="indicator-item">
                  <input 
                    type="checkbox" 
                    id={`indicator-${indicator.id}`}
                    checked={activeIndicators.some(ind => ind.id === indicator.id)}
                    onChange={() => toggleIndicator(indicator.id)}
                  />
                  <label htmlFor={`indicator-${indicator.id}`}>
                    <span className="indicator-color" style={{ backgroundColor: indicator.color }}></span>
                    {indicator.fullName}
                  </label>
                </div>
              ))}
            </div>
          </div>
          
          <div className="patterns-section">
            <h4>Pattern Detection</h4>
            <div className="pattern-list">
              {Object.values(PatternType).map(patternType => (
                <div key={patternType} className="pattern-item">
                  <input 
                    type="checkbox" 
                    id={`pattern-${patternType}`}
                    checked={patternVisibility[patternType]}
                    onChange={() => setPatternVisibility(prev => ({
                      ...prev,
                      [patternType]: !prev[patternType]
                    }))}
                  />
                  <label htmlFor={`pattern-${patternType}`}>
                    <span className="pattern-color" style={{ 
                      backgroundColor: PATTERN_COLORS[patternType] || '#888888' 
                    }}></span>
                    {patternType}
                  </label>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      
      <div 
        ref={chartRef} 
        className="chart-container"
        style={{ height: '400px', width: '100%' }}
      />
      
      {isLoading && (
        <div className="chart-loader">
          <div className="loader-spinner"></div>
          <p>Loading chart data...</p>
        </div>
      )}
      
      {error && (
        <div className="chart-error">
          <p>Error: {error instanceof Error ? error.message : error}</p>
        </div>
      )}
    </div>
  );
};

export default TechnicalAnalysisChart;
