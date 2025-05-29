import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart } from 'lightweight-charts';
import './LiveCryptoChart.css';

interface LiveCryptoChartProps {
  symbol: string;
  interval?: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d';
  darkMode?: boolean;
  height?: number;
  width?: number;
  showVolume?: boolean;
  onSymbolChange?: (symbol: string) => void;
}

interface OHLCVData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

// Available crypto trading pairs
const AVAILABLE_PAIRS = [
  // MEXC pairs
  'BTC/USDC', 'ETH/USDC', 'BNB/USDC', 'SOL/USDC', 'XRP/USDC',
  // Standard pairs
  'BTC/USD', 'ETH/USD', 'BTC/USDT', 'ETH/USDT', 
  'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 
  'DOGE/USDT', 'DOT/USDT', 'AVAX/USDT'
];

// Available data sources
const DATA_SOURCES = [
  { value: 'mexc', label: 'MEXC' },
  { value: 'twelvedata', label: 'Twelve Data' },
  { value: 'mock', label: 'Mock Data' }
];

const LiveCryptoChart: React.FC<LiveCryptoChartProps> = ({
  symbol = 'BTC/USD',
  interval = '1m',
  darkMode = true,
  height = 500,
  width = 800,
  showVolume = true,
  onSymbolChange
}) => {
  // Refs
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candleSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  
  // State
  const [ohlcvData, setOhlcvData] = useState<OHLCVData[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [lastPrice, setLastPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [wsInstance, setWsInstance] = useState<WebSocket | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbol);
  const [selectedInterval, setSelectedInterval] = useState<string>(interval);
  const [lastUpdateTime, setLastUpdateTime] = useState<string>('');
  const [selectedDataSource, setSelectedDataSource] = useState<string>('mexc'); // Default to MEXC

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    // Close existing connection if any
    if (wsInstance) {
      wsInstance.close();
    }

    // Format symbol for API (replace / with -)
    const formattedSymbol = selectedSymbol.replace('/', '-');
    let wsUrl = '';
    
    // Choose appropriate WebSocket URL based on selected data source
    if (selectedDataSource === 'mexc') {
      // MEXC WebSocket endpoint
      wsUrl = `ws://localhost:8000/ws/mexc/ticker/${formattedSymbol}`;
      if (selectedInterval !== '1m') {
        // For klines, use the kline endpoint with interval
        wsUrl = `ws://localhost:8000/ws/mexc/kline/${formattedSymbol}/${selectedInterval}`;
      }
    } else if (selectedDataSource === 'twelvedata') {
      // Original TwelveData WebSocket endpoint
      wsUrl = `ws://localhost:8000/ws/crypto/${formattedSymbol}/${selectedInterval}`;
    } else {
      // Mock data endpoint
      wsUrl = `ws://localhost:8000/ws/mock/${formattedSymbol}/${selectedInterval}`;
    }
    
    console.log(`Connecting to ${wsUrl}`);
    
    try {
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log(`WebSocket connected for ${selectedSymbol} using ${selectedDataSource}`);
        setIsConnected(true);
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        
        // Attempt to reconnect after 2 seconds
        setTimeout(() => {
          connectWebSocket();
        }, 2000);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // console.log('Received WebSocket data:', data); // Debug incoming data (commented to reduce console noise)
          
          // Handle MEXC-specific message formats
          if (selectedDataSource === 'mexc') {
            if (data.type === 'ticker') {
              // MEXC ticker update
              const tickerData = data.data;
              const newPrice = parseFloat(tickerData.price);
              const prevPrice = lastPrice;
              setLastPrice(newPrice);
              
              // Calculate price change
              if (prevPrice !== null) {
                const change = ((newPrice - prevPrice) / prevPrice) * 100;
                setPriceChange(change);
              } else if (tickerData.change) {
                // Use change directly from ticker if available
                setPriceChange(parseFloat(tickerData.change));
              }
              
              setLastUpdateTime(new Date().toLocaleTimeString());
            }
            else if (data.type === 'kline' || data.type === 'crypto_kline') {
              // MEXC kline update
              const klineData = data.data;
              
              if (klineData) {
                const candle: OHLCVData = {
                  time: new Date(klineData.timestamp).getTime() / 1000, // Convert to seconds
                  open: parseFloat(klineData.open),
                  high: parseFloat(klineData.high),
                  low: parseFloat(klineData.low),
                  close: parseFloat(klineData.close),
                  volume: parseFloat(klineData.volume)
                };
                
                // Update chart data
                setOhlcvData(prevData => {
                  // Check if this is an update to the last candle
                  if (prevData.length > 0 && prevData[prevData.length - 1].time === candle.time) {
                    const newData = [...prevData];
                    newData[newData.length - 1] = candle;
                    return newData;
                  }
                  
                  // Add new candle
                  return [...prevData, candle];
                });
                
                // Update last price and last update time
                setLastPrice(candle.close);
                setLastUpdateTime(new Date().toLocaleTimeString());
              }
            }
            else if (data.type === 'error') {
              console.error('MEXC WebSocket error:', data.message);
            }
          } 
          // Handle standard message formats (from TwelveData or mock)
          else {
            if (data.type === 'price') {
              // Single price update
              const prevPrice = lastPrice;
              const newPrice = parseFloat(data.price);
              setLastPrice(newPrice);
              
              // Calculate price change
              if (prevPrice !== null) {
                const change = ((newPrice - prevPrice) / prevPrice) * 100;
                setPriceChange(change);
              }
              
              setLastUpdateTime(new Date().toLocaleTimeString());
            } 
            else if (data.type === 'ohlcv') {
              // OHLCV update for a candle
              const candle: OHLCVData = {
                time: data.timestamp / 1000, // Convert to seconds for lightweight-charts
                open: parseFloat(data.open),
                high: parseFloat(data.high),
                low: parseFloat(data.low),
                close: parseFloat(data.close),
                volume: parseFloat(data.volume)
              };
              
              // Update chart data
              setOhlcvData(prevData => {
                // Check if this is an update to the last candle
                if (prevData.length > 0 && prevData[prevData.length - 1].time === candle.time) {
                  const newData = [...prevData];
                  newData[newData.length - 1] = candle;
                  return newData;
                }
                
                // Add new candle
                return [...prevData, candle];
              });
              
              // Update last price and last update time
              setLastPrice(candle.close);
              setLastUpdateTime(new Date().toLocaleTimeString());
            }
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      // Store WebSocket instance
      setWsInstance(ws);
      
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      setIsConnected(false);
    }
  }, [selectedSymbol, selectedInterval, lastPrice]);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;
    
    // Create chart with type assertion to avoid TypeScript errors
    const chart = createChart(chartContainerRef.current, {
      width: width,
      height: height,
      layout: {
        backgroundColor: darkMode ? '#1E222D' : '#ffffff',
        textColor: darkMode ? '#d9d9d9' : '#191919',
      },
      grid: {
        vertLines: { color: darkMode ? '#2B2B43' : '#e1e1e1' },
        horzLines: { color: darkMode ? '#2B2B43' : '#e1e1e1' },
      },
      // Use type assertion for properties not in the TypeScript definition
      timeScale: {
        timeVisible: true,
        secondsVisible: true,
      },
      rightPriceScale: {
        borderVisible: true,
      },
    } as any);
    
    // Create candle series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });
    
    // Create volume series if enabled
    let volumeSeries = null;
    if (showVolume) {
      volumeSeries = chart.addHistogramSeries({
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
    }
    
    // Save references
    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;
    
    // Fit the chart to data on first load
    chart.timeScale().fitContent();
    
    // Cleanup
    return () => {
      if (chart) {
        // Use type assertion to handle the 'remove' method that exists at runtime but not in TypeScript definitions
        (chart as any).remove?.();
      }
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
    };
  }, [darkMode, height, width, showVolume]);

  // Connect to WebSocket on component mount or when data source/symbol/interval changes
  useEffect(() => {
    connectWebSocket();
    
    // Cleanup WebSocket on component unmount
    return () => {
      if (wsInstance) {
        wsInstance.close();
      }
    };
  }, [connectWebSocket, selectedDataSource, selectedSymbol, selectedInterval]);

  // Update chart data when ohlcvData changes
  useEffect(() => {
    if (!candleSeriesRef.current || ohlcvData.length === 0) return;
    
    // Update candle series
    candleSeriesRef.current.setData(ohlcvData);
    
    // Update volume series if enabled
    if (showVolume && volumeSeriesRef.current) {
      const volumeData = ohlcvData.map(candle => ({
        time: candle.time,
        value: candle.volume || 0,
        color: candle.close >= candle.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
      }));
      volumeSeriesRef.current.setData(volumeData);
    }
    
    // Fit the chart to data
    if (chartRef.current && ohlcvData.length > 0) {
      chartRef.current.timeScale().fitContent();
    }
  }, [ohlcvData, showVolume]);

  // Handle symbol change
  const handleSymbolChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newSymbol = e.target.value;
    setSelectedSymbol(newSymbol);
    
    // Reset data for new symbol
    setOhlcvData([]);
    setLastPrice(null);
    setPriceChange(0);
    
    // Notify parent if callback provided
    if (onSymbolChange) {
      onSymbolChange(newSymbol);
    }
  };

  // Handle interval change
  const handleIntervalChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newInterval = e.target.value as '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d';
    setSelectedInterval(newInterval);
    
    // Reset data for new interval
    setOhlcvData([]);
  };
  
  // Handle data source change
  const handleDataSourceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newDataSource = e.target.value;
    setSelectedDataSource(newDataSource);
    
    // Reset data for new data source
    setOhlcvData([]);
    setLastPrice(null);
    setPriceChange(0);
  };

  return (
    <div className="live-crypto-chart">
      <div className="chart-header">
        <div className="chart-controls">
          <div className="control-group">
            <label>Symbol:</label>
            <div className="symbol-selector">
              <select value={selectedSymbol} onChange={handleSymbolChange}>
                {AVAILABLE_PAIRS.map(pair => (
                  <option key={pair} value={pair}>{pair}</option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="control-group">
            <label>Interval:</label>
            <div className="interval-selector">
              <select value={selectedInterval} onChange={handleIntervalChange}>
                <option value="1m">1m</option>
                <option value="5m">5m</option>
                <option value="15m">15m</option>
                <option value="30m">30m</option>
                <option value="1h">1h</option>
                <option value="4h">4h</option>
                <option value="1d">1d</option>
              </select>
            </div>
          </div>
          
          <div className="control-group">
            <label>Data Source:</label>
            <div className="data-source-selector">
              <select value={selectedDataSource} onChange={handleDataSourceChange}>
                {DATA_SOURCES.map(source => (
                  <option key={source.value} value={source.value}>{source.label}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
        
        <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>
      
      <div className="price-info">
        {lastPrice !== null && (
          <>
            <div className="current-price">
              <span className="price-label">Price:</span>
              <span className="price-value">{lastPrice.toFixed(2)}</span>
            </div>
            
            <div className={`price-change ${priceChange >= 0 ? 'positive' : 'negative'}`}>
              {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
            </div>
          </>
        )}
        
        {lastUpdateTime && (
          <div className="last-update">
            Last update: {lastUpdateTime}
          </div>
        )}
      </div>
      
      <div ref={chartContainerRef} className="chart-container" />
      
      <div className="chart-footer">
          <div className="data-source">
            <strong>Data Source:</strong> {selectedDataSource === 'mexc' ? 'MEXC Exchange' : 
                            selectedDataSource === 'twelvedata' ? 'Twelve Data' : 'Mock Data'}
            {isConnected && selectedDataSource === 'mexc' && 
              <span className="mexc-badge">MEXC V3 API</span>}
          </div>
      </div>
    </div>
  );
};

export default LiveCryptoChart;
