import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart } from 'lightweight-charts';
import './LiveCryptoChart.css';

interface MexcLiveChartProps {
  symbol?: string;
  interval?: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d';
  darkMode?: boolean;
  height?: number;
  width?: number;
  showVolume?: boolean;
  showOrderbook?: boolean;
}

interface OHLCVData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface OrderbookData {
  timestamp: number;
  bid: number;
  ask: number;
  bid_qty: number;
  ask_qty: number;
}

const MexcLiveChart: React.FC<MexcLiveChartProps> = ({
  symbol = 'BTC/USDC',
  interval = '1m',
  darkMode = true,
  height = 500,
  width = 800,
  showVolume = true,
  showOrderbook = true
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
  const [selectedInterval, setSelectedInterval] = useState<string>(interval);
  const [lastUpdateTime, setLastUpdateTime] = useState<string>('');
  const [orderbookData, setOrderbookData] = useState<OrderbookData | null>(null);
  const [klineWs, setKlineWs] = useState<WebSocket | null>(null);
  const [tickerWs, setTickerWs] = useState<WebSocket | null>(null);
  const [orderbookWs, setOrderbookWs] = useState<WebSocket | null>(null);
  
  // Format symbol for API paths
  const getFormattedSymbol = useCallback(() => {
    return symbol.replace('/', '-').toLowerCase();
  }, [symbol]);

  // Connect to kline WebSocket
  const connectKlineWebSocket = useCallback(() => {
    const formattedSymbol = getFormattedSymbol();
    
    try {
      // Close existing connection if any
      if (klineWs) {
        klineWs.close();
      }
      
      // Connect to our specialized MEXC WebSocket endpoint
      const ws = new WebSocket(`ws://localhost:8000/ws/mexc/kline/${formattedSymbol}/${selectedInterval}`);
      
      ws.onopen = () => {
        console.log(`MEXC Kline WebSocket connected for ${symbol} (${selectedInterval})`);
        setIsConnected(true);
      };
      
      ws.onclose = () => {
        console.log('MEXC Kline WebSocket disconnected');
        setIsConnected(false);
        
        // Attempt to reconnect after 2 seconds
        setTimeout(() => {
          connectKlineWebSocket();
        }, 2000);
      };
      
      ws.onerror = (error) => {
        console.error('MEXC Kline WebSocket error:', error);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'kline') {
            // Process kline data
            const candle: OHLCVData = {
              time: Math.floor(data.data.timestamp / 1000), // Convert to seconds for lightweight-charts
              open: data.data.open,
              high: data.data.high,
              low: data.data.low,
              close: data.data.close,
              volume: data.data.volume
            };
            
            // Update chart data
            setOhlcvData(prevData => {
              // Check if this is an update to the last candle
              if (prevData.length > 0 && prevData[prevData.length - 1].time === candle.time) {
                const newData = [...prevData];
                newData[newData.length - 1] = candle;
                return newData;
              } else {
                // New candle
                return [...prevData, candle].slice(-300); // Keep last 300 candles
              }
            });
            
            // Update price and timestamp
            setLastPrice(data.data.close);
            setLastUpdateTime(new Date().toLocaleTimeString());
          } 
          else if (data.type === 'historical_klines') {
            // Process historical kline data
            if (Array.isArray(data.data) && data.data.length > 0) {
              const historicalCandles: OHLCVData[] = data.data.map((item: any) => ({
                time: Math.floor(item.timestamp / 1000),
                open: item.open,
                high: item.high,
                low: item.low,
                close: item.close,
                volume: item.volume
              }));
              
              setOhlcvData(historicalCandles);
              
              // Update price with the latest candle
              const latestCandle = historicalCandles[historicalCandles.length - 1];
              setLastPrice(latestCandle.close);
              setLastUpdateTime(new Date().toLocaleTimeString());
            }
          }
        } catch (error) {
          console.error('Error parsing MEXC kline data:', error);
        }
      };
      
      setKlineWs(ws);
    } catch (error) {
      console.error('Error connecting to MEXC kline WebSocket:', error);
      setIsConnected(false);
    }
  }, [getFormattedSymbol, selectedInterval, klineWs, symbol]);
  
  // Connect to ticker WebSocket
  const connectTickerWebSocket = useCallback(() => {
    const formattedSymbol = getFormattedSymbol();
    
    try {
      // Close existing connection if any
      if (tickerWs) {
        tickerWs.close();
      }
      
      // Connect to our specialized MEXC WebSocket endpoint
      const ws = new WebSocket(`ws://localhost:8000/ws/mexc/ticker/${formattedSymbol}`);
      
      ws.onopen = () => {
        console.log(`MEXC Ticker WebSocket connected for ${symbol}`);
      };
      
      ws.onclose = () => {
        console.log('MEXC Ticker WebSocket disconnected');
        
        // Attempt to reconnect after 2 seconds
        setTimeout(() => {
          connectTickerWebSocket();
        }, 2000);
      };
      
      ws.onerror = (error) => {
        console.error('MEXC Ticker WebSocket error:', error);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'ticker') {
            // Process ticker data
            const newPrice = data.data.price;
            const prevPrice = lastPrice;
            
            setLastPrice(newPrice);
            
            // Calculate price change
            if (prevPrice !== null) {
              const change = ((newPrice - prevPrice) / prevPrice) * 100;
              setPriceChange(change);
            }
            
            setLastUpdateTime(new Date().toLocaleTimeString());
          }
        } catch (error) {
          console.error('Error parsing MEXC ticker data:', error);
        }
      };
      
      setTickerWs(ws);
    } catch (error) {
      console.error('Error connecting to MEXC ticker WebSocket:', error);
    }
  }, [getFormattedSymbol, lastPrice, symbol]);
  
  // Connect to orderbook WebSocket
  const connectOrderbookWebSocket = useCallback(() => {
    if (!showOrderbook) return;
    
    const formattedSymbol = getFormattedSymbol();
    
    try {
      // Close existing connection if any
      if (orderbookWs) {
        orderbookWs.close();
      }
      
      // Connect to our specialized MEXC WebSocket endpoint
      const ws = new WebSocket(`ws://localhost:8000/ws/mexc/orderbook/${formattedSymbol}`);
      
      ws.onopen = () => {
        console.log(`MEXC Orderbook WebSocket connected for ${symbol}`);
      };
      
      ws.onclose = () => {
        console.log('MEXC Orderbook WebSocket disconnected');
        
        // Attempt to reconnect after 2 seconds
        setTimeout(() => {
          connectOrderbookWebSocket();
        }, 2000);
      };
      
      ws.onerror = (error) => {
        console.error('MEXC Orderbook WebSocket error:', error);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'orderbook') {
            // Process orderbook data
            setOrderbookData({
              timestamp: data.data.timestamp,
              bid: data.data.bid,
              ask: data.data.ask,
              bid_qty: data.data.bid_qty,
              ask_qty: data.data.ask_qty
            });
          }
        } catch (error) {
          console.error('Error parsing MEXC orderbook data:', error);
        }
      };
      
      setOrderbookWs(ws);
    } catch (error) {
      console.error('Error connecting to MEXC orderbook WebSocket:', error);
    }
  }, [getFormattedSymbol, showOrderbook, symbol, orderbookWs]);

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
        // Use proper chart cleanup method
        try {
          // For newer versions that have the remove method
          // Use type assertion to avoid TypeScript errors
          const chartInstance = chart as any;
          if (typeof chartInstance.remove === 'function') {
            chartInstance.remove();
          } else {
            // For older versions or as a fallback, try another approach
            // Since we can't directly access chart.element, we clear the container
            if (chartContainerRef.current) {
              chartContainerRef.current.innerHTML = '';
            }
          }
        } catch (e) {
          console.error('Error cleaning up chart:', e);
        }
      }
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
    };
  }, [darkMode, height, width, showVolume]);

  // Connect to WebSockets on component mount
  useEffect(() => {
    connectKlineWebSocket();
    connectTickerWebSocket();
    if (showOrderbook) {
      connectOrderbookWebSocket();
    }
    
    // Cleanup WebSockets on component unmount
    return () => {
      if (klineWs) {
        klineWs.close();
      }
      if (tickerWs) {
        tickerWs.close();
      }
      if (orderbookWs) {
        orderbookWs.close();
      }
    };
  }, [connectKlineWebSocket, connectTickerWebSocket, connectOrderbookWebSocket, showOrderbook]);

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

  // Handle interval change
  const handleIntervalChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newInterval = e.target.value as '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d';
    setSelectedInterval(newInterval);
    
    // Reset data for new interval
    setOhlcvData([]);
  };

  // Calculate spread
  const getSpread = () => {
    if (!orderbookData) return null;
    
    const spread = orderbookData.ask - orderbookData.bid;
    const spreadPercentage = (spread / orderbookData.bid) * 100;
    
    return {
      value: spread,
      percentage: spreadPercentage
    };
  };

  const spread = getSpread();

  return (
    <div className="live-crypto-chart mexc-chart">
      <div className="chart-header">
        <div className="symbol-info">
          <span className="symbol-label">MEXC</span>
          <span className="symbol-value">{symbol}</span>
        </div>
        
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
            
            {orderbookData && spread && (
              <div className="orderbook-info">
                <div className="spread-info">
                  <span className="spread-label">Spread:</span>
                  <span className="spread-value">{spread.value.toFixed(2)} ({spread.percentage.toFixed(2)}%)</span>
                </div>
                <div className="bid-ask">
                  <span className="bid">Bid: {orderbookData.bid.toFixed(2)} ({orderbookData.bid_qty.toFixed(4)})</span>
                  <span className="ask">Ask: {orderbookData.ask.toFixed(2)} ({orderbookData.ask_qty.toFixed(4)})</span>
                </div>
              </div>
            )}
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
          Live Data from MEXC Exchange
        </div>
      </div>
    </div>
  );
};

export default MexcLiveChart;
