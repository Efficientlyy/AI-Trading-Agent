import React, { useState, ChangeEvent } from 'react';
import MexcLiveChart from '../components/crypto/MexcLiveChart';

// Custom styles
import '../styles/MexcLiveChartDemo.css';

const MexcLiveChartDemo: React.FC = () => {
  const [symbol, setSymbol] = useState<string>('BTC/USDC');
  const [interval, setInterval] = useState<'1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d'>('1m');
  const [darkMode, setDarkMode] = useState<boolean>(true);
  const [showVolume, setShowVolume] = useState<boolean>(true);
  const [showOrderbook, setShowOrderbook] = useState<boolean>(true);

  // List of available trading pairs on MEXC
  const availablePairs = [
    'BTC/USDC', 
    'ETH/USDC', 
    'SOL/USDC', 
    'XRP/USDC',
    'BNB/USDC',
    'ADA/USDC',
    'DOGE/USDC',
    'AVAX/USDC'
  ];

  // List of available intervals
  const availableIntervals = [
    { value: '1m', label: '1 Minute' },
    { value: '5m', label: '5 Minutes' },
    { value: '15m', label: '15 Minutes' },
    { value: '30m', label: '30 Minutes' },
    { value: '1h', label: '1 Hour' },
    { value: '4h', label: '4 Hours' },
    { value: '1d', label: '1 Day' }
  ];

  return (
    <div className="container mt-4">
      <h2>MEXC Live Chart Demo</h2>
      <p className="text-muted">
        Real-time BTC/USDC price data from MEXC Exchange. This component demonstrates
        the integration with MEXC for obtaining live market data using WebSockets.
      </p>
      
      <div className="row mb-4">
        <div className="col-md-3">
          <div className="card shadow-sm">
            <div className="card-header">Chart Settings</div>
            <div className="card-body">
              <form>
                <div className="mb-3">
                  <label>Trading Pair</label>
                  <select 
                    className="form-select"
                    value={symbol}
                    onChange={(e: ChangeEvent<HTMLSelectElement>) => setSymbol(e.target.value)}
                  >
                    {availablePairs.map(pair => (
                      <option key={pair} value={pair}>{pair}</option>
                    ))}
                  </select>
                </div>
                
                <div className="mb-3">
                  <label>Time Interval</label>
                  <select
                    className="form-select" 
                    value={interval}
                    onChange={(e: ChangeEvent<HTMLSelectElement>) => setInterval(e.target.value as '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d')}
                  >
                    {availableIntervals.map(int => (
                      <option key={int.value} value={int.value}>{int.label}</option>
                    ))}
                  </select>
                </div>
                
                <div className="mb-3">
                  <div className="form-check form-switch">
                    <input 
                      className="form-check-input"
                      type="checkbox"
                      id="dark-mode-switch"
                      checked={darkMode}
                      onChange={(e: ChangeEvent<HTMLInputElement>) => setDarkMode(e.target.checked)}
                    />
                    <label className="form-check-label" htmlFor="dark-mode-switch">Dark Mode</label>
                  </div>
                </div>
                
                <div className="mb-3">
                  <div className="form-check form-switch">
                    <input 
                      className="form-check-input"
                      type="checkbox"
                      id="volume-switch"
                      checked={showVolume}
                      onChange={(e: ChangeEvent<HTMLInputElement>) => setShowVolume(e.target.checked)}
                    />
                    <label className="form-check-label" htmlFor="volume-switch">Show Volume</label>
                  </div>
                </div>
                
                <div className="mb-3">
                  <div className="form-check form-switch">
                    <input 
                      className="form-check-input"
                      type="checkbox"
                      id="orderbook-switch"
                      checked={showOrderbook}
                      onChange={(e: ChangeEvent<HTMLInputElement>) => setShowOrderbook(e.target.checked)}
                    />
                    <label className="form-check-label" htmlFor="orderbook-switch">Show Orderbook</label>
                  </div>
                </div>
              </form>
            </div>
          </div>
          
          <div className="card mt-3 shadow-sm">
            <div className="card-header">Connection Info</div>
            <div className="card-body">
              <p className="mb-1"><strong>Data Source:</strong> MEXC Exchange</p>
              <p className="mb-1"><strong>Connection Type:</strong> WebSocket</p>
              <p className="mb-1"><strong>Update Frequency:</strong> Real-time</p>
              <p className="mb-0"><strong>Default Pair:</strong> BTC/USDC</p>
            </div>
          </div>
        </div>
        
        <div className="col-md-9">
          <div className="card shadow">
            <div className="card-body">
              <MexcLiveChart 
                symbol={symbol}
                interval={interval}
                darkMode={darkMode}
                showVolume={showVolume}
                showOrderbook={showOrderbook}
                height={600}
                width={1000}
              />
            </div>
          </div>
          
          <div className="card mt-3 shadow-sm">
            <div className="card-header">About This Chart</div>
            <div className="card-body">
              <p>
                This chart displays real-time market data from MEXC Exchange for the selected trading pair.
                It uses WebSockets to establish a direct connection to the exchange, ensuring low-latency
                updates for price, volume, and orderbook data.
              </p>
              <p>
                Features include:
              </p>
              <ul>
                <li>Real-time price updates via WebSocket</li>
                <li>Candlestick chart with customizable timeframes</li>
                <li>Volume indicator</li>
                <li>Orderbook data (best bid/ask)</li>
                <li>Spread calculation</li>
                <li>Dark/light mode toggle</li>
              </ul>
              <p>
                This component is part of the AI Trading Agent project, which aims to provide
                algorithmic trading capabilities for various asset classes including cryptocurrencies.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MexcLiveChartDemo;
