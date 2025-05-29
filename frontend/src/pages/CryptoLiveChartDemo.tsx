import React, { useState } from 'react';
import LiveCryptoChart from '../components/crypto/LiveCryptoChart';
import '../styles/pages/CryptoLiveChartDemo.css';

const CryptoLiveChartDemo: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');

  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol);
  };

  return (
    <div className="crypto-live-chart-demo">
      <header className="demo-header">
        <h1>Live Cryptocurrency Chart</h1>
        <p className="demo-description">
          High-performance real-time chart powered by Twelve Data
        </p>
      </header>

      <div className="chart-container">
        <LiveCryptoChart 
          symbol={selectedSymbol}
          onSymbolChange={handleSymbolChange}
          height={600}
          showVolume={true}
        />
      </div>

      <div className="demo-footer">
        <p>
          This chart uses WebSocket technology to display real-time price data with minimal latency. 
          The backend connects to Twelve Data's premium API service which provides sub-second updates
          for cryptocurrency trading pairs.
        </p>
        <p>
          Features:
          <ul>
            <li>Real-time price updates with millisecond precision</li>
            <li>Multiple timeframes from 1-minute to daily</li>
            <li>Volume indicator</li>
            <li>Price change percentage</li>
            <li>Automatic reconnection if connection is lost</li>
          </ul>
        </p>
      </div>
    </div>
  );
};

export default CryptoLiveChartDemo;
