import React, { useState, useEffect } from 'react';
import SignalChart from '../TradingSignals/SignalChart';
import SentimentPriceChart from '../TradingSignals/SentimentPriceChart';
import { SignalData as SignalDataType } from '../../types/signals';

// Use the imported SignalData type but with a local alias to avoid conflicts
type LocalSignalData = SignalDataType;

interface SentimentData {
  timestamp: string | Date;
  sentiment_score: number;
  source: string;
  symbol?: string;
  article_count?: number;
}

interface PriceData {
  timestamp: string | Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TradingDashboardProps {
  symbol?: string;
  timeframe?: string;
  className?: string;
}

const TradingDashboard: React.FC<TradingDashboardProps> = ({
  symbol = 'BTC',
  timeframe = '1D',
  className
}) => {
  const [signals, setSignals] = useState<LocalSignalData[]>([]);
  const [sentimentData, setSentimentData] = useState<SentimentData[]>([]);
  const [priceData, setPriceData] = useState<PriceData[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState<string>(symbol);
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>(timeframe);
  const [dataSource, setDataSource] = useState<string>('all');

  // Fetch data when component mounts or when filters change
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // In a real implementation, these would be actual API calls
        // For now, we'll simulate with mock data
        await Promise.all([
          fetchTradingSignals(),
          fetchSentimentData(),
          fetchPriceData()
        ]);
        
        setIsLoading(false);
      } catch (err) {
        setError('Failed to fetch data. Please try again later.');
        setIsLoading(false);
        console.error('Error fetching data:', err);
      }
    };

    fetchData();
  }, [selectedSymbol, selectedTimeframe, dataSource]);

  // Mock data fetching functions
  const fetchTradingSignals = async () => {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Generate mock signals
    const mockSignals: LocalSignalData[] = [];
    const now = new Date();
    const sources = ['Twitter', 'Reddit', 'News', 'Technical', 'Combined'];
    const types = ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL', 'NEUTRAL'];
    
    // Generate signals for the past 30 days
    for (let i = 0; i < 50; i++) {
      const date = new Date(now);
      date.setDate(now.getDate() - Math.floor(Math.random() * 30));
      date.setHours(Math.floor(Math.random() * 24));
      
      mockSignals.push({
        id: `signal-${Date.now()}-${i}`,
        timestamp: date,
        type: types[Math.floor(Math.random() * types.length)],
        strength: Math.random() * 0.8 + 0.2, // Random between 0.2 and 1.0
        source: sources[Math.floor(Math.random() * sources.length)],
        symbol: selectedSymbol, // Symbol is now required
        confidence: Math.random() * 0.7 + 0.3, // Random between 0.3 and 1.0
        description: `${types[Math.floor(Math.random() * types.length)]} signal from ${sources[Math.floor(Math.random() * sources.length)]}`
      });
    }
    
    // Sort by timestamp
    mockSignals.sort((a, b) => {
      return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
    });
    
    setSignals(mockSignals);
    return mockSignals;
  };
  
  const fetchSentimentData = async () => {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Generate mock sentiment data
    const mockSentimentData: SentimentData[] = [];
    const now = new Date();
    const sources = ['Twitter', 'Reddit', 'Alpha Vantage', 'News API', 'Fear & Greed Index'];
    
    // Generate data points for the past 30 days
    for (let i = 0; i < 100; i++) {
      const date = new Date(now);
      date.setDate(now.getDate() - Math.floor(Math.random() * 30));
      date.setHours(Math.floor(Math.random() * 24));
      
      // Sentiment score between -1 and 1
      const sentimentScore = Math.random() * 2 - 1;
      
      mockSentimentData.push({
        timestamp: date,
        sentiment_score: sentimentScore,
        source: sources[Math.floor(Math.random() * sources.length)],
        symbol: selectedSymbol,
        article_count: Math.floor(Math.random() * 20) + 1
      });
    }
    
    // Sort by timestamp
    mockSentimentData.sort((a, b) => {
      return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
    });
    
    setSentimentData(mockSentimentData);
    return mockSentimentData;
  };
  
  const fetchPriceData = async () => {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1200));
    
    // Generate mock price data
    const mockPriceData: PriceData[] = [];
    const now = new Date();
    
    // Base price for different symbols
    const basePrice = selectedSymbol === 'BTC' ? 50000 : 
                     selectedSymbol === 'ETH' ? 3000 :
                     selectedSymbol === 'AAPL' ? 150 :
                     selectedSymbol === 'MSFT' ? 300 : 100;
    
    // Generate price data for the past 30 days
    let currentPrice = basePrice;
    for (let i = 30; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(now.getDate() - i);
      
      // Generate daily OHLC data
      const changePercent = (Math.random() * 6 - 3) / 100; // -3% to +3%
      const open = currentPrice;
      const close = open * (1 + changePercent);
      const high = Math.max(open, close) * (1 + Math.random() * 0.02); // Up to 2% higher
      const low = Math.min(open, close) * (1 - Math.random() * 0.02); // Up to 2% lower
      const volume = Math.floor(Math.random() * 10000) + 5000;
      
      mockPriceData.push({
        timestamp: date,
        open,
        high,
        low,
        close,
        volume
      });
      
      // Update current price for next iteration
      currentPrice = close;
    }
    
    // Add intraday data for today
    const today = new Date(now);
    today.setHours(0, 0, 0, 0);
    
    for (let hour = 0; hour < now.getHours(); hour++) {
      const date = new Date(today);
      date.setHours(hour);
      
      const changePercent = (Math.random() * 2 - 1) / 100; // -1% to +1%
      const open = currentPrice;
      const close = open * (1 + changePercent);
      const high = Math.max(open, close) * (1 + Math.random() * 0.01);
      const low = Math.min(open, close) * (1 - Math.random() * 0.01);
      const volume = Math.floor(Math.random() * 1000) + 500;
      
      mockPriceData.push({
        timestamp: date,
        open,
        high,
        low,
        close,
        volume
      });
      
      currentPrice = close;
    }
    
    // Sort by timestamp
    mockPriceData.sort((a, b) => {
      return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
    });
    
    setPriceData(mockPriceData);
    return mockPriceData;
  };

  // Filter signals by source
  const filteredSignals = dataSource === 'all' 
    ? signals 
    : signals.filter(signal => signal.source?.toLowerCase() === dataSource.toLowerCase());

  // Filter sentiment data by source
  const filteredSentiment = dataSource === 'all'
    ? sentimentData
    : sentimentData.filter(data => data.source?.toLowerCase().includes(dataSource.toLowerCase()));

  return (
    <div className={`trading-dashboard ${className || ''}`}>
      <div className="dashboard-header">
        <h2>Trading Dashboard</h2>
        <div className="dashboard-controls">
          <div className="control-group">
            <label htmlFor="symbol-select">Symbol:</label>
            <select 
              id="symbol-select"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
            >
              <option value="BTC">Bitcoin (BTC)</option>
              <option value="ETH">Ethereum (ETH)</option>
              <option value="AAPL">Apple (AAPL)</option>
              <option value="MSFT">Microsoft (MSFT)</option>
              <option value="AMZN">Amazon (AMZN)</option>
            </select>
          </div>
          
          <div className="control-group">
            <label htmlFor="timeframe-select">Timeframe:</label>
            <select 
              id="timeframe-select"
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
            >
              <option value="1D">1 Day</option>
              <option value="1W">1 Week</option>
              <option value="1M">1 Month</option>
              <option value="3M">3 Months</option>
              <option value="6M">6 Months</option>
              <option value="1Y">1 Year</option>
              <option value="ALL">All Time</option>
            </select>
          </div>
          
          <div className="control-group">
            <label htmlFor="source-select">Data Source:</label>
            <select 
              id="source-select"
              value={dataSource}
              onChange={(e) => setDataSource(e.target.value)}
            >
              <option value="all">All Sources</option>
              <option value="twitter">Twitter</option>
              <option value="reddit">Reddit</option>
              <option value="news">News</option>
              <option value="alpha">Alpha Vantage</option>
              <option value="technical">Technical</option>
            </select>
          </div>
        </div>
      </div>
      
      {isLoading ? (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading data...</p>
        </div>
      ) : error ? (
        <div className="error-container">
          <p>{error}</p>
          <button onClick={() => window.location.reload()}>Retry</button>
        </div>
      ) : (
        <div className="dashboard-content">
          <div className="dashboard-row">
            <div className="dashboard-card full-width">
              <h3>Price & Sentiment Analysis</h3>
              <SentimentPriceChart 
                sentimentData={filteredSentiment}
                priceData={priceData}
                symbol={selectedSymbol}
                timeframe={selectedTimeframe}
                width={1200}
                height={500}
              />
            </div>
          </div>
          
          <div className="dashboard-row">
            <div className="dashboard-card">
              <h3>Trading Signals</h3>
              <SignalChart 
                signals={filteredSignals}
                width={580}
                height={350}
                title={`${selectedSymbol} Trading Signals`}
              />
            </div>
            
            <div className="dashboard-card">
              <h3>Signal Statistics</h3>
              <div className="stats-container">
                <div className="stat-item">
                  <div className="stat-label">Total Signals</div>
                  <div className="stat-value">{filteredSignals.length}</div>
                </div>
                
                <div className="stat-item">
                  <div className="stat-label">Buy Signals</div>
                  <div className="stat-value">{
                    filteredSignals.filter(s => s.type === 'BUY' || s.type === 'STRONG_BUY').length
                  }</div>
                </div>
                
                <div className="stat-item">
                  <div className="stat-label">Sell Signals</div>
                  <div className="stat-value">{
                    filteredSignals.filter(s => s.type === 'SELL' || s.type === 'STRONG_SELL').length
                  }</div>
                </div>
                
                <div className="stat-item">
                  <div className="stat-label">Average Strength</div>
                  <div className="stat-value">{
                    filteredSignals.length > 0 
                      ? (filteredSignals.reduce((sum, s) => sum + s.strength, 0) / filteredSignals.length).toFixed(2)
                      : '0.00'
                  }</div>
                </div>
                
                <div className="stat-item">
                  <div className="stat-label">Average Sentiment</div>
                  <div className="stat-value">{
                    filteredSentiment.length > 0
                      ? (filteredSentiment.reduce((sum, s) => sum + s.sentiment_score, 0) / filteredSentiment.length).toFixed(2)
                      : '0.00'
                  }</div>
                </div>
                
                <div className="stat-item">
                  <div className="stat-label">Signal Confidence</div>
                  <div className="stat-value">{
                    filteredSignals.length > 0 && filteredSignals.some(s => s.confidence !== undefined)
                      ? (filteredSignals.reduce((sum, s) => sum + (s.confidence || 0), 0) / 
                         filteredSignals.filter(s => s.confidence !== undefined).length).toFixed(2)
                      : '0.00'
                  }</div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="dashboard-row">
            <div className="dashboard-card full-width">
              <h3>Recent Trading Signals</h3>
              <div className="signals-table-container">
                <table className="signals-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Symbol</th>
                      <th>Type</th>
                      <th>Strength</th>
                      <th>Source</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredSignals
                      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
                      .slice(0, 10)
                      .map((signal, index) => (
                        <tr key={index}>
                          <td>{new Date(signal.timestamp).toLocaleString()}</td>
                          <td>{signal.symbol}</td>
                          <td className={`signal-type ${signal.type.toLowerCase()}`}>{signal.type}</td>
                          <td>{signal.strength.toFixed(2)}</td>
                          <td>{signal.source}</td>
                          <td>{signal.confidence ? signal.confidence.toFixed(2) : 'N/A'}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
      
      <style>{`
        .trading-dashboard {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
          padding: 20px;
          background-color: #f5f7fa;
          border-radius: 8px;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 24px;
        }
        
        .dashboard-header h2 {
          margin: 0;
          font-size: 24px;
          font-weight: 600;
          color: #333;
        }
        
        .dashboard-controls {
          display: flex;
          gap: 16px;
        }
        
        .control-group {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .control-group label {
          font-size: 14px;
          font-weight: 500;
          color: #666;
        }
        
        .control-group select {
          padding: 8px 12px;
          border: 1px solid #ddd;
          border-radius: 4px;
          background-color: white;
          font-size: 14px;
          color: #333;
        }
        
        .dashboard-content {
          display: flex;
          flex-direction: column;
          gap: 24px;
        }
        
        .dashboard-row {
          display: flex;
          gap: 24px;
        }
        
        .dashboard-card {
          flex: 1;
          background-color: white;
          border-radius: 8px;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
          padding: 16px;
        }
        
        .dashboard-card h3 {
          margin-top: 0;
          margin-bottom: 16px;
          font-size: 18px;
          font-weight: 600;
          color: #333;
        }
        
        .full-width {
          width: 100%;
        }
        
        .loading-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 300px;
        }
        
        .loading-spinner {
          width: 40px;
          height: 40px;
          border: 4px solid rgba(0, 0, 0, 0.1);
          border-radius: 50%;
          border-top-color: #2196F3;
          animation: spin 1s ease-in-out infinite;
          margin-bottom: 16px;
        }
        
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        
        .error-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 200px;
          color: #d32f2f;
        }
        
        .error-container button {
          margin-top: 16px;
          padding: 8px 16px;
          background-color: #2196F3;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }
        
        .stats-container {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 16px;
        }
        
        .stat-item {
          background-color: #f5f7fa;
          border-radius: 8px;
          padding: 16px;
          text-align: center;
        }
        
        .stat-label {
          font-size: 14px;
          color: #666;
          margin-bottom: 8px;
        }
        
        .stat-value {
          font-size: 24px;
          font-weight: 600;
          color: #333;
        }
        
        .signals-table-container {
          overflow-x: auto;
        }
        
        .signals-table {
          width: 100%;
          border-collapse: collapse;
        }
        
        .signals-table th,
        .signals-table td {
          padding: 12px 16px;
          text-align: left;
          border-bottom: 1px solid #eee;
        }
        
        .signals-table th {
          font-weight: 600;
          color: #666;
          background-color: #f5f7fa;
        }
        
        .signals-table tr:hover {
          background-color: #f9f9f9;
        }
        
        .signal-type {
          font-weight: 600;
        }
        
        .signal-type.buy,
        .signal-type.strong_buy {
          color: #4CAF50;
        }
        
        .signal-type.sell,
        .signal-type.strong_sell {
          color: #F44336;
        }
        
        .signal-type.neutral {
          color: #9E9E9E;
        }
      `}</style>
    </div>
  );
};

export default TradingDashboard;
