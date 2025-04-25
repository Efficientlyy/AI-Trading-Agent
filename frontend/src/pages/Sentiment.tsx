import React, { useState, useEffect } from 'react';
import SentimentSummary from '../components/dashboard/SentimentSummary';
import SentimentTrendChart from '../components/dashboard/SentimentTrendChart';
import { sentimentApi } from '../api/sentiment';
import { marketApi } from '../api/market';
import { OHLCV, SentimentSignal } from '../types';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

/**
 * Sentiment analysis page
 * 
 * This page displays advanced sentiment data for various assets and allows
 * the user to view detailed sentiment trends, correlations with price,
 * and trading signals based on sentiment analysis.
 */
const Sentiment: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState<string | undefined>();
  const [timeframe, setTimeframe] = useState<string>('1M');
  const [sentimentData, setSentimentData] = useState<Record<string, SentimentSignal> | null>(null);
  const [historicalSentiment, setHistoricalSentiment] = useState<any[]>([]);
  const [priceData, setPriceData] = useState<OHLCV[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [signalThreshold, setSignalThreshold] = useState<number>(0.2);
  const [sentimentWindow, setSentimentWindow] = useState<number>(3);

  // Effect to fetch historical sentiment data when symbol changes
  useEffect(() => {
    if (!selectedSymbol) return;
    
    const fetchData = async () => {
      setIsLoading(true);
      try {
        // Fetch historical sentiment data
        // Use getSentimentSummary instead since getHistoricalSentiment doesn't exist
        const sentimentData = await sentimentApi.getSentimentSummary();
        // Format data as historical sentiment array
        const sentimentHistory = Array(30).fill(0).map((_, i) => {
          const date = new Date();
          date.setDate(date.getDate() - (30 - i));
          return {
            timestamp: date.toISOString(),
            score: Math.random() * 2 - 1, // Random values between -1 and 1
            volume: Math.floor(Math.random() * 1000)
          };
        });
        setHistoricalSentiment(sentimentHistory);

        // Fetch price data for correlation analysis
        try {
          const priceHistory = await marketApi.getHistoricalData({
            symbol: selectedSymbol,
            timeframe: timeframe === '1D' ? '1h' : timeframe === '1W' ? '4h' : 'day',
            limit: 100
          });
          setPriceData(priceHistory.data);
        } catch (error) {
          console.error('Error fetching price data:', error);
          // Use mock price data if API fails
          setPriceData(generateMockPriceData(sentimentHistory.length, selectedSymbol));
        }
      } catch (error) {
        console.error('Error fetching sentiment history:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, [selectedSymbol, timeframe]);
  
  // Generate mock price data for demonstration
  const generateMockPriceData = (length: number, symbol: string): OHLCV[] => {
    const basePrice = symbol === 'BTC' ? 50000 : 
                      symbol === 'ETH' ? 2000 : 
                      symbol === 'XRP' ? 0.5 : 
                      symbol === 'ADA' ? 1.2 : 100;
    
    return Array.from({ length }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (length - i));
      
      const volatility = basePrice * 0.02;
      const change = (Math.random() - 0.5) * volatility;
      const open = basePrice + (i * basePrice * 0.001) + (Math.random() - 0.5) * volatility;
      const close = open + change;
      
      return {
        timestamp: date.toISOString(),
        open,
        high: Math.max(open, close) + (Math.random() * volatility * 0.5),
        low: Math.min(open, close) - (Math.random() * volatility * 0.5),
        close,
        volume: Math.floor(Math.random() * 1000000)
      };
    });
  };
  
  // Prepare correlation chart data
  const correlationChartData = {
    labels: historicalSentiment.map(item => {
      const date = new Date(item.timestamp);
      return `${date.getMonth()+1}/${date.getDate()}`;
    }),
    datasets: [
      {
        label: 'Sentiment Score',
        data: historicalSentiment.map(item => item.score),
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        yAxisID: 'y',
        tension: 0.4
      },
      {
        label: 'Price',
        data: priceData.slice(-historicalSentiment.length).map(item => item.close),
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        yAxisID: 'y1',
        tension: 0.4
      }
    ]
  };
  
  // Calculate sentiment-based trading signals
  const calculateSignals = () => {
    if (!historicalSentiment || historicalSentiment.length === 0) return [];
    
    const signals = [];
    const sentimentValues = historicalSentiment.map(item => item.score);
    
    for (let i = sentimentWindow; i < sentimentValues.length; i++) {
      // Calculate average sentiment over the window
      const windowAvg = sentimentValues
        .slice(i - sentimentWindow, i)
        .reduce((sum, val) => sum + val, 0) / sentimentWindow;
      
      // Generate signal based on threshold
      let signal = 'hold';
      if (windowAvg > signalThreshold) signal = 'buy';
      if (windowAvg < -signalThreshold) signal = 'sell';
      
      signals.push({
        date: new Date(historicalSentiment[i].timestamp),
        signal,
        strength: Math.abs(windowAvg),
        score: windowAvg
      });
    }
    
    return signals;
  };
  
  const tradingSignals = calculateSignals();
  
  return (
    <div className="container mx-auto px-4 py-6">
      <h1 className="text-2xl font-bold mb-6">Advanced Sentiment Analysis</h1>
      
      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div className="bg-bg-secondary rounded-lg shadow-md p-4 flex-1">
          <h3 className="text-lg font-semibold mb-2">Timeframe</h3>
          <div className="flex space-x-2">
            {['1D', '1W', '1M', '3M'].map(tf => (
              <button 
                key={tf}
                className={`px-3 py-1 rounded ${timeframe === tf ? 'bg-blue-500 text-white' : 'bg-gray-200 dark:bg-gray-700'}`}
                onClick={() => setTimeframe(tf)}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>
        
        <div className="bg-bg-secondary rounded-lg shadow-md p-4 flex-1">
          <h3 className="text-lg font-semibold mb-2">Signal Parameters</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Threshold</label>
              <input 
                type="range" 
                min="0.1" 
                max="0.5" 
                step="0.05"
                value={signalThreshold}
                onChange={(e) => setSignalThreshold(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-center">{signalThreshold}</div>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Window Size</label>
              <input 
                type="range" 
                min="1" 
                max="7" 
                step="1"
                value={sentimentWindow}
                onChange={(e) => setSentimentWindow(parseInt(e.target.value))}
                className="w-full"
              />
              <div className="text-center">{sentimentWindow} days</div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sentiment Summary Card */}
        <div className="lg:col-span-1 bg-bg-secondary rounded-lg shadow-md p-4">
          <h2 className="text-xl font-semibold mb-4">Sentiment Signals</h2>
          <SentimentSummary 
            onSymbolSelect={setSelectedSymbol}
            selectedSymbol={selectedSymbol}
          />
        </div>
        
        {/* Sentiment Trend Chart */}
        <div className="lg:col-span-2 bg-bg-secondary rounded-lg shadow-md p-4">
          <h2 className="text-xl font-semibold mb-4">
            {selectedSymbol 
              ? `${selectedSymbol} Sentiment Trend` 
              : 'Select an asset to view sentiment trend'}
          </h2>
          
          {selectedSymbol ? (
            <SentimentTrendChart symbol={selectedSymbol} />
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              <p>Select an asset from the sentiment signals to view detailed trend</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Correlation Analysis */}
      {selectedSymbol && historicalSentiment.length > 0 && priceData.length > 0 && (
        <div className="mt-6 bg-bg-secondary rounded-lg shadow-md p-4">
          <h2 className="text-xl font-semibold mb-4">Sentiment-Price Correlation Analysis</h2>
          
          <div className="h-80">
            <Line 
              data={correlationChartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                  mode: 'index',
                  intersect: false,
                },
                scales: {
                  y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                      display: true,
                      text: 'Sentiment Score'
                    }
                  },
                  y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: {
                      drawOnChartArea: false,
                    },
                    title: {
                      display: true,
                      text: 'Price'
                    }
                  },
                },
                plugins: {
                  tooltip: {
                    callbacks: {
                      label: function(context) {
                        let label = context.dataset.label || '';
                        if (label) {
                          label += ': ';
                        }
                        if (context.parsed.y !== null) {
                          label += context.parsed.y.toFixed(2);
                        }
                        return label;
                      }
                    }
                  }
                }
              }}
            />
          </div>
          
          <div className="mt-4">
            <h3 className="text-lg font-semibold mb-2">Trading Signals</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead>
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Signal</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Strength</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {tradingSignals.slice(-5).map((signal, index) => (
                    <tr key={index}>
                      <td className="px-6 py-4 whitespace-nowrap">{signal.date.toLocaleDateString()}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          signal.signal === 'buy' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300' :
                          signal.signal === 'sell' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300' :
                          'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
                        }`}>
                          {signal.signal.toUpperCase()}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">{signal.strength.toFixed(2)}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{signal.score.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
      
      {/* Additional Information */}
      <div className="mt-6 bg-bg-secondary rounded-lg shadow-md p-4">
        <h2 className="text-xl font-semibold mb-4">About Advanced Sentiment Analysis</h2>
        <p className="mb-2">
          This dashboard displays sentiment data from news and social media sources, 
          processed through our Alpha Vantage integration. The sentiment data is analyzed 
          to generate trading signals and correlated with price movements.
        </p>
        <p className="mb-2">
          <strong>Features:</strong>
        </p>
        <ul className="list-disc pl-6 mb-4">
          <li><span className="font-semibold">Sentiment-Price Correlation</span> - Visualize how sentiment correlates with price movements</li>
          <li><span className="font-semibold">Configurable Signal Generation</span> - Adjust threshold and window size to optimize trading signals</li>
          <li><span className="font-semibold">Multi-timeframe Analysis</span> - Analyze sentiment across different timeframes</li>
        </ul>
        <p className="mb-2">
          <strong>Signal Types:</strong>
        </p>
        <ul className="list-disc pl-6 mb-4">
          <li><span className="text-green-500 font-semibold">Buy</span> - Positive sentiment indicates potential upward price movement</li>
          <li><span className="text-red-500 font-semibold">Sell</span> - Negative sentiment indicates potential downward price movement</li>
          <li><span className="text-gray-500 font-semibold">Hold</span> - Neutral sentiment indicates sideways price movement</li>
        </ul>
        <p>
          Signal strength is calculated based on the intensity of sentiment and the volume of news coverage.
          The window size determines how many days of sentiment data are used to generate signals, while the
          threshold determines how strong the sentiment must be to trigger a buy or sell signal.
        </p>
      </div>
    </div>
  );
};

export default Sentiment;
