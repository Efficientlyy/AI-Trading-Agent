import React, { useState, useEffect, useCallback } from 'react';
import { format } from 'date-fns';
import { Link } from 'react-router-dom';
import { Card, CardHeader, CardBody } from '../Card';
import { Select, Spinner, Button } from '../Form';
import { Alert } from '../Alert';
import { Table } from '../Table';
import { alphaVantageApi, SentimentSummary } from '../../api/alphaVantage';
import { SignalData } from '../../types/signals';
import { createSignalNotification } from '../../services/NotificationService';
import { NotificationType } from '../../types/notifications';

// Sentiment score visualization component
const SentimentScoreGauge: React.FC<{ score: number; size?: number }> = ({ score, size = 120 }) => {
  // Map score from -1 to 1 range to 0 to 100 for the gauge
  const normalizedScore = ((score + 1) / 2) * 100;
  
  // Determine color based on sentiment score
  const getColor = (score: number) => {
    if (score < -0.3) return '#e74c3c'; // Red for negative
    if (score > 0.3) return '#2ecc71'; // Green for positive
    return '#f39c12'; // Yellow for neutral
  };
  
  const color = getColor(score);
  const radius = size / 2;
  const strokeWidth = size / 10;
  const circumference = 2 * Math.PI * (radius - strokeWidth / 2);
  const progress = (normalizedScore / 100) * circumference;
  
  return (
    <div className="sentiment-gauge" style={{ width: size, height: size, position: 'relative' }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background circle */}
        <circle
          cx={radius}
          cy={radius}
          r={radius - strokeWidth / 2}
          fill="transparent"
          stroke="#e0e0e0"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={0}
        />
        
        {/* Progress arc */}
        <circle
          cx={radius}
          cy={radius}
          r={radius - strokeWidth / 2}
          fill="transparent"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={circumference - progress}
          strokeLinecap="round"
          transform={`rotate(-90 ${radius} ${radius})`}
        />
        
        {/* Score text */}
        <text
          x={radius}
          y={radius}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={size / 5}
          fontWeight="bold"
          fill={color}
        >
          {score.toFixed(2)}
        </text>
      </svg>
    </div>
  );
};

// Sentiment distribution chart
const SentimentDistribution: React.FC<{ 
  positive: number; 
  negative: number; 
  neutral: number; 
  width?: number;
  height?: number;
}> = ({ positive, negative, neutral, width = 300, height = 100 }) => {
  const total = positive + negative + neutral;
  const positiveWidth = (positive / total) * width;
  const negativeWidth = (negative / total) * width;
  const neutralWidth = (neutral / total) * width;
  
  return (
    <div className="sentiment-distribution" style={{ width, height: height + 30 }}>
      <div className="bars" style={{ display: 'flex', height }}>
        <div 
          className="positive" 
          style={{ 
            width: positiveWidth, 
            backgroundColor: '#2ecc71',
            position: 'relative',
            borderTopLeftRadius: '4px',
            borderBottomLeftRadius: '4px'
          }}
        >
          <span className="count" style={{ 
            position: 'absolute', 
            top: '50%', 
            left: '50%', 
            transform: 'translate(-50%, -50%)',
            color: 'white',
            fontWeight: 'bold',
            fontSize: '14px'
          }}>
            {positive}
          </span>
        </div>
        <div 
          className="neutral" 
          style={{ 
            width: neutralWidth, 
            backgroundColor: '#f39c12',
            position: 'relative'
          }}
        >
          <span className="count" style={{ 
            position: 'absolute', 
            top: '50%', 
            left: '50%', 
            transform: 'translate(-50%, -50%)',
            color: 'white',
            fontWeight: 'bold',
            fontSize: '14px'
          }}>
            {neutral}
          </span>
        </div>
        <div 
          className="negative" 
          style={{ 
            width: negativeWidth, 
            backgroundColor: '#e74c3c',
            position: 'relative',
            borderTopRightRadius: '4px',
            borderBottomRightRadius: '4px'
          }}
        >
          <span className="count" style={{ 
            position: 'absolute', 
            top: '50%', 
            left: '50%', 
            transform: 'translate(-50%, -50%)',
            color: 'white',
            fontWeight: 'bold',
            fontSize: '14px'
          }}>
            {negative}
          </span>
        </div>
      </div>
      <div className="labels" style={{ display: 'flex', marginTop: '8px', fontSize: '12px' }}>
        <div style={{ width: positiveWidth, textAlign: 'center' }}>Positive</div>
        <div style={{ width: neutralWidth, textAlign: 'center' }}>Neutral</div>
        <div style={{ width: negativeWidth, textAlign: 'center' }}>Negative</div>
      </div>
    </div>
  );
};

// Ticker sentiment card
const TickerSentimentCard: React.FC<{ 
  ticker: string; 
  sentimentScore: number; 
  relevanceScore: number;
}> = ({ ticker, sentimentScore, relevanceScore }) => {
  return (
    <div className="ticker-sentiment-card" style={{ 
      backgroundColor: 'white',
      borderRadius: '8px',
      padding: '16px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      width: '180px',
      margin: '8px'
    }}>
      <div className="ticker" style={{ 
        fontSize: '24px', 
        fontWeight: 'bold',
        marginBottom: '8px'
      }}>
        {ticker}
      </div>
      <SentimentScoreGauge score={sentimentScore} size={100} />
      <div className="relevance" style={{ 
        marginTop: '8px',
        fontSize: '14px',
        color: '#666'
      }}>
        Relevance: {(relevanceScore * 100).toFixed(0)}%
      </div>
    </div>
  );
};

// Available topics for sentiment analysis
const AVAILABLE_TOPICS = [
  'blockchain',
  'crypto',
  'bitcoin',
  'ethereum',
  'defi',
  'nft',
  'metaverse',
  'web3',
  'fintech',
  'stocks',
  'forex',
  'commodities'
];

const SentimentDashboard: React.FC = () => {
  // State for filters and data
  const [selectedTopic, setSelectedTopic] = useState<string>('blockchain');
  const [sentimentData, setSentimentData] = useState<SentimentSummary | null>(null);
  const [signals, setSignals] = useState<SignalData[]>([]);
  
  // UI state
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Fetch sentiment data
  const fetchSentimentData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await alphaVantageApi.getSentimentByTopic(selectedTopic);
      setSentimentData(data);
      
      // Convert sentiment data to signals
      const newSignals = alphaVantageApi.convertSentimentToSignals(data);
      setSignals(newSignals);
      
      // Create notifications for strong signals
      newSignals.forEach(signal => {
        if (signal.strength > 0.7) {
          createSignalNotification(signal);
        }
      });
    } catch (error) {
      console.error('Error fetching sentiment data:', error);
      setError('Failed to fetch sentiment data. Please try again later.');
    } finally {
      setLoading(false);
    }
  }, [selectedTopic]);
  
  // Fetch data when topic changes
  useEffect(() => {
    fetchSentimentData();
  }, [fetchSentimentData]);
  
  // Handle topic change
  const handleTopicChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedTopic(event.target.value);
  };
  
  // Handle refresh
  const handleRefresh = () => {
    fetchSentimentData();
  };
  
  return (
    <div className="sentiment-dashboard">
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Sentiment Analysis Dashboard</h2>
            <div className="flex space-x-4">
              <Link to="/advanced-signals" className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
                View Advanced Signals
              </Link>
              <div className="w-48">
                <Select
                  value={selectedTopic}
                  onChange={handleTopicChange}
                  label="Topic"
                  className="w-full"
                >
                  {AVAILABLE_TOPICS.map(topic => (
                    <option key={topic} value={topic}>
                      {topic.charAt(0).toUpperCase() + topic.slice(1)}
                    </option>
                  ))}
                </Select>
              </div>
              <Button
                onClick={handleRefresh}
                disabled={loading}
                variant="primary"
                className="ml-2"
              >
                {loading ? <Spinner size="sm" /> : 'Refresh'}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardBody>
          {error && (
            <Alert type="error" className="mb-4">
              {error}
            </Alert>
          )}
          
          {loading ? (
            <div className="flex items-center justify-center h-80">
              <Spinner size="lg" color="text-blue-500" />
            </div>
          ) : sentimentData ? (
            <div className="sentiment-content">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {/* Overall sentiment */}
                <div className="bg-white p-6 rounded-lg shadow-sm">
                  <h3 className="text-lg font-medium mb-4">Overall Sentiment</h3>
                  <div className="flex items-center justify-center">
                    <SentimentScoreGauge score={sentimentData.average_sentiment} size={160} />
                  </div>
                  <div className="text-center mt-4">
                    <p className="text-sm text-gray-600">
                      Based on {sentimentData.total_count} articles from {format(new Date(sentimentData.time_from), 'MMM d, yyyy')} to {format(new Date(sentimentData.time_to), 'MMM d, yyyy')}
                    </p>
                  </div>
                </div>
                
                {/* Sentiment distribution */}
                <div className="bg-white p-6 rounded-lg shadow-sm">
                  <h3 className="text-lg font-medium mb-4">Sentiment Distribution</h3>
                  <div className="flex items-center justify-center h-40">
                    <SentimentDistribution
                      positive={sentimentData.positive_count}
                      neutral={sentimentData.neutral_count}
                      negative={sentimentData.negative_count}
                      width={400}
                      height={80}
                    />
                  </div>
                </div>
              </div>
              
              {/* Top tickers by sentiment */}
              <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
                <h3 className="text-lg font-medium mb-4">Top Tickers by Sentiment</h3>
                <div className="flex flex-wrap justify-center">
                  {sentimentData.top_tickers.map((ticker, index) => (
                    <TickerSentimentCard
                      key={index}
                      ticker={ticker.ticker}
                      sentimentScore={ticker.sentiment_score}
                      relevanceScore={ticker.relevance_score}
                    />
                  ))}
                </div>
              </div>
              
              {/* Generated signals */}
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <h3 className="text-lg font-medium mb-4">Generated Trading Signals</h3>
                {signals.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No signals generated from current sentiment data.
                  </div>
                ) : (
                  <Table>
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Strength</th>
                        <th>Confidence</th>
                        <th>Source</th>
                        <th>Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {signals.map((signal, index) => (
                        <tr key={index} className="hover:bg-gray-50">
                          <td className="font-medium">{signal.symbol}</td>
                          <td>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              signal.type.includes('BUY') ? 'bg-green-100 text-green-800' : 
                              signal.type.includes('SELL') ? 'bg-red-100 text-red-800' : 
                              'bg-yellow-100 text-yellow-800'
                            }`}>
                              {signal.type}
                            </span>
                          </td>
                          <td>
                            <div className="w-full bg-gray-200 rounded-full h-2.5">
                              <div 
                                className={`h-2.5 rounded-full ${
                                  signal.type.includes('BUY') ? 'bg-green-500' : 
                                  signal.type.includes('SELL') ? 'bg-red-500' : 'bg-yellow-500'
                                }`}
                                style={{ width: `${signal.strength * 100}%` }}
                              ></div>
                            </div>
                          </td>
                          <td>{signal.confidence ? `${(signal.confidence * 100).toFixed(0)}%` : 'N/A'}</td>
                          <td>{signal.source}</td>
                          <td>{format(new Date(signal.timestamp), 'MMM d, yyyy HH:mm')}</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              No sentiment data available. Please select a topic and refresh.
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
};

export default SentimentDashboard;

// Export additional components for reuse
export { SentimentScoreGauge, SentimentDistribution };
