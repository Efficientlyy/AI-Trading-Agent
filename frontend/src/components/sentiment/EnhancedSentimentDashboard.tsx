import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardBody } from '../Card';
import { Tabs, Tab } from '../Tabs';
import Toggle from '../Form/Toggle';
import Alert from '../Alert';
import { Table } from '../Table';
import SentimentTrendChart, { SentimentDataPoint } from './SentimentTrendChart';
import IntegratedSignalChart from './IntegratedSignalChart';
import SentimentHeatmap from './SentimentHeatmap';
import AssetComparisonChart, { AssetSentimentData } from './AssetComparisonChart';
import FilterControls from './FilterControls';
import SentimentInsightWidget from './SentimentInsightWidget';
import { alphaVantageApi } from '../../api/alphaVantage';
import { sentimentApi } from '../../api/sentimentApi';
import { SignalData, SignalType, SignalSource } from '../../types/signals';
import { format } from 'date-fns';

// Type for the AlphaVantage API response
interface AlphaVantageResponse {
  average_sentiment?: number;
  positive_count?: number;
  negative_count?: number;
  neutral_count?: number;
  total_count?: number;
  top_tickers?: Array<{
    ticker: string;
    sentiment_score: number;
    relevance_score: number;
  }>;
}

// Type definitions
interface SentimentData {
  average_sentiment: number;
  top_tickers: Array<{
    symbol: string;
    sentiment: number;
    volume?: number;
    technical_signal?: string;
    sentiment_signal?: string;
  }>;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  total_count: number;
  // For compatibility with Alpha Vantage API response
  ticker?: string;
  sentiment_score?: number;
  relevance_score?: number;
}

type TrendData = SentimentDataPoint;

type ComparisonData = AssetSentimentData;

// List of available assets for selection
const AVAILABLE_ASSETS = [
  { value: 'BTC', label: 'Bitcoin' },
  { value: 'ETH', label: 'Ethereum' },
  { value: 'SOL', label: 'Solana' },
  { value: 'XRP', label: 'Ripple' },
  { value: 'ADA', label: 'Cardano' },
  { value: 'AVAX', label: 'Avalanche' },
  { value: 'DOGE', label: 'Dogecoin' },
  { value: 'DOT', label: 'Polkadot' },
  { value: 'MATIC', label: 'Polygon' },
  { value: 'LINK', label: 'Chainlink' }
];

const EnhancedSentimentDashboard: React.FC = () => {
  // UI state
  const [activeTab, setActiveTab] = useState('overview');
  const [activeChartTab, setActiveChartTab] = useState('trends');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Data state
  const [sentimentData, setSentimentData] = useState<SentimentData | null>(null);
  const [trendData, setTrendData] = useState<TrendData[]>([]);
  const [integratedSignals, setIntegratedSignals] = useState<SignalData[]>([]);
  const [comparisonData, setComparisonData] = useState<ComparisonData[]>([]);
  
  // Filter state
  const [showRealTimeData, setShowRealTimeData] = useState(false);
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(['BTC', 'ETH', 'SOL']);
  const [timeRange, setTimeRange] = useState('7d');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.3);
  const [normalizeData, setNormalizeData] = useState(false);
  
  // Handle filter changes
  const handleTimeRangeChange = (range: string) => {
    setTimeRange(range);
  };
  
  const handleSymbolsChange = (symbols: string[]) => {
    setSelectedSymbols(symbols.length > 0 ? symbols : ['BTC']); // Always keep at least one symbol selected
  };
  
  // Handle visualization type change
  const handleVisualizationTypeChange = (type: string) => {
    setNormalizeData(type === 'normalized');
  };
  
  // Fetch data function (extracted for reuse)
  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // For demonstration, we're using the Alpha Vantage API
      // In a real implementation, you would use the actual sentiment API
      const alphaVantageData: AlphaVantageResponse = await alphaVantageApi.getSentimentByTopic('blockchain');
      
      // Transform the data to match our SentimentData interface
      const transformedData: SentimentData = {
        average_sentiment: alphaVantageData.average_sentiment || 0,
        positive_count: alphaVantageData.positive_count || 0,
        negative_count: alphaVantageData.negative_count || 0,
        neutral_count: alphaVantageData.neutral_count || 0,
        total_count: alphaVantageData.total_count || 0,
        // Transform top_tickers array if available
        top_tickers: (alphaVantageData.top_tickers || []).map(ticker => ({
          symbol: ticker.ticker, // Map ticker field to symbol
          sentiment: ticker.sentiment_score, // Map sentiment_score to sentiment
          volume: 0, // Default value
          technical_signal: undefined,
          sentiment_signal: undefined
        }))
      };
      
      setSentimentData(transformedData);
      
      // Fetch sentiment trend data
      const trendResponse = await sentimentApi.getSentimentTrend({ 
        symbols: selectedSymbols, 
        timeRange 
      });
      setTrendData(trendResponse.data);
      
      // Fetch integrated signals
      const signalsResponse = await sentimentApi.getIntegratedSignals({ 
        symbols: selectedSymbols, 
        timeRange,
        includeRawSignals: true
      });
      setIntegratedSignals(signalsResponse.data);
      
      // Fetch comparison data for all selected symbols
      if (selectedSymbols.length > 0) {
        const comparisonPromises = selectedSymbols.map(async symbol => {
          const response = await sentimentApi.getSentimentTrend({ 
            symbols: [symbol], 
            timeRange 
          });
          return {
            symbol,
            name: AVAILABLE_ASSETS.find(asset => asset.value === symbol)?.label || symbol,
            data: response.data
          };
        });
        
        const comparisonResults = await Promise.all(comparisonPromises);
        setComparisonData(comparisonResults);
      }
    } catch (error) {
      console.error('Error fetching sentiment data:', error);
      setError('Failed to fetch sentiment data. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  }, [selectedSymbols, timeRange]);
  
  // Initial data load and refresh setup
  useEffect(() => {
    fetchData();
    
    // If real-time data is enabled, set up interval to refresh data
    let intervalId: NodeJS.Timeout | null = null;
    if (showRealTimeData) {
      intervalId = setInterval(() => {
        fetchData();
      }, 60000); // Refresh every minute
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [fetchData, showRealTimeData]);
  
  // Generate sentiment summary cards
  const renderSummaryCards = () => {
    if (!sentimentData) return null;
    
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="border-gray-700">
            <h3 className="text-lg font-semibold text-white">Overall Sentiment</h3>
          </CardHeader>
          <CardBody className="text-gray-300">
            <div className="flex items-center justify-center">
              <div className="text-4xl font-bold">
                {sentimentData.average_sentiment > 0.3 ? '🔥' : 
                 sentimentData.average_sentiment > 0 ? '📈' : 
                 sentimentData.average_sentiment > -0.3 ? '📉' : '❄️'}
              </div>
              <div className="ml-4">
                <div className="text-3xl font-bold">
                  {sentimentData.average_sentiment !== undefined ? sentimentData.average_sentiment.toFixed(2) : 'N/A'}
                </div>
                <div className={`text-sm font-medium ${
                  sentimentData.average_sentiment > 0.3 ? 'text-green-600' : 
                  sentimentData.average_sentiment > 0 ? 'text-green-500' : 
                  sentimentData.average_sentiment > -0.3 ? 'text-red-500' : 'text-red-600'
                }`}>
                  {sentimentData.average_sentiment > 0.3 ? 'Very Bullish' : 
                   sentimentData.average_sentiment > 0 ? 'Bullish' : 
                   sentimentData.average_sentiment > -0.3 ? 'Bearish' : 'Very Bearish'}
                </div>
              </div>
            </div>
          </CardBody>
        </Card>
        
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="border-gray-700">
            <h3 className="text-lg font-semibold text-white">Sentiment Distribution</h3>
          </CardHeader>
          <CardBody className="text-gray-300">
            <div className="flex items-center space-x-2">
              <div className="w-full bg-gray-700 rounded-full h-2.5">
                <div 
                  className="bg-green-600 h-2.5 rounded-full" 
                  style={{ width: `${(sentimentData.positive_count / sentimentData.total_count) * 100}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium text-green-600">
                {Math.round((sentimentData.positive_count / sentimentData.total_count) * 100)}%
              </span>
            </div>
            <div className="flex items-center mt-2 space-x-2">
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className="bg-gray-500 h-2.5 rounded-full" 
                  style={{ width: `${(sentimentData.neutral_count / sentimentData.total_count) * 100}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium text-gray-500">
                {Math.round((sentimentData.neutral_count / sentimentData.total_count) * 100)}%
              </span>
            </div>
            <div className="flex items-center mt-2 space-x-2">
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className="bg-red-600 h-2.5 rounded-full" 
                  style={{ width: `${(sentimentData.negative_count / sentimentData.total_count) * 100}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium text-red-600">
                {Math.round((sentimentData.negative_count / sentimentData.total_count) * 100)}%
              </span>
            </div>
          </CardBody>
        </Card>
        
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="border-gray-700">
            <h3 className="text-lg font-semibold text-white">Top Assets by Sentiment</h3>
          </CardHeader>
          <CardBody className="text-gray-300">
            <div className="space-y-2">
              {sentimentData.top_tickers.slice(0, 5).map((ticker: any, index: number) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="font-medium">{ticker.ticker}</div>
                  <div className={`px-2 py-1 rounded text-xs font-medium ${
                    ticker.sentiment_score > 0.3 ? 'bg-green-100 text-green-800' : 
                    ticker.sentiment_score > 0 ? 'bg-green-50 text-green-600' : 
                    ticker.sentiment_score > -0.3 ? 'bg-red-50 text-red-600' : 'bg-red-100 text-red-800'
                  }`}>
                    {ticker.sentiment_score !== undefined ? ticker.sentiment_score.toFixed(2) : 'N/A'}
                  </div>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      </div>
    );
  };
  
  // Generate trading signals table
  const renderSignalsTable = () => {
    if (!integratedSignals || integratedSignals.length === 0) {
      return (
        <div className="text-center py-6">
          <p className="text-gray-400">No signals available for the selected time range.</p>
        </div>
      );
    }
    
    return (
      <Table className="bg-gray-800 text-gray-300">
        <thead>
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Date & Time</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Symbol</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Signal</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Source</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Strength</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {integratedSignals.map((signal, index) => (
            <tr key={index} className={index % 2 === 0 ? 'bg-gray-700' : 'bg-gray-800'}>
              <td className="px-6 py-4 whitespace-nowrap">{format(new Date(signal.timestamp), 'yyyy-MM-dd HH:mm')}</td>
              <td className="px-6 py-4 whitespace-nowrap">{signal.symbol}</td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  signal.type.includes('BUY') ? 'bg-green-100 text-green-800' : 
                  signal.type.includes('SELL') ? 'bg-red-100 text-red-800' : 
                  'bg-gray-100 text-gray-800'
                }`}>
                  {signal.type}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  signal.source === 'TECHNICAL' ? 'bg-blue-100 text-blue-800' : 
                  signal.source === 'SENTIMENT' ? 'bg-purple-100 text-purple-800' : 
                  signal.source === 'COMBINED' ? 'bg-indigo-100 text-indigo-800' : 
                  'bg-gray-100 text-gray-800'
                }`}>
                  {signal.source}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="w-full bg-gray-700 rounded-full h-2.5">
                  <div 
                    className={`h-2.5 rounded-full ${
                      signal.strength > 0.8 ? 'bg-green-600' : 
                      signal.strength > 0.6 ? 'bg-green-500' : 
                      signal.strength > 0.4 ? 'bg-yellow-500' : 
                      'bg-gray-400'
                    }`}
                    style={{ width: `${signal.strength * 100}%` }}
                  ></div>
                </div>
                <span className="text-xs">{signal.strength !== undefined ? (signal.strength * 100).toFixed(0) : 0}%</span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                {signal.confidence !== undefined ? `${(signal.confidence * 100).toFixed(0)}%` : 'N/A'}
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    );
  };
  
  return (
    <div className="enhanced-sentiment-dashboard">
      {/* Dashboard Header */}
      <div className="mb-6 flex flex-col md:flex-row justify-between items-start md:items-center">
        <div>
          <h1 className="text-2xl font-bold">Enhanced Sentiment Dashboard</h1>
          <p className="text-gray-500">
            Integrated analysis of sentiment and technical signals across multiple assets
          </p>
        </div>
        <div className="mt-4 md:mt-0 flex items-center">
          <Toggle 
            label="Real-time updates"
            checked={showRealTimeData}
            onChange={() => setShowRealTimeData(!showRealTimeData)}
            className="mr-4"
          />
          <select 
            className="bg-white border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <option value="1d">Last 24 hours</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
        </div>
      </div>
      
      {/* Error Alert */}
      {error && (
        <Alert type="error">
          <div className="flex flex-col">
            <span className="font-bold">Error</span>
            <span>{error}</span>
          </div>
        </Alert>
      )}
      
      <div className="mb-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold">Enhanced Sentiment Dashboard</h1>
        <button 
          onClick={fetchData} 
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          disabled={isLoading}
        >
          {isLoading ? 'Refreshing...' : 'Refresh Data'}
        </button>
      </div>
      
      {/* New filter controls component */}
      <FilterControls
        selectedSymbols={selectedSymbols}
        availableSymbols={AVAILABLE_ASSETS}
        timeRange={timeRange}
        showRealTimeData={showRealTimeData}
        confidenceThreshold={confidenceThreshold}
        onSymbolsChange={handleSymbolsChange}
        onTimeRangeChange={handleTimeRangeChange}
        onRealTimeToggle={setShowRealTimeData}
        onConfidenceThresholdChange={setConfidenceThreshold}
      />
      
      {sentimentData && (
        <>
          {/* Summary cards at the top */}
          {renderSummaryCards()}
          
          {/* Insights widget - new component */}
          <SentimentInsightWidget
            sentimentData={sentimentData}
            selectedSymbols={selectedSymbols}
            confidenceThreshold={confidenceThreshold}
          />
          
          {/* Main content tabs */}
          <Tabs activeTab={activeTab} onChange={setActiveTab}>
            <Tab id="overview" label="Overview">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <Card>
                  <CardHeader>
                    <h3 className="text-lg font-semibold">Sentiment Trend</h3>
                  </CardHeader>
                  <CardBody>
                    <SentimentTrendChart 
                      data={trendData} 
                      height={300} 
                      symbol={selectedSymbols[0]}
                      showConfidence={confidenceThreshold > 0}
                    />
                  </CardBody>
                </Card>
                
                <Card>
                  <CardHeader>
                    <h3 className="text-lg font-semibold">Sentiment vs. Technical Signals</h3>
                  </CardHeader>
                  <CardBody>
                    <IntegratedSignalChart 
                      data={integratedSignals.map(signal => ({
                        timestamp: signal.timestamp,
                        symbol: signal.symbol || selectedSymbols[0],
                        technicalSignal: signal.source === 'TECHNICAL' ? signal.strength : 0,
                        sentimentSignal: signal.source === 'SENTIMENT' ? signal.strength : 0,
                        combinedSignal: signal.source === 'COMBINED' ? signal.strength : 0,
                        technicalConfidence: signal.source === 'TECHNICAL' ? (signal.confidence || 0.5) : 0.5,
                        sentimentConfidence: signal.source === 'SENTIMENT' ? (signal.confidence || 0.5) : 0.5,
                        combinedConfidence: signal.source === 'COMBINED' ? (signal.confidence || 0.5) : 0.5,
                        signalType: signal.type.toLowerCase()
                      }))} 
                      height={300}
                    />
                  </CardBody>
                </Card>
              </div>
              
              {/* New comparison chart section */}
              <Card className="mb-4">
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <h3 className="text-lg font-semibold">Asset Comparison</h3>
                    <div className="flex space-x-2">
                      <button
                        className={`px-3 py-1 text-sm rounded-md ${!normalizeData ? 'bg-blue-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
                        onClick={() => handleVisualizationTypeChange('standard')}
                      >
                        Absolute
                      </button>
                      <button
                        className={`px-3 py-1 text-sm rounded-md ${normalizeData ? 'bg-blue-500 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
                        onClick={() => handleVisualizationTypeChange('normalized')}
                      >
                        Normalized
                      </button>
                    </div>
                  </div>
                </CardHeader>
                <CardBody>
                  <AssetComparisonChart
                    assets={comparisonData}
                    height={350}
                    showConfidence={false}
                    normalizeData={normalizeData}
                    timeRange={timeRange}
                  />
                </CardBody>
              </Card>
            </Tab>
            
            <Tab id="heatmap" label="Sentiment Heatmap">
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">Asset Sentiment Heatmap</h3>
                </CardHeader>
                <CardBody>
                  <SentimentHeatmap 
                    data={sentimentData?.top_tickers.map(ticker => ({
                      ...ticker,
                      timestamp: new Date().toISOString() // Add required timestamp
                    })) || []} 
                    height={500}
                  />
                </CardBody>
              </Card>
            </Tab>
            
            <Tab id="signals" label="Trading Signals">
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">Generated Trading Signals</h3>
                </CardHeader>
                <CardBody>
                  {renderSignalsTable()}
                </CardBody>
              </Card>
            </Tab>
            
            <Tab id="charts" label="Advanced Charts">
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">Advanced Sentiment Analysis</h3>
                </CardHeader>
                <CardBody>
                  <Tabs activeTab={activeChartTab} onChange={setActiveChartTab}>
                    <Tab id="trends" label="Trend Analysis">
                      <div className="p-4">
                        <h4 className="text-lg font-medium mb-4">Sentiment Trend Analysis</h4>
                        <SentimentTrendChart 
                          data={trendData} 
                          height={400} 
                          symbol={selectedSymbols[0]}
                          showConfidence={true}
                          title="Long-term Sentiment Trend"
                        />
                      </div>
                    </Tab>
                    <Tab id="correlation" label="Correlation Analysis">
                      <div className="p-4">
                        <h4 className="text-lg font-medium mb-4">Sentiment-Price Correlation</h4>
                        <p className="text-gray-600 mb-4">
                          This chart visualizes the correlation between sentiment score and price movements over time.
                          A high correlation suggests that sentiment is a leading indicator of price changes.
                        </p>
                        {/* Placeholder for correlation chart */}
                        <div className="h-64 bg-gray-800 rounded flex items-center justify-center">
                          <p className="text-gray-400">Correlation visualization coming soon</p>
                        </div>
                      </div>
                    </Tab>
                    <Tab id="volatility" label="Sentiment Volatility">
                      <div className="p-4">
                        <h4 className="text-lg font-medium mb-4">Sentiment Volatility Index</h4>
                        <p className="text-gray-600 mb-4">
                          This chart shows the volatility of sentiment over time, which can be an indicator of market uncertainty.
                          High sentiment volatility often precedes major market moves.
                        </p>
                        {/* Placeholder for volatility chart */}
                        <div className="h-64 bg-gray-800 rounded flex items-center justify-center">
                          <p className="text-gray-400">Volatility analysis coming soon</p>
                        </div>
                      </div>
                    </Tab>
                  </Tabs>
                </CardBody>
              </Card>
            </Tab>
            
            <Tab id="sources" label="Data Sources">
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">Sentiment Data Sources</h3>
                </CardHeader>
                <CardBody>
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-lg font-medium mb-2">News & Media</h4>
                      <p className="text-gray-300">
                        Financial news articles, press releases, and media coverage from over 100 sources
                        are analyzed using NLP to extract sentiment about cryptocurrencies and blockchain projects.
                      </p>
                      <div className="mt-2">
                        <span className="text-sm text-gray-400">
                          Last updated: {format(new Date(), 'yyyy-MM-dd HH:mm')}
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="text-lg font-medium mb-2">Social Media</h4>
                      <p className="text-gray-300">
                        Twitter, Reddit, and specialized crypto forums are monitored for mentions and 
                        sentiment analysis is performed on posts with high engagement.
                      </p>
                      <div className="mt-2">
                        <span className="text-sm text-gray-400">
                          Last updated: {format(new Date(), 'yyyy-MM-dd HH:mm')}
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="text-lg font-medium mb-2">Market Data</h4>
                      <p className="text-gray-300">
                        Technical indicators from price and volume data are integrated with sentiment analysis
                        to provide a comprehensive view of market conditions.
                      </p>
                      <div className="mt-2">
                        <span className="text-sm text-gray-400">
                          Last updated: {format(new Date(), 'yyyy-MM-dd HH:mm')}
                        </span>
                      </div>
                    </div>
                  </div>
                </CardBody>
              </Card>
            </Tab>
          </Tabs>
        </>
      )}
    </div>
  );
};

export default EnhancedSentimentDashboard;
