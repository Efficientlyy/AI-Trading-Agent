import React, { useState, useEffect, useCallback } from 'react';
import { format } from 'date-fns';
import { Card, CardHeader, CardBody } from '../Card';
import { Select, Spinner, Button } from '../Form';
import Toggle from '../Form/Toggle';
import { Alert } from '../Alert';
import { Tab, Tabs } from '../Tabs/index';

// Import our custom sentiment visualization components
import SentimentTrendChart, { SentimentDataPoint } from '../sentiment/SentimentTrendChart';
import IntegratedSignalChart, { IntegratedSignalData } from '../sentiment/IntegratedSignalChart';
import SentimentHeatmap from '../sentiment/SentimentHeatmap';
import { SentimentScoreGauge, SentimentDistribution } from '../dashboard/SentimentDashboard';

// Import sentimentApi for data fetching
import { sentimentApiModule } from '../../api';
const { fetchSentimentData, fetchIntegratedSignals } = sentimentApiModule.default;

interface TimeRangeOption {
  label: string;
  value: string;
  days: number;
}

const TIME_RANGES: TimeRangeOption[] = [
  { label: '24 Hours', value: '1d', days: 1 },
  { label: '7 Days', value: '1w', days: 7 },
  { label: '30 Days', value: '1m', days: 30 },
  { label: '90 Days', value: '3m', days: 90 }
];

const AVAILABLE_ASSETS = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'SOL', 'DOGE'];

const AVAILABLE_TOPICS = [
  'blockchain',
  'cryptocurrency',
  'bitcoin',
  'ethereum',
  'defi',
  'nft',
  'metaverse',
  'regulation',
  'market'
];

interface SentimentSummary {
  average_sentiment: number;
  positive_count: number;
  neutral_count: number;
  negative_count: number;
  last_updated: string;
}

const EnhancedSentimentDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [selectedAsset, setSelectedAsset] = useState<string>('BTC');
  const [selectedTopic, setSelectedTopic] = useState<string>('cryptocurrency');
  const [timeRange, setTimeRange] = useState<string>('1w');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showConfidence, setShowConfidence] = useState<boolean>(true);
  
  // Data states
  const [sentimentData, setSentimentData] = useState<SentimentDataPoint[]>([]);
  const [integratedSignals, setIntegratedSignals] = useState<IntegratedSignalData[]>([]);
  const [sentimentSummary, setSentimentSummary] = useState<SentimentSummary | null>(null);
  const [topicSentimentData, setTopicSentimentData] = useState<Record<string, SentimentDataPoint[]>>({});
  
  // Load data function
  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Get selected time range in days
      const days = TIME_RANGES.find(range => range.value === timeRange)?.days || 7;
      
      // Fetch sentiment data for the selected asset
      const assetSentimentData = await fetchSentimentData(selectedAsset, days);
      setSentimentData(assetSentimentData);
      
      // Calculate sentiment summary
      const posCount = assetSentimentData.filter((d: SentimentDataPoint) => d.sentiment > 0.3).length;
      const negCount = assetSentimentData.filter((d: SentimentDataPoint) => d.sentiment < -0.3).length;
      const neutCount = assetSentimentData.length - posCount - negCount;
      const avgSent = assetSentimentData.reduce((sum: number, d: SentimentDataPoint) => sum + d.sentiment, 0) / 
                     (assetSentimentData.length || 1);
      
      setSentimentSummary({
        average_sentiment: avgSent,
        positive_count: posCount,
        neutral_count: neutCount,
        negative_count: negCount,
        last_updated: new Date().toISOString()
      });
      
      // Fetch integrated signals for all assets
      const signals = await fetchIntegratedSignals(days);
      setIntegratedSignals(signals);
      
      // Fetch sentiment data for topics
      const topicData: Record<string, SentimentDataPoint[]> = {};
      for (const topic of AVAILABLE_TOPICS.slice(0, 5)) { // Limit to first 5 topics for performance
        topicData[topic] = await fetchSentimentData(topic, days, true);
      }
      setTopicSentimentData(topicData);
      
    } catch (err) {
      console.error('Error loading sentiment data:', err);
      setError('Failed to load sentiment data. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  }, [selectedAsset, timeRange]);
  
  // Load data on initial mount and when dependencies change
  useEffect(() => {
    loadData();
  }, [loadData]);
  
  // Handle asset change
  const handleAssetChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedAsset(event.target.value);
  };
  
  // Handle topic change
  const handleTopicChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedTopic(event.target.value);
  };
  
  // Handle time range change
  const handleTimeRangeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setTimeRange(event.target.value);
  };
  
  // Handle refresh button click
  const handleRefresh = () => {
    loadData();
  };
  
  // Prepare heatmap data
  const heatmapData = AVAILABLE_ASSETS.flatMap(asset => {
    // Find the most recent integrated signal for this asset
    const signal = integratedSignals.find(s => s.symbol === asset);
    if (!signal) return [];
    
    // Create a heatmap data point with null checks
    return [{
      symbol: asset,
      sentiment: signal.sentimentSignal !== undefined ? signal.sentimentSignal : 0,
      confidence: signal.sentimentConfidence !== undefined ? signal.sentimentConfidence : 0.5,
      timestamp: new Date().toISOString(),
      volume: Math.random() * 10000 // Mock volume data
    }];
  });
  
  return (
    <div className="enhanced-sentiment-dashboard p-4">
      <Card>
        <CardHeader>
          <div className="flex flex-col md:flex-row justify-between items-center space-y-2 md:space-y-0">
            <h2 className="text-xl font-bold">Sentiment Analysis Dashboard</h2>
            <div className="flex flex-wrap items-center space-x-2">
              <Select
                value={selectedAsset}
                onChange={handleAssetChange}
                className="w-32"
              >
                {AVAILABLE_ASSETS.map(asset => (
                  <option key={asset} value={asset}>{asset}</option>
                ))}
              </Select>
              
              <Select
                value={timeRange}
                onChange={handleTimeRangeChange}
                className="w-32"
              >
                {TIME_RANGES.map(range => (
                  <option key={range.value} value={range.value}>{range.label}</option>
                ))}
              </Select>
              
              <Toggle
                label="Show Confidence"
                checked={showConfidence}
                onChange={() => setShowConfidence(!showConfidence)}
              />
              
              <Button
                onClick={handleRefresh}
                className="ml-2"
                disabled={isLoading}
              >
                {isLoading ? <Spinner size="sm" className="mr-2" /> : null}
                Refresh
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
          
          <Tabs activeTab={activeTab} onChange={setActiveTab}>
            <Tab id="overview" label="Overview">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {/* Sentiment Summary Card */}
                <div className="bg-white p-6 rounded-lg shadow-sm">
                  <h3 className="text-lg font-medium mb-4">Current Sentiment</h3>
                  <div className="flex items-center justify-center mb-4">
                    {sentimentSummary && (
                      <SentimentScoreGauge score={sentimentSummary.average_sentiment} size={150} />
                    )}
                  </div>
                  {sentimentSummary && (
                    <>
                      <div className="text-center mb-4">
                        <p className="text-sm text-gray-500">
                          Last updated: {format(new Date(sentimentSummary.last_updated), 'MMM d, yyyy HH:mm')}
                        </p>
                      </div>
                      <div className="flex items-center justify-center">
                        <SentimentDistribution
                          positive={sentimentSummary.positive_count}
                          neutral={sentimentSummary.neutral_count}
                          negative={sentimentSummary.negative_count}
                          width={300}
                          height={80}
                        />
                      </div>
                    </>
                  )}
                </div>
                
                {/* Sentiment Trend Chart */}
                <div className="bg-gray-800 text-gray-300 p-6 rounded-lg shadow-sm border border-gray-700">
                  <h3 className="text-lg font-medium mb-4 text-white">{selectedAsset} Sentiment Trend</h3>
                  {sentimentData.length > 0 ? (
                    <SentimentTrendChart
                      data={sentimentData}
                      symbol={selectedAsset}
                      showConfidence={showConfidence}
                      height={250}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-64 text-gray-400">
                      No sentiment data available
                    </div>
                  )}
                </div>
              </div>
              
              {/* Integrated Signals Chart */}
              <div className="bg-gray-800 text-gray-300 p-6 rounded-lg shadow-sm mb-6 border border-gray-700">
                <h3 className="text-lg font-medium mb-4 text-white">Integrated Trading Signals</h3>
                {integratedSignals.length > 0 ? (
                  <IntegratedSignalChart
                    data={integratedSignals}
                    showConfidence={showConfidence}
                    height={300}
                  />
                ) : (
                  <div className="flex items-center justify-center h-64 text-gray-500">
                    No integrated signals available
                  </div>
                )}
              </div>
              
              {/* Sentiment Heatmap */}
              <div className="bg-gray-800 text-gray-300 p-6 rounded-lg shadow-sm border border-gray-700">
                <SentimentHeatmap
                  data={heatmapData}
                  height={40 * AVAILABLE_ASSETS.length}
                  title="Asset Sentiment Heatmap"
                />
              </div>
            </Tab>
            
            <Tab id="topics" label="Topic Analysis">
              <div className="mb-6">
                <div className="flex items-center space-x-2 mb-4">
                  <Select
                    value={selectedTopic}
                    onChange={handleTopicChange}
                    className="w-48"
                  >
                    {AVAILABLE_TOPICS.map(topic => (
                      <option key={topic} value={topic}>{topic}</option>
                    ))}
                  </Select>
                </div>
                
                {/* Topic Sentiment Chart */}
                <div className="bg-gray-800 text-gray-300 p-6 rounded-lg shadow-sm mb-6 border border-gray-700">
                  <h3 className="text-lg font-medium mb-4 text-white">{selectedTopic} Sentiment Trend</h3>
                  {topicSentimentData[selectedTopic]?.length > 0 ? (
                    <SentimentTrendChart
                      data={topicSentimentData[selectedTopic] || []}
                      symbol={selectedTopic}
                      showConfidence={showConfidence}
                      height={300}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-64 text-gray-400">
                      No sentiment data available for this topic
                    </div>
                  )}
                </div>
                
                {/* Topic Comparison */}
                <div className="bg-gray-800 text-gray-300 p-6 rounded-lg shadow-sm border border-gray-700">
                  <h3 className="text-lg font-medium mb-4 text-white">Topic Sentiment Comparison</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(topicSentimentData).map(([topic, data]) => {
                      if (data.length === 0) return null;
                      
                      // Calculate average sentiment
                      const avgSentiment = data.reduce((sum, d) => sum + d.sentiment, 0) / data.length;
                      
                      return (
                        <div 
                          key={topic}
                          className="bg-gray-50 p-4 rounded-lg cursor-pointer hover:bg-gray-100"
                          onClick={() => setSelectedTopic(topic)}
                        >
                          <h4 className="font-medium mb-2">{topic}</h4>
                          <div className="flex items-center">
                            <SentimentScoreGauge score={avgSentiment} size={80} />
                            <div className="ml-4">
                              <p className="text-sm">
                                Average: <span className="font-medium">{avgSentiment !== undefined ? avgSentiment.toFixed(2) : 'N/A'}</span>
                              </p>
                              <p className="text-sm">
                                Samples: <span className="font-medium">{data.length}</span>
                              </p>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </Tab>
            
            <Tab id="comparison" label="Technical vs. Sentiment">
              <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
                <h3 className="text-lg font-medium mb-4">Technical vs. Sentiment Analysis</h3>
                {integratedSignals.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Signal Comparison Chart */}
                    <IntegratedSignalChart
                      data={integratedSignals.filter(s => s.symbol === selectedAsset)}
                      showConfidence={showConfidence}
                      height={200}
                      title={`${selectedAsset} Signal Comparison`}
                    />
                    
                    {/* Summary Card */}
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-4">Signal Analysis</h4>
                      {integratedSignals.find(s => s.symbol === selectedAsset) ? (
                        <div>
                          {(() => {
                            const signal = integratedSignals.find(s => s.symbol === selectedAsset)!;
                            const techStronger = Math.abs(signal.technicalSignal) > Math.abs(signal.sentimentSignal);
                            const sentStronger = Math.abs(signal.sentimentSignal) > Math.abs(signal.technicalSignal);
                            const agreement = Math.sign(signal.technicalSignal) === Math.sign(signal.sentimentSignal);
                            
                            return (
                              <>
                                <p className="mb-2">
                                  <span className="font-medium">Technical Signal:</span> {signal.technicalSignal !== undefined ? signal.technicalSignal.toFixed(2) : 'N/A'}
                                  {signal.technicalConfidence && ` (${(signal.technicalConfidence * 100).toFixed(0)}% confidence)`}
                                </p>
                                <p className="mb-2">
                                  <span className="font-medium">Sentiment Signal:</span> {signal.sentimentSignal !== undefined ? signal.sentimentSignal.toFixed(2) : 'N/A'}
                                  {signal.sentimentConfidence && ` (${(signal.sentimentConfidence * 100).toFixed(0)}% confidence)`}
                                </p>
                                <p className="mb-2">
                                  <span className="font-medium">Combined Signal:</span> {signal.combinedSignal !== undefined ? signal.combinedSignal.toFixed(2) : 'N/A'}
                                  {signal.combinedConfidence && ` (${(signal.combinedConfidence * 100).toFixed(0)}% confidence)`}
                                </p>
                                <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                                  <h5 className="font-medium mb-2">Analysis</h5>
                                  {agreement ? (
                                    <p className="text-green-600">
                                      Technical and sentiment signals are in agreement, suggesting a {signal.combinedSignal > 0 ? 'bullish' : 'bearish'} outlook.
                                    </p>
                                  ) : (
                                    <p className="text-yellow-600">
                                      Technical and sentiment signals are in disagreement, with {techStronger ? 'technical' : 'sentiment'} analysis showing a stronger signal.
                                    </p>
                                  )}
                                  <p className="mt-2">
                                    {techStronger && 'Technical analysis is currently the dominant factor.'}
                                    {sentStronger && 'Sentiment analysis is currently the dominant factor.'}
                                    {!techStronger && !sentStronger && 'Both factors are contributing equally to the signal.'}
                                  </p>
                                </div>
                              </>
                            );
                          })()}
                        </div>
                      ) : (
                        <div className="text-center py-8 text-gray-500">
                          No integrated signal available for {selectedAsset}
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-64 text-gray-500">
                    No integrated signals available
                  </div>
                )}
              </div>
              
              {/* All Assets Comparison */}
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <h3 className="text-lg font-medium mb-4">Cross-Asset Comparison</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="py-2 px-4 text-left">Asset</th>
                        <th className="py-2 px-4 text-left">Technical Signal</th>
                        <th className="py-2 px-4 text-left">Sentiment Signal</th>
                        <th className="py-2 px-4 text-left">Combined Signal</th>
                        <th className="py-2 px-4 text-left">Agreement</th>
                      </tr>
                    </thead>
                    <tbody>
                      {integratedSignals.map(signal => (
                        <tr 
                          key={signal.symbol}
                          className={`hover:bg-gray-50 cursor-pointer ${signal.symbol === selectedAsset ? 'bg-blue-50' : ''}`}
                          onClick={() => setSelectedAsset(signal.symbol)}
                        >
                          <td className="py-2 px-4 font-medium">{signal.symbol}</td>
                          <td className={`py-2 px-4 ${signal.technicalSignal > 0 ? 'text-green-600' : signal.technicalSignal < 0 ? 'text-red-600' : ''}`}>
                            {signal.technicalSignal !== undefined ? signal.technicalSignal.toFixed(2) : 'N/A'}
                          </td>
                          <td className={`py-2 px-4 ${signal.sentimentSignal > 0 ? 'text-green-600' : signal.sentimentSignal < 0 ? 'text-red-600' : ''}`}>
                            {signal.sentimentSignal !== undefined ? signal.sentimentSignal.toFixed(2) : 'N/A'}
                          </td>
                          <td className={`py-2 px-4 ${signal.combinedSignal > 0 ? 'text-green-600' : signal.combinedSignal < 0 ? 'text-red-600' : ''}`}>
                            {signal.combinedSignal !== undefined ? signal.combinedSignal.toFixed(2) : 'N/A'}
                          </td>
                          <td className="py-2 px-4">
                            {Math.sign(signal.technicalSignal) === Math.sign(signal.sentimentSignal) ? (
                              <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Agreement</span>
                            ) : (
                              <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs">Divergence</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </Tab>
          </Tabs>
        </CardBody>
      </Card>
    </div>
  );
};

export default EnhancedSentimentDashboard;
