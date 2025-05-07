import React, { useState, useEffect, useCallback } from 'react';
import { format } from 'date-fns';
import { Link } from 'react-router-dom';
import { Card, CardHeader, CardBody } from '../Card';
import { Select, Spinner, Button } from '../Form';
import { Alert } from '../Alert';
import { Table } from '../Table';
import { SignalData } from '../../types/signals';
import advancedSignalService, { MarketRegime } from '../../services/AdvancedSignalService';
import SignalComparisonChart from './SignalComparisonChart';
import SignalWeightVisualizer from './SignalWeightVisualizer';
import SignalWeightSettings from './SignalWeightSettings';
import SignalContributionBreakdown from './SignalContributionBreakdown';
import SignalPerformanceTracker from './SignalPerformanceTracker';

// Market regime badge component
const MarketRegimeBadge: React.FC<{ regime: MarketRegime }> = ({ regime }) => {
  const getBadgeColor = () => {
    switch (regime) {
      case MarketRegime.BULLISH:
        return 'bg-green-100 text-green-800';
      case MarketRegime.BEARISH:
        return 'bg-red-100 text-red-800';
      case MarketRegime.VOLATILE:
        return 'bg-purple-100 text-purple-800';
      case MarketRegime.SIDEWAYS:
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getBadgeColor()}`}>
      {regime}
    </span>
  );
};

// Signal strength indicator component
const SignalStrengthIndicator: React.FC<{ 
  strength: number; 
  type: string;
}> = ({ strength, type }) => {
  const getColor = () => {
    if (type.includes('BUY')) {
      return 'bg-green-500';
    } else if (type.includes('SELL')) {
      return 'bg-red-500';
    } else {
      return 'bg-yellow-500';
    }
  };

  return (
    <div className="w-full bg-gray-200 rounded-full h-2.5">
      <div 
        className={`h-2.5 rounded-full ${getColor()}`}
        style={{ width: `${strength * 100}%` }}
      ></div>
    </div>
  );
};

// Signal type badge component
const SignalTypeBadge: React.FC<{ type: string }> = ({ type }) => {
  const getBadgeColor = () => {
    if (type === 'STRONG_BUY') {
      return 'bg-green-600 text-white';
    } else if (type === 'BUY') {
      return 'bg-green-400 text-white';
    } else if (type === 'STRONG_SELL') {
      return 'bg-red-600 text-white';
    } else if (type === 'SELL') {
      return 'bg-red-400 text-white';
    } else {
      return 'bg-yellow-400 text-gray-800';
    }
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getBadgeColor()}`}>
      {type}
    </span>
  );
};

// Signal source badge component
const SignalSourceBadge: React.FC<{ source: string }> = ({ source }) => {
  const getBadgeColor = () => {
    switch (source) {
      case 'TECHNICAL':
        return 'bg-blue-100 text-blue-800';
      case 'SENTIMENT':
        return 'bg-purple-100 text-purple-800';
      case 'COMBINED':
        return 'bg-indigo-100 text-indigo-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getBadgeColor()}`}>
      {source}
    </span>
  );
};

// Available symbols for trading
const AVAILABLE_SYMBOLS = [
  'BTC/USDT', 
  'ETH/USDT', 
  'XRP/USDT', 
  'ADA/USDT', 
  'SOL/USDT',
  'DOT/USDT',
  'DOGE/USDT',
  'AVAX/USDT'
];

// Available timeframes for analysis
const AVAILABLE_TIMEFRAMES = [
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: '1 Day' },
  { value: '1w', label: '1 Week' }
];

// Available topics for sentiment analysis
const AVAILABLE_TOPICS = [
  'blockchain',
  'crypto',
  'bitcoin',
  'ethereum',
  'defi',
  'nft',
  'metaverse',
  'web3'
];

// Advanced Signal Dashboard component
const AdvancedSignalDashboard: React.FC = () => {
  // State for filters and data
  const [selectedSymbol, setSelectedSymbol] = useState<string>('BTC/USDT');
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('1d');
  const [selectedTopic, setSelectedTopic] = useState<string>('blockchain');
  const [marketRegime, setMarketRegime] = useState<MarketRegime>(MarketRegime.UNKNOWN);
  const [signals, setSignals] = useState<SignalData[]>([]);
  
  // State for custom weights
  const [technicalWeight, setTechnicalWeight] = useState<number>(0.5);
  const [sentimentWeight, setSentimentWeight] = useState<number>(0.5);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState<boolean>(false);
  
  // UI state
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Fetch signals data
  const fetchSignals = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const combinedSignals = await advancedSignalService.generateCombinedSignals(
        selectedSymbol,
        selectedTopic,
        selectedTimeframe
      );
      
      setSignals(combinedSignals);
      
      // Get the market regime from the first combined signal
      const combinedSignal = combinedSignals.find((signal: SignalData) => signal.source === 'COMBINED');
      if (combinedSignal && combinedSignal.description) {
        const regimeMatch = combinedSignal.description.match(/in (\w+) market regime/);
        if (regimeMatch && regimeMatch[1]) {
          setMarketRegime(regimeMatch[1] as MarketRegime);
        }
      }
    } catch (error) {
      console.error('Error fetching advanced signals:', error);
      setError('Failed to fetch advanced signals. Please try again later.');
    } finally {
      setLoading(false);
    }
  }, [selectedSymbol, selectedTopic, selectedTimeframe]);
  
  // Fetch data when filters change
  useEffect(() => {
    fetchSignals();
  }, [fetchSignals]);
  
  // Handle symbol change
  const handleSymbolChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedSymbol(event.target.value);
  };
  
  // Handle timeframe change
  const handleTimeframeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedTimeframe(event.target.value);
  };
  
  // Handle topic change
  const handleTopicChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedTopic(event.target.value);
  };
  
  // Handle refresh
  const handleRefresh = () => {
    fetchSignals();
  };
  
  // Handle weight changes
  const handleWeightsChange = (technical: number, sentiment: number) => {
    setTechnicalWeight(technical);
    setSentimentWeight(sentiment);
    // In a real implementation, we would use these weights to regenerate signals
    console.log(`Weights updated - Technical: ${technical}, Sentiment: ${sentiment}`);
  };
  
  // Toggle advanced settings
  const toggleAdvancedSettings = () => {
    setShowAdvancedSettings(!showAdvancedSettings);
  };
  
  // Filter signals by source
  const getSignalsBySource = (source: string) => {
    return signals.filter(signal => signal.source === source);
  };
  
  // Get combined signals
  const combinedSignals = getSignalsBySource('COMBINED');
  
  // Get technical signals
  const technicalSignals = getSignalsBySource('TECHNICAL');
  
  // Get sentiment signals
  const sentimentSignals = getSignalsBySource('SENTIMENT');
  
  return (
    <div className="advanced-signal-dashboard">
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <h2 className="text-xl font-semibold">Advanced Trading Signals</h2>
              {marketRegime !== MarketRegime.UNKNOWN && (
                <MarketRegimeBadge regime={marketRegime} />
              )}
              <Link to="/sentiment-analysis" className="ml-4 px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors">
                View Sentiment Analysis
              </Link>
            </div>
            <div className="flex flex-wrap space-x-2 space-y-2 sm:space-y-0 sm:space-x-4">
              <div className="w-40">
                <Select
                  value={selectedSymbol}
                  onChange={handleSymbolChange}
                  label="Symbol"
                  className="w-full"
                >
                  {AVAILABLE_SYMBOLS.map(symbol => (
                    <option key={symbol} value={symbol}>
                      {symbol}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="w-40">
                <Select
                  value={selectedTimeframe}
                  onChange={handleTimeframeChange}
                  label="Timeframe"
                  className="w-full"
                >
                  {AVAILABLE_TIMEFRAMES.map(timeframe => (
                    <option key={timeframe.value} value={timeframe.value}>
                      {timeframe.label}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="w-40">
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
                className="ml-2 mt-6"
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
          ) : (
            <div className="signal-content">
              {/* Signal Combination Visualization */}
              <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium">Signal Combination Visualization</h3>
                  <button 
                    onClick={toggleAdvancedSettings}
                    className="px-3 py-1 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors text-sm"
                  >
                    {showAdvancedSettings ? 'Hide Advanced Settings' : 'Show Advanced Settings'}
                  </button>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                  {/* Signal Weight Visualizer or Settings */}
                  <div className="lg:col-span-1">
                    {showAdvancedSettings ? (
                      <SignalWeightSettings 
                        marketRegime={marketRegime}
                        onWeightsChange={handleWeightsChange}
                      />
                    ) : (
                      <SignalWeightVisualizer marketRegime={marketRegime} />
                    )}
                  </div>
                  
                  {/* Signal Comparison Chart */}
                  <div className="lg:col-span-2">
                    {signals.length > 0 ? (
                      <SignalComparisonChart 
                        technicalSignals={technicalSignals}
                        sentimentSignals={sentimentSignals}
                        combinedSignals={combinedSignals}
                        width={600}
                        height={300}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-80 bg-gray-50 rounded-lg">
                        <p className="text-gray-500">No signals available for comparison</p>
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Signal Contribution Breakdown */}
                {combinedSignals.length > 0 && (
                  <div className="mt-6">
                    <SignalContributionBreakdown
                      combinedSignal={combinedSignals[0]}
                      technicalSignals={technicalSignals}
                      sentimentSignals={sentimentSignals}
                    />
                  </div>
                )}
              </div>
              
              {/* Signal Performance Tracker */}
              <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
                <SignalPerformanceTracker
                  symbol={selectedSymbol}
                  timeframe={selectedTimeframe}
                  signals={signals}
                />
              </div>
              
              {/* Combined Signals */}
              <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
                <h3 className="text-lg font-medium mb-4">Combined Signals</h3>
                {combinedSignals.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No combined signals available.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {combinedSignals.map((signal, index) => (
                      <div key={index} className="border rounded-lg p-4 bg-gray-50">
                        <div className="flex justify-between items-center mb-3">
                          <div className="flex items-center space-x-2">
                            <SignalTypeBadge type={signal.type} />
                            <span className="font-medium">{signal.symbol}</span>
                          </div>
                          <span className="text-sm text-gray-500">
                            {format(new Date(signal.timestamp), 'MMM d, yyyy HH:mm')}
                          </span>
                        </div>
                        <div className="mb-3">
                          <div className="flex justify-between mb-1">
                            <span className="text-sm text-gray-600">Strength</span>
                            <span className="text-sm font-medium">{(signal.strength * 100).toFixed(0)}%</span>
                          </div>
                          <SignalStrengthIndicator strength={signal.strength} type={signal.type} />
                        </div>
                        <div className="mb-3">
                          <div className="flex justify-between mb-1">
                            <span className="text-sm text-gray-600">Confidence</span>
                            <span className="text-sm font-medium">{signal.confidence ? `${(signal.confidence * 100).toFixed(0)}%` : 'N/A'}</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div 
                              className="h-2.5 rounded-full bg-blue-500"
                              style={{ width: `${signal.confidence ? signal.confidence * 100 : 0}%` }}
                            ></div>
                          </div>
                        </div>
                        <p className="text-sm text-gray-700">{signal.description}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              {/* Technical Signals */}
              <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
                <h3 className="text-lg font-medium mb-4">Technical Signals</h3>
                {technicalSignals.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No technical signals available.
                  </div>
                ) : (
                  <Table>
                    <thead>
                      <tr>
                        <th>Type</th>
                        <th>Symbol</th>
                        <th>Strength</th>
                        <th>Confidence</th>
                        <th>Time</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {technicalSignals.map((signal, index) => (
                        <tr key={index} className="hover:bg-gray-50">
                          <td><SignalTypeBadge type={signal.type} /></td>
                          <td className="font-medium">{signal.symbol}</td>
                          <td>
                            <div className="w-32">
                              <SignalStrengthIndicator strength={signal.strength} type={signal.type} />
                            </div>
                          </td>
                          <td>{signal.confidence ? `${(signal.confidence * 100).toFixed(0)}%` : 'N/A'}</td>
                          <td>{format(new Date(signal.timestamp), 'MMM d, HH:mm')}</td>
                          <td className="max-w-md truncate">{signal.description}</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                )}
              </div>
              
              {/* Sentiment Signals */}
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <h3 className="text-lg font-medium mb-4">Sentiment Signals</h3>
                {sentimentSignals.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    No sentiment signals available.
                  </div>
                ) : (
                  <Table>
                    <thead>
                      <tr>
                        <th>Type</th>
                        <th>Symbol</th>
                        <th>Strength</th>
                        <th>Confidence</th>
                        <th>Time</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sentimentSignals.map((signal, index) => (
                        <tr key={index} className="hover:bg-gray-50">
                          <td><SignalTypeBadge type={signal.type} /></td>
                          <td className="font-medium">{signal.symbol}</td>
                          <td>
                            <div className="w-32">
                              <SignalStrengthIndicator strength={signal.strength} type={signal.type} />
                            </div>
                          </td>
                          <td>{signal.confidence ? `${(signal.confidence * 100).toFixed(0)}%` : 'N/A'}</td>
                          <td>{format(new Date(signal.timestamp), 'MMM d, HH:mm')}</td>
                          <td className="max-w-md truncate">{signal.description}</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                )}
              </div>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
};

export default AdvancedSignalDashboard;
