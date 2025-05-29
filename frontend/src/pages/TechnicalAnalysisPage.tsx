import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import TechnicalAnalysisChart from '../components/dashboard/TechnicalAnalysisChart';
import PatternDashboard from '../components/dashboard/PatternDashboard';
import PatternVisualizationControls from '../components/dashboard/PatternVisualizationControls';
import { PatternDetectionResult, PatternType } from '../types/patterns';
// Import UI components directly from our component folder
import { Card, CardHeader, CardBody } from '../components/ui/Card';
import { Spinner, Tabs, Tab } from '../components/ui/index';
import { Switch } from '../components/ui/Switch';

// Create a simple Select component to replace the imported one
interface SelectProps {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  options: Array<{ value: string; label: string }>;
  className?: string;
}

const Select: React.FC<SelectProps> = ({ value, onChange, options, className }) => {
  return (
    <select value={value} onChange={onChange} className={`bg-gray-700 text-white rounded px-3 py-2 ${className}`}>
      {options.map(option => (
        <option key={option.value} value={option.value}>{option.label}</option>
      ))}
    </select>
  );
};

const availableSymbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'ADA/USD', 'SOL/USD'];
const availableTimeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'] as const;

const TechnicalAnalysisPage: React.FC = () => {
  // State
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USD');
  const [selectedTimeframe, setSelectedTimeframe] = useState<typeof availableTimeframes[number]>('1d');
  const [isLoading, setIsLoading] = useState(false);
  const [ohlcvData, setOhlcvData] = useState<any[]>([]);
  const [patternData, setPatternData] = useState<PatternDetectionResult[]>([]);
  const [showPatterns, setShowPatterns] = useState(true);
  const [patternVisibility, setPatternVisibility] = useState<Record<PatternType, boolean>>(
    Object.values(PatternType).reduce((acc, type) => ({ ...acc, [type]: true }), {} as Record<PatternType, boolean>)
  );
  const [displayedPatterns, setDisplayedPatterns] = useState<PatternType[]>([]);
  const [activeTab, setActiveTab] = useState('chart');
  const [useMockData, setUseMockData] = useState(true);

  // Web socket
  const { data: wsData, getOHLCVStream, getTechnicalPatterns, status: wsStatus, setDataMode } = useWebSocket([]);

  // Calculate pattern counts for the selected symbol
  const patternCounts = patternData
    .filter(p => p.symbol === selectedSymbol)
    .reduce((acc, pattern) => {
      if (!acc[pattern.pattern_type]) {
        acc[pattern.pattern_type] = 0;
      }
      acc[pattern.pattern_type]++;
      return acc;
    }, {} as Record<PatternType, number>);

  // Fetch initial data
  useEffect(() => {
    if (wsStatus === 'connected') {
      // Set initial data mode
      setDataMode(useMockData);
      
      // Subscribe to OHLCV data for the selected symbol
      getOHLCVStream(selectedSymbol, selectedTimeframe);
      
      // Subscribe to technical patterns
      getTechnicalPatterns(selectedSymbol);
      
      setIsLoading(false);
    } else {
      setIsLoading(true);
    }
  }, [wsStatus, selectedSymbol, selectedTimeframe, useMockData, getOHLCVStream, getTechnicalPatterns, setDataMode]);

  // Process OHLCV data from websocket
  useEffect(() => {
    if (wsData?.ohlcv && wsData.ohlcv.symbol === selectedSymbol && wsData.ohlcv.timeframe === selectedTimeframe) {
      const data = Array.isArray(wsData.ohlcv.data) ? wsData.ohlcv.data : [wsData.ohlcv.data];
      setOhlcvData(prev => {
        // If it's a full snapshot, replace the data
        if (Array.isArray(wsData.ohlcv?.data)) {
          return data;
        }
        
        // Otherwise update the last candle or add a new one
        const lastCandle = prev[prev.length - 1];
        if (lastCandle && lastCandle.timestamp === data[0].timestamp) {
          return [...prev.slice(0, -1), data[0]];
        }
        return [...prev, data[0]];
      });
    }
  }, [wsData?.ohlcv, selectedSymbol, selectedTimeframe]);

  // Process pattern data from websocket
  useEffect(() => {
    if (wsData?.patterns && wsData.patterns.symbol === selectedSymbol) {
      // Convert backend pattern data to match the PatternDetectionResult interface
      const formattedPatterns = wsData.patterns.data.map((p: any) => ({
        pattern_type: p.pattern_type,
        symbol: p.symbol,
        confidence: p.confidence,
        start_time: p.start_time || new Date().toISOString(),
        end_time: p.end_time || new Date().toISOString(),
        start_idx: p.start_idx,
        end_idx: p.end_idx,
        target_price: p.target_price,
        price_level: p.price_level,
        additional_info: p
      }));
      
      setPatternData(formattedPatterns);
      
      // Update displayed pattern types
      const uniqueTypes = Array.from(
        new Set(
          wsData.patterns.data
            .filter((p: { symbol: string; pattern_type: string }) => p.symbol === selectedSymbol)
            .map((p: { symbol: string; pattern_type: string }) => p.pattern_type)
        )
      );
      setDisplayedPatterns(uniqueTypes as PatternType[]);
    }
  }, [wsData?.patterns, selectedSymbol]);

  // Handle symbol change
  const handleSymbolChange = (newSymbol: string) => {
    setSelectedSymbol(newSymbol);
  };

  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe: typeof availableTimeframes[number]) => {
    setSelectedTimeframe(newTimeframe);
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4 text-white">Technical Analysis</h1>
      
      <div className="flex flex-wrap gap-4 mb-4">
        <div className="w-64">
          <label className="block text-sm font-medium text-gray-300 mb-1">Symbol</label>
          <Select
            value={selectedSymbol}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => handleSymbolChange(e.target.value)}
            options={availableSymbols.map(symbol => ({ value: symbol, label: symbol }))}
            className="w-full"
          />
        </div>
        <div className="w-64">
          <label className="block text-sm font-medium text-gray-300 mb-1">Timeframe</label>
          <Select
            value={selectedTimeframe}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) => 
              handleTimeframeChange(e.target.value as typeof availableTimeframes[number])}
            options={availableTimeframes.map(tf => ({ value: tf, label: tf }))}
            className="w-full"
          />
          
          <div className="mt-4">
            <label className="block text-sm font-medium text-gray-300 mb-1">Data Mode</label>
            <div className="flex items-center justify-between px-3 py-2 bg-gray-700 rounded">
              <span className={`text-sm ${useMockData ? 'text-blue-400 font-medium' : 'text-gray-400'}`}>Mock</span>
              <Switch 
                checked={!useMockData} 
                onChange={() => {
                  const newMode = !useMockData;
                  setUseMockData(!newMode);
                  setDataMode(!newMode);
                  // Reset data when switching modes
                  setOhlcvData([]);
                  setPatternData([]);
                  setIsLoading(true);
                  setTimeout(() => setIsLoading(false), 1000);
                }}
                className="mx-2"
              />
              <span className={`text-sm ${!useMockData ? 'text-green-400 font-medium' : 'text-gray-400'}`}>Real</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Main content area - Chart or Pattern Dashboard based on active tab */}
        <div className="lg:col-span-3">
          <Card className="bg-slate-800 shadow-lg">
            <CardHeader className="border-b border-slate-700">
              <div className="flex space-x-4">
                <button 
                  className={`py-2 px-4 ${activeTab === 'chart' ? 'border-b-2 border-blue-500 text-blue-500' : 'text-gray-400'}`}
                  onClick={() => setActiveTab('chart')}
                >
                  Chart
                </button>
                <button 
                  className={`py-2 px-4 ${activeTab === 'patterns' ? 'border-b-2 border-blue-500 text-blue-500' : 'text-gray-400'}`}
                  onClick={() => setActiveTab('patterns')}
                >
                  Patterns
                </button>
              </div>
            </CardHeader>
            <CardBody className="p-0">
              {isLoading ? (
                <div className="flex justify-center items-center h-96">
                  <Spinner size="lg" />
                </div>
              ) : (
                <>
                  {activeTab === 'chart' ? (
                    <TechnicalAnalysisChart
                      symbol={selectedSymbol}
                      data={ohlcvData}
                      isLoading={isLoading}
                      availableSymbols={availableSymbols}
                      onSymbolChange={handleSymbolChange}
                      onTimeframeChange={handleTimeframeChange}
                      timeframe={selectedTimeframe}
                      patterns={patternData}
                      showPatterns={showPatterns}
                      className="h-[600px]"
                    />
                  ) : (
                    <PatternDashboard
                      patterns={patternData}
                      symbol={selectedSymbol}
                      onPatternVisibilityChange={setPatternVisibility}
                      onShowPatternsChange={setShowPatterns}
                      showPatterns={showPatterns}
                      className="h-[600px] overflow-auto"
                    />
                  )}
                </>
              )}
            </CardBody>
          </Card>
        </div>
        
        {/* Sidebar - Pattern Controls */}
        <div className="lg:col-span-1">
          <PatternVisualizationControls
            patternVisibility={patternVisibility}
            onPatternVisibilityChange={setPatternVisibility}
            displayedPatterns={displayedPatterns}
            showPatterns={showPatterns}
            onShowPatternsChange={setShowPatterns}
            patternCounts={patternCounts}
          />
          
          <Card className="bg-slate-800 shadow-md mt-4">
            <CardHeader>
              <h3 className="text-lg font-medium text-white">Technical Analysis</h3>
            </CardHeader>
            <CardBody>
              <div className="text-sm text-gray-300">
                <p className="mb-2">
                  Patterns help identify potential price movements based on historical chart formations.
                </p>
                <p className="mb-2">
                  Toggle pattern visibility using the controls above.
                </p>
                <div className="mt-4">
                  <h4 className="font-medium text-white mb-1">Pattern Legend</h4>
                  <ul className="space-y-1">
                    <li className="flex items-center">
                      <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
                      <span>Support & Trendlines (Bullish)</span>
                    </li>
                    <li className="flex items-center">
                      <div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div>
                      <span>Resistance & Trendlines (Bearish)</span>
                    </li>
                    <li className="flex items-center">
                      <div className="w-3 h-3 rounded-full bg-orange-500 mr-2"></div>
                      <span>Head & Shoulders Patterns</span>
                    </li>
                    <li className="flex items-center">
                      <div className="w-3 h-3 rounded-full bg-purple-500 mr-2"></div>
                      <span>Triangle Patterns</span>
                    </li>
                  </ul>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default TechnicalAnalysisPage;
