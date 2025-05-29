import React from 'react';
import Toggle from '../Form/Toggle';

interface FilterControlsProps {
  selectedSymbols: string[];
  availableSymbols: { value: string; label: string }[];
  timeRange: string;
  showRealTimeData: boolean;
  confidenceThreshold: number;
  onSymbolsChange: (symbols: string[]) => void;
  onTimeRangeChange: (range: string) => void;
  onRealTimeToggle: (enabled: boolean) => void;
  onConfidenceThresholdChange: (threshold: number) => void;
}

const FilterControls: React.FC<FilterControlsProps> = ({
  selectedSymbols,
  availableSymbols,
  timeRange,
  showRealTimeData,
  confidenceThreshold,
  onSymbolsChange,
  onTimeRangeChange,
  onRealTimeToggle,
  onConfidenceThresholdChange
}) => {
  // Handle symbol checkbox changes
  const handleSymbolChange = (symbol: string) => {
    if (selectedSymbols.includes(symbol)) {
      onSymbolsChange(selectedSymbols.filter(s => s !== symbol));
    } else {
      onSymbolsChange([...selectedSymbols, symbol]);
    }
  };

  // Handle select all/none
  const handleSelectAll = () => {
    onSymbolsChange(availableSymbols.map(s => s.value));
  };

  const handleSelectNone = () => {
    onSymbolsChange([]);
  };

  return (
    <div className="bg-gray-800 text-gray-300 rounded-lg shadow p-4 border border-gray-700">
      <div className="mb-4">
        <h3 className="text-lg font-medium mb-2 text-white">Dashboard Controls</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Time Range Selection */}
          <div>
            <label className="block text-sm font-medium mb-1 text-gray-300">Time Range</label>
            <div className="flex space-x-2">
              {['24h', '7d', '30d', '90d'].map(range => (
                <button
                  key={range}
                  className={`px-3 py-1 text-sm rounded-md ${
                    timeRange === range
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                  onClick={() => onTimeRangeChange(range)}
                >
                  {range}
                </button>
              ))}
            </div>
          </div>
          
          {/* Real-time Data Toggle */}
          <div>
            <label className="block text-sm font-medium mb-1 text-gray-300">Real-time Updates</label>
            <Toggle
              id="real-time-toggle"
              checked={showRealTimeData}
              onChange={() => onRealTimeToggle(!showRealTimeData)}
              label={showRealTimeData ? "Enabled" : "Disabled"}
            />
          </div>
          
          {/* Confidence Threshold Slider */}
          <div>
            <label className="block text-sm font-medium mb-1 text-gray-300">
              Confidence Threshold: {confidenceThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => onConfidenceThresholdChange(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
          
          {/* Visualization Type Selection (placeholder) */}
          <div>
            <label className="block text-sm font-medium mb-1 text-gray-300">Visualization Type</label>
            <select 
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="standard">Standard</option>
              <option value="normalized">Normalized</option>
              <option value="relative">Relative Change</option>
            </select>
          </div>
        </div>
      </div>
      
      {/* Asset Selection */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-medium text-white">Assets</h3>
          <div className="space-x-2">
            <button 
              className="px-3 py-1 text-xs bg-blue-100 text-blue-800 rounded-md dark:bg-blue-800 dark:text-blue-100"
              onClick={handleSelectAll}
            >
              Select All
            </button>
            <button 
              className="px-3 py-1 text-xs bg-gray-100 text-gray-800 rounded-md dark:bg-gray-700 dark:text-gray-300"
              onClick={handleSelectNone}
            >
              Clear
            </button>
          </div>
        </div>
        
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
          {availableSymbols.map(({ value, label }) => (
            <div key={value} className="flex items-center">
              <input
                type="checkbox"
                id={`symbol-${value}`}
                checked={selectedSymbols.includes(value)}
                onChange={() => handleSymbolChange(value)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label
                htmlFor={`symbol-${value}`}
                className="ml-2 block text-sm text-gray-900 dark:text-gray-300"
              >
                {label}
              </label>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FilterControls;
