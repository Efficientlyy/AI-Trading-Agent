import React, { useState } from 'react';
import { MarketRegime } from '../../services/AdvancedSignalService';

interface SignalWeightSettingsProps {
  marketRegime: MarketRegime;
  onWeightsChange: (technicalWeight: number, sentimentWeight: number) => void;
}

const SignalWeightSettings: React.FC<SignalWeightSettingsProps> = ({
  marketRegime,
  onWeightsChange
}) => {
  // Define default weights based on market regime
  const getDefaultWeights = () => {
    switch (marketRegime) {
      case MarketRegime.BULLISH:
        return { technical: 0.6, sentiment: 0.4 };
      case MarketRegime.BEARISH:
        return { technical: 0.7, sentiment: 0.3 };
      case MarketRegime.VOLATILE:
        return { technical: 0.3, sentiment: 0.7 };
      case MarketRegime.SIDEWAYS:
        return { technical: 0.5, sentiment: 0.5 };
      default:
        return { technical: 0.5, sentiment: 0.5 };
    }
  };

  const defaultWeights = getDefaultWeights();
  const [technicalWeight, setTechnicalWeight] = useState<number>(defaultWeights.technical);
  const [sentimentWeight, setSentimentWeight] = useState<number>(defaultWeights.sentiment);
  const [isCustom, setIsCustom] = useState<boolean>(false);
  
  // Handle technical weight change
  const handleTechnicalWeightChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTechnicalWeight = parseFloat(e.target.value) / 100;
    setTechnicalWeight(newTechnicalWeight);
    setSentimentWeight(1 - newTechnicalWeight);
    setIsCustom(true);
    onWeightsChange(newTechnicalWeight, 1 - newTechnicalWeight);
  };
  
  // Handle sentiment weight change
  const handleSentimentWeightChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newSentimentWeight = parseFloat(e.target.value) / 100;
    setSentimentWeight(newSentimentWeight);
    setTechnicalWeight(1 - newSentimentWeight);
    setIsCustom(true);
    onWeightsChange(1 - newSentimentWeight, newSentimentWeight);
  };
  
  // Reset to default weights
  const handleReset = () => {
    const defaults = getDefaultWeights();
    setTechnicalWeight(defaults.technical);
    setSentimentWeight(defaults.sentiment);
    setIsCustom(false);
    onWeightsChange(defaults.technical, defaults.sentiment);
  };
  
  return (
    <div className="signal-weight-settings bg-white p-4 rounded-lg shadow-sm">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-lg font-medium">Signal Weight Settings</h3>
        {isCustom && (
          <button 
            onClick={handleReset}
            className="text-sm text-blue-600 hover:text-blue-800"
          >
            Reset to Default
          </button>
        )}
      </div>
      
      <p className="text-sm text-gray-600 mb-4">
        Adjust the weights of technical and sentiment signals to customize your trading strategy:
      </p>
      
      {/* Technical Weight Slider */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-1">
          <label className="text-sm font-medium text-blue-700">Technical Weight</label>
          <span className="text-sm font-medium">{Math.round(technicalWeight * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          step="5"
          value={Math.round(technicalWeight * 100)}
          onChange={handleTechnicalWeightChange}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
      </div>
      
      {/* Sentiment Weight Slider */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-1">
          <label className="text-sm font-medium text-purple-700">Sentiment Weight</label>
          <span className="text-sm font-medium">{Math.round(sentimentWeight * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          step="5"
          value={Math.round(sentimentWeight * 100)}
          onChange={handleSentimentWeightChange}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
      </div>
      
      {/* Visual representation */}
      <div className="h-8 bg-gray-200 rounded-full overflow-hidden mb-4">
        <div 
          className="h-full bg-blue-500 rounded-l-full flex items-center justify-end"
          style={{ width: `${technicalWeight * 100}%` }}
        >
          {technicalWeight >= 0.15 && (
            <span className="text-white text-xs font-bold mr-2">
              Technical {Math.round(technicalWeight * 100)}%
            </span>
          )}
        </div>
        <div 
          className="h-full bg-purple-500 rounded-r-full flex items-center justify-start ml-auto"
          style={{ 
            width: `${sentimentWeight * 100}%`, 
            marginTop: '-2rem' // Hack to position it in the same row
          }}
        >
          {sentimentWeight >= 0.15 && (
            <span className="text-white text-xs font-bold ml-2">
              Sentiment {Math.round(sentimentWeight * 100)}%
            </span>
          )}
        </div>
      </div>
      
      <div className="p-3 bg-gray-50 rounded border border-gray-200">
        <h4 className="text-sm font-medium mb-1">Strategy Insight</h4>
        <p className="text-xs text-gray-700">
          {technicalWeight > 0.7 ? (
            "Your strategy is heavily focused on technical analysis. This works well in trending markets but may miss sentiment shifts."
          ) : sentimentWeight > 0.7 ? (
            "Your strategy is heavily focused on sentiment analysis. This works well for catching market mood shifts but may be more volatile."
          ) : Math.abs(technicalWeight - sentimentWeight) < 0.2 ? (
            "Your strategy is balanced between technical and sentiment analysis, providing a good mix of signals."
          ) : technicalWeight > sentimentWeight ? (
            "Your strategy leans toward technical analysis while still considering sentiment."
          ) : (
            "Your strategy leans toward sentiment analysis while still considering technical factors."
          )}
        </p>
      </div>
    </div>
  );
};

export default SignalWeightSettings;
