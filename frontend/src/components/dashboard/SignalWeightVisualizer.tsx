import React from 'react';
import { MarketRegime } from '../../services/AdvancedSignalService';

interface SignalWeightVisualizerProps {
  marketRegime: MarketRegime;
}

const SignalWeightVisualizer: React.FC<SignalWeightVisualizerProps> = ({ marketRegime }) => {
  // Define weights based on market regime
  const getWeights = () => {
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

  const weights = getWeights();
  
  // Get color based on market regime
  const getRegimeColor = () => {
    switch (marketRegime) {
      case MarketRegime.BULLISH:
        return 'bg-green-500';
      case MarketRegime.BEARISH:
        return 'bg-red-500';
      case MarketRegime.VOLATILE:
        return 'bg-purple-500';
      case MarketRegime.SIDEWAYS:
        return 'bg-yellow-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="signal-weight-visualizer p-4 bg-white rounded-lg shadow-sm">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-medium">Signal Weights in {marketRegime} Market</h3>
        <div className={`w-4 h-4 rounded-full ${getRegimeColor()}`}></div>
      </div>
      
      <div className="mb-4">
        <p className="text-sm text-gray-600 mb-1">
          How technical and sentiment signals are combined in the current market regime:
        </p>
      </div>
      
      <div className="flex items-center mb-2">
        <div className="w-24 text-right pr-2">
          <span className="text-blue-600 font-medium">Technical</span>
        </div>
        <div className="flex-grow h-8 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className="h-full bg-blue-500 rounded-full flex items-center justify-end pr-2"
            style={{ width: `${weights.technical * 100}%` }}
          >
            <span className="text-white text-xs font-bold">{Math.round(weights.technical * 100)}%</span>
          </div>
        </div>
      </div>
      
      <div className="flex items-center">
        <div className="w-24 text-right pr-2">
          <span className="text-purple-600 font-medium">Sentiment</span>
        </div>
        <div className="flex-grow h-8 bg-gray-200 rounded-full overflow-hidden">
          <div 
            className="h-full bg-purple-500 rounded-full flex items-center justify-end pr-2"
            style={{ width: `${weights.sentiment * 100}%` }}
          >
            <span className="text-white text-xs font-bold">{Math.round(weights.sentiment * 100)}%</span>
          </div>
        </div>
      </div>
      
      <div className="mt-4 p-3 bg-gray-50 rounded border border-gray-200">
        <h4 className="text-sm font-medium mb-2">How Weights Are Determined</h4>
        <ul className="text-xs text-gray-700 space-y-1">
          <li><span className="font-medium text-green-600">Bullish Markets:</span> Technical analysis has higher weight (60/40)</li>
          <li><span className="font-medium text-red-600">Bearish Markets:</span> Technical analysis has even higher weight (70/30)</li>
          <li><span className="font-medium text-purple-600">Volatile Markets:</span> Sentiment analysis has higher weight (30/70)</li>
          <li><span className="font-medium text-yellow-600">Sideways Markets:</span> Equal weighting (50/50)</li>
        </ul>
      </div>
    </div>
  );
};

export default SignalWeightVisualizer;
