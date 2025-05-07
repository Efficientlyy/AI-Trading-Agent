import React from 'react';
import { SignalData } from '../../types/signals';

interface SignalContributionBreakdownProps {
  combinedSignal: SignalData;
  technicalSignals: SignalData[];
  sentimentSignals: SignalData[];
}

const SignalContributionBreakdown: React.FC<SignalContributionBreakdownProps> = ({
  combinedSignal,
  technicalSignals,
  sentimentSignals
}) => {
  // Calculate the contribution of each indicator to the final signal
  const calculateContributions = () => {
    // Get the most recent technical signals (up to 5)
    const recentTechnicalSignals = [...technicalSignals]
      .sort((a, b) => {
        const dateA = new Date(a.timestamp);
        const dateB = new Date(b.timestamp);
        return dateB.getTime() - dateA.getTime();
      })
      .slice(0, 5);
    
    // Get the most recent sentiment signals (up to 3)
    const recentSentimentSignals = [...sentimentSignals]
      .sort((a, b) => {
        const dateA = new Date(a.timestamp);
        const dateB = new Date(b.timestamp);
        return dateB.getTime() - dateA.getTime();
      })
      .slice(0, 3);
    
    // Calculate technical contribution
    const technicalContribution = recentTechnicalSignals.reduce((acc, signal) => {
      // Normalize signal type to a value
      const signalValue = signal.type.includes('BUY') 
        ? signal.strength 
        : signal.type.includes('SELL') 
          ? -signal.strength 
          : 0;
      
      // Get indicator type from description
      const indicatorMatch = signal.description?.match(/based on (\w+)/) || [];
      const indicator = indicatorMatch[1] || 'Unknown';
      
      return {
        ...acc,
        [indicator]: (acc[indicator] || 0) + signalValue
      };
    }, {} as Record<string, number>);
    
    // Calculate sentiment contribution
    const sentimentContribution = recentSentimentSignals.reduce((acc, signal) => {
      // Normalize signal type to a value
      const signalValue = signal.type.includes('BUY') 
        ? signal.strength 
        : signal.type.includes('SELL') 
          ? -signal.strength 
          : 0;
      
      // Get sentiment source from description
      const sourceMatch = signal.description?.match(/for (\w+)/) || [];
      const source = sourceMatch[1] || 'Unknown';
      
      return {
        ...acc,
        [source]: (acc[source] || 0) + signalValue
      };
    }, {} as Record<string, number>);
    
    return {
      technical: technicalContribution,
      sentiment: sentimentContribution
    };
  };
  
  const contributions = calculateContributions();
  
  // Get the signal direction (positive for buy, negative for sell)
  const signalDirection = combinedSignal.type.includes('BUY') ? 1 : -1;
  
  // Format contribution value
  const formatContribution = (value: number) => {
    const absValue = Math.abs(value);
    const formattedValue = (absValue * 100).toFixed(0);
    return `${formattedValue}%`;
  };
  
  // Get color based on contribution value
  const getContributionColor = (value: number) => {
    if (value * signalDirection > 0) {
      return 'text-green-600';
    } else if (value * signalDirection < 0) {
      return 'text-red-600';
    }
    return 'text-gray-600';
  };
  
  // Get icon based on contribution value
  const getContributionIcon = (value: number) => {
    if (value * signalDirection > 0) {
      return '↑';
    } else if (value * signalDirection < 0) {
      return '↓';
    }
    return '→';
  };
  
  return (
    <div className="signal-contribution-breakdown bg-white p-4 rounded-lg shadow-sm">
      <h3 className="text-lg font-medium mb-3">Signal Contribution Breakdown</h3>
      <p className="text-sm text-gray-600 mb-4">
        How different indicators contributed to the {combinedSignal.type} signal:
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Technical Indicators Contribution */}
        <div className="border rounded-lg p-3 bg-blue-50">
          <h4 className="text-md font-medium text-blue-800 mb-2">Technical Indicators</h4>
          <div className="space-y-2">
            {Object.entries(contributions.technical).length > 0 ? (
              Object.entries(contributions.technical).map(([indicator, value]) => (
                <div key={indicator} className="flex justify-between items-center">
                  <span className="text-sm">{indicator}</span>
                  <span className={`text-sm font-medium ${getContributionColor(value)}`}>
                    {getContributionIcon(value)} {formatContribution(value)}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-sm text-gray-500 italic">No technical indicators</div>
            )}
          </div>
        </div>
        
        {/* Sentiment Sources Contribution */}
        <div className="border rounded-lg p-3 bg-purple-50">
          <h4 className="text-md font-medium text-purple-800 mb-2">Sentiment Sources</h4>
          <div className="space-y-2">
            {Object.entries(contributions.sentiment).length > 0 ? (
              Object.entries(contributions.sentiment).map(([source, value]) => (
                <div key={source} className="flex justify-between items-center">
                  <span className="text-sm">{source}</span>
                  <span className={`text-sm font-medium ${getContributionColor(value)}`}>
                    {getContributionIcon(value)} {formatContribution(value)}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-sm text-gray-500 italic">No sentiment sources</div>
            )}
          </div>
        </div>
      </div>
      
      <div className="mt-4 p-3 bg-gray-50 rounded border border-gray-200">
        <h4 className="text-sm font-medium mb-1">How to Interpret</h4>
        <p className="text-xs text-gray-700">
          <span className="text-green-600">↑</span> indicates a positive contribution toward the signal direction,
          <span className="text-red-600 ml-1">↓</span> indicates a negative contribution (conflicting with the signal direction).
          Larger percentages indicate stronger influence on the final signal.
        </p>
      </div>
    </div>
  );
};

export default SignalContributionBreakdown;
