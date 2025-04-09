import React, { useMemo } from 'react';
import { SentimentSignal } from '../../types';
import { ArrowUpIcon, ArrowDownIcon, MinusIcon } from '@heroicons/react/24/solid';

interface SentimentSummaryProps {
  sentimentData: Record<string, SentimentSignal> | null;
  isLoading: boolean;
}

const SentimentSummary: React.FC<SentimentSummaryProps> = ({ sentimentData, isLoading }) => {
  const sentimentAnalysis = useMemo(() => {
    if (!sentimentData) return { buyCount: 0, sellCount: 0, holdCount: 0, signals: [] };

    const signals = Object.entries(sentimentData).map(([symbol, data]) => ({
      symbol,
      signal: data.signal,
      strength: data.strength
    }));

    // Sort by strength (descending) and then by symbol
    signals.sort((a, b) => {
      if (b.strength !== a.strength) return b.strength - a.strength;
      return a.symbol.localeCompare(b.symbol);
    });

    // Count signals by type
    const buyCount = signals.filter(s => s.signal === 'buy').length;
    const sellCount = signals.filter(s => s.signal === 'sell').length;
    const holdCount = signals.filter(s => s.signal === 'hold').length;

    return { buyCount, sellCount, holdCount, signals };
  }, [sentimentData]);

  // Calculate overall market sentiment
  const overallSentiment = useMemo(() => {
    if (!sentimentData || Object.keys(sentimentData).length === 0) return 'neutral';

    const { buyCount, sellCount } = sentimentAnalysis;
    
    if (buyCount > sellCount * 2) return 'strongly_bullish';
    if (buyCount > sellCount) return 'bullish';
    if (sellCount > buyCount * 2) return 'strongly_bearish';
    if (sellCount > buyCount) return 'bearish';
    return 'neutral';
  }, [sentimentData, sentimentAnalysis]);

  // Get signal icon and color
  const getSignalDisplay = (signal: 'buy' | 'sell' | 'hold', strength: number) => {
    const strengthClass = strength > 0.7 ? 'font-bold' : strength > 0.4 ? 'font-medium' : 'font-normal';
    
    if (signal === 'buy') {
      return {
        icon: <ArrowUpIcon className="h-4 w-4 text-green-500" />,
        textColor: 'text-green-600 dark:text-green-400',
        bgColor: 'bg-green-100 dark:bg-green-900',
        label: 'Buy',
        strengthClass
      };
    } else if (signal === 'sell') {
      return {
        icon: <ArrowDownIcon className="h-4 w-4 text-red-500" />,
        textColor: 'text-red-600 dark:text-red-400',
        bgColor: 'bg-red-100 dark:bg-red-900',
        label: 'Sell',
        strengthClass
      };
    } else {
      return {
        icon: <MinusIcon className="h-4 w-4 text-gray-500" />,
        textColor: 'text-gray-600 dark:text-gray-400',
        bgColor: 'bg-gray-100 dark:bg-gray-800',
        label: 'Hold',
        strengthClass
      };
    }
  };

  // Get overall sentiment display
  const getOverallSentimentDisplay = () => {
    switch (overallSentiment) {
      case 'strongly_bullish':
        return {
          label: 'Strongly Bullish',
          bgColor: 'bg-green-600',
          textColor: 'text-white'
        };
      case 'bullish':
        return {
          label: 'Bullish',
          bgColor: 'bg-green-400',
          textColor: 'text-white'
        };
      case 'strongly_bearish':
        return {
          label: 'Strongly Bearish',
          bgColor: 'bg-red-600',
          textColor: 'text-white'
        };
      case 'bearish':
        return {
          label: 'Bearish',
          bgColor: 'bg-red-400',
          textColor: 'text-white'
        };
      default:
        return {
          label: 'Neutral',
          bgColor: 'bg-gray-400',
          textColor: 'text-white'
        };
    }
  };

  if (isLoading) {
    return (
      <div className="dashboard-widget col-span-1">
        <h2 className="text-lg font-semibold mb-3">Market Sentiment</h2>
        <div className="animate-pulse space-y-2">
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded"></div>
          <div className="h-24 bg-gray-200 dark:bg-gray-700 rounded"></div>
          <div className="h-36 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  const sentimentDisplay = getOverallSentimentDisplay();

  return (
    <div className="dashboard-widget col-span-1">
      <h2 className="text-lg font-semibold mb-3">Market Sentiment</h2>
      
      {!sentimentData || Object.keys(sentimentData).length === 0 ? (
        <div className="text-gray-500 dark:text-gray-400 text-center py-24">
          No sentiment data available
        </div>
      ) : (
        <>
          {/* Overall Sentiment */}
          <div className={`${sentimentDisplay.bgColor} ${sentimentDisplay.textColor} p-3 rounded-md mb-4 text-center`}>
            <h3 className="font-medium">Overall Market Sentiment</h3>
            <p className="text-xl font-bold">{sentimentDisplay.label}</p>
          </div>
          
          {/* Signal Distribution */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Signal Distribution</h3>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-green-100 dark:bg-green-900 p-2 rounded">
                <div className="text-green-600 dark:text-green-400 font-medium">Buy</div>
                <div className="text-lg font-bold">{sentimentAnalysis.buyCount}</div>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded">
                <div className="text-gray-600 dark:text-gray-400 font-medium">Hold</div>
                <div className="text-lg font-bold">{sentimentAnalysis.holdCount}</div>
              </div>
              <div className="bg-red-100 dark:bg-red-900 p-2 rounded">
                <div className="text-red-600 dark:text-red-400 font-medium">Sell</div>
                <div className="text-lg font-bold">{sentimentAnalysis.sellCount}</div>
              </div>
            </div>
          </div>
          
          {/* Signal List */}
          <div>
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Top Signals</h3>
            <div className="max-h-64 overflow-y-auto pr-1">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-800">
                  <tr>
                    <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Symbol
                    </th>
                    <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Signal
                    </th>
                    <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Strength
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                  {sentimentAnalysis.signals.map((item) => {
                    const display = getSignalDisplay(item.signal, item.strength);
                    return (
                      <tr key={item.symbol}>
                        <td className="px-3 py-2 whitespace-nowrap text-sm font-medium">
                          {item.symbol}
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs ${display.bgColor} ${display.textColor}`}>
                            {display.icon}
                            <span className="ml-1">{display.label}</span>
                          </span>
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm">
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${
                                item.signal === 'buy' 
                                  ? 'bg-green-500' 
                                  : item.signal === 'sell' 
                                    ? 'bg-red-500' 
                                    : 'bg-gray-500'
                              }`}
                              style={{ width: `${item.strength * 100}%` }}
                            ></div>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default SentimentSummary;
