import React, { useMemo } from 'react';
import { SentimentSignal } from '../../types';
import { ArrowUpIcon, ArrowDownIcon, MinusIcon } from '@heroicons/react/24/solid';

export interface SentimentSummaryProps {
  sentimentData: Record<string, SentimentSignal> | null;
  isLoading: boolean;
  onSymbolSelect?: (symbol: string) => void;
  selectedSymbol?: string;
}

const SentimentSummary: React.FC<SentimentSummaryProps> = ({ sentimentData, isLoading, onSymbolSelect, selectedSymbol }) => {
  const sentimentAnalysis = useMemo(() => {
    if (!sentimentData) return { buyCount: 0, sellCount: 0, holdCount: 0, signals: [] };

    const signals = Object.entries(sentimentData).map(([symbol, data]) => ({
      symbol,
      signal: data.signal,
      strength: data.strength,
    }));

    // Sort by absolute strength (strongest signals first)
    signals.sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength));

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

  // Handle symbol click
  const handleSymbolClick = (symbol: string) => {
    if (onSymbolSelect) {
      onSymbolSelect(symbol);
    }
  };

  if (isLoading) {
    return (
      <div className="dashboard-widget col-span-1">
        <h2 className="text-lg font-semibold mb-3">Market Sentiment</h2>
        <div className="animate-pulse space-y-2">
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
        </div>
      </div>
    );
  }

  const sentimentDisplay = getOverallSentimentDisplay();

  return (
    <div className="dashboard-widget col-span-1">
      <h2 className="text-lg font-semibold mb-3">Market Sentiment</h2>
      
      {/* Sentiment Summary */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="bg-green-50 dark:bg-green-900/20 p-2 rounded text-center">
          <div className="text-green-600 dark:text-green-400 text-lg font-bold">{sentimentAnalysis.buyCount}</div>
          <div className="text-xs text-green-800 dark:text-green-300">Bullish</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-800 p-2 rounded text-center">
          <div className="text-gray-600 dark:text-gray-400 text-lg font-bold">{sentimentAnalysis.holdCount}</div>
          <div className="text-xs text-gray-800 dark:text-gray-300">Neutral</div>
        </div>
        <div className="bg-red-50 dark:bg-red-900/20 p-2 rounded text-center">
          <div className="text-red-600 dark:text-red-400 text-lg font-bold">{sentimentAnalysis.sellCount}</div>
          <div className="text-xs text-red-800 dark:text-red-300">Bearish</div>
        </div>
      </div>
      
      {/* Overall Sentiment */}
      <div className={`${sentimentDisplay.bgColor} ${sentimentDisplay.textColor} p-3 rounded-md mb-4 text-center`}>
        <h3 className="font-medium">Overall Market Sentiment</h3>
        <p className="text-xl font-bold">{sentimentDisplay.label}</p>
      </div>
      
      {/* Top Signals */}
      <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Top Signals</h3>
      {sentimentAnalysis.signals.length === 0 ? (
        <div className="text-gray-500 dark:text-gray-400 text-center py-4">
          No sentiment data available
        </div>
      ) : (
        <div className="space-y-2">
          {sentimentAnalysis.signals.slice(0, 5).map((signal) => {
            const display = getSignalDisplay(signal.signal, signal.strength);
            return (
              <div 
                key={signal.symbol}
                className={`p-2 rounded border ${
                  selectedSymbol === signal.symbol 
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                    : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800'
                } cursor-pointer transition-colors`}
                onClick={() => handleSymbolClick(signal.symbol)}
              >
                <div className="flex justify-between items-center">
                  <span className="font-medium">{signal.symbol}</span>
                  <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs ${display.bgColor} ${display.textColor}`}>
                    {display.icon}
                    <span className="ml-1">{display.label}</span>
                  </span>
                </div>
                <div className="flex justify-between items-center mt-1 text-sm">
                  <span className="text-gray-600 dark:text-gray-400">
                    Strength: {signal.strength.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
      
      {/* Info text */}
      {onSymbolSelect && (
        <div className="mt-3 text-xs text-gray-500 dark:text-gray-400 text-center">
          Click on a symbol to analyze it
        </div>
      )}
    </div>
  );
};

export default SentimentSummary;
