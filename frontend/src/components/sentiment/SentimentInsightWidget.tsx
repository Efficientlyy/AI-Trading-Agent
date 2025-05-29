import React, { useMemo } from 'react';
import { Card, CardHeader, CardBody } from '../Card';
import { Alert } from '../Alert';

interface InsightItem {
  type: 'positive' | 'negative' | 'neutral' | 'alert';
  title: string;
  description: string;
  strength?: number; // 0-1 value representing insight strength/confidence
  relatedSymbols?: string[];
}

// Use a union type to ensure all variations are properly typed
type InsightWithOrWithoutOptionals = 
  | InsightItem 
  | { type: string; title: string; description: string; }

interface SentimentInsightWidgetProps {
  sentimentData: any; // The sentiment data object
  selectedSymbols: string[];
  confidenceThreshold: number;
}

const SentimentInsightWidget: React.FC<SentimentInsightWidgetProps> = ({ 
  sentimentData, 
  selectedSymbols,
  confidenceThreshold
}) => {
  // Generate insights based on sentiment data
  const insights = useMemo(() => {
    if (!sentimentData) return [];
    
    const results: InsightItem[] = [];
    
    // Check if we have the required data structures
    if (!sentimentData.top_tickers || !Array.isArray(sentimentData.top_tickers)) {
      return [{
        type: 'neutral',
        title: 'No detailed data available',
        description: 'Unable to generate insights due to missing data.'
      }];
    }
    
    // Filter to selected symbols
    const filteredTickers = sentimentData.top_tickers.filter(
      (ticker: any) => selectedSymbols.includes(ticker.symbol)
    );
    
    // Find strongest positive sentiment
    const mostPositive = [...filteredTickers].sort(
      (a: any, b: any) => b.sentiment - a.sentiment
    )[0];
    
    if (mostPositive && mostPositive.sentiment > 0.3) {
      results.push({
        type: 'positive',
        title: `Strong positive sentiment for ${mostPositive.symbol}`,
        description: `${mostPositive.symbol} shows a sentiment score of ${mostPositive.sentiment.toFixed(2)}, indicating bullish market sentiment.`,
        strength: mostPositive.sentiment,
        relatedSymbols: [mostPositive.symbol]
      });
    }
    
    // Find strongest negative sentiment
    const mostNegative = [...filteredTickers].sort(
      (a: any, b: any) => a.sentiment - b.sentiment
    )[0];
    
    if (mostNegative && mostNegative.sentiment < -0.3) {
      results.push({
        type: 'negative',
        title: `Strong negative sentiment for ${mostNegative.symbol}`,
        description: `${mostNegative.symbol} shows a sentiment score of ${mostNegative.sentiment.toFixed(2)}, indicating bearish market sentiment.`,
        strength: Math.abs(mostNegative.sentiment),
        relatedSymbols: [mostNegative.symbol]
      });
    }
    
    // Detect sentiment divergence (technical vs sentiment)
    const divergences = filteredTickers.filter((ticker: any) => {
      return ticker.technical_signal && ticker.sentiment_signal && 
             ticker.technical_signal !== ticker.sentiment_signal &&
             Math.abs(ticker.sentiment) > 0.2;
    });
    
    if (divergences.length > 0) {
      results.push({
        type: 'alert',
        title: `Signal divergence detected`,
        description: `${divergences.length} assets show divergence between technical and sentiment signals, suggesting potential market reversals.`,
        relatedSymbols: divergences.map((d: any) => d.symbol)
      });
    }
    
    // Look for correlated sentiment movements
    const positiveGroup = filteredTickers.filter((t: any) => t.sentiment > 0.3);
    const negativeGroup = filteredTickers.filter((t: any) => t.sentiment < -0.3);
    
    if (positiveGroup.length > 1 && positiveGroup.length >= filteredTickers.length * 0.5) {
      results.push({
        type: 'positive',
        title: 'Market-wide positive sentiment',
        description: `${positiveGroup.length} of ${filteredTickers.length} selected assets show positive sentiment, suggesting a bullish market trend.`,
        strength: 0.7,
        relatedSymbols: positiveGroup.map((t: any) => t.symbol)
      });
    }
    
    if (negativeGroup.length > 1 && negativeGroup.length >= filteredTickers.length * 0.5) {
      results.push({
        type: 'negative',
        title: 'Market-wide negative sentiment',
        description: `${negativeGroup.length} of ${filteredTickers.length} selected assets show negative sentiment, suggesting a bearish market trend.`,
        strength: 0.7,
        relatedSymbols: negativeGroup.map((t: any) => t.symbol)
      });
    }
    
    // Apply confidence threshold filtering
    const filteredInsights = results.filter(insight => 
      !insight.strength || insight.strength >= confidenceThreshold
    );
    
    // Return default insight if none found
    if (filteredInsights.length === 0) {
      return [{
        type: 'neutral',
        title: 'No significant insights detected',
        description: 'Current sentiment levels do not indicate any noteworthy market trends for the selected assets.'
      }];
    }
    
    return filteredInsights;
  }, [sentimentData, selectedSymbols, confidenceThreshold]);
  
  if (!sentimentData) {
    return (
      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">Market Insights</h3>
        </CardHeader>
        <CardBody>
          <div className="text-center py-4 text-gray-300">Loading insights...</div>
        </CardBody>
      </Card>
    );
  }
  
  return (
    <Card>
      <CardHeader>
        <h3 className="text-lg font-semibold">Market Insights</h3>
      </CardHeader>
      <CardBody>
        <div className="space-y-3">
          {insights.map((insight, index) => (
            <Alert 
              key={index}
              type={insight.type === 'positive' ? 'success' : 
                    insight.type === 'negative' ? 'error' : 
                    insight.type === 'alert' ? 'warning' : 'info'}
            >
              <div className="flex flex-col">
                <span className="font-bold text-white">{insight.title}</span>
                <p>{insight.description}</p>
                {/* Only show related symbols if they exist */}
                {('relatedSymbols' in insight) && insight.relatedSymbols && insight.relatedSymbols.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {insight.relatedSymbols.map((symbol: string) => (
                      <span 
                        key={symbol}
                        className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300"
                      >
                        {symbol}
                      </span>
                    ))}
                  </div>
                )}
                {/* Only show strength bar if it exists */}
                {('strength' in insight) && insight.strength !== undefined && (
                  <div className="mt-2">
                    <div className="text-xs text-gray-400 mb-1">Confidence: {(insight.strength * 100).toFixed(0)}%</div>
                    <div className="w-full bg-gray-200 rounded-full h-1.5 dark:bg-gray-700">
                      <div 
                        className={`h-1.5 rounded-full ${
                          insight.type === 'positive' ? 'bg-green-500' : 
                          insight.type === 'negative' ? 'bg-red-500' : 
                          'bg-yellow-500'
                        }`} 
                        style={{ width: `${insight.strength * 100}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            </Alert>
          ))}
        </div>
      </CardBody>
    </Card>
  );
};

export default SentimentInsightWidget;
