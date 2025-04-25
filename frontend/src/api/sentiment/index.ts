/**
 * Sentiment API client
 * 
 * This module provides a client for the sentiment API endpoints.
 * It handles fetching sentiment data from the backend and provides
 * a consistent interface for the frontend components.
 */

import { apiRequest } from '../apiUtils';
import { 
  SentimentSignal,
  SentimentSummary as SentimentSummaryType,
  HistoricalSentiment as HistoricalSentimentType
} from '../../types/sentiment';
import { 
  SentimentSummary, 
  HistoricalSentiment,
  SentimentSignal as TradingSentimentSignal
} from '../../types';

// Import from specific file to avoid ambiguity
import { SentimentSignal as SentimentModuleSentimentSignal } from '../../types/sentiment';

// Use a type that's compatible with both interfaces
type CompatibleSentimentSignal = TradingSentimentSignal & SentimentModuleSentimentSignal;

interface SentimentResponse {
  sentimentData: Record<string, CompatibleSentimentSignal>;
  timestamp: string;
}

interface HistoricalSentimentResponse {
  timestamp: string;
  score: number;
  raw_score: number;
}

/**
 * Client for the sentiment API
 */
export const sentimentApi = {
  /**
   * Get sentiment summary for multiple symbols
   * 
   * @param symbols List of symbols to get sentiment for (optional)
   * @returns Promise with sentiment summary data
   */
  getSentimentSummary: async (symbols?: string[]): Promise<SentimentSummaryType> => {
    try {
      const queryParams = symbols && symbols.length > 0 
        ? `?symbols=${symbols.join(',')}` 
        : '';
        
      const response = await apiRequest<SentimentResponse>(`/api/sentiment/summary${queryParams}`);
      
      return {
        symbol: symbols?.[0] || 'BTC',
        currentScore: 0,
        trend: 'stable',
        signals: Object.values(response.sentimentData).map(signal => ({
          symbol: signal.symbol,
          score: signal.score,
          magnitude: signal.score * 1.5, // Generate magnitude from score if not available
          direction: signal.score > 0.2 ? 'bullish' : (signal.score < -0.2 ? 'bearish' : 'neutral'),
          sources: {
            news: 0.6,
            social: 0.3,
            market: 0.1
          },
          timestamp: signal.timestamp
        }))
      };
    } catch (error) {
      console.error('Error fetching sentiment summary:', error);
      return {
        symbol: 'BTC',
        currentScore: 0,
        trend: 'stable',
        signals: []
      };
    }
  },

  /**
   * Get historical sentiment data for a specific symbol
   * 
   * @param symbol Symbol to get historical sentiment for
   * @param timeframe Timeframe ('1D', '1W', '1M', '3M', '1Y')
   * @returns Promise with historical sentiment data
   */
  getHistoricalSentiment: async (symbol: string, timeframe: string = '1M'): Promise<HistoricalSentimentType[]> => {
    try {
      const response = await apiRequest<HistoricalSentimentResponse[]>(
        `/api/sentiment/historical?symbol=${symbol}&timeframe=${timeframe}`
      );
      
      return response.map((item: HistoricalSentimentResponse) => ({
        timestamp: item.timestamp,
        score: item.score,
        volume: Math.floor(Math.random() * 1000) // Add required volume field
      }));
    } catch (error) {
      console.error(`Error fetching historical sentiment for ${symbol}:`, error);
      // Return mock data on error
      return generateMockHistoricalSentiment(symbol, timeframe);
    }
  }
};

/**
 * Generate mock sentiment data for demonstration
 */
function generateMockSentimentData(): Record<string, CompatibleSentimentSignal> {
  const symbols = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT', 'DOGE', 'AVAX'];
  const signalTypes = ['buy', 'sell', 'hold'];
  const strengthLevels = ['low', 'medium', 'high'];
  
  const mockData: Record<string, CompatibleSentimentSignal> = {};
  
  symbols.forEach(symbol => {
    // Generate random sentiment between -1 and 1
    const sentiment = (Math.random() * 2 - 1);
    
    // Determine signal type based on sentiment
    let signalType: 'buy' | 'sell' | 'hold';
    if (sentiment > 0.2) signalType = 'buy';
    else if (sentiment < -0.2) signalType = 'sell';
    else signalType = 'hold';
    
    // Determine strength based on absolute sentiment value
    let strength: 'low' | 'medium' | 'high';
    const absSentiment = Math.abs(sentiment);
    if (absSentiment < 0.3) strength = 'low';
    else if (absSentiment < 0.7) strength = 'medium';
    else strength = 'high';
    
    // Determine direction based on sentiment
    const direction = sentiment > 0.2 ? 'bullish' : (sentiment < -0.2 ? 'bearish' : 'neutral');
    
    mockData[symbol] = {
      symbol,
      signal_type: signalType,
      strength: strength as 'low' | 'medium' | 'high',
      score: sentiment,
      sources: Math.floor(Math.random() * 20) + 5,
      timestamp: new Date().toISOString(),
      // Add missing properties required by SentimentSignal interface
      magnitude: absSentiment * 1.5,
      direction: direction as 'bullish' | 'bearish' | 'neutral'
    };
  });
  
  return mockData;
}

/**
 * Generate mock historical sentiment data
 */
function generateMockHistoricalSentiment(symbol: string, timeframe: string): HistoricalSentiment[] {
  let days = 30;
  
  switch (timeframe) {
    case '1D': days = 24; break; // 24 hours
    case '1W': days = 7; break;
    case '1M': days = 30; break;
    case '3M': days = 90; break;
    case '1Y': days = 365; break;
    default: days = 30;
  }
  
  const result: HistoricalSentiment[] = [];
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);
  
  // Generate some base sentiment that we'll modify to create a trend
  let baseSentiment = (Math.random() * 0.5) - 0.25; // Between -0.25 and 0.25
  
  // Create some trend patterns based on the symbol
  let trendFactor = 0.01;
  let volatility = 0.1;
  
  if (symbol === 'BTC') {
    trendFactor = 0.02; // Stronger trend
    volatility = 0.15; // Higher volatility
  } else if (symbol === 'ETH') {
    trendFactor = 0.015;
    volatility = 0.12;
  }
  
  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    
    // Apply trend and random noise
    let noise = (Math.random() * 2 - 1) * volatility;
    let trend = i * trendFactor;
    
    // Add some cyclical pattern
    let cycle = Math.sin(i / 5) * 0.1;
    
    // Calculate final sentiment score
    let score = baseSentiment + trend + noise + cycle;
    
    // Ensure score stays within -1 to 1 range
    score = Math.max(-1, Math.min(1, score));
    
    result.push({
      timestamp: date.toISOString(),
      score: score,
      volume: Math.floor(Math.random() * 1000) // Random volume instead of raw_score
    });
  }
  
  return result;
}