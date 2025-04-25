/**
 * Sentiment API client
 * 
 * This module provides a client for the sentiment API endpoints.
 * It handles fetching sentiment data from the backend and provides
 * a consistent interface for the frontend components.
 */

import { apiRequest } from '../apiUtils';
import { 
  SentimentSummary, 
  HistoricalSentiment,
  SentimentSignal 
} from '../../types';

interface SentimentResponse {
  sentimentData: Record<string, SentimentSignal>;
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
  getSentimentSummary: async (symbols?: string[]): Promise<SentimentSummary> => {
    try {
      const queryParams = symbols && symbols.length > 0 
        ? `?symbols=${symbols.join(',')}` 
        : '';
        
      const response = await apiRequest<SentimentResponse>(`/api/sentiment/summary${queryParams}`);
      
      return {
        sentimentData: response.sentimentData,
        timestamp: response.timestamp
      };
    } catch (error) {
      console.error('Error fetching sentiment summary:', error);
      return {
        sentimentData: {},
        timestamp: new Date().toISOString()
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
  getHistoricalSentiment: async (symbol: string, timeframe: string = '1M'): Promise<HistoricalSentiment[]> => {
    try {
      const response = await apiRequest<HistoricalSentimentResponse[]>(
        `/api/sentiment/historical?symbol=${symbol}&timeframe=${timeframe}`
      );
      
      return response.map(item => ({
        timestamp: item.timestamp,
        score: item.score,
        raw_score: item.raw_score
      }));
    } catch (error) {
      console.error(`Error fetching historical sentiment for ${symbol}:`, error);
      return [];
    }
  }
};