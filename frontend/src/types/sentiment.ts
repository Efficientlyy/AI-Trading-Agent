/**
 * Sentiment Types
 * 
 * This file contains type definitions related to sentiment analysis.
 */

// Sentiment data from various sources
export type SentimentSource = 'news' | 'social' | 'market' | 'combined';

// Sentiment strength levels
export type SentimentStrength = 'weak' | 'moderate' | 'strong';

// Market sentiment types for overall market analysis
export type MarketSentiment = 'strongly_bullish' | 'bullish' | 'neutral' | 'bearish' | 'strongly_bearish';

// Sentiment data structure
export interface SentimentData {
  score: number;
  magnitude: number;
  source: SentimentSource;
  timestamp: string;
  confidence?: number;
}

// Sentiment signal with trading direction
export interface SentimentSignal {
  symbol: string;
  score: number;
  magnitude: number;
  direction: 'bullish' | 'bearish' | 'neutral';
  sources: {
    news?: number;
    social?: number;
    market?: number;
  };
  timestamp: string;
}

// Historical sentiment data point
export interface HistoricalSentiment {
  timestamp: string;
  score: number;
  volume: number;
}

// Sentiment summary for dashboard display
export interface SentimentSummary {
  symbol: string;
  currentScore: number;
  trend: 'up' | 'down' | 'stable';
  signals: SentimentSignal[];
}

/**
 * Helper function to determine sentiment strength based on score
 */
export function getSentimentStrength(score: number): SentimentStrength {
  const absScore = Math.abs(score);
  if (absScore >= 0.7) return 'strong';
  if (absScore >= 0.4) return 'moderate';
  return 'weak';
}

/**
 * Helper function to calculate overall market sentiment
 */
export function calculateMarketSentiment(signals: SentimentSignal[]): MarketSentiment {
  const buyCount = signals.filter(s => s.direction === 'bullish').length;
  const sellCount = signals.filter(s => s.direction === 'bearish').length;
  
  if (buyCount > sellCount * 2) return 'strongly_bullish';
  if (buyCount > sellCount) return 'bullish';
  if (sellCount > buyCount * 2) return 'strongly_bearish';
  if (sellCount > buyCount) return 'bearish';
  return 'neutral';
}