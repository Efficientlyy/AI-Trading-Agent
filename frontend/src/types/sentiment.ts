/**
 * Sentiment data types
 * 
 * These types define the structure of sentiment data used throughout the application.
 */

/**
 * Signal type - buy, sell, or hold
 */
export type SignalType = 'buy' | 'sell' | 'hold';

/**
 * Sentiment strength level
 */
export type SentimentStrength = 'strong' | 'moderate' | 'weak';

/**
 * Overall market sentiment
 */
export type MarketSentiment = 'strongly_bullish' | 'bullish' | 'neutral' | 'bearish' | 'strongly_bearish';

/**
 * Sentiment signal for a specific asset
 */
export interface SentimentSignal {
  symbol: string;
  signal: SignalType;
  strength: number;
  score: number;
  trend: number;
  volatility: number;
  timestamp: string;
}

/**
 * Summary of sentiment across multiple assets
 */
export interface SentimentSummary {
  sentimentData: Record<string, SentimentSignal>;
  timestamp: string;
}

/**
 * Historical sentiment data point
 */
export interface HistoricalSentiment {
  timestamp: string;
  score: number;
  raw_score: number;
}

/**
 * Sentiment news article
 */
export interface SentimentArticle {
  title: string;
  url: string;
  time_published: string;
  authors?: string[];
  summary?: string;
  source: string;
  overall_sentiment_score: number;
  overall_sentiment_label: string;
  ticker_sentiment_score?: number;
  ticker_sentiment_label?: string;
  relevance_score?: number;
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
  const buyCount = signals.filter(s => s.signal === 'buy').length;
  const sellCount = signals.filter(s => s.signal === 'sell').length;
  
  if (buyCount > sellCount * 2) return 'strongly_bullish';
  if (buyCount > sellCount) return 'bullish';
  if (sellCount > buyCount * 2) return 'strongly_bearish';
  if (sellCount > buyCount) return 'bearish';
  return 'neutral';
}