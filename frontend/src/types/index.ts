/**
 * Type definitions export
 * 
 * This file exports all types used throughout the application.
 */

// Re-export types from other files
export * from './trading';

// Re-export sentiment types but handle SentimentSignal explicitly to avoid ambiguity
export { getSentimentStrength, calculateMarketSentiment } from './sentiment';

// Use 'export type' for type re-exports to satisfy the isolatedModules flag
export type { 
  SentimentData,
  SentimentSource,
  SentimentStrength,
  MarketSentiment,
  HistoricalSentiment,
  SentimentSummary
} from './sentiment';