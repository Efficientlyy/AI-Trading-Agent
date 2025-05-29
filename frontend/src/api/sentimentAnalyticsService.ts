import { format } from 'date-fns';
import clientCache from '../utils/clientCache';
// Import the API client with an explicit type to help TypeScript
import axios from 'axios';
import type { AxiosInstance } from 'axios';

// Use the apiClient from the separate file
import apiClient from './apiClient';

// Fallback client in case of import issues
const fallbackClient: AxiosInstance = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' }
});

// Use the imported client or fallback
const client = apiClient || fallbackClient;

// API Endpoints
const API_ENDPOINTS = {
  HISTORICAL_SENTIMENT: '/api/sentiment/historical',
  ALL_SYMBOLS_SENTIMENT: '/api/sentiment/all-symbols',
  MONITORED_SYMBOLS: '/api/sentiment/monitored',
  SIGNAL_QUALITY_METRICS: '/api/sentiment/quality-metrics'
};

// Interface for sentiment data point
export interface SentimentDataPoint {
  timestamp: string;
  sentiment_score: number;
  confidence: number;
  volume: number;
}

// Interface for historical sentiment data
export interface SentimentHistoricalData {
  agent_id: string;
  symbol: string;
  timeframe: TimeFrame;
  dataPoints: SentimentDataPoint[];
  metadata: {
    total_records: number;
    average_sentiment: number;
    average_confidence: number;
    updated_at: string;
  };
}

// Interface for symbol sentiment data
export interface SymbolSentimentData {
  symbol: string;
  current_sentiment: number;
  sentiment_trend: 'up' | 'down' | 'neutral';
  volume: number;
  updated_at: string;
}

// Interface for sentiment metrics
export interface SignalQualityMetrics {
  timeframe: TimeFrame;
  overall_accuracy: number;
  overall_precision: number;
  overall_recall: number;
  overall_f1_score: number;
  total_signals: number;
  total_successful_signals: number;
  signal_distribution: {
    positive: number;
    negative: number;
    neutral: number;
  };
  performance_by_symbol: Record<string, {
    accuracy: number;
    signal_count: number;
    avg_confidence: number;
  }>;
}

// Time frames for data aggregation
export type TimeFrame = '24h' | '7d' | '30d' | '90d';

// Cache key generation helper functions
const getCacheKeys = {
  historicalData: (agentId: string, symbol: string, timeframe: TimeFrame) => 
    `sentiment_historical_${agentId}_${symbol}_${timeframe}`,
  allSymbolsData: (agentId: string, timeframe: TimeFrame) => 
    `sentiment_all_symbols_${agentId}_${timeframe}`,
  monitoredSymbols: (agentId: string) => 
    `sentiment_monitored_symbols_${agentId}`,
  signalQualityMetrics: (agentId: string, timeframe: TimeFrame) => 
    `sentiment_quality_metrics_${agentId}_${timeframe}_${new Date().toISOString().split('T')[0]}`
};

// Cache expiry times (in milliseconds)
const CACHE_EXPIRY = {
  SHORT: 2 * 60 * 1000,       // 2 minutes for real-time data
  MEDIUM: 15 * 60 * 1000,     // 15 minutes for frequently changing data
  LONG: 60 * 60 * 1000,       // 1 hour for relatively stable data
  DAILY: 24 * 60 * 60 * 1000  // 24 hours for data that changes daily
};

// Error handling utility
const handleApiError = (error: any, endpoint: string) => {
  console.error(`API Error (${endpoint}):`, error);
  throw new Error(`Failed to fetch data from ${endpoint}: ${error.message || 'Unknown error'}`);
};

// Sentiment analytics service with client-side caching
const sentimentAnalyticsService = {
  // Get historical sentiment data for a specific symbol
  getHistoricalSentimentData: async (
    agentId: string,
    symbol: string | null,
    timeframe: TimeFrame
  ): Promise<SentimentHistoricalData> => {
    if (!symbol) {
      // Default to BTC if no symbol is provided
      symbol = 'BTC';
    }
    
    // Generate cache key
    const cacheKey = getCacheKeys.historicalData(agentId, symbol, timeframe);
    
    // Check cache first
    const cachedData = clientCache.get<SentimentHistoricalData>(cacheKey);
    if (cachedData) {
      console.log(`Using cached sentiment data for ${symbol}, timeframe: ${timeframe}`);
      return cachedData;
    }
    
    console.log(`Fetching fresh sentiment data for ${symbol}, timeframe: ${timeframe}`);
    
    try {
      // Call the real API endpoint
      const response = await client.get(API_ENDPOINTS.HISTORICAL_SENTIMENT, {
        params: { agentId, symbol, timeframe }
      });
      
      const result: SentimentHistoricalData = response.data;
      
      // Store in cache with expiry
      clientCache.set(cacheKey, result, CACHE_EXPIRY.MEDIUM);
      return result;
    } catch (error) {
      handleApiError(error, API_ENDPOINTS.HISTORICAL_SENTIMENT);
      throw error; // Re-throw for component-level error handling
    }
  },
  
  // Get sentiment data for all monitored symbols in a timeframe
  getAllSymbolsSentimentData: async (
    agentId: string,
    timeframe: TimeFrame
  ): Promise<Record<string, SentimentDataPoint[]>> => {
    // Generate cache key
    const cacheKey = getCacheKeys.allSymbolsData(agentId, timeframe);
    
    // Check cache first
    const cachedData = clientCache.get<Record<string, SentimentDataPoint[]>>(cacheKey);
    if (cachedData) {
      console.log(`Using cached sentiment data for all symbols, timeframe: ${timeframe}`);
      return cachedData;
    }
    
    console.log(`Fetching fresh sentiment data for all symbols, timeframe: ${timeframe}`);
    
    try {
      // Call the real API endpoint
      const response = await client.get(API_ENDPOINTS.ALL_SYMBOLS_SENTIMENT, {
        params: { agentId, timeframe }
      });
      
      const result: Record<string, SentimentDataPoint[]> = response.data;
      
      // Store in cache with expiry
      clientCache.set(cacheKey, result, CACHE_EXPIRY.MEDIUM);
      return result;
    } catch (error) {
      handleApiError(error, API_ENDPOINTS.ALL_SYMBOLS_SENTIMENT);
      throw error; // Re-throw for component-level error handling
    }
  },
  
  // Get list of all monitored symbols with their latest sentiment
  getMonitoredSymbolsWithSentiment: async (agentId: string): Promise<{
    symbol: string;
    latest_sentiment: number;
    sentiment_change: number;
    confidence: number;
  }[]> => {
    // Generate cache key
    const cacheKey = getCacheKeys.monitoredSymbols(agentId);
    
    // Check cache first
    const cachedData = clientCache.get<{
      symbol: string;
      latest_sentiment: number;
      sentiment_change: number;
      confidence: number;
    }[]>(cacheKey);
    if (cachedData) {
      console.log('Using cached monitored symbols with sentiment');
      return cachedData;
    }
    
    console.log('Fetching fresh monitored symbols with sentiment');
    
    try {
      // Call the real API endpoint
      const response = await client.get(API_ENDPOINTS.MONITORED_SYMBOLS, {
        params: { agentId }
      });
      
      const result = response.data;
      
      // Store in cache with expiry
      clientCache.set(cacheKey, result, CACHE_EXPIRY.SHORT);
      return result;
    } catch (error) {
      handleApiError(error, API_ENDPOINTS.MONITORED_SYMBOLS);
      throw error; // Re-throw for component-level error handling
    }
  },
  
  // Get detailed signal quality metrics
  getSignalQualityMetrics: async (
    agentId: string,
    timeframe: TimeFrame
  ): Promise<SignalQualityMetrics> => {
    // Generate cache key
    const cacheKey = getCacheKeys.signalQualityMetrics(agentId, timeframe);
    
    // Check cache first
    const cachedData = clientCache.get<SignalQualityMetrics>(cacheKey);
    if (cachedData) {
      console.log(`Using cached signal quality metrics for timeframe: ${timeframe}`);
      return cachedData;
    }
    
    console.log(`Fetching fresh signal quality metrics for timeframe: ${timeframe}`);
    
    try {
      // Call the real API endpoint
      const response = await client.get(API_ENDPOINTS.SIGNAL_QUALITY_METRICS, {
        params: { agentId, timeframe }
      });
      
      const result: SignalQualityMetrics = response.data;
      
      // Store in cache with expiry
      clientCache.set(cacheKey, result, CACHE_EXPIRY.DAILY);
      return result;
    } catch (error) {
      handleApiError(error, API_ENDPOINTS.SIGNAL_QUALITY_METRICS);
      throw error; // Re-throw for component-level error handling
    }
  }
};

export default sentimentAnalyticsService;
