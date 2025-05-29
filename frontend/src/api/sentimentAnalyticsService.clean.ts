import { format, subDays, subHours, subMonths, subWeeks } from 'date-fns';
import clientCache from '../utils/clientCache';

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

// Helper function to generate mock historical data
function generateMockHistoricalData(symbol: string, timeframe: TimeFrame): SentimentDataPoint[] {
  const dataPoints: SentimentDataPoint[] = [];
  const now = new Date();
  let numberOfPoints: number;
  let intervalHours: number;
  
  // Set number of data points and interval based on timeframe
  switch (timeframe) {
    case '24h':
      numberOfPoints = 24;
      intervalHours = 1;
      break;
    case '7d':
      numberOfPoints = 7 * 6; // 6 data points per day
      intervalHours = 4;
      break;
    case '30d':
      numberOfPoints = 30;
      intervalHours = 24; // Daily data
      break;
    case '90d':
      numberOfPoints = 90;
      intervalHours = 24; // Daily data
      break;
    default:
      numberOfPoints = 24;
      intervalHours = 1;
  }
  
  // Generate seed for deterministic but varied results per symbol
  const seed = symbol.charCodeAt(0) + symbol.charCodeAt(symbol.length - 1);
  
  // Generate data points
  for (let i = 0; i < numberOfPoints; i++) {
    const timestamp = subHours(now, intervalHours * i);
    
    // Base sentiment and trends that vary by symbol
    let baseSentiment = 0;
    const timePosition = i / numberOfPoints; // 0 to 1 position in time series
    
    // Different trends for different symbols for variety
    switch (symbol) {
      case 'BTC':
        // Upward trend with volatility
        baseSentiment = 0.3 + (timePosition * 0.3) + (Math.sin(timePosition * Math.PI * 4) * 0.2);
        break;
      case 'ETH':
        // Moderately bullish with recent increase
        baseSentiment = 0.2 + (Math.pow(timePosition, 2) * 0.4);
        break;
      case 'XRP':
        // Declining trend
        baseSentiment = 0.1 - (timePosition * 0.5) + (Math.sin(timePosition * Math.PI * 3) * 0.15);
        break;
      case 'SOL':
        // Volatile upward
        baseSentiment = 0.25 + (timePosition * 0.2) + (Math.sin(timePosition * Math.PI * 6) * 0.3);
        break;
      case 'ADA':
        // Slight uptrend with volatility
        baseSentiment = 0.1 + (timePosition * 0.1) + (Math.sin(timePosition * Math.PI * 5) * 0.2);
        break;
      default:
        // Random walk pattern for other symbols
        baseSentiment = 0.2 + ((Math.sin(timePosition * Math.PI * (seed % 5 + 2)) * 0.3));
    }
    
    // Add some noise to the base sentiment
    const sentiment = Math.max(-1, Math.min(1, baseSentiment + (Math.random() * 0.2 - 0.1)));
    
    // Generate the data point
    dataPoints.push({
      timestamp: format(timestamp, 'yyyy-MM-dd HH:mm:ss'),
      sentiment_score: sentiment,
      confidence: 0.6 + Math.random() * 0.4, // Random confidence between 0.6 and 1.0
      volume: Math.floor(Math.random() * 1000) + 100
    });
  }
  
  return dataPoints;
}

// Mock sentiment analytics service with client-side caching
const sentimentAnalyticsService = {
  // Get historical sentiment data for a specific symbol
  getHistoricalSentimentData: async (
    agentId: string,
    symbol: string | null,
    timeframe: TimeFrame
  ): Promise<SentimentHistoricalData> => {
    // Generate cache key
    const cacheKey = getCacheKeys.historicalData(agentId, symbol || 'ALL', timeframe);
    
    // Check cache first
    const cachedData = clientCache.get<SentimentHistoricalData>(cacheKey);
    if (cachedData) {
      console.log(`Using cached historical sentiment data for ${symbol || 'ALL'}, timeframe: ${timeframe}`);
      return cachedData;
    }
    
    console.log(`Fetching fresh historical sentiment data for ${symbol || 'ALL'}, timeframe: ${timeframe}`);
    await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API delay
    
    // Use the helper to generate mock data
    const dataPoints = generateMockHistoricalData(symbol || 'ALL', timeframe);
    
    const result: SentimentHistoricalData = {
      agent_id: agentId,
      symbol: symbol || 'ALL',
      timeframe,
      dataPoints,
      metadata: {
        total_records: dataPoints.length,
        average_sentiment: dataPoints.reduce((sum, point) => sum + point.sentiment_score, 0) / dataPoints.length,
        average_confidence: dataPoints.reduce((sum, point) => sum + point.confidence, 0) / dataPoints.length,
        updated_at: new Date().toISOString()
      }
    };
    
    // Store in cache with appropriate expiry based on timeframe
    const expiry = timeframe === '24h' ? CACHE_EXPIRY.SHORT : 
                  timeframe === '7d' ? CACHE_EXPIRY.MEDIUM : 
                  CACHE_EXPIRY.LONG;
    
    clientCache.set(cacheKey, result, expiry);
    return result;
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
    await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API delay
    
    // Mock symbols
    const symbols = ['BTC', 'ETH', 'XRP', 'SOL', 'ADA', 'DOT'];
    const result: Record<string, SentimentDataPoint[]> = {};
    
    for (const symbol of symbols) {
      result[symbol] = generateMockHistoricalData(symbol, timeframe);
    }
    
    // Store in cache with expiry
    clientCache.set(cacheKey, result, CACHE_EXPIRY.MEDIUM);
    return result;
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
    await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API delay
    
    // Mock data
    const result = [
      { symbol: 'BTC', latest_sentiment: 0.42, sentiment_change: 0.05, confidence: 0.78 },
      { symbol: 'ETH', latest_sentiment: 0.28, sentiment_change: -0.02, confidence: 0.72 },
      { symbol: 'XRP', latest_sentiment: -0.18, sentiment_change: -0.12, confidence: 0.62 },
      { symbol: 'SOL', latest_sentiment: 0.35, sentiment_change: 0.08, confidence: 0.68 },
      { symbol: 'ADA', latest_sentiment: 0.12, sentiment_change: 0.01, confidence: 0.66 },
      { symbol: 'DOT', latest_sentiment: -0.05, sentiment_change: -0.04, confidence: 0.58 }
    ];
    
    // Store in cache with expiry
    clientCache.set(cacheKey, result, CACHE_EXPIRY.SHORT);
    return result;
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
    await new Promise(resolve => setTimeout(resolve, 700)); // Simulate API delay
    
    // Mock data
    const result: SignalQualityMetrics = {
      timeframe,
      overall_accuracy: 0.68,
      overall_precision: 0.72,
      overall_recall: 0.65,
      overall_f1_score: 0.68,
      total_signals: 148,
      total_successful_signals: 96,
      signal_distribution: {
        positive: 82,
        negative: 42,
        neutral: 24
      },
      performance_by_symbol: {
        'BTC': { accuracy: 0.84, signal_count: 35, avg_confidence: 0.78 },
        'ETH': { accuracy: 0.76, signal_count: 32, avg_confidence: 0.72 },
        'XRP': { accuracy: 0.62, signal_count: 24, avg_confidence: 0.65 },
        'SOL': { accuracy: 0.68, signal_count: 22, avg_confidence: 0.69 },
        'ADA': { accuracy: 0.58, signal_count: 12, avg_confidence: 0.60 },
        'DOT': { accuracy: 0.73, signal_count: 23, avg_confidence: 0.62 }
      }
    };
    
    // Store in cache with expiry
    clientCache.set(cacheKey, result, CACHE_EXPIRY.DAILY);
    return result;
  }
};

export default sentimentAnalyticsService;
