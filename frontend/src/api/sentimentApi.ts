import { SentimentDataPoint } from '../components/sentiment/SentimentTrendChart';
import { IntegratedSignalData } from '../components/sentiment/IntegratedSignalChart';
import { SignalData, SignalType, SignalSource } from '../types/signals';

/**
 * Mock API functions to simulate fetching sentiment data
 * These will be replaced with actual API calls when the backend is ready
 */

// Generate random sentiment data points
const generateSentimentData = (
  symbol: string, 
  days: number, 
  isTopic: boolean = false
): SentimentDataPoint[] => {
  const now = new Date();
  const data: SentimentDataPoint[] = [];
  
  // Base sentiment based on symbol or topic
  let baseSentiment = 0;
  if (symbol === 'BTC' || symbol === 'ETH' || symbol === 'cryptocurrency' || symbol === 'blockchain') {
    baseSentiment = 0.3; // More positive
  } else if (symbol === 'XRP' || symbol === 'regulation') {
    baseSentiment = -0.2; // More negative
  }
  
  // Generate data points
  for (let i = 0; i < days * 4; i++) {
    // Time: go back 'i' intervals (6 hours each)
    const timestamp = new Date(now.getTime() - (i * 6 * 60 * 60 * 1000));
    
    // Add some randomness to sentiment
    const randomFactor = Math.random() * 0.4 - 0.2; // -0.2 to 0.2
    
    // Add a trend component
    const trendFactor = isTopic ? 
      (Math.sin(i / 10) * 0.2) : // Topics have cyclical trends
      ((days - i) / days * 0.3); // Assets have linear trends
    
    // Calculate sentiment with all factors
    let sentiment = baseSentiment + randomFactor + trendFactor;
    
    // Ensure sentiment is within -1 to 1 range
    sentiment = Math.max(-0.95, Math.min(0.95, sentiment));
    
    // Add confidence (higher for recent data)
    const confidence = Math.max(0.3, 1 - (i / (days * 6)));
    
    // Determine source
    const sourceTypes = isTopic ? 
      ['news', 'social_media', 'blogs'] : 
      ['news', 'social_media', 'financial_reports'];
    const source = sourceTypes[Math.floor(Math.random() * sourceTypes.length)];
    
    data.push({
      timestamp: timestamp.toISOString(),
      sentiment,
      confidence,
      source
    });
  }
  
  return data;
};

// Generate integrated signal data
const generateIntegratedSignals = (days: number): IntegratedSignalData[] => {
  // Mock assets
  const assets = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'SOL'];
  
  return assets.map(symbol => {
    // Base values by asset
    let techBase = 0;
    let sentBase = 0;
    
    // Set different base values for different assets
    if (symbol === 'BTC') {
      techBase = 0.6;
      sentBase = 0.4;
    } else if (symbol === 'ETH') {
      techBase = 0.4;
      sentBase = 0.6;
    } else if (symbol === 'XRP') {
      techBase = -0.3;
      sentBase = -0.2;
    } else if (symbol === 'ADA') {
      techBase = 0.2;
      sentBase = -0.3; // Divergent signals
    } else if (symbol === 'DOT') {
      techBase = -0.4;
      sentBase = 0.1; // Divergent signals
    } else if (symbol === 'SOL') {
      techBase = 0.3;
      sentBase = 0.2;
    }
    
    // Add some randomness
    const techRandom = Math.random() * 0.3 - 0.15;
    const sentRandom = Math.random() * 0.3 - 0.15;
    
    // Calculate signals
    const technicalSignal = techBase + techRandom;
    const sentimentSignal = sentBase + sentRandom;
    
    // Calculate combined signal (weighted average)
    const techWeight = 0.6;
    const sentWeight = 0.4;
    const combinedSignal = (technicalSignal * techWeight) + (sentimentSignal * sentWeight);
    
    // Calculate confidence scores
    const technicalConfidence = Math.random() * 0.3 + 0.6; // 0.6-0.9
    const sentimentConfidence = Math.random() * 0.3 + 0.5; // 0.5-0.8
    const combinedConfidence = (technicalConfidence * techWeight) + (sentimentConfidence * sentWeight);
    
    // Determine signal type
    let signalType = 'neutral';
    if (combinedSignal > 0.5) signalType = 'strong_buy';
    else if (combinedSignal > 0.2) signalType = 'buy';
    else if (combinedSignal < -0.5) signalType = 'strong_sell';
    else if (combinedSignal < -0.2) signalType = 'sell';
    
    return {
      symbol,
      technicalSignal,
      sentimentSignal,
      combinedSignal,
      technicalConfidence,
      sentimentConfidence,
      combinedConfidence,
      signalType
    };
  });
};

// Fetch sentiment data for a symbol or topic
export const fetchSentimentData = async (
  query: string, 
  days: number = 7,
  isTopic: boolean = false
): Promise<SentimentDataPoint[]> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 500));
  
  // Generate mock data
  return generateSentimentData(query, days, isTopic);
};

// Fetch integrated signals
export const fetchIntegratedSignals = async (
  days: number = 7
): Promise<IntegratedSignalData[]> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 700));
  
  // Generate mock data
  return generateIntegratedSignals(days);
};

// Fetch all sentiment data for available topics
export const fetchAllTopicSentiment = async (
  days: number = 7
): Promise<Record<string, SentimentDataPoint[]>> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
  // Available topics
  const topics = [
    'blockchain',
    'cryptocurrency',
    'bitcoin',
    'ethereum',
    'defi',
    'nft',
    'metaverse',
    'regulation',
    'market'
  ];
  
  // Generate data for each topic
  const result: Record<string, SentimentDataPoint[]> = {};
  for (const topic of topics) {
    result[topic] = generateSentimentData(topic, days, true);
  }
  
  return result;
};

// Fetch sentiment and technical correlation data
export const fetchCorrelationData = async (
  symbol: string,
  days: number = 30
): Promise<{
  timestamp: string;
  price: number;
  sentiment: number;
  correlation: number;
}[]> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 600));
  
  const now = new Date();
  const data = [];
  
  // Base price by symbol
  let basePrice = 0;
  if (symbol === 'BTC') basePrice = 40000;
  else if (symbol === 'ETH') basePrice = 2500;
  else if (symbol === 'XRP') basePrice = 1.2;
  else if (symbol === 'ADA') basePrice = 2.8;
  else if (symbol === 'DOT') basePrice = 35;
  else if (symbol === 'SOL') basePrice = 150;
  else basePrice = 100;
  
  // Generate correlation data
  let lastPrice = basePrice;
  let correlation = 0.3; // Start with moderate correlation
  
  for (let i = 0; i < days; i++) {
    // Time: go back 'i' days
    const timestamp = new Date(now.getTime() - (i * 24 * 60 * 60 * 1000));
    
    // Generate sentiment data
    const sentimentPoints = generateSentimentData(symbol, 1);
    const avgSentiment = sentimentPoints.reduce((sum, point) => sum + point.sentiment, 0) / sentimentPoints.length;
    
    // Adjust price based partially on sentiment (to create some correlation)
    const sentimentEffect = avgSentiment * basePrice * 0.03; // 3% max impact
    const randomEffect = (Math.random() - 0.5) * basePrice * 0.05; // 5% random fluctuation
    
    // Calculate new price
    lastPrice = lastPrice + sentimentEffect + randomEffect;
    lastPrice = Math.max(lastPrice, basePrice * 0.5); // Ensure price doesn't go too low
    
    // Calculate correlation (simplified simulation)
    correlation = Math.min(0.95, Math.max(-0.95, correlation + (Math.random() - 0.5) * 0.1));
    
    data.push({
      timestamp: timestamp.toISOString(),
      price: lastPrice,
      sentiment: avgSentiment,
      correlation
    });
  }
  
  return data.reverse(); // Most recent first
};

// Get sentiment trend data for multiple symbols
export const getSentimentTrend = async (
  params: {
    symbols: string[];
    timeRange: string;
  }
): Promise<{
  data: SentimentDataPoint[];
}> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
  // Convert timeRange to days
  let days = 7;
  switch (params.timeRange) {
    case '1d':
      days = 1;
      break;
    case '7d':
      days = 7;
      break;
    case '30d':
      days = 30;
      break;
    case '90d':
      days = 90;
      break;
    default:
      days = 7;
  }
  
  // Generate data for each symbol
  let allData: SentimentDataPoint[] = [];
  for (const symbol of params.symbols) {
    const symbolData = generateSentimentData(symbol, days, false);
    
    // Add symbol to each data point
    const dataWithSymbol = symbolData.map(point => ({
      ...point,
      symbol
    }));
    
    allData = [...allData, ...dataWithSymbol];
  }
  
  return { data: allData };
};

// Get integrated signals (combines technical and sentiment)
export const getIntegratedSignals = async (
  params: {
    symbols: string[];
    timeRange: string;
    includeRawSignals?: boolean;
  }
): Promise<{
  data: SignalData[];
}> => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 700));
  
  // Convert timeRange to days
  let days = 7;
  switch (params.timeRange) {
    case '1d':
      days = 1;
      break;
    case '7d':
      days = 7;
      break;
    case '30d':
      days = 30;
      break;
    case '90d':
      days = 90;
      break;
    default:
      days = 7;
  }
  
  // Get integrated signals from our mock function
  const integratedSignals = generateIntegratedSignals(days);
  
  // Generate dates for signals
  const now = new Date();
  const daysAgo = new Date(now.getTime() - (days * 24 * 60 * 60 * 1000));
  
  // Create signal objects for each integrated signal
  const signals: SignalData[] = [];
  
  // Filter symbols if provided
  const filteredSignals = params.symbols.length > 0 ?
    integratedSignals.filter(s => params.symbols.includes(s.symbol)) :
    integratedSignals;
  
  // Generate signal data
  for (const signal of filteredSignals) {
    // Generate a timestamp between daysAgo and now
    const timestamp = new Date(daysAgo.getTime() + Math.random() * (now.getTime() - daysAgo.getTime()));
    
    // Convert signalType to SignalType
    let type: SignalType;
    switch (signal.signalType) {
      case 'strong_buy':
        type = 'STRONG_BUY';
        break;
      case 'buy':
        type = 'BUY';
        break;
      case 'strong_sell':
        type = 'STRONG_SELL';
        break;
      case 'sell':
        type = 'SELL';
        break;
      default:
        type = 'NEUTRAL';
    }
    
    // Add combined signal
    signals.push({
      id: `combined_${signal.symbol}_${timestamp.getTime()}`,
      timestamp,
      type,
      symbol: signal.symbol,
      strength: Math.abs(signal.combinedSignal),
      confidence: signal.combinedConfidence,
      description: `${type} signal based on combined technical and sentiment analysis`,
      source: 'COMBINED'
    });
    
    // If includeRawSignals is true, add technical and sentiment signals separately
    if (params.includeRawSignals) {
      // Add technical signal
      signals.push({
        id: `technical_${signal.symbol}_${timestamp.getTime()}`,
        timestamp: new Date(timestamp.getTime() - 1000 * 60 * 60), // 1 hour earlier
        type: signal.technicalSignal > 0.2 ? 'BUY' : signal.technicalSignal < -0.2 ? 'SELL' : 'NEUTRAL',
        symbol: signal.symbol,
        strength: Math.abs(signal.technicalSignal),
        confidence: signal.technicalConfidence,
        description: `Signal based on technical indicators`,
        source: 'TECHNICAL'
      });
      
      // Add sentiment signal
      signals.push({
        id: `sentiment_${signal.symbol}_${timestamp.getTime()}`,
        timestamp: new Date(timestamp.getTime() - 2000 * 60 * 60), // 2 hours earlier
        type: signal.sentimentSignal > 0.2 ? 'BUY' : signal.sentimentSignal < -0.2 ? 'SELL' : 'NEUTRAL',
        symbol: signal.symbol,
        strength: Math.abs(signal.sentimentSignal),
        confidence: signal.sentimentConfidence,
        description: `Signal based on sentiment analysis`,
        source: 'SENTIMENT'
      });
    }
  }
  
  // Sort by timestamp (newest first)
  signals.sort((a, b) => {
    const timeA = a.timestamp instanceof Date ? a.timestamp.getTime() : new Date(a.timestamp).getTime();
    const timeB = b.timestamp instanceof Date ? b.timestamp.getTime() : new Date(b.timestamp).getTime();
    return timeB - timeA;
  });
  
  return { data: signals };
};

// Export the API functions as an object
export const sentimentApi = {
  getSentimentTrend,
  getIntegratedSignals,
  fetchSentimentData,
  fetchIntegratedSignals,
  fetchAllTopicSentiment,
  fetchCorrelationData
};

export default sentimentApi;
