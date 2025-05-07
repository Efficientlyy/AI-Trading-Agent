import { createAuthenticatedClient } from './client';

export interface AlphaVantageSentimentResponse {
  sentiment_score: number;
  sentiment_label: string;
  relevance_score: number;
  topics: string[];
  ticker_sentiment: {
    ticker: string;
    relevance_score: number;
    ticker_sentiment_score: number;
    ticker_sentiment_label: string;
  }[];
  title: string;
  url: string;
  time_published: string;
  authors: string[];
  summary: string;
  overall_sentiment_score: number;
  overall_sentiment_label: string;
}

export interface AlphaVantageSentimentData {
  items: AlphaVantageSentimentResponse[];
  metadata: {
    total_count: number;
    topic: string;
    time_from: string;
    time_to: string;
  };
}

export interface SentimentSummary {
  topic: string;
  average_sentiment: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  total_count: number;
  top_tickers: {
    ticker: string;
    sentiment_score: number;
    relevance_score: number;
  }[];
  time_from: string;
  time_to: string;
}

export const alphaVantageApi = {
  // Get sentiment data by topic
  getSentimentByTopic: async (topic: string = 'blockchain'): Promise<SentimentSummary> => {
    try {
      const client = createAuthenticatedClient();
      const response = await client.get<AlphaVantageSentimentData>('/api/alpha-vantage/sentiment', {
        params: { topic }
      });
      
      // Process the response to create a summary
      const data = response.data;
      const items = data.items || [];
      
      // Calculate sentiment statistics
      let totalSentiment = 0;
      let positiveCount = 0;
      let negativeCount = 0;
      let neutralCount = 0;
      
      // Process all sentiment items
      items.forEach(item => {
        totalSentiment += item.overall_sentiment_score;
        
        if (item.overall_sentiment_label === 'Bullish') {
          positiveCount++;
        } else if (item.overall_sentiment_label === 'Bearish') {
          negativeCount++;
        } else {
          neutralCount++;
        }
      });
      
      // Collect ticker sentiment data
      const tickerMap = new Map<string, { 
        sentiment_total: number, 
        relevance_total: number, 
        count: number 
      }>();
      
      items.forEach(item => {
        item.ticker_sentiment.forEach(ticker => {
          if (!tickerMap.has(ticker.ticker)) {
            tickerMap.set(ticker.ticker, {
              sentiment_total: 0,
              relevance_total: 0,
              count: 0
            });
          }
          
          const current = tickerMap.get(ticker.ticker)!;
          current.sentiment_total += ticker.ticker_sentiment_score;
          current.relevance_total += ticker.relevance_score;
          current.count++;
        });
      });
      
      // Create top tickers list
      const topTickers = Array.from(tickerMap.entries())
        .map(([ticker, data]) => ({
          ticker,
          sentiment_score: data.sentiment_total / data.count,
          relevance_score: data.relevance_total / data.count
        }))
        .sort((a, b) => b.relevance_score - a.relevance_score)
        .slice(0, 10);
      
      return {
        topic: data.metadata.topic,
        average_sentiment: items.length > 0 ? totalSentiment / items.length : 0,
        positive_count: positiveCount,
        negative_count: negativeCount,
        neutral_count: neutralCount,
        total_count: items.length,
        top_tickers: topTickers,
        time_from: data.metadata.time_from,
        time_to: data.metadata.time_to
      };
    } catch (error) {
      console.error('Error fetching Alpha Vantage sentiment data:', error);
      
      // Return mock data for development
      return {
        topic: topic,
        average_sentiment: 0.35,
        positive_count: 15,
        negative_count: 5,
        neutral_count: 10,
        total_count: 30,
        top_tickers: [
          { ticker: 'BTC', sentiment_score: 0.65, relevance_score: 0.85 },
          { ticker: 'ETH', sentiment_score: 0.58, relevance_score: 0.82 },
          { ticker: 'SOL', sentiment_score: 0.42, relevance_score: 0.75 },
          { ticker: 'ADA', sentiment_score: 0.38, relevance_score: 0.70 },
          { ticker: 'DOT', sentiment_score: 0.32, relevance_score: 0.65 }
        ],
        time_from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        time_to: new Date().toISOString()
      };
    }
  },
  
  // Convert Alpha Vantage sentiment data to trading signals
  convertSentimentToSignals: (sentimentData: SentimentSummary) => {
    // Create trading signals from top tickers
    return sentimentData.top_tickers.map(ticker => {
      // Determine signal type based on sentiment score
      let type = 'NEUTRAL';
      if (ticker.sentiment_score > 0.6) {
        type = 'STRONG_BUY';
      } else if (ticker.sentiment_score > 0.3) {
        type = 'BUY';
      } else if (ticker.sentiment_score < -0.6) {
        type = 'STRONG_SELL';
      } else if (ticker.sentiment_score < -0.3) {
        type = 'SELL';
      }
      
      return {
        timestamp: new Date(),
        type,
        strength: Math.abs(ticker.sentiment_score),
        source: 'Alpha Vantage',
        symbol: ticker.ticker,
        confidence: ticker.relevance_score
      };
    });
  }
};

export default alphaVantageApi;
