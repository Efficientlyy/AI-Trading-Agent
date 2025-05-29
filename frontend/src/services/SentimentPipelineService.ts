import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface SentimentPipelineStage {
  name: string;
  status: string;
  metrics: Record<string, any>;
  last_updated: string;
  description: string;
}

export interface SentimentPipelineData {
  agent_id: string;
  pipeline_status: string;
  stages: SentimentPipelineStage[];
  global_metrics: Record<string, any>;
  pipeline_updated: string;
  pipeline_latency: number;
}

export interface SentimentSignal {
  symbol: string;
  sentiment_score: number;
  signal_strength: number;
  trend: string;
  source_articles: number;
  confidence: number;
  momentum: number;
  recency_score: number;
  volume_score: number;
  analysis_time: string;
  suggested_action: string;
  action_confidence: number;
  cached_at?: string;
}

export interface SentimentSignalsResponse {
  agent_id: string;
  signals_count: number;
  signals: SentimentSignal[];
}

export interface SymbolsResponse {
  agent_id: string;
  symbols: string[];
  topic_mappings: Record<string, string>;
}

/**
 * Service for fetching sentiment pipeline data from the API
 */
class SentimentPipelineService {
  /**
   * Get detailed pipeline data for a specific sentiment analysis agent
   * @param agentId The ID of the sentiment analysis agent
   * @returns Promise with the pipeline data
   */
  async getPipelineData(agentId: string): Promise<SentimentPipelineData> {
    try {
      const response = await axios.get<SentimentPipelineData>(`${API_BASE_URL}/api/sentiment/pipeline/${agentId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching sentiment pipeline data:', error);
      // Return fallback data in case of error
      return this.getFallbackPipelineData(agentId);
    }
  }

  /**
   * Get the latest sentiment signals from an agent
   * @param agentId The ID of the sentiment analysis agent
   * @param limit Maximum number of signals to return
   * @returns Promise with the signals data
   */
  async getLatestSignals(agentId: string, limit: number = 5): Promise<SentimentSignalsResponse> {
    try {
      const response = await axios.get<SentimentSignalsResponse>(
        `${API_BASE_URL}/api/sentiment/signals/${agentId}?limit=${limit}`
      );
      return response.data;
    } catch (error) {
      console.error('Error fetching sentiment signals:', error);
      return {
        agent_id: agentId,
        signals_count: 0,
        signals: []
      };
    }
  }

  /**
   * Get symbols monitored by the sentiment analysis agent
   * @param agentId The ID of the sentiment analysis agent
   * @returns Promise with the symbols data
   */
  async getMonitoredSymbols(agentId: string): Promise<SymbolsResponse> {
    try {
      const response = await axios.get<SymbolsResponse>(`${API_BASE_URL}/api/sentiment/symbols/${agentId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching monitored symbols:', error);
      return {
        agent_id: agentId,
        symbols: [],
        topic_mappings: {}
      };
    }
  }

  /**
   * Generate fallback pipeline data when the API fails
   * This ensures the UI doesn't break if the backend is unavailable
   */
  private getFallbackPipelineData(agentId: string): SentimentPipelineData {
    return {
      agent_id: agentId,
      pipeline_status: 'unknown',
      stages: [
        {
          name: 'Alpha Vantage Client',
          status: 'offline',
          metrics: {
            status: 'offline',
            apiCalls: 0,
            cacheHits: 0,
            errors: 0
          },
          last_updated: new Date().toISOString(),
          description: 'Fetches news sentiment data from Alpha Vantage API'
        },
        {
          name: 'Sentiment Processor',
          status: 'offline',
          metrics: {
            status: 'offline',
            processedItems: 0,
            avgProcessingTime: 0,
            errors: 0
          },
          last_updated: new Date().toISOString(),
          description: 'Processes raw sentiment data into analyzable format'
        },
        {
          name: 'Signal Generator',
          status: 'offline',
          metrics: {
            status: 'offline',
            signalsGenerated: 0,
            bullishSignals: 0,
            bearishSignals: 0,
            neutralSignals: 0
          },
          last_updated: new Date().toISOString(),
          description: 'Generates trading signals from processed sentiment'
        },
        {
          name: 'Cache Manager',
          status: 'offline',
          metrics: {
            status: 'offline',
            cacheSize: 0,
            hitRatio: 0
          },
          last_updated: new Date().toISOString(),
          description: 'Manages caching of sentiment data and signals'
        }
      ],
      global_metrics: {
        total_signals: 0,
        avg_sentiment_score: 0,
        success_rate: 0,
        error_rate: 0
      },
      pipeline_updated: new Date().toISOString(),
      pipeline_latency: 0
    };
  }
}

export default new SentimentPipelineService();
