import { AxiosResponse } from 'axios';
import { createAuthenticatedClient } from './client';

// Type definitions for trading signals
export interface SignalModel {
  symbol: string;
  signal_type: string;
  direction: string;
  strength: number;
  confidence: number;
  timeframe: string;
  source: string;
  timestamp: string;
  metadata: Record<string, any>;
}

export interface OrderModel {
  symbol: string;
  side: string;
  order_type: string;
  quantity: number;
  price: number;
  stop_price?: number;
  take_profit_price?: number;
  created_at: string;
}

export interface SignalResponseModel {
  signals: SignalModel[];
  orders: OrderModel[];
  timestamp: string;
  metadata: Record<string, any>;
}

export interface StrategyConfigModel {
  sentiment_threshold?: number;
  window_size?: number;
  sentiment_weight?: number;
  min_confidence?: number;
  enable_regime_detection?: boolean;
  volatility_window?: number;
  trend_window?: number;
  volatility_threshold?: number;
  trend_threshold?: number;
  range_threshold?: number;
  risk_per_trade?: number;
  max_position_size?: number;
  stop_loss_pct?: number;
  take_profit_pct?: number;
  timeframe?: string;
  assets?: string[];
  topics?: string[];
  days_back?: number;
}

export interface AggregatorConfigModel {
  conflict_strategy?: string;
  min_confidence?: number;
  min_strength?: number;
  min_signals?: number;
  max_signal_age_hours?: number;
  enable_regime_detection?: boolean;
  signal_weights?: Record<string, number>;
  timeframe_weights?: Record<string, number>;
  source_weights?: Record<string, number>;
}

export interface PerformanceMetrics {
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_return: number;
  avg_return_per_trade: number;
  num_trades: number;
  [key: string]: any;
}

/**
 * Trading Signals API client
 * 
 * This client provides methods for interacting with the trading signals API endpoints.
 */
export class TradingSignalsApi {
  private static instance: TradingSignalsApi;

  /**
   * Get the singleton instance
   */
  public static getInstance(): TradingSignalsApi {
    if (!TradingSignalsApi.instance) {
      TradingSignalsApi.instance = new TradingSignalsApi();
    }
    return TradingSignalsApi.instance;
  }
  private client = createAuthenticatedClient();
  private baseUrl = '/trading-signals';

  /**
   * Get sentiment-based trading signals
   * 
   * @param config Optional strategy configuration
   * @param symbols Optional list of symbols to get signals for
   * @returns Promise with signal response
   */
  public async getSentimentSignals(
    config?: StrategyConfigModel,
    symbols?: string[]
  ): Promise<SignalResponseModel> {
    try {
      const response: AxiosResponse<SignalResponseModel> = await this.client.post(
        `${this.baseUrl}/sentiment`,
        {
          strategy_config: config,
          symbols: symbols
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error fetching sentiment signals:', error);
      throw error;
    }
  }

  /**
   * Aggregate trading signals from multiple sources
   * 
   * @param aggregatorConfig Optional aggregator configuration
   * @param strategyConfig Optional strategy configuration
   * @param symbols Optional list of symbols to aggregate signals for
   * @returns Promise with aggregated signal response
   */
  public async aggregateSignals(
    aggregatorConfig?: AggregatorConfigModel,
    strategyConfig?: StrategyConfigModel,
    symbols?: string[]
  ): Promise<SignalResponseModel> {
    try {
      const response: AxiosResponse<SignalResponseModel> = await this.client.post(
        `${this.baseUrl}/aggregate`,
        {
          aggregator_config: aggregatorConfig,
          strategy_config: strategyConfig,
          symbols: symbols
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error aggregating signals:', error);
      throw error;
    }
  }

  /**
   * Get performance metrics for a specific strategy
   * 
   * @param strategyType Strategy type (enhanced, trend, divergence, shock)
   * @param daysBack Number of days to look back for performance evaluation
   * @returns Promise with performance metrics
   */
  public async getSignalPerformance(
    strategyType: string = 'enhanced',
    daysBack: number = 30
  ): Promise<PerformanceMetrics> {
    try {
      const response: AxiosResponse<PerformanceMetrics> = await this.client.get(
        `${this.baseUrl}/performance`,
        {
          params: {
            strategy_type: strategyType,
            days_back: daysBack
          }
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error fetching signal performance:', error);
      throw error;
    }
  }
}

// Export the class for direct access to static methods
export default TradingSignalsApi;

// Export an instance for direct import
export const tradingSignalsApi = TradingSignalsApi.getInstance();
