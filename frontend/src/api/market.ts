import axios from 'axios';
import { createAuthenticatedClient } from './client';
import { OHLCV } from '../types';

// Enhanced asset interface with additional fields
export interface Asset {
  symbol: string;
  name: string;
  type: string;
  price: number;
  change_24h?: number;
  volume_24h?: number;
  source?: string;
  confidence?: number;
  description?: string;
  color?: string;
  icon?: string;
  volatility?: string;
  market_cap?: number;
  high_24h?: number;
  low_24h?: number;
  current_regime?: string;
  volatility_regime?: string;
}

export interface AssetsResponse {
  assets: Asset[];
}

// New interfaces for historical data
export interface HistoricalBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export enum TimeFrame {
  MINUTE_1 = "1m",
  MINUTE_5 = "5m",
  MINUTE_15 = "15m",
  MINUTE_30 = "30m",
  HOUR_1 = "1h",
  HOUR_4 = "4h",
  DAY_1 = "1d",
  WEEK_1 = "1w",
  MONTH_1 = "1M"
}

export interface HistoricalDataRequest {
  symbol: string;
  timeframe: TimeFrame;
  limit?: number;
  start_date?: string;
  end_date?: string;
}

export interface HistoricalDataResponse {
  symbol: string;
  timeframe: TimeFrame;
  data: HistoricalBar[];
  currency: string;
}

// Technical indicators interface
export interface TechnicalIndicators {
  sma_20?: number;
  sma_50?: number;
  sma_200?: number;
  ema_12?: number;
  ema_26?: number;
  macd?: number;
  macd_signal?: number;
  macd_histogram?: number;
  rsi_14?: number;
  bollinger_upper?: number;
  bollinger_middle?: number;
  bollinger_lower?: number;
  signal_macd?: string;
  signal_rsi?: string;
  signal_ma?: string;
}

// Asset detail response
export interface AssetDetailResponse {
  asset: Asset;
  market_status: string;
  last_updated: string;
  historical_data?: HistoricalBar[];
  related_assets?: Asset[];
  market_sentiment?: number;
  technical_indicators?: TechnicalIndicators;
}

// Market overview response
export interface MarketOverviewResponse {
  timestamp: string;
  market_status: string;
  top_gainers: Asset[];
  top_losers: Asset[];
  most_volatile: Asset[];
  market_sentiment?: number;
  trading_volume?: number;
}

// Timeframe conversion utility to handle string-to-enum conversion safely
export function convertToTimeFrame(timeframe: string): TimeFrame {
  switch (timeframe) {
    case '1m': return TimeFrame.MINUTE_1;
    case '5m': return TimeFrame.MINUTE_5;
    case '15m': return TimeFrame.MINUTE_15;
    case '30m': return TimeFrame.MINUTE_30;
    case '1h': return TimeFrame.HOUR_1;
    case '4h': return TimeFrame.HOUR_4;
    case '1d': return TimeFrame.DAY_1;
    case 'day': return TimeFrame.DAY_1;
    case '1w': return TimeFrame.WEEK_1;
    case '1M': return TimeFrame.MONTH_1;
    default: return TimeFrame.DAY_1; // Default to daily timeframe
  }
}

// Mock data for development if API is not available
const MOCK_ASSETS: Asset[] = [
  {
    symbol: 'BTC/USD',
    name: 'Bitcoin',
    type: 'crypto',
    price: 107000,
    change_24h: 2.5,
    volume_24h: 28000000000,
    color: '#F7931A',
    description: 'The original cryptocurrency and largest by market capitalization',
    volatility: 'high'
  },
  {
    symbol: 'ETH/USD',
    name: 'Ethereum',
    type: 'crypto',
    price: 2500,
    change_24h: 3.2,
    volume_24h: 12000000000,
    color: '#627EEA',
    description: 'Smart contract platform enabling decentralized applications',
    volatility: 'high'
  }
];

export const marketApi = {
  /**
   * Get all assets with their current prices
   * 
   * This endpoint provides real-time price data for all available assets
   * with enhanced metadata and market information.
   */
  getAssets: async (): Promise<{ assets: Asset[] }> => {
    try {
      // Try to get real market data from the backend API
      const client = createAuthenticatedClient();
      const response = await client.get<{ assets: Asset[] }>('/market/assets');
      
      // If we have real data, return it
      if (response.data && response.data.assets && response.data.assets.length > 0) {
        console.log('Successfully retrieved real-time market data:', response.data.assets.length, 'assets');
        return response.data;
      }
    } catch (error) {
      console.warn('Error fetching real market data, falling back to mock data:', error);
    }
    
    // Fallback to mock data if the API call fails
    return {
      assets: MOCK_ASSETS
    };
  },
  
  /**
   * Get detailed information for a specific asset
   * 
   * This endpoint provides comprehensive data about a single asset including
   * current price, market status, historical data, technical indicators,
   * and related assets.
   * 
   * @param symbol Trading symbol (e.g., 'BTC/USD')
   */
  getAssetDetail: async (symbol: string): Promise<AssetDetailResponse | null> => {
    try {
      const client = createAuthenticatedClient();
      const response = await client.get<AssetDetailResponse>(`/market/asset/${symbol}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching asset details for ${symbol}:`, error);
      return null;
    }
  },
  
  /**
   * Get market overview with top gainers, losers, and market sentiment
   * 
   * This endpoint provides a high-level overview of the market status
   * including top performing and underperforming assets.
   */
  getMarketOverview: async (): Promise<MarketOverviewResponse | null> => {
    try {
      const client = createAuthenticatedClient();
      const response = await client.get<MarketOverviewResponse>('/market/overview');
      return response.data;
    } catch (error) {
      console.error('Error fetching market overview:', error);
      return null;
    }
  },
  
  /**
   * Get historical price data for a trading symbol
   * 
   * This endpoint handles string or enum timeframes and converts them appropriately.
   * It attempts to fetch data from the backend API but falls back to generated data
   * if the API is unavailable.
   * 
   * @param request The historical data request parameters
   */
  getHistoricalData: async (request: { 
    symbol: string, 
    timeframe: string | TimeFrame, 
    limit?: number,
    start?: string,
    end?: string 
  }): Promise<{ data: OHLCV[] }> => {
    try {
      // Convert timeframe string to enum if necessary
      const timeframeEnum = typeof request.timeframe === 'string' 
        ? convertToTimeFrame(request.timeframe)
        : request.timeframe;
      
      // Prepare the properly typed request object
      const properRequest: HistoricalDataRequest = {
        symbol: request.symbol,
        timeframe: timeframeEnum,
        limit: request.limit,
        start_date: request.start,
        end_date: request.end
      };
      
      // Attempt to fetch from the real API
      const client = createAuthenticatedClient();
      const response = await client.post<HistoricalDataResponse>('/market/historical', properRequest);
      
      // Convert the API response to the expected OHLCV format
      const ohlcvData: OHLCV[] = response.data.data.map(bar => ({
        timestamp: bar.timestamp,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        volume: bar.volume || 0
      }));
      
      return { data: ohlcvData };
    } catch (error) {
      console.warn('Error fetching historical data from API, falling back to mock data:', error);
      
      // MOCKED OHLCV data for offline/dev use
      // Generates 30 days of fake daily candles for any symbol
      const now = Date.now();
      const dayMs = 24 * 60 * 60 * 1000;
      // Ensure at least some price movement for SMA to work
      let prevClose = 40000 + Math.floor(Math.random() * 10000);
      const data: OHLCV[] = Array.from({ length: 30 }).map((_, i) => {
        // Simulate a realistic walk so SMA is not flat/null
        const drift = Math.floor((Math.random() - 0.5) * 500);
        const open = prevClose + drift;
        const close = open + Math.floor((Math.random() - 0.5) * 1000);
        const high = Math.max(open, close) + Math.floor(Math.random() * 200);
        const low = Math.min(open, close) - Math.floor(Math.random() * 200);
        const volume = 10 + Math.floor(Math.random() * 5);
        prevClose = close;
        return {
          timestamp: new Date(now - (29 - i) * dayMs).toISOString(),
          open,
          high,
          low,
          close,
          volume,
        };
      });
      return { data };  
    }
  },
  
  getSentiment: async (symbol?: string): Promise<{ sentiment: string }> => {
    const client = createAuthenticatedClient();
    const response = await client.get<{ sentiment: string }>('/sentiment', {
      params: { symbol }
    });
    return response.data;
  },
  
  /**
   * Get historical prices for a symbol
   * 
   * This is a compatibility wrapper around getHistoricalData that provides
   * a simpler interface with defaults for common use cases
   * 
   * @param symbol Trading symbol (e.g. 'BTC/USD')
   * @param timeframe Timeframe string (e.g. '1d', '4h')
   * @param limit Maximum number of data points to retrieve
   */
  getHistoricalPrices: async (symbol: string, timeframe: string = '1d', limit: number = 100): Promise<{ data: OHLCV[] }> => {
    // This function uses the convertToTimeFrame utility to handle string timeframes
    return marketApi.getHistoricalData({
      symbol,
      timeframe, // The getHistoricalData method now handles string timeframes
      limit
    });
  },
};
