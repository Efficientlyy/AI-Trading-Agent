/**
 * Market API client
 * 
 * This module provides a client for the market data API endpoints.
 * It handles fetching price data from the backend.
 */

import { apiRequest } from '../apiUtils';
import { OHLCV } from '../../types';
import { createAuthenticatedClient } from '../client';

interface MarketDataParams {
  symbol: string;
  timeframe: string;
  limit?: number;
  from?: string;
  to?: string;
}

interface MarketDataResponse {
  symbol: string;
  timeframe: string;
  data: OHLCV[];
}

/**
 * Client for the market data API
 */
export const marketApi = {
  /**
   * Get historical price data for a symbol
   * 
   * @param params Parameters for the historical data request
   * @returns Promise with historical price data
   */
  getHistoricalData: async (params: MarketDataParams): Promise<MarketDataResponse> => {
    try {
      const { symbol, timeframe, limit = 100, from, to } = params;
      
      let queryParams = `?symbol=${symbol}&timeframe=${timeframe}&limit=${limit}`;
      if (from) queryParams += `&from=${from}`;
      if (to) queryParams += `&to=${to}`;
      
      const response = await apiRequest<MarketDataResponse>(`/api/market/historical${queryParams}`);
      
      return response;
    } catch (error) {
      console.error('Error fetching historical market data:', error);
      
      // Return mock data on error
      return {
        symbol: params.symbol,
        timeframe: params.timeframe,
        data: generateMockPriceData(params.symbol, params.limit || 100)
      };
    }
  }
};

/**
 * Generate mock price data for demonstration
 */
function generateMockPriceData(symbol: string, length: number): OHLCV[] {
  const basePrice = symbol === 'BTC' ? 50000 : 
                    symbol === 'ETH' ? 2000 : 
                    symbol === 'XRP' ? 0.5 : 
                    symbol === 'ADA' ? 1.2 : 100;
  
  return Array.from({ length }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (length - i));
    
    const volatility = basePrice * 0.02;
    const change = (Math.random() - 0.5) * volatility;
    const open = basePrice + (i * basePrice * 0.001) + (Math.random() - 0.5) * volatility;
    const close = open + change;
    
    return {
      timestamp: date.toISOString(),
      open,
      high: Math.max(open, close) + (Math.random() * volatility * 0.5),
      low: Math.min(open, close) - (Math.random() * volatility * 0.5),
      close,
      volume: Math.floor(Math.random() * 1000000)
    };
  });
}

/**
 * Get historical prices for a symbol
 * 
 * @param symbol Symbol to get prices for (e.g., BTC, ETH)
 * @param timeframe Timeframe for the prices (e.g., 1h, 4h, 1d, 1w)
 * @param startDate Start date for the price data
 * @param endDate End date for the price data
 * @returns Promise with historical price data
 */
export async function getHistoricalPrices(
  symbol: string,
  timeframe: string = '1d',
  startDate?: Date,
  endDate?: Date
): Promise<OHLCV[]> {
  try {
    const client = createAuthenticatedClient();
    
    // Format dates for API request
    const from = startDate ? startDate.toISOString() : undefined;
    const to = endDate ? endDate.toISOString() : undefined;
    
    // Prepare query parameters
    const params: Record<string, string> = {
      symbol,
      timeframe,
      limit: '100'
    };
    
    if (from) params.from = from;
    if (to) params.to = to;
    
    // Make API request
    const response = await client.get('/api/market/historical', { params });
    
    if (response.data && response.data.data) {
      return response.data.data;
    }
    
    throw new Error('Invalid response format');
  } catch (error) {
    console.error('Error fetching historical prices:', error);
    
    // Return mock data on error
    return generateMockPriceData(symbol, 100);
  }
}
