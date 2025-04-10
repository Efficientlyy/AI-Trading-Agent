import { useState, useEffect, useCallback } from 'react';
import { OHLCV } from '../types';
import { getMockHistoricalData } from '../api/mockData/historicalData';

interface UseHistoricalDataParams {
  symbol: string;
  timeframe?: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w';
  limit?: number;
}

interface UseHistoricalDataResult {
  data: OHLCV[] | null;
  isLoading: boolean;
  error: Error | null;
  changeTimeframe: (timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w') => void;
  refresh: () => void;
}

/**
 * Custom hook to fetch and manage historical price data
 * @param params Configuration parameters for historical data
 * @returns Historical data state and control functions
 */
const useHistoricalData = (params: UseHistoricalDataParams): UseHistoricalDataResult => {
  const { symbol, timeframe = '1d', limit = 200 } = params;
  
  const [data, setData] = useState<OHLCV[] | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const [currentTimeframe, setCurrentTimeframe] = useState<string>(timeframe);
  
  // Function to fetch historical data
  const fetchData = useCallback(async () => {
    if (!symbol) {
      setData(null);
      setIsLoading(false);
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would be an API call to a backend service
      // For now, we're using mock data
      const result = await getMockHistoricalData(symbol, currentTimeframe as any);
      setData(result);
    } catch (err) {
      console.error('Error fetching historical data:', err);
      setError(err instanceof Error ? err : new Error('Failed to fetch historical data'));
      setData(null);
    } finally {
      setIsLoading(false);
    }
  }, [symbol, currentTimeframe]);
  
  // Fetch data when symbol or timeframe changes
  useEffect(() => {
    fetchData();
  }, [fetchData]);
  
  // Function to change the timeframe
  const changeTimeframe = useCallback((newTimeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w') => {
    setCurrentTimeframe(newTimeframe);
  }, []);
  
  // Function to manually refresh the data
  const refresh = useCallback(() => {
    fetchData();
  }, [fetchData]);
  
  return {
    data,
    isLoading,
    error,
    changeTimeframe,
    refresh
  };
};

export default useHistoricalData;
