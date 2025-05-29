// Import React hooks properly
import { useState, useEffect, useRef } from 'react';
import mexcService from '../api/mexcService';

// Export all hooks from this index file
export * from './useMexcData';

// Export the useMexcTickers hook and its types
export interface TickerItem {
  symbol: string;
  price: number;
  change: number;
  volume: number;
}

type NodeJSTimeout = ReturnType<typeof setTimeout>;

export function useMexcTickers() {
  // This constant should match the one in useMexcData.ts
  const USE_MOCK_DATA = true;
  
  const [tickers, setTickers] = useState<TickerItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  // Generate mock ticker data to ensure consistent experience
  const generateMockTickers = (): TickerItem[] => {
    return [
      { symbol: 'BTC/USDT', price: 60245.75 + (Math.random() * 200 - 100), change: 2.3 + (Math.random() - 0.5), volume: 1245789023 },
      { symbol: 'ETH/USDT', price: 3420.50 + (Math.random() * 20 - 10), change: 1.8 + (Math.random() - 0.5), volume: 545672301 },
      { symbol: 'SOL/USDT', price: 154.25 + (Math.random() * 5 - 2.5), change: 5.2 + (Math.random() - 0.5), volume: 324567890 },
      { symbol: 'ADA/USDT', price: 0.85 + (Math.random() * 0.02 - 0.01), change: -0.7 + (Math.random() - 0.5), volume: 98765432 },
      { symbol: 'XRP/USDT', price: 0.52 + (Math.random() * 0.01 - 0.005), change: -1.2 + (Math.random() - 0.5), volume: 76543210 },
      { symbol: 'DOGE/USDT', price: 0.12 + (Math.random() * 0.005 - 0.0025), change: 3.5 + (Math.random() - 0.5), volume: 43210987 },
      { symbol: 'DOT/USDT', price: 7.45 + (Math.random() * 0.2 - 0.1), change: 0.5 + (Math.random() - 0.5), volume: 32109876 },
      { symbol: 'AVAX/USDT', price: 38.90 + (Math.random() * 1 - 0.5), change: 4.1 + (Math.random() - 0.5), volume: 21098765 }
    ];
  };

  // Mock data update interval
  const mockDataIntervalRef = useRef<NodeJSTimeout | null>(null);

  useEffect(() => {
    let isMounted = true;
    let interval: NodeJSTimeout | null = null;
    
    const fetchTickers = async () => {
      try {
        setLoading(true);
        
        if (USE_MOCK_DATA) {
          // Use mock data for consistent performance
          const mockTickers = generateMockTickers();
          
          if (isMounted) {
            setTickers(mockTickers);
            setLoading(false);
            setError(null);
          }
          
          // Set up a mock update interval
          if (mockDataIntervalRef.current) {
            clearInterval(mockDataIntervalRef.current);
          }
          
          mockDataIntervalRef.current = setInterval(() => {
            if (isMounted) {
              setTickers(generateMockTickers());
            }
          }, 5000); // Update every 5 seconds with slightly different values
        } else {
          // Use real API data
          const response = await mexcService.getTickers();
          
          // Filter out just the tickers we're interested in
          const supportedPairs = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 
            'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT'
          ];
          
          const filteredTickers = Array.isArray(response) 
            ? response.filter((ticker: any) => supportedPairs.includes(ticker.symbol))
            : [response].filter((ticker: any) => supportedPairs.includes(ticker.symbol));
          
          // Convert to our format
          const formattedTickers: TickerItem[] = filteredTickers.map((ticker: any) => ({
            symbol: ticker.symbol.replace(/(\w+)USDT/, '$1/USDT'),
            price: parseFloat(ticker.lastPrice),
            change: parseFloat(ticker.priceChangePercent),
            volume: parseFloat(ticker.quoteVolume)
          }));
          
          if (isMounted) {
            setTickers(formattedTickers);
            setLoading(false);
            setError(null);
          }
        }
      } catch (err) {
        console.error('Error fetching tickers:', err);
        if (isMounted) {
          setError(err instanceof Error ? err : new Error('Unknown error fetching tickers'));
          setLoading(false);
          
          // Provide fallback data on error if not using mock data
          if (!USE_MOCK_DATA) {
            setTickers(generateMockTickers());
          }
        }
      }
    };
    
    // Initial fetch
    fetchTickers();
    
    // Set up periodic refresh - only if not using mock data
    // (mock data has its own refresh interval)
    if (!USE_MOCK_DATA) {
      interval = setInterval(fetchTickers, 30000); // Refresh every 30 seconds
    }
    
    return () => {
      isMounted = false;
      if (interval) clearInterval(interval);
      if (mockDataIntervalRef.current) {
        clearInterval(mockDataIntervalRef.current);
        mockDataIntervalRef.current = null;
      }
    };
  }, []);

  return { tickers, loading, error };
}
