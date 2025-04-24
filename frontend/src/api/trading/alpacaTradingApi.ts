import axios from 'axios';
import { TradingMode } from '../../config';
import { Order, OrderRequest, Portfolio, Position } from '../../types';
import { createAuthenticatedClient } from '../client';
import { ApiError, NetworkError } from '../utils/errorHandling';
import { executeTrading } from '../utils/tradingCircuitBreaker';
import { TradingApi } from './index';

// Alpaca API endpoints
const ALPACA_LIVE_API_URL = 'https://api.alpaca.markets';
const ALPACA_PAPER_API_URL = 'https://paper-api.alpaca.markets';
const ALPACA_DATA_API_URL = 'https://data.alpaca.markets';

interface AlpacaConfig {
  apiKey: string;
  apiSecret: string;
  paperTrading: boolean;
}

// Create Alpaca API client
const createAlpacaClient = (tradingMode: TradingMode, config: AlpacaConfig) => {
  // For paper trading mode, always use paper API
  // For live mode, use the configuration setting
  const useRealMoney = tradingMode === 'live' && !config.paperTrading;
  const baseURL = useRealMoney ? ALPACA_LIVE_API_URL : ALPACA_PAPER_API_URL;

  const client = axios.create({
    baseURL,
    headers: {
      'APCA-API-KEY-ID': config.apiKey,
      'APCA-API-SECRET-KEY': config.apiSecret,
    },
  });

  // Create data API client
  const dataClient = axios.create({
    baseURL: ALPACA_DATA_API_URL,
    headers: {
      'APCA-API-KEY-ID': config.apiKey,
      'APCA-API-SECRET-KEY': config.apiSecret,
    },
  });

  return {
    client,
    dataClient,
  };
};

// Convert Alpaca order to our Order type
const convertAlpacaOrder = (alpacaOrder: any): Order => {
  return {
    id: alpacaOrder.id,
    symbol: alpacaOrder.symbol,
    type: alpacaOrder.type,
    side: alpacaOrder.side,
    quantity: parseFloat(alpacaOrder.qty),
    price: parseFloat(alpacaOrder.limit_price || alpacaOrder.filled_avg_price || '0'),
    status: alpacaOrder.status,
    createdAt: new Date(alpacaOrder.created_at),
    updatedAt: new Date(alpacaOrder.updated_at),
    clientOrderId: alpacaOrder.client_order_id,
    timeInForce: alpacaOrder.time_in_force,
    filledQuantity: parseFloat(alpacaOrder.filled_qty || '0'),
  };
};

// Helper function to fetch market price
const fetchMarketPrice = async (symbol: string, client: any): Promise<number | null> => {
  try {
    // Try to get latest trade price
    const response = await client.get(`/v2/stocks/${symbol}/trades/latest`);
    if (response.data && response.data.trade && response.data.trade.p) {
      return parseFloat(response.data.trade.p);
    }

    // Fallback to latest quote
    const quoteResponse = await client.get(`/v2/stocks/${symbol}/quotes/latest`);
    if (quoteResponse.data && quoteResponse.data.quote) {
      const quote = quoteResponse.data.quote;
      // Use mid price
      if (quote.ap && quote.bp) {
        return (parseFloat(quote.ap) + parseFloat(quote.bp)) / 2;
      }
      // Use ask price
      if (quote.ap) {
        return parseFloat(quote.ap);
      }
      // Use bid price
      if (quote.bp) {
        return parseFloat(quote.bp);
      }
    }

    return null;
  } catch (error) {
    console.error(`Error fetching market price for ${symbol}:`, error);
    return null;
  }
};

// Create Alpaca trading API
export const createAlpacaTradingApi = (config: AlpacaConfig): TradingApi => {
  // Create Alpaca client
  const { client, dataClient } = createAlpacaClient('live', config);

  // Create authenticated client for our backend
  const backendClient = createAuthenticatedClient();

  // Helper function to get cached portfolio
  const getCachedPortfolio = async (): Promise<Portfolio | null> => {
    try {
      // Try to get from local storage first
      const cachedData = localStorage.getItem(`Alpaca:portfolio`);
      if (cachedData) {
        const parsed = JSON.parse(cachedData);
        const cacheTime = parsed.timestamp;

        // Check if cache is fresh enough (15 minutes)
        if (Date.now() - cacheTime < 15 * 60 * 1000) {
          return parsed.data;
        }
      }

      return null;
    } catch (e) {
      console.error('Error retrieving cached portfolio:', e);
      return null;
    }
  };

  // Get portfolio information with enhanced fallback
  const getPortfolio = async (): Promise<Portfolio> => {
    return executeTrading<Portfolio>(
      async () => {
        // Fetch account information
        const accountResponse = await client.get('/v2/account');
        const account = accountResponse.data;

        // Fetch positions
        const positionsResponse = await client.get('/v2/positions');
        const positions = positionsResponse.data;

        // Format portfolio data
        const portfolio: Portfolio = {
          cash: parseFloat(account.cash),
          total_value: parseFloat(account.portfolio_value),
          positions: {},
          daily_pnl: parseFloat(account.equity) - parseFloat(account.last_equity),
          margin_multiplier: parseFloat(account.multiplier),
        };

        // Process positions
        positions.forEach((position: any) => {
          portfolio.positions[position.symbol] = {
            symbol: position.symbol,
            quantity: parseFloat(position.qty),
            entry_price: parseFloat(position.avg_entry_price),
            current_price: parseFloat(position.current_price),
            market_value: parseFloat(position.market_value),
            unrealized_pnl: parseFloat(position.unrealized_pl),
            realized_pnl: parseFloat(position.realized_pl || '0'),
          };
        });

        // Cache the portfolio
        try {
          localStorage.setItem(`Alpaca:portfolio`, JSON.stringify({
            data: portfolio,
            timestamp: Date.now()
          }));
        } catch (e) {
          console.error('Failed to cache portfolio:', e);
        }

        return portfolio;
      },
      'Alpaca',
      'GET_PORTFOLIO',
      {
        primaryFallback: async () => {
          console.log('Using primary fallback (backend API) for portfolio');
          const response = await backendClient.get('/portfolio');
          return response.data.portfolio;
        },
        secondaryFallback: async () => {
          console.log('Using secondary fallback (cached with updates) for portfolio');
          // Try to get cached portfolio data
          const cachedPortfolio = await getCachedPortfolio();
          if (!cachedPortfolio) throw new Error('No cached portfolio available');

          // Update prices for positions if possible
          try {
            for (const symbol of Object.keys(cachedPortfolio.positions)) {
              try {
                // Get market price from a market data service or API
                const currentPrice = await fetchMarketPrice(symbol, dataClient);
                if (currentPrice && cachedPortfolio.positions[symbol]) {
                  const position = cachedPortfolio.positions[symbol];
                  position.current_price = currentPrice;
                  position.market_value = position.quantity * currentPrice;
                  position.unrealized_pnl = position.market_value - (position.quantity * position.entry_price);
                }
              } catch (e) {
                console.warn(`Failed to update price for ${symbol}:`, e);
                // Continue with other positions
              }
            }

            // Recalculate total value
            let positionsValue = 0;
            Object.values(cachedPortfolio.positions).forEach(position => {
              positionsValue += position.market_value;
            });
            cachedPortfolio.total_value = cachedPortfolio.cash + positionsValue;

            return cachedPortfolio;
          } catch (e) {
            console.error('Failed to update cached portfolio:', e);
            return cachedPortfolio; // Return original cached data
          }
        },
        cacheRetrieval: getCachedPortfolio,
        validator: (portfolio: Portfolio) => {
          return !!portfolio &&
            typeof portfolio.cash === 'number' &&
            typeof portfolio.total_value === 'number' &&
            !!portfolio.positions;
        }
      }
    );
  };

  return {
    // Account and portfolio methods
    async getPortfolio(): Promise<Portfolio> {
      return getPortfolio();
    },

    async getPositions(): Promise<Record<string, Position>> {
      return executeTrading<Record<string, Position>>(
        async () => {
          const { data: alpacaPositions } = await client.get('/v2/positions');

          // Convert positions to our format
          const positions: Record<string, Position> = {};
          alpacaPositions.forEach((pos: any) => {
            positions[pos.symbol] = {
              symbol: pos.symbol,
              quantity: parseFloat(pos.qty),
              entry_price: parseFloat(pos.avg_entry_price),
              current_price: parseFloat(pos.current_price),
              market_value: parseFloat(pos.market_value),
              unrealized_pnl: parseFloat(pos.unrealized_pl),
              realized_pnl: parseFloat(pos.realized_pl || '0'),
            };
          });

          return positions;
        },
        'Alpaca',
        'GET_POSITIONS',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              const { data } = await backendClient.get('/positions');
              return data.positions;
            } catch {
              return {};
            }
          }
        }
      );
    },

    async getBalance(asset?: string): Promise<number> {
      return executeTrading<number>(
        async () => {
          const { data: account } = await client.get('/v2/account');

          if (asset) {
            // Alpaca doesn't have a direct way to get balance for a specific asset
            // We need to check positions
            const positions = await this.getPositions();
            const position = positions[asset];
            return position ? position.market_value : 0;
          }

          return parseFloat(account.cash);
        },
        'Alpaca',
        'GET_ACCOUNT_INFO',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              const { data } = await backendClient.get('/balance');
              return data.balance;
            } catch {
              return 0;
            }
          }
        }
      );
    },

    // Order management methods
    async createOrder(orderRequest: OrderRequest): Promise<Order> {
      return executeTrading<Order>(
        async () => {
          // Convert our order format to Alpaca format
          const alpacaOrder: Record<string, any> = {
            symbol: orderRequest.symbol,
            qty: orderRequest.quantity.toString(),
            side: orderRequest.side,
            type: orderRequest.order_type,
            time_in_force: 'day',
          };

          // Add price for limit orders
          if (orderRequest.order_type === 'limit' && orderRequest.price) {
            alpacaOrder.limit_price = orderRequest.price.toString();
          }

          // Add stop price for stop orders
          if (orderRequest.order_type === 'stop' && orderRequest.stop_price) {
            alpacaOrder.stop_price = orderRequest.stop_price.toString();
          }

          // Add both prices for stop limit orders
          if (orderRequest.order_type === 'stop_limit') {
            if (orderRequest.stop_price) {
              alpacaOrder.stop_price = orderRequest.stop_price.toString();
            }
            if (orderRequest.price) {
              alpacaOrder.limit_price = orderRequest.price.toString();
            }
          }

          // Send order to Alpaca
          const { data } = await client.post('/v2/orders', alpacaOrder);

          // Convert Alpaca order to our format
          return convertAlpacaOrder(data);
        },
        'Alpaca',
        'CREATE_ORDER',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            const { data } = await backendClient.post('/orders', orderRequest);
            return data.order;
          },
          isRetryable: (error: Error) => {
            // Don't retry order submission errors that might lead to duplicate orders
            if (error instanceof ApiError) {
              // Only retry on network errors or server overload
              return error.status >= 500;
            }
            return error instanceof NetworkError;
          }
        }
      );
    },

    async cancelOrder(orderId: string): Promise<boolean> {
      return executeTrading<boolean>(
        async () => {
          await client.delete(`/v2/orders/${orderId}`);
          return true;
        },
        'Alpaca',
        'CANCEL_ORDER',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              await backendClient.delete(`/orders/${orderId}`);
              return true;
            } catch {
              return false;
            }
          },
          maxRetries: 5 // Try more times for cancel operations since they're important
        }
      );
    },

    async getOrders(status?: string): Promise<Order[]> {
      return executeTrading<Order[]>(
        async () => {
          // Default to open orders
          const params: Record<string, any> = {};

          if (status) {
            params.status = status;
          }

          const { data: alpacaOrders } = await client.get('/v2/orders', { params });

          // Convert to our format
          return alpacaOrders.map(convertAlpacaOrder);
        },
        'Alpaca',
        'GET_ORDERS',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            const { data } = await backendClient.get('/orders');
            return data.orders;
          }
        }
      );
    },

    async getOrder(orderId: string): Promise<Order | null> {
      return executeTrading<Order | null>(
        async () => {
          const { data } = await client.get(`/v2/orders/${orderId}`);
          return convertAlpacaOrder(data);
        },
        'Alpaca',
        'GET_ORDERS',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              const { data } = await backendClient.get(`/orders/${orderId}`);
              return data.order;
            } catch {
              return null;
            }
          }
        }
      );
    },

    // Market data methods
    async getMarketPrice(symbol: string): Promise<number> {
      return executeTrading<number>(
        async () => {
          const { data } = await dataClient.get(`/v2/stocks/${symbol}/quotes/latest`);
          return (parseFloat(data.quote.ap) + parseFloat(data.quote.bp)) / 2;
        },
        'Alpaca',
        'GET_MARKET_DATA',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              const { data } = await backendClient.get(`/market/price/${symbol}`);
              return data.price;
            } catch {
              return 0;
            }
          }
        }
      );
    },

    async getOrderBook(symbol: string, limit: number = 10): Promise<{ bids: any[], asks: any[] }> {
      return executeTrading<{ bids: any[], asks: any[] }>(
        async () => {
          // Alpaca doesn't have a direct orderbook API for most symbols
          // We'll use the quotes API to get the best bid/ask
          const { data } = await dataClient.get(`/v2/stocks/${symbol}/quotes/latest`);

          // Create a simple order book with just the best bid/ask
          const bids = [{ price: parseFloat(data.quote.bp), size: parseFloat(data.quote.bs) }];
          const asks = [{ price: parseFloat(data.quote.ap), size: parseFloat(data.quote.as) }];

          return { bids, asks };
        },
        'Alpaca',
        'GET_MARKET_DATA',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              const { data } = await backendClient.get(`/market/orderbook/${symbol}`);
              return data.orderBook;
            } catch {
              return { bids: [], asks: [] };
            }
          }
        }
      );
    },

    async getTicker(symbol: string): Promise<{ price: number, volume: number, change: number }> {
      return executeTrading<{ price: number, volume: number, change: number }>(
        async () => {
          // Get latest trade
          const { data: tradeData } = await dataClient.get(`/v2/stocks/${symbol}/trades/latest`);

          // Get daily bar for change calculation
          const now = new Date();
          const yesterday = new Date(now);
          yesterday.setDate(yesterday.getDate() - 1);

          const startDate = yesterday.toISOString().split('T')[0];
          const endDate = now.toISOString().split('T')[0];

          const { data: barData } = await dataClient.get(`/v2/stocks/${symbol}/bars`, {
            params: {
              start: startDate,
              end: endDate,
              timeframe: '1D',
            },
          });

          const price = parseFloat(tradeData.trade.p);
          const volume = barData.bars.length > 0 ? parseFloat(barData.bars[0].v) : 0;

          // Calculate change percentage
          let change = 0;
          if (barData.bars.length > 0) {
            const bar = barData.bars[0];
            const openPrice = parseFloat(bar.o);
            change = (price - openPrice) / openPrice;
          }

          return { price, volume, change };
        },
        'Alpaca',
        'GET_MARKET_DATA',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              const { data } = await backendClient.get(`/market/ticker/${symbol}`);
              return data.ticker;
            } catch {
              return { price: 0, volume: 0, change: 0 };
            }
          }
        }
      );
    },

    // Exchange info methods
    async getExchangeInfo(): Promise<any> {
      return executeTrading<any>(
        async () => {
          // Alpaca doesn't have a direct equivalent to exchange info
          // We'll create a basic version with the assets we can trade

          // Get account to check if we're in paper mode
          const { data: account } = await client.get('/v2/account');

          // Get assets
          const { data: assets } = await client.get('/v2/assets', {
            params: { status: 'active' },
          });

          return {
            name: account.account_blocked ? 'Alpaca Paper Trading' : 'Alpaca',
            symbols: assets.map((asset: any) => asset.symbol),
            tradingFees: 0, // Alpaca has zero commission
            withdrawalFees: {},
            serverTime: Date.now(),
          };
        },
        'Alpaca',
        'GET_MARKET_DATA',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              const { data } = await backendClient.get('/market/exchange-info');
              return data.exchangeInfo;
            } catch {
              return {
                name: 'Alpaca',
                symbols: [],
                tradingFees: 0,
                withdrawalFees: {},
                serverTime: Date.now(),
              };
            }
          }
        }
      );
    },

    async getSymbols(): Promise<string[]> {
      return executeTrading<string[]>(
        async () => {
          const { data: assets } = await client.get('/v2/assets', {
            params: { status: 'active' },
          });

          return assets.map((asset: any) => asset.symbol);
        },
        'Alpaca',
        'GET_MARKET_DATA',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              const { data } = await backendClient.get('/market/symbols');
              return data.symbols;
            } catch {
              return [];
            }
          }
        }
      );
    },

    async getAssetInfo(symbol: string): Promise<any> {
      return executeTrading<any>(
        async () => {
          const { data } = await client.get(`/v2/assets/${symbol}`);

          return {
            symbol: data.symbol,
            baseAsset: data.symbol,
            quoteAsset: 'USD',
            minQuantity: data.min_order_size ? parseFloat(data.min_order_size) : 1,
            maxQuantity: data.max_order_size ? parseFloat(data.max_order_size) : 1000000,
            quantityPrecision: 0, // Alpaca uses whole shares for most assets
            pricePrecision: 2,
            minNotional: 1, // Minimum order value in USD
            tickSize: 0.01,
          };
        },
        'Alpaca',
        'GET_MARKET_DATA',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              const { data } = await backendClient.get(`/market/asset-info/${symbol}`);
              return data.assetInfo;
            } catch {
              return {
                symbol,
                baseAsset: symbol,
                quoteAsset: 'USD',
                minQuantity: 1,
                maxQuantity: 1000000,
                quantityPrecision: 0,
                pricePrecision: 2,
                minNotional: 1,
                tickSize: 0.01,
              };
            }
          }
        }
      );
    },
  };
};
