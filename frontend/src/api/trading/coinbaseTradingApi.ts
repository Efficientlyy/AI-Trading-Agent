import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import crypto from 'crypto';
import { TradingMode } from '../../config';
import { Order, OrderRequest, OrderSide, OrderStatus, OrderType, Portfolio, Position } from '../../types';
import { createAuthenticatedClient } from '../client';
import { executeWithCircuitBreaker } from '../utils/circuitBreakerExecutor';
import { ApiError, NetworkError } from '../utils/errorHandling';
import { TradingApi } from './index';

// Coinbase API endpoints
const COINBASE_API_URL = 'https://api.exchange.coinbase.com';
const COINBASE_SANDBOX_API_URL = 'https://api-public.sandbox.exchange.coinbase.com';

interface CoinbaseConfig {
  apiKey: string;
  apiSecret: string;
  passphrase: string;
}

// Create Coinbase API client
const createCoinbaseClient = (tradingMode: TradingMode, coinbaseConfig: CoinbaseConfig): AxiosInstance => {
  const baseURL = tradingMode === 'live' ? COINBASE_API_URL : COINBASE_SANDBOX_API_URL;

  const client = axios.create({
    baseURL,
    // Add timeouts to prevent hanging requests
    timeout: 30000, // 30 seconds
  });

  // Add request interceptor for authentication
  client.interceptors.request.use((config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const method = config.method?.toUpperCase() || 'GET';
    const path = config.url || '';
    const body = config.data ? JSON.stringify(config.data) : '';

    // Create signature
    const message = timestamp + method + path + body;
    const signature = crypto
      .createHmac('sha256', coinbaseConfig.apiSecret)
      .update(message)
      .digest('base64');

    // Add headers
    // Use the set method of AxiosHeaders
    config.headers.set('CB-ACCESS-KEY', coinbaseConfig.apiKey);
    config.headers.set('CB-ACCESS-SIGN', signature);
    config.headers.set('CB-ACCESS-TIMESTAMP', timestamp);
    config.headers.set('CB-ACCESS-PASSPHRASE', coinbaseConfig.passphrase);

    return config;
  });

  // Add response interceptor for error handling
  client.interceptors.response.use(
    response => response,
    error => {
      // Handle specific Coinbase error codes
      if (axios.isAxiosError(error) && error.response) {
        const { status, data } = error.response;

        // Rate limiting
        if (status === 429) {
          return Promise.reject(new ApiError(
            'Coinbase rate limit exceeded',
            429,
            data,
            true
          ));
        }

        // Server errors
        if (status >= 500) {
          return Promise.reject(new ApiError(
            'Coinbase server error',
            status,
            data,
            true
          ));
        }

        // Authentication errors
        if (status === 401) {
          return Promise.reject(new ApiError(
            'Coinbase authentication failed',
            401,
            data,
            false // Not retryable
          ));
        }
      }

      return Promise.reject(error);
    }
  );

  return client;
};

// Convert Coinbase order to our Order type
const convertCoinbaseOrder = (coinbaseOrder: any): Order => {
  return {
    id: coinbaseOrder.id,
    symbol: coinbaseOrder.product_id,
    type: coinbaseOrder.type.toUpperCase() as OrderType,
    side: coinbaseOrder.side.toUpperCase() as OrderSide,
    quantity: parseFloat(coinbaseOrder.size),
    price: parseFloat(coinbaseOrder.price || coinbaseOrder.executed_value || '0'),
    status: coinbaseOrder.status.toUpperCase() as OrderStatus,
    created_at: coinbaseOrder.created_at,
    createdAt: new Date(coinbaseOrder.created_at),
    updated_at: coinbaseOrder.done_at || coinbaseOrder.created_at,
    updatedAt: new Date(coinbaseOrder.done_at || coinbaseOrder.created_at),
    clientOrderId: coinbaseOrder.client_oid || '',
    timeInForce: coinbaseOrder.time_in_force || 'GTC',
    filledQuantity: parseFloat(coinbaseOrder.filled_size || '0'),
    filled_quantity: parseFloat(coinbaseOrder.filled_size || '0'),
  };
};

// Execute Coinbase API call with circuit breaker pattern
const executeCoinbaseCall = async <T>(
  method: string,
  apiCall: () => Promise<T>,
  config: CoinbaseConfig,
  primaryFallback?: () => Promise<T>,
  secondaryFallback?: () => Promise<T>,
  cacheRetrieval?: () => Promise<T | null>,
  validator?: (result: T) => boolean
): Promise<T> => {
  return executeWithCircuitBreaker(apiCall, {
    exchange: 'Coinbase',
    method,
    primaryFallback,
    secondaryFallback,
    cacheRetrieval,
    validator,
    isRetryable: (error: Error) => {
      // Retry on network errors and server errors (5xx)
      if (error instanceof NetworkError) return true;
      if (error instanceof ApiError && (error as ApiError).isRetryable) return true;
      return false;
    },
    maxRetries: 3,
    initialDelayMs: 1000,
    maxDelayMs: 10000,
    isCritical: method.includes('Order') // Order operations are considered critical
  });
};

// Create Coinbase trading API
export const coinbaseTradingApi = (tradingMode: TradingMode, config: CoinbaseConfig): TradingApi => {
  // Create Coinbase client
  const client = createCoinbaseClient(tradingMode, config);

  // Create authenticated client for our backend
  const backendClient = createAuthenticatedClient();

  // Helper function to get cached portfolio
  const getCachedPortfolio = async (): Promise<Portfolio | null> => {
    try {
      // Try to get from local storage first
      const cachedData = localStorage.getItem(`Coinbase:portfolio`);
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

  // Helper function to fetch market price
  const fetchMarketPrice = async (symbol: string): Promise<number | null> => {
    try {
      const productId = symbol.replace('/', '-'); // Convert "BTC/USD" to "BTC-USD"
      const { data } = await client.get(`/products/${productId}/ticker`);
      if (data && data.price) {
        return parseFloat(data.price);
      }
      return null;
    } catch (error) {
      console.error(`Error fetching market price for ${symbol}:`, error);
      return null;
    }
  };

  // Exchange name for logging
  const EXCHANGE = 'Coinbase';

  return {
    // Account and portfolio methods
    async getPortfolio(): Promise<Portfolio> {
      return executeCoinbaseCall<Portfolio>(
        'getPortfolio',
        async () => {
          // Get accounts (balances)
          const { data: accounts } = await client.get('/accounts');

          // Build portfolio
          const portfolio: Portfolio = {
            cash: 0, // Will be calculated
            total_value: 0, // Will be calculated
            positions: {},
            daily_pnl: 0, // Not directly available from Coinbase API
            margin_multiplier: 1 // Coinbase Pro doesn't offer margin by default
          };

          // Process accounts
          let totalValue = 0;
          let cashValue = 0;

          for (const account of accounts) {
            const currency = account.currency;
            const balance = parseFloat(account.balance);

            // Skip zero balances
            if (balance <= 0) continue;

            // Stablecoins and fiat are considered cash
            const isStablecoinOrFiat = ['USD', 'USDC', 'USDT', 'DAI', 'EUR', 'GBP'].includes(currency);

            if (isStablecoinOrFiat) {
              cashValue += balance;
              totalValue += balance;
            } else {
              try {
                // Get price in USD
                const productId = `${currency}-USD`;
                const { data: ticker } = await client.get(`/products/${productId}/ticker`);
                const price = parseFloat(ticker.price);

                if (price > 0) {
                  const marketValue = balance * price;
                  totalValue += marketValue;

                  // Add to positions
                  portfolio.positions[currency] = {
                    symbol: `${currency}/USD`,
                    quantity: balance,
                    entry_price: 0, // Not available from Coinbase API
                    current_price: price,
                    market_value: marketValue,
                    unrealized_pnl: 0, // Not available without order history
                    realized_pnl: 0 // Not available without order history
                  };
                }
              } catch (error) {
                console.warn(`Could not get price for ${currency}:`, error);
                // Skip this asset if we can't get a price
              }
            }
          }

          portfolio.cash = cashValue;
          portfolio.total_value = totalValue;

          // Cache the portfolio for future fallbacks
          localStorage.setItem('Coinbase:portfolio', JSON.stringify({
            timestamp: Date.now(),
            data: portfolio
          }));

          return portfolio;
        },
        config,
        // Primary fallback - backend API
        async () => {
          console.log('Using primary fallback (backend API) for Coinbase portfolio');
          const { data } = await backendClient.get('/portfolio');
          return data.portfolio;
        },
        // Secondary fallback - cached portfolio with price updates
        async () => {
          console.log('Using secondary fallback (cached with updates) for Coinbase portfolio');
          // Try to get cached portfolio data
          const cachedPortfolio = await getCachedPortfolio();
          if (!cachedPortfolio) throw new Error('No cached portfolio available');

          // Update prices for positions if possible
          try {
            for (const symbol of Object.keys(cachedPortfolio.positions)) {
              try {
                const currentPrice = await fetchMarketPrice(`${symbol}/USD`);
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
        // Cache retrieval function
        getCachedPortfolio,
        // Validate portfolio data
        (portfolio: Portfolio) => {
          return !!portfolio &&
            typeof portfolio.cash === 'number' &&
            typeof portfolio.total_value === 'number' &&
            !!portfolio.positions;
        }
      );
    },

    async getPositions(): Promise<Record<string, Position>> {
      return executeCoinbaseCall(
        'getPositions',
        async () => {
          const portfolio = await this.getPortfolio();
          return portfolio.positions || {};
        },
        config,
        async () => ({})
      );
    },

    async getBalance(asset?: string): Promise<number> {
      return executeCoinbaseCall(
        'getBalance',
        async () => {
          const { data: accounts } = await client.get('/accounts');

          if (asset) {
            // Convert asset format from "BTC/USD" to "BTC"
            const currency = asset.split('/')[0];
            const account = accounts.find((a: any) => a.currency === currency);
            return account ? parseFloat(account.balance) : 0;
          } else {
            // Return USD balance as default
            const usdAccount = accounts.find((a: any) => a.currency === 'USD');
            return usdAccount ? parseFloat(usdAccount.balance) : 0;
          }
        },
        config,
        async () => 0
      );
    },

    // Order management methods
    async createOrder(orderRequest: OrderRequest): Promise<Order> {
      return executeCoinbaseCall(
        'createOrder',
        async () => {
          // Convert our order format to Coinbase format
          const productId = orderRequest.symbol.replace('/', '-'); // Convert "BTC/USD" to "BTC-USD"

          // Base parameters for all order types
          const params: Record<string, any> = {
            product_id: productId,
            side: orderRequest.side.toLowerCase(),
            size: orderRequest.quantity.toString(),
          };

          // Add parameters based on order type
          if (orderRequest.order_type === 'limit' && orderRequest.price) {
            params.type = 'limit';
            params.price = orderRequest.price.toString();
            params.time_in_force = 'GTC'; // Good Till Cancelled
          } else {
            params.type = 'market';
          }

          // Send order to Coinbase
          const { data } = await client.post('/orders', params);

          // Convert Coinbase order to our format
          return convertCoinbaseOrder(data);
        },
        config,
        async () => {
          // Fallback to our backend
          const { data } = await backendClient.post('/orders', orderRequest);
          return data.order;
        }
      );
    },

    async cancelOrder(orderId: string): Promise<boolean> {
      return executeCoinbaseCall(
        'cancelOrder',
        async () => {
          await client.delete(`/orders/${orderId}`);
          return true;
        },
        config,
        async () => {
          // Fallback to our backend
          try {
            await backendClient.delete(`/orders/${orderId}`);
            return true;
          } catch {
            return false;
          }
        }
      );
    },

    async getOrders(status?: string): Promise<Order[]> {
      return executeCoinbaseCall(
        'getOrders',
        async () => {
          // Get orders from Coinbase
          const { data } = await client.get('/orders');

          // Convert to our format
          const orders = data.map(convertCoinbaseOrder);

          // Filter by status if provided
          if (status) {
            return orders.filter((order: Order) => order.status === status);
          }

          return orders;
        },
        config,
        async () => {
          // Fallback to our backend
          const { data } = await backendClient.get('/orders');
          return data.orders;
        }
      );
    },

    async getOrder(orderId: string): Promise<Order | null> {
      return executeCoinbaseCall(
        'getOrder',
        async () => {
          const { data } = await client.get(`/orders/${orderId}`);
          return convertCoinbaseOrder(data);
        },
        config,
        async () => {
          // Fallback to our backend
          try {
            const { data } = await backendClient.get(`/orders/${orderId}`);
            return data.order;
          } catch {
            return null;
          }
        }
      );
    },

    // Market data methods
    async getMarketPrice(symbol: string): Promise<number> {
      return executeCoinbaseCall(
        'getMarketPrice',
        async () => {
          const productId = symbol.replace('/', '-'); // Convert "BTC/USD" to "BTC-USD"
          const { data } = await client.get(`/products/${productId}/ticker`);
          return parseFloat(data.price);
        },
        config,
        async () => {
          // Fallback to our backend
          try {
            const { data } = await backendClient.get(`/market/price/${symbol}`);
            return data.price;
          } catch {
            return 0;
          }
        }
      );
    },

    async getOrderBook(symbol: string, limit: number = 10): Promise<{ bids: any[], asks: any[] }> {
      return executeCoinbaseCall(
        'getOrderBook',
        async () => {
          const productId = symbol.replace('/', '-'); // Convert "BTC/USD" to "BTC-USD"
          const { data } = await client.get(`/products/${productId}/book`, {
            params: { level: 2 }, // Level 2 provides the top 50 bids and asks
          });

          // Convert to our format
          const bids = data.bids.slice(0, limit).map((bid: any) => ({
            price: parseFloat(bid[0]),
            size: parseFloat(bid[1]),
          }));

          const asks = data.asks.slice(0, limit).map((ask: any) => ({
            price: parseFloat(ask[0]),
            size: parseFloat(ask[1]),
          }));

          return { bids, asks };
        },
        config,
        async () => {
          // Fallback to our backend
          try {
            const { data } = await backendClient.get(`/market/orderbook/${symbol}`);
            return data.orderBook;
          } catch {
            return { bids: [], asks: [] };
          }
        }
      );
    },

    async getTicker(symbol: string): Promise<{ price: number, volume: number, change: number }> {
      return executeCoinbaseCall(
        'getTicker',
        async () => {
          const productId = symbol.replace('/', '-'); // Convert "BTC/USD" to "BTC-USD"

          // Get ticker
          const { data: ticker } = await client.get(`/products/${productId}/ticker`);

          // Get 24h stats
          const { data: stats } = await client.get(`/products/${productId}/stats`);

          const price = parseFloat(ticker.price);
          const volume = parseFloat(stats.volume);

          // Calculate change percentage
          const open = parseFloat(stats.open);
          const change = open > 0 ? (price - open) / open : 0;

          return { price, volume, change };
        },
        config,
        async () => {
          // Fallback to our backend
          try {
            const { data } = await backendClient.get(`/market/ticker/${symbol}`);
            return data.ticker;
          } catch {
            return { price: 0, volume: 0, change: 0 };
          }
        }
      );
    },

    // Exchange info methods
    async getExchangeInfo(): Promise<any> {
      return executeCoinbaseCall(
        'getExchangeInfo',
        async () => {
          // Get products
          const { data: products } = await client.get('/products');

          // Convert to our format
          return {
            name: 'Coinbase',
            symbols: products.map((p: any) => `${p.base_currency}/${p.quote_currency}`),
            tradingFees: 0.005, // Default Coinbase fee (0.5%)
            withdrawalFees: {},
            serverTime: Date.now(),
          };
        },
        config,
        async () => {
          // Fallback to our backend
          try {
            const { data } = await backendClient.get('/market/exchange-info');
            return data.exchangeInfo;
          } catch {
            return {
              name: 'Coinbase',
              symbols: [],
              tradingFees: 0.005,
              withdrawalFees: {},
              serverTime: Date.now(),
            };
          }
        }
      );
    },

    async getSymbols(): Promise<string[]> {
      return executeCoinbaseCall(
        'getSymbols',
        async () => {
          const { data: products } = await client.get('/products');
          return products.map((p: any) => `${p.base_currency}/${p.quote_currency}`);
        },
        config,
        async () => {
          // Fallback to our backend
          try {
            const { data } = await backendClient.get('/market/symbols');
            return data.symbols;
          } catch {
            return [];
          }
        }
      );
    },

    async getAssetInfo(symbol: string): Promise<any> {
      return executeCoinbaseCall(
        'getAssetInfo',
        async () => {
          const productId = symbol.replace('/', '-'); // Convert "BTC/USD" to "BTC-USD"
          const { data } = await client.get(`/products/${productId}`);

          return {
            symbol,
            baseAsset: data.base_currency,
            quoteAsset: data.quote_currency,
            minQuantity: parseFloat(data.base_min_size),
            maxQuantity: parseFloat(data.base_max_size),
            quantityPrecision: data.base_increment.split('.')[1]?.length || 0,
            pricePrecision: data.quote_increment.split('.')[1]?.length || 0,
            minNotional: parseFloat(data.min_market_funds),
            tickSize: parseFloat(data.quote_increment),
          };
        },
        config,
        async () => {
          // Fallback to our backend
          try {
            const { data } = await backendClient.get(`/market/asset-info/${symbol}`);
            return data.assetInfo;
          } catch {
            return {
              symbol,
              baseAsset: symbol.split('/')[0],
              quoteAsset: symbol.split('/')[1],
              minQuantity: 0.0001,
              maxQuantity: 1000,
              quantityPrecision: 8,
              pricePrecision: 2,
              minNotional: 1,
              tickSize: 0.01,
            };
          }
        }
      );
    },
  };
};
