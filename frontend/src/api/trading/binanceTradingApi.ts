import axios, { AxiosInstance } from 'axios';
import crypto from 'crypto';
import { TradingMode } from '../../config';
import { Order, OrderRequest, OrderSide, OrderStatus, OrderType, Portfolio, Position } from '../../types';
import { createAuthenticatedClient } from '../client';
import { ApiError, NetworkError } from '../utils/errorHandling';
import { executeTrading } from '../utils/tradingCircuitBreaker';
import { TradingApi } from './index';

// Binance API endpoints
const BINANCE_API_URL = 'https://api.binance.com';
const BINANCE_TEST_API_URL = 'https://testnet.binance.vision'; // Testnet for paper trading

interface BinanceConfig {
  apiKey: string;
  apiSecret: string;
  testnet?: boolean;
}

// Create Binance API client
const createBinanceClient = (tradingMode: TradingMode, config: BinanceConfig): AxiosInstance => {
  // Use testnet if explicitly requested or if in paper trading mode
  const useTestnet = config.testnet || tradingMode === 'paper';
  const baseURL = useTestnet ? BINANCE_TEST_API_URL : BINANCE_API_URL;

  const client = axios.create({
    baseURL,
    headers: {
      'X-MBX-APIKEY': config.apiKey,
    },
    // Add timeouts to prevent hanging requests
    timeout: 30000, // 30 seconds
  });

  // Add response interceptor for error handling
  client.interceptors.response.use(
    response => response,
    error => {
      // Handle specific Binance error codes
      if (axios.isAxiosError(error) && error.response) {
        const { status, data } = error.response;

        // Rate limiting
        if (status === 429 || data?.code === -1003) {
          const retryAfter = error.response.headers['retry-after']
            ? parseInt(error.response.headers['retry-after'], 10) * 1000
            : 60000; // Default to 60 seconds if no header

          return Promise.reject(new ApiError(
            'Binance rate limit exceeded',
            429,
            data,
            true
          ));
        }

        // Server errors
        if (status >= 500) {
          return Promise.reject(new ApiError(
            'Binance server error',
            status,
            data,
            true
          ));
        }

        // Authentication errors
        if (status === 401 || data?.code === -2015) {
          return Promise.reject(new ApiError(
            'Binance authentication failed',
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

// Generate signature for authenticated requests
const generateSignature = (queryString: string, config: BinanceConfig): string => {
  return crypto
    .createHmac('sha256', config.apiSecret)
    .update(queryString)
    .digest('hex');
};

// Create Binance trading API
export const binanceTradingApi = (tradingMode: TradingMode, config: BinanceConfig): TradingApi => {
  // Create Binance client
  const client = createBinanceClient(tradingMode, config);

  // Create authenticated client for our backend
  const backendClient = createAuthenticatedClient();

  // Helper function to get cached portfolio
  const getCachedPortfolio = async (): Promise<Portfolio | null> => {
    try {
      // Try to get from local storage first
      const cachedData = localStorage.getItem(`Binance:portfolio`);
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
      const { data } = await client.get(`/api/v3/ticker/price?symbol=${symbol}USDT`);
      if (data && data.price) {
        return parseFloat(data.price);
      }
      return null;
    } catch (error) {
      console.error(`Error fetching market price for ${symbol}:`, error);
      return null;
    }
  };

  return {
    // Account and portfolio methods
    async getPortfolio(): Promise<Portfolio> {
      return executeTrading<Portfolio>(
        async () => {
          // Get account information
          const timestamp = Date.now();
          const queryString = `timestamp=${timestamp}`;
          const signature = generateSignature(queryString, config);

          const { data: accountInfo } = await client.get(
            `/api/v3/account?${queryString}&signature=${signature}`
          );

          // Get ticker prices for all assets
          const { data: prices } = await client.get('/api/v3/ticker/price');
          const priceMap: Record<string, number> = {};

          prices.forEach((ticker: any) => {
            priceMap[ticker.symbol] = parseFloat(ticker.price);
          });

          // Build portfolio
          const portfolio: Portfolio = {
            cash: 0, // Will be calculated
            total_value: 0, // Will be calculated
            positions: {},
            daily_pnl: 0, // Not directly available from Binance API
            margin_multiplier: parseFloat(accountInfo.leverage || '1')
          };

          // Process balances
          let totalValue = 0;
          let cashValue = 0;

          accountInfo.balances.forEach((balance: any) => {
            const asset = balance.asset;
            const free = parseFloat(balance.free);
            const locked = parseFloat(balance.locked);
            const total = free + locked;

            // Skip zero balances
            if (total <= 0) return;

            // Stablecoins are considered cash
            const isStablecoin = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD'].includes(asset);

            if (isStablecoin) {
              cashValue += total;
              totalValue += total;
            } else {
              // Find price for this asset against USDT
              const symbol = `${asset}USDT`;
              const price = priceMap[symbol] || 0;

              if (price > 0) {
                const marketValue = total * price;
                totalValue += marketValue;

                // Add to positions
                portfolio.positions[asset] = {
                  symbol: asset,
                  quantity: total,
                  entry_price: 0, // Not available from Binance API
                  current_price: price,
                  market_value: marketValue,
                  unrealized_pnl: 0, // Not available without order history
                  realized_pnl: 0 // Not available without order history
                };
              }
            }
          });

          portfolio.cash = cashValue;
          portfolio.total_value = totalValue;

          // Cache the portfolio for future fallbacks
          localStorage.setItem('Binance:portfolio', JSON.stringify({
            timestamp: Date.now(),
            data: portfolio
          }));

          return portfolio;
        },
        'Binance',
        'GET_PORTFOLIO',
        {
          primaryFallback: async () => {
            console.log('Using primary fallback (backend API) for Binance portfolio');
            const { data } = await backendClient.get('/portfolio');
            return data.portfolio;
          },
          secondaryFallback: async () => {
            console.log('Using secondary fallback (cached with updates) for Binance portfolio');
            // Try to get cached portfolio data
            const cachedPortfolio = await getCachedPortfolio();
            if (!cachedPortfolio) throw new Error('No cached portfolio available');

            // Update prices for positions if possible
            try {
              for (const symbol of Object.keys(cachedPortfolio.positions)) {
                try {
                  const currentPrice = await fetchMarketPrice(symbol);
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
    },

    async getPositions(): Promise<Record<string, Position>> {
      return executeTrading<Record<string, Position>>(
        async () => {
          const portfolio = await this.getPortfolio();
          return portfolio.positions || {};
        },
        'Binance',
        'GET_POSITIONS',
        {
          primaryFallback: async () => {
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
          // Get account information
          const timestamp = Date.now();
          const queryString = `timestamp=${timestamp}`;
          const signature = generateSignature(queryString, config);

          const { data: account } = await client.get(
            `/api/v3/account?${queryString}&signature=${signature}`
          );

          if (asset) {
            // Convert asset format from "BTC/USDT" to "BTC"
            const baseAsset = asset.split('/')[0];
            const balance = account.balances.find((b: any) => b.asset === baseAsset);
            return balance ? parseFloat(balance.free) : 0;
          } else {
            // Return USDT balance as default
            const usdtBalance = account.balances.find((b: any) => b.asset === 'USDT');
            return usdtBalance ? parseFloat(usdtBalance.free) : 0;
          }
        },
        'Binance',
        'GET_ACCOUNT_INFO',
        {
          primaryFallback: async () => {
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
          // Convert our order format to Binance format
          const symbol = orderRequest.symbol.replace('/', '');

          // Base parameters for all order types
          const params: Record<string, any> = {
            symbol,
            side: orderRequest.side.toUpperCase(),
            type: orderRequest.order_type.toUpperCase(),
            quantity: orderRequest.quantity,
            timestamp: Date.now(),
          };

          // Add parameters based on order type
          if (orderRequest.order_type === 'limit' && orderRequest.price) {
            params.timeInForce = 'GTC';
            params.price = orderRequest.price;
          } else if (orderRequest.order_type === 'stop' && orderRequest.stop_price) {
            params.stopPrice = orderRequest.stop_price;
          } else if (orderRequest.order_type === 'stop_limit' && orderRequest.price && orderRequest.stop_price) {
            params.timeInForce = 'GTC';
            params.price = orderRequest.price;
            params.stopPrice = orderRequest.stop_price;
          }

          // Generate signature
          const queryString = Object.entries(params)
            .map(([key, value]) => `${key}=${value}`)
            .join('&');
          const signature = generateSignature(queryString, config);

          // Send order to Binance
          const { data } = await client.post(
            '/api/v3/order',
            null,
            {
              params: {
                ...params,
                signature,
              },
            }
          );

          // Convert Binance order to our format
          return {
            id: data.orderId.toString(),
            symbol: orderRequest.symbol,
            type: data.type as OrderType,
            side: data.side as OrderSide,
            quantity: parseFloat(data.origQty),
            price: parseFloat(data.price) || undefined,
            status: data.status as OrderStatus,
            createdAt: new Date(data.transactTime),
            updatedAt: new Date(data.transactTime),
            clientOrderId: data.clientOrderId,
            timeInForce: data.timeInForce,
            filledQuantity: parseFloat(data.executedQty),
          };
        },
        'Binance',
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
          // Need symbol for Binance cancel
          const orders = await this.getOrders();
          const order = orders.find(o => o.id === orderId);

          if (!order) {
            return false;
          }

          const symbol = order.symbol.replace('/', ''); // Convert "BTC/USDT" to "BTCUSDT"

          const params = {
            symbol,
            orderId,
            timestamp: Date.now(),
          };
          const queryString = Object.entries(params)
            .map(([key, value]) => `${key}=${value}`)
            .join('&');
          const signature = generateSignature(queryString, config);

          await client.delete('/api/v3/order', {
            params: {
              ...params,
              signature,
            },
          });
          return true;
        },
        'Binance',
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
          // Get open orders
          const openOrdersParams = {
            timestamp: Date.now(),
          };
          const queryString = Object.entries(openOrdersParams)
            .map(([key, value]) => `${key}=${value}`)
            .join('&');
          const signature = generateSignature(queryString, config);
          const { data: openOrders } = await client.get('/api/v3/openOrders', {
            params: {
              ...openOrdersParams,
              signature,
            },
          });

          // Get order history (last 7 days by default)
          const historyParams = {
            timestamp: Date.now(),
          };
          const historyQueryString = Object.entries(historyParams)
            .map(([key, value]) => `${key}=${value}`)
            .join('&');
          const historySignature = generateSignature(historyQueryString, config);
          const { data: orderHistory } = await client.get('/api/v3/allOrders', {
            params: {
              ...historyParams,
              signature: historySignature,
            },
          });

          // Combine and convert to our format
          const allOrders = [...openOrders, ...orderHistory].map((order: any) => ({
            id: order.orderId.toString(),
            symbol: order.symbol,
            type: order.type as OrderType,
            side: order.side as OrderSide,
            quantity: parseFloat(order.origQty),
            price: parseFloat(order.price) || undefined,
            status: order.status as OrderStatus,
            createdAt: new Date(order.time || order.transactTime || Date.now()),
            updatedAt: new Date(order.updateTime || order.transactTime || Date.now()),
            clientOrderId: order.clientOrderId,
            timeInForce: order.timeInForce,
            filledQuantity: parseFloat(order.executedQty),
          }));

          // Filter by status if provided
          if (status) {
            return allOrders.filter(order => order.status === status);
          }

          return allOrders;
        },
        'Binance',
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
          // Need symbol for Binance order query
          const orders = await this.getOrders();
          const order = orders.find(o => o.id === orderId);

          if (!order) {
            return null;
          }

          const symbol = order.symbol.replace('/', ''); // Convert "BTC/USDT" to "BTCUSDT"

          const params = {
            symbol,
            orderId,
            timestamp: Date.now(),
          };
          const queryString = Object.entries(params)
            .map(([key, value]) => `${key}=${value}`)
            .join('&');
          const signature = generateSignature(queryString, config);

          const { data } = await client.get('/api/v3/order', {
            params: {
              ...params,
              signature,
            },
          });
          return {
            id: data.orderId.toString(),
            symbol: order.symbol,
            type: data.type as OrderType,
            side: data.side as OrderSide,
            quantity: parseFloat(data.origQty),
            price: parseFloat(data.price) || undefined,
            status: data.status as OrderStatus,
            createdAt: new Date(data.time || data.transactTime || Date.now()),
            updatedAt: new Date(data.updateTime || data.transactTime || Date.now()),
            clientOrderId: data.clientOrderId,
            timeInForce: data.timeInForce,
            filledQuantity: parseFloat(data.executedQty),
          };
        },
        'Binance',
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
          // Convert symbol format
          const binanceSymbol = symbol.replace('/', '');

          // Get ticker price from Binance
          const { data } = await client.get('/api/v3/ticker/price', {
            params: { symbol: binanceSymbol },
          });

          return parseFloat(data.price);
        },
        'Binance',
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
          // Convert symbol format
          const binanceSymbol = symbol.replace('/', '');

          // Get order book from Binance
          const { data } = await client.get('/api/v3/depth', {
            params: { symbol: binanceSymbol, limit },
          });

          // Convert Binance format to our format
          const bids = data.bids.map((bid: string[]) => ({
            price: parseFloat(bid[0]),
            size: parseFloat(bid[1]),
          }));

          const asks = data.asks.map((ask: string[]) => ({
            price: parseFloat(ask[0]),
            size: parseFloat(ask[1]),
          }));

          return { bids, asks };
        },
        'Binance',
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
          const binanceSymbol = symbol.replace('/', ''); // Convert "BTC/USDT" to "BTCUSDT"
          const { data } = await client.get('/api/v3/ticker/24hr', { params: { symbol: binanceSymbol } });

          return {
            price: parseFloat(data.lastPrice),
            volume: parseFloat(data.volume),
            change: parseFloat(data.priceChangePercent) / 100,
          };
        },
        'Binance',
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
          const { data } = await client.get('/api/v3/exchangeInfo');

          // Convert to a more usable format
          return {
            name: 'Binance',
            symbols: data.symbols.map((s: any) => `${s.baseAsset}/${s.quoteAsset}`),
            tradingFees: 0.001, // Default Binance fee
            withdrawalFees: {},
            serverTime: data.serverTime,
          };
        },
        'Binance',
        'GET_MARKET_DATA',
        {
          primaryFallback: async () => {
            // Fallback to our backend
            try {
              const { data } = await backendClient.get('/market/exchange-info');
              return data.exchangeInfo;
            } catch {
              return {
                name: 'Binance',
                symbols: [],
                tradingFees: 0.001,
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
          const { data } = await client.get('/api/v3/exchangeInfo');
          return data.symbols.map((s: any) => `${s.baseAsset}/${s.quoteAsset}`);
        },
        'Binance',
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
          const binanceSymbol = symbol.replace('/', ''); // Convert "BTC/USDT" to "BTCUSDT"
          const { data } = await client.get('/api/v3/exchangeInfo');

          const symbolInfo = data.symbols.find((s: any) => s.symbol === binanceSymbol);

          if (!symbolInfo) {
            throw new Error(`Symbol ${symbol} not found`);
          }

          // Find quantity filter
          const lotSizeFilter = symbolInfo.filters.find((f: any) => f.filterType === 'LOT_SIZE');
          const priceFilter = symbolInfo.filters.find((f: any) => f.filterType === 'PRICE_FILTER');
          const minNotionalFilter = symbolInfo.filters.find((f: any) => f.filterType === 'MIN_NOTIONAL');

          return {
            symbol,
            baseAsset: symbolInfo.baseAsset,
            quoteAsset: symbolInfo.quoteAsset,
            minQuantity: lotSizeFilter ? parseFloat(lotSizeFilter.minQty) : 0,
            maxQuantity: lotSizeFilter ? parseFloat(lotSizeFilter.maxQty) : 0,
            quantityPrecision: symbolInfo.baseAssetPrecision,
            pricePrecision: symbolInfo.quotePrecision,
            minNotional: minNotionalFilter ? parseFloat(minNotionalFilter.minNotional) : 0,
            tickSize: priceFilter ? parseFloat(priceFilter.tickSize) : 0,
          };
        },
        'Binance',
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
                baseAsset: symbol.split('/')[0],
                quoteAsset: symbol.split('/')[1],
                minQuantity: 0.0001,
                maxQuantity: 1000,
                quantityPrecision: 4,
                pricePrecision: 2,
                minNotional: 10,
                tickSize: 0.01,
              };
            }
          }
        }
      );
    },
  };
};
