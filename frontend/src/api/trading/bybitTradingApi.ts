// filepath: bybitTradingApi.ts
import axios from 'axios';
import { createAuthenticatedClient } from '../client';
import { TradingApi } from './index';
import {
  Order,
  OrderRequest,
  OrderSide,
  OrderStatus,
  OrderType,
  Portfolio,
  Position
} from '../../types';
import { canMakeApiCall, recordApiCall, recordCircuitBreakerResult } from '../utils/monitoring';

/**
 * Bybit trading API implementation
 * @param mode Trading mode (live or paper)
 * @param config API configuration
 */
export const bybitTradingApi = (mode: 'live' | 'paper', config: { apiKey: string, apiSecret: string }): TradingApi => {
  const client = axios.create({
    baseURL: 'https://api.bybit.com',
    headers: {
      'X-BAPI-API-KEY': config.apiKey,
      'X-BAPI-TIMESTAMP': Date.now().toString(),
      'X-BAPI-SIGN': '',  // Would be generated per request
      'X-BAPI-RECV-WINDOW': '5000',
      'Content-Type': 'application/json',
    }
  });
  
  // Backend client as fallback
  const backendClient = createAuthenticatedClient();

  // Helper for circuit breaker pattern
  const withCircuitBreaker = async <T>(
    methodName: string,
    apiCall: () => Promise<T>,
    fallback: () => Promise<T>
  ): Promise<T> => {
    if (!canMakeApiCall('Bybit', methodName)) {
      return fallback();
    }
    
    try {
      recordApiCall('Bybit', methodName, 'attempt');
      const result = await apiCall();
      recordApiCall('Bybit', methodName, 'success');
      recordCircuitBreakerResult('Bybit', methodName, true);
      return result;
    } catch (error) {
      recordApiCall('Bybit', methodName, 'failure');
      recordCircuitBreakerResult('Bybit', methodName, false);
      console.error(`Bybit API error (${methodName}):`, error);
      return fallback();
    }
  };

  // Map Bybit-specific order types to our app's order types
  const mapBybitOrderStatus = (status: string): OrderStatus => {
    switch (status) {
      case 'Created':
      case 'New':
        return OrderStatus.NEW;
      case 'PartiallyFilled':
        return OrderStatus.PARTIALLY_FILLED;
      case 'Filled':
        return OrderStatus.FILLED;
      case 'Cancelled':
      case 'Rejected':
        return OrderStatus.CANCELED;
      default:
        return OrderStatus.NEW;
    }
  };

  // Map Bybit order response to our app's Order type
  const mapBybitOrderToOrder = (bybitOrder: any): Order => {
    return {
      id: bybitOrder.order_id,
      symbol: bybitOrder.symbol.replace('USDT', '/USDT'),
      side: bybitOrder.side === 'Buy' ? OrderSide.BUY : OrderSide.SELL,
      type: bybitOrder.order_type === 'Market' ? OrderType.MARKET : OrderType.LIMIT,
      quantity: Number(bybitOrder.qty),
      price: Number(bybitOrder.price) || 0,
      status: mapBybitOrderStatus(bybitOrder.order_status),
      created_at: bybitOrder.create_time || new Date().toISOString(),
      updated_at: bybitOrder.update_time || new Date().toISOString(),
      clientOrderId: bybitOrder.client_order_id,
      timeInForce: bybitOrder.time_in_force || 'GTC',
      filledQuantity: Number(bybitOrder.cum_exec_qty) || 0,
    };
  };

  return {
    // Portfolio methods
    async getPortfolio(): Promise<Portfolio> {
      return withCircuitBreaker(
        'getPortfolio',
        async () => {
          // Get account balances
          const accountResponse = await client.get('/v5/account/wallet-balance', {
            params: {
              accountType: 'SPOT',
            }
          });
          
          // Get ticker prices for all assets
          const tickerResponse = await client.get('/v5/market/tickers', {
            params: { category: 'spot' }
          });

          // Process account data
          const balances = accountResponse.data.result.list[0].coin;
          const prices: Record<string, number> = {};
          
          tickerResponse.data.result.list.forEach((ticker: any) => {
            prices[ticker.symbol] = Number(ticker.lastPrice);
          });
          
          // Calculate portfolio data
          let cash = 0;
          const positions: Record<string, Position> = {};
          
          balances.forEach((balance: any) => {
            const asset = balance.coin;
            const free = Number(balance.free);
            const locked = Number(balance.locked || 0);
            const total = free + locked;
            
            if (asset === 'USDT') {
              cash = total;
              return;
            }
            
            const symbol = `${asset}/USDT`;
            const tickerSymbol = `${asset}USDT`;
            const price = prices[tickerSymbol] || 0;
            
            if (total > 0 && price > 0) {
              positions[symbol] = {
                symbol,
                quantity: total,
                entry_price: 0, // Bybit doesn't provide this directly
                current_price: price,
                market_value: total * price,
                unrealized_pnl: 0, // Would need to calculate based on trades
              };
            }
          });
          
          // Calculate total portfolio value
          let totalValue = cash;
          Object.values(positions).forEach(pos => {
            totalValue += pos.market_value;
          });
          
          return {
            cash,
            total_value: totalValue,
            positions,
          };
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.get('/portfolio');
          return response.data.portfolio;
        }
      );
    },

    async getPositions(): Promise<Record<string, Position>> {
      return withCircuitBreaker(
        'getPositions',
        async () => {
          // Get positions from portfolio
          const portfolio = await this.getPortfolio();
          return portfolio.positions;
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.get('/positions');
          return response.data.positions;
        }
      );
    },

    async getBalance(asset?: string): Promise<number> {
      return withCircuitBreaker(
        'getBalance',
        async () => {
          const accountResponse = await client.get('/v5/account/wallet-balance', {
            params: { accountType: 'SPOT' }
          });
          
          const balances = accountResponse.data.result.list[0].coin;
          
          if (asset) {
            // Find specific asset balance
            const assetSymbol = asset.split('/')[0];
            const assetBalance = balances.find((b: any) => b.coin === assetSymbol);
            return assetBalance ? Number(assetBalance.free) + Number(assetBalance.locked || 0) : 0;
          } else {
            // Return USDT balance as cash
            const usdtBalance = balances.find((b: any) => b.coin === 'USDT');
            return usdtBalance ? Number(usdtBalance.free) + Number(usdtBalance.locked || 0) : 0;
          }
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.get('/balance', { params: { asset } });
          return response.data.balance;
        }
      );
    },

    // Order methods
    async createOrder(orderRequest: OrderRequest): Promise<Order> {
      return withCircuitBreaker(
        'createOrder',
        async () => {
          // Format symbol for Bybit (remove slash)
          const bybitSymbol = orderRequest.symbol.replace('/', '');
          
          // Create order request object
          const requestData: any = {
            category: 'spot',
            symbol: bybitSymbol,
            side: orderRequest.side.toUpperCase(),
            orderType: orderRequest.order_type?.toUpperCase() || 'MARKET',
            qty: orderRequest.quantity.toString(),
            timeInForce: 'GTC',
          };
          
          // Add price if limit order
          if (orderRequest.order_type === 'limit' && orderRequest.price) {
            requestData.price = orderRequest.price.toString();
          }
          
          // Send order request
          const response = await client.post('/v5/order/create', requestData);
          
          // Check for success
          if (response.data.ret_code !== 0) {
            throw new Error(`Bybit order error: ${response.data.ret_msg}`);
          }
          
          // Get order details
          const orderResponse = await client.get('/v5/order/realtime', {
            params: {
              category: 'spot',
              orderId: response.data.result.orderId,
            }
          });
          
          return mapBybitOrderToOrder(orderResponse.data.result.list[0]);
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.post('/orders', orderRequest);
          return response.data.order;
        }
      );
    },

    async cancelOrder(orderId: string): Promise<boolean> {
      return withCircuitBreaker(
        'cancelOrder',
        async () => {
          const response = await client.post('/v5/order/cancel', {
            category: 'spot',
            orderId: orderId,
          });
          
          return response.data.ret_code === 0;
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.delete(`/orders/${orderId}`);
          return response.data.success;
        }
      );
    },

    async getOrders(status?: string): Promise<Order[]> {
      return withCircuitBreaker(
        'getOrders',
        async () => {
          const params: any = { category: 'spot' };
          
          if (status) {
            params.orderStatus = status;
          }
          
          const response = await client.get('/v5/order/history', { params });
          
          return response.data.result.list.map(mapBybitOrderToOrder);
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.get('/orders', { params: { status } });
          return response.data.orders;
        }
      );
    },

    async getOrder(orderId: string): Promise<Order | null> {
      return withCircuitBreaker(
        'getOrder',
        async () => {
          const response = await client.get('/v5/order/realtime', {
            params: {
              category: 'spot',
              orderId: orderId,
            }
          });
          
          if (response.data.result.list.length === 0) {
            return null;
          }
          
          return mapBybitOrderToOrder(response.data.result.list[0]);
        },
        async () => {
          // Fallback to backend API
          try {
            const response = await backendClient.get(`/orders/${orderId}`);
            return response.data.order;
          } catch (error) {
            return null;
          }
        }
      );
    },

    // Market data methods
    async getMarketPrice(symbol: string): Promise<number> {
      return withCircuitBreaker(
        'getMarketPrice',
        async () => {
          const bybitSymbol = symbol.replace('/', '');
          
          const response = await client.get('/v5/market/tickers', {
            params: {
              category: 'spot',
              symbol: bybitSymbol,
            }
          });
          
          if (response.data.result.list.length === 0) {
            throw new Error(`Symbol ${symbol} not found`);
          }
          
          return Number(response.data.result.list[0].lastPrice);
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.get(`/market/${symbol}/price`);
          return response.data.price;
        }
      );
    },

    async getOrderBook(symbol: string, limit: number = 10): Promise<{ bids: any[], asks: any[] }> {
      return withCircuitBreaker(
        'getOrderBook',
        async () => {
          const bybitSymbol = symbol.replace('/', '');
          
          const response = await client.get('/v5/market/orderbook', {
            params: {
              category: 'spot',
              symbol: bybitSymbol,
              limit: limit,
            }
          });
          
          return {
            bids: response.data.result.b.map((bid: any) => ({
              price: Number(bid[0]),
              quantity: Number(bid[1]),
            })),
            asks: response.data.result.a.map((ask: any) => ({
              price: Number(ask[0]),
              quantity: Number(ask[1]),
            })),
          };
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.get(`/market/${symbol}/orderbook`, {
            params: { limit }
          });
          return response.data;
        }
      );
    },

    async getTicker(symbol: string): Promise<{ price: number, volume: number, change: number }> {
      return withCircuitBreaker(
        'getTicker',
        async () => {
          const bybitSymbol = symbol.replace('/', '');
          
          const response = await client.get('/v5/market/tickers', {
            params: {
              category: 'spot',
              symbol: bybitSymbol,
            }
          });
          
          const ticker = response.data.result.list[0];
          
          return {
            price: Number(ticker.lastPrice),
            volume: Number(ticker.volume24h),
            change: Number(ticker.price24hPcnt) * 100,
          };
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.get(`/market/${symbol}/ticker`);
          return response.data;
        }
      );
    },

    // Exchange info methods
    async getExchangeInfo(): Promise<any> {
      return withCircuitBreaker(
        'getExchangeInfo',
        async () => {
          const response = await client.get('/v5/market/instruments-info', {
            params: { category: 'spot' }
          });
          
          return {
            name: 'Bybit',
            symbols: response.data.result.list.map((item: any) => 
              `${item.baseCoin}/${item.quoteCoin}`
            ),
            tradingFees: 0.001, // Default fee
          };
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.get('/exchange/info');
          return response.data;
        }
      );
    },

    async getSymbols(): Promise<string[]> {
      return withCircuitBreaker(
        'getSymbols',
        async () => {
          const info = await this.getExchangeInfo();
          return info.symbols;
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.get('/symbols');
          return response.data.symbols;
        }
      );
    },

    async getAssetInfo(symbol: string): Promise<any> {
      return withCircuitBreaker(
        'getAssetInfo',
        async () => {
          const bybitSymbol = symbol.replace('/', '');
          
          const response = await client.get('/v5/market/instruments-info', {
            params: {
              category: 'spot',
              symbol: bybitSymbol,
            }
          });
          
          if (response.data.result.list.length === 0) {
            throw new Error(`Symbol ${symbol} not found`);
          }
          
          const info = response.data.result.list[0];
          
          return {
            symbol,
            basePrecision: parseInt(info.lotSizeFilter.basePrecision),
            quotePrecision: parseInt(info.lotSizeFilter.quotePrecision),
            minQty: parseFloat(info.lotSizeFilter.minOrderQty),
            maxQty: parseFloat(info.lotSizeFilter.maxOrderQty),
            stepSize: parseFloat(info.lotSizeFilter.basePrecision),
            minNotional: parseFloat(info.lotSizeFilter.minOrderAmt),
            status: info.status === 'Trading',
          };
        },
        async () => {
          // Fallback to backend API
          const response = await backendClient.get(`/assets/${symbol}`);
          return response.data;
        }
      );
    },
  };
};