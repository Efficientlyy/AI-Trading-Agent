import { TradingApi } from './index';
import { OrderRequest, Order, Portfolio, Position, OrderStatus, OrderType, OrderSide } from '../../types';
import { createAuthenticatedClient } from '../client';
// @ts-ignore
import { v4 as uuidv4 } from 'uuid';

// Paper trading implementation that wraps around a real exchange API
// but simulates order execution without real money
export const createPaperTradingApi = (baseApi: TradingApi): TradingApi => {
  // Create authenticated client for our backend
  const backendClient = createAuthenticatedClient();
  
  // In-memory storage for paper trading
  const paperState = {
    portfolio: {
      total_value: 100000, // Default starting balance of $100,000
      cash: 100000,
      positions: {} as Record<string, Position>,
      daily_pnl: 0,
    },
    orders: [] as Order[],
    orderIdMap: new Map<string, string>(), // Maps real order IDs to paper order IDs
  };
  
  // Define order status and type constants to match our types
  const ORDER_STATUS = {
    NEW: OrderStatus.NEW,
    PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
    FILLED: OrderStatus.FILLED,
    CANCELED: OrderStatus.CANCELED,
    REJECTED: OrderStatus.REJECTED,
  };
  
  const ORDER_TYPE = {
    MARKET: OrderType.MARKET,
    LIMIT: OrderType.LIMIT,
    STOP: OrderType.STOP,
    STOP_LIMIT: OrderType.STOP_LIMIT,
  };
  
  // Define order side constants
  const ORDER_SIDE = {
    BUY: OrderSide.BUY,
    SELL: OrderSide.SELL,
  };
  
  // Load paper trading state from local storage
  const loadState = () => {
    try {
      const savedState = localStorage.getItem('paperTradingState');
      if (savedState) {
        const parsed = JSON.parse(savedState);
        paperState.portfolio = parsed.portfolio;
        paperState.orders = parsed.orders;
        paperState.orderIdMap = new Map(parsed.orderIdMap);
      }
    } catch (error) {
      console.error('Error loading paper trading state:', error);
    }
  };
  
  // Save paper trading state to local storage
  const saveState = () => {
    try {
      localStorage.setItem('paperTradingState', JSON.stringify({
        portfolio: paperState.portfolio,
        orders: paperState.orders,
        orderIdMap: Array.from(paperState.orderIdMap.entries()),
      }));
    } catch (error) {
      console.error('Error saving paper trading state:', error);
    }
  };
  
  // Load state on initialization
  loadState();
  
  // Process pending orders
  const processOrders = async () => {
    const currentTime = new Date();
    const marketHours = isMarketOpen();
    
    // Only process orders during market hours
    if (!marketHours.isOpen) {
      return;
    }
    
    // Get pending orders
    const pendingOrders = paperState.orders.filter(
      order => order.status === ORDER_STATUS.NEW || order.status === ORDER_STATUS.PARTIALLY_FILLED
    );
    
    for (const order of pendingOrders) {
      try {
        // Get current market price
        const marketPrice = await baseApi.getMarketPrice(order.symbol);
        
        // Skip if we couldn't get a market price
        if (!marketPrice) continue;
        
        // Process order based on type
        switch (order.type) {
          case ORDER_TYPE.MARKET:
            await executeMarketOrder(order, marketPrice);
            break;
          case ORDER_TYPE.LIMIT:
            await executeLimitOrder(order, marketPrice);
            break;
          case ORDER_TYPE.STOP:
            await executeStopOrder(order, marketPrice);
            break;
          case ORDER_TYPE.STOP_LIMIT:
            await executeStopLimitOrder(order, marketPrice);
            break;
        }
      } catch (error) {
        console.error(`Error processing paper order ${order.id}:`, error);
      }
    }
    
    // Save state after processing
    saveState();
  };
  
  // Check if market is open
  const isMarketOpen = () => {
    const now = new Date();
    const day = now.getDay();
    const hour = now.getHours();
    const minute = now.getMinutes();
    
    // Default to always open for crypto
    const isOpen = true;
    const nextOpenTime = new Date();
    const nextCloseTime = new Date();
    
    return {
      isOpen,
      nextOpenTime,
      nextCloseTime,
    };
  };
  
  // Execute a market order
  const executeMarketOrder = async (order: Order, marketPrice: number) => {
    // For market orders, execute immediately at market price
    const remainingQty = order.quantity - (order.filledQuantity || 0);
    
    if (remainingQty <= 0) {
      return;
    }
    
    // Update portfolio
    if (order.side === 'BUY') {
      // Check if we have enough cash
      const cost = remainingQty * marketPrice;
      if (paperState.portfolio.cash < cost) {
        // Partial fill if we don't have enough cash
        const affordableQty = paperState.portfolio.cash / marketPrice;
        if (affordableQty <= 0) {
          order.status = ORDER_STATUS.REJECTED;
          return;
        }
        
        // Update order with partial fill
        order.filledQuantity = (order.filledQuantity || 0) + affordableQty;
        order.price = marketPrice;
        order.status = ORDER_STATUS.PARTIALLY_FILLED;
        
        // Update portfolio
        updatePortfolio(order.symbol, affordableQty, marketPrice, ORDER_SIDE.BUY);
      } else {
        // Full fill
        order.filledQuantity = order.quantity;
        order.price = marketPrice;
        order.status = ORDER_STATUS.FILLED;
        order.updatedAt = new Date();
        
        // Update portfolio
        updatePortfolio(order.symbol, remainingQty, marketPrice, ORDER_SIDE.BUY);
      }
    } else {
      // SELL order
      // Check if we have enough of the asset
      const position = paperState.portfolio.positions[order.symbol];
      const availableQty = position ? position.quantity : 0;
      
      if (availableQty < remainingQty) {
        // Partial fill if we don't have enough of the asset
        if (availableQty <= 0) {
          order.status = ORDER_STATUS.REJECTED;
          return;
        }
        
        // Update order with partial fill
        order.filledQuantity = (order.filledQuantity || 0) + availableQty;
        order.price = marketPrice;
        order.status = ORDER_STATUS.PARTIALLY_FILLED;
        
        // Update portfolio
        updatePortfolio(order.symbol, availableQty, marketPrice, ORDER_SIDE.SELL);
      } else {
        // Full fill
        order.filledQuantity = order.quantity;
        order.price = marketPrice;
        order.status = ORDER_STATUS.FILLED;
        order.updatedAt = new Date();
        
        // Update portfolio
        updatePortfolio(order.symbol, remainingQty, marketPrice, ORDER_SIDE.SELL);
      }
    }
  };
  
  // Execute a limit order
  const executeLimitOrder = async (order: Order, marketPrice: number) => {
    const remainingQty = order.quantity - (order.filledQuantity || 0);
    
    if (remainingQty <= 0) {
      return;
    }
    
    // For limit orders, execute only if price conditions are met
    if (order.side === 'BUY') {
      // For buy limit orders, execute if market price <= limit price
      if (marketPrice <= (order.price || 0)) {
        // Check if we have enough cash
        const cost = remainingQty * marketPrice;
        if (paperState.portfolio.cash < cost) {
          // Partial fill if we don't have enough cash
          const affordableQty = paperState.portfolio.cash / marketPrice;
          if (affordableQty <= 0) {
            return; // Keep the order open
          }
          
          // Update order with partial fill
          order.filledQuantity = (order.filledQuantity || 0) + affordableQty;
          order.status = ORDER_STATUS.PARTIALLY_FILLED;
          
          // Update portfolio
          updatePortfolio(order.symbol, affordableQty, marketPrice, ORDER_SIDE.BUY);
        } else {
          // Full fill
          order.filledQuantity = order.quantity;
          order.status = ORDER_STATUS.FILLED;
          order.updatedAt = new Date();
          
          // Update portfolio
          updatePortfolio(order.symbol, remainingQty, marketPrice, ORDER_SIDE.BUY);
        }
      }
    } else {
      // For sell limit orders, execute if market price >= limit price
      if (marketPrice >= (order.price || 0)) {
        // Check if we have enough of the asset
        const position = paperState.portfolio.positions[order.symbol];
        const availableQty = position ? position.quantity : 0;
        
        if (availableQty < remainingQty) {
          // Partial fill if we don't have enough of the asset
          if (availableQty <= 0) {
            return; // Keep the order open
          }
          
          // Update order with partial fill
          order.filledQuantity = (order.filledQuantity || 0) + availableQty;
          order.status = ORDER_STATUS.PARTIALLY_FILLED;
          
          // Update portfolio
          updatePortfolio(order.symbol, availableQty, marketPrice, ORDER_SIDE.SELL);
        } else {
          // Full fill
          order.filledQuantity = order.quantity;
          order.status = ORDER_STATUS.FILLED;
          order.updatedAt = new Date();
          
          // Update portfolio
          updatePortfolio(order.symbol, remainingQty, marketPrice, ORDER_SIDE.SELL);
        }
      }
    }
  };
  
  // Execute a stop order
  const executeStopOrder = async (order: Order, marketPrice: number) => {
    // For stop orders, check if stop price has been triggered
    if (order.side === 'BUY') {
      // For buy stop orders, trigger if market price >= stop price
      if (marketPrice >= (order.stopPrice || 0)) {
        // Convert to market order once triggered
        order.type = ORDER_TYPE.MARKET;
        await executeMarketOrder(order, marketPrice);
      }
    } else {
      // For sell stop orders, trigger if market price <= stop price
      if (marketPrice <= (order.stopPrice || 0)) {
        // Convert to market order once triggered
        order.type = ORDER_TYPE.MARKET;
        await executeMarketOrder(order, marketPrice);
      }
    }
  };
  
  // Execute a stop-limit order
  const executeStopLimitOrder = async (order: Order, marketPrice: number) => {
    // For stop-limit orders, first check if stop price has been triggered
    if (order.side === 'BUY') {
      // For buy stop-limit orders, trigger if market price >= stop price
      if (marketPrice >= (order.stopPrice || 0)) {
        // Convert to limit order once triggered
        order.type = ORDER_TYPE.LIMIT;
        await executeLimitOrder(order, marketPrice);
      }
    } else {
      // For sell stop-limit orders, trigger if market price <= stop price
      if (marketPrice <= (order.stopPrice || 0)) {
        // Convert to limit order once triggered
        order.type = ORDER_TYPE.LIMIT;
        await executeLimitOrder(order, marketPrice);
      }
    }
  };
  
  // Update portfolio with a trade
  const updatePortfolio = (
    symbol: string,
    quantity: number,
    price: number,
    side: OrderSide
  ) => {
    // Get current position or create a new one
    const position = paperState.portfolio.positions[symbol] || {
      symbol,
      quantity: 0,
      entry_price: 0,
      current_price: price,
      market_value: 0,
      unrealized_pnl: 0,
      realized_pnl: 0,
    };
    
    const cost = quantity * price;
    
    if (side === 'BUY') {
      // Update cash
      paperState.portfolio.cash -= cost;
      
      // Update position
      const totalCost = position.quantity * position.entry_price + cost;
      const totalQuantity = position.quantity + quantity;
      
      position.entry_price = totalCost / totalQuantity;
      position.quantity = totalQuantity;
      position.current_price = price;
      position.market_value = position.quantity * price;
      position.unrealized_pnl = position.market_value - (position.entry_price * position.quantity);
    } else {
      // Update cash
      paperState.portfolio.cash += cost;
      
      // Calculate realized P&L
      const realizedPnl = (price - position.entry_price) * quantity;
      // Initialize realized_pnl if undefined
      position.realized_pnl = position.realized_pnl || 0;
      position.realized_pnl += realizedPnl;
      
      // Update position
      position.quantity -= quantity;
      position.current_price = price;
      position.market_value = position.quantity * price;
      position.unrealized_pnl = position.market_value - (position.entry_price * position.quantity);
      
      // Remove position if quantity is zero
      if (position.quantity <= 0) {
        delete paperState.portfolio.positions[symbol];
      }
    }
    
    // If position still exists, update it
    if (position.quantity > 0) {
      paperState.portfolio.positions[symbol] = position;
    }
    
    // Update total portfolio value
    updatePortfolioValue();
  };
  
  // Update total portfolio value
  const updatePortfolioValue = async () => {
    let totalValue = paperState.portfolio.cash;
    
    // Add value of all positions
    for (const symbol in paperState.portfolio.positions) {
      const position = paperState.portfolio.positions[symbol];
      
      // Try to get current market price
      try {
        const marketPrice = await baseApi.getMarketPrice(symbol);
        if (marketPrice) {
          // Update position with current price
          position.current_price = marketPrice;
          position.market_value = position.quantity * marketPrice;
          position.unrealized_pnl = position.market_value - (position.entry_price * position.quantity);
        }
      } catch (error) {
        console.error(`Error updating price for ${symbol}:`, error);
      }
      
      totalValue += position.market_value;
    }
    
    // Update total value
    paperState.portfolio.total_value = totalValue;
  };
  
  // Start order processing
  const orderProcessingInterval = setInterval(processOrders, 5000);
  
  // Clean up on unmount
  window.addEventListener('beforeunload', () => {
    clearInterval(orderProcessingInterval);
    saveState();
  });
  
  // Return the paper trading API implementation
  return {
    // Account and portfolio methods
    async getPortfolio(): Promise<Portfolio> {
      // Process any pending orders first
      await processOrders();
      
      // Return the paper portfolio
      return paperState.portfolio;
    },
    
    async getPositions(): Promise<Record<string, Position>> {
      // Process any pending orders first
      await processOrders();
      
      // Return the paper positions
      return paperState.portfolio.positions;
    },
    
    async getBalance(asset?: string): Promise<number> {
      // Process any pending orders first
      await processOrders();
      
      if (asset) {
        // Return balance for specific asset
        const position = paperState.portfolio.positions[asset];
        return position ? position.market_value : 0;
      } else {
        // Return cash balance
        return paperState.portfolio.cash;
      }
    },
    
    // Order management methods
    async createOrder(orderRequest: OrderRequest): Promise<Order> {
      // Create a new paper order
      const orderId = uuidv4();
      const now = new Date();
      
      // Get current market price
      const marketPrice = await baseApi.getMarketPrice(orderRequest.symbol);
      
      // Map order type and side to our enum values
      let orderType: OrderType;
      switch (orderRequest.order_type) {
        case 'market':
          orderType = OrderType.MARKET;
          break;
        case 'limit':
          orderType = OrderType.LIMIT;
          break;
        case 'stop':
          orderType = OrderType.STOP;
          break;
        case 'stop_limit':
          orderType = OrderType.STOP_LIMIT;
          break;
        default:
          orderType = OrderType.MARKET;
      }
      
      const orderSide = orderRequest.side === 'buy' ? OrderSide.BUY : OrderSide.SELL;
      
      const newOrder: Order = {
        id: orderId,
        symbol: orderRequest.symbol,
        type: orderType,
        side: orderSide,
        quantity: orderRequest.quantity,
        price: marketPrice,
        stopPrice: orderRequest.stop_price,
        status: ORDER_STATUS.NEW,
        created_at: new Date().toISOString(),
        createdAt: new Date(),
        updated_at: new Date().toISOString(),
        updatedAt: new Date(),
        clientOrderId: `paper-${Date.now()}`,
        timeInForce: 'GTC',
        filledQuantity: 0,
        filled_quantity: 0
      };
      
      // Add to orders list
      paperState.orders.push(newOrder);
      
      // Process the order immediately
      await processOrders();
      
      // Save state
      saveState();
      
      // Return the order
      return newOrder;
    },
    
    async cancelOrder(orderId: string): Promise<boolean> {
      // Find the order
      const orderIndex = paperState.orders.findIndex(order => order.id === orderId);
      
      if (orderIndex === -1) {
        return false;
      }
      
      // Check if order can be canceled
      const order = paperState.orders[orderIndex];
      if (order.status === ORDER_STATUS.FILLED || order.status === ORDER_STATUS.CANCELED || order.status === ORDER_STATUS.REJECTED) {
        return false;
      }
      
      // Cancel the order
      order.status = ORDER_STATUS.CANCELED;
      order.updatedAt = new Date();
      
      // Save state
      saveState();
      
      return true;
    },
    
    async getOrders(status?: string): Promise<Order[]> {
      // Process any pending orders first
      await processOrders();
      
      // Filter orders by status if provided
      if (status) {
        return paperState.orders.filter(order => order.status === status.toUpperCase());
      }
      
      // Return all orders
      return paperState.orders;
    },
    
    async getOrder(orderId: string): Promise<Order | null> {
      // Process any pending orders first
      await processOrders();
      
      // Find the order
      const order = paperState.orders.find(order => order.id === orderId);
      
      return order || null;
    },
    
    // Market data methods - delegate to base API
    async getMarketPrice(symbol: string): Promise<number> {
      return baseApi.getMarketPrice(symbol);
    },
    
    async getOrderBook(symbol: string, limit: number = 10): Promise<{ bids: any[], asks: any[] }> {
      return baseApi.getOrderBook(symbol, limit);
    },
    
    async getTicker(symbol: string): Promise<{ price: number, volume: number, change: number }> {
      return baseApi.getTicker(symbol);
    },
    
    // Exchange info methods - delegate to base API
    async getExchangeInfo(): Promise<any> {
      const info = await baseApi.getExchangeInfo();
      
      // Modify to indicate paper trading
      return {
        ...info,
        name: `${info.name} (Paper)`,
      };
    },
    
    async getSymbols(): Promise<string[]> {
      return baseApi.getSymbols();
    },
    
    async getAssetInfo(symbol: string): Promise<any> {
      return baseApi.getAssetInfo(symbol);
    },
  };
};
