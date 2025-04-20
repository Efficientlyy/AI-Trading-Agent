/**
 * Mock data for trading integration tests
 */

// Mock order response
export const mockOrderResponse = {
  id: 'ord123456789',
  clientOrderId: 'client-ord-123',
  symbol: 'BTC-USD',
  side: 'buy',
  type: 'MARKET',
  status: 'FILLED',
  quantity: 0.1,
  price: 50000,
  filledQuantity: 0.1,
  filledPrice: 50000,
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  fees: 25,
  totalValue: 5000
};

// Mock market data for different assets
export const mockMarketData = {
  'BTC-USD': {
    symbol: 'BTC-USD',
    price: 50000,
    change24h: 1200,
    changePercent24h: 2.4,
    high24h: 51200,
    low24h: 49200,
    volume24h: 1500000000,
    marketCap: 950000000000,
    lastUpdated: new Date().toISOString()
  },
  'ETH-USD': {
    symbol: 'ETH-USD',
    price: 3500,
    change24h: 120,
    changePercent24h: 3.5,
    high24h: 3600,
    low24h: 3400,
    volume24h: 800000000,
    marketCap: 420000000000,
    lastUpdated: new Date().toISOString()
  },
  'SOL-USD': {
    symbol: 'SOL-USD',
    price: 150,
    change24h: 8,
    changePercent24h: 5.6,
    high24h: 155,
    low24h: 145,
    volume24h: 300000000,
    marketCap: 60000000000,
    lastUpdated: new Date().toISOString()
  }
};

// Mock available assets
export const mockAssets = [
  {
    symbol: 'BTC-USD',
    name: 'Bitcoin',
    minOrderSize: 0.0001,
    maxOrderSize: 100,
    pricePrecision: 2,
    quantityPrecision: 8,
    status: 'ACTIVE'
  },
  {
    symbol: 'ETH-USD',
    name: 'Ethereum',
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    pricePrecision: 2,
    quantityPrecision: 6,
    status: 'ACTIVE'
  },
  {
    symbol: 'SOL-USD',
    name: 'Solana',
    minOrderSize: 0.01,
    maxOrderSize: 10000,
    pricePrecision: 2,
    quantityPrecision: 4,
    status: 'ACTIVE'
  }
];

// Mock portfolio data
export const mockPortfolio = {
  totalValue: 25000,
  availableCash: 10000,
  positions: [
    {
      symbol: 'BTC-USD',
      quantity: 0.2,
      averageEntryPrice: 48000,
      currentPrice: 50000,
      marketValue: 10000,
      unrealizedPnL: 400,
      unrealizedPnLPercent: 4,
      allocation: 40
    },
    {
      symbol: 'ETH-USD',
      quantity: 1.5,
      averageEntryPrice: 3300,
      currentPrice: 3500,
      marketValue: 5250,
      unrealizedPnL: 300,
      unrealizedPnLPercent: 6.06,
      allocation: 21
    }
  ]
};

// Mock order book
export const mockOrderBook = {
  bids: [
    { price: 49800, size: 1.5 },
    { price: 49750, size: 2.3 },
    { price: 49700, size: 3.1 },
    { price: 49650, size: 1.8 },
    { price: 49600, size: 4.2 }
  ],
  asks: [
    { price: 50000, size: 1.2 },
    { price: 50050, size: 2.0 },
    { price: 50100, size: 1.5 },
    { price: 50150, size: 3.0 },
    { price: 50200, size: 2.5 }
  ]
};

// Mock recent trades
export const mockRecentTrades = [
  { id: 'trade1', price: 49950, size: 0.1, side: 'buy', time: new Date(Date.now() - 30000).toISOString() },
  { id: 'trade2', price: 49975, size: 0.05, side: 'sell', time: new Date(Date.now() - 25000).toISOString() },
  { id: 'trade3', price: 50000, size: 0.2, side: 'buy', time: new Date(Date.now() - 20000).toISOString() },
  { id: 'trade4', price: 50025, size: 0.15, side: 'buy', time: new Date(Date.now() - 15000).toISOString() },
  { id: 'trade5', price: 50000, size: 0.3, side: 'sell', time: new Date(Date.now() - 10000).toISOString() }
];

// Mock open orders
export const mockOpenOrders = [
  {
    id: 'open-ord1',
    clientOrderId: 'client-open-ord1',
    symbol: 'BTC-USD',
    side: 'sell',
    type: 'LIMIT',
    status: 'OPEN',
    quantity: 0.05,
    price: 52000,
    filledQuantity: 0,
    filledPrice: null,
    createdAt: new Date(Date.now() - 3600000).toISOString(),
    updatedAt: new Date(Date.now() - 3600000).toISOString()
  },
  {
    id: 'open-ord2',
    clientOrderId: 'client-open-ord2',
    symbol: 'ETH-USD',
    side: 'buy',
    type: 'LIMIT',
    status: 'OPEN',
    quantity: 0.5,
    price: 3300,
    filledQuantity: 0,
    filledPrice: null,
    createdAt: new Date(Date.now() - 1800000).toISOString(),
    updatedAt: new Date(Date.now() - 1800000).toISOString()
  }
];

// Mock order history
export const mockOrderHistory = [
  {
    id: 'hist-ord1',
    clientOrderId: 'client-hist-ord1',
    symbol: 'BTC-USD',
    side: 'buy',
    type: 'MARKET',
    status: 'FILLED',
    quantity: 0.1,
    price: null,
    filledQuantity: 0.1,
    filledPrice: 49800,
    createdAt: new Date(Date.now() - 86400000).toISOString(),
    updatedAt: new Date(Date.now() - 86400000).toISOString(),
    fees: 24.9,
    totalValue: 4980
  },
  {
    id: 'hist-ord2',
    clientOrderId: 'client-hist-ord2',
    symbol: 'ETH-USD',
    side: 'sell',
    type: 'LIMIT',
    status: 'FILLED',
    quantity: 0.2,
    price: 3450,
    filledQuantity: 0.2,
    filledPrice: 3450,
    createdAt: new Date(Date.now() - 172800000).toISOString(),
    updatedAt: new Date(Date.now() - 172800000).toISOString(),
    fees: 3.45,
    totalValue: 690
  },
  {
    id: 'hist-ord3',
    clientOrderId: 'client-hist-ord3',
    symbol: 'SOL-USD',
    side: 'buy',
    type: 'LIMIT',
    status: 'CANCELED',
    quantity: 5,
    price: 140,
    filledQuantity: 0,
    filledPrice: null,
    createdAt: new Date(Date.now() - 259200000).toISOString(),
    updatedAt: new Date(Date.now() - 259100000).toISOString(),
    fees: 0,
    totalValue: 0
  }
];

// Mock API errors
export const mockApiErrors = {
  networkError: {
    message: 'Network Error',
    status: 0,
    code: 'NETWORK_ERROR'
  },
  authError: {
    message: 'Authentication failed',
    status: 401,
    code: 'UNAUTHORIZED'
  },
  insufficientFunds: {
    message: 'Insufficient funds',
    status: 400,
    code: 'INSUFFICIENT_FUNDS'
  },
  invalidOrder: {
    message: 'Invalid order parameters',
    status: 400,
    code: 'INVALID_ORDER'
  },
  rateLimitExceeded: {
    message: 'Rate limit exceeded',
    status: 429,
    code: 'RATE_LIMIT_EXCEEDED'
  },
  serverError: {
    message: 'Internal server error',
    status: 500,
    code: 'SERVER_ERROR'
  }
};
