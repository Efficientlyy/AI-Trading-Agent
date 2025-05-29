/**
 * Trading and portfolio types
 */

export interface User {
  id: string;
  username: string;
  email: string;
  role?: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;
}

export interface AuthContextType {
  authState: AuthState;
  login: (email: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  clearError: () => void;
}

export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent?: number;
  realized_pnl?: number; // Added to fix errors with Position object usage
}

export interface Portfolio {
  cash: number;
  total_value: number;
  positions: Record<string, Position>;
  daily_pnl?: number;
  daily_pnl_percent?: number;
  margin_multiplier?: number; // Added for compatibility
}

export enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP = 'STOP',
  STOP_LIMIT = 'STOP_LIMIT'
}

export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL'
}

export enum OrderStatus {
  NEW = 'NEW',
  FILLED = 'FILLED',
  PARTIALLY_FILLED = 'PARTIALLY_FILLED',
  CANCELED = 'CANCELED',
  REJECTED = 'REJECTED',
  PENDING = 'PENDING'
}

export interface Order {
  id: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  quantity: number;
  price?: number;
  stop_price?: number;
  limitPrice?: number; // Added for compatibility
  stopPrice?: number; // Added for compatibility
  status: OrderStatus;
  filled_quantity?: number;
  filledQuantity?: number; // Added for compatibility
  filled_price?: number;
  created_at: string | Date; // Changed to allow both string and Date
  createdAt?: Date; // For compatibility
  updated_at?: string | Date; // Changed to allow both string and Date
  updatedAt?: Date; // For compatibility
  clientOrderId?: string; // Added for compatibility
  timeInForce?: string; // Added for compatibility
  realized_pnl?: number; // Added for TradeStatistics
}

export interface OrderRequest {
  symbol: string;
  side: OrderSide | 'buy' | 'sell'; // Added string literals for compatibility
  type: OrderType;
  order_type?: string; // Added for compatibility with existing code
  quantity: number;
  price?: number;
  stop_price?: number;
}

export interface Trade {
  id: string;
  symbol: string;
  side: OrderSide | 'buy' | 'sell'; // Updated to support both enum and string literals
  quantity: number;
  price: number;
  timestamp: string | number; // Updated to support both string and number timestamps
  order_id?: string;
  commission?: number;
  pnl?: number;
  status?: string; // Added to fix errors with Trade.status property references
  realized_pnl?: number; // Added for compatibility with TradeStatistics
}

export interface Asset {
  symbol: string;
  name: string;
  type: string;
  price: number;
  change_24h: number;
  volume_24h: number;
}

// WebSocket types
export type TopicType = 'portfolio' | 'sentiment_signals' | 'recent_trades' | 'ohlcv';

export interface WebSocketMessage {
  action: 'subscribe' | 'unsubscribe';
  topic: TopicType;
  symbol?: string;
  timeframe?: string;
}

export interface OHLCV {
  timestamp: string;
  time?: number;  // Unix timestamp in seconds for chart libraries
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OHLCVLiveUpdate {
  topic: 'ohlcv';
  symbol: string;
  timeframe: string;
  data: OHLCV[] | OHLCV;
  timestamp: string;
}

export interface WebSocketUpdate {
  portfolio?: Portfolio;
  sentiment_signal?: Record<string, any>;
  performance?: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate?: number;
    profit_factor?: number;
    avg_trade?: number;
  };
  agent_status?: {
    status: 'running' | 'stopped' | 'error';
    reasoning: string;
    timestamp: string;
  };
  recent_trades?: Trade[];
  ohlcv?: OHLCVLiveUpdate;
}

// Sentiment types
export interface SentimentSignal {
  signal?: 'buy' | 'sell' | 'hold';  // In trading.ts
  signal_type?: 'buy' | 'sell' | 'hold'; // In sentiment.ts
  strength: 'low' | 'medium' | 'high' | number; // Support both string and number
  score?: number;
  symbol?: string;
  sources?: number;
  timestamp?: string;
}

// Strategy and backtesting types
export interface Strategy {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, any>;
  asset_class: string;
  timeframe: string;
  created_at: string;
  updated_at: string;
}

export interface BacktestParams {
  strategy_id?: string;
  strategy_name: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  symbol?: string;           // Single symbol option
  symbols?: string[];        // Multiple symbols option
  parameters?: Record<string, any>;
}

export interface BacktestResult {
  id: string;
  strategy_id: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_capital: number;
  total_return: number;
  annualized_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  trades: Order[];
  equity_curve: Array<{
    timestamp: string;
    equity: number;
  }>;
  parameters: Record<string, any>;
  metrics?: PerformanceMetrics; // Added for compatibility
  params?: {                   // Added for compatibility
    strategy_name: string;
    [key: string]: any;
  };
  created_at?: string;        // Added for compatibility
}

export interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
  max_consecutive_wins: number;
  max_consecutive_losses: number;
  avg_trade?: number;     // Added for compatibility with existing code
  volatility?: number;    // Added for compatibility
  beta?: number;          // Added for compatibility
  alpha?: number;         // Added for compatibility
}

export interface HistoricalDataRequest {
  symbol: string;
  timeframe: string;
  start?: string;  // Added for compatibility
  end?: string;    // Added for compatibility
  start_date?: string;
  end_date?: string;
  limit?: number;
}
