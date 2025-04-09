// Authentication types
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

// Trading and portfolio types
export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  realized_pnl: number;
}

export interface Portfolio {
  cash: number;
  total_value: number;
  positions: Record<string, Position>;
  daily_pnl?: number;
  margin_multiplier?: number;
}

export interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  order_type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stop_price?: number;
  status: 'pending' | 'filled' | 'partially_filled' | 'canceled' | 'rejected';
  filled_quantity: number;
  created_at: string;
  updated_at: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: number | string;
  status: 'pending' | 'filled' | 'partial' | 'cancelled' | 'rejected';
  fee?: number;
  total?: number;
}

export interface OrderRequest {
  symbol: string;
  side: 'buy' | 'sell';
  order_type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stop_price?: number;
}

// Performance metrics types
export interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  volatility: number;
  beta: number;
  alpha: number;
}

// Backtest types
export interface BacktestParams {
  strategy_name: string;
  parameters: Record<string, any>;
  start_date: string;
  end_date: string;
  initial_capital: number;
  symbols: string[];
}

export interface BacktestResult {
  id: string;
  params: BacktestParams;
  metrics: PerformanceMetrics;
  equity_curve: Array<{ timestamp: string; equity: number }>;
  trades: Order[];
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
}

// Sentiment types
export interface SentimentSignal {
  signal: 'buy' | 'sell' | 'hold';
  strength: number;
}

// Strategy types
export interface Strategy {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, any>;
  created_at: string;
  updated_at: string;
}

// Asset types
export interface Asset {
  symbol: string;
  name: string;
  type: 'crypto' | 'stock' | 'forex' | 'commodity';
  price: number;
  change_24h: number;
  volume_24h: number;
}

// Historical data types
export interface OHLCV {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface HistoricalDataRequest {
  symbol: string;
  start: string;
  end: string;
  timeframe: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w';
}

// WebSocket subscription types
export type TopicType = 'portfolio' | 'sentiment_signal' | 'performance';

export interface WebSocketMessage {
  action: 'subscribe' | 'unsubscribe';
  topic: TopicType;
}

export interface WebSocketUpdate {
  portfolio?: Portfolio;
  sentiment_signal?: Record<string, SentimentSignal>;
  performance?: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate?: number;
    profit_factor?: number;
    avg_trade?: number;
  };
}
