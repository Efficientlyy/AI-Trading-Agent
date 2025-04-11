"""
Python implementation of the backtesting models from the Rust extension.
This serves as a temporary replacement until the Rust extension build issues are resolved.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime


class OrderSide(Enum):
    """Order side enum (Buy or Sell)"""
    Buy = auto()
    Sell = auto()


class OrderType(Enum):
    """Order type enum"""
    Market = auto()
    Limit = auto()
    Stop = auto()
    StopLimit = auto()


class OrderStatus(Enum):
    """Order status enum"""
    Created = auto()
    Submitted = auto()
    Partial = auto()
    Filled = auto()
    Canceled = auto()
    Rejected = auto()


@dataclass
class Fill:
    """Fill struct representing a partial or complete order execution"""
    quantity: float
    price: float
    timestamp: int  # Unix timestamp


@dataclass
class Order:
    """Order struct representing a trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.Created
    fills: List[Fill] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(datetime.now().timestamp()))


@dataclass
class Trade:
    """Trade struct representing a completed trade"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: int  # Unix timestamp


@dataclass
class Position:
    """Position struct representing a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    market_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Portfolio:
    """Portfolio struct representing the full trading portfolio"""
    cash: float
    total_value: float
    positions: Dict[str, Position] = field(default_factory=dict)


@dataclass
class PortfolioSnapshot:
    """PortfolioSnapshot struct for recording portfolio state at a point in time"""
    timestamp: int  # Unix timestamp
    cash: float
    total_value: float
    positions: Dict[str, Position]


@dataclass
class OHLCVBar:
    """OHLCV bar struct for price data"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestConfig:
    """BacktestConfig struct for configuring the backtest"""
    initial_capital: float
    commission_rate: float
    slippage: float
    enable_fractional: bool


@dataclass
class PerformanceMetrics:
    """PerformanceMetrics struct containing calculated performance metrics"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_profit_per_trade: float = 0.0
    avg_loss_per_trade: float = 0.0
    avg_profit_loss_ratio: float = 0.0


@dataclass
class BacktestResult:
    """BacktestResult struct containing all backtest results"""
    portfolio_history: List[PortfolioSnapshot]
    trade_history: List[Trade]
    order_history: List[Order]
    metrics: PerformanceMetrics
