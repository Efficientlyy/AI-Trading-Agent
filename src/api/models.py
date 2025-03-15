"""Data models for the Market Regime Detection API."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

class DataPoint(BaseModel):
    """Single data point for market data."""
    date: datetime
    price: float
    volume: Optional[float] = None
    return_value: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    spread: Optional[float] = None

class MarketData(BaseModel):
    """Market data for a single symbol."""
    symbol: str
    data: List[DataPoint]
    
class RegimeRequest(BaseModel):
    """Request model for regime detection."""
    market_data: Union[MarketData, List[MarketData]]
    methods: Optional[List[str]] = Field(default_factory=lambda: [
        "volatility", "momentum", "hmm"
    ])
    lookback_window: Optional[int] = 63
    include_statistics: Optional[bool] = True
    include_visualization: Optional[bool] = False

class BacktestRequest(BaseModel):
    """Request model for backtesting."""
    market_data: Union[MarketData, List[MarketData]]
    strategy_type: str = "trend_following"
    regime_methods: List[str] = Field(default_factory=lambda: [
        "volatility", "momentum"
    ])
    n_regimes: Optional[int] = 3
    initial_capital: float = 10000.0
    position_sizing: str = "fixed"
    max_position_size: float = 1.0
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    include_transaction_costs: bool = True
    train_test_split: Optional[float] = 0.7
    walk_forward: Optional[bool] = False
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class AnalysisResponse(BaseModel):
    """Response model for regime detection."""
    request_id: str
    symbol: str
    regimes: Dict[str, List[int]]
    statistics: Optional[Dict[str, Any]] = None
    visualization_urls: Optional[Dict[str, str]] = None
    execution_time: float

class BacktestResponse(BaseModel):
    """Response model for backtesting."""
    request_id: str
    symbol: str
    strategy: str
    performance_metrics: Dict[str, float]
    regime_metrics: Optional[Dict[str, Dict[str, float]]] = None
    equity_curve_url: Optional[str] = None
    regime_chart_url: Optional[str] = None
    trades: Optional[List[Dict[str, Any]]] = None
    execution_time: float 