from typing import Dict, List, Any, Optional
import pandas as pd
from ..trading_engine.models import Order
from ai_trading_agent.strategies.base_strategy import BaseStrategy
from ..trading_engine.enums import OrderSide, OrderType
import numpy as np
from scipy.optimize import minimize

def kelly_position_size(win_rate: float, payoff_ratio: float, max_fraction: float = 1.0) -> float:
    """
    Calculate the Kelly criterion position size.
    win_rate: Probability of winning (0-1)
    payoff_ratio: Average win / average loss
    max_fraction: Maximum allowed fraction (risk cap)
    Returns: Optimal fraction of capital to allocate (0-1)
    """
    if payoff_ratio <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    kelly = win_rate - (1 - win_rate) / payoff_ratio
    return max(0.0, min(kelly, max_fraction))

def dynamic_stop_loss(price: float, atr: float, atr_multiplier: float = 2.0, direction: str = "long") -> float:
    """
    Calculate a dynamic stop-loss based on ATR.
    price: current price
    atr: average true range
    atr_multiplier: multiplier for ATR
    direction: "long" or "short"
    Returns: stop-loss price
    """
    if direction == "long":
        return price - atr * atr_multiplier
    else:
        return price + atr * atr_multiplier

def take_profit_optimization(price: float, stop_loss: float, risk_reward: float = 2.0, direction: str = "long") -> float:
    """
    Calculate a take-profit level based on risk/reward ratio.
    price: current price
    stop_loss: stop-loss price
    risk_reward: desired risk/reward ratio (e.g., 2.0)
    direction: "long" or "short"
    Returns: take-profit price
    """
    risk = abs(price - stop_loss)
    if direction == "long":
        return price + risk * risk_reward
    else:
        return price - risk * risk_reward

def risk_parity_weights(cov_matrix: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """
    Compute risk parity weights for a given covariance matrix.
    Returns: np.ndarray of weights summing to 1.
    """
    n = cov_matrix.shape[0]
    x0 = np.ones(n) / n

    def risk_contribution(weights):
        portfolio_var = weights @ cov_matrix @ weights
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_var)
        return risk_contrib

    def objective(weights):
        rc = risk_contribution(weights)
        return np.sum((rc - np.mean(rc)) ** 2)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(objective, x0, bounds=bounds, constraints=constraints, tol=tol, options={'maxiter': max_iter})
    if not result.success:
        return x0
    return result.x

def volatility_adjusted_position_size(volatility: float, risk_per_trade: float = 0.01, price: float = 1.0, max_fraction: float = 1.0) -> float:
    """
    Calculate position size inversely proportional to volatility.
    volatility: e.g., ATR or rolling std
    risk_per_trade: fraction of capital to risk per trade
    price: current price (for scaling)
    max_fraction: maximum allowed fraction (risk cap)
    Returns: Fraction of capital to allocate (0-1)
    """
    if volatility <= 0 or price <= 0:
        return 0.0
    size = risk_per_trade / (volatility / price)
    return min(size, max_fraction)

def trailing_stop(price: float, highest_price: float, trail_percent: float = 0.05, direction: str = "long") -> float:
    """
    Calculate a trailing stop price.
    price: current price
    highest_price: highest price since entry (for long), lowest for short
    trail_percent: trailing stop as a fraction (e.g., 0.05 for 5%)
    direction: "long" or "short"
    Returns: stop-loss price
    """
    if direction == "long":
        return highest_price * (1 - trail_percent)
    else:
        return highest_price * (1 + trail_percent)

def drawdown_based_position_reduction(current_drawdown: float, max_drawdown: float = 0.2, reduction_factor: float = 0.5) -> float:
    """
    Reduce position size if current drawdown exceeds max_drawdown.
    current_drawdown: current portfolio drawdown (0-1)
    max_drawdown: drawdown threshold (0-1)
    reduction_factor: fraction to reduce position size by (e.g., 0.5 for 50%)
    Returns: position size multiplier (1.0 if no reduction, <1.0 if reduced)
    """
    if current_drawdown > max_drawdown:
        return reduction_factor
    return 1.0

class SentimentStrategy(BaseStrategy):
    """
    Abstract base class for sentiment-based trading strategies.
    """
    def _initialize_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        pass

    def risk_parity_allocation(self, cov_matrix: np.ndarray) -> np.ndarray:
        return risk_parity_weights(cov_matrix)

    def volatility_adjusted_position_size(self, volatility: float, risk_per_trade: float = 0.01, price: float = 1.0, max_fraction: float = 1.0) -> float:
        return volatility_adjusted_position_size(volatility, risk_per_trade, price, max_fraction)

    def kelly_position_size(self, win_rate: float, payoff_ratio: float, max_fraction: float = 1.0) -> float:
        return kelly_position_size(win_rate, payoff_ratio, max_fraction)

    def dynamic_stop_loss(self, price: float, atr: float, atr_multiplier: float = 2.0, direction: str = "long") -> float:
        return dynamic_stop_loss(price, atr, atr_multiplier, direction)

    def take_profit_optimization(self, price: float, stop_loss: float, risk_reward: float = 2.0, direction: str = "long") -> float:
        return take_profit_optimization(price, stop_loss, risk_reward, direction)

    def trailing_stop(self, price: float, highest_price: float, trail_percent: float = 0.05, direction: str = "long") -> float:
        return trailing_stop(price, highest_price, trail_percent, direction)

    def drawdown_based_position_reduction(self, current_drawdown: float, max_drawdown: float = 0.2, reduction_factor: float = 0.5) -> float:
        return drawdown_based_position_reduction(current_drawdown, max_drawdown, reduction_factor)

class DummySentimentStrategy(SentimentStrategy):
    """
    Concrete dummy implementation of SentimentStrategy to fix instantiation errors.
    """
    def __init__(self, symbols: Optional[List[str]] = None, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(symbols or [], parameters or {}, "DummySentimentStrategy")
        self.source_weights = (parameters or {}).get('source_weights', {})
        self.trade_history = []

    def _initialize_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        # No-op dummy implementation
        pass

    def generate_orders(self, signals: pd.DataFrame, timestamp: pd.Timestamp,
                       current_positions: Dict[str, Any]) -> List[Order]:
        # Dummy implementation returns a mock order if there's a signal
        orders = []
        if not signals.empty and 'signal' in signals.columns:
            for idx, row in signals.iterrows():
                if row['signal'] != 0:
                    # Create a basic mock order
                    orders.append(Order(
                        order_id=f"dummy_{row['symbol']}_{timestamp.isoformat()}",
                        symbol=row['symbol'],
                        side=OrderSide.BUY if row['signal'] > 0 else OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=row.get('position_size', 0.01), # Use position_size if available
                        timestamp=timestamp
                    ))
        return orders

    def generate_signals(
        self,
        data: Any = None,
        portfolio: Any = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        # Dummy implementation returns a simple DataFrame with signals
        if data is not None and 'symbol' in data.columns:
            signals_df = pd.DataFrame(index=data.index)
            signals_df['symbol'] = data['symbol']
            # Generate alternating signals for testing
            signals_df['signal'] = [1 if i % 2 == 0 else -1 for i in range(len(data))]
            signals_df['position_size'] = 0.1 # Dummy size
            signals_df['signal_strength'] = 0.8 # Dummy strength
            signals_df['stop_loss'] = data['close'] * 0.95 # Dummy SL
            signals_df['take_profit'] = data['close'] * 1.05 # Dummy TP
            return signals_df
        else:
            return pd.DataFrame()

    def update_trade_history(self, trade_result: Dict[str, Any]):
        # Dummy implementation appends to history, limiting size
        self.trade_history.append(trade_result)
        max_history = self.parameters.get('max_history', 100)
        if len(self.trade_history) > max_history:
            self.trade_history = self.trade_history[-max_history:]
