"""
Risk Management and Position Sizing Utilities

Functions related to calculating position sizes, stop-losses, take-profits,
and other risk management techniques.
"""

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
        # Handle optimization failure, e.g., return equal weights or raise error
        print("Warning: Risk parity optimization failed. Returning equal weights.") # Or use logger
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
    # Ensure volatility is scaled relative to price if not already (e.g., ATR needs dividing by price)
    # Assuming volatility is already appropriately scaled or represents a percentage.
    # If volatility is absolute (like ATR), it should be scaled: volatility_scaled = volatility / price
    # size = risk_per_trade / volatility_scaled
    size = risk_per_trade / (volatility / price) # Assuming vol needs scaling like ATR

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
        # Stop should be below the highest price achieved
        stop_price = highest_price * (1 - trail_percent)
        # Ensure stop doesn't move down if price pulls back slightly
        # This logic might need adjustment based on specific trailing stop implementation
        # Typically, the stop only moves up for longs.
        # We return the calculated stop based on highest_price.
        return stop_price
    else: # short
        # Stop should be above the lowest price achieved
        stop_price = highest_price * (1 + trail_percent) # 'highest_price' is lowest price for shorts
        # Stop only moves down for shorts.
        return stop_price

def drawdown_based_position_reduction(current_drawdown: float, max_drawdown: float = 0.2, reduction_factor: float = 0.5) -> float:
    """
    Reduce position size if current drawdown exceeds max_drawdown.
    current_drawdown: current portfolio drawdown (0-1, e.g., 0.1 for 10%)
    max_drawdown: drawdown threshold (0-1)
    reduction_factor: fraction to reduce *new* position sizes by (e.g., 0.5 for 50%)
    Returns: position size multiplier (1.0 if no reduction, <1.0 if reduced)
    """
    if current_drawdown > max_drawdown:
        return reduction_factor
    else:
        return 1.0

