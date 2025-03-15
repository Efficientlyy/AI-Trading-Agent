"""Risk management module for trading strategies.

This module provides risk management functionality including position sizing,
risk metrics calculation, and portfolio-level risk controls.
"""

from typing import Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    risk_reward_ratio: float
    time_in_trade: float  # in hours

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_exposure: float
    margin_used: float
    free_margin: float
    portfolio_var: float  # Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    current_drawdown: float
    risk_per_trade: float
    correlation_risk: float

class RiskManager:
    """Risk manager for trading strategies."""
    
    def __init__(
        self,
        initial_capital: float,
        max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
        max_position_risk: float = 0.01,    # 1% max position risk
        max_correlation_risk: float = 0.7,   # 70% max correlation
        target_risk_reward: float = 2.0     # 2:1 minimum RR ratio
    ):
        """Initialize risk manager.
        
        Args:
            initial_capital: Starting capital
            max_portfolio_risk: Maximum portfolio risk as decimal
            max_position_risk: Maximum position risk as decimal
            max_correlation_risk: Maximum correlation risk threshold
            target_risk_reward: Target risk/reward ratio
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation_risk = max_correlation_risk
        self.target_risk_reward = target_risk_reward
        
        self.positions: Dict[str, PositionRisk] = {}
        self.portfolio_risk = PortfolioRisk(
            total_exposure=0.0,
            margin_used=0.0,
            free_margin=initial_capital,
            portfolio_var=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            current_drawdown=0.0,
            risk_per_trade=0.0,
            correlation_risk=0.0
        )
        
        self.position_history: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]
        self.drawdown_history: List[float] = [0.0]
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        volatility: float,
        correlation_score: float,
        confidence_score: float
    ) -> float:
        """Calculate optimal position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            volatility: Current volatility
            correlation_score: Asset correlation score
            confidence_score: Strategy confidence score
            
        Returns:
            Optimal position size
        """
        # Calculate base position size from risk per trade
        risk_amount = self.current_capital * self.max_position_risk
        price_risk = abs(entry_price - stop_loss)
        base_size = risk_amount / price_risk
        
        # Adjust for volatility
        vol_factor = 1.0 - (volatility / 2.0)  # Reduce size in high volatility
        
        # Adjust for correlation
        corr_factor = 1.0 - (correlation_score / self.max_correlation_risk)
        
        # Adjust for confidence
        conf_factor = confidence_score
        
        # Calculate risk/reward ratio
        rr_ratio = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        rr_factor = min(rr_ratio / self.target_risk_reward, 1.5)
        
        # Calculate final position size
        position_size = (
            base_size *
            vol_factor *
            corr_factor *
            conf_factor *
            rr_factor
        )
        
        # Ensure position size doesn't exceed portfolio limits
        max_size = self.calculate_max_position_size()
        position_size = min(position_size, max_size)
        
        return position_size
    
    def calculate_max_position_size(self) -> float:
        """Calculate maximum allowed position size based on portfolio risk.
        
        Returns:
            Maximum position size
        """
        available_risk = (
            self.max_portfolio_risk * self.current_capital -
            self.portfolio_risk.total_exposure
        )
        return max(0, available_risk)
    
    def update_position_risk(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime
    ) -> None:
        """Update risk metrics for a position.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            timestamp: Current timestamp
        """
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        # Update unrealized P&L
        pos.current_price = current_price
        price_change = current_price - pos.entry_price
        pos.unrealized_pnl = price_change * pos.position_size
        
        # Update time in trade
        time_diff = timestamp - datetime.fromtimestamp(pos.time_in_trade)
        pos.time_in_trade = time_diff.total_seconds() / 3600  # Convert to hours
        
        # Update risk/reward ratio
        current_risk = abs(current_price - pos.stop_loss)
        current_reward = abs(pos.take_profit - current_price)
        pos.risk_reward_ratio = current_reward / current_risk if current_risk > 0 else 0
    
    def update_portfolio_risk(
        self,
        returns: List[float],
        correlations: Dict[str, float]
    ) -> None:
        """Update portfolio risk metrics.
        
        Args:
            returns: List of portfolio returns
            correlations: Dictionary of asset correlations
        """
        if not returns:
            return
        
        # Update Value at Risk (VaR)
        self.portfolio_risk.portfolio_var = self._calculate_var(returns)
        
        # Update drawdown metrics
        current_equity = self.current_capital
        peak_equity = max(self.equity_curve)
        self.portfolio_risk.current_drawdown = (
            (peak_equity - current_equity) / peak_equity
        )
        self.portfolio_risk.max_drawdown = max(
            self.portfolio_risk.max_drawdown,
            self.portfolio_risk.current_drawdown
        )
        
        # Update Sharpe ratio
        self.portfolio_risk.sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Update correlation risk
        self.portfolio_risk.correlation_risk = max(correlations.values())
        
        # Update exposure metrics
        total_exposure = sum(
            abs(pos.position_size * pos.current_price)
            for pos in self.positions.values()
        )
        self.portfolio_risk.total_exposure = total_exposure
        self.portfolio_risk.margin_used = total_exposure * 0.1  # Assume 10x leverage
        self.portfolio_risk.free_margin = (
            self.current_capital - self.portfolio_risk.margin_used
        )
    
    def _calculate_var(
        self,
        returns: List[float],
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Value at Risk.
        
        Args:
            returns: List of returns
            confidence_level: VaR confidence level
            
        Returns:
            Value at Risk
        """
        if not returns:
            return 0.0
        return float(-np.percentile(returns, (1 - confidence_level) * 100))
    
    def _calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: List of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if not returns:
            return 0.0
        
        excess_returns = float(np.mean(returns)) - (risk_free_rate / 252)  # Daily risk-free rate
        return float(excess_returns / (np.std(returns) + 1e-10))  # Avoid division by zero
    
    def should_close_position(
        self,
        symbol: str,
        current_price: float
    ) -> bool:
        """Check if a position should be closed based on risk metrics.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            
        Returns:
            True if position should be closed
        """
        if symbol not in self.positions:
            return False
        
        pos = self.positions[symbol]
        
        # Check stop loss
        if current_price <= pos.stop_loss:
            return True
        
        # Check take profit
        if current_price >= pos.take_profit:
            return True
        
        # Check time-based exit (e.g., close after 48 hours)
        if pos.time_in_trade > 48:
            return True
        
        # Check deteriorating risk/reward
        if pos.risk_reward_ratio < 1.0:
            return True
        
        return False
    
    def can_open_position(
        self,
        symbol: str,
        position_size: float,
        entry_price: float
    ) -> bool:
        """Check if a new position can be opened based on risk limits.
        
        Args:
            symbol: Trading symbol
            position_size: Proposed position size
            entry_price: Entry price
            
        Returns:
            True if position can be opened
        """
        # Check if we have enough free margin
        required_margin = position_size * entry_price * 0.1  # Assume 10x leverage
        if required_margin > self.portfolio_risk.free_margin:
            return False
        
        # Check if total exposure would exceed limits
        new_exposure = (
            self.portfolio_risk.total_exposure +
            position_size * entry_price
        )
        if new_exposure > self.current_capital * 2:  # Max 2x leverage
            return False
        
        # Check if we're already at max positions (e.g., 5)
        if len(self.positions) >= 5:
            return False
        
        return True 