"""
Risk Management Module

This module provides sophisticated risk management capabilities for the backtesting framework,
including dynamic risk adjustment, drawdown-based position sizing, and correlation-aware exposure.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import math
import uuid
from collections import defaultdict

from ..common.utils import get_logger
from .position_manager import PositionManager, Position

class RiskManager:
    """
    Sophisticated risk management system with dynamic risk adjustment capabilities.
    
    Features:
    - Drawdown-based risk reduction
    - Volatility-adjusted position sizing
    - Correlation-aware exposure management
    - Risk budget allocation
    - Profit protection rules
    - Streak-based risk adjustment
    """
    
    def __init__(self, position_manager: PositionManager, config: Dict[str, Any] = None):
        """
        Initialize the risk manager.
        
        Args:
            position_manager: PositionManager instance to manage positions
            config: Configuration dictionary with risk parameters
        """
        self.logger = get_logger("RiskManager")
        self.position_manager = position_manager
        self.config = config or {}
        
        # Extract configuration
        self.enable_drawdown_protection = self.config.get("enable_drawdown_protection", True)
        self.drawdown_thresholds = self.config.get("drawdown_thresholds", [
            {"threshold": 5.0, "reduction": 20.0},   # 5% drawdown -> 20% risk reduction
            {"threshold": 10.0, "reduction": 50.0},  # 10% drawdown -> 50% risk reduction
            {"threshold": 15.0, "reduction": 75.0},  # 15% drawdown -> 75% risk reduction
            {"threshold": 20.0, "reduction": 100.0}  # 20% drawdown -> 100% risk reduction (trading halt)
        ])
        
        self.enable_volatility_adjustment = self.config.get("enable_volatility_adjustment", True)
        self.volatility_lookback = self.config.get("volatility_lookback", 20)
        self.volatility_risk_factor = self.config.get("volatility_risk_factor", 1.0)
        
        self.enable_correlation_adjustment = self.config.get("enable_correlation_adjustment", True)
        self.correlation_lookback = self.config.get("correlation_lookback", 60)
        self.max_correlation_exposure = self.config.get("max_correlation_exposure", 10.0)
        
        self.enable_streak_adjustment = self.config.get("enable_streak_adjustment", True)
        self.winning_streak_increase = self.config.get("winning_streak_increase", 10.0)
        self.losing_streak_decrease = self.config.get("losing_streak_decrease", 20.0)
        self.max_streak_adjustment = self.config.get("max_streak_adjustment", 50.0)
        
        self.enable_profit_protection = self.config.get("enable_profit_protection", True)
        self.profit_protection_threshold = self.config.get("profit_protection_threshold", 20.0)
        self.profit_protection_reduction = self.config.get("profit_protection_reduction", 25.0)
        
        # State tracking
        self.current_drawdown = 0.0
        self.current_drawdown_risk_reduction = 0.0
        self.current_volatility_adjustment = 1.0
        self.correlation_matrix = pd.DataFrame()
        self.current_winning_streak = 0
        self.current_losing_streak = 0
        self.streak_risk_adjustment = 0.0
        self.symbol_volatilities = {}
        self.highest_equity = position_manager.initial_capital
        
        # Historical tracking
        self.risk_adjustments = []
        self.position_sizings = []
        
        self.logger.info("Risk Manager initialized with drawdown protection, volatility and correlation adjustment")
        
    def update_market_state(self, market_data: Dict[str, pd.DataFrame], current_date: datetime = None):
        """
        Update risk manager with current market data to adjust risk parameters.
        
        Args:
            market_data: Dictionary mapping symbols to DataFrame with market data
            current_date: Current simulation date
        """
        # Update position manager first
        self.position_manager.update_positions(market_data, current_date)
        
        # Get current equity
        if self.position_manager.equity_curve:
            current_equity = self.position_manager.equity_curve[-1]["equity"]
            
            # Update highest equity
            if current_equity > self.highest_equity:
                self.highest_equity = current_equity
                
            # Calculate current drawdown
            if self.highest_equity > 0:
                self.current_drawdown = (self.highest_equity - current_equity) / self.highest_equity * 100
            else:
                self.current_drawdown = 0.0
                
            # Update drawdown-based risk reduction
            self._update_drawdown_risk_reduction()
        
        # Update volatility-based adjustment if enabled
        if self.enable_volatility_adjustment:
            self._update_volatility_adjustment(market_data)
            
        # Update correlation matrix if enabled
        if self.enable_correlation_adjustment:
            self._update_correlation_matrix(market_data)
            
        # Record risk adjustment
        self._record_risk_adjustment(current_date)
        
    def _update_drawdown_risk_reduction(self):
        """Update risk reduction based on current drawdown."""
        if not self.enable_drawdown_protection:
            self.current_drawdown_risk_reduction = 0.0
            return
            
        # Find the appropriate risk reduction based on drawdown thresholds
        self.current_drawdown_risk_reduction = 0.0
        for threshold in self.drawdown_thresholds:
            if self.current_drawdown >= threshold["threshold"]:
                self.current_drawdown_risk_reduction = threshold["reduction"]
                
        if self.current_drawdown_risk_reduction > 0:
            self.logger.info(f"Drawdown protection activated: {self.current_drawdown:.2f}% drawdown -> "
                           f"{self.current_drawdown_risk_reduction:.2f}% risk reduction")
            
    def _update_volatility_adjustment(self, market_data: Dict[str, pd.DataFrame]):
        """
        Update volatility-based adjustment factors for each symbol.
        
        Args:
            market_data: Dictionary mapping symbols to DataFrame with market data
        """
        for symbol, df in market_data.items():
            if df.empty or len(df) < self.volatility_lookback:
                continue
                
            # Calculate historical volatility (standard deviation of returns)
            returns = df["close"].pct_change().dropna()
            if len(returns) >= self.volatility_lookback:
                current_vol = returns.tail(self.volatility_lookback).std() * (252 ** 0.5)  # Annualized
                
                # Calculate baseline if this is the first time
                if symbol not in self.symbol_volatilities:
                    self.symbol_volatilities[symbol] = {
                        "baseline": current_vol,
                        "current": current_vol,
                        "adjustment": 1.0
                    }
                else:
                    self.symbol_volatilities[symbol]["current"] = current_vol
                    
                # Calculate adjustment factor
                baseline = self.symbol_volatilities[symbol]["baseline"]
                if baseline > 0:
                    # Inverse relationship: higher volatility -> lower position size
                    adjustment = baseline / max(current_vol, 0.0001)
                    
                    # Apply volatility risk factor and cap the adjustment
                    adjustment = max(0.2, min(2.0, adjustment ** self.volatility_risk_factor))
                    self.symbol_volatilities[symbol]["adjustment"] = adjustment
        
        # Calculate average adjustment across all symbols
        if self.symbol_volatilities:
            adjustments = [data["adjustment"] for data in self.symbol_volatilities.values()]
            self.current_volatility_adjustment = sum(adjustments) / len(adjustments)
        else:
            self.current_volatility_adjustment = 1.0
            
    def _update_correlation_matrix(self, market_data: Dict[str, pd.DataFrame]):
        """
        Update correlation matrix between symbols.
        
        Args:
            market_data: Dictionary mapping symbols to DataFrame with market data
        """
        # Need at least two symbols for correlation
        if len(market_data) < 2:
            return
            
        # Extract close prices and align dates
        price_data = {}
        for symbol, df in market_data.items():
            if not df.empty and len(df) >= self.correlation_lookback:
                price_data[symbol] = df["close"]
                
        if len(price_data) < 2:
            return
            
        # Create DataFrame with aligned dates
        price_df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Calculate correlation matrix using recent data
        if len(returns_df) >= self.correlation_lookback:
            recent_returns = returns_df.tail(self.correlation_lookback)
            self.correlation_matrix = recent_returns.corr()
            
    def _record_risk_adjustment(self, date: datetime = None):
        """
        Record current risk adjustment factors.
        
        Args:
            date: Current date
        """
        if date is None:
            date = datetime.now()
            
        self.risk_adjustments.append({
            "date": date,
            "drawdown": self.current_drawdown,
            "drawdown_reduction": self.current_drawdown_risk_reduction,
            "volatility_adjustment": self.current_volatility_adjustment,
            "streak_adjustment": self.streak_risk_adjustment,
            "total_adjustment": self.get_total_risk_adjustment()
        })
        
    def calculate_position_size(self, symbol: str, direction: str, entry_price: float,
                              stop_loss: float, base_risk_pct: float) -> Tuple[float, float]:
        """
        Calculate adjusted position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            direction: Position direction ('long' or 'short')
            entry_price: Entry price
            stop_loss: Stop loss price
            base_risk_pct: Base risk percentage before adjustments
            
        Returns:
            Tuple of (adjusted_quantity, adjusted_risk_pct)
        """
        # Start with base risk
        adjusted_risk_pct = base_risk_pct
        
        # Apply total risk adjustment
        total_adjustment = self.get_total_risk_adjustment()
        adjusted_risk_pct = adjusted_risk_pct * (1 - total_adjustment / 100.0)
        
        # Apply symbol-specific volatility adjustment if available
        vol_adjustment = 1.0
        if symbol in self.symbol_volatilities and self.enable_volatility_adjustment:
            vol_adjustment = self.symbol_volatilities[symbol]["adjustment"]
            adjusted_risk_pct = adjusted_risk_pct * vol_adjustment
            
        # Apply correlation adjustment if needed
        if self.enable_correlation_adjustment:
            corr_adjustment = self._calculate_correlation_adjustment(symbol, direction)
            adjusted_risk_pct = adjusted_risk_pct * corr_adjustment
            
        # Ensure risk is not negative
        adjusted_risk_pct = max(0.0, adjusted_risk_pct)
        
        # Calculate quantity based on adjusted risk
        quantity = self._calculate_quantity(symbol, direction, entry_price, stop_loss, adjusted_risk_pct)
        
        # Record position sizing
        self._record_position_sizing(symbol, base_risk_pct, adjusted_risk_pct, 
                                   total_adjustment, vol_adjustment)
        
        return quantity, adjusted_risk_pct
        
    def _calculate_quantity(self, symbol: str, direction: str, entry_price: float,
                          stop_loss: float, risk_pct: float) -> float:
        """
        Calculate position quantity based on risk percentage.
        
        Args:
            symbol: Trading symbol
            direction: Position direction ('long' or 'short')
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_pct: Risk percentage
            
        Returns:
            Position quantity
        """
        # Use position manager's calculation method
        return self.position_manager._calculate_position_size(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_pct=risk_pct
        )
        
    def _calculate_correlation_adjustment(self, symbol: str, direction: str) -> float:
        """
        Calculate adjustment factor based on correlation with existing positions.
        
        Args:
            symbol: Trading symbol
            direction: Position direction ('long' or 'short')
            
        Returns:
            Correlation adjustment factor (0.0-1.0)
        """
        # If correlation matrix is empty or symbol is not in it, no adjustment
        if self.correlation_matrix.empty or symbol not in self.correlation_matrix.index:
            return 1.0
            
        # Get open positions
        open_positions = self.position_manager.get_open_positions()
        
        # If no open positions, no adjustment needed
        if not open_positions:
            return 1.0
            
        # Calculate average correlation with existing positions
        correlations = []
        for position in open_positions:
            pos_symbol = position.symbol
            
            # Skip if it's the same symbol or not in correlation matrix
            if pos_symbol == symbol or pos_symbol not in self.correlation_matrix.index:
                continue
                
            # Get correlation
            correlation = self.correlation_matrix.loc[symbol, pos_symbol]
            
            # Adjust correlation based on direction
            # For opposite directions, we want negative correlation to be good
            if position.direction != direction:
                correlation = -correlation
                
            correlations.append(correlation)
            
        # If no valid correlations, no adjustment
        if not correlations:
            return 1.0
            
        # Calculate average correlation
        avg_correlation = sum(correlations) / len(correlations)
        
        # Convert to adjustment factor:
        # High positive correlation -> reduce size
        # Negative correlation -> increase size (up to a limit)
        if avg_correlation > 0:
            # Linear reduction: 0 to 0.8 correlation -> 1.0 to 0.2 adjustment
            adjustment = max(0.2, 1.0 - avg_correlation)
        else:
            # Negative correlation bonus: max 30% increase
            adjustment = min(1.3, 1.0 + abs(avg_correlation) * 0.3)
            
        return adjustment
        
    def _record_position_sizing(self, symbol: str, base_risk: float, adjusted_risk: float,
                              total_adjustment: float, vol_adjustment: float):
        """
        Record position sizing data for analysis.
        
        Args:
            symbol: Trading symbol
            base_risk: Base risk percentage
            adjusted_risk: Adjusted risk percentage
            total_adjustment: Total risk adjustment percentage
            vol_adjustment: Volatility adjustment factor
        """
        self.position_sizings.append({
            "date": datetime.now(),
            "symbol": symbol,
            "base_risk_pct": base_risk,
            "adjusted_risk_pct": adjusted_risk,
            "total_adjustment_pct": total_adjustment,
            "volatility_adjustment": vol_adjustment,
            "drawdown": self.current_drawdown,
            "drawdown_reduction": self.current_drawdown_risk_reduction
        })
        
    def get_total_risk_adjustment(self) -> float:
        """
        Get total risk adjustment percentage.
        
        Returns:
            Total risk adjustment percentage (0-100)
        """
        # Start with drawdown-based reduction
        total_reduction = self.current_drawdown_risk_reduction
        
        # Add streak-based adjustment (can be positive or negative)
        if self.enable_streak_adjustment:
            total_reduction -= self.streak_risk_adjustment
            
        # Add profit protection if enabled
        if self.enable_profit_protection and self.position_manager.equity_curve:
            # Calculate profit percentage
            initial_capital = self.position_manager.initial_capital
            current_equity = self.position_manager.equity_curve[-1]["equity"]
            profit_pct = ((current_equity / initial_capital) - 1) * 100
            
            # Apply profit protection reduction if threshold reached
            if profit_pct >= self.profit_protection_threshold:
                profit_reduction = self.profit_protection_reduction
                total_reduction += profit_reduction
                
        # Ensure total reduction is within bounds (0-100%)
        return max(0.0, min(100.0, total_reduction))
        
    def update_trade_streak(self, trade_result: float):
        """
        Update trading streak and adjust risk based on win/loss streaks.
        
        Args:
            trade_result: P&L from the trade (positive for win, negative for loss)
        """
        if not self.enable_streak_adjustment:
            return
            
        # Update streaks
        if trade_result > 0:
            # Winning trade
            self.current_winning_streak += 1
            self.current_losing_streak = 0
        elif trade_result < 0:
            # Losing trade
            self.current_losing_streak += 1
            self.current_winning_streak = 0
        else:
            # Breakeven trade, no streak change
            return
            
        # Calculate streak-based risk adjustment
        if self.current_winning_streak >= 2:
            # Increase risk after winning streak
            streak_adjustment = min(
                self.max_streak_adjustment,
                self.winning_streak_increase * (self.current_winning_streak - 1)
            )
        elif self.current_losing_streak >= 2:
            # Decrease risk after losing streak (negative adjustment)
            streak_adjustment = -min(
                self.max_streak_adjustment,
                self.losing_streak_decrease * (self.current_losing_streak - 1)
            )
        else:
            streak_adjustment = 0.0
            
        self.streak_risk_adjustment = streak_adjustment
        
        if streak_adjustment != 0:
            self.logger.info(f"Streak adjustment updated: {streak_adjustment:.2f}% "
                           f"(W{self.current_winning_streak}/L{self.current_losing_streak})")
            
    def process_position_closed(self, position: Position):
        """
        Process a closed position to update risk parameters.
        
        Args:
            position: The closed Position object
        """
        # Update trade streak
        self.update_trade_streak(position.realized_pnl)
        
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            "current_drawdown": self.current_drawdown,
            "drawdown_risk_reduction": self.current_drawdown_risk_reduction,
            "volatility_adjustment": self.current_volatility_adjustment,
            "streak_adjustment": self.streak_risk_adjustment,
            "total_risk_adjustment": self.get_total_risk_adjustment(),
            "winning_streak": self.current_winning_streak,
            "losing_streak": self.current_losing_streak,
            "highest_equity": self.highest_equity
        }
        
    def get_risk_adjustment_history(self) -> pd.DataFrame:
        """
        Get history of risk adjustments.
        
        Returns:
            DataFrame with risk adjustment history
        """
        return pd.DataFrame(self.risk_adjustments).set_index("date") if self.risk_adjustments else pd.DataFrame()
        
    def get_position_sizing_history(self) -> pd.DataFrame:
        """
        Get history of position sizing decisions.
        
        Returns:
            DataFrame with position sizing history
        """
        return pd.DataFrame(self.position_sizings).set_index("date") if self.position_sizings else pd.DataFrame()
        
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get the current correlation matrix.
        
        Returns:
            DataFrame with correlation matrix
        """
        return self.correlation_matrix.copy()
        
    def should_halt_trading(self) -> Tuple[bool, str]:
        """
        Check if trading should be halted based on risk rules.
        
        Returns:
            Tuple of (should_halt, reason)
        """
        # Check drawdown protection
        if self.enable_drawdown_protection:
            for threshold in self.drawdown_thresholds:
                if threshold["reduction"] >= 100.0 and self.current_drawdown >= threshold["threshold"]:
                    return True, f"Maximum drawdown reached: {self.current_drawdown:.2f}% >= {threshold['threshold']}%"
                    
        return False, "Trading allowed"
