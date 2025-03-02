#!/usr/bin/env python3
"""
Dynamic Risk Limits Module

This module provides functionality for implementing adaptive risk limits
based on market conditions. It includes:
1. Market volatility-based position sizing
2. Adjustable drawdown protection
3. Circuit breakers for rapid market movements
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, Any

logger = logging.getLogger(__name__)

# Enums for risk limit types and status
class RiskLimitType(Enum):
    """Enum for different types of risk limits"""
    VOLATILITY_BASED = auto()
    DRAWDOWN = auto()
    CIRCUIT_BREAKER = auto()
    EXPOSURE = auto()
    CONCENTRATION = auto()
    
class RiskLimitStatus(Enum):
    """Enum for risk limit status"""
    NORMAL = auto()
    WARNING = auto()
    BREACHED = auto()

@dataclass
class RiskLimit:
    """Base class for risk limits"""
    limit_type: RiskLimitType
    threshold: float
    current_value: float = 0.0
    status: RiskLimitStatus = RiskLimitStatus.NORMAL
    last_updated: datetime = datetime.now()
    
    def check_status(self) -> RiskLimitStatus:
        """Check and update the status of this risk limit"""
        if self.current_value >= self.threshold:
            self.status = RiskLimitStatus.BREACHED
        elif self.current_value >= self.threshold * 0.8:
            self.status = RiskLimitStatus.WARNING
        else:
            self.status = RiskLimitStatus.NORMAL
        
        self.last_updated = datetime.now()
        return self.status

# Simple data classes for specific limit parameters
@dataclass
class VolatilityParameters:
    """Parameters for volatility-based limits"""
    lookback_period: int = 20  # days
    volatility_scale: float = 1.0  # multiplier for position sizing

@dataclass
class DrawdownParameters:
    """Parameters for drawdown limits"""
    max_drawdown_allowed: float  # as a percentage (e.g., 0.05 for 5%)
    recovery_threshold: float = 0.5  # threshold to reduce/reset limits after recovery

@dataclass
class CircuitBreakerParameters:
    """Parameters for circuit breaker limits"""
    time_window: timedelta  # time window to monitor
    price_move_threshold: float  # percentage move that triggers the breaker
    cooldown_period: timedelta  # time to wait before re-enabling trading
    last_triggered: Optional[datetime] = None

class DynamicRiskLimits:
    """
    DynamicRiskLimits manages a set of risk limits that adapt to changing 
    market conditions. It enforces various risk constraints and can 
    adjust position sizes and trading permissions based on current risk levels.
    """
    
    def __init__(self):
        self.risk_limits: Dict[str, List[RiskLimit]] = {}
        self.volatility_params: Dict[str, VolatilityParameters] = {}
        self.drawdown_params: Dict[str, DrawdownParameters] = {}
        self.circuit_breaker_params: Dict[str, CircuitBreakerParameters] = {}
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.portfolio_value_history: List[Tuple[datetime, float]] = []
        self.peak_portfolio_value: float = 0.0
    
    def add_volatility_limit(self, symbol: str, threshold: float, 
                             lookback_period: int = 20, 
                             volatility_scale: float = 1.0) -> None:
        """
        Add a volatility-based risk limit for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            threshold: Maximum allowed volatility
            lookback_period: Period in days to calculate volatility
            volatility_scale: Scaling factor for position sizing
        """
        limit = RiskLimit(
            limit_type=RiskLimitType.VOLATILITY_BASED,
            threshold=threshold
        )
        
        self.volatility_params[symbol] = VolatilityParameters(
            lookback_period=lookback_period,
            volatility_scale=volatility_scale
        )
        
        if symbol not in self.risk_limits:
            self.risk_limits[symbol] = []
            
        self.risk_limits[symbol].append(limit)
        logger.info(f"Added volatility limit for {symbol} with threshold {threshold}")
    
    def add_drawdown_limit(self, symbol: str, max_drawdown: float, 
                           recovery_threshold: float = 0.5) -> None:
        """
        Add a drawdown limit for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            max_drawdown: Maximum allowed drawdown as percentage (e.g., 0.05 for 5%)
            recovery_threshold: Percentage of max drawdown for recovery
        """
        limit = RiskLimit(
            limit_type=RiskLimitType.DRAWDOWN,
            threshold=max_drawdown
        )
        
        self.drawdown_params[symbol] = DrawdownParameters(
            max_drawdown_allowed=max_drawdown,
            recovery_threshold=recovery_threshold
        )
        
        if symbol not in self.risk_limits:
            self.risk_limits[symbol] = []
            
        self.risk_limits[symbol].append(limit)
        logger.info(f"Added drawdown limit for {symbol} with max drawdown {max_drawdown}")
    
    def add_circuit_breaker(self, symbol: str, price_move_threshold: float,
                           time_window_minutes: int = 15,
                           cooldown_minutes: int = 60) -> None:
        """
        Add a circuit breaker for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            price_move_threshold: Price move percentage that triggers the breaker
            time_window_minutes: Time window to monitor for price moves (minutes)
            cooldown_minutes: Cooldown period after triggering (minutes)
        """
        limit = RiskLimit(
            limit_type=RiskLimitType.CIRCUIT_BREAKER,
            threshold=price_move_threshold
        )
        
        self.circuit_breaker_params[symbol] = CircuitBreakerParameters(
            time_window=timedelta(minutes=time_window_minutes),
            price_move_threshold=price_move_threshold,
            cooldown_period=timedelta(minutes=cooldown_minutes)
        )
        
        if symbol not in self.risk_limits:
            self.risk_limits[symbol] = []
            
        self.risk_limits[symbol].append(limit)
        logger.info(f"Added circuit breaker for {symbol} with threshold {price_move_threshold}")
    
    def update_market_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Update market data cache for a symbol
        
        Args:
            symbol: Trading symbol
            data: Market data DataFrame with at least 'timestamp' and 'close' columns
        """
        self.market_data_cache[symbol] = data
        logger.debug(f"Updated market data for {symbol}, {len(data)} data points")
    
    def update_portfolio_value(self, value: float) -> None:
        """
        Update portfolio value history and track peak value
        
        Args:
            value: Current portfolio value
        """
        timestamp = datetime.now()
        self.portfolio_value_history.append((timestamp, value))
        
        # Keep only the last 1000 data points to manage memory
        if len(self.portfolio_value_history) > 1000:
            self.portfolio_value_history = self.portfolio_value_history[-1000:]
        
        # Update peak value if current value is higher
        if value > self.peak_portfolio_value:
            self.peak_portfolio_value = value
        
        logger.debug(f"Updated portfolio value: {value}, peak value: {self.peak_portfolio_value}")
    
    def calculate_current_drawdown(self) -> float:
        """
        Calculate current drawdown from peak portfolio value
        
        Returns:
            Current drawdown as a percentage
        """
        if not self.portfolio_value_history or self.peak_portfolio_value == 0:
            return 0.0
        
        current_value = self.portfolio_value_history[-1][1]
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        return drawdown
    
    def calculate_volatility(self, symbol: str) -> float:
        """
        Calculate current volatility for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current volatility as a percentage
        """
        if symbol not in self.market_data_cache:
            logger.warning(f"No market data available for {symbol}")
            return 0.0
        
        data = self.market_data_cache[symbol]
        
        if len(data) < 2:
            return 0.0
        
        # Calculate daily returns
        data['returns'] = data['close'].pct_change().dropna()
        
        # Calculate rolling volatility (standard deviation of returns)
        volatility = data['returns'].std()
        
        return volatility
    
    def check_circuit_breaker(self, symbol: str, current_price: float) -> bool:
        """
        Check if a circuit breaker should be triggered
        
        Args:
            symbol: Trading symbol
            current_price: Current price of the symbol
            
        Returns:
            True if circuit breaker is triggered, False otherwise
        """
        if (symbol not in self.risk_limits or
            symbol not in self.circuit_breaker_params or
            symbol not in self.market_data_cache):
            return False
        
        # Get the circuit breaker parameters
        cb_params = self.circuit_breaker_params[symbol]
        
        # Find the circuit breaker limit
        cb_limit = None
        for limit in self.risk_limits[symbol]:
            if limit.limit_type == RiskLimitType.CIRCUIT_BREAKER:
                cb_limit = limit
                break
                
        if not cb_limit:
            return False
            
        data = self.market_data_cache[symbol]
        
        if len(data) < 2:
            return False
        
        # Get data within the time window
        now = datetime.now()
        window_start = now - cb_params.time_window
        recent_data = data[data['timestamp'] >= window_start]
        
        if len(recent_data) < 2:
            return False
        
        # Calculate price move within the window
        first_price = recent_data.iloc[0]['close']
        price_move = abs(current_price - first_price) / first_price
        
        # Update current value
        cb_limit.current_value = price_move
        
        # Check if the circuit breaker should be triggered
        if price_move >= cb_params.price_move_threshold:
            # If already in cooldown period, check if it has ended
            if cb_params.last_triggered:
                cooldown_end = cb_params.last_triggered + cb_params.cooldown_period
                if now < cooldown_end:
                    logger.info(f"Circuit breaker for {symbol} in cooldown until {cooldown_end}")
                    return True
            
            # Trigger the circuit breaker
            cb_params.last_triggered = now
            cb_limit.status = RiskLimitStatus.BREACHED
            
            logger.warning(
                f"Circuit breaker triggered for {symbol}: price move {price_move:.2%} "
                f"exceeds threshold {cb_params.price_move_threshold:.2%}"
            )
            
            return True
    
        return False
    
    def get_position_size_multiplier(self, symbol: str) -> float:
        """
        Get multiplier for position sizing based on volatility
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position size multiplier (1.0 is standard, <1.0 reduces size, >1.0 increases size)
        """
        if (symbol not in self.risk_limits or
            symbol not in self.volatility_params):
            return 1.0
        
        volatility = self.calculate_volatility(symbol)
        vol_params = self.volatility_params[symbol]
        
        # Find the volatility limit
        vol_limit = None
        for limit in self.risk_limits[symbol]:
            if limit.limit_type == RiskLimitType.VOLATILITY_BASED:
                vol_limit = limit
                break
                
        if not vol_limit:
            return 1.0
        
        # Update current value
        vol_limit.current_value = volatility
        vol_limit.check_status()
        
        # Calculate position size multiplier inversely proportional to volatility
        multiplier = 1.0
        
        if volatility > 0:
            # Reduce position size as volatility increases
            target_vol = vol_limit.threshold / 2  # Target half the threshold as ideal
            multiplier = target_vol / volatility if volatility > 0 else 1.0
            
            # Apply scaling factor
            multiplier *= vol_params.volatility_scale
            
            # Cap the multiplier between 0.1 and 2.0
            multiplier = max(0.1, min(2.0, multiplier))
        
        logger.debug(
            f"Position size multiplier for {symbol}: {multiplier:.2f} "
            f"(volatility: {volatility:.2%}, threshold: {vol_limit.threshold:.2%})"
        )
        
        return multiplier
    
    def check_drawdown_limits(self) -> Tuple[bool, float]:
        """
        Check if drawdown limits are breached
        
        Returns:
            Tuple of (is_breached, current_drawdown_percentage)
        """
        current_drawdown = self.calculate_current_drawdown()
        
        # Check if any drawdown limits are breached
        for symbol, limits in self.risk_limits.items():
            for limit in limits:
                if limit.limit_type != RiskLimitType.DRAWDOWN:
                    continue
                
                # Update current value
                limit.current_value = current_drawdown
                status = limit.check_status()
                
                if status == RiskLimitStatus.BREACHED:
                    logger.warning(
                        f"Drawdown limit breached for {symbol}: current drawdown {current_drawdown:.2%} "
                        f"exceeds threshold {limit.threshold:.2%}"
                    )
                    return True, current_drawdown
                
                elif status == RiskLimitStatus.WARNING:
                    logger.info(
                        f"Drawdown warning for {symbol}: current drawdown {current_drawdown:.2%} "
                        f"approaching threshold {limit.threshold:.2%}"
                    )
        
        return False, current_drawdown
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all risk limits and their current status
        
        Returns:
            Dictionary with risk summary information
        """
        summary = {
            "portfolio": {
                "current_value": self.portfolio_value_history[-1][1] if self.portfolio_value_history else 0,
                "peak_value": self.peak_portfolio_value,
                "drawdown": self.calculate_current_drawdown()
            },
            "symbols": {}
        }
        
        for symbol, limits in self.risk_limits.items():
            symbol_summary = {
                "volatility": self.calculate_volatility(symbol),
                "position_size_multiplier": self.get_position_size_multiplier(symbol),
                "limits": []
            }
            
            for limit in limits:
                limit_summary = {
                    "type": limit.limit_type.name,
                    "threshold": limit.threshold,
                    "current_value": limit.current_value,
                    "status": limit.status.name,
                    "last_updated": limit.last_updated.isoformat()
                }
                
                # Add specific information for different limit types
                if limit.limit_type == RiskLimitType.CIRCUIT_BREAKER and symbol in self.circuit_breaker_params:
                    cb_params = self.circuit_breaker_params[symbol]
                    limit_summary["last_triggered"] = cb_params.last_triggered.isoformat() if cb_params.last_triggered else None
                    limit_summary["cooldown_period_minutes"] = cb_params.cooldown_period.total_seconds() / 60
                
                symbol_summary["limits"].append(limit_summary)
            
            summary["symbols"][symbol] = symbol_summary
        
        return summary 