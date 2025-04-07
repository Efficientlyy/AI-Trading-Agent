"""
Moving Average Crossover Strategy module for AI Trading Agent.

This module provides a simple moving average crossover strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from src.trading_engine.models import Order, Portfolio
from src.trading_engine.enums import OrderSide, OrderType
from src.strategies.base_strategy import BaseStrategy
from src.rust_integration.indicators import calculate_sma
from src.common import logger


class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    This strategy generates buy signals when the fast moving average crosses above
    the slow moving average, and sell signals when the fast moving average crosses
    below the slow moving average.
    """
    
    def __init__(
        self,
        symbols: List[str],
        fast_period: int = 10,
        slow_period: int = 30,
        risk_pct: float = 0.02,
        max_position_pct: float = 0.2,
        name: str = "MA Crossover",
    ):
        """
        Initialize the MA Crossover strategy.
        
        Args:
            symbols: List of symbols to trade
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            risk_pct: Percentage of portfolio to risk on each trade
            max_position_pct: Maximum percentage of portfolio for a single position
            name: Name of the strategy
        """
        parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "risk_pct": risk_pct,
            "max_position_pct": max_position_pct,
        }
        
        super().__init__(symbols, parameters, name)
        
        # Initialize indicators dictionary
        self.indicators = {}
    
    def _initialize_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize technical indicators for the strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
        """
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        
        for symbol in self.symbols:
            if symbol in data:
                # Calculate fast and slow moving averages
                close_prices = data[symbol]["close"]
                
                fast_ma = calculate_sma(close_prices, fast_period)
                slow_ma = calculate_sma(close_prices, slow_period)
                
                # Store indicators
                self.indicators[symbol] = {
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                }
                
                logger.info(f"Initialized indicators for {symbol}: fast_ma({fast_period}), slow_ma({slow_period})")
    
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> List[Order]:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            portfolio: Current portfolio state
            timestamp: Current timestamp (optional)
            
        Returns:
            List[Order]: List of orders to execute
        """
        orders = []
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        risk_pct = self.parameters["risk_pct"]
        max_position_pct = self.parameters["max_position_pct"]
        
        for symbol in self.symbols:
            if symbol not in data:
                continue
            
            # Get current data
            df = data[symbol]
            if len(df) < slow_period + 1:
                logger.warning(f"Not enough data for {symbol} to generate signals")
                continue
            
            # Calculate moving averages
            close_prices = df["close"]
            
            fast_ma = calculate_sma(close_prices, fast_period)
            slow_ma = calculate_sma(close_prices, slow_period)
            
            # Store updated indicators
            self.indicators[symbol] = {
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
            }
            
            # Check for crossover
            if len(fast_ma) < 2 or len(slow_ma) < 2:
                continue
            
            # Get current and previous values
            current_fast = fast_ma[-1]
            current_slow = slow_ma[-1]
            prev_fast = fast_ma[-2]
            prev_slow = slow_ma[-2]
            
            # Check for crossover
            bullish_crossover = prev_fast < prev_slow and current_fast > current_slow
            bearish_crossover = prev_fast > prev_slow and current_fast < current_slow
            
            # Get current position
            current_position = 0
            if symbol in portfolio.positions:
                current_position = portfolio.positions[symbol].quantity
            
            # Get current price
            current_price = df["close"].iloc[-1]
            
            # Generate signals
            if bullish_crossover and current_position <= 0:
                # Calculate position size
                position_size = self.calculate_position_size(
                    symbol, portfolio, current_price, risk_pct, max_position_pct
                )
                
                # Create buy order
                if position_size > 0:
                    order = self.create_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=position_size,
                        order_type=OrderType.MARKET,
                    )
                    
                    orders.append(order)
                    logger.info(f"Generated BUY signal for {symbol} at {current_price:.2f}")
            
            elif bearish_crossover and current_position > 0:
                # Create sell order
                order = self.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=current_position,
                    order_type=OrderType.MARKET,
                )
                
                orders.append(order)
                logger.info(f"Generated SELL signal for {symbol} at {current_price:.2f}")
        
        return orders
