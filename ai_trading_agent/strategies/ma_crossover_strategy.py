"""
Moving Average Crossover Strategy module for AI Trading Agent.

This module provides a simple moving average crossover strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from ..trading_engine.models import Order, Portfolio
from ..trading_engine.enums import OrderSide, OrderType
from ..agent.strategy import BaseStrategy, RichSignalsDict 
from ..rust_integration.indicators import calculate_sma
from ..common import logger


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
        # Consolidate parameters into a config dict for the new base class
        parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            # Keep risk params in config for potential future use, though not used for signal gen
            "risk_pct": risk_pct, 
            "max_position_pct": max_position_pct,
            "symbols": symbols # Store symbols in config as well
        }
        
        # Call the __init__ of the canonical BaseStrategy (name, config)
        super().__init__(name=name, config=parameters)
        
        # Keep symbols easily accessible if needed, though also in config
        self.symbols = symbols 
        
        # Initialize indicators dictionary
        self.indicators = {}
    
    def _initialize_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize technical indicators for the strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
        """
        # Retrieve periods from config
        fast_period = self.config.get("fast_period", 10)
        slow_period = self.config.get("slow_period", 30)
        
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
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Updates the strategy's configuration.

        Args:
            new_config: Dictionary containing parameters to update.
        """
        logger.info(f"Updating {self.name} config. Old: {self.config}, New partial: {new_config}")
        self.config.update(new_config)
        # Optionally, re-initialize indicators if parameters change
        # self._initialize_indicators(data) # Need access to data if re-initializing here
        logger.info(f"Updated {self.name} config: {self.config}")

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame], 
        current_portfolio: Optional[Dict[str, Any]] = None 
    ) -> RichSignalsDict:
        signals: RichSignalsDict = {}
        # Retrieve periods from config
        fast_period = self.config.get("fast_period", 10)
        slow_period = self.config.get("slow_period", 30)
        
        for symbol in self.symbols:
            # Default signal is HOLD (rich format)
            default_metadata = {
                'indicator': 'MA Crossover',
                'fast': fast_period,
                'slow': slow_period,
                'crossover_detected': False
            }
            signals[symbol] = {
                'signal_strength': 0.0,
                'confidence_score': 0.5, # Default confidence for hold/no data
                'signal_type': 'technical',
                'metadata': default_metadata
            }
            
            if symbol not in data:
                logger.warning(f"{self.name}: No data for {symbol}, using default HOLD signal.")
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
            
            # Construct metadata first
            metadata = {
                'indicator': 'MA Crossover',
                'fast': fast_period,
                'slow': slow_period,
                'crossover_detected': bullish_crossover or bearish_crossover,
                'fast_ma_value': current_fast,
                'slow_ma_value': current_slow
            }
            
            # Get current position from the optional portfolio dictionary
            current_position = 0
            if current_portfolio and 'positions' in current_portfolio:
                current_position = current_portfolio['positions'].get(symbol, 0)
            
            # Get current price (still needed for logging, might remove later)
            current_price = df["close"].iloc[-1]
            
            # Generate rich signals
            if bullish_crossover:
                signals[symbol] = {
                    'signal_strength': 1.0,
                    'confidence_score': 0.75, # Example: Fixed confidence for clear crossover
                    'signal_type': 'technical',
                    'metadata': metadata
                }
                logger.info(f"{self.name}: Generated BUY signal for {symbol} at {current_price:.2f} (Fast MA crossed above Slow MA)")

            elif bearish_crossover:
                signals[symbol] = {
                    'signal_strength': -1.0,
                    'confidence_score': 0.75, # Example: Fixed confidence for clear crossover
                    'signal_type': 'technical',
                    'metadata': metadata
                }
                logger.info(f"{self.name}: Generated SELL signal for {symbol} at {current_price:.2f} (Fast MA crossed below Slow MA)")
            
            # else: the default HOLD signal set at the start of the loop remains
            
        # Ensure all configured symbols have a signal (redundant due to default init, but safe)
        for symbol in self.symbols:
            if symbol not in signals:
                signals[symbol] = {
                    'signal_strength': 0.0,
                    'confidence_score': 0.5, 
                    'signal_type': 'technical',
                    'metadata': default_metadata
                }
                logger.warning(f"{self.name}: Symbol {symbol} somehow missed, setting HOLD signal.")
        
        return signals
