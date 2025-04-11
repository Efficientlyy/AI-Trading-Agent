"""
Strategies module for AI Trading Agent.

This module provides base classes and implementations for trading strategies.
"""

from .base_strategy import BaseStrategy
from .ma_crossover_strategy import MACrossoverStrategy

__all__ = ["BaseStrategy", "MACrossoverStrategy"]
