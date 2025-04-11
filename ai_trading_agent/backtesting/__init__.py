"""
Backtesting module for AI Trading Agent.

This module provides tools for backtesting trading strategies.
"""

from .backtester import Backtester
from .performance_metrics import calculate_metrics, PerformanceMetrics

# Import Rust backtester if available
try:
    from .rust_backtester import RustBacktester
    RUST_AVAILABLE = True
    __all__ = ["Backtester", "RustBacktester", "calculate_metrics", "PerformanceMetrics"]
except ImportError:
    RUST_AVAILABLE = False
    __all__ = ["Backtester", "calculate_metrics", "PerformanceMetrics"]
