"""
Rust integration module for AI Trading Agent.

This module provides Python wrappers for Rust-accelerated functions.
"""

from .indicators import (
    calculate_sma,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    RUST_AVAILABLE
)

from .features import (
    create_lag_features,
    create_lag_features_df
)

__all__ = [
    'calculate_sma',
    'calculate_ema',
    'calculate_macd',
    'calculate_rsi',
    'create_lag_features',
    'create_lag_features_df',
    'RUST_AVAILABLE'
]