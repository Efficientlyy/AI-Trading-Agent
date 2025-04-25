"""
Rust-accelerated lag features for time series analysis.

This module provides high-performance implementations of time series feature engineering
functions using Rust.
"""

# Import the Rust extension module
try:
    from rust_lag_extension import (
        create_lag_features,
        create_diff_features,
        create_pct_change_features,
        create_rolling_window_features
    )
    RUST_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"Failed to import Rust extension: {e}")
    RUST_AVAILABLE = False
