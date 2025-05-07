"""
Rust Extensions Module

This module provides direct access to the Rust-accelerated functions.
It serves as a bridge between the Python codebase and the Rust implementations.
"""

import logging

logger = logging.getLogger(__name__)

# Try to import the Rust extensions
try:
    from ai_trading_agent_rs import (
        create_lag_features_rs,
        create_diff_features_rs,
        create_pct_change_features_rs,
        create_rolling_window_features_rs
    )
    
    __all__ = [
        'create_lag_features_rs',
        'create_diff_features_rs',
        'create_pct_change_features_rs',
        'create_rolling_window_features_rs'
    ]
    
    logger.info("Successfully imported Rust extensions")
    
except ImportError as e:
    logger.warning(f"Failed to import Rust extensions: {e}")
    
    # Define placeholder functions that raise ImportError when called
    def _missing_rust_extension(*args, **kwargs):
        raise ImportError("Rust extensions are not available")
    
    create_lag_features_rs = _missing_rust_extension
    create_diff_features_rs = _missing_rust_extension
    create_pct_change_features_rs = _missing_rust_extension
    create_rolling_window_features_rs = _missing_rust_extension
    
    __all__ = [
        'create_lag_features_rs',
        'create_diff_features_rs',
        'create_pct_change_features_rs',
        'create_rolling_window_features_rs'
    ]
