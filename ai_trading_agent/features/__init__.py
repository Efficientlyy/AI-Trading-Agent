"""
Features package for AI Trading Agent.

This package contains modules for feature engineering and transformation
used in the trading system.
"""

from .lag_features import (
    lag_feature,
    diff_feature,
    pct_change_feature,
    create_lag_features,
    create_diff_features,
    create_pct_change_features,
    RUST_AVAILABLE
)

__all__ = [
    'lag_feature',
    'diff_feature',
    'pct_change_feature',
    'create_lag_features',
    'create_diff_features',
    'create_pct_change_features',
    'RUST_AVAILABLE'
]
