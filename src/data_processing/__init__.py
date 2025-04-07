"""
Data Processing Package

Modules for calculating indicators, feature engineering, and preprocessing.
"""

from .indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd
)

from .feature_engineering import (
    add_lagged_features,
    add_technical_indicators,
    scale_features
)

# You might add preprocessing functions here later if needed
# from .preprocessing import normalize_data, etc.

__all__ = [
    # Indicators
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi',
    'calculate_macd',
    # Feature Engineering
    'add_lagged_features',
    'add_technical_indicators',
    'scale_features',
    # Preprocessing (add later)
]
