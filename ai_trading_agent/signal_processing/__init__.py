"""
signal_processing/__init__.py

Makes signal processing utilities available as a package.
"""

from .filters import exponential_moving_average, savitzky_golay_filter, rolling_zscore
from .regime import volatility_regime, rolling_kmeans_regime
