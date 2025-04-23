"""
signal_processing/regime.py

Provides market regime detection utilities (e.g., volatility-based, clustering-based).
"""

import numpy as np
import pandas as pd


def volatility_regime(prices: pd.Series, window: int = 20, threshold: float = 0.02) -> pd.Series:
    """
    Simple volatility-based regime detector.
    Returns 'high_vol' or 'low_vol' for each timestamp.
    """
    returns = prices.pct_change()
    rolling_vol = returns.rolling(window=window).std()
    regime = np.where(rolling_vol > threshold, 'high_vol', 'low_vol')
    return pd.Series(regime, index=prices.index)


def rolling_kmeans_regime(prices: pd.Series, window: int = 60, n_clusters: int = 2) -> pd.Series:
    """
    Regime detection using rolling k-means clustering on returns volatility.
    Returns regime labels (0, 1, ...) for each timestamp.
    """
    from sklearn.cluster import KMeans
    returns = prices.pct_change().rolling(window=window).std().dropna()
    regimes = pd.Series(index=prices.index, dtype=int)
    if len(returns) < window:
        return regimes  # Not enough data
    X = returns.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    regimes.iloc[-len(labels):] = labels
    return regimes
