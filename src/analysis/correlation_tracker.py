"""Correlation tracking module.

This module provides tools for tracking correlations between different assets
and market indicators, including:
1. Price correlations
2. Volume correlations
3. Volatility correlations
4. Lead/lag relationships
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict


class CorrelationTracker:
    """Asset correlation tracker."""
    
    def __init__(self, window_sizes: List[int] = [24, 168, 720]):
        """Initialize the correlation tracker.
        
        Args:
            window_sizes: List of window sizes in hours for correlation calculation
                        (default: [24h, 1w, 1m])
        """
        self.window_sizes = window_sizes
        
        # Price history
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.volume_history: Dict[str, List[float]] = defaultdict(list)
        self.volatility_history: Dict[str, List[float]] = defaultdict(list)
        
        # Correlation matrices
        self.price_correlations: Dict[int, Dict[Tuple[str, str], float]] = {
            w: {} for w in window_sizes
        }
        self.volume_correlations: Dict[int, Dict[Tuple[str, str], float]] = {
            w: {} for w in window_sizes
        }
        self.volatility_correlations: Dict[int, Dict[Tuple[str, str], float]] = {
            w: {} for w in window_sizes
        }
        
        # Lead/lag relationships
        self.lead_lag: Dict[Tuple[str, str], Dict[str, float]] = {}
    
    def update_data(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime
    ) -> None:
        """Update price and volume data.
        
        Args:
            symbol: Asset symbol
            price: Current price
            volume: Current volume
            timestamp: Data timestamp
        """
        # Update price history
        self.price_history[symbol].append(price)
        
        # Update volume history
        self.volume_history[symbol].append(volume)
        
        # Calculate and update volatility
        if len(self.price_history[symbol]) >= 2:
            returns = np.log(price / self.price_history[symbol][-2])
            volatility = abs(returns)
        else:
            volatility = 0.0
        self.volatility_history[symbol].append(volatility)
        
        # Maintain buffer sizes
        max_window = max(self.window_sizes)
        for history in [self.price_history[symbol], 
                       self.volume_history[symbol],
                       self.volatility_history[symbol]]:
            if len(history) > max_window:
                history.pop(0)
    
    def calculate_correlations(self) -> None:
        """Calculate correlations for all pairs and windows."""
        symbols = list(self.price_history.keys())
        
        for window in self.window_sizes:
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    # Price correlations
                    price_corr = self._calculate_correlation(
                        self.price_history[symbol1][-window:],
                        self.price_history[symbol2][-window:]
                    )
                    self.price_correlations[window][(symbol1, symbol2)] = price_corr
                    self.price_correlations[window][(symbol2, symbol1)] = price_corr
                    
                    # Volume correlations
                    volume_corr = self._calculate_correlation(
                        self.volume_history[symbol1][-window:],
                        self.volume_history[symbol2][-window:]
                    )
                    self.volume_correlations[window][(symbol1, symbol2)] = volume_corr
                    self.volume_correlations[window][(symbol2, symbol1)] = volume_corr
                    
                    # Volatility correlations
                    vol_corr = self._calculate_correlation(
                        self.volatility_history[symbol1][-window:],
                        self.volatility_history[symbol2][-window:]
                    )
                    self.volatility_correlations[window][(symbol1, symbol2)] = vol_corr
                    self.volatility_correlations[window][(symbol2, symbol1)] = vol_corr
                    
                    # Lead/lag relationships
                    lead_lag = self._calculate_lead_lag(
                        self.price_history[symbol1][-window:],
                        self.price_history[symbol2][-window:]
                    )
                    self.lead_lag[(symbol1, symbol2)] = {
                        "coefficient": float(lead_lag[0]),
                        "lag": int(lead_lag[1])
                    }
    
    def _calculate_correlation(
        self,
        series1: List[float],
        series2: List[float]
    ) -> float:
        """Calculate correlation between two series.
        
        Args:
            series1: First time series
            series2: Second time series
            
        Returns:
            Correlation coefficient
        """
        if len(series1) != len(series2) or len(series1) < 2:
            return 0.0
        
        return float(np.corrcoef(series1, series2)[0, 1])
    
    def _calculate_lead_lag(
        self,
        series1: List[float],
        series2: List[float],
        max_lag: int = 12
    ) -> Tuple[float, int]:
        """Calculate lead/lag relationship between two series.
        
        Args:
            series1: First time series
            series2: Second time series
            max_lag: Maximum lag to consider
            
        Returns:
            Tuple of (correlation coefficient, lag)
        """
        if len(series1) != len(series2) or len(series1) < max_lag + 1:
            return 0.0, 0
        
        # Convert to numpy arrays
        s1 = np.array(series1)
        s2 = np.array(series2)
        
        # Calculate correlations for different lags
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(s1[:lag], s2[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(s1[lag:], s2[:-lag])[0, 1]
            else:
                corr = np.corrcoef(s1, s2)[0, 1]
            correlations.append((corr, lag))
        
        # Find lag with maximum correlation
        max_corr, best_lag = max(correlations, key=lambda x: abs(x[0]))
        
        return float(max_corr), int(best_lag)
    
    def get_correlations(
        self,
        symbol: str,
        window: int
    ) -> Dict[str, Dict[str, float]]:
        """Get correlations for a symbol.
        
        Args:
            symbol: Asset symbol
            window: Window size
            
        Returns:
            Dictionary of correlation metrics
        """
        if window not in self.window_sizes:
            raise ValueError(f"Invalid window size: {window}")
        
        correlations = {
            "price": {},
            "volume": {},
            "volatility": {}
        }
        
        for other_symbol in self.price_history.keys():
            if other_symbol != symbol:
                correlations["price"][other_symbol] = self.price_correlations[window].get(
                    (symbol, other_symbol), 0.0
                )
                correlations["volume"][other_symbol] = self.volume_correlations[window].get(
                    (symbol, other_symbol), 0.0
                )
                correlations["volatility"][other_symbol] = self.volatility_correlations[window].get(
                    (symbol, other_symbol), 0.0
                )
        
        return correlations
    
    def get_lead_lag_relationships(
        self,
        symbol: str
    ) -> Dict[str, Dict[str, Union[float, int]]]:
        """Get lead/lag relationships for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary of lead/lag relationships
        """
        relationships = {}
        
        for other_symbol in self.price_history.keys():
            if other_symbol != symbol:
                rel = self.lead_lag.get((symbol, other_symbol))
                if rel is not None:
                    relationships[other_symbol] = rel
        
        return relationships
    
    def get_correlation_clusters(
        self,
        window: int,
        threshold: float = 0.7
    ) -> List[List[str]]:
        """Find clusters of correlated assets.
        
        Args:
            window: Window size
            threshold: Correlation threshold for clustering
            
        Returns:
            List of asset clusters
        """
        if window not in self.window_sizes:
            raise ValueError(f"Invalid window size: {window}")
        
        # Build adjacency matrix
        symbols = list(self.price_history.keys())
        n = len(symbols)
        adjacency = np.zeros((n, n))
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    corr = abs(self.price_correlations[window].get((symbol1, symbol2), 0.0))
                    if corr >= threshold:
                        adjacency[i, j] = 1
        
        # Find connected components (clusters)
        clusters = []
        visited = set()
        
        def dfs(node: int, cluster: List[str]) -> None:
            visited.add(node)
            cluster.append(symbols[node])
            for neighbor in range(n):
                if adjacency[node, neighbor] == 1 and neighbor not in visited:
                    dfs(neighbor, cluster)
        
        for i in range(n):
            if i not in visited:
                cluster = []
                dfs(i, cluster)
                if len(cluster) > 1:  # Only include clusters with multiple assets
                    clusters.append(cluster)
        
        return clusters 