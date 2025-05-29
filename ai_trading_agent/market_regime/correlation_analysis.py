"""
Correlation Analysis Module

This module provides tools for analyzing correlations between assets and market sectors,
which helps identify different market regimes such as risk-on/risk-off environments,
sector rotation, or market fragmentation/convergence.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import networkx as nx

from .core_definitions import (
    CorrelationRegimeType,
    MarketRegimeConfig
)

# Set up logger
logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Class for analyzing correlation patterns across assets and markets.
    
    Detects regimes such as risk-on/risk-off, sector rotation, and market fragmentation.
    """
    
    def __init__(self, config: Optional[MarketRegimeConfig] = None):
        """
        Initialize the correlation analyzer.
        
        Args:
            config: Configuration for correlation analysis
        """
        self.config = config or MarketRegimeConfig()
        self.historical_correlations = {}
        self.regime_history = []
        self.asset_groups = {}  # For tracking asset categories/sectors
    
    def register_asset_group(self, group_name: str, assets: List[str]) -> None:
        """
        Register a group of related assets (e.g., sector, asset class).
        
        Args:
            group_name: Name of the asset group
            assets: List of asset identifiers in this group
        """
        self.asset_groups[group_name] = assets
        logger.info(f"Registered asset group {group_name} with {len(assets)} assets")
    
    def analyze_correlations(self, 
                             returns_df: pd.DataFrame,
                             lookback_periods: Optional[List[int]] = None) -> Dict[str, any]:
        """
        Analyze correlation patterns in a dataframe of asset returns.
        
        Args:
            returns_df: DataFrame where columns are assets and rows are return observations
            lookback_periods: List of periods to calculate correlations over (default: [20, 60, 120])
            
        Returns:
            Dictionary with correlation analysis results
        """
        if returns_df is None or returns_df.empty or returns_df.shape[1] < 2:
            logger.warning(f"Insufficient data for correlation analysis: {returns_df.shape if returns_df is not None else 'None'}")
            return {
                "correlation_regime": CorrelationRegimeType.UNKNOWN.value,
                "avg_correlation": None,
                "correlation_dispersion": None,
                "risk_on_off_score": None,
                "group_cohesion": {}
            }
        
        # Default lookback periods if none provided
        if lookback_periods is None:
            lookback_periods = [20, 60, 120]
        
        # Ensure periods don't exceed available data
        max_lookback = min(lookback_periods[-1], len(returns_df) - 1)
        lookback_periods = [p for p in lookback_periods if p <= max_lookback]
        
        if not lookback_periods:
            lookback_periods = [max_lookback]
        
        # Calculate correlation matrices for different periods
        correlation_matrices = {}
        for period in lookback_periods:
            if len(returns_df) >= period:
                # Use the most recent 'period' observations
                recent_returns = returns_df.iloc[-period:]
                # Calculate correlation matrix
                corr_matrix = recent_returns.corr()
                correlation_matrices[period] = corr_matrix
        
        if not correlation_matrices:
            logger.warning("Could not calculate correlation matrices due to insufficient data")
            return {
                "correlation_regime": CorrelationRegimeType.UNKNOWN.value,
                "avg_correlation": None,
                "correlation_dispersion": None,
                "risk_on_off_score": None,
                "group_cohesion": {}
            }
        
        # Focus on the shortest period correlation matrix for regime detection
        primary_period = min(lookback_periods)
        primary_corr_matrix = correlation_matrices[primary_period]
        
        # Calculate average correlation (excluding self-correlations)
        mask = ~np.eye(primary_corr_matrix.shape[0], dtype=bool)
        avg_correlation = primary_corr_matrix.values[mask].mean()
        
        # Calculate correlation dispersion (standard deviation of correlations)
        correlation_dispersion = primary_corr_matrix.values[mask].std()
        
        # Analyze correlation patterns
        risk_on_off_score = self._calculate_risk_on_off_score(primary_corr_matrix)
        
        # Calculate group cohesion if asset groups are defined
        group_cohesion = {}
        if self.asset_groups:
            for group_name, assets in self.asset_groups.items():
                # Find assets in this group that are present in the data
                available_assets = [asset for asset in assets if asset in primary_corr_matrix.columns]
                
                if len(available_assets) >= 2:
                    # Extract the submatrix for this group
                    group_corr = primary_corr_matrix.loc[available_assets, available_assets]
                    # Calculate within-group average correlation
                    mask = ~np.eye(group_corr.shape[0], dtype=bool)
                    within_corr = group_corr.values[mask].mean()
                    
                    # Calculate correlation between this group and others
                    other_assets = [col for col in primary_corr_matrix.columns if col not in available_assets]
                    if other_assets:
                        between_corr = primary_corr_matrix.loc[available_assets, other_assets].values.mean()
                    else:
                        between_corr = None
                    
                    group_cohesion[group_name] = {
                        "within_correlation": within_corr,
                        "between_correlation": between_corr,
                        "cohesion_score": within_corr - between_corr if between_corr is not None else None
                    }
        
        # Identify correlation regime
        correlation_regime = self._identify_correlation_regime(
            avg_correlation, 
            correlation_dispersion,
            risk_on_off_score,
            group_cohesion
        )
        
        # Prepare result dictionary
        result = {
            "correlation_regime": correlation_regime.value,
            "avg_correlation": avg_correlation,
            "correlation_dispersion": correlation_dispersion,
            "risk_on_off_score": risk_on_off_score,
            "group_cohesion": group_cohesion,
            "lookback_period": primary_period
        }
        
        # Store historical correlations
        timestamp = returns_df.index[-1] if hasattr(returns_df.index[-1], 'timestamp') else pd.Timestamp.now()
        self.historical_correlations[timestamp] = {
            "avg_correlation": avg_correlation,
            "correlation_dispersion": correlation_dispersion,
            "risk_on_off_score": risk_on_off_score,
            "correlation_regime": correlation_regime.value
        }
        
        # Track in regime history
        self.regime_history.append({
            "timestamp": timestamp,
            "correlation_regime": correlation_regime.value,
            "avg_correlation": avg_correlation,
            "risk_on_off_score": risk_on_off_score
        })
        
        # Trim history if too long
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        return result
    
    def _calculate_risk_on_off_score(self, corr_matrix: pd.DataFrame) -> float:
        """
        Calculate a score indicating whether the market is in risk-on or risk-off mode.
        
        Risk-on/off is characterized by high correlations between risk assets and negative
        correlations between risk assets and safe haven assets.
        
        Args:
            corr_matrix: Correlation matrix of asset returns
            
        Returns:
            Risk-on/off score (-1.0 to 1.0, with positive values indicating risk-on)
        """
        # This is a simplified implementation - in a full system, we would
        # have predefined lists of "risk assets" and "safe haven assets"
        score = 0.0
        
        # If we have registered "risk" and "safe" asset groups, use them
        if "risk_assets" in self.asset_groups and "safe_assets" in self.asset_groups:
            risk_assets = [a for a in self.asset_groups["risk_assets"] if a in corr_matrix.columns]
            safe_assets = [a for a in self.asset_groups["safe_assets"] if a in corr_matrix.columns]
            
            if risk_assets and safe_assets:
                # Calculate average correlation within risk assets
                risk_corr = corr_matrix.loc[risk_assets, risk_assets]
                mask = ~np.eye(risk_corr.shape[0], dtype=bool)
                within_risk_corr = risk_corr.values[mask].mean() if mask.sum() > 0 else 0.0
                
                # Calculate average correlation between risk and safe assets
                risk_safe_corr = corr_matrix.loc[risk_assets, safe_assets].values.mean()
                
                # Risk-on: high within-risk correlation, negative risk-safe correlation
                score = within_risk_corr - risk_safe_corr
        
        # If we don't have predefined groups, use a simpler heuristic based on
        # the overall correlation distribution
        else:
            # Extract the correlations (excluding self-correlations)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            correlations = corr_matrix.values[mask]
            
            # In risk-on/off, we expect high absolute correlations
            # A bimodal distribution with strong positive and negative correlations
            # suggests risk-on/off dynamics
            if len(correlations) > 0:
                abs_correlations = np.abs(correlations)
                # Stronger absolute correlations indicate stronger risk-on/off dynamics
                avg_abs_corr = abs_correlations.mean()
                # The skewness of raw correlations can indicate risk-on (positive) or risk-off (negative)
                skew = stats.skew(correlations)
                score = avg_abs_corr * 2 * (1 if skew > 0 else -1)
        
        return max(-1.0, min(1.0, score))
    
    def _identify_correlation_regime(self, 
                                    avg_correlation: float, 
                                    correlation_dispersion: float,
                                    risk_on_off_score: float,
                                    group_cohesion: Dict[str, Dict[str, float]]) -> CorrelationRegimeType:
        """
        Identify the correlation regime based on correlation metrics.
        
        Args:
            avg_correlation: Average correlation between assets
            correlation_dispersion: Standard deviation of correlations
            risk_on_off_score: Score indicating risk-on vs risk-off dynamics
            group_cohesion: Dictionary of group cohesion metrics
            
        Returns:
            CorrelationRegimeType enum
        """
        # Handle missing data
        if avg_correlation is None or correlation_dispersion is None:
            return CorrelationRegimeType.UNKNOWN
        
        # Detect sector rotation based on group cohesion differences
        if group_cohesion:
            cohesion_scores = [g["cohesion_score"] for g in group_cohesion.values() 
                              if g["cohesion_score"] is not None]
            if cohesion_scores:
                # High variance in cohesion scores suggests sector rotation
                cohesion_variance = np.var(cohesion_scores)
                if cohesion_variance > 0.1:  # Threshold for sector rotation
                    return CorrelationRegimeType.SECTOR_ROTATION
        
        # Detect risk-on and risk-off regimes
        if avg_correlation > self.config.correlation_threshold:
            # High average correlation
            if risk_on_off_score > 0.3:
                return CorrelationRegimeType.RISK_ON
            elif risk_on_off_score < -0.3:
                return CorrelationRegimeType.RISK_OFF
            else:
                return CorrelationRegimeType.CONVERGENT
        else:
            # Low average correlation
            if correlation_dispersion > 0.3:  # High dispersion
                return CorrelationRegimeType.DISPERSED
            else:
                return CorrelationRegimeType.DISPERSED  # Default to dispersed for low correlation
    
    def detect_regime_change(self,
                            current_regime: CorrelationRegimeType,
                            prev_regime: CorrelationRegimeType) -> bool:
        """
        Detect if there has been a significant change in correlation regime.
        
        Args:
            current_regime: Current correlation regime
            prev_regime: Previous correlation regime
            
        Returns:
            Boolean indicating whether regime has changed
        """
        if current_regime is None or prev_regime is None:
            return False
            
        # Change between any different regimes is considered significant
        return current_regime != prev_regime
    
    def visualize_correlation_network(self, corr_matrix: pd.DataFrame, 
                                    threshold: float = 0.5) -> Dict[str, any]:
        """
        Create a network visualization of the correlation structure.
        
        Args:
            corr_matrix: Correlation matrix of asset returns
            threshold: Minimum absolute correlation to include as edge
            
        Returns:
            Dictionary with network metrics and layout information
        """
        try:
            # Create a network from the correlation matrix
            G = nx.Graph()
            
            # Add nodes
            for asset in corr_matrix.columns:
                G.add_node(asset)
            
            # Add edges for correlations above threshold
            for i, asset1 in enumerate(corr_matrix.columns):
                for j, asset2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates and self-loops
                        corr = corr_matrix.loc[asset1, asset2]
                        if abs(corr) >= threshold:
                            G.add_edge(asset1, asset2, weight=corr, color='green' if corr > 0 else 'red')
            
            # Calculate network metrics
            density = nx.density(G)
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                avg_path_length = None
                
            clustering = nx.average_clustering(G)
            
            # Calculate modularity if there are communities
            if nx.number_of_edges(G) > 0:
                communities = list(nx.algorithms.community.greedy_modularity_communities(G))
                modularity = nx.algorithms.community.modularity(G, communities)
            else:
                communities = []
                modularity = 0
            
            # Create a layout for visualization
            if len(G.nodes) > 0:
                layout = nx.spring_layout(G)
                node_positions = {node: [float(pos[0]), float(pos[1])] for node, pos in layout.items()}
            else:
                node_positions = {}
            
            # Prepare result
            result = {
                "nodes": list(G.nodes),
                "edges": [(u, v, d['weight']) for u, v, d in G.edges(data=True)],
                "layout": node_positions,
                "metrics": {
                    "density": density,
                    "average_path_length": avg_path_length,
                    "clustering_coefficient": clustering,
                    "modularity": modularity,
                    "communities": len(communities)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating correlation network: {str(e)}")
            return {
                "nodes": [],
                "edges": [],
                "layout": {},
                "metrics": {}
            }
    
    def get_correlation_statistics(self) -> Dict[str, any]:
        """
        Get summary statistics of correlation history.
        
        Returns:
            Dictionary with correlation statistics
        """
        if not self.regime_history:
            return {
                "mean_correlation": None,
                "correlation_volatility": None,
                "regime_counts": {}
            }
        
        # Extract correlations and regimes
        correlations = [entry["avg_correlation"] for entry in self.regime_history 
                       if entry["avg_correlation"] is not None]
        regimes = [entry["correlation_regime"] for entry in self.regime_history]
        
        # Calculate statistics
        stats_dict = {
            "mean_correlation": np.mean(correlations) if correlations else None,
            "correlation_volatility": np.std(correlations) if correlations else None
        }
        
        # Count regime occurrences
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        stats_dict["regime_counts"] = regime_counts
        
        return stats_dict
