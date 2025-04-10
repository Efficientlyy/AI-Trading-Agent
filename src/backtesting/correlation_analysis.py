"""
Correlation Analysis for Multi-Asset Backtesting.

This module provides tools for analyzing correlations between assets,
calculating diversification benefits, and visualizing correlation matrices.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class CorrelationStats:
    """Container for correlation statistics."""
    
    correlation_matrix: pd.DataFrame
    avg_correlation: float
    min_correlation: float
    max_correlation: float
    diversification_score: float
    highly_correlated_pairs: List[Tuple[str, str, float]]
    correlation_stability: float


def calculate_correlation_matrix(
    returns_data: Dict[str, List[float]],
    min_periods: int = 30
) -> pd.DataFrame:
    """
    Calculate correlation matrix from returns data.
    
    Args:
        returns_data: Dictionary mapping symbols to lists of returns
        min_periods: Minimum number of periods required to calculate correlation
        
    Returns:
        DataFrame containing the correlation matrix
    """
    # Convert to DataFrame
    returns_df = pd.DataFrame(returns_data)
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr(min_periods=min_periods)
    
    return correlation_matrix


def calculate_rolling_correlations(
    returns_data: Dict[str, List[float]],
    window_size: int = 60
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Calculate rolling correlations between assets.
    
    Args:
        returns_data: Dictionary mapping symbols to lists of returns
        window_size: Window size for rolling correlation
        
    Returns:
        Nested dictionary mapping pairs of symbols to Series of rolling correlations
    """
    # Convert to DataFrame
    returns_df = pd.DataFrame(returns_data)
    
    # Get list of symbols
    symbols = list(returns_data.keys())
    
    # Initialize result dictionary
    rolling_correlations = {}
    
    # Calculate rolling correlations for each pair of symbols
    for i, symbol1 in enumerate(symbols):
        rolling_correlations[symbol1] = {}
        for symbol2 in symbols[i+1:]:
            # Calculate rolling correlation
            corr = returns_df[symbol1].rolling(window=window_size).corr(returns_df[symbol2])
            rolling_correlations[symbol1][symbol2] = corr
    
    return rolling_correlations


def analyze_correlations(
    correlation_matrix: pd.DataFrame,
    high_correlation_threshold: float = 0.7
) -> CorrelationStats:
    """
    Analyze correlation matrix and extract key statistics.
    
    Args:
        correlation_matrix: Correlation matrix as DataFrame
        high_correlation_threshold: Threshold for identifying highly correlated pairs
        
    Returns:
        CorrelationStats object with correlation statistics
    """
    # Get symbols
    symbols = correlation_matrix.columns.tolist()
    
    # Calculate average correlation (excluding self-correlations)
    corr_values = []
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            if i != j:  # Exclude self-correlations
                corr_values.append(correlation_matrix.loc[symbol1, symbol2])
    
    avg_correlation = np.mean(corr_values) if corr_values else 0
    min_correlation = np.min(corr_values) if corr_values else 0
    max_correlation = np.max(corr_values) if corr_values else 0
    
    # Calculate diversification score (1 - avg absolute correlation)
    diversification_score = 1 - np.mean(np.abs(corr_values)) if corr_values else 1
    
    # Identify highly correlated pairs
    highly_correlated_pairs = []
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols[i+1:], i+1):
            corr = correlation_matrix.loc[symbol1, symbol2]
            if abs(corr) >= high_correlation_threshold:
                highly_correlated_pairs.append((symbol1, symbol2, corr))
    
    # Calculate correlation stability (standard deviation of correlations)
    correlation_stability = np.std(corr_values) if corr_values else 0
    
    return CorrelationStats(
        correlation_matrix=correlation_matrix,
        avg_correlation=avg_correlation,
        min_correlation=min_correlation,
        max_correlation=max_correlation,
        diversification_score=diversification_score,
        highly_correlated_pairs=highly_correlated_pairs,
        correlation_stability=correlation_stability
    )


def calculate_conditional_correlations(
    returns_data: Dict[str, List[float]],
    market_index: str,
    threshold_percentile: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate conditional correlations during market stress periods.
    
    Args:
        returns_data: Dictionary mapping symbols to lists of returns
        market_index: Symbol to use as market index
        threshold_percentile: Percentile threshold for identifying stress periods
        
    Returns:
        Tuple of (normal_correlation_matrix, stress_correlation_matrix)
    """
    # Convert to DataFrame
    returns_df = pd.DataFrame(returns_data)
    
    # Check if market index exists in data
    if market_index not in returns_df.columns:
        raise ValueError(f"Market index {market_index} not found in returns data")
    
    # Identify stress periods (bottom percentile of market returns)
    market_returns = returns_df[market_index]
    threshold = np.percentile(market_returns, threshold_percentile)
    stress_mask = market_returns <= threshold
    
    # Split data into normal and stress periods
    normal_returns = returns_df[~stress_mask]
    stress_returns = returns_df[stress_mask]
    
    # Calculate correlation matrices
    normal_correlation = normal_returns.corr()
    stress_correlation = stress_returns.corr()
    
    return normal_correlation, stress_correlation


def plot_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    title: str = "Asset Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "coolwarm",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation matrix as a heatmap.
    
    Args:
        correlation_matrix: Correlation matrix as DataFrame
        title: Plot title
        figsize: Figure size
        cmap: Colormap for heatmap
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_correlation_network(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.5,
    title: str = "Asset Correlation Network",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation network where nodes are assets and edges represent correlations.
    
    Args:
        correlation_matrix: Correlation matrix as DataFrame
        threshold: Minimum absolute correlation to draw an edge
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    try:
        import networkx as nx
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        symbols = correlation_matrix.columns.tolist()
        for symbol in symbols:
            G.add_node(symbol)
        
        # Add edges for correlations above threshold
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                corr = correlation_matrix.loc[symbol1, symbol2]
                if abs(corr) >= threshold:
                    # Edge weight based on absolute correlation
                    weight = abs(corr)
                    # Edge color based on sign of correlation
                    color = "green" if corr > 0 else "red"
                    G.add_edge(symbol1, symbol2, weight=weight, color=color)
        
        # Get edge colors and weights
        edge_colors = [G[u][v]["color"] for u, v in G.edges()]
        edge_weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]
        
        # Calculate node sizes based on degree centrality
        centrality = nx.degree_centrality(G)
        node_sizes = [centrality[node] * 2000 + 500 for node in G.nodes()]
        
        # Draw graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue", alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_colors, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
        
        # Set title
        ax.set_title(title, fontsize=14)
        
        # Remove axis
        ax.axis("off")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    except ImportError:
        print("NetworkX library is required for correlation network plots.")
        return None


def plot_rolling_correlations(
    rolling_correlations: Dict[str, Dict[str, pd.Series]],
    pairs_to_plot: Optional[List[Tuple[str, str]]] = None,
    window_size: int = 60,
    title: str = "Rolling Correlations",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling correlations between asset pairs.
    
    Args:
        rolling_correlations: Nested dictionary mapping pairs of symbols to Series of rolling correlations
        pairs_to_plot: List of symbol pairs to plot (if None, plot all pairs)
        window_size: Window size used for rolling correlations (for title only)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all pairs if not specified
    if pairs_to_plot is None:
        pairs_to_plot = []
        for symbol1, inner_dict in rolling_correlations.items():
            for symbol2 in inner_dict.keys():
                pairs_to_plot.append((symbol1, symbol2))
    
    # Plot rolling correlations for each pair
    for symbol1, symbol2 in pairs_to_plot:
        if symbol1 in rolling_correlations and symbol2 in rolling_correlations[symbol1]:
            corr = rolling_correlations[symbol1][symbol2]
            ax.plot(corr.index, corr.values, label=f"{symbol1} - {symbol2}")
        elif symbol2 in rolling_correlations and symbol1 in rolling_correlations[symbol2]:
            corr = rolling_correlations[symbol2][symbol1]
            ax.plot(corr.index, corr.values, label=f"{symbol1} - {symbol2}")
    
    # Add horizontal lines at 0, 0.5, and -0.5
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
    ax.axhline(y=-0.5, color="green", linestyle="--", alpha=0.3)
    
    # Set title and labels
    ax.set_title(f"{title} ({window_size}-period window)", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Correlation", fontsize=12)
    
    # Set y-axis limits
    ax.set_ylim(-1.1, 1.1)
    
    # Add legend
    ax.legend(loc="best", fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig
