"""
Portfolio Visualization for Multi-Asset Backtesting.

This module provides visualization tools for portfolio performance,
asset allocation, correlation analysis, and risk metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime

from .performance_metrics import PerformanceMetrics
from .correlation_analysis import plot_correlation_matrix, plot_correlation_network, plot_rolling_correlations


def plot_portfolio_performance(
    metrics: PerformanceMetrics,
    title: str = "Portfolio Performance",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive portfolio performance visualization.
    
    Args:
        metrics: PerformanceMetrics object with performance data
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create a grid of subplots
    gs = fig.add_gridspec(3, 2)
    
    # Equity curve and drawdown
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(metrics.equity_curve.index, metrics.equity_curve.values, label="Equity Curve", color="blue")
    ax1.set_title("Equity Curve", fontsize=12)
    ax1.set_ylabel("Portfolio Value")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Add drawdown on secondary axis
    ax1b = ax1.twinx()
    ax1b.fill_between(
        metrics.drawdown_curve.index, 
        metrics.drawdown_curve.values, 
        0, 
        alpha=0.3, 
        color="red", 
        label="Drawdown"
    )
    ax1b.set_ylabel("Drawdown (%)")
    ax1b.set_ylim(min(metrics.drawdown_curve.min() * 1.1, -0.05), 0.01)  # Set y-axis limits for drawdown
    ax1b.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax1b.legend(loc="upper right")
    
    # Daily returns histogram
    ax2 = fig.add_subplot(gs[1, 0])
    sns.histplot(metrics.daily_returns, bins=50, kde=True, ax=ax2, color="green")
    ax2.set_title("Daily Returns Distribution", fontsize=12)
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    
    # Add vertical line at mean
    mean_return = metrics.daily_returns.mean()
    ax2.axvline(mean_return, color="red", linestyle="--", label=f"Mean: {mean_return:.4f}")
    ax2.legend()
    
    # Drawdown periods
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.fill_between(
        metrics.drawdown_curve.index, 
        metrics.drawdown_curve.values, 
        0, 
        alpha=0.7, 
        color="red"
    )
    ax3.set_title("Drawdown Periods", fontsize=12)
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Format x-axis dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Key metrics table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    
    metrics_data = [
        ["Total Return", f"{metrics.total_return:.2%}"],
        ["Annualized Return", f"{metrics.annualized_return:.2%}"],
        ["Volatility", f"{metrics.volatility:.2%}"],
        ["Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}"],
        ["Sortino Ratio", f"{metrics.sortino_ratio:.2f}"],
        ["Max Drawdown", f"{metrics.max_drawdown:.2%}"],
        ["Max Drawdown Duration", f"{metrics.max_drawdown_duration} days"],
        ["Win Rate", f"{metrics.win_rate:.2%}"],
        ["Profit Factor", f"{metrics.profit_factor:.2f}"],
        ["Calmar Ratio", f"{metrics.calmar_ratio:.2f}"],
        ["Omega Ratio", f"{metrics.omega_ratio:.2f}"],
        ["Avg Exposure", f"{metrics.avg_exposure:.2%}"],
        ["Time in Market", f"{metrics.time_in_market:.2%}"],
    ]
    
    # Create table
    table = ax4.table(
        cellText=metrics_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.4, 0.4]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Set title for the entire figure
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_asset_allocation(
    allocations: Dict[str, List[float]],
    timestamps: List[datetime],
    title: str = "Asset Allocation Over Time",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot asset allocation over time.
    
    Args:
        allocations: Dictionary mapping symbols to lists of allocation values
        timestamps: List of timestamps corresponding to allocation values
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Convert to DataFrame
    df = pd.DataFrame(allocations, index=timestamps)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stacked area chart
    ax.stackplot(
        df.index, 
        *[df[col] for col in df.columns], 
        labels=df.columns,
        alpha=0.8
    )
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Allocation", fontsize=12)
    
    # Add legend
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_asset_performance_comparison(
    asset_metrics: Dict[str, Dict[str, float]],
    benchmark_return: Optional[float] = None,
    title: str = "Asset Performance Comparison",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance comparison between assets.
    
    Args:
        asset_metrics: Dictionary mapping symbols to dictionaries of metrics
        benchmark_return: Optional benchmark return for comparison
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Extract returns and volatilities
    symbols = []
    returns = []
    volatilities = []
    drawdowns = []
    
    for symbol, metrics in asset_metrics.items():
        symbols.append(symbol)
        returns.append(metrics.get("total_return", 0))
        volatilities.append(metrics.get("volatility", 0))
        drawdowns.append(metrics.get("max_drawdown", 0))
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Risk-return scatter plot
    ax1.scatter(volatilities, returns, s=100, alpha=0.7)
    
    # Add labels to points
    for i, symbol in enumerate(symbols):
        ax1.annotate(
            symbol, 
            (volatilities[i], returns[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10
        )
    
    # Add benchmark if provided
    if benchmark_return is not None:
        ax1.axhline(y=benchmark_return, color="red", linestyle="--", label="Benchmark")
    
    # Set title and labels
    ax1.set_title("Risk-Return Profile", fontsize=12)
    ax1.set_xlabel("Volatility", fontsize=10)
    ax1.set_ylabel("Total Return", fontsize=10)
    
    # Format axes as percentage
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add legend if benchmark is provided
    if benchmark_return is not None:
        ax1.legend()
    
    # Return vs drawdown bar chart
    x = np.arange(len(symbols))
    width = 0.35
    
    ax2.bar(x - width/2, returns, width, label="Return", alpha=0.7, color="green")
    ax2.bar(x + width/2, drawdowns, width, label="Max Drawdown", alpha=0.7, color="red")
    
    # Set title and labels
    ax2.set_title("Return vs Max Drawdown", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(symbols, rotation=45)
    ax2.set_ylabel("Percentage", fontsize=10)
    
    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    ax2.legend()
    
    # Set title for the entire figure
    fig.suptitle(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_drawdown_comparison(
    asset_drawdowns: Dict[str, pd.Series],
    portfolio_drawdown: pd.Series,
    title: str = "Drawdown Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot drawdown comparison between assets and portfolio.
    
    Args:
        asset_drawdowns: Dictionary mapping symbols to drawdown Series
        portfolio_drawdown: Portfolio drawdown Series
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot asset drawdowns
    for symbol, drawdown in asset_drawdowns.items():
        ax.plot(drawdown.index, drawdown.values, alpha=0.5, linewidth=1, label=symbol)
    
    # Plot portfolio drawdown
    ax.plot(
        portfolio_drawdown.index, 
        portfolio_drawdown.values, 
        color="black", 
        linewidth=2, 
        label="Portfolio"
    )
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown", fontsize=12)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc="lower right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def create_performance_dashboard(
    metrics: PerformanceMetrics,
    asset_metrics: Dict[str, Dict[str, float]],
    correlation_matrix: pd.DataFrame,
    asset_allocations: Dict[str, List[float]],
    allocation_timestamps: List[datetime],
    asset_drawdowns: Dict[str, pd.Series],
    output_dir: str = "./reports"
) -> None:
    """
    Create a comprehensive performance dashboard with multiple visualizations.
    
    Args:
        metrics: PerformanceMetrics object with portfolio performance data
        asset_metrics: Dictionary mapping symbols to dictionaries of metrics
        correlation_matrix: Correlation matrix as DataFrame
        asset_allocations: Dictionary mapping symbols to lists of allocation values
        allocation_timestamps: List of timestamps for allocations
        asset_drawdowns: Dictionary mapping symbols to drawdown Series
        output_dir: Directory to save visualizations
        
    Returns:
        None
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Portfolio performance
    plot_portfolio_performance(
        metrics, 
        save_path=os.path.join(output_dir, "portfolio_performance.png")
    )
    
    # 2. Asset allocation
    plot_asset_allocation(
        asset_allocations, 
        allocation_timestamps,
        save_path=os.path.join(output_dir, "asset_allocation.png")
    )
    
    # 3. Asset performance comparison
    plot_asset_performance_comparison(
        asset_metrics,
        save_path=os.path.join(output_dir, "asset_performance.png")
    )
    
    # 4. Correlation matrix
    plot_correlation_matrix(
        correlation_matrix,
        save_path=os.path.join(output_dir, "correlation_matrix.png")
    )
    
    # 5. Correlation network
    plot_correlation_network(
        correlation_matrix,
        save_path=os.path.join(output_dir, "correlation_network.png")
    )
    
    # 6. Drawdown comparison
    plot_drawdown_comparison(
        asset_drawdowns,
        metrics.drawdown_curve,
        save_path=os.path.join(output_dir, "drawdown_comparison.png")
    )
    
    print(f"Performance dashboard created in {output_dir}")
