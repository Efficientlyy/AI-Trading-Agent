"""
Multi-Asset Backtesting Framework.

This package implements a comprehensive framework for testing trading strategies
across multiple assets simultaneously, with portfolio-level analysis and risk management.
"""

from src.backtesting.multi_asset.core import MultiAssetBacktester
from src.backtesting.multi_asset.metrics import calculate_portfolio_metrics
from src.backtesting.multi_asset.visualization import (
    plot_equity_curve,
    plot_asset_allocation,
    plot_correlation_heatmap,
    plot_drawdowns
)

__all__ = [
    'MultiAssetBacktester',
    'calculate_portfolio_metrics',
    'plot_equity_curve',
    'plot_asset_allocation',
    'plot_correlation_heatmap',
    'plot_drawdowns'
]
