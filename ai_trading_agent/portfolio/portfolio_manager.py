"""
Portfolio Manager Module for AI Trading Agent

This module exports the PortfolioManager class for use throughout the application.
It serves as a central entry point for portfolio management functionality.

The PortfolioManager provides comprehensive portfolio management capabilities including:
- Position sizing based on risk parameters and stop-loss levels
- Portfolio rebalancing to target weights
- Correlation-based position management
- Drawdown-based risk management
- Performance metrics calculation
"""

# Import the AdvancedPortfolioManager as the main implementation
from .advanced_portfolio_manager import AdvancedPortfolioManager

# Export AdvancedPortfolioManager as PortfolioManager for application-wide consistency
PortfolioManager = AdvancedPortfolioManager

# Export the class directly for backward compatibility
__all__ = ['PortfolioManager', 'AdvancedPortfolioManager']