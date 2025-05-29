"""
Coordination Module for AI Trading Agent

This package provides coordination capabilities for the AI Trading Agent system:
1. Strategy coordination for optimal combined signals
2. Performance attribution to understand contribution factors
3. Cross-strategy allocation and conflict resolution
"""

from .strategy_coordinator import StrategyCoordinator
from .performance_attribution import PerformanceAttributor

__all__ = [
    'StrategyCoordinator',
    'PerformanceAttributor'
]
