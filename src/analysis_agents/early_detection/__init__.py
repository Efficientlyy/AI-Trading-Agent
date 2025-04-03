"""
Early Event Detection System for Market-Moving Events.

This package implements a system for detecting market-moving events before they go viral,
allowing the trading agent to take positions ahead of mainstream market reactions.
"""

from src.analysis_agents.early_detection.system import EarlyEventDetectionSystem

__all__ = ["EarlyEventDetectionSystem"]