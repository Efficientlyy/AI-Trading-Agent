"""
Market Regime Classification Package

This package provides tools for detecting and classifying market regimes
based on multiple factors including volatility clustering, momentum,
correlation patterns, and market liquidity.

It also includes temporal pattern recognition components for seasonality detection,
regime transition probability modeling, and multi-timeframe confirmation logic.
"""

from .core_definitions import (
    MarketRegimeType,
    VolatilityRegimeType,
    LiquidityRegimeType,
    CorrelationRegimeType,
    RegimeChangeSignificance,
    RegimeDetectionMethod,
    MarketRegimeConfig,
    MarketRegimeInfo
)

from .volatility_clustering import VolatilityClusteringDetector
from .momentum_analysis import MomentumFactorAnalyzer
from .correlation_analysis import CorrelationAnalyzer
from .liquidity_analysis import LiquidityAnalyzer
from .regime_classifier import MarketRegimeClassifier
from .seasonality import SeasonalityDetector
from .transition_probability import TransitionProbabilityModel
from .multi_timeframe import MultiTimeframeConfirmation
from .temporal_patterns import TemporalPatternRecognition, TemporalPatternOptimizer

__all__ = [
    'MarketRegimeType',
    'VolatilityRegimeType',
    'LiquidityRegimeType',
    'CorrelationRegimeType',
    'RegimeChangeSignificance',
    'RegimeDetectionMethod',
    'MarketRegimeConfig',
    'MarketRegimeInfo',
    'VolatilityClusteringDetector',
    'MomentumFactorAnalyzer',
    'CorrelationAnalyzer',
    'LiquidityAnalyzer',
    'MarketRegimeClassifier',
    'SeasonalityDetector',
    'TransitionProbabilityModel',
    'MultiTimeframeConfirmation',
    'TemporalPatternRecognition',
    'TemporalPatternOptimizer'
]
