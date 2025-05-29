"""
Adaptive Parameters Module

This module provides functionality for dynamically tuning strategy parameters
based on market conditions and historical performance.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path

# Optional imports for optimization
try:
    from scipy.optimize import minimize
    from sklearn.cluster import KMeans
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

from ..common.utils import get_logger


class MarketRegimeClassifier:
    """
    Classifies market regimes (trending, ranging, volatile) to help with parameter selection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the market regime classifier.
        
        Args:
            config: Configuration dictionary with parameters
                - window_size: Number of periods to analyze for regime classification
                - regime_thresholds: Dict with thresholds for different regimes
        """
        self.logger = get_logger("MarketRegimeClassifier")
        self.config = config or {}
        
        # Extract configuration
        self.window_size = self.config.get("window_size", 20)
        self.regime_thresholds = self.config.get("regime_thresholds", {
            "trending": {
                "directional_strength": 0.6,  # ADX or similar
                "volatility": 0.4  # Normalized ATR
            },
            "ranging": {
                "directional_strength": 0.4,
                "mean_reversion": 0.6  # How often price reverts to mean
            },
            "volatile": {
                "volatility": 0.6,
                "avg_candle_size": 0.5  # Relative to historical
            }
        })
    
    def classify_regime(
        self, 
        market_data: pd.DataFrame, 
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify the current market regime based on price action and indicators.
        
        Args:
            market_data: DataFrame with OHLCV data
            indicators: Dictionary with technical indicators
            
        Returns:
            Dictionary with regime classification and metrics
        """
        if market_data.empty:
            return {"regime": "unknown", "confidence": 0.0}
            
        try:
            # Calculate regime metrics
            metrics = self._calculate_regime_metrics(market_data, indicators)
            
            # Determine the regime based on metrics and thresholds
            regime_scores = {
                "trending": self._calculate_trending_score(metrics),
                "ranging": self._calculate_ranging_score(metrics),
                "volatile": self._calculate_volatile_score(metrics)
            }
            
            # Find the regime with the highest score
            regime = max(regime_scores.items(), key=lambda x: x[1])
            
            # Calculate confidence as the relative strength of the winning regime
            total_score = sum(regime_scores.values()) or 1  # Avoid division by zero
            confidence = regime[1] / total_score
            
            result = {
                "regime": regime[0],
                "confidence": float(confidence),
                "scores": {k: float(v) for k, v in regime_scores.items()},
                "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                            for k, v in metrics.items()}
            }
            
            self.logger.info(
                f"Classified market regime as {regime[0]} with "
                f"confidence {confidence:.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying market regime: {str(e)}")
            return {"regime": "unknown", "confidence": 0.0, "error": str(e)}
    
    def _calculate_regime_metrics(
        self, 
        market_data: pd.DataFrame, 
        indicators: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate metrics used for regime classification."""
        metrics = {}
        
        # Get window of recent data
        window = market_data.tail(self.window_size)
        
        # Calculate directional strength (simplified ADX-like measure)
        up_moves = window['high'].diff().clip(lower=0)
        down_moves = -window['low'].diff().clip(upper=0)
        
        # Simplified directional movement index
        plus_di = up_moves.mean() / (up_moves.mean() + down_moves.mean()) if (up_moves.mean() + down_moves.mean()) > 0 else 0.5
        minus_di = down_moves.mean() / (up_moves.mean() + down_moves.mean()) if (up_moves.mean() + down_moves.mean()) > 0 else 0.5
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
        
        metrics["directional_strength"] = dx
        
        # Calculate volatility (normalized ATR)
        high_low = window['high'] - window['low']
        high_close = abs(window['high'] - window['close'].shift(1))
        low_close = abs(window['low'] - window['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.mean()
        avg_price = window['close'].mean()
        metrics["volatility"] = atr / avg_price if avg_price > 0 else 0
        
        # Calculate mean reversion tendency
        # Correlation between returns and previous returns (negative = mean-reverting)
        returns = window['close'].pct_change()
        metrics["mean_reversion"] = max(0, -np.corrcoef(returns[1:], returns.shift(1)[1:])[0, 1]) if len(returns) > 2 else 0
        
        # Calculate average candle size relative to price
        candle_sizes = (window['high'] - window['low']) / window['close']
        metrics["avg_candle_size"] = candle_sizes.mean()
        
        # Calculate trend consistency
        price_direction = np.sign(window['close'].diff())
        # Count how often the direction changes (lower = more consistent trend)
        direction_changes = (price_direction.diff() != 0).sum()
        metrics["trend_consistency"] = 1 - (direction_changes / len(window))
        
        # Use indicators if available
        if "adx" in indicators:
            adx_values = indicators["adx"]
            if hasattr(adx_values, 'iloc'):
                recent_adx = float(adx_values.iloc[-1])
            else:
                recent_adx = float(adx_values[-1])
            metrics["adx"] = recent_adx / 100  # Normalize to 0-1
        
        return metrics
    
    def _calculate_trending_score(self, metrics: Dict[str, float]) -> float:
        """Calculate the score for trending regime."""
        score = 0.0
        
        # High directional strength indicates trend
        score += metrics.get("directional_strength", 0) * 2.0
        
        # Consistent trend direction adds to trend score
        score += metrics.get("trend_consistency", 0) * 1.5
        
        # ADX directly measures trend strength
        score += metrics.get("adx", 0) * 2.0
        
        # Mean reversion works against trend
        score -= metrics.get("mean_reversion", 0) * 1.0
        
        # Normalize score to 0-1 range
        return max(0, min(1, score / 5.5))
    
    def _calculate_ranging_score(self, metrics: Dict[str, float]) -> float:
        """Calculate the score for ranging regime."""
        score = 0.0
        
        # Low directional strength indicates range
        score += (1 - metrics.get("directional_strength", 0)) * 1.5
        
        # Mean reversion is key characteristic of ranging markets
        score += metrics.get("mean_reversion", 0) * 2.0
        
        # Low volatility suggests range-bound
        score += (1 - metrics.get("volatility", 0)) * 1.0
        
        # Normalize score to 0-1 range
        return max(0, min(1, score / 4.5))
    
    def _calculate_volatile_score(self, metrics: Dict[str, float]) -> float:
        """Calculate the score for volatile regime."""
        score = 0.0
        
        # High volatility directly indicates volatile regime
        score += metrics.get("volatility", 0) * 2.0
        
        # Large candles indicate volatility
        score += metrics.get("avg_candle_size", 0) * 1.5
        
        # Low trend consistency can indicate volatility
        score += (1 - metrics.get("trend_consistency", 0)) * 1.0
        
        # Normalize score to 0-1 range
        return max(0, min(1, score / 4.5))


class AdaptiveParameterManager:
    """
    Manages strategy parameters that adapt to changing market conditions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adaptive parameter manager.
        
        Args:
            config: Configuration dictionary with parameters
                - base_parameters: Dictionary with base parameters for strategies
                - regime_adjustments: Dictionary with parameter adjustments for different regimes
                - optimization_method: Method for parameter optimization
        """
        self.logger = get_logger("AdaptiveParameterManager")
        self.config = config or {}
        
        # Extract configuration
        self.base_parameters = self.config.get("base_parameters", {
            "MA_Cross": {
                "fast_period": 9,
                "slow_period": 21,
                "signal_period": 9,
                "confirmation_periods": 2,
                "min_divergence": 0.001,
                "max_volatility": 0.05,
                "volatility_adjustment": 0.5
            },
            "RSI_OB_OS": {
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30,
                "confirmation_periods": 2,
                "buffer_zones": 5,
                "max_volatility": 0.05,
                "volatility_adjustment": 0.5
            }
        })
        
        self.regime_adjustments = self.config.get("regime_adjustments", {
            "trending": {
                "MA_Cross": {
                    "fast_period": -2,  # Faster response
                    "slow_period": +5,  # Longer trend capture
                    "confirmation_periods": -1,  # Less confirmation needed
                    "min_divergence": 0.0005,  # Lower threshold to catch trend
                    "max_volatility": 0.07  # Allow more volatility
                },
                "RSI_OB_OS": {
                    "overbought": +5,  # Higher threshold for strong trends
                    "oversold": -5,  # Lower threshold for strong trends
                    "confirmation_periods": -1,  # Less confirmation needed
                    "buffer_zones": -2  # Narrower buffer in trending markets
                }
            },
            "ranging": {
                "MA_Cross": {
                    "fast_period": +1,  # Slower response to avoid noise
                    "slow_period": -3,  # Shorter period to catch range
                    "confirmation_periods": +1,  # More confirmation needed
                    "min_divergence": 0.002  # Higher threshold for noise
                },
                "RSI_OB_OS": {
                    "overbought": -5,  # Lower to catch range reversals
                    "oversold": +5,  # Higher to catch range reversals
                    "confirmation_periods": 0,  # Standard confirmation
                    "buffer_zones": +3  # Wider buffer in ranging markets
                }
            },
            "volatile": {
                "MA_Cross": {
                    "fast_period": +3,  # Much slower to filter noise
                    "slow_period": +5,  # Much longer for stability
                    "confirmation_periods": +2,  # More confirmation needed
                    "min_divergence": 0.003,  # Higher threshold for noise
                    "max_volatility": 0.03,  # Lower volatility threshold
                    "volatility_adjustment": 0.7  # Higher adjustment
                },
                "RSI_OB_OS": {
                    "overbought": +10,  # Much higher threshold
                    "oversold": -10,  # Much lower threshold
                    "confirmation_periods": +2,  # More confirmation needed
                    "buffer_zones": +5,  # Wider buffer in volatile markets
                    "max_volatility": 0.03,  # Lower volatility threshold
                    "volatility_adjustment": 0.7  # Higher adjustment
                }
            }
        })
        
        self.optimization_method = self.config.get("optimization_method", "regime_based")
        self.performance_history = {}
        
        # Initialize regime classifier
        self.regime_classifier = MarketRegimeClassifier(
            self.config.get("regime_classifier", {})
        )
    
    def get_strategy_parameters(
        self, 
        strategy_name: str, 
        market_data: pd.DataFrame, 
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get optimized parameters for a strategy based on current market conditions.
        
        Args:
            strategy_name: Name of the strategy
            market_data: DataFrame with OHLCV data
            indicators: Dictionary with technical indicators
            
        Returns:
            Dictionary with optimized parameters
        """
        if strategy_name not in self.base_parameters:
            self.logger.warning(f"No base parameters for strategy {strategy_name}")
            return {}
            
        try:
            # Get base parameters
            params = self.base_parameters[strategy_name].copy()
            
            # Classify market regime
            regime_result = self.regime_classifier.classify_regime(market_data, indicators)
            regime = regime_result["regime"]
            confidence = regime_result["confidence"]
            
            # Apply regime-based adjustments
            if regime in self.regime_adjustments and strategy_name in self.regime_adjustments[regime]:
                adjustments = self.regime_adjustments[regime][strategy_name]
                
                for param, adjustment in adjustments.items():
                    if param in params:
                        # Apply adjustment with confidence weighting
                        weighted_adjustment = adjustment * confidence
                        params[param] += weighted_adjustment
                        
                        # Ensure integer parameters remain integers
                        if param in ["fast_period", "slow_period", "signal_period", 
                                    "rsi_period", "confirmation_periods"]:
                            params[param] = round(params[param])
                        
                        # Ensure parameters don't go below 1 for periods
                        if param.endswith("_period") and params[param] < 1:
                            params[param] = 1
            
            # Add metadata about adaptation
            params["_adaptive_metadata"] = {
                "regime": regime,
                "confidence": float(confidence),
                "original_parameters": self.base_parameters[strategy_name].copy(),
                "adjustment_method": self.optimization_method,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(
                f"Optimized parameters for {strategy_name} in {regime} regime "
                f"(confidence: {confidence:.4f})"
            )
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            return self.base_parameters.get(strategy_name, {})
    
    def update_performance(
        self, 
        strategy_name: str, 
        parameters: Dict[str, Any], 
        performance_metrics: Dict[str, float]
    ):
        """
        Update performance history for a set of parameters.
        
        Args:
            strategy_name: Name of the strategy
            parameters: Parameters used
            performance_metrics: Dictionary with performance metrics
        """
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
            
        # Store performance record
        record = {
            "parameters": {k: v for k, v in parameters.items() if not k.startswith("_")},
            "metrics": performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add regime information if available
        if "_adaptive_metadata" in parameters and "regime" in parameters["_adaptive_metadata"]:
            record["regime"] = parameters["_adaptive_metadata"]["regime"]
            
        self.performance_history[strategy_name].append(record)
        
        # Trim history if it gets too large
        if len(self.performance_history[strategy_name]) > 1000:
            self.performance_history[strategy_name] = self.performance_history[strategy_name][-1000:]
    
    def reset_to_defaults(self, strategy_name: str = None):
        """
        Reset parameters to default values.
        
        Args:
            strategy_name: Name of the strategy to reset, or None for all
        """
        if strategy_name is None:
            # Reset all strategies
            self.logger.info("Reset all strategy parameters to defaults")
        else:
            self.logger.info(f"Reset {strategy_name} parameters to defaults")
    
    def save_performance_history(self, file_path: str) -> bool:
        """
        Save performance history to a JSON file.
        
        Args:
            file_path: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert performance history to serializable format
            serializable_history = {}
            for strategy, history in self.performance_history.items():
                serializable_history[strategy] = []
                for record in history:
                    serializable_record = {
                        "parameters": {k: float(v) if isinstance(v, np.number) else v 
                                      for k, v in record["parameters"].items()},
                        "metrics": {k: float(v) if isinstance(v, np.number) else v 
                                   for k, v in record["metrics"].items()},
                        "timestamp": record["timestamp"]
                    }
                    if "regime" in record:
                        serializable_record["regime"] = record["regime"]
                    
                    serializable_history[strategy].append(serializable_record)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
                
            self.logger.info(f"Saved performance history to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving performance history: {str(e)}")
            return False
    
    def load_performance_history(self, file_path: str) -> bool:
        """
        Load performance history from a JSON file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"Performance history file not found: {file_path}")
                return False
                
            with open(file_path, 'r') as f:
                self.performance_history = json.load(f)
                
            self.logger.info(f"Loaded performance history from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading performance history: {str(e)}")
            return False
