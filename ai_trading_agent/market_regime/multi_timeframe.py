"""
Multi-Timeframe Confirmation Module

This module provides tools for confirming market regime classifications
across multiple timeframes, enhancing the reliability of regime detection
and reducing false signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
from collections import Counter

from .core_definitions import (
    MarketRegimeType,
    MarketRegimeInfo,
    MarketRegimeConfig,
    RegimeChangeSignificance
)
from .regime_classifier import MarketRegimeClassifier

# Set up logger
logger = logging.getLogger(__name__)


class MultiTimeframeConfirmation:
    """
    Class for confirming market regime classifications across multiple timeframes.
    
    Uses hierarchical timeframe analysis to validate regime classifications
    and reduce false signals, improving the robustness of the system.
    """
    
    def __init__(self,
               timeframes: Optional[List[str]] = None,
               weights: Optional[Dict[str, float]] = None,
               config: Optional[MarketRegimeConfig] = None):
        """
        Initialize the multi-timeframe confirmation system.
        
        Args:
            timeframes: List of timeframes to analyze (e.g. ["1D", "1W", "1M"])
            weights: Dictionary mapping timeframes to their weights in analysis
            config: Configuration for market regime detection
        """
        self.timeframes = timeframes or ["1H", "4H", "1D", "1W"]
        self.weights = weights or self._default_weights()
        self.config = config or MarketRegimeConfig()
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        for tf in self.weights:
            self.weights[tf] /= total_weight
        
        # Create regime classifiers for each timeframe
        self.classifiers = {
            tf: MarketRegimeClassifier(config=self.config) for tf in self.timeframes
        }
        
        # Store results history
        self.confirmation_history = {}
    
    def _default_weights(self) -> Dict[str, float]:
        """
        Get default weights for timeframes based on common best practices.
        
        Returns:
            Dictionary mapping timeframes to weights
        """
        weights = {}
        for tf in self.timeframes:
            # Common prioritization: higher weight to longer timeframes
            if tf in ["1M", "1mo", "monthly"]:
                weights[tf] = 4.0
            elif tf in ["1W", "1w", "weekly"]:
                weights[tf] = 3.0
            elif tf in ["1D", "1d", "daily"]:
                weights[tf] = 2.0
            elif tf in ["4H", "4h"]:
                weights[tf] = 1.5
            elif tf in ["1H", "1h", "hourly"]:
                weights[tf] = 1.0
            else:
                weights[tf] = 1.0
                
        return weights
    
    def _resample_data(self, 
                      data: pd.DataFrame, 
                      timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to the specified timeframe.
        
        Args:
            data: OHLCV DataFrame
            timeframe: Target timeframe for resampling
            
        Returns:
            Resampled DataFrame
        """
        # Convert timeframe string to pandas offset alias
        offset = self._timeframe_to_offset(timeframe)
        
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("Data must have a DatetimeIndex for resampling")
            return data
        
        try:
            # Resample data
            resampled = data.resample(offset).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data to {timeframe}: {str(e)}")
            return data
    
    def _timeframe_to_offset(self, timeframe: str) -> str:
        """
        Convert timeframe string to pandas offset alias.
        
        Args:
            timeframe: Timeframe string (e.g. "1D", "4H", "1W")
            
        Returns:
            Pandas offset alias
        """
        # Extract number and unit
        import re
        match = re.match(r'(\d+)([a-zA-Z]+)', timeframe)
        
        if not match:
            logger.warning(f"Invalid timeframe format: {timeframe}, using '1D' as default")
            return '1D'
            
        number, unit = match.groups()
        
        # Map to pandas offset alias
        unit_map = {
            'M': 'M',  # Month
            'mo': 'M',
            'month': 'M',
            'monthly': 'M',
            'W': 'W',  # Week
            'w': 'W',
            'week': 'W',
            'weekly': 'W',
            'D': 'D',  # Day
            'd': 'D',
            'day': 'D',
            'daily': 'D',
            'H': 'H',  # Hour
            'h': 'H',
            'hour': 'H',
            'hourly': 'H',
            'min': 'min',  # Minute
            'm': 'min',
            'minute': 'min'
        }
        
        if unit.lower() in unit_map:
            return f"{number}{unit_map[unit.lower()]}"
        else:
            logger.warning(f"Unknown timeframe unit: {unit}, using '1D' as default")
            return '1D'
    
    def analyze_multi_timeframe(self,
                               data: pd.DataFrame,
                               asset_id: str = "default",
                               volumes: Optional[pd.Series] = None) -> Dict[str, any]:
        """
        Analyze market regime across multiple timeframes for confirmation.
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
            asset_id: Identifier for the asset
            volumes: Optional volume data if not in DataFrame
            
        Returns:
            Dictionary with multi-timeframe analysis results
        """
        if data is None or data.empty:
            logger.warning(f"Empty data provided for multi-timeframe analysis")
            return {
                "confirmed_regime": MarketRegimeType.UNKNOWN.value,
                "agreement_score": 0.0,
                "timeframe_regimes": {}
            }
            
        # Check data format
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("Data must have a DatetimeIndex for multi-timeframe analysis")
            return {
                "confirmed_regime": MarketRegimeType.UNKNOWN.value,
                "agreement_score": 0.0,
                "timeframe_regimes": {}
            }
            
        # Extract volume data if not provided separately
        if volumes is None and 'Volume' in data.columns:
            volumes = data['Volume']
            
        # Prepare result containers
        timeframe_regimes = {}
        regime_votes = Counter()
        
        # Analyze each timeframe
        for tf in self.timeframes:
            try:
                # Resample data to this timeframe
                resampled_data = self._resample_data(data, tf)
                
                # Resample volumes if available
                resampled_volumes = None
                if volumes is not None:
                    if isinstance(volumes, pd.Series):
                        resampled_volumes = volumes.resample(self._timeframe_to_offset(tf)).sum()
                    else:
                        resampled_volumes = volumes
                
                # Detect regime for this timeframe
                if 'Close' in resampled_data.columns:
                    prices = resampled_data['Close']
                else:
                    prices = resampled_data.iloc[:, 0]  # Use first column if no Close
                    
                classifier = self.classifiers[tf]
                regime_info = classifier.classify_regime(
                    prices=prices,
                    volumes=resampled_volumes,
                    asset_id=f"{asset_id}_{tf}"
                )
                
                # Store result
                timeframe_regimes[tf] = {
                    "regime": regime_info.regime_type.value,
                    "confidence": regime_info.confidence,
                    "volatility_regime": regime_info.volatility_regime.value,
                    "data_points": len(resampled_data)
                }
                
                # Add weighted vote for this regime
                regime_votes[regime_info.regime_type.value] += (
                    self.weights[tf] * regime_info.confidence
                )
                
            except Exception as e:
                logger.error(f"Error analyzing timeframe {tf}: {str(e)}")
                timeframe_regimes[tf] = {
                    "regime": MarketRegimeType.UNKNOWN.value,
                    "error": str(e)
                }
        
        # Determine the confirmed regime (highest weighted vote)
        confirmed_regime = MarketRegimeType.UNKNOWN.value
        highest_vote = 0.0
        
        for regime, vote in regime_votes.items():
            if vote > highest_vote:
                highest_vote = vote
                confirmed_regime = regime
                
        # Calculate agreement score (0 to 1)
        total_weight = sum(self.weights.values())
        max_score = sum(self.weights[tf] for tf in timeframe_regimes.keys() 
                       if timeframe_regimes[tf].get("regime") != MarketRegimeType.UNKNOWN.value)
        
        if max_score > 0:
            agreement_score = regime_votes[confirmed_regime] / max_score
        else:
            agreement_score = 0.0
        
        # Store in history
        if asset_id not in self.confirmation_history:
            self.confirmation_history[asset_id] = []
            
        self.confirmation_history[asset_id].append({
            "timestamp": data.index[-1],
            "confirmed_regime": confirmed_regime,
            "agreement_score": agreement_score,
            "timeframe_regimes": {tf: info["regime"] for tf, info in timeframe_regimes.items()}
        })
        
        # Trim history if too long
        if len(self.confirmation_history[asset_id]) > 100:
            self.confirmation_history[asset_id] = self.confirmation_history[asset_id][-100:]
            
        # Return results
        return {
            "confirmed_regime": confirmed_regime,
            "agreement_score": agreement_score,
            "timeframe_regimes": timeframe_regimes,
            "regime_votes": {k: float(v) for k, v in regime_votes.items()}
        }
    
    def get_timeframe_agreement(self, asset_id: str = "default") -> float:
        """
        Calculate the agreement between timeframes based on recent history.
        
        Args:
            asset_id: Identifier for the asset
            
        Returns:
            Agreement score (0 to 1)
        """
        if asset_id not in self.confirmation_history or not self.confirmation_history[asset_id]:
            return 0.0
            
        # Get average agreement score from recent history
        recent_scores = [item["agreement_score"] for item in self.confirmation_history[asset_id][-5:]]
        return sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
    
    def detect_divergence(self, asset_id: str = "default") -> Dict[str, any]:
        """
        Detect divergence between different timeframes.
        
        Args:
            asset_id: Identifier for the asset
            
        Returns:
            Dictionary with divergence analysis
        """
        if asset_id not in self.confirmation_history or not self.confirmation_history[asset_id]:
            return {
                "has_divergence": False,
                "divergence_score": 0.0,
                "divergent_timeframes": []
            }
            
        # Get most recent confirmation
        recent = self.confirmation_history[asset_id][-1]
        
        # Check for divergence
        confirmed_regime = recent["confirmed_regime"]
        divergent_timeframes = []
        
        for tf, regime in recent["timeframe_regimes"].items():
            if regime != confirmed_regime and regime != MarketRegimeType.UNKNOWN.value:
                divergent_timeframes.append({
                    "timeframe": tf,
                    "regime": regime
                })
                
        # Calculate divergence score
        divergence_score = 1.0 - recent["agreement_score"]
        
        return {
            "has_divergence": len(divergent_timeframes) > 0,
            "divergence_score": divergence_score,
            "divergent_timeframes": divergent_timeframes,
            "confirmed_regime": confirmed_regime
        }
    
    def get_timeframe_transition_signals(self, asset_id: str = "default") -> Dict[str, List[str]]:
        """
        Get transition signals from different timeframes.
        
        Transition signals occur when a timeframe changes regime,
        and can provide early indicators of larger trend changes.
        
        Args:
            asset_id: Identifier for the asset
            
        Returns:
            Dictionary mapping timeframes to their transition signals
        """
        if asset_id not in self.confirmation_history or len(self.confirmation_history[asset_id]) < 2:
            return {}
            
        # Look for regime changes in each timeframe
        transitions = {}
        history = self.confirmation_history[asset_id]
        
        # Extract the last two observations
        current = history[-1]["timeframe_regimes"]
        previous = history[-2]["timeframe_regimes"]
        
        # Check for transitions
        for tf in self.timeframes:
            if tf in current and tf in previous:
                current_regime = current[tf]
                prev_regime = previous[tf]
                
                if current_regime != prev_regime and current_regime != MarketRegimeType.UNKNOWN.value:
                    transitions[tf] = {
                        "from": prev_regime,
                        "to": current_regime,
                        "timeframe": tf
                    }
                    
        return transitions
