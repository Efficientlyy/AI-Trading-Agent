"""
Temporal Pattern Recognition Module

This module integrates different temporal pattern analysis components including:
1. Seasonality detection
2. Regime transition probability modeling
3. Multi-timeframe confirmation logic

Together, these components allow the trading system to detect and predict
temporal patterns in market regimes, enhancing the adaptability and
robustness of trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta

from .core_definitions import (
    MarketRegimeType,
    MarketRegimeInfo,
    MarketRegimeConfig,
    RegimeChangeSignificance
)
from .regime_classifier import MarketRegimeClassifier
from .seasonality import SeasonalityDetector
from .transition_probability import TransitionProbabilityModel
from .multi_timeframe import MultiTimeframeConfirmation

# Set up logger
logger = logging.getLogger(__name__)


class TemporalPatternRecognition:
    """
    Main class for temporal pattern recognition in market regimes.
    
    Integrates seasonality detection, regime transition probability modeling,
    and multi-timeframe confirmation logic to provide comprehensive
    temporal pattern analysis for market regimes.
    """
    
    def __init__(self, config: Optional[MarketRegimeConfig] = None):
        """
        Initialize the temporal pattern recognition system.
        
        Args:
            config: Configuration for market regime detection and analysis
        """
        self.config = config or MarketRegimeConfig()
        
        # Initialize component systems
        self.seasonality_detector = SeasonalityDetector()
        self.transition_model = TransitionProbabilityModel()
        self.multi_timeframe = MultiTimeframeConfirmation(config=self.config)
        
        # For standalone regime detection
        self.regime_classifier = MarketRegimeClassifier(config=self.config)
        
        # Result history
        self.analysis_history = {}
    
    def analyze_temporal_patterns(self,
                                 prices: pd.Series,
                                 asset_id: str = "default",
                                 volumes: Optional[pd.Series] = None,
                                 ohlcv_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform comprehensive temporal pattern analysis on market data.
        
        Args:
            prices: Series of price data
            asset_id: Identifier for the asset
            volumes: Optional volume data
            ohlcv_data: Optional OHLCV DataFrame for multi-timeframe analysis
            
        Returns:
            Dictionary with comprehensive temporal pattern analysis
        """
        if prices is None or len(prices) < 60:
            logger.warning(f"Insufficient data for temporal pattern analysis")
            return {
                "current_regime": MarketRegimeType.UNKNOWN.value,
                "has_seasonality": False,
                "next_regime_probabilities": {},
                "timeframe_agreement": 0.0
            }
        
        # 1. Current regime classification
        regime_info = self.regime_classifier.classify_regime(
            prices=prices,
            volumes=volumes,
            asset_id=asset_id
        )
        
        # Add to transition model history
        self.transition_model.add_regime_observation(regime_info, asset_id)
        
        # 2. Seasonality detection
        seasonality = self.seasonality_detector.detect_seasonality(
            series=prices,
            asset_id=asset_id
        )
        
        # 3. Regime transition probabilities
        # Build transition matrix if we have enough history
        if len(self.transition_model.regime_history.get(asset_id, [])) >= 30:
            transition_matrix = self.transition_model.build_transition_matrix(asset_id)
            next_regime_probs = self.transition_model.predict_next_regime(
                current_regime=regime_info.regime_type,
                asset_id=asset_id
            )
            most_likely_transition, transition_prob = self.transition_model.get_most_likely_transition(
                current_regime=regime_info.regime_type,
                asset_id=asset_id
            )
        else:
            next_regime_probs = {}
            most_likely_transition = None
            transition_prob = 0.0
        
        # 4. Multi-timeframe confirmation (if OHLCV data provided)
        if ohlcv_data is not None and isinstance(ohlcv_data, pd.DataFrame):
            multi_tf_analysis = self.multi_timeframe.analyze_multi_timeframe(
                data=ohlcv_data,
                asset_id=asset_id,
                volumes=volumes
            )
            timeframe_agreement = self.multi_timeframe.get_timeframe_agreement(asset_id)
            divergence = self.multi_timeframe.detect_divergence(asset_id)
            timeframe_transitions = self.multi_timeframe.get_timeframe_transition_signals(asset_id)
        else:
            multi_tf_analysis = None
            timeframe_agreement = 0.0
            divergence = {"has_divergence": False}
            timeframe_transitions = {}
        
        # 5. Seasonal forecast if seasonality detected
        if seasonality.get("has_seasonality", False):
            seasonal_forecast = self.seasonality_detector.get_seasonal_forecast(
                series=prices,
                asset_id=asset_id,
                forecast_periods=20
            )
        else:
            seasonal_forecast = {"forecast": None}
        
        # Compile comprehensive result
        result = {
            "timestamp": pd.Timestamp.now(),
            "asset_id": asset_id,
            "current_regime": {
                "regime_type": regime_info.regime_type.value,
                "volatility_regime": regime_info.volatility_regime.value,
                "liquidity_regime": regime_info.liquidity_regime.value,
                "confidence": regime_info.confidence,
                "regime_change": regime_info.regime_change.value if regime_info.regime_change else None
            },
            "seasonality": {
                "has_seasonality": seasonality.get("has_seasonality", False),
                "seasonal_periods": seasonality.get("acf_results", {}).get("seasonal_periods", []),
                "calendar_patterns": seasonality.get("calendar_patterns", {})
            },
            "transition_probabilities": {
                "next_regime_probabilities": next_regime_probs,
                "most_likely_transition": most_likely_transition,
                "transition_probability": transition_prob
            },
            "multi_timeframe": {
                "confirmed_regime": multi_tf_analysis.get("confirmed_regime") if multi_tf_analysis else None,
                "agreement_score": multi_tf_analysis.get("agreement_score") if multi_tf_analysis else 0.0,
                "timeframe_regimes": multi_tf_analysis.get("timeframe_regimes") if multi_tf_analysis else {},
                "has_divergence": divergence.get("has_divergence", False),
                "timeframe_transitions": timeframe_transitions
            },
            "forecasts": {
                "seasonal_forecast": seasonal_forecast.get("forecast")
            }
        }
        
        # Store in history
        if asset_id not in self.analysis_history:
            self.analysis_history[asset_id] = []
            
        self.analysis_history[asset_id].append(result)
        
        # Trim history if too long
        if len(self.analysis_history[asset_id]) > 100:
            self.analysis_history[asset_id] = self.analysis_history[asset_id][-100:]
            
        return result
    
    def get_regime_continuation_probability(self, 
                                          current_regime: MarketRegimeType, 
                                          asset_id: str = "default") -> float:
        """
        Get the probability that the current regime will continue.
        
        Args:
            current_regime: Current market regime
            asset_id: Identifier for the asset
            
        Returns:
            Probability of regime continuation
        """
        # Build transition matrix if needed
        if asset_id not in self.transition_model.transition_matrices:
            self.transition_model.build_transition_matrix(asset_id)
            
        # If no transition matrix available, return default
        if asset_id not in self.transition_model.transition_matrices:
            return 0.5
            
        matrix = self.transition_model.transition_matrices[asset_id]["matrix"]
        
        # Check if current regime is in matrix
        if current_regime.value in matrix and current_regime.value in matrix[current_regime.value]:
            return matrix[current_regime.value][current_regime.value]
        else:
            return 0.5
    
    def get_timeframe_alignment_signal(self, 
                                     asset_id: str = "default", 
                                     min_agreement: float = 0.75) -> Dict[str, Any]:
        """
        Detect when multiple timeframes align on the same regime.
        
        This can be a powerful confirmation signal for trading decisions.
        
        Args:
            asset_id: Identifier for the asset
            min_agreement: Minimum agreement score to generate signal
            
        Returns:
            Dictionary with alignment signal information
        """
        if asset_id not in self.multi_timeframe.confirmation_history:
            return {
                "has_alignment": False,
                "aligned_regime": None,
                "agreement_score": 0.0
            }
            
        # Get most recent confirmation
        recent = self.multi_timeframe.confirmation_history[asset_id][-1]
        
        # Check for sufficient agreement
        if recent["agreement_score"] >= min_agreement:
            return {
                "has_alignment": True,
                "aligned_regime": recent["confirmed_regime"],
                "agreement_score": recent["agreement_score"],
                "aligned_timeframes": [
                    tf for tf, regime in recent["timeframe_regimes"].items() 
                    if regime == recent["confirmed_regime"]
                ]
            }
        else:
            return {
                "has_alignment": False,
                "aligned_regime": None,
                "agreement_score": recent["agreement_score"]
            }
    
    def detect_regime_transition_opportunity(self, asset_id: str = "default") -> Dict[str, Any]:
        """
        Detect potential regime transition opportunities based on combined signals.
        
        Looks for alignment between:
        1. Current regime transition probability
        2. Seasonal pattern transitions
        3. Multi-timeframe leading indicators
        
        Args:
            asset_id: Identifier for the asset
            
        Returns:
            Dictionary with transition opportunity information
        """
        # Check for sufficient history
        if asset_id not in self.analysis_history or len(self.analysis_history[asset_id]) < 2:
            return {
                "transition_opportunity": False,
                "confidence": 0.0
            }
            
        current_analysis = self.analysis_history[asset_id][-1]
        
        # 1. Check transition probability
        transition_prob = current_analysis["transition_probabilities"]["transition_probability"]
        next_regime = current_analysis["transition_probabilities"]["most_likely_transition"]
        
        # 2. Check for timeframe transitions in shorter timeframes
        tf_transitions = current_analysis["multi_timeframe"]["timeframe_transitions"]
        has_shorter_tf_transition = False
        
        # Check if shorter timeframes are showing transitions
        for tf in ["1H", "4H", "1D"]:
            if tf in tf_transitions and tf_transitions[tf]["to"] == next_regime:
                has_shorter_tf_transition = True
                break
                
        # 3. Check for seasonal pattern alignment
        seasonal_alignment = False
        if current_analysis["seasonality"]["has_seasonality"]:
            # Simple check - are we at a period boundary for a detected seasonal pattern?
            for period in current_analysis["seasonality"]["seasonal_periods"]:
                if period.get("period") and len(self.analysis_history[asset_id]) % period["period"] == 0:
                    seasonal_alignment = True
                    break
        
        # Calculate opportunity confidence
        confidence = 0.0
        if next_regime:
            confidence = transition_prob * 0.4  # Base 40% on transition probability
            if has_shorter_tf_transition:
                confidence += 0.3  # Add 30% if shorter timeframes confirm
            if seasonal_alignment:
                confidence += 0.2  # Add 20% if seasonal pattern aligns
        
        return {
            "transition_opportunity": confidence > 0.5,
            "confidence": confidence,
            "potential_next_regime": next_regime,
            "current_regime": current_analysis["current_regime"]["regime_type"],
            "has_timeframe_confirmation": has_shorter_tf_transition,
            "has_seasonal_alignment": seasonal_alignment
        }


class TemporalPatternOptimizer:
    """
    Class for optimizing strategies based on temporal pattern recognition.
    
    Uses pattern recognition to adjust strategy parameters dynamically
    based on detected regimes, transitions, and seasonal patterns.
    """
    
    def __init__(self, 
                pattern_recognizer: TemporalPatternRecognition,
                strategy_parameters: Dict[str, Dict[str, Any]] = None):
        """
        Initialize the temporal pattern optimizer.
        
        Args:
            pattern_recognizer: TemporalPatternRecognition instance
            strategy_parameters: Dictionary mapping regimes to optimal parameters
        """
        self.pattern_recognizer = pattern_recognizer
        
        # Default strategy parameters by regime if none provided
        self.strategy_parameters = strategy_parameters or self._default_parameters()
        
        # Track parameter history
        self.parameter_history = {}
    
    def _default_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get default strategy parameters for different regimes.
        
        Returns:
            Dictionary mapping regimes to optimal parameters
        """
        return {
            MarketRegimeType.BULL.value: {
                "stop_loss_atr_multiple": 3.0,
                "take_profit_atr_multiple": 6.0,
                "entry_momentum_threshold": 0.2,
                "position_size_pct": 1.0
            },
            MarketRegimeType.BEAR.value: {
                "stop_loss_atr_multiple": 2.0,
                "take_profit_atr_multiple": 4.0,
                "entry_momentum_threshold": 0.3,
                "position_size_pct": 0.5
            },
            MarketRegimeType.SIDEWAYS.value: {
                "stop_loss_atr_multiple": 2.5,
                "take_profit_atr_multiple": 2.5,
                "entry_momentum_threshold": 0.4,
                "position_size_pct": 0.5
            },
            MarketRegimeType.VOLATILE.value: {
                "stop_loss_atr_multiple": 4.0,
                "take_profit_atr_multiple": 3.0,
                "entry_momentum_threshold": 0.5,
                "position_size_pct": 0.3
            },
            MarketRegimeType.TRENDING.value: {
                "stop_loss_atr_multiple": 3.0,
                "take_profit_atr_multiple": 5.0,
                "entry_momentum_threshold": 0.3,
                "position_size_pct": 0.8
            },
        }
    
    def get_optimized_parameters(self,
                               asset_id: str = "default",
                               base_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get optimized strategy parameters based on detected temporal patterns.
        
        Args:
            asset_id: Identifier for the asset
            base_parameters: Base parameters to adjust (or use defaults)
            
        Returns:
            Dictionary with optimized strategy parameters
        """
        # Use provided base parameters or empty dict
        parameters = base_parameters.copy() if base_parameters else {}
        
        # Check if we have analysis for this asset
        if asset_id not in self.pattern_recognizer.analysis_history:
            logger.warning(f"No temporal pattern analysis available for {asset_id}")
            return parameters
        
        # Get most recent analysis
        analysis = self.pattern_recognizer.analysis_history[asset_id][-1]
        
        # Get current regime and multi-timeframe confirmation
        current_regime = analysis["current_regime"]["regime_type"]
        multi_tf = analysis["multi_timeframe"]
        
        # 1. Base parameters on current regime
        if current_regime in self.strategy_parameters:
            regime_params = self.strategy_parameters[current_regime].copy()
            for param, value in regime_params.items():
                if param not in parameters:
                    parameters[param] = value
        
        # 2. Adjust for multi-timeframe agreement/disagreement
        if multi_tf and "agreement_score" in multi_tf:
            agreement = multi_tf["agreement_score"]
            
            # Higher confidence with strong agreement
            if agreement > 0.8:
                # Increase position size with high confidence
                if "position_size_pct" in parameters:
                    parameters["position_size_pct"] *= min(1.2, 1 + (agreement - 0.8) * 2)
                    
                # Tighter stops with high confidence
                if "stop_loss_atr_multiple" in parameters:
                    parameters["stop_loss_atr_multiple"] *= max(0.9, 1 - (agreement - 0.8))
                
            # Lower confidence with disagreement
            elif agreement < 0.5:
                # Reduce position size with low confidence
                if "position_size_pct" in parameters:
                    parameters["position_size_pct"] *= max(0.5, agreement)
                
                # Wider stops with low confidence
                if "stop_loss_atr_multiple" in parameters:
                    parameters["stop_loss_atr_multiple"] *= min(1.5, 1 + (0.5 - agreement))
        
        # 3. Adjust for seasonality
        seasonality = analysis["seasonality"]
        if seasonality["has_seasonality"] and seasonality["seasonal_periods"]:
            # If we have strong seasonality, adjust profit targets
            if seasonality["seasonal_periods"][0].get("acf_value", 0) > 0.5:
                # Extend profit targets to capture seasonal moves
                if "take_profit_atr_multiple" in parameters:
                    parameters["take_profit_atr_multiple"] *= 1.2
        
        # 4. Adjust for transition probability
        if analysis["transition_probabilities"]["transition_probability"] > 0.6:
            # Higher risk of regime change ahead
            # Reduce position size and tighten take profit
            if "position_size_pct" in parameters:
                parameters["position_size_pct"] *= 0.8
            
            if "take_profit_atr_multiple" in parameters:
                parameters["take_profit_atr_multiple"] *= 0.8
        
        # Store in history
        if asset_id not in self.parameter_history:
            self.parameter_history[asset_id] = []
            
        self.parameter_history[asset_id].append({
            "timestamp": datetime.now().isoformat(),
            "parameters": parameters.copy(),
            "regime": current_regime,
            "agreement_score": multi_tf.get("agreement_score", 0)
        })
        
        # Trim history if too long
        if len(self.parameter_history[asset_id]) > 100:
            self.parameter_history[asset_id] = self.parameter_history[asset_id][-100:]
        
        return parameters
