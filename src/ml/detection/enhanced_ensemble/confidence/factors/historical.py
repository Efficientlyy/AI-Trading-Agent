"""
Historical Accuracy Factor

This module implements a confidence factor that evaluates prediction confidence
based on historical detector accuracy.
"""

from typing import Dict, Any, Optional, List, Tuple, DefaultDict
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

from src.ml.detection.enhanced_ensemble.confidence.factors.base import ConfidenceFactor


class HistoricalAccuracyFactor(ConfidenceFactor):
    """
    Evaluates confidence based on detector historical accuracy.
    
    This factor assesses how accurate detectors have been historically,
    with a focus on similar market conditions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the historical accuracy factor.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._window_size = self.config.get('window_size', 100)
        self._accuracy_history = defaultdict(lambda: deque(maxlen=self._window_size))
        self._regime_accuracy_history: DefaultDict[str, DefaultDict[str, deque]] = \
            defaultdict(lambda: defaultdict(lambda: deque(maxlen=self._window_size)))
        self._lock = threading.RLock()
        self._last_calculation_metadata = {}
    
    def calculate(self, 
                detector_outputs: Dict[str, Dict[str, Any]],
                market_context: Dict[str, Any],
                historical_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate confidence based on historical accuracy.
        
        Args:
            detector_outputs: Dictionary mapping detector names to their outputs
            market_context: Dictionary containing current market conditions
            historical_data: Optional historical market data
            
        Returns:
            Confidence factor score between 0.0 and 1.0
        """
        if not self._validate_inputs(detector_outputs, market_context):
            return self.config.get('default_error_score', 0.5)
            
        # Clear previous metadata
        self._last_calculation_metadata = {}
        
        # Get accuracy for each detector
        with self._lock:
            accuracies = {}
            for detector in detector_outputs.keys():
                accuracies[detector] = self.get_accuracy(detector)
                
            self._last_calculation_metadata["overall_accuracies"] = accuracies
            
            if not accuracies:
                return 0.5
            
            # Calculate weighted average accuracy based on market conditions
            current_regime = market_context.get('regime', 'unknown')
            regime_specific_accuracies = self._get_regime_specific_accuracies(
                detector_outputs.keys(), current_regime)
            
            self._last_calculation_metadata["regime_specific_accuracies"] = regime_specific_accuracies
            
            if regime_specific_accuracies:
                # Weight regime-specific accuracy higher when available
                overall_accuracy = 0.4 * np.mean(list(accuracies.values()))
                regime_accuracy = 0.6 * np.mean(list(regime_specific_accuracies.values()))
                final_score = overall_accuracy + regime_accuracy
            else:
                # Fall back to overall accuracy when regime-specific not available
                final_score = np.mean(list(accuracies.values()))
                
            # Apply recency weighting if enabled
            if self.config.get('apply_recency_weighting', True):
                recency_modifier = self._calculate_recency_modifier()
                final_score = final_score * recency_modifier
                self._last_calculation_metadata["recency_modifier"] = recency_modifier
            
            return min(1.0, max(0.0, final_score))
    
    def update_accuracy(self, detector: str, was_correct: bool, regime: Optional[str] = None) -> None:
        """
        Update historical accuracy for a detector.
        
        Args:
            detector: Name of the detector
            was_correct: Whether the prediction was correct
            regime: Optional regime for which this prediction was made
        """
        with self._lock:
            # Update overall accuracy
            self._accuracy_history[detector].append(was_correct)
            
            # Update regime-specific accuracy if regime is provided
            if regime:
                self._regime_accuracy_history[detector][regime].append(was_correct)
    
    def get_accuracy(self, detector: str) -> float:
        """
        Get the current accuracy for a detector.
        
        Args:
            detector: Name of the detector
            
        Returns:
            Historical accuracy between 0.0 and 1.0
        """
        with self._lock:
            history = self._accuracy_history[detector]
            if not history:
                return 0.5  # Default when no history
            return sum(history) / len(history)
    
    def get_regime_accuracy(self, detector: str, regime: str) -> float:
        """
        Get the accuracy for a detector in a specific regime.
        
        Args:
            detector: Name of the detector
            regime: Market regime
            
        Returns:
            Regime-specific accuracy between 0.0 and 1.0
        """
        with self._lock:
            history = self._regime_accuracy_history[detector][regime]
            if not history:
                return self.get_accuracy(detector)  # Fall back to overall accuracy
            return sum(history) / len(history)
    
    def _get_regime_specific_accuracies(self, 
                                      detectors: List[str],
                                      current_regime: str) -> Dict[str, float]:
        """
        Get regime-specific accuracies for the current regime.
        
        Args:
            detectors: List of detector names
            current_regime: Current market regime
            
        Returns:
            Dictionary mapping detector names to their regime-specific accuracies
        """
        regime_accuracies = {}
        
        for detector in detectors:
            # Only include if we have regime-specific history
            if current_regime in self._regime_accuracy_history[detector] and \
               len(self._regime_accuracy_history[detector][current_regime]) > 0:
                regime_accuracies[detector] = self.get_regime_accuracy(detector, current_regime)
                
        return regime_accuracies
    
    def _calculate_recency_modifier(self) -> float:
        """
        Calculate a recency modifier based on how recent our accuracy data is.
        
        Less recent data should reduce confidence slightly.
        
        Returns:
            Recency modifier between 0.5 and 1.0
        """
        # This is a placeholder implementation that could be enhanced with actual
        # timestamps of when accuracy data was collected
        return 1.0
            
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get additional metadata about this factor calculation.
        
        Returns:
            Dictionary of metadata that can be used for analysis and debugging
        """
        with self._lock:
            history_counts = {detector: len(history) 
                            for detector, history in self._accuracy_history.items()}
                            
            regime_history_counts = {
                detector: {
                    regime: len(history)
                    for regime, history in regime_histories.items()
                }
                for detector, regime_histories in self._regime_accuracy_history.items()
            }
            
            return {
                'history_counts': history_counts,
                'regime_history_counts': regime_history_counts,
                'calculation_data': self._last_calculation_metadata
            }
