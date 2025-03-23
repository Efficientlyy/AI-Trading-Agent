"""
Regime Boundary Proximity Factor

This module implements a confidence factor that evaluates prediction confidence
based on proximity to regime boundaries.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

from src.ml.detection.enhanced_ensemble.confidence.factors.base import ConfidenceFactor


class RegimeBoundaryProximityFactor(ConfidenceFactor):
    """
    Evaluates confidence based on proximity to regime boundaries.
    
    This factor assesses how close the current market conditions are to
    regime boundaries, where transitions are more likely and predictions
    are less certain.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the regime boundary proximity factor.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._boundary_metadata = {}
        
    def calculate(self, 
                detector_outputs: Dict[str, Dict[str, Any]],
                market_context: Dict[str, Any],
                historical_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate confidence based on proximity to regime boundaries.
        
        Args:
            detector_outputs: Dictionary mapping detector names to their outputs
            market_context: Dictionary containing current market conditions
            historical_data: Optional historical market data
            
        Returns:
            Confidence factor score between 0.0 and 1.0
        """
        # Clear previous metadata
        self._boundary_metadata = {}
        
        # Check if we have probability distributions
        has_probabilities = any('probabilities' in output 
                               for output in detector_outputs.values())
                               
        if has_probabilities:
            # Use probability distribution method if available
            proximity_score = self._calculate_from_probabilities(detector_outputs)
            method = "probability_distribution"
        elif historical_data is not None and not historical_data.empty:
            # Fall back to feature-based method if historical data available
            proximity_score = self._calculate_from_features(market_context, historical_data)
            method = "feature_based"
        else:
            # No good way to calculate boundary proximity
            self.logger.warning("Insufficient data for boundary proximity calculation")
            proximity_score = self.config.get('default_error_score', 0.5)
            method = "default"
            
        self._boundary_metadata['calculation_method'] = method
        self._boundary_metadata['proximity_score'] = proximity_score
            
        return proximity_score
    
    def _calculate_from_probabilities(self, 
                                    detector_outputs: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate boundary proximity using probability distributions.
        
        When probabilities are close for multiple regimes, we're likely near a boundary.
        
        Args:
            detector_outputs: Dictionary mapping detector names to their outputs
            
        Returns:
            Proximity-based confidence score between 0.0 and 1.0
        """
        # Collect all probability distributions
        distributions = []
        
        for name, output in detector_outputs.items():
            if 'probabilities' in output:
                distributions.append(output['probabilities'])
                
        if not distributions:
            return self.config.get('default_error_score', 0.5)
            
        # Calculate average margin between top probabilities for each distribution
        margins = []
        
        for dist in distributions:
            if not dist:
                continue
                
            # Sort probabilities in descending order
            sorted_probs = sorted(dist.values(), reverse=True)
            
            if len(sorted_probs) < 2:
                # Only one regime, can't calculate margin
                continue
                
            # Margin is difference between top two probabilities
            margin = sorted_probs[0] - sorted_probs[1]
            margins.append(margin)
            
        if not margins:
            return self.config.get('default_error_score', 0.5)
            
        # Average margin across all distributions
        avg_margin = sum(margins) / len(margins)
        
        # Store for metadata
        self._boundary_metadata['probability_margins'] = margins
        self._boundary_metadata['average_margin'] = avg_margin
        
        # Convert margin to confidence score
        # Small margin (close to 0) → low confidence (near boundary)
        # Large margin (close to 1) → high confidence (far from boundary)
        
        # Apply non-linear scaling to emphasize boundaries
        if avg_margin < 0.1:
            # Very close to boundary
            confidence = 0.2 + avg_margin * 3
        elif avg_margin < 0.3:
            # Somewhat close to boundary
            confidence = 0.5 + (avg_margin - 0.1) * 1.5
        else:
            # Far from boundary
            confidence = 0.8 + (avg_margin - 0.3) * 0.5
            
        return min(1.0, max(0.2, confidence))
    
    def _calculate_from_features(self, 
                               market_context: Dict[str, Any],
                               historical_data: pd.DataFrame) -> float:
        """
        Calculate boundary proximity using market features.
        
        This is a fallback method when probability distributions are not available.
        
        Args:
            market_context: Dictionary containing current market conditions
            historical_data: Historical market data
            
        Returns:
            Proximity-based confidence score between 0.0 and 1.0
        """
        # Extract key market features that indicate regime boundaries
        volatility = market_context.get('volatility')
        avg_volatility = market_context.get('average_volatility')
        trend_strength = market_context.get('trend_strength')
        
        # If missing key features, use the most recent data point
        if volatility is None or avg_volatility is None:
            if 'volatility' in historical_data.columns:
                volatility = historical_data['volatility'].iloc[-1]
                avg_volatility = historical_data['volatility'].mean()
            else:
                # No volatility data available
                self.logger.warning("No volatility data available for boundary proximity calculation")
                volatility = 0
                avg_volatility = 1
                
        if trend_strength is None:
            if 'trend_strength' in historical_data.columns:
                trend_strength = historical_data['trend_strength'].iloc[-1]
            else:
                # No trend strength data available
                self.logger.warning("No trend strength data available for boundary proximity calculation")
                trend_strength = 0.5
                
        # Store values for metadata
        self._boundary_metadata['feature_values'] = {
            'volatility': volatility,
            'avg_volatility': avg_volatility,
            'trend_strength': trend_strength
        }
        
        # Calculate volatility ratio (how current volatility compares to average)
        volatility_ratio = volatility / max(avg_volatility, 0.001)
        
        # Calculate confidence scores from features
        
        # 1. Volatility-based confidence:
        # - Very high or very low volatility often indicates regime boundaries
        if volatility_ratio > 2.0:
            # Volatility spike - likely near boundary
            volatility_confidence = 0.3
        elif volatility_ratio < 0.5:
            # Abnormally low volatility - could be calm before storm
            volatility_confidence = 0.7
        else:
            # Normal volatility range - more confident
            volatility_confidence = 0.9
            
        # 2. Trend-based confidence:
        # - Weak trends often occur near regime boundaries
        if trend_strength is not None:
            # Remap trend strength [0,1] to confidence
            # Strong trends (close to 1) -> high confidence
            # Weak trends (close to 0) -> low confidence
            trend_confidence = 0.5 + trend_strength * 0.5
        else:
            trend_confidence = 0.5
            
        # Combine confidence scores
        weights = self.config.get('feature_weights', {
            'volatility': 0.6,
            'trend': 0.4
        })
        
        confidence = (
            weights.get('volatility', 0.6) * volatility_confidence +
            weights.get('trend', 0.4) * trend_confidence
        )
        
        # Store combined confidence for metadata
        self._boundary_metadata['feature_confidence'] = {
            'volatility_confidence': volatility_confidence,
            'trend_confidence': trend_confidence,
            'combined_confidence': confidence
        }
        
        return confidence
            
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get additional metadata about this factor calculation.
        
        Returns:
            Dictionary of metadata that can be used for analysis and debugging
        """
        return {
            'boundary_data': self._boundary_metadata
        }
