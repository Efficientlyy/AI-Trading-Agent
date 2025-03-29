"""Adaptive weights module for sentiment analysis.

This module provides functionality for adjusting sentiment source weights
based on historical performance to improve prediction accuracy.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class AdaptiveSentimentWeights:
    """Adjusts sentiment source weights based on historical performance."""
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 decay_factor: float = 0.95,
                 performance_window: int = 30,
                 min_samples: int = 10):
        """Initialize the adaptive weights system.
        
        Args:
            learning_rate: Rate of weight adjustment
            decay_factor: Factor for time-based decay of old performance data
            performance_window: Days to keep performance data
            min_samples: Minimum samples needed before adjustment
        """
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.performance_window = performance_window
        self.min_samples = min_samples
        
        # Default source weights
        self.source_weights = {
            "social_media": 0.25,
            "news": 0.25,
            "market_sentiment": 0.3,
            "onchain": 0.2
        }
        
        # Performance history
        self.source_performance: Dict[str, List[Dict[str, Any]]] = {
            "social_media": [],
            "news": [],
            "market_sentiment": [],
            "onchain": []
        }
        
        # Last update timestamp
        self.last_update = datetime.utcnow()
        
    def record_performance(self, 
                          source: str, 
                          prediction: float,
                          actual_outcome: float,
                          timestamp: datetime) -> None:
        """Record the performance of a sentiment source prediction.
        
        Args:
            source: The sentiment source
            prediction: The predicted sentiment value (0-1)
            actual_outcome: The actual market movement normalized to 0-1
            timestamp: When the prediction was made
        """
        if source not in self.source_performance:
            self.source_performance[source] = []
            
        # Calculate accuracy (inverted mean squared error)
        error = (prediction - actual_outcome) ** 2
        accuracy = max(0, 1 - error)
        
        # Record performance
        self.source_performance[source].append({
            "prediction": prediction,
            "actual": actual_outcome,
            "accuracy": accuracy,
            "timestamp": timestamp
        })
        
        # Clean up old performance data
        cutoff = datetime.utcnow() - timedelta(days=self.performance_window)
        self.source_performance[source] = [
            p for p in self.source_performance[source]
            if p["timestamp"] > cutoff
        ]
        
    def update_weights(self) -> Dict[str, float]:
        """Update weights based on recent performance.
        
        Returns:
            The updated source weights
        """
        # Check if we have enough data
        has_enough_data = all(
            len(perfs) >= self.min_samples
            for perfs in self.source_performance.values()
        )
        
        if not has_enough_data:
            return self.source_weights
            
        # Calculate performance-based adjustments
        adjustments = {}
        total_adjustment = 0
        
        for source, performances in self.source_performance.items():
            if not performances:
                continue
                
            # Calculate recent average accuracy with time decay
            now = datetime.utcnow()
            weighted_sum = 0
            weight_sum = 0
            
            for perf in performances:
                # More recent performances get higher weight
                days_old = (now - perf["timestamp"]).total_seconds() / 86400
                time_weight = self.decay_factor ** days_old
                
                weighted_sum += perf["accuracy"] * time_weight
                weight_sum += time_weight
                
            if weight_sum > 0:
                avg_accuracy = weighted_sum / weight_sum
            else:
                avg_accuracy = 0.5
                
            # Calculate adjustment (positive if accuracy > 0.5, negative otherwise)
            current_weight = self.source_weights.get(source, 0.25)
            target_weight = current_weight * (1 + (avg_accuracy - 0.5) * self.learning_rate)
            
            # Ensure weight stays in reasonable range
            target_weight = max(0.05, min(0.5, target_weight))
            
            adjustments[source] = target_weight - current_weight
            total_adjustment += abs(adjustments[source])
            
        # Apply adjustments while maintaining sum of weights = 1
        if total_adjustment > 0:
            # Apply raw adjustments first
            for source in self.source_weights:
                if source in adjustments:
                    self.source_weights[source] += adjustments[source]
                    
            # Normalize weights to sum to 1
            weight_sum = sum(self.source_weights.values())
            for source in self.source_weights:
                self.source_weights[source] /= weight_sum
                
        # Update timestamp
        self.last_update = datetime.utcnow()
                
        return self.source_weights
        
    def get_weights(self) -> Dict[str, float]:
        """Get the current source weights.
        
        Returns:
            The current source weights
        """
        # Update weights if it's been more than a day
        if (datetime.utcnow() - self.last_update) > timedelta(days=1):
            self.update_weights()
            
        return self.source_weights
