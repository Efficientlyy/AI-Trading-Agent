"""Sentiment validator module.

This module provides functionality for validating sentiment data to filter out noise
and detect anomalies in sentiment signals.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

class SentimentValidator:
    """Validates sentiment data to filter out noise and detect anomalies."""
    
    def __init__(self, 
                 anomaly_threshold: float = 3.0,
                 min_credibility: float = 0.5,
                 history_size: int = 100):
        """Initialize the sentiment validator.
        
        Args:
            anomaly_threshold: Z-score threshold for anomaly detection
            min_credibility: Minimum source credibility score
            history_size: Maximum size of historical data to keep
        """
        self.anomaly_threshold = anomaly_threshold
        self.min_credibility = min_credibility
        self.history_size = history_size
        self.historical_data: Dict[str, Dict[str, Any]] = {}
        self.source_credibility: Dict[str, float] = {
            "social_media": 0.7,
            "news": 0.8,
            "market_sentiment": 0.9,
            "onchain": 0.85
        }
        
    def validate_sentiment(self, 
                          symbol: str, 
                          source: str, 
                          sentiment_value: float,
                          confidence: float,
                          timestamp: datetime) -> Tuple[bool, str, float]:
        """Validate sentiment data.
        
        Args:
            symbol: The trading pair symbol
            source: The sentiment source
            sentiment_value: The sentiment value (0-1)
            confidence: The confidence value (0-1)
            timestamp: The timestamp of the sentiment data
            
        Returns:
            Tuple of (is_valid, reason, adjusted_confidence)
        """
        # Check source credibility
        source_cred = self.source_credibility.get(source, 0.6)
        if source_cred < self.min_credibility:
            return False, f"Source credibility below threshold: {source_cred:.2f}", confidence
            
        # Initialize historical data for this symbol if needed
        if symbol not in self.historical_data:
            self.historical_data[symbol] = {
                "values": [],
                "timestamps": [],
                "sources": []
            }
            
        history = self.historical_data[symbol]
        
        # Check for anomalies using z-score if we have enough history
        adjusted_confidence = confidence
        if len(history["values"]) >= 5:
            # Get recent values for this source
            source_indices = [i for i, s in enumerate(history["sources"]) if s == source]
            if source_indices:
                source_values = [history["values"][i] for i in source_indices[-10:]]
                if source_values:
                    mean = np.mean(source_values)
                    std = np.std(source_values) if len(source_values) > 1 else 0.1
                    z_score = abs(sentiment_value - mean) / std
                    
                    if z_score > self.anomaly_threshold:
                        # Reduce confidence based on how extreme the anomaly is
                        confidence_reduction = min(0.5, (z_score - self.anomaly_threshold) / 10)
                        adjusted_confidence = max(0.1, confidence - confidence_reduction)
                        
                        if z_score > self.anomaly_threshold * 2:
                            return False, f"Extreme anomaly detected (z-score: {z_score:.2f})", adjusted_confidence
        
        # Update historical data
        history["values"].append(sentiment_value)
        history["timestamps"].append(timestamp)
        history["sources"].append(source)
        
        # Trim history if needed
        if len(history["values"]) > self.history_size:
            history["values"] = history["values"][-self.history_size:]
            history["timestamps"] = history["timestamps"][-self.history_size:]
            history["sources"] = history["sources"][-self.history_size:]
            
        return True, "Validation passed", adjusted_confidence
        
    def get_source_credibility(self, source: str) -> float:
        """Get the credibility score for a source.
        
        Args:
            source: The sentiment source
            
        Returns:
            The credibility score (0-1)
        """
        return self.source_credibility.get(source, 0.6)
        
    def update_source_credibility(self, source: str, score: float) -> None:
        """Update the credibility score for a source.
        
        Args:
            source: The sentiment source
            score: The new credibility score (0-1)
        """
        self.source_credibility[source] = max(0.1, min(1.0, score))
