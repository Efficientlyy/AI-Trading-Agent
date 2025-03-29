"""Advanced sentiment validation module.

This module provides enhanced functionality for validating sentiment data,
including content filtering, anomaly detection, and source credibility scoring.
"""

import re
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set

class ContentFilter:
    """Filters content for spam, manipulation, and low-quality signals."""
    
    def __init__(self):
        """Initialize the content filter."""
        # Spam patterns (simplified for demonstration)
        self.spam_patterns = [
            r'(?i)(to\s+the\s+moon|pump\s+it|dump\s+it|buy\s+now|sell\s+now|get\s+rich)',
            r'(?i)(guaranteed|100%\s+profit|double\s+your|millionaire)',
            r'(?i)(join\s+my|follow\s+me|sign\s+up|register\s+now|click\s+here)',
            r'(?i)(airdrop|free\s+tokens|free\s+coins|giveaway)',
            r'(?i)(scam|fraud|fake|ponzi|pyramid)'
        ]
        
        # Manipulation indicators
        self.manipulation_indicators = [
            r'(?i)(short\s+squeeze|pump\s+and\s+dump|manipulate|manipulated|manipulation)',
            r'(?i)(coordinated|attack|raid|group\s+buy|mass\s+sell)',
            r'(?i)(bot|bots|automated|algorithm|trading\s+group)'
        ]
        
        # Extreme sentiment indicators
        self.extreme_indicators = [
            r'(?i)(best\s+investment\s+ever|worst\s+investment\s+ever)',
            r'(?i)(going\s+to\s+zero|worthless|useless|dead\s+coin)',
            r'(?i)(1000x|10000x|million|billion|trillion)',
            r'(?i)(all\s+time\s+high|all\s+time\s+low|ath|atl)'
        ]
        
        # Compiled patterns for performance
        self.compiled_spam = [re.compile(pattern) for pattern in self.spam_patterns]
        self.compiled_manipulation = [re.compile(pattern) for pattern in self.manipulation_indicators]
        self.compiled_extreme = [re.compile(pattern) for pattern in self.extreme_indicators]
        
        # Banned phrases (exact matches)
        self.banned_phrases = set([
            "financial advice",
            "guaranteed profit",
            "insider information",
            "secret group"
        ])
        
    def filter_content(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Filter content for problematic patterns.
        
        Args:
            text: The text content to filter
            
        Returns:
            Tuple of (is_acceptable, details)
        """
        if not text:
            return True, {"reason": "Empty content"}
            
        # Check for banned phrases
        lower_text = text.lower()
        for phrase in self.banned_phrases:
            if phrase in lower_text:
                return False, {"reason": f"Contains banned phrase: {phrase}"}
        
        # Check for spam patterns
        spam_matches = []
        for i, pattern in enumerate(self.compiled_spam):
            matches = pattern.findall(text)
            if matches:
                spam_matches.extend(matches)
                if len(matches) > 2:  # Multiple spam indicators
                    return False, {
                        "reason": "Multiple spam patterns detected",
                        "matches": spam_matches[:5]  # Limit to first 5
                    }
        
        # Check for manipulation indicators
        manip_matches = []
        for i, pattern in enumerate(self.compiled_manipulation):
            matches = pattern.findall(text)
            if matches:
                manip_matches.extend(matches)
        
        # Check for extreme sentiment indicators
        extreme_matches = []
        for i, pattern in enumerate(self.compiled_extreme):
            matches = pattern.findall(text)
            if matches:
                extreme_matches.extend(matches)
        
        # Calculate problematic score
        spam_score = min(1.0, len(spam_matches) * 0.2)
        manip_score = min(1.0, len(manip_matches) * 0.3)
        extreme_score = min(1.0, len(extreme_matches) * 0.25)
        
        total_score = spam_score + manip_score + extreme_score
        
        # Determine if content is acceptable
        is_acceptable = total_score < 0.7
        
        return is_acceptable, {
            "spam_score": spam_score,
            "manipulation_score": manip_score,
            "extreme_score": extreme_score,
            "total_score": total_score,
            "spam_matches": spam_matches[:3],
            "manipulation_matches": manip_matches[:3],
            "extreme_matches": extreme_matches[:3],
            "reason": "Content quality score too low" if not is_acceptable else "Acceptable content"
        }
        
    def adjust_confidence(self, confidence: float, filter_details: Dict[str, Any]) -> float:
        """Adjust confidence based on filter results.
        
        Args:
            confidence: Original confidence value
            filter_details: Details from filter_content
            
        Returns:
            Adjusted confidence value
        """
        if "total_score" not in filter_details:
            return confidence
            
        # Reduce confidence based on problematic content
        total_score = filter_details["total_score"]
        
        # Linear reduction based on total score
        # At score 0.5, no reduction; at score 1.0, reduce by 50%
        if total_score > 0.5:
            reduction_factor = (total_score - 0.5) * 1.0  # Scale to 0-0.5
            return max(0.1, confidence * (1.0 - reduction_factor))
            
        return confidence


class SourceCredibilityTracker:
    """Tracks and updates source credibility based on historical performance."""
    
    def __init__(self):
        """Initialize the source credibility tracker."""
        # Default credibility scores
        self.default_scores = {
            "social_media": 0.7,
            "news": 0.8,
            "market_sentiment": 0.9,
            "onchain": 0.85
        }
        
        # Current credibility scores
        self.credibility_scores = self.default_scores.copy()
        
        # Historical performance data
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {
            "social_media": [],
            "news": [],
            "market_sentiment": [],
            "onchain": []
        }
        
        # Source metadata
        self.source_metadata: Dict[str, Dict[str, Any]] = {
            "social_media": {
                "volatility": 0.3,  # Higher volatility
                "lag": 0.1,  # Quick to react
                "noise": 0.4,  # Higher noise
                "last_update": datetime.utcnow()
            },
            "news": {
                "volatility": 0.2,
                "lag": 0.2,
                "noise": 0.3,
                "last_update": datetime.utcnow()
            },
            "market_sentiment": {
                "volatility": 0.1,
                "lag": 0.3,
                "noise": 0.2,
                "last_update": datetime.utcnow()
            },
            "onchain": {
                "volatility": 0.1,
                "lag": 0.4,  # Slower to react
                "noise": 0.1,  # Lower noise
                "last_update": datetime.utcnow()
            }
        }
        
    def get_credibility(self, source: str) -> float:
        """Get the credibility score for a source.
        
        Args:
            source: The sentiment source
            
        Returns:
            The credibility score (0-1)
        """
        return self.credibility_scores.get(source, 0.6)
        
    def record_performance(self, 
                          source: str, 
                          prediction_value: float,
                          actual_value: float,
                          prediction_time: datetime,
                          actual_time: datetime) -> None:
        """Record performance data for a source.
        
        Args:
            source: The sentiment source
            prediction_value: The predicted sentiment value (0-1)
            actual_value: The actual market movement normalized to 0-1
            prediction_time: When the prediction was made
            actual_time: When the actual outcome was measured
        """
        if source not in self.performance_history:
            self.performance_history[source] = []
            
        # Calculate accuracy (inverted mean squared error)
        error = (prediction_value - actual_value) ** 2
        accuracy = max(0, 1 - error)
        
        # Calculate lag (time between prediction and outcome)
        lag_hours = (actual_time - prediction_time).total_seconds() / 3600
        
        # Record performance
        self.performance_history[source].append({
            "prediction": prediction_value,
            "actual": actual_value,
            "accuracy": accuracy,
            "prediction_time": prediction_time,
            "actual_time": actual_time,
            "lag_hours": lag_hours
        })
        
        # Limit history size
        max_history = 100
        if len(self.performance_history[source]) > max_history:
            self.performance_history[source] = self.performance_history[source][-max_history:]
            
        # Update source metadata
        if source in self.source_metadata:
            # Update last update time
            self.source_metadata[source]["last_update"] = datetime.utcnow()
            
            # Update volatility (standard deviation of recent predictions)
            if len(self.performance_history[source]) >= 5:
                recent_predictions = [p["prediction"] for p in self.performance_history[source][-5:]]
                self.source_metadata[source]["volatility"] = float(np.std(recent_predictions))
                
            # Update noise (average error)
            if len(self.performance_history[source]) >= 5:
                recent_errors = [abs(p["prediction"] - p["actual"]) for p in self.performance_history[source][-5:]]
                self.source_metadata[source]["noise"] = float(np.mean(recent_errors))
                
            # Update lag (average lag hours)
            if len(self.performance_history[source]) >= 5:
                recent_lags = [p["lag_hours"] for p in self.performance_history[source][-5:]]
                self.source_metadata[source]["lag"] = float(np.mean(recent_lags)) / 24.0  # Normalize to 0-1 range
        
    def update_credibility_scores(self) -> Dict[str, float]:
        """Update credibility scores based on performance history.
        
        Returns:
            Updated credibility scores
        """
        for source in self.credibility_scores.keys():
            if source not in self.performance_history or not self.performance_history[source]:
                continue
                
            # Get recent performance data
            recent_data = self.performance_history[source][-20:]
            
            if len(recent_data) < 5:
                continue
                
            # Calculate average accuracy
            avg_accuracy = np.mean([p["accuracy"] for p in recent_data])
            
            # Get metadata
            metadata = self.source_metadata.get(source, {
                "volatility": 0.2,
                "lag": 0.2,
                "noise": 0.3
            })
            
            # Calculate base credibility from accuracy
            base_credibility = 0.5 + (avg_accuracy * 0.5)  # Scale to 0.5-1.0
            
            # Adjust for metadata factors
            volatility_factor = 1.0 - (metadata["volatility"] * 0.5)  # Higher volatility reduces credibility
            lag_factor = 1.0 - (metadata["lag"] * 0.3)  # Higher lag reduces credibility
            noise_factor = 1.0 - (metadata["noise"] * 0.5)  # Higher noise reduces credibility
            
            # Calculate final credibility
            credibility = base_credibility * volatility_factor * lag_factor * noise_factor
            
            # Ensure credibility is in valid range
            credibility = max(0.1, min(1.0, credibility))
            
            # Update credibility score
            self.credibility_scores[source] = credibility
            
        return self.credibility_scores
        
    def get_source_metadata(self, source: str) -> Dict[str, Any]:
        """Get metadata for a source.
        
        Args:
            source: The sentiment source
            
        Returns:
            Source metadata
        """
        return self.source_metadata.get(source, {
            "volatility": 0.2,
            "lag": 0.2,
            "noise": 0.3,
            "last_update": datetime.utcnow()
        })


class EnhancedSentimentValidator:
    """Enhanced validator for sentiment data with advanced filtering and credibility tracking."""
    
    def __init__(self):
        """Initialize the enhanced sentiment validator."""
        # Create content filter
        self.content_filter = ContentFilter()
        
        # Create source credibility tracker
        self.credibility_tracker = SourceCredibilityTracker()
        
        # Anomaly detection parameters
        self.anomaly_threshold = 2.5
        self.extreme_anomaly_threshold = 4.0
        
        # Historical data for anomaly detection
        self.historical_data: Dict[str, Dict[str, Any]] = {}
        
        # Validation statistics
        self.validation_stats = {
            "total_processed": 0,
            "rejected_count": 0,
            "anomaly_count": 0,
            "adjusted_count": 0,
            "last_update": datetime.utcnow()
        }
        
    def validate_sentiment(self,
                          symbol: str,
                          source: str,
                          sentiment_value: float,
                          confidence: float,
                          timestamp: datetime,
                          content: Optional[str] = None) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Validate sentiment data with enhanced filtering.
        
        Args:
            symbol: The trading pair symbol
            source: The sentiment source
            sentiment_value: The sentiment value (0-1)
            confidence: The confidence value (0-1)
            timestamp: The timestamp of the sentiment data
            content: Optional text content that generated the sentiment
            
        Returns:
            Tuple of (is_valid, reason, adjusted_confidence, details)
        """
        # Update statistics
        self.validation_stats["total_processed"] += 1
        self.validation_stats["last_update"] = datetime.utcnow()
        
        # Initialize result details
        details = {
            "original_confidence": confidence,
            "adjustments": []
        }
        
        # Step 1: Check source credibility
        source_credibility = self.credibility_tracker.get_credibility(source)
        if source_credibility < 0.5:
            self.validation_stats["rejected_count"] += 1
            return False, f"Source credibility too low: {source_credibility:.2f}", confidence, details
            
        # Apply initial credibility adjustment
        adjusted_confidence = confidence * source_credibility
        details["adjustments"].append({
            "type": "credibility",
            "factor": source_credibility,
            "result": adjusted_confidence
        })
        
        # Step 2: Content filtering (if content provided)
        if content:
            is_acceptable, filter_details = self.content_filter.filter_content(content)
            details["content_filter"] = filter_details
            
            if not is_acceptable:
                self.validation_stats["rejected_count"] += 1
                return False, f"Content filtering failed: {filter_details['reason']}", adjusted_confidence, details
                
            # Adjust confidence based on content quality
            prev_confidence = adjusted_confidence
            adjusted_confidence = self.content_filter.adjust_confidence(adjusted_confidence, filter_details)
            
            if adjusted_confidence < prev_confidence:
                self.validation_stats["adjusted_count"] += 1
                details["adjustments"].append({
                    "type": "content_quality",
                    "factor": adjusted_confidence / prev_confidence if prev_confidence > 0 else 1.0,
                    "result": adjusted_confidence
                })
        
        # Step 3: Anomaly detection
        # Initialize historical data for this symbol if needed
        if symbol not in self.historical_data:
            self.historical_data[symbol] = {
                "values": [],
                "timestamps": [],
                "sources": [],
                "mean": 0.5,
                "std": 0.1
            }
            
        history = self.historical_data[symbol]
        
        # Check for anomalies if we have enough history
        if len(history["values"]) >= 5:
            # Get recent values for this source
            source_indices = [i for i, s in enumerate(history["sources"]) if s == source]
            if source_indices:
                source_values = [history["values"][i] for i in source_indices[-10:]]
                if source_values:
                    mean = np.mean(source_values)
                    std = np.std(source_values) if len(source_values) > 1 else 0.1
                    z_score = abs(sentiment_value - mean) / std
                    
                    details["anomaly_detection"] = {
                        "mean": float(mean),
                        "std": float(std),
                        "z_score": float(z_score),
                        "threshold": self.anomaly_threshold
                    }
                    
                    if z_score > self.extreme_anomaly_threshold:
                        # Extreme anomaly - reject
                        self.validation_stats["rejected_count"] += 1
                        self.validation_stats["anomaly_count"] += 1
                        return False, f"Extreme anomaly detected (z-score: {z_score:.2f})", adjusted_confidence, details
                    elif z_score > self.anomaly_threshold:
                        # Anomaly - reduce confidence
                        self.validation_stats["anomaly_count"] += 1
                        
                        # Calculate confidence reduction factor
                        reduction_factor = min(0.7, (z_score - self.anomaly_threshold) / 5)
                        prev_confidence = adjusted_confidence
                        adjusted_confidence = max(0.1, adjusted_confidence * (1.0 - reduction_factor))
                        
                        details["adjustments"].append({
                            "type": "anomaly",
                            "factor": adjusted_confidence / prev_confidence if prev_confidence > 0 else 1.0,
                            "z_score": float(z_score),
                            "result": adjusted_confidence
                        })
        
        # Update historical data
        history["values"].append(sentiment_value)
        history["timestamps"].append(timestamp)
        history["sources"].append(source)
        
        # Trim history if needed
        max_history = 100
        if len(history["values"]) > max_history:
            history["values"] = history["values"][-max_history:]
            history["timestamps"] = history["timestamps"][-max_history:]
            history["sources"] = history["sources"][-max_history:]
            
        # Update mean and std
        history["mean"] = float(np.mean(history["values"]))
        history["std"] = float(np.std(history["values"])) if len(history["values"]) > 1 else 0.1
        
        # Final validation result
        return True, "Validation passed", adjusted_confidence, details
        
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics.
        
        Returns:
            Validation statistics
        """
        return self.validation_stats
        
    def get_source_credibility(self, source: str) -> float:
        """Get the credibility score for a source.
        
        Args:
            source: The sentiment source
            
        Returns:
            The credibility score (0-1)
        """
        return self.credibility_tracker.get_credibility(source)
        
    def record_performance(self,
                          source: str,
                          prediction_value: float,
                          actual_value: float,
                          prediction_time: datetime,
                          actual_time: datetime) -> None:
        """Record performance data for a source.
        
        Args:
            source: The sentiment source
            prediction_value: The predicted sentiment value (0-1)
            actual_value: The actual market movement normalized to 0-1
            prediction_time: When the prediction was made
            actual_time: When the actual outcome was measured
        """
        self.credibility_tracker.record_performance(
            source=source,
            prediction_value=prediction_value,
            actual_value=actual_value,
            prediction_time=prediction_time,
            actual_time=actual_time
        )
        
    def update_credibility_scores(self) -> Dict[str, float]:
        """Update credibility scores based on performance history.
        
        Returns:
            Updated credibility scores
        """
        return self.credibility_tracker.update_credibility_scores()
