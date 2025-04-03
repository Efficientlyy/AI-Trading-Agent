"""Performance tracking for sentiment analysis system.

This module tracks the performance of various sentiment sources and models
to improve overall system reliability and accuracy.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, DefaultDict, Deque

import numpy as np
import pandas as pd

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event


class PerformanceTracker:
    """Tracks performance of sentiment sources and models.
    
    This class maintains a history of predictions and their outcomes,
    calculating metrics such as accuracy, precision, recall, and
    calibrating confidence scores based on historical performance.
    """
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.logger = get_logger("analysis_agents", "performance_tracker")
        
        # Configuration
        self.tracking_window = config.get("sentiment_analysis.consensus_system.tracking_window", 720)  # 12 hours default
        self.min_predictions = config.get("sentiment_analysis.consensus_system.min_predictions", 5)
        self.save_interval = config.get("sentiment_analysis.consensus_system.save_interval", 3600)  # 1 hour default
        
        # Initialize data structures
        self.predictions: DefaultDict[str, DefaultDict[str, Deque[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.tracking_window))
        )
        
        # Performance metrics for each source/model
        self.performance_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        # Last save timestamp
        self.last_save_time = 0
        
        # Create directory for performance data
        os.makedirs("data/performance", exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the performance tracker."""
        self.logger.info("Initializing performance tracker")
        
        # Load historical performance data if available
        self._load_performance_data()
        
        # Subscribe to events
        event_bus.subscribe("sentiment_prediction", self.handle_prediction)
        event_bus.subscribe("market_direction_update", self.handle_market_update)
    
    async def _load_performance_data(self) -> None:
        """Load historical performance data from files."""
        performance_file = "data/performance/sentiment_performance.json"
        
        if os.path.exists(performance_file):
            try:
                with open(performance_file, 'r') as f:
                    data = json.load(f)
                
                # Restore performance metrics
                self.performance_metrics = data.get("metrics", {})
                
                # Restore recent predictions (up to tracking window)
                predictions_data = data.get("predictions", {})
                for symbol, sources in predictions_data.items():
                    for source, preds in sources.items():
                        # Convert list to deque with maxlen
                        self.predictions[symbol][source] = deque(preds, maxlen=self.tracking_window)
                
                self.logger.info(f"Loaded performance data for {len(self.performance_metrics)} sources")
                
            except Exception as e:
                self.logger.error(f"Error loading performance data: {str(e)}")
    
    async def save_performance_data(self) -> None:
        """Save performance data to file."""
        current_time = time.time()
        
        # Only save if enough time has passed since last save
        if current_time - self.last_save_time < self.save_interval:
            return
        
        self.last_save_time = current_time
        performance_file = "data/performance/sentiment_performance.json"
        
        try:
            # Convert predictions deques to lists for serialization
            predictions_data = {}
            for symbol, sources in self.predictions.items():
                predictions_data[symbol] = {}
                for source, preds in sources.items():
                    predictions_data[symbol][source] = list(preds)
            
            # Prepare data for saving
            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": self.performance_metrics,
                "predictions": predictions_data
            }
            
            # Save to file
            with open(performance_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved performance data to {performance_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance data: {str(e)}")
    
    async def handle_prediction(self, event: Event) -> None:
        """Handle a new sentiment prediction event.
        
        Args:
            event: The sentiment prediction event
        """
        # Extract relevant data from event
        data = event.data
        symbol = data.get("symbol")
        source = data.get("source")
        sentiment_value = data.get("sentiment_value")
        direction = data.get("direction")
        confidence = data.get("confidence")
        timestamp = data.get("timestamp", datetime.utcnow().isoformat())
        
        if not all([symbol, source, sentiment_value is not None, direction, confidence is not None]):
            self.logger.warning(f"Missing data in prediction event: {data}")
            return
        
        # Store prediction for later verification
        prediction = {
            "timestamp": timestamp,
            "sentiment_value": sentiment_value,
            "direction": direction,
            "confidence": confidence,
            "verified": False,
            "correct_direction": None,
            "correct_value": None,
            "verification_time": None
        }
        
        self.predictions[symbol][source].append(prediction)
        
        # Schedule periodic save
        asyncio.create_task(self.save_performance_data())
    
    async def handle_market_update(self, event: Event) -> None:
        """Handle a market direction update event.
        
        This event indicates the actual market direction and is used to
        evaluate prediction accuracy.
        
        Args:
            event: The market update event
        """
        # Extract relevant data
        data = event.data
        symbol = data.get("symbol")
        actual_direction = data.get("direction")
        actual_value = data.get("value")
        update_time = data.get("timestamp", datetime.utcnow().isoformat())
        lookback_period = data.get("lookback_period", 3600)  # Default 1 hour
        
        if not all([symbol, actual_direction, actual_value is not None]):
            self.logger.warning(f"Missing data in market update event: {data}")
            return
        
        # Calculate the cutoff time for prediction evaluation
        if isinstance(update_time, str):
            update_time = datetime.fromisoformat(update_time)
        
        cutoff_time = update_time - timedelta(seconds=lookback_period)
        cutoff_time_str = cutoff_time.isoformat()
        
        # Update relevant predictions
        sources_updated = []
        
        for source, predictions in self.predictions[symbol].items():
            predictions_updated = False
            
            for pred in predictions:
                # Skip already verified predictions
                if pred["verified"]:
                    continue
                
                # Skip predictions made after the cutoff time
                pred_time = pred["timestamp"]
                if isinstance(pred_time, str):
                    pred_time = datetime.fromisoformat(pred_time)
                else:
                    pred_time = datetime.fromtimestamp(pred_time)
                
                if pred_time > cutoff_time:
                    continue
                
                # Verify the prediction
                pred["verified"] = True
                pred["correct_direction"] = pred["direction"] = = actual_direction                pred["correct_direction"] = pred["direction"] = = actual_direction
                pred["correct_value"] = abs(pred["sentiment_value"] - actual_value) < 0.25  # Within 0.25 range
                pred["verification_time"] = update_time.isoformat()
                
                predictions_updated = True
            
            if predictions_updated:
                sources_updated.append(source)
                
                # Update performance metrics for this source
                self._update_source_metrics(symbol, source)
        
        if sources_updated:
            self.logger.info(f"Updated predictions for {symbol} from sources: {', '.join(sources_updated)}")
            
            # Save updated performance data
            asyncio.create_task(self.save_performance_data())
    
    def _update_source_metrics(self, symbol: str, source: str) -> None:
        """Update performance metrics for a specific source.
        
        Args:
            symbol: The trading symbol
            source: The sentiment source
        """
        # Get verified predictions for this source
        all_predictions = [p for p in self.predictions[symbol][source] if p["verified"]]
        
        # Skip if not enough predictions
        if len(all_predictions) < self.min_predictions:
            return
        
        # Calculate metrics
        total = len(all_predictions)
        direction_correct = sum(1 for p in all_predictions if p["correct_direction"])
        value_correct = sum(1 for p in all_predictions if p["correct_value"])
        
        # Direction accuracy
        direction_accuracy = direction_correct / total if total > 0 else 0
        
        # Value accuracy (how close predictions were to actual values)
        value_accuracy = value_correct / total if total > 0 else 0
        
        # Calculate calibration metrics (comparing confidence to accuracy)
        confidence_values = np.array([p["confidence"] for p in all_predictions])
        direction_correctness = np.array([1 if p["correct_direction"] else 0 for p in all_predictions])
        
        # Calculate calibration error (difference between confidence and actual accuracy)
        calibration_error = abs(np.mean(confidence_values) - direction_accuracy)
        
        # Calculate recency-weighted metrics (more recent predictions matter more)
        weights = np.linspace(0.5, 1.0, len(all_predictions))
        weighted_direction_accuracy = np.average(
            direction_correctness, 
            weights=weights
        ) if len(all_predictions) > 0 else 0
        
        # Store metrics
        if symbol not in self.performance_metrics:
            self.performance_metrics[symbol] = {}
        
        self.performance_metrics[symbol][source] = {
            "direction_accuracy": direction_accuracy,
            "value_accuracy": value_accuracy,
            "calibration_error": calibration_error,
            "weighted_accuracy": weighted_direction_accuracy,
            "total_predictions": total,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        self.logger.info(
            f"Updated metrics for {symbol}/{source}: "
            f"Direction accuracy: {direction_accuracy:.2f}, "
            f"Value accuracy: {value_accuracy:.2f}"
        )
    
    def get_source_performance(self, symbol: str, source: str) -> Dict[str, Any]:
        """Get performance metrics for a specific source.
        
        Args:
            symbol: The trading symbol
            source: The sentiment source
            
        Returns:
            Performance metrics for the source
        """
        return self.performance_metrics.get(symbol, {}).get(source, {})
    
    def get_all_performance(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get all performance metrics.
        
        Returns:
            All performance metrics
        """
        return self.performance_metrics
    
    def get_calibrated_confidence(self, symbol: str, source: str, raw_confidence: float) -> float:
        """Get calibrated confidence based on historical performance.
        
        Args:
            symbol: The trading symbol
            source: The sentiment source
            raw_confidence: The raw confidence score
            
        Returns:
            Calibrated confidence score
        """
        # Get performance metrics for this source
        metrics = self.get_source_performance(symbol, source)
        
        if not metrics:
            return raw_confidence  # No metrics available, return raw confidence
        
        # Get the source's direction accuracy
        direction_accuracy = metrics.get("direction_accuracy", 0.5)
        
        # Get the calibration error
        calibration_error = metrics.get("calibration_error", 0.0)
        
        # If the source tends to be overconfident, reduce the confidence
        if calibration_error > 0.1:  # Threshold for overconfidence
            calibration_factor = 1.0 - (calibration_error * 0.5)  # Reduce by up to 50% based on calibration error
            calibrated_confidence = raw_confidence * calibration_factor
        else:
            calibrated_confidence = raw_confidence
        
        # Anchor the confidence to historical accuracy
        if direction_accuracy < 0.5:
            # If worse than random, significantly reduce confidence
            calibrated_confidence *= (direction_accuracy / 0.5)
        else:
            # Adjust based on accuracy above random
            accuracy_factor = 0.5 + (direction_accuracy - 0.5) * 0.5  # Scale the above-random portion by 50%
            calibrated_confidence = (calibrated_confidence + accuracy_factor) / 2  # Average with accuracy factor
        
        # Ensure confidence is in valid range
        return max(0.1, min(0.99, calibrated_confidence))
    
    def get_optimal_weights(self, symbol: str) -> Dict[str, float]:
        """Calculate optimal source weights based on performance.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Dictionary of source weights
        """
        if symbol not in self.performance_metrics:
            # Default weights if no metrics available
            return {
                "llm": 0.4,
                "news": 0.2,
                "social_media": 0.2,
                "market": 0.1,
                "onchain": 0.1
            }
        
        sources = self.performance_metrics[symbol]
        
        # If less than 2 sources have metrics, use default weights
        if len(sources) < 2:
            return {
                "llm": 0.4,
                "news": 0.2,
                "social_media": 0.2,
                "market": 0.1,
                "onchain": 0.1
            }
        
        # Calculate weights based on weighted accuracy
        weights = {}
        total_accuracy = 0
        
        for source, metrics in sources.items():
            # Use the weighted accuracy for weight calculation
            accuracy = metrics.get("weighted_accuracy", 0.5)
            
            # Ensure a minimum weight even for poor performers (avoid zero weights)
            adjusted_accuracy = max(0.3, accuracy)
            
            weights[source] = adjusted_accuracy
            total_accuracy += adjusted_accuracy
        
        # Normalize weights to sum to 1.0
        if total_accuracy > 0:
            normalized_weights = {source: weight / total_accuracy for source, weight in weights.items()}
        else:
            # Fallback to equal weights
            normalized_weights = {source: 1.0 / len(sources) for source in sources}
        
        return normalized_weights
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report.
        
        Returns:
            Performance report data
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {},
            "symbols": {}
        }
        
        # Overall metrics across all symbols and sources
        all_direction_accuracies = []
        all_value_accuracies = []
        all_calibration_errors = []
        
        for symbol, sources in self.performance_metrics.items():
            symbol_metrics = {
                "direction_accuracy": {},
                "value_accuracy": {},
                "calibration_error": {},
                "total_predictions": {},
                "optimal_weights": self.get_optimal_weights(symbol)
            }
            
            for source, metrics in sources.items():
                direction_accuracy = metrics.get("direction_accuracy", 0)
                value_accuracy = metrics.get("value_accuracy", 0)
                calibration_error = metrics.get("calibration_error", 0)
                
                all_direction_accuracies.append(direction_accuracy)
                all_value_accuracies.append(value_accuracy)
                all_calibration_errors.append(calibration_error)
                
                symbol_metrics["direction_accuracy"][source] = direction_accuracy
                symbol_metrics["value_accuracy"][source] = value_accuracy
                symbol_metrics["calibration_error"][source] = calibration_error
                symbol_metrics["total_predictions"][source] = metrics.get("total_predictions", 0)
            
            report["symbols"][symbol] = symbol_metrics
        
        # Calculate overall summary
        report["summary"] = {
            "avg_direction_accuracy": np.mean(all_direction_accuracies) if all_direction_accuracies else 0,
            "avg_value_accuracy": np.mean(all_value_accuracies) if all_value_accuracies else 0,
            "avg_calibration_error": np.mean(all_calibration_errors) if all_calibration_errors else 0,
            "total_symbols": len(self.performance_metrics),
            "total_sources": sum(len(sources) for sources in self.performance_metrics.values())
        }
        
        return report


# Singleton instance
performance_tracker = PerformanceTracker()