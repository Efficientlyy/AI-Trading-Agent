"""Prediction aggregator for the Decision Engine.

This module provides components for aggregating predictions from multiple 
analysis agents into a single consolidated prediction.
"""

import asyncio
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from src.common.logging import get_logger
from src.common.config import config
from src.common.datetime_utils import utc_now
from src.decision_engine.models import (
    Prediction, AggregatedPrediction, Direction, 
    SignalType, PredictionSource
)


class PredictionAggregator:
    """Aggregates predictions from multiple sources."""
    
    def __init__(self):
        """Initialize the prediction aggregator."""
        self.logger = get_logger("decision_engine", "prediction_aggregator")
        
        # Load config
        self.min_confidence = config.get("decision_engine.aggregator.min_confidence", 0.7)
        self.min_agent_agreement = config.get("decision_engine.aggregator.min_agent_agreement", 2)
        self.prediction_ttl = config.get("decision_engine.aggregator.prediction_ttl", 3600)  # seconds
        self.time_decay_factor = config.get("decision_engine.aggregator.time_decay_factor", 0.9)
        
        # Default agent weights
        self.default_weights = {
            PredictionSource.TECHNICAL.value: 0.4,
            PredictionSource.PATTERN.value: 0.4,
            PredictionSource.SENTIMENT.value: 0.2,
            PredictionSource.FUNDAMENTAL.value: 0.3,
            PredictionSource.ENSEMBLE.value: 0.5,
        }
        
        # Agent-specific weights (can be updated based on performance)
        self.agent_weights = config.get(
            "decision_engine.aggregator.agent_weights", 
            {}
        )
        
        # Active predictions by symbol
        self.active_predictions: Dict[str, List[Prediction]] = {}
        
        # Previously aggregated predictions
        self.aggregated_predictions: Dict[str, AggregatedPrediction] = {}
        
    def add_prediction(self, prediction: Prediction) -> None:
        """Add a new prediction to the aggregator.
        
        Args:
            prediction: The prediction to add
        """
        # Skip predictions with low confidence
        if prediction.confidence < self.min_confidence:
            self.logger.debug("Skipping low-confidence prediction", 
                           confidence=prediction.confidence,
                           min_threshold=self.min_confidence)
            return
        
        # Initialize predictions list for this symbol if needed
        if prediction.symbol not in self.active_predictions:
            self.active_predictions[prediction.symbol] = []
        
        # Check if this prediction already exists
        for i, existing in enumerate(self.active_predictions[prediction.symbol]):
            if existing.id == prediction.id:
                # Update the existing prediction
                self.active_predictions[prediction.symbol][i] = prediction
                self.logger.debug("Updated existing prediction", 
                               prediction_id=prediction.id,
                               symbol=prediction.symbol)
                return
        
        # Add new prediction
        self.active_predictions[prediction.symbol].append(prediction)
        self.logger.info("Added new prediction", 
                       prediction_id=prediction.id,
                       source=prediction.source.value,
                       agent=prediction.agent_id,
                       symbol=prediction.symbol,
                       direction=prediction.direction.value,
                       confidence=prediction.confidence)
    
    def get_active_predictions(self, symbol: str) -> List[Prediction]:
        """Get all active predictions for a symbol.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            List of active predictions
        """
        if symbol not in self.active_predictions:
            return []
            
        # Filter valid predictions and remove expired ones
        valid_predictions = []
        now = utc_now()
        
        for prediction in self.active_predictions[symbol]:
            # Check if prediction is expired based on TTL
            is_expired = False
            if prediction.expiration:
                is_expired = now > prediction.expiration
            else:
                # Use default TTL if no expiration is specified
                is_expired = (now - prediction.timestamp).total_seconds() > self.prediction_ttl
                
            if not is_expired and prediction.is_valid():
                valid_predictions.append(prediction)
        
        # Update the active predictions list
        self.active_predictions[symbol] = valid_predictions
        
        return valid_predictions
    
    def clear_expired_predictions(self) -> None:
        """Remove expired predictions from all symbols."""
        now = utc_now()
        count = 0
        
        for symbol in list(self.active_predictions.keys()):
            valid_predictions = []
            
            for prediction in self.active_predictions[symbol]:
                # Check if prediction is expired
                is_expired = False
                if prediction.expiration:
                    is_expired = now > prediction.expiration
                else:
                    # Use default TTL if no expiration is specified
                    is_expired = (now - prediction.timestamp).total_seconds() > self.prediction_ttl
                    
                if not is_expired:
                    valid_predictions.append(prediction)
                else:
                    count += 1
            
            if valid_predictions:
                self.active_predictions[symbol] = valid_predictions
            else:
                del self.active_predictions[symbol]
        
        if count > 0:
            self.logger.debug("Cleared expired predictions", count=count)
    
    def aggregate_predictions(self, symbol: str) -> Optional[AggregatedPrediction]:
        """Aggregate predictions for a symbol into a single prediction.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            An aggregated prediction, or None if not enough predictions
        """
        # Get active predictions for this symbol
        predictions = self.get_active_predictions(symbol)
        
        # If not enough predictions, return None
        if len(predictions) < self.min_agent_agreement:
            self.logger.debug("Not enough predictions to aggregate", 
                           count=len(predictions),
                           min_required=self.min_agent_agreement)
            return None
        
        # Group predictions by timeframe and signal type
        grouped_predictions = self._group_predictions_by_timeframe_and_signal(predictions)
        
        # If no groups have enough predictions, return None
        if not grouped_predictions:
            self.logger.debug("No prediction groups with enough agreement")
            return None
        
        # Find the group with the highest confidence
        best_group = max(grouped_predictions, key=lambda g: g["aggregate_confidence"])
        
        # Create the aggregated prediction
        aggregated = self._create_aggregated_prediction(symbol, best_group)
        
        # Store the aggregated prediction
        self.aggregated_predictions[aggregated.id] = aggregated
        
        self.logger.info("Created aggregated prediction", 
                       prediction_id=aggregated.id,
                       symbol=symbol,
                       direction=aggregated.direction.value,
                       confidence=aggregated.confidence,
                       signal_type=aggregated.signal_type.value,
                       timeframe=aggregated.timeframe,
                       prediction_count=len(best_group["predictions"]))
        
        return aggregated
    
    def _group_predictions_by_timeframe_and_signal(self, predictions: List[Prediction]) -> List[Dict]:
        """Group predictions by timeframe and signal type.
        
        Args:
            predictions: List of predictions to group
            
        Returns:
            List of prediction groups
        """
        # Group predictions by timeframe and signal type
        groups = {}
        
        for prediction in predictions:
            key = (prediction.timeframe, prediction.signal_type.value)
            if key not in groups:
                groups[key] = []
            groups[key].append(prediction)
        
        # Filter groups with enough predictions
        filtered_groups = []
        
        for (timeframe, signal_type), group_predictions in groups.items():
            if len(group_predictions) >= self.min_agent_agreement:
                # Calculate aggregate confidence for this group
                aggregate_confidence = self._calculate_aggregate_confidence(group_predictions)
                
                filtered_groups.append({
                    "timeframe": timeframe,
                    "signal_type": signal_type,
                    "predictions": group_predictions,
                    "aggregate_confidence": aggregate_confidence
                })
        
        return filtered_groups
    
    def _calculate_aggregate_confidence(self, predictions: List[Prediction]) -> float:
        """Calculate aggregated confidence from a list of predictions.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Aggregated confidence score
        """
        if not predictions:
            return 0.0
        
        # Calculate weighted confidence for each prediction
        weighted_confidences = []
        total_weight = 0.0
        
        for prediction in predictions:
            # Get weight based on source and agent
            source_weight = self.default_weights.get(prediction.source.value, 0.5)
            agent_weight = self.agent_weights.get(prediction.agent_id, source_weight)
            
            # Apply time decay to older predictions
            age_seconds = (utc_now() - prediction.timestamp).total_seconds()
            time_decay = max(0.5, self.time_decay_factor ** (age_seconds / 3600.0))
            
            # Direction agreement
            bullish_count = sum(1 for p in predictions if p.direction == Direction.BULLISH)
            bearish_count = sum(1 for p in predictions if p.direction == Direction.BEARISH)
            
            # Direction consensus factor (higher if more agreement)
            if prediction.direction == Direction.BULLISH:
                direction_factor = bullish_count / len(predictions)
            elif prediction.direction == Direction.BEARISH:
                direction_factor = bearish_count / len(predictions)
            else:
                direction_factor = 0.5
            
            # Combined weight
            weight = agent_weight * time_decay * direction_factor
            weighted_confidences.append(prediction.confidence * weight)
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            return sum(weighted_confidences) / total_weight
        else:
            return 0.0
    
    def _get_direction_consensus(self, predictions: List[Prediction]) -> Direction:
        """Get consensus direction from predictions.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Consensus direction
        """
        bullish_count = sum(1 for p in predictions if p.direction == Direction.BULLISH)
        bearish_count = sum(1 for p in predictions if p.direction == Direction.BEARISH)
        neutral_count = sum(1 for p in predictions if p.direction == Direction.NEUTRAL)
        
        if bullish_count > bearish_count and bullish_count > neutral_count:
            return Direction.BULLISH
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            return Direction.BEARISH
        else:
            return Direction.NEUTRAL
    
    def _calculate_price_targets(self, predictions: List[Prediction], 
                               direction: Direction) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate consensus price targets from predictions.
        
        Args:
            predictions: List of predictions
            direction: Consensus direction
            
        Returns:
            Tuple containing (entry_price, stop_loss, take_profit)
        """
        # Filter predictions that match the consensus direction
        matching_predictions = [p for p in predictions if p.direction == direction]
        
        # If no matching predictions, return None for all targets
        if not matching_predictions:
            return None, None, None
        
        # Filter predictions with price targets
        entry_prices = [p.entry_price for p in matching_predictions if p.entry_price is not None]
        stop_losses = [p.stop_loss for p in matching_predictions if p.stop_loss is not None]
        take_profits = [p.take_profit for p in matching_predictions if p.take_profit is not None]
        
        # Calculate weighted average for each target
        entry_price = float(np.median(entry_prices)) if entry_prices else None
        stop_loss = float(np.median(stop_losses)) if stop_losses else None
        take_profit = float(np.median(take_profits)) if take_profits else None
        
        return entry_price, stop_loss, take_profit
    
    def _create_aggregated_prediction(self, symbol: str, group: Dict) -> AggregatedPrediction:
        """Create an aggregated prediction from a group of predictions.
        
        Args:
            symbol: The trading pair symbol
            group: Prediction group information
            
        Returns:
            Aggregated prediction
        """
        predictions = group["predictions"]
        
        # Get consensus direction and signal type
        direction = self._get_direction_consensus(predictions)
        signal_type = SignalType(group["signal_type"])
        
        # Calculate price targets
        entry_price, stop_loss, take_profit = self._calculate_price_targets(predictions, direction)
        
        # Create weights dictionary
        weights = {}
        for prediction in predictions:
            source_weight = self.default_weights.get(prediction.source.value, 0.5)
            agent_weight = self.agent_weights.get(prediction.agent_id, source_weight)
            weights[prediction.id] = agent_weight
        
        # Build aggregated rationale
        rationale_parts = []
        for prediction in sorted(predictions, key=lambda p: p.confidence, reverse=True)[:3]:
            rationale_parts.append(f"{prediction.source.value.capitalize()} ({prediction.agent_id}): {prediction.rationale}")
        
        rationale = "\n".join(rationale_parts)
        
        # Create the aggregated prediction
        return AggregatedPrediction(
            id=f"agg_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            timestamp=utc_now(),
            timeframe=group["timeframe"],
            direction=direction,
            confidence=group["aggregate_confidence"],
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            predictions=predictions,
            weights=weights,
            rationale=rationale,
            tags=[f"aggregated", f"source_count:{len(predictions)}"]
        )
    
    def get_aggregated_prediction(self, prediction_id: str) -> Optional[AggregatedPrediction]:
        """Get an aggregated prediction by ID.
        
        Args:
            prediction_id: The prediction ID
            
        Returns:
            The aggregated prediction, or None if not found
        """
        return self.aggregated_predictions.get(prediction_id)
