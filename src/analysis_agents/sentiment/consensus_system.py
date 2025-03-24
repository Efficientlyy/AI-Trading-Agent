"""Consensus System for Multi-Model Sentiment Analysis.

This module implements a sophisticated consensus system for aggregating
sentiment signals from multiple LLMs and traditional sentiment sources.
It handles model disagreements, confidence calibration, and provides
weighted ensemble outputs.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import numpy as np
from collections import defaultdict

from src.common.config import config
from src.common.logging import get_logger
from src.common.monitoring import metrics
from src.common.caching import Cache
from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent


class ConsensusSystem:
    """Multi-model consensus system for combining sentiment signals.
    
    This system implements ensemble methods to combine outputs from multiple
    LLMs and sentiment sources, handling disagreements and calibrating
    confidence scores.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the consensus system.
        
        Args:
            agent_id: The agent id that owns this consensus system
        """
        self.logger = get_logger("analysis_agents", "consensus")
        
        # Load configuration
        self.min_sources = config.get(
            f"analysis_agents.{agent_id}.consensus.min_sources", 
            2  # At least 2 sources for consensus
        )
        self.confidence_threshold = config.get(
            f"analysis_agents.{agent_id}.consensus.confidence_threshold", 
            0.6  # Minimum confidence to include in consensus
        )
        self.max_age_hours = config.get(
            f"analysis_agents.{agent_id}.consensus.max_age_hours", 
            24  # Maximum age of sentiment data to include
        )
        self.use_bayesian = config.get(
            f"analysis_agents.{agent_id}.consensus.use_bayesian", 
            True  # Use Bayesian aggregation when possible
        )
        
        # Model weights configuration
        self.model_weights = config.get(
            f"analysis_agents.{agent_id}.consensus.model_weights",
            {
                "gpt-4o": 1.0,
                "claude-3-opus": 1.0,
                "llama-3-70b": 0.9,
                "gpt-3.5-turbo": 0.8,
                "claude-3-sonnet": 0.8,
                "mistral-7b": 0.7,
                "llama-3-8b": 0.7,
                "finbert": 0.6
            }
        )
        
        # Source type weights configuration
        self.source_type_weights = config.get(
            f"analysis_agents.{agent_id}.consensus.source_type_weights",
            {
                "llm": 1.0,
                "social_media": 0.8,
                "news": 0.9,
                "market_sentiment": 0.7,
                "onchain": 0.6,
                "technical": 0.5
            }
        )
        
        # Historical performance tracking
        self.performance_memory = Cache(ttl=86400 * 30)  # 30 day memory
        
        # Disagreement thresholds
        self.minor_disagreement_threshold = 0.2
        self.major_disagreement_threshold = 0.4
    
    def compute_consensus(
        self,
        sentiment_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute consensus from multiple sentiment sources/models.
        
        Args:
            sentiment_data: List of sentiment data points, each containing:
                - value: sentiment value (0-1)
                - direction: sentiment direction (bullish/bearish/neutral)
                - confidence: confidence score (0-1) 
                - model: model name (optional)
                - source_type: source type (optional)
                - timestamp: timestamp of the data
                - additional fields as needed
                
        Returns:
            Consensus result with aggregated sentiment values
        """
        if not sentiment_data or len(sentiment_data) < self.min_sources:
            self.logger.debug(f"Insufficient sources for consensus: {len(sentiment_data) if sentiment_data else 0}")
            return self._get_default_result()
        
        # Filter data by age and confidence
        now = datetime.utcnow()
        filtered_data = []
        
        for data in sentiment_data:
            # Check timestamp
            timestamp = data.get("timestamp")
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        timestamp = now
                
                age_hours = (now - timestamp).total_seconds() / 3600
                if age_hours > self.max_age_hours:
                    continue
            
            # Check confidence
            confidence = data.get("confidence", 0.0)
            if confidence < self.confidence_threshold:
                continue
                
            # Add to filtered data
            filtered_data.append(data)
        
        # Check if we still have enough data
        if len(filtered_data) < self.min_sources:
            self.logger.debug(f"Insufficient valid sources after filtering: {len(filtered_data)}")
            return self._get_default_result()
        
        # Compute weighted consensus
        return self._weighted_ensemble(filtered_data)
    
    def _weighted_ensemble(self, sentiment_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute weighted ensemble from multiple sentiment sources.
        
        Args:
            sentiment_data: Filtered sentiment data
                
        Returns:
            Consensus result with aggregated sentiment
        """
        # Extract values, weights, and confidences
        values = []
        weights = []
        confidences = []
        directions = []
        source_types = set()
        models = set()
        
        for data in sentiment_data:
            # Get basic sentiment data
            value = data.get("value", 0.5)
            confidence = data.get("confidence", 0.5)
            
            # Get source and model info
            source_type = data.get("source_type", "unknown")
            model = data.get("model", "unknown")
            
            # Track unique sources and models
            source_types.add(source_type)
            models.add(model)
            
            # Calculate base weight from source type and model
            source_weight = self.source_type_weights.get(source_type, 0.5)
            model_weight = self.model_weights.get(model, 0.5)
            
            # Adjust weight using performance history if available
            performance_key = f"{source_type}:{model}"
            performance_score = self.performance_memory.get(performance_key)
            if performance_score is not None:
                performance_modifier = 0.5 + (performance_score / 2)  # Scale to 0.5-1.5
            else:
                performance_modifier = 1.0
            
            # Calculate final weight: source weight * model weight * performance * confidence
            final_weight = source_weight * model_weight * performance_modifier * confidence
            
            # Apply recency weighting if timestamp exists
            if "timestamp" in data:
                timestamp = data.get("timestamp")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        timestamp = datetime.utcnow()
                
                age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
                recency_factor = max(0.5, 1.0 - (age_hours / self.max_age_hours))
                final_weight *= recency_factor
            
            # Add to arrays
            values.append(value)
            weights.append(final_weight)
            confidences.append(confidence)
            directions.append(data.get("direction", "neutral"))
        
        # Convert to numpy arrays for efficient computation
        values_array = np.array(values)
        weights_array = np.array(weights)
        confidences_array = np.array(confidences)
        
        # Check for disagreements
        disagreement_level = self._calculate_disagreement(values_array)
        
        # If strong disagreement and we have enough data, try using Bayesian aggregation
        if self.use_bayesian and disagreement_level >= self.major_disagreement_threshold and len(values) >= 3:
            consensus_value, consensus_confidence = self._bayesian_aggregation(values_array, confidences_array, weights_array)
        else:
            # Otherwise use weighted average
            if np.sum(weights_array) > 0:
                consensus_value = float(np.sum(values_array * weights_array) / np.sum(weights_array))
            else:
                consensus_value = float(np.mean(values_array))
                
            # Calculate consensus confidence based on agreement and individual confidences
            agreement_factor = 1.0 - disagreement_level
            base_confidence = float(np.mean(confidences_array))
            consensus_confidence = float(base_confidence * (0.7 + (0.3 * agreement_factor)))
        
        # Determine consensus direction
        if consensus_value > 0.6:
            consensus_direction = "bullish"
        elif consensus_value < 0.4:
            consensus_direction = "bearish"
        else:
            consensus_direction = "neutral"
        
        # Count direction frequencies
        direction_counts = {
            "bullish": directions.count("bullish"),
            "bearish": directions.count("bearish"),
            "neutral": directions.count("neutral")
        }
        
        # Create consensus result
        result = {
            "value": consensus_value,
            "direction": consensus_direction,
            "confidence": consensus_confidence,
            "source_count": len(sentiment_data),
            "unique_source_types": len(source_types),
            "unique_models": len(models),
            "disagreement_level": disagreement_level,
            "direction_counts": direction_counts,
            "source_types": list(source_types),
            "models": list(models),
            "last_update": datetime.utcnow()
        }
        
        # Add disagreement indication if significant
        if disagreement_level >= self.major_disagreement_threshold:
            result["has_major_disagreement"] = True
            result["consensus_method"] = "bayesian" if self.use_bayesian else "weighted_average"
        elif disagreement_level >= self.minor_disagreement_threshold:
            result["has_minor_disagreement"] = True
            
        return result
    
    def _calculate_disagreement(self, values: np.ndarray) -> float:
        """Calculate disagreement level among sentiment values.
        
        Args:
            values: Array of sentiment values
            
        Returns:
            Disagreement level (0-1)
        """
        if len(values) <= 1:
            return 0.0
            
        # Calculate standard deviation and normalize to 0-1 range
        std_dev = float(np.std(values))
        max_possible_std = 0.5  # Maximum possible std dev for values in 0-1 range
        disagreement = min(1.0, std_dev / max_possible_std)
        
        return disagreement
    
    def _bayesian_aggregation(
        self, 
        values: np.ndarray, 
        confidences: np.ndarray, 
        weights: np.ndarray
    ) -> Tuple[float, float]:
        """Perform Bayesian aggregation of sentiment values.
        
        This implements a simplified Bayesian aggregation that considers
        both confidence and weights of individual models.
        
        Args:
            values: Array of sentiment values
            confidences: Array of confidence scores
            weights: Array of weights
            
        Returns:
            Tuple of (aggregated value, aggregated confidence)
        """
        # Convert sentiment values to log-odds (logits)
        # Add small epsilon to avoid log(0) issues
        epsilon = 1e-6
        values_clipped = np.clip(values, epsilon, 1.0 - epsilon)
        logits = np.log(values_clipped / (1.0 - values_clipped))
        
        # Adjust by confidence (higher confidence = more weight)
        adjusted_logits = logits * confidences
        
        # Calculate precision (inverse variance) based on confidence
        # Higher confidence = lower variance = higher precision
        precisions = confidences * weights
        
        # Calculate weighted sum of logits
        if np.sum(precisions) > 0:
            posterior_logit = np.sum(adjusted_logits * precisions) / np.sum(precisions)
        else:
            posterior_logit = np.mean(logits)  # Fallback
        
        # Convert back to probability
        posterior_prob = 1.0 / (1.0 + np.exp(-posterior_logit))
        
        # Calculate posterior confidence based on total precision
        # More models with high confidence = higher overall confidence
        posterior_confidence = min(0.95, np.tanh(np.sum(precisions) / len(precisions)))
        
        return float(posterior_prob), float(posterior_confidence)
    
    def record_performance(
        self,
        source_type: str,
        model: str,
        prediction: float,
        actual_outcome: float,
        timestamp: datetime = None
    ) -> None:
        """Record performance for a sentiment source/model.
        
        Args:
            source_type: Source type (e.g., "social_media", "llm")
            model: Model name
            prediction: Predicted sentiment value (0-1)
            actual_outcome: Actual market outcome (0-1)
            timestamp: Timestamp of the prediction (optional)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        # Calculate error (lower is better)
        error = abs(prediction - actual_outcome)
        
        # Convert to performance score (0-1, higher is better)
        # Use a non-linear scoring that heavily penalizes large errors
        performance_score = 1.0 - min(1.0, (error * 2) ** 1.5)
        
        # Create unique key
        key = f"{source_type}:{model}"
        
        # Get existing score if available
        existing_score = self.performance_memory.get(key)
        
        if existing_score is not None:
            # Apply exponential moving average (more weight to recent performance)
            alpha = 0.2  # Weight for new data
            updated_score = (alpha * performance_score) + ((1 - alpha) * existing_score)
        else:
            # First record
            updated_score = performance_score
            
        # Store updated score
        self.performance_memory.set(key, updated_score)
        
        self.logger.debug(f"Recorded performance for {source_type}:{model}: {performance_score:.3f}",
                        error=error, updated_score=updated_score)
    
    def get_model_performance(self, source_type: str = None, model: str = None) -> Dict[str, float]:
        """Get performance scores for models or sources.
        
        Args:
            source_type: Filter by source type (optional)
            model: Filter by model name (optional)
            
        Returns:
            Dictionary of performance scores
        """
        results = {}
        
        # Scan all keys in performance memory
        for key, score in self.performance_memory.items():
            if ":" not in key:
                continue
                
            key_source, key_model = key.split(":", 1)
            
            # Apply filters if specified
            if source_type and key_source != source_type:
                continue
            if model and key_model != model:
                continue
                
            results[key] = score
            
        return results
    
    def _get_default_result(self) -> Dict[str, Any]:
        """Get default consensus result when insufficient data.
        
        Returns:
            Default consensus result
        """
        return {
            "value": 0.5,
            "direction": "neutral",
            "confidence": 0.0,
            "source_count": 0,
            "unique_source_types": 0,
            "unique_models": 0,
            "disagreement_level": 0.0,
            "last_update": datetime.utcnow()
        }


class MultiModelConsensusAgent(BaseSentimentAgent):
    """Sentiment agent that uses multi-model consensus.
    
    This agent collects sentiment from multiple sources and models,
    and uses the consensus system to produce high-confidence signals.
    """
    
    def __init__(self, agent_id: str):
        """Initialize the multi-model consensus agent.
        
        Args:
            agent_id: The unique identifier for this agent
        """
        super().__init__(agent_id)
        self.logger = get_logger("analysis_agents", "consensus_agent")
        
        # Setup consensus system
        self.consensus = ConsensusSystem(agent_id)
        
        # Configure sources
        self.data_sources = config.get(
            f"analysis_agents.{agent_id}.data_sources", 
            ["llm", "social_media", "news", "market_sentiment", "onchain"]
        )
        
        # Update interval
        self.update_interval = config.get(
            f"analysis_agents.{agent_id}.update_interval", 
            1800  # Default: 30 minutes
        )
        
        # Collection of sentiment data points for consensus
        self.sentiment_points: Dict[str, List[Dict[str, Any]]] = {}
        
        # Cache for consensus results
        self.consensus_cache: Dict[str, Dict[str, Any]] = {}
        
        # Market data for price comparisons
        self.market_data_cache: Dict[str, List[CandleData]] = {}
    
    async def _initialize(self) -> None:
        """Initialize the consensus agent."""
        await super()._initialize()
        
        if not self.enabled:
            return
            
        self.logger.info("Initializing multi-model consensus agent",
                       data_sources=self.data_sources)
        
        # Initialize sentiment points container
        self.sentiment_points = {}
        
        # Record initialization metric
        metrics.counter("consensus_agent_initialized", tags={
            "agent_id": self.agent_id
        })
    
    async def _start(self) -> None:
        """Start the consensus agent."""
        await super()._start()
        
        if not self.enabled:
            return
            
        # Start update task for consensus processing
        self.update_task = self.create_task(
            self._process_consensus_periodically()
        )
        
        # Record start metric
        metrics.counter("consensus_agent_started", tags={
            "agent_id": self.agent_id
        })
    
    async def _stop(self) -> None:
        """Stop the consensus agent."""
        # Cancel update task
        if hasattr(self, "update_task") and not self.update_task.done():
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Record stop metric
        metrics.counter("consensus_agent_stopped", tags={
            "agent_id": self.agent_id
        })
        
        await super()._stop()
    
    async def submit_sentiment(
        self, 
        symbol: str, 
        value: float,
        direction: str,
        confidence: float,
        source_type: str,
        model: str = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Submit a sentiment data point for consensus processing.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            value: Sentiment value (0-1)
            direction: Sentiment direction (bullish/bearish/neutral)
            confidence: Confidence level (0-1)
            source_type: Source type (e.g., "social_media", "news")
            model: Model name (optional)
            metadata: Additional metadata (optional)
        """
        if not self.enabled:
            return
            
        # Create sentiment data point
        data_point = {
            "value": value,
            "direction": direction,
            "confidence": confidence,
            "source_type": source_type,
            "timestamp": datetime.utcnow(),
        }
        
        # Add model if provided
        if model:
            data_point["model"] = model
            
        # Add metadata if provided
        if metadata:
            data_point.update(metadata)
            
        # Add to sentiment points
        if symbol not in self.sentiment_points:
            self.sentiment_points[symbol] = []
            
        self.sentiment_points[symbol].append(data_point)
        
        # Log the submission
        self.logger.debug(f"Submitted sentiment for {symbol}: {direction} ({value:.2f})",
                        source=source_type, model=model, confidence=confidence)
        
        # If we have accumulated enough points, compute consensus
        min_points = 3
        if len(self.sentiment_points[symbol]) >= min_points:
            self.create_task(self._process_consensus(symbol))
    
    async def _process_consensus_periodically(self) -> None:
        """Process consensus periodically for all symbols."""
        try:
            while True:
                # Process consensus for all symbols with data
                for symbol in list(self.sentiment_points.keys()):
                    if self.sentiment_points[symbol]:
                        await self._process_consensus(symbol)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
        except asyncio.CancelledError:
            self.logger.debug("Consensus processing task cancelled")
            raise
        except Exception as e:
            self.logger.error("Error in consensus processing task", error=str(e))
            
            # Record error metric
            metrics.counter("consensus_agent_error", tags={
                "agent_id": self.agent_id,
                "error_type": type(e).__name__
            })
            
            await asyncio.sleep(60)  # Wait a minute and retry
    
    async def _process_consensus(self, symbol: str) -> None:
        """Process consensus for a single symbol.
        
        Args:
            symbol: The trading pair symbol
        """
        if not self.sentiment_points.get(symbol):
            return
            
        self.logger.debug(f"Processing consensus for {symbol} with {len(self.sentiment_points[symbol])} data points")
        
        # Compute consensus
        consensus_result = self.consensus.compute_consensus(self.sentiment_points[symbol])
        
        # Save to consensus cache
        previous_consensus = self.consensus_cache.get(symbol, {}).get("value", 0.5)
        consensus_shift = abs(consensus_result["value"] - previous_consensus)
        
        self.consensus_cache[symbol] = consensus_result
        
        # Log the consensus result
        self.logger.info(f"Consensus for {symbol}: {consensus_result['direction']} ({consensus_result['value']:.2f})",
                       confidence=consensus_result["confidence"],
                       sources=consensus_result["source_count"],
                       disagreement=consensus_result["disagreement_level"])
        
        # Record metrics
        metrics.gauge("consensus_sentiment_value", consensus_result["value"], tags={
            "symbol": symbol,
            "direction": consensus_result["direction"]
        })
        
        metrics.gauge("consensus_confidence", consensus_result["confidence"], tags={
            "symbol": symbol
        })
        
        metrics.gauge("consensus_disagreement", consensus_result["disagreement_level"], tags={
            "symbol": symbol
        })
        
        # Update the sentiment cache for other components
        self._update_sentiment_cache(
            symbol=symbol,
            source_type="consensus",
            sentiment_value=consensus_result["value"],
            direction=consensus_result["direction"],
            confidence=consensus_result["confidence"],
            additional_data={
                "disagreement_level": consensus_result["disagreement_level"],
                "source_count": consensus_result["source_count"],
                "source_types": consensus_result["source_types"],
                "models": consensus_result.get("models", []),
                "direction_counts": consensus_result.get("direction_counts", {})
            }
        )
        
        # Publish sentiment event if significant or high confidence
        is_significant = consensus_shift > self.sentiment_shift_threshold
        is_high_confidence = consensus_result["confidence"] > 0.85
        
        if is_significant or is_high_confidence:
            event_type = "consensus_shift" if is_significant else "high_confidence_consensus"
            
            details = {
                "disagreement_level": consensus_result["disagreement_level"],
                "source_count": consensus_result["source_count"],
                "source_types": consensus_result["source_types"],
                "models": consensus_result.get("models", []),
                "direction_counts": consensus_result.get("direction_counts", {}),
                "event_type": event_type
            }
            
            # Add flags for disagreement if present
            if consensus_result.get("has_major_disagreement"):
                details["has_major_disagreement"] = True
                details["consensus_method"] = consensus_result.get("consensus_method", "weighted_average")
            elif consensus_result.get("has_minor_disagreement"):
                details["has_minor_disagreement"] = True
            
            await self.publish_sentiment_event(
                symbol=symbol,
                direction=consensus_result["direction"],
                value=consensus_result["value"],
                confidence=consensus_result["confidence"],
                is_extreme=abs(consensus_result["value"] - 0.5) > 0.3,
                signal_type="consensus",
                sources=consensus_result["source_types"],
                details=details
            )
            
            self.logger.info("Published consensus sentiment event", 
                           symbol=symbol,
                           direction=consensus_result["direction"],
                           value=consensus_result["value"], 
                           confidence=consensus_result["confidence"],
                           event_type=event_type)
            
            # Record event metric
            metrics.counter("consensus_event_published", tags={
                "symbol": symbol,
                "direction": consensus_result["direction"],
                "event_type": event_type
            })
            
        # Clear old data points to avoid memory bloat
        # Keep only data from the last day
        cutoff_time = datetime.utcnow() - timedelta(days=1)
        self.sentiment_points[symbol] = [
            point for point in self.sentiment_points[symbol]
            if point.get("timestamp", datetime.utcnow()) > cutoff_time
        ]
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        candles: List[Any]
    ) -> None:
        """Analyze market data for performance tracking.
        
        Args:
            symbol: The trading pair symbol
            exchange: Exchange name
            timeframe: Time frame
            candles: List of candle data
        """
        await super().analyze_market_data(symbol, exchange, timeframe, candles)
        
        if not candles or len(candles) < 3:
            return
            
        # Save market data for this symbol
        self.market_data_cache[symbol] = candles
        
        # Track performance if we have consensus data
        if symbol in self.consensus_cache:
            # Get consensus from several candles ago
            lookback = 3  # Number of candles to look back
            
            # Get consensus timestamp
            consensus_time = self.consensus_cache[symbol].get("last_update")
            
            # If this consensus was made around the time of the old candle
            if consensus_time and len(candles) > lookback:
                old_candle_time = candles[-lookback].timestamp
                
                # If the consensus was made within 1 hour of the old candle
                if abs((consensus_time - old_candle_time).total_seconds()) < 3600:
                    # Calculate actual market movement since then
                    price_change = (candles[-1].close / candles[-lookback].close) - 1
                    
                    # Normalize to 0-1 range (scale price change to sentiment range)
                    # A 10% price change in either direction maps to 0 or 1
                    normalized_outcome = 0.5 + (price_change * 5)  # Scale to 0-1 range
                    normalized_outcome = max(0, min(1, normalized_outcome))
                    
                    # Record performance for all models and sources in this consensus
                    consensus_data = self.consensus_cache[symbol]
                    consensus_value = consensus_data.get("value", 0.5)
                    
                    # Record overall consensus performance
                    self.consensus.record_performance(
                        source_type="consensus",
                        model="ensemble",
                        prediction=consensus_value,
                        actual_outcome=normalized_outcome,
                        timestamp=consensus_time
                    )
                    
                    # Get individual data points that were used in this consensus
                    for point in self.sentiment_points[symbol]:
                        point_time = point.get("timestamp")
                        if point_time and abs((point_time - old_candle_time).total_seconds()) < 7200:
                            # This point was probably used in that consensus
                            source_type = point.get("source_type", "unknown")
                            model = point.get("model", "unknown")
                            value = point.get("value", 0.5)
                            
                            # Record individual performance
                            self.consensus.record_performance(
                                source_type=source_type,
                                model=model,
                                prediction=value,
                                actual_outcome=normalized_outcome,
                                timestamp=point_time
                            )
                            
                    self.logger.debug(f"Recorded performance for {symbol} consensus",
                                    consensus_value=consensus_value,
                                    actual_outcome=normalized_outcome,
                                    price_change=price_change)
    
    def get_consensus(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest consensus for a symbol.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Latest consensus result or None if not available
        """
        return self.consensus_cache.get(symbol)
    
    def get_performance_metrics(self, symbol: str = None) -> Dict[str, Any]:
        """Get performance metrics for models and sources.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Dictionary of performance metrics
        """
        # Get raw performance scores
        performance_scores = self.consensus.get_model_performance()
        
        # Group by source type and model
        source_performance = defaultdict(float)
        model_performance = defaultdict(float)
        source_counts = defaultdict(int)
        model_counts = defaultdict(int)
        
        for key, score in performance_scores.items():
            source, model = key.split(":", 1)
            
            # Add to source metrics
            source_performance[source] += score
            source_counts[source] += 1
            
            # Add to model metrics
            model_performance[model] += score
            model_counts[model] += 1
        
        # Calculate averages
        avg_source_performance = {
            source: score / source_counts[source]
            for source, score in source_performance.items()
        }
        
        avg_model_performance = {
            model: score / model_counts[model]
            for model, score in model_performance.items()
        }
        
        # Create final metrics
        return {
            "source_performance": avg_source_performance,
            "model_performance": avg_model_performance,
            "raw_scores": performance_scores,
            "timestamp": datetime.utcnow().isoformat()
        }