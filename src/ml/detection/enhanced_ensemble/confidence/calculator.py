"""
Confidence Calculator

This module implements the main calculator for regime detection confidence scoring.
"""

import time
import threading
import logging
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, TypeVar, Generic, Callable
from collections import deque
import pandas as pd
from datetime import datetime

from src.ml.detection.enhanced_ensemble.confidence.models import (
    ConfidenceConfig, ConfidenceResult, ConfidenceLevel, 
    CacheEntry, FactorScore, ConfidenceHistoryEntry, PerformanceMetrics
)
from src.ml.detection.enhanced_ensemble.confidence.factors import (
    ConfidenceFactor, InterDetectorAgreementFactor, 
    HistoricalAccuracyFactor, DataQualityFactor, RegimeBoundaryProximityFactor
)


class ConfidenceCalculator:
    """
    Thread-safe calculator for regime detection confidence scores.
    
    This class orchestrates the calculation of confidence scores by evaluating
    multiple confidence factors and combining their results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the confidence calculator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._validate_config(config or {})
        self._lock = threading.RLock()
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_cache_cleanup = time.time()
        self._history_window_size = self.config.get('history_window_size', 100)
        self._confidence_history: deque = deque(maxlen=self._history_window_size)
        self._initialize_factors()
        
    def calculate_confidence(self, 
                           detector_outputs: Dict[str, Dict[str, Any]],
                           market_context: Dict[str, Any],
                           historical_data: Optional[pd.DataFrame] = None) -> ConfidenceResult:
        """
        Calculate overall confidence score in a thread-safe manner.
        
        Args:
            detector_outputs: Dictionary mapping detector names to their outputs
            market_context: Dictionary containing current market conditions
            historical_data: Optional historical market data
            
        Returns:
            ConfidenceResult containing overall confidence score and detailed data
        """
        start_time = time.perf_counter()
        
        # Generate cache key if caching is enabled
        cache_key = None
        if self.config.get('enable_caching', True):
            cache_key = self._generate_cache_key(detector_outputs, market_context)
            
            # Check cache for existing result
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                # Add performance data to cached result
                calculation_time = time.perf_counter() - start_time
                
                cached_result['performance'] = {
                    'calculation_time_ms': calculation_time * 1000,
                    'cache_hit': True
                }
                
                return cached_result
        
        # Process factors (in parallel if configured)
        if self.config.get('parallel_calculation', True):
            factor_scores = self._calculate_factor_scores_parallel(
                detector_outputs, market_context, historical_data)
        else:
            factor_scores = self._calculate_factor_scores_sequential(
                detector_outputs, market_context, historical_data)
            
        # Calculate overall confidence with volatility adjustment
        volatility_modifier = self._calculate_volatility_modifier(market_context)
        
        # Calculate weighted average of factor scores
        result = self._aggregate_confidence(factor_scores, volatility_modifier)
            
        # Add calculation timestamp
        result['calculation_timestamp'] = datetime.now()
        
        # Add performance metrics
        calculation_time = time.perf_counter() - start_time
        result['performance'] = {
            'calculation_time_ms': calculation_time * 1000,
            'cache_hit': False,
            'factor_calculation_times': {
                name: score.get('metadata', {}).get('calculation_time_ms', 0.0)
                for name, score in factor_scores.items()
            }
        }
        
        # Update cache if caching is enabled
        if cache_key is not None:
            self._add_to_cache(cache_key, result)
            
        # Update confidence history
        self._update_confidence_history(result, market_context)
        
        # Periodically clean expired cache entries
        self._clean_expired_cache_entries()
        
        return result
    
    def get_confidence_history(self) -> List[ConfidenceHistoryEntry]:
        """
        Get the history of confidence scores.
        
        Returns:
            List of historical confidence entries
        """
        with self._lock:
            return list(self._confidence_history)
    
    def update_historical_accuracy(self, 
                                 detector: str, 
                                 was_correct: bool, 
                                 regime: Optional[str] = None) -> None:
        """
        Update historical accuracy for a detector.
        
        This method should be called after ground truth becomes available.
        
        Args:
            detector: Name of the detector
            was_correct: Whether the prediction was correct
            regime: Optional regime for which this prediction was made
        """
        historical_factor = self._get_historical_accuracy_factor()
        if historical_factor:
            historical_factor.update_accuracy(detector, was_correct, regime)
    
    def _initialize_factors(self) -> None:
        """Initialize confidence factors."""
        with self._lock:
            # Create the default factors with their respective configurations
            factor_configs = self.config.get('factor_configs', {})
            
            self.factors: Dict[str, ConfidenceFactor] = {
                'agreement': InterDetectorAgreementFactor(
                    factor_configs.get('agreement', {})),
                'historical': HistoricalAccuracyFactor(
                    factor_configs.get('historical', {})),
                'data_quality': DataQualityFactor(
                    factor_configs.get('data_quality', {})),
                'boundary': RegimeBoundaryProximityFactor(
                    factor_configs.get('boundary', {}))
            }
            
            # Add any custom factors from configuration
            custom_factors = self.config.get('custom_factors', {})
            self.factors.update(custom_factors)
    
    def _get_historical_accuracy_factor(self) -> Optional[HistoricalAccuracyFactor]:
        """
        Get the historical accuracy factor for updating accuracy.
        
        Returns:
            HistoricalAccuracyFactor if found, None otherwise
        """
        with self._lock:
            for factor in self.factors.values():
                if isinstance(factor, HistoricalAccuracyFactor):
                    return factor
            return None
    
    def _calculate_factor_scores_sequential(self, 
                                          detector_outputs: Dict[str, Dict[str, Any]],
                                          market_context: Dict[str, Any],
                                          historical_data: Optional[pd.DataFrame] = None) -> Dict[str, FactorScore]:
        """
        Calculate all factor scores sequentially.
        
        Args:
            detector_outputs: Dictionary of detector outputs
            market_context: Dictionary of market context data
            historical_data: Optional historical market data
            
        Returns:
            Dictionary mapping factor names to their scores
        """
        factor_scores = {}
        
        for name, factor in self.factors.items():
            start_time = time.perf_counter()
            
            try:
                score = factor.calculate(detector_outputs, market_context, historical_data)
                metadata = factor.get_metadata()
                
                # Add calculation time to metadata
                calculation_time = time.perf_counter() - start_time
                if 'calculation_time_ms' not in metadata:
                    metadata['calculation_time_ms'] = calculation_time * 1000
                
                # Get weight for this factor
                weight = self.config.get('factor_weights', {}).get(name, 1.0)
                
                factor_scores[name] = {
                    'name': name,
                    'score': score,
                    'weight': weight,
                    'weighted_score': score * weight,
                    'metadata': metadata
                }
            except Exception as e:
                self.logger.error(f"Error calculating {name} factor: {str(e)}")
                
                # Use default error score
                default_score = self.config.get('default_error_score', 0.5)
                weight = self.config.get('factor_weights', {}).get(name, 1.0)
                
                factor_scores[name] = {
                    'name': name,
                    'score': default_score,
                    'weight': weight,
                    'weighted_score': default_score * weight,
                    'metadata': {
                        'error': str(e),
                        'calculation_time_ms': (time.perf_counter() - start_time) * 1000
                    }
                }
        
        return factor_scores
    
    def _calculate_factor_scores_parallel(self, 
                                        detector_outputs: Dict[str, Dict[str, Any]],
                                        market_context: Dict[str, Any],
                                        historical_data: Optional[pd.DataFrame] = None) -> Dict[str, FactorScore]:
        """
        Calculate all factor scores in parallel.
        
        Args:
            detector_outputs: Dictionary of detector outputs
            market_context: Dictionary of market context data
            historical_data: Optional historical market data
            
        Returns:
            Dictionary mapping factor names to their scores
        """
        factor_scores = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all factor calculations to the thread pool
            factor_futures = {}
            
            for name, factor in self.factors.items():
                start_time = time.perf_counter()
                
                # Capture start time for each factor
                factor_futures[executor.submit(
                    self._calculate_factor_score, name, factor,
                    (detector_outputs, market_context, historical_data)
                )] = (name, start_time)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(factor_futures):
                name, start_time = factor_futures[future]
                
                try:
                    score_result = future.result()
                    
                    # Add result to factor scores
                    factor_scores[name] = score_result
                    
                except Exception as e:
                    self.logger.error(f"Error calculating {name} factor: {str(e)}")
                    
                    # Use default error score
                    default_score = self.config.get('default_error_score', 0.5)
                    weight = self.config.get('factor_weights', {}).get(name, 1.0)
                    
                    factor_scores[name] = {
                        'name': name,
                        'score': default_score,
                        'weight': weight,
                        'weighted_score': default_score * weight,
                        'metadata': {
                            'error': str(e),
                            'calculation_time_ms': (time.perf_counter() - start_time) * 1000
                        }
                    }
        
        return factor_scores
    
    def _calculate_factor_score(self, 
                              factor_name: str, 
                              factor: ConfidenceFactor,
                              inputs: Tuple) -> FactorScore:
        """
        Calculate a single factor score with comprehensive error handling.
        
        Args:
            factor_name: Name of the factor
            factor: Factor instance
            inputs: Tuple of (detector_outputs, market_context, historical_data)
            
        Returns:
            FactorScore with score and metadata
        """
        detector_outputs, market_context, historical_data = inputs
        start_time = time.perf_counter()
        
        try:
            score = factor.calculate(detector_outputs, market_context, historical_data)
            metadata = factor.get_metadata()
            
            # Add calculation time to metadata
            calculation_time = time.perf_counter() - start_time
            if 'calculation_time_ms' not in metadata:
                metadata['calculation_time_ms'] = calculation_time * 1000
            
            # Get weight for this factor
            weight = self.config.get('factor_weights', {}).get(factor_name, 1.0)
            
            return {
                'name': factor_name,
                'score': score,
                'weight': weight,
                'weighted_score': score * weight,
                'metadata': metadata
            }
        except Exception as e:
            # Capture detailed error context
            error_context = {
                'factor_name': factor_name,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'market_data_length': len(historical_data) if historical_data is not None else 0,
                'detector_count': len(detector_outputs) if detector_outputs else 0
            }
            
            # Log detailed error information
            self.logger.error(
                f"Error calculating confidence factor {factor_name}: {str(e)}",
                extra={'error_context': error_context}
            )
            
            # Return safe default value
            default_score = self.config.get('default_error_score', 0.5)
            weight = self.config.get('factor_weights', {}).get(factor_name, 1.0)
            
            return {
                'name': factor_name,
                'score': default_score,
                'weight': weight,
                'weighted_score': default_score * weight,
                'metadata': {
                    'error': str(e),
                    'error_context': error_context,
                    'calculation_time_ms': (time.perf_counter() - start_time) * 1000
                }
            }
    
    def _calculate_volatility_modifier(self, market_context: Dict[str, Any]) -> float:
        """
        Calculate confidence modifier based on market volatility.
        
        Args:
            market_context: Dictionary of market context data
            
        Returns:
            Volatility modifier between 0.5 and 1.1
        """
        # Extract volatility metrics from market context
        volatility = market_context.get('volatility', 0.0)
        avg_volatility = market_context.get('average_volatility', volatility)
        
        # Calculate relative volatility (how current volatility compares to average)
        if avg_volatility > 0:
            relative_volatility = volatility / avg_volatility
        else:
            relative_volatility = 1.0
        
        # High volatility should reduce confidence ceiling
        if relative_volatility > 2.5:
            # Extreme volatility: significant reduction in confidence
            return 0.6
        elif relative_volatility > 2.0:
            # Very high volatility: substantial reduction
            return 0.7
        elif relative_volatility > 1.5:
            # High volatility: moderate reduction
            return 0.85
        elif relative_volatility < 0.5:
            # Very low volatility: slight increase in confidence
            return 1.1
        else:
            # Normal volatility: no adjustment
            return 1.0
    
    def _aggregate_confidence(self, 
                            factor_scores: Dict[str, FactorScore],
                            volatility_modifier: float) -> ConfidenceResult:
        """
        Aggregate factor scores into an overall confidence score.
        
        Args:
            factor_scores: Dictionary mapping factor names to their scores
            volatility_modifier: Modifier based on market volatility
            
        Returns:
            ConfidenceResult with overall confidence score and detailed data
        """
        if not factor_scores:
            # No factors calculated, return default score
            overall_confidence = self.config.get('default_error_score', 0.5)
            confidence_level = ConfidenceLevel.from_score(overall_confidence)
            
            return {
                'overall_confidence': overall_confidence,
                'confidence_level': confidence_level,
                'factor_scores': {},
                'adjusted_by_volatility': False,
                'volatility_modifier': 1.0,
                'performance': None,
                'calculation_timestamp': datetime.now()
            }
        
        # Calculate total weight
        total_weight = sum(score['weight'] for score in factor_scores.values())
        
        if total_weight == 0:
            # All weights are zero, use equal weights
            total_weight = len(factor_scores)
            weighted_sum = sum(score['score'] for score in factor_scores.values())
        else:
            # Use weighted sum
            weighted_sum = sum(score['weighted_score'] for score in factor_scores.values())
        
        # Calculate raw confidence score
        raw_confidence = weighted_sum / total_weight
        
        # Apply volatility modifier
        adjusted_confidence = raw_confidence * volatility_modifier
        
        # Ensure confidence is in valid range
        overall_confidence = min(1.0, max(0.0, adjusted_confidence))
        
        # Get confidence level
        confidence_level = ConfidenceLevel.from_score(overall_confidence)
        
        # Return comprehensive result
        return {
            'overall_confidence': overall_confidence,
            'confidence_level': confidence_level,
            'factor_scores': factor_scores,
            'adjusted_by_volatility': volatility_modifier != 1.0,
            'volatility_modifier': volatility_modifier,
            'performance': None,  # Will be filled later
            'calculation_timestamp': datetime.now()
        }
    
    def _update_confidence_history(self, 
                                 result: ConfidenceResult,
                                 market_context: Dict[str, Any]) -> None:
        """
        Update the confidence history with the new result.
        
        Args:
            result: Confidence calculation result
            market_context: Market context data
        """
        with self._lock:
            # Extract current regime from context or detector outputs
            current_regime = market_context.get('regime', 'unknown')
            
            # Create history entry
            entry: ConfidenceHistoryEntry = {
                'timestamp': result['calculation_timestamp'],
                'confidence': result['overall_confidence'],
                'factors': {name: score['score'] for name, score in result['factor_scores'].items()},
                'regime': current_regime,
                'market_context': {
                    k: v for k, v in market_context.items()
                    if k in ['volatility', 'average_volatility', 'trend_strength']
                }
            }
            
            # Add to history
            self._confidence_history.append(entry)
    
    def _generate_cache_key(self, 
                          detector_outputs: Dict[str, Dict[str, Any]],
                          market_context: Dict[str, Any]) -> str:
        """
        Generate a cache key for the given inputs.
        
        Args:
            detector_outputs: Dictionary of detector outputs
            market_context: Dictionary of market context data
            
        Returns:
            Cache key string
        """
        # Extract only essential data for cache key
        key_parts = []
        
        # Add detector regimes
        for name, output in sorted(detector_outputs.items()):
            if 'regime' in output:
                key_parts.append(f"{name}:{output['regime']}")
        
        # Add essential market context
        for key in ['timestamp', 'volatility', 'regime']:
            if key in market_context:
                key_parts.append(f"{key}:{market_context[key]}")
        
        # Join all parts
        return "|".join(key_parts)
    
    def _add_to_cache(self, key: str, value: ConfidenceResult) -> None:
        """
        Add result to cache with expiration timestamp.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            expiration_time = time.time() + self.config.get('cache_expiration_seconds', 300)
            
            self._cache[key] = {
                'value': value,
                'expires_at': expiration_time,
                'created_at': time.time()
            }
            
            self._cache_misses += 1
    
    def _get_from_cache(self, key: str) -> Optional[ConfidenceResult]:
        """
        Get value from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                return None
                
            cache_entry = self._cache[key]
            if time.time() > cache_entry['expires_at']:
                # Entry expired
                del self._cache[key]
                return None
                
            self._cache_hits += 1
            return cache_entry['value']
    
    def _clean_expired_cache_entries(self) -> None:
        """Clean expired entries from the cache."""
        current_time = time.time()
        
        # Only clean if enough time has passed since last cleanup
        cleanup_interval = self.config.get('cache_cleanup_interval_seconds', 60)
        if current_time - self._last_cache_cleanup < cleanup_interval:
            return
            
        with self._lock:
            # Find expired keys
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time > entry['expires_at']
            ]
            
            # Remove expired entries
            for key in expired_keys:
                del self._cache[key]
                
            # Update last cleanup time
            self._last_cache_cleanup = current_time
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration with defaults applied
        """
        if not config:
            return self._default_config()
            
        # Set defaults for missing values
        validated = self._default_config()
        validated.update(config)
        
        # Validate factor weights
        weights = validated.get('factor_weights', {})
        if not isinstance(weights, dict):
            raise TypeError("factor_weights must be a dictionary")
            
        # Ensure all weights are positive
        for factor, weight in weights.items():
            if weight < 0:
                raise ValueError(f"Weight for factor {factor} must be positive")
        
        # Validate thresholds are in proper range
        for key, value in validated.items():
            if key.endswith('_threshold') and (value < 0 or value > 1):
                raise ValueError(f"Threshold {key} must be between 0 and 1")
                
        return validated
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'factor_weights': {
                'agreement': 0.35,
                'historical': 0.30,
                'data_quality': 0.15,
                'boundary': 0.20
            },
            'enable_caching': True,
            'cache_expiration_seconds': 300,
            'cache_cleanup_interval_seconds': 60,
            'default_error_score': 0.5,
            'min_data_quality_threshold': 0.3,
            'history_window_size': 100,
            'parallel_calculation': True,
            'factor_configs': {
                'agreement': {},
                'historical': {'window_size': 100},
                'data_quality': {'expected_update_hours': 24},
                'boundary': {}
            }
        }
        
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                'cache_size': len(self._cache),
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
    
    def to_visualization_format(self) -> Dict[str, Any]:
        """
        Convert confidence data to visualization-friendly format.
        
        Returns:
            Dictionary with visualization-friendly data
        """
        with self._lock:
            history = list(self._confidence_history)
            
            return {
                'cache_statistics': self.get_cache_statistics(),
                'history': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'confidence': float(entry['confidence']),
                        'regime': entry['regime'],
                        'factors': {k: float(v) for k, v in entry['factors'].items()}
                    }
                    for entry in history
                ]
            }
