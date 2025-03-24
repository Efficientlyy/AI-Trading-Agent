"""A/B Testing Framework for sentiment analysis.

This module provides a comprehensive A/B testing framework for experimenting with
various aspects of the sentiment analysis system including prompt templates,
model parameters, and analysis strategies.
"""

import asyncio
import json
import os
import random
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.stats as stats
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event
from src.common.caching import Cache
from src.analysis_agents.sentiment.performance_tracker import performance_tracker
from src.analysis_agents.sentiment.prompt_tuning import prompt_tuning_system


class ExperimentType(Enum):
    """Types of experiments that can be conducted."""
    PROMPT_TEMPLATE = "prompt_template"
    MODEL_SELECTION = "model_selection"
    TEMPERATURE = "temperature"
    CONTEXT_STRATEGY = "context_strategy"
    AGGREGATION_WEIGHTS = "aggregation_weights"
    UPDATE_FREQUENCY = "update_frequency"
    CONFIDENCE_THRESHOLD = "confidence_threshold"


class ExperimentStatus(Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ANALYZED = "analyzed"
    IMPLEMENTED = "implemented"
    ARCHIVED = "archived"


class TargetingCriteria(Enum):
    """Targeting criteria for experiment variants."""
    ALL_TRAFFIC = "all_traffic"
    SYMBOL_SPECIFIC = "symbol_specific"
    SOURCE_SPECIFIC = "source_specific"
    TIME_SPECIFIC = "time_specific"
    RANDOM_ASSIGNMENT = "random_assignment"
    MARKET_CONDITION = "market_condition"


class VariantAssignmentStrategy(Enum):
    """Strategy for assigning users to variants."""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    SESSION_STICKY = "session_sticky"
    SYMBOL_HASH = "symbol_hash"
    TIME_BASED = "time_based"
    CONTEXT_BASED = "context_based"


@dataclass
class ExperimentVariant:
    """Represents a variant within an experiment."""
    id: str
    name: str
    description: str
    weight: float = 0.5
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    control: bool = False


@dataclass
class ExperimentMetrics:
    """Metrics recorded for an experiment."""
    requests: int = 0
    successes: int = 0
    errors: int = 0
    total_latency: float = 0.0
    sentiment_accuracy: float = 0.0
    calibration_error: float = 0.0
    direction_accuracy: float = 0.0
    confidence_score: float = 0.0
    user_overrides: int = 0
    
    def get_success_rate(self) -> float:
        """Get the success rate."""
        if self.requests == 0:
            return 0.0
        return self.successes / self.requests
    
    def get_average_latency(self) -> float:
        """Get the average latency."""
        if self.successes == 0:
            return 0.0
        return self.total_latency / self.successes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "requests": self.requests,
            "successes": self.successes,
            "errors": self.errors,
            "success_rate": self.get_success_rate(),
            "average_latency": self.get_average_latency(),
            "sentiment_accuracy": self.sentiment_accuracy,
            "calibration_error": self.calibration_error,
            "direction_accuracy": self.direction_accuracy,
            "confidence_score": self.confidence_score,
            "user_overrides": self.user_overrides
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentMetrics':
        """Create metrics from dictionary."""
        metrics = cls()
        metrics.requests = data.get("requests", 0)
        metrics.successes = data.get("successes", 0)
        metrics.errors = data.get("errors", 0)
        metrics.total_latency = data.get("total_latency", 0.0)
        metrics.sentiment_accuracy = data.get("sentiment_accuracy", 0.0)
        metrics.calibration_error = data.get("calibration_error", 0.0)
        metrics.direction_accuracy = data.get("direction_accuracy", 0.0)
        metrics.confidence_score = data.get("confidence_score", 0.0)
        metrics.user_overrides = data.get("user_overrides", 0)
        return metrics


class Experiment:
    """Represents an A/B testing experiment."""
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        experiment_type: ExperimentType,
        variants: List[ExperimentVariant],
        targeting: List[TargetingCriteria] = None,
        assignment_strategy: VariantAssignmentStrategy = VariantAssignmentStrategy.RANDOM,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        sample_size: Optional[int] = None,
        min_confidence: float = 0.95,
        owner: str = "system",
        status: ExperimentStatus = ExperimentStatus.DRAFT,
        metadata: Dict[str, Any] = None
    ):
        """Initialize an experiment.
        
        Args:
            id: Unique identifier for the experiment
            name: Name of the experiment
            description: Description of the experiment
            experiment_type: Type of experiment
            variants: List of variants to test
            targeting: Targeting criteria for the experiment
            assignment_strategy: Strategy for assigning variants
            start_time: Start time for the experiment
            end_time: End time for the experiment
            sample_size: Target sample size for statistical significance
            min_confidence: Minimum confidence level for significance
            owner: Owner of the experiment
            status: Current status of the experiment
            metadata: Additional metadata
        """
        self.id = id
        self.name = name
        self.description = description
        self.experiment_type = experiment_type
        self.variants = variants
        self.targeting = targeting or [TargetingCriteria.ALL_TRAFFIC]
        self.assignment_strategy = assignment_strategy
        self.start_time = start_time
        self.end_time = end_time
        self.sample_size = sample_size
        self.min_confidence = min_confidence
        self.owner = owner
        self.status = status
        self.metadata = metadata or {}
        
        # Initialize metrics for each variant
        self.variant_metrics: Dict[str, ExperimentMetrics] = {
            variant.id: ExperimentMetrics() for variant in variants
        }
        
        # Track assignments
        self.assignments: Dict[str, str] = {}  # key -> variant_id
        
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at
        self.results: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "experiment_type": self.experiment_type.value,
            "variants": [
                {
                    "id": v.id,
                    "name": v.name,
                    "description": v.description,
                    "weight": v.weight,
                    "config": v.config,
                    "metadata": v.metadata,
                    "control": v.control
                } for v in self.variants
            ],
            "targeting": [t.value for t in self.targeting],
            "assignment_strategy": self.assignment_strategy.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "sample_size": self.sample_size,
            "min_confidence": self.min_confidence,
            "owner": self.owner,
            "status": self.status.value,
            "metadata": self.metadata,
            "variant_metrics": {
                variant_id: metrics.to_dict()
                for variant_id, metrics in self.variant_metrics.items()
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "results": self.results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create an experiment from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Experiment instance
        """
        # Parse variants
        variants = []
        for v_data in data.get("variants", []):
            variant = ExperimentVariant(
                id=v_data.get("id", str(uuid.uuid4())),
                name=v_data.get("name", ""),
                description=v_data.get("description", ""),
                weight=v_data.get("weight", 0.5),
                config=v_data.get("config", {}),
                metadata=v_data.get("metadata", {}),
                control=v_data.get("control", False)
            )
            variants.append(variant)
        
        # Parse dates
        start_time = None
        if data.get("start_time"):
            try:
                start_time = datetime.fromisoformat(data["start_time"])
            except (ValueError, TypeError):
                pass
                
        end_time = None
        if data.get("end_time"):
            try:
                end_time = datetime.fromisoformat(data["end_time"])
            except (ValueError, TypeError):
                pass
        
        # Create experiment
        experiment = cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            experiment_type=ExperimentType(data.get("experiment_type", "prompt_template")),
            variants=variants,
            targeting=[TargetingCriteria(t) for t in data.get("targeting", ["all_traffic"])],
            assignment_strategy=VariantAssignmentStrategy(data.get("assignment_strategy", "random")),
            start_time=start_time,
            end_time=end_time,
            sample_size=data.get("sample_size"),
            min_confidence=data.get("min_confidence", 0.95),
            owner=data.get("owner", "system"),
            status=ExperimentStatus(data.get("status", "draft")),
            metadata=data.get("metadata", {})
        )
        
        # Set timestamps
        if data.get("created_at"):
            try:
                experiment.created_at = datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                pass
                
        if data.get("updated_at"):
            try:
                experiment.updated_at = datetime.fromisoformat(data["updated_at"])
            except (ValueError, TypeError):
                pass
        
        # Load metrics
        for variant_id, metrics_data in data.get("variant_metrics", {}).items():
            experiment.variant_metrics[variant_id] = ExperimentMetrics.from_dict(metrics_data)
        
        # Load results
        experiment.results = data.get("results", {})
        
        return experiment
    
    def assign_variant(self, context: Dict[str, Any]) -> ExperimentVariant:
        """Assign a variant based on the experiment's assignment strategy.
        
        Args:
            context: Context for the assignment (e.g., symbol, session_id)
            
        Returns:
            Assigned variant
        """
        # Check if this context already has an assignment
        context_key = self._get_context_key(context)
        if context_key in self.assignments:
            variant_id = self.assignments[context_key]
            for variant in self.variants:
                if variant.id == variant_id:
                    return variant
        
        # Apply targeting filters
        if not self._matches_targeting(context):
            # Return the control variant if targeting doesn't match
            for variant in self.variants:
                if variant.control:
                    return variant
            return self.variants[0]  # Fallback to first variant
        
        # Apply assignment strategy
        if self.assignment_strategy == VariantAssignmentStrategy.RANDOM:
            # Random assignment with weights
            weights = [v.weight for v in self.variants]
            variant = random.choices(self.variants, weights=weights, k=1)[0]
        
        elif self.assignment_strategy == VariantAssignmentStrategy.ROUND_ROBIN:
            # Round robin assignment
            assignment_counts = {v.id: 0 for v in self.variants}
            for assigned_variant_id in self.assignments.values():
                if assigned_variant_id in assignment_counts:
                    assignment_counts[assigned_variant_id] += 1
            
            # Choose the variant with the fewest assignments
            min_assignments = min(assignment_counts.values())
            candidates = [v for v in self.variants 
                         if assignment_counts[v.id] == min_assignments]
            variant = random.choice(candidates)
        
        elif self.assignment_strategy == VariantAssignmentStrategy.SESSION_STICKY:
            # Session-based assignment
            session_id = context.get("session_id", "")
            if not session_id:
                # Fallback to random if no session ID
                weights = [v.weight for v in self.variants]
                variant = random.choices(self.variants, weights=weights, k=1)[0]
            else:
                # Deterministic assignment based on session ID
                seed = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % 10000
                random.seed(seed)
                weights = [v.weight for v in self.variants]
                variant = random.choices(self.variants, weights=weights, k=1)[0]
                # Reset random seed
                random.seed()
        
        elif self.assignment_strategy == VariantAssignmentStrategy.SYMBOL_HASH:
            # Symbol-based assignment
            symbol = context.get("symbol", "")
            if not symbol:
                # Fallback to random if no symbol
                weights = [v.weight for v in self.variants]
                variant = random.choices(self.variants, weights=weights, k=1)[0]
            else:
                # Deterministic assignment based on symbol
                seed = sum(ord(c) for c in symbol) % 10000
                random.seed(seed)
                weights = [v.weight for v in self.variants]
                variant = random.choices(self.variants, weights=weights, k=1)[0]
                # Reset random seed
                random.seed()
        
        elif self.assignment_strategy == VariantAssignmentStrategy.TIME_BASED:
            # Time-based assignment
            hour = datetime.utcnow().hour
            hour_bucket = hour // 6  # 4 buckets of 6 hours each
            variant_index = hour_bucket % len(self.variants)
            variant = self.variants[variant_index]
        
        elif self.assignment_strategy == VariantAssignmentStrategy.CONTEXT_BASED:
            # Context-based assignment using a hash of all context values
            context_str = json.dumps(sorted(context.items()), sort_keys=True)
            seed = int(hashlib.md5(context_str.encode()).hexdigest(), 16) % 10000
            random.seed(seed)
            weights = [v.weight for v in self.variants]
            variant = random.choices(self.variants, weights=weights, k=1)[0]
            # Reset random seed
            random.seed()
        
        else:
            # Fallback to random
            weights = [v.weight for v in self.variants]
            variant = random.choices(self.variants, weights=weights, k=1)[0]
        
        # Store assignment
        self.assignments[context_key] = variant.id
        return variant
    
    def _get_context_key(self, context: Dict[str, Any]) -> str:
        """Generate a stable key for the context.
        
        Args:
            context: Assignment context
            
        Returns:
            Stable context key
        """
        # Priority keys for different assignment strategies
        if self.assignment_strategy == VariantAssignmentStrategy.SESSION_STICKY:
            session_id = context.get("session_id", "")
            if session_id:
                return f"session:{session_id}"
        
        elif self.assignment_strategy == VariantAssignmentStrategy.SYMBOL_HASH:
            symbol = context.get("symbol", "")
            if symbol:
                return f"symbol:{symbol}"
        
        # Default: hash the relevant context
        key_parts = []
        
        # Include symbol if available and relevant
        if "symbol" in context and TargetingCriteria.SYMBOL_SPECIFIC in self.targeting:
            key_parts.append(f"symbol:{context['symbol']}")
        
        # Include source if available and relevant
        if "source" in context and TargetingCriteria.SOURCE_SPECIFIC in self.targeting:
            key_parts.append(f"source:{context['source']}")
        
        # Include session if available
        if "session_id" in context:
            key_parts.append(f"session:{context['session_id']}")
        
        # Fallback: use generic request ID
        if not key_parts and "request_id" in context:
            key_parts.append(f"req:{context['request_id']}")
        
        return ":".join(key_parts) or str(uuid.uuid4())
    
    def _matches_targeting(self, context: Dict[str, Any]) -> bool:
        """Check if the context matches the targeting criteria.
        
        Args:
            context: Context to check
            
        Returns:
            True if targeting matches, False otherwise
        """
        # ALL_TRAFFIC always matches
        if TargetingCriteria.ALL_TRAFFIC in self.targeting:
            return True
        
        # Check each targeting criterion
        for criterion in self.targeting:
            if criterion == TargetingCriteria.SYMBOL_SPECIFIC:
                # Check if symbol matches
                symbol = context.get("symbol", "")
                target_symbols = self.metadata.get("target_symbols", [])
                if symbol and target_symbols and symbol in target_symbols:
                    return True
            
            elif criterion == TargetingCriteria.SOURCE_SPECIFIC:
                # Check if source matches
                source = context.get("source", "")
                target_sources = self.metadata.get("target_sources", [])
                if source and target_sources and source in target_sources:
                    return True
            
            elif criterion == TargetingCriteria.TIME_SPECIFIC:
                # Check if current time is within target hours
                now = datetime.utcnow()
                target_hours = self.metadata.get("target_hours", [])
                if target_hours and now.hour in target_hours:
                    return True
            
            elif criterion == TargetingCriteria.RANDOM_ASSIGNMENT:
                # Randomly include based on sample_rate
                sample_rate = self.metadata.get("sample_rate", 1.0)
                if random.random() < sample_rate:
                    return True
            
            elif criterion == TargetingCriteria.MARKET_CONDITION:
                # Check if market condition matches
                market_condition = context.get("market_condition", "")
                target_conditions = self.metadata.get("target_market_conditions", [])
                if market_condition and target_conditions and market_condition in target_conditions:
                    return True
        
        return False
    
    def record_result(self, variant_id: str, result: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Record a result for a variant.
        
        Args:
            variant_id: Variant ID
            result: Result data (including metrics)
            context: Context of the result
        """
        if variant_id not in self.variant_metrics:
            return
        
        metrics = self.variant_metrics[variant_id]
        
        # Update basic metrics
        metrics.requests += 1
        success = result.get("success", False)
        
        if success:
            metrics.successes += 1
            latency = result.get("latency_ms", 0)
            metrics.total_latency += latency
        else:
            metrics.errors += 1
        
        # Update accuracy metrics if available
        sentiment_accuracy = result.get("sentiment_accuracy")
        if sentiment_accuracy is not None:
            # Weighted update to avoid wild swings
            if metrics.sentiment_accuracy == 0:
                metrics.sentiment_accuracy = sentiment_accuracy
            else:
                metrics.sentiment_accuracy = (
                    0.9 * metrics.sentiment_accuracy + 
                    0.1 * sentiment_accuracy
                )
        
        # Update calibration error if available
        calibration_error = result.get("calibration_error")
        if calibration_error is not None:
            if metrics.calibration_error == 0:
                metrics.calibration_error = calibration_error
            else:
                metrics.calibration_error = (
                    0.9 * metrics.calibration_error + 
                    0.1 * calibration_error
                )
        
        # Update direction accuracy if available
        direction_accuracy = result.get("direction_accuracy")
        if direction_accuracy is not None:
            if metrics.direction_accuracy == 0:
                metrics.direction_accuracy = direction_accuracy
            else:
                metrics.direction_accuracy = (
                    0.9 * metrics.direction_accuracy + 
                    0.1 * direction_accuracy
                )
        
        # Update confidence score if available
        confidence_score = result.get("confidence_score")
        if confidence_score is not None:
            if metrics.confidence_score == 0:
                metrics.confidence_score = confidence_score
            else:
                metrics.confidence_score = (
                    0.9 * metrics.confidence_score + 
                    0.1 * confidence_score
                )
        
        # Track user overrides if provided
        if result.get("user_override", False):
            metrics.user_overrides += 1
        
        # Update timestamp
        self.updated_at = datetime.utcnow()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results.
        
        Returns:
            Analysis results
        """
        # Get control variant
        control_variant = None
        for variant in self.variants:
            if variant.control:
                control_variant = variant
                break
        
        if not control_variant:
            # Use the first variant as control if none specified
            control_variant = self.variants[0]
        
        # Get control metrics
        control_metrics = self.variant_metrics[control_variant.id]
        
        # Analysis results
        analysis = {
            "control_variant": control_variant.name,
            "experiment_id": self.id,
            "total_traffic": sum(m.requests for m in self.variant_metrics.values()),
            "variants_analyzed": len(self.variants),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_analyzed": [
                "success_rate", "average_latency", "sentiment_accuracy", 
                "calibration_error", "direction_accuracy", "confidence_score"
            ],
            "variant_results": {},
            "has_significant_results": False,
            "has_clear_winner": False,
            "winning_variant": None,
            "recommendation": None
        }
        
        # Analyze each variant against control
        for variant in self.variants:
            if variant.id == control_variant.id:
                continue
                
            variant_metrics = self.variant_metrics[variant.id]
            
            # Skip variants with too little data
            if variant_metrics.requests < 10 or control_metrics.requests < 10:
                analysis["variant_results"][variant.name] = {
                    "status": "insufficient_data",
                    "message": "Not enough data for statistical analysis",
                    "control_requests": control_metrics.requests,
                    "variant_requests": variant_metrics.requests,
                    "metrics": {}
                }
                continue
            
            # Analyze each metric
            metrics_analysis = {}
            
            # Success rate analysis
            control_success = control_metrics.get_success_rate()
            variant_success = variant_metrics.get_success_rate()
            
            success_p_value = self._calculate_proportion_p_value(
                control_metrics.successes, control_metrics.requests,
                variant_metrics.successes, variant_metrics.requests
            )
            
            metrics_analysis["success_rate"] = {
                "control_value": control_success,
                "variant_value": variant_success,
                "absolute_difference": variant_success - control_success,
                "percent_change": (variant_success - control_success) / max(0.0001, control_success) * 100,
                "p_value": success_p_value,
                "is_significant": success_p_value < (1 - self.min_confidence),
                "sample_size": {
                    "control": control_metrics.requests,
                    "variant": variant_metrics.requests
                }
            }
            
            # Latency analysis
            control_latency = control_metrics.get_average_latency()
            variant_latency = variant_metrics.get_average_latency()
            
            metrics_analysis["average_latency"] = {
                "control_value": control_latency,
                "variant_value": variant_latency,
                "absolute_difference": variant_latency - control_latency,
                "percent_change": (variant_latency - control_latency) / max(0.0001, control_latency) * 100,
                "is_significant": False,  # Simplified for now
                "sample_size": {
                    "control": control_metrics.successes,
                    "variant": variant_metrics.successes
                }
            }
            
            # Accuracy analysis
            metrics_analysis["sentiment_accuracy"] = {
                "control_value": control_metrics.sentiment_accuracy,
                "variant_value": variant_metrics.sentiment_accuracy,
                "absolute_difference": variant_metrics.sentiment_accuracy - control_metrics.sentiment_accuracy,
                "percent_change": (variant_metrics.sentiment_accuracy - control_metrics.sentiment_accuracy) / 
                                max(0.0001, control_metrics.sentiment_accuracy) * 100,
                "is_significant": False,  # Simplified
                "sample_size": {
                    "control": control_metrics.successes,
                    "variant": variant_metrics.successes
                }
            }
            
            # Direction accuracy analysis
            metrics_analysis["direction_accuracy"] = {
                "control_value": control_metrics.direction_accuracy,
                "variant_value": variant_metrics.direction_accuracy,
                "absolute_difference": variant_metrics.direction_accuracy - control_metrics.direction_accuracy,
                "percent_change": (variant_metrics.direction_accuracy - control_metrics.direction_accuracy) / 
                                max(0.0001, control_metrics.direction_accuracy) * 100,
                "is_significant": False,  # Simplified
                "sample_size": {
                    "control": control_metrics.successes,
                    "variant": variant_metrics.successes
                }
            }
            
            # Calibration error analysis (lower is better)
            metrics_analysis["calibration_error"] = {
                "control_value": control_metrics.calibration_error,
                "variant_value": variant_metrics.calibration_error,
                "absolute_difference": control_metrics.calibration_error - variant_metrics.calibration_error,
                "percent_change": (control_metrics.calibration_error - variant_metrics.calibration_error) / 
                                max(0.0001, control_metrics.calibration_error) * 100,
                "is_significant": False,  # Simplified
                "sample_size": {
                    "control": control_metrics.successes,
                    "variant": variant_metrics.successes
                }
            }
            
            # Confidence score analysis
            metrics_analysis["confidence_score"] = {
                "control_value": control_metrics.confidence_score,
                "variant_value": variant_metrics.confidence_score,
                "absolute_difference": variant_metrics.confidence_score - control_metrics.confidence_score,
                "percent_change": (variant_metrics.confidence_score - control_metrics.confidence_score) / 
                                max(0.0001, control_metrics.confidence_score) * 100,
                "is_significant": False,  # Simplified
                "sample_size": {
                    "control": control_metrics.successes,
                    "variant": variant_metrics.successes
                }
            }
            
            # Overall variant results
            significant_metrics = [
                name for name, analysis in metrics_analysis.items()
                if analysis.get("is_significant", False)
            ]
            
            positive_metrics = [
                name for name, analysis in metrics_analysis.items()
                if (name != "calibration_error" and analysis.get("percent_change", 0) > 0) or
                   (name == "calibration_error" and analysis.get("percent_change", 0) < 0)
            ]
            
            negative_metrics = [
                name for name, analysis in metrics_analysis.items()
                if (name != "calibration_error" and analysis.get("percent_change", 0) < 0) or
                   (name == "calibration_error" and analysis.get("percent_change", 0) > 0)
            ]
            
            analysis["variant_results"][variant.name] = {
                "status": "analyzed",
                "significant_metrics": significant_metrics,
                "positive_metrics": positive_metrics,
                "negative_metrics": negative_metrics,
                "metrics": metrics_analysis
            }
            
            # Update overall analysis
            if significant_metrics:
                analysis["has_significant_results"] = True
        
        # Determine if there's a clear winner
        if analysis["has_significant_results"]:
            variant_scores = {}
            
            for variant_name, variant_result in analysis["variant_results"].items():
                if variant_result["status"] != "analyzed":
                    continue
                
                # Calculate score based on significant improvements
                score = 0
                for metric_name in variant_result["significant_metrics"]:
                    metric_analysis = variant_result["metrics"][metric_name]
                    
                    # Different handling for calibration error (lower is better)
                    if metric_name == "calibration_error":
                        if metric_analysis["percent_change"] < 0:
                            score += abs(metric_analysis["percent_change"]) * 0.1
                    else:
                        if metric_analysis["percent_change"] > 0:
                            score += metric_analysis["percent_change"] * 0.1
                
                variant_scores[variant_name] = score
            
            if variant_scores:
                max_score = max(variant_scores.values())
                if max_score > 5:  # Threshold for a clear winner
                    winning_variants = [
                        name for name, score in variant_scores.items()
                        if score == max_score
                    ]
                    if len(winning_variants) == 1:
                        analysis["has_clear_winner"] = True
                        analysis["winning_variant"] = winning_variants[0]
                        
                        # Generate recommendation
                        analysis["recommendation"] = f"Implement variant '{winning_variants[0]}' as it shows significant improvements in {len(analysis['variant_results'][winning_variants[0]]['significant_metrics'])} metrics."
        
        # Store the analysis
        self.results = analysis
        self.updated_at = datetime.utcnow()
        
        return analysis
    
    def _calculate_proportion_p_value(
        self,
        control_successes: int,
        control_total: int,
        variant_successes: int,
        variant_total: int
    ) -> float:
        """Calculate p-value for comparing two proportions.
        
        Args:
            control_successes: Number of successes in control
            control_total: Total number in control
            variant_successes: Number of successes in variant
            variant_total: Total number in variant
            
        Returns:
            P-value for the comparison
        """
        # Calculate proportions
        p1 = control_successes / control_total
        p2 = variant_successes / variant_total
        
        # Pooled proportion
        p_pooled = (control_successes + variant_successes) / (control_total + variant_total)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/variant_total))
        
        # Z-score
        z = (p1 - p2) / se
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return p_value
    
    def needs_more_data(self) -> Tuple[bool, int]:
        """Check if the experiment needs more data for statistical significance.
        
        Returns:
            Tuple of (needs_more_data, additional_samples_needed)
        """
        # If a sample size was specified, check against that
        if self.sample_size is not None:
            total_requests = sum(m.requests for m in self.variant_metrics.values())
            if total_requests < self.sample_size:
                return True, self.sample_size - total_requests
        
        # If no significant results, estimate samples needed
        if not self.results.get("has_significant_results", False):
            # Get metrics for control variant
            control_variant = None
            for variant in self.variants:
                if variant.control:
                    control_variant = variant
                    break
            
            if not control_variant:
                control_variant = self.variants[0]
            
            control_metrics = self.variant_metrics[control_variant.id]
            
            # Calculate for each non-control variant
            max_needed = 0
            
            for variant in self.variants:
                if variant.id == control_variant.id:
                    continue
                
                variant_metrics = self.variant_metrics[variant.id]
                
                # Skip if not enough data to make an estimate
                if variant_metrics.requests < 10 or control_metrics.requests < 10:
                    max_needed = max(max_needed, 100)  # Default: need at least 100 samples
                    continue
                
                # For success rate, calculate sample size needed
                p1 = control_metrics.get_success_rate()
                p2 = variant_metrics.get_success_rate()
                
                # If almost identical, we need a lot of samples
                if abs(p1 - p2) < 0.01:
                    needed = 10000
                else:
                    # Rough estimate using the power test formula
                    effect_size = abs(p1 - p2)
                    # Simplified calculation for a ~95% confidence level
                    needed = int(16 * (p1 * (1 - p1) + p2 * (1 - p2)) / (effect_size * effect_size))
                
                max_needed = max(max_needed, needed)
            
            # Check against current total
            total_requests = sum(m.requests for m in self.variant_metrics.values())
            if total_requests < max_needed:
                return True, max_needed - total_requests
        
        return False, 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the experiment.
        
        Returns:
            Status information
        """
        # Basic status info
        status_info = {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "experiment_type": self.experiment_type.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (
                (self.end_time - self.start_time).total_seconds() 
                if self.end_time and self.start_time else None
            ),
            "total_traffic": sum(m.requests for m in self.variant_metrics.values()),
            "variant_counts": {
                variant.name: self.variant_metrics[variant.id].requests
                for variant in self.variants
            }
        }
        
        # Check if needs more data
        needs_more, samples_needed = self.needs_more_data()
        status_info["needs_more_data"] = needs_more
        status_info["additional_samples_needed"] = samples_needed
        
        # Add analysis results if available
        if self.results:
            status_info["has_results"] = True
            status_info["has_significant_results"] = self.results.get("has_significant_results", False)
            status_info["has_clear_winner"] = self.results.get("has_clear_winner", False)
            status_info["winning_variant"] = self.results.get("winning_variant")
        else:
            status_info["has_results"] = False
        
        return status_info
    
    def start(self) -> None:
        """Start the experiment."""
        if self.status not in [ExperimentStatus.DRAFT, ExperimentStatus.PAUSED]:
            return
            
        self.status = ExperimentStatus.ACTIVE
        self.start_time = datetime.utcnow()
        self.updated_at = self.start_time
    
    def pause(self) -> None:
        """Pause the experiment."""
        if self.status != ExperimentStatus.ACTIVE:
            return
            
        self.status = ExperimentStatus.PAUSED
        self.updated_at = datetime.utcnow()
    
    def complete(self) -> None:
        """Mark the experiment as completed."""
        if self.status not in [ExperimentStatus.ACTIVE, ExperimentStatus.PAUSED]:
            return
            
        self.status = ExperimentStatus.COMPLETED
        self.end_time = datetime.utcnow()
        self.updated_at = self.end_time
    
    def analyze(self) -> None:
        """Analyze the experiment results."""
        if self.status != ExperimentStatus.COMPLETED:
            self.complete()
            
        self.analyze_results()
        self.status = ExperimentStatus.ANALYZED
        self.updated_at = datetime.utcnow()
    
    def implement(self) -> None:
        """Mark the experiment as implemented."""
        if self.status != ExperimentStatus.ANALYZED:
            return
            
        self.status = ExperimentStatus.IMPLEMENTED
        self.updated_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the experiment."""
        if self.status == ExperimentStatus.DRAFT:
            return
            
        self.status = ExperimentStatus.ARCHIVED
        self.updated_at = datetime.utcnow()


class ABTestingFramework:
    """Framework for creating and managing A/B tests.
    
    This framework integrates with various components of the sentiment analysis
    system to enable experimentation and analysis of results.
    """
    
    def __init__(self):
        """Initialize the A/B testing framework."""
        self.logger = get_logger("analysis_agents", "ab_testing")
        
        # Load configuration
        self.config = config.get("sentiment_analysis.ab_testing", {})
        self.experiments_dir = self.config.get("storage_dir", "data/experiments")
        self.max_active_experiments = self.config.get("max_active_experiments", 5)
        self.results_ttl = self.config.get("results_ttl", 90)  # Days to keep results
        
        # Experiments
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiment_ids: Set[str] = set()
        
        # Context tracker for sticky assignment
        self.context_assignments: Dict[str, Dict[str, str]] = {}  # exp_id -> context_key -> variant_id
        
        # Ensure directory exists
        os.makedirs(self.experiments_dir, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the A/B testing framework."""
        self.logger.info("Initializing A/B testing framework")
        
        # Load experiments
        self._load_experiments()
        
        # Subscribe to events
        event_bus.subscribe("llm_api_request", self.handle_api_request)
        event_bus.subscribe("llm_api_response", self.handle_api_response)
        event_bus.subscribe("model_performance", self.handle_performance_update)
        
        self.logger.info(f"A/B testing framework initialized with {len(self.experiments)} experiments")
    
    async def _load_experiments(self) -> None:
        """Load experiments from storage."""
        try:
            # Scan experiments directory
            for filename in os.listdir(self.experiments_dir):
                if not filename.endswith(".json"):
                    continue
                
                file_path = os.path.join(self.experiments_dir, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        experiment_data = json.load(f)
                        experiment = Experiment.from_dict(experiment_data)
                        
                        # Add to experiments
                        self.experiments[experiment.id] = experiment
                        
                        # Track active experiments
                        if experiment.status == ExperimentStatus.ACTIVE:
                            self.active_experiment_ids.add(experiment.id)
                except Exception as e:
                    self.logger.error(f"Error loading experiment from {file_path}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading experiments: {str(e)}")
    
    async def _save_experiment(self, experiment: Experiment) -> None:
        """Save an experiment to storage.
        
        Args:
            experiment: The experiment to save
        """
        try:
            # Create filename from ID
            filename = f"{experiment.id}.json"
            file_path = os.path.join(self.experiments_dir, filename)
            
            # Convert to dict and save
            with open(file_path, 'w') as f:
                json.dump(experiment.to_dict(), f, indent=2)
                
            self.logger.debug(f"Saved experiment {experiment.id} to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving experiment {experiment.id}: {str(e)}")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        experiment_type: ExperimentType,
        variants: List[Dict[str, Any]],
        targeting: List[TargetingCriteria] = None,
        assignment_strategy: VariantAssignmentStrategy = VariantAssignmentStrategy.RANDOM,
        sample_size: Optional[int] = None,
        min_confidence: float = 0.95,
        owner: str = "system",
        metadata: Dict[str, Any] = None
    ) -> Experiment:
        """Create a new experiment.
        
        Args:
            name: Name of the experiment
            description: Description of the experiment
            experiment_type: Type of experiment
            variants: List of variant configurations
            targeting: Targeting criteria
            assignment_strategy: Assignment strategy
            sample_size: Target sample size
            min_confidence: Minimum confidence level
            owner: Experiment owner
            metadata: Additional metadata
            
        Returns:
            Created experiment
        """
        # Validate variants
        if not variants or len(variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")
        
        # Ensure there's a control variant
        has_control = any(v.get("control", False) for v in variants)
        if not has_control:
            variants[0]["control"] = True
        
        # Create experiment ID
        experiment_id = f"{experiment_type.value}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Create variant objects
        variant_objects = []
        for i, v in enumerate(variants):
            variant_id = v.get("id", f"{experiment_id}_variant_{i}")
            variant_objects.append(ExperimentVariant(
                id=variant_id,
                name=v.get("name", f"Variant {i+1}"),
                description=v.get("description", ""),
                weight=v.get("weight", 1.0),
                config=v.get("config", {}),
                metadata=v.get("metadata", {}),
                control=v.get("control", False)
            ))
        
        # Create experiment
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            experiment_type=experiment_type,
            variants=variant_objects,
            targeting=targeting,
            assignment_strategy=assignment_strategy,
            sample_size=sample_size,
            min_confidence=min_confidence,
            owner=owner,
            status=ExperimentStatus.DRAFT,
            metadata=metadata or {}
        )
        
        # Add to experiments
        self.experiments[experiment.id] = experiment
        
        # Save experiment
        asyncio.create_task(self._save_experiment(experiment))
        
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment or None if not found
        """
        return self.experiments.get(experiment_id)
    
    def list_experiments(
        self, 
        status: Optional[List[ExperimentStatus]] = None,
        experiment_type: Optional[List[ExperimentType]] = None,
        owner: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering.
        
        Args:
            status: Filter by status
            experiment_type: Filter by type
            owner: Filter by owner
            
        Returns:
            List of experiment summaries
        """
        results = []
        
        for experiment in self.experiments.values():
            # Apply filters
            if status and experiment.status not in status:
                continue
                
            if experiment_type and experiment.experiment_type not in experiment_type:
                continue
                
            if owner and experiment.owner != owner:
                continue
            
            # Create summary
            results.append({
                "id": experiment.id,
                "name": experiment.name,
                "status": experiment.status.value,
                "type": experiment.experiment_type.value,
                "variants": len(experiment.variants),
                "start_time": experiment.start_time.isoformat() if experiment.start_time else None,
                "end_time": experiment.end_time.isoformat() if experiment.end_time else None,
                "total_traffic": sum(m.requests for m in experiment.variant_metrics.values()),
                "owner": experiment.owner,
                "has_results": bool(experiment.results),
                "has_winner": experiment.results.get("has_clear_winner", False) if experiment.results else False,
                "created_at": experiment.created_at.isoformat(),
                "updated_at": experiment.updated_at.isoformat()
            })
        
        # Sort by updated_at (newest first)
        results.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return results
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if started successfully, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
        
        # Check max active experiments
        if (len(self.active_experiment_ids) >= self.max_active_experiments and
                experiment_id not in self.active_experiment_ids):
            self.logger.warning(
                f"Cannot start experiment {experiment_id}: "
                f"Maximum active experiments ({self.max_active_experiments}) reached"
            )
            return False
        
        # Start the experiment
        experiment.start()
        
        # Track active experiments
        self.active_experiment_ids.add(experiment_id)
        
        # Save changes
        asyncio.create_task(self._save_experiment(experiment))
        
        self.logger.info(f"Started experiment {experiment_id}: {experiment.name}")
        return True
    
    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if paused successfully, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
        
        # Pause the experiment
        experiment.pause()
        
        # Update active experiments
        self.active_experiment_ids.discard(experiment_id)
        
        # Save changes
        asyncio.create_task(self._save_experiment(experiment))
        
        self.logger.info(f"Paused experiment {experiment_id}: {experiment.name}")
        return True
    
    def complete_experiment(self, experiment_id: str) -> bool:
        """Complete an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if completed successfully, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
        
        # Complete the experiment
        experiment.complete()
        
        # Update active experiments
        self.active_experiment_ids.discard(experiment_id)
        
        # Analyze results
        experiment.analyze()
        
        # Save changes
        asyncio.create_task(self._save_experiment(experiment))
        
        self.logger.info(f"Completed experiment {experiment_id}: {experiment.name}")
        return True
    
    def archive_experiment(self, experiment_id: str) -> bool:
        """Archive an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if archived successfully, False otherwise
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
        
        # Archive the experiment
        experiment.archive()
        
        # Update active experiments
        self.active_experiment_ids.discard(experiment_id)
        
        # Save changes
        asyncio.create_task(self._save_experiment(experiment))
        
        self.logger.info(f"Archived experiment {experiment_id}: {experiment.name}")
        return True
    
    def get_active_experiments_by_type(self, experiment_type: ExperimentType) -> List[Experiment]:
        """Get active experiments of a specific type.
        
        Args:
            experiment_type: Type of experiment to find
            
        Returns:
            List of active experiments of the specified type
        """
        return [
            exp for exp in self.experiments.values()
            if exp.status == ExperimentStatus.ACTIVE and 
               exp.experiment_type == experiment_type and
               exp.id in self.active_experiment_ids
        ]
    
    def get_experiment_variant(
        self, 
        experiment_type: ExperimentType,
        context: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Get a variant for an active experiment of the specified type.
        
        Args:
            experiment_type: Type of experiment
            context: Context for variant assignment
            
        Returns:
            Tuple of (experiment_id, variant_config) or (None, None) if no experiment
        """
        # Get active experiments of this type
        active_experiments = self.get_active_experiments_by_type(experiment_type)
        
        if not active_experiments:
            return None, None
        
        # Select one experiment (for now, just use the first one)
        experiment = active_experiments[0]
        
        # Assign a variant
        variant = experiment.assign_variant(context)
        
        return experiment.id, variant.config
    
    async def handle_api_request(self, event: Event) -> None:
        """Handle an API request event.
        
        Args:
            event: API request event
        """
        data = event.data
        request_id = data.get("request_id")
        experiment_id = data.get("experiment_id")
        variant_id = data.get("variant_id")
        
        # Skip if not part of an experiment
        if not all([request_id, experiment_id, variant_id]):
            return
        
        # Get the experiment
        experiment = self.get_experiment(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.ACTIVE:
            return
        
        # Extract context for results tracking
        context = {
            "request_id": request_id,
            "model": data.get("model", ""),
            "provider": data.get("provider", ""),
            "operation": data.get("operation", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in request context for matching with response
        event_bus.set_request_context(request_id, {
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "context": context
        })
    
    async def handle_api_response(self, event: Event) -> None:
        """Handle an API response event.
        
        Args:
            event: API response event
        """
        data = event.data
        request_id = data.get("request_id")
        
        # Get experiment context
        context = event_bus.get_request_context(request_id)
        if not context:
            return
            
        experiment_id = context.get("experiment_id")
        variant_id = context.get("variant_id")
        request_context = context.get("context", {})
        
        # Get the experiment
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return
        
        # Create result data
        result = {
            "success": data.get("success", False),
            "latency_ms": data.get("latency_ms", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Record the result
        experiment.record_result(variant_id, result, request_context)
        
        # Save experiment periodically (don't save on every request)
        if random.random() < 0.05:  # ~5% of responses trigger a save
            asyncio.create_task(self._save_experiment(experiment))
    
    async def handle_performance_update(self, event: Event) -> None:
        """Handle a performance update event.
        
        Args:
            event: Performance update event
        """
        data = event.data
        experiment_id = data.get("experiment_id")
        variant_id = data.get("variant_id")
        
        # Skip if not part of an experiment
        if not experiment_id or not variant_id:
            return
        
        # Get the experiment
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return
        
        # Extract metrics
        metrics = {
            "sentiment_accuracy": data.get("accuracy"),
            "calibration_error": data.get("calibration_error"),
            "direction_accuracy": data.get("direction_accuracy"),
            "confidence_score": data.get("confidence")
        }
        
        # Record the result
        experiment.record_result(variant_id, metrics, {})
        
        # Save experiment
        asyncio.create_task(self._save_experiment(experiment))
    
    def create_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """Create a comprehensive report for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Report data
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        # Ensure results are analyzed
        if not experiment.results:
            experiment.analyze_results()
        
        # Create report data
        report = {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "type": experiment.experiment_type.value,
            "status": experiment.status.value,
            "created_at": experiment.created_at.isoformat(),
            "start_time": experiment.start_time.isoformat() if experiment.start_time else None,
            "end_time": experiment.end_time.isoformat() if experiment.end_time else None,
            "duration_hours": (
                (experiment.end_time - experiment.start_time).total_seconds() / 3600
                if experiment.end_time and experiment.start_time else None
            ),
            "owner": experiment.owner,
            "variants": [
                {
                    "id": v.id,
                    "name": v.name,
                    "description": v.description,
                    "is_control": v.control,
                    "traffic_weight": v.weight,
                    "metrics": experiment.variant_metrics[v.id].to_dict()
                }
                for v in experiment.variants
            ],
            "total_traffic": sum(m.requests for m in experiment.variant_metrics.values()),
            "results": experiment.results,
            "needs_more_data": experiment.needs_more_data()[0],
            "recommendation": experiment.results.get("recommendation", "No recommendation available")
        }
        
        # Check if experiment is conclusive
        report["is_conclusive"] = (
            report["results"].get("has_significant_results", False) and
            not report["needs_more_data"]
        )
        
        return report
    
    def generate_visualization_data(
        self,
        experiment_id: str,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Generate data for visualizing experiment results.
        
        Args:
            experiment_id: Experiment ID
            metrics: List of metrics to include (default: all)
            
        Returns:
            Visualization data
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        # Default metrics
        if not metrics:
            metrics = [
                "success_rate", "average_latency", "sentiment_accuracy",
                "calibration_error", "direction_accuracy", "confidence_score"
            ]
        
        # Extract data for each variant
        variants_data = []
        for variant in experiment.variants:
            metrics_data = experiment.variant_metrics[variant.id].to_dict()
            
            variant_data = {
                "name": variant.name,
                "is_control": variant.control,
                "traffic": metrics_data["requests"],
                "metrics": {
                    metric: metrics_data.get(metric, 0)
                    for metric in metrics
                }
            }
            
            variants_data.append(variant_data)
        
        # Analysis results
        analysis = {}
        if experiment.results:
            for variant_name, result in experiment.results.get("variant_results", {}).items():
                if result["status"] != "analyzed":
                    continue
                    
                analysis[variant_name] = {
                    "significant_metrics": result.get("significant_metrics", []),
                    "metric_details": {
                        metric: {
                            "percent_change": details.get("percent_change", 0),
                            "is_significant": details.get("is_significant", False)
                        }
                        for metric, details in result.get("metrics", {}).items()
                        if metric in metrics
                    }
                }
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "metrics": metrics,
            "variants": variants_data,
            "analysis": analysis,
            "is_conclusive": experiment.results.get("has_significant_results", False) and not experiment.needs_more_data()[0],
            "winning_variant": experiment.results.get("winning_variant")
        }
    
    def check_experiments_completion(self) -> List[str]:
        """Check if any experiments have reached completion criteria.
        
        Returns:
            List of experiment IDs that should be completed
        """
        completed_ids = []
        
        for experiment_id in list(self.active_experiment_ids):
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                self.active_experiment_ids.discard(experiment_id)
                continue
            
            # Skip if not active
            if experiment.status != ExperimentStatus.ACTIVE:
                self.active_experiment_ids.discard(experiment_id)
                continue
            
            # Check if end date reached
            if experiment.end_time and datetime.utcnow() >= experiment.end_time:
                completed_ids.append(experiment_id)
                continue
            
            # Check if sample size reached
            if experiment.sample_size:
                total_requests = sum(m.requests for m in experiment.variant_metrics.values())
                if total_requests >= experiment.sample_size:
                    completed_ids.append(experiment_id)
                    continue
            
            # Check if we have significant results
            if experiment.results.get("has_clear_winner", False):
                # Only complete if we've got enough data
                needs_more, _ = experiment.needs_more_data()
                if not needs_more:
                    completed_ids.append(experiment_id)
                    continue
        
        return completed_ids
    
    async def run_maintenance(self) -> None:
        """Run maintenance tasks for the A/B testing framework."""
        self.logger.info("Running A/B testing framework maintenance")
        
        # Check for experiments that should be completed
        completed_ids = self.check_experiments_completion()
        for experiment_id in completed_ids:
            self.logger.info(f"Automatically completing experiment {experiment_id}")
            self.complete_experiment(experiment_id)
        
        # Archive old experiments
        cutoff_date = datetime.utcnow() - timedelta(days=self.results_ttl)
        for experiment in list(self.experiments.values()):
            if (experiment.status == ExperimentStatus.IMPLEMENTED and
                    experiment.updated_at < cutoff_date):
                self.logger.info(f"Archiving old experiment {experiment.id}")
                self.archive_experiment(experiment.id)


# Singleton instance
ab_testing_framework = ABTestingFramework()