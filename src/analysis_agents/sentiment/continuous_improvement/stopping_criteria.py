"""Automatic Stopping Criteria for Continuous Improvement Experiments.

This module provides utilities for automatically determining when experiments
have collected sufficient data to make reliable decisions, enabling more
efficient resource allocation and faster iterations.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta

from src.common.logging import get_logger
from src.common.events import event_bus, Event
from src.analysis_agents.sentiment.ab_testing import (
    ExperimentStatus, Experiment, ExperimentVariant, ExperimentMetrics
)
from src.analysis_agents.sentiment.continuous_improvement.bayesian_analysis import (
    BayesianAnalyzer, BayesianAnalysisResults
)

# Initialize logger
logger = get_logger("analysis_agents", "stopping_criteria")


class StoppingCriterion:
    """Base class for experiment stopping criteria."""
    
    def __init__(self, name: str, description: str):
        """Initialize stopping criterion.
        
        Args:
            name: Name of the criterion
            description: Description of the criterion
        """
        self.name = name
        self.description = description
    
    def should_stop(self, experiment: Experiment) -> Tuple[bool, str]:
        """Check if the experiment should be stopped based on this criterion.
        
        Args:
            experiment: The experiment to check
            
        Returns:
            Tuple of (should_stop, reason)
        """
        raise NotImplementedError("Subclasses must implement should_stop")


class SampleSizeCriterion(StoppingCriterion):
    """Stopping criterion based on reaching a target sample size."""
    
    def __init__(self, min_samples_per_variant: int = 100):
        """Initialize sample size criterion.
        
        Args:
            min_samples_per_variant: Minimum samples per variant
        """
        super().__init__(
            name="sample_size",
            description=f"Stop when each variant has at least {min_samples_per_variant} samples"
        )
        self.min_samples_per_variant = min_samples_per_variant
    
    def should_stop(self, experiment: Experiment) -> Tuple[bool, str]:
        """Check if all variants have reached the minimum sample size.
        
        Args:
            experiment: The experiment to check
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if experiment.status != ExperimentStatus.ACTIVE:
            return False, "Experiment is not active"
        
        # Check sample size for each variant
        min_samples = float('inf')
        for variant_id, metrics in experiment.variant_metrics.items():
            min_samples = min(min_samples, metrics.requests)
        
        if min_samples >= self.min_samples_per_variant:
            return True, f"All variants have at least {self.min_samples_per_variant} samples"
        
        return False, f"Need more samples (min: {min_samples}, target: {self.min_samples_per_variant})"


class BayesianProbabilityThresholdCriterion(StoppingCriterion):
    """Stopping criterion based on Bayesian probability of being best."""
    
    def __init__(
        self,
        probability_threshold: float = 0.95,
        min_samples_per_variant: int = 50,
        metrics_weight: Dict[str, float] = None
    ):
        """Initialize Bayesian probability threshold criterion.
        
        Args:
            probability_threshold: Minimum probability to declare a winner
            min_samples_per_variant: Minimum samples before applying criterion
            metrics_weight: Weights for different metrics in the decision
        """
        super().__init__(
            name="bayesian_probability",
            description=f"Stop when a variant has at least {probability_threshold:.0%} probability of being best"
        )
        self.probability_threshold = probability_threshold
        self.min_samples_per_variant = min_samples_per_variant
        self.metrics_weight = metrics_weight or {
            "sentiment_accuracy": 0.4,
            "direction_accuracy": 0.3,
            "calibration_error": 0.2,
            "confidence_score": 0.1
        }
        # Normalize weights
        total_weight = sum(self.metrics_weight.values())
        self.metrics_weight = {k: v / total_weight for k, v in self.metrics_weight.items()}
        
        self.bayesian_analyzer = BayesianAnalyzer()
    
    def should_stop(self, experiment: Experiment) -> Tuple[bool, str]:
        """Check if any variant has reached the probability threshold.
        
        Args:
            experiment: The experiment to check
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if experiment.status != ExperimentStatus.ACTIVE:
            return False, "Experiment is not active"
        
        # Check minimum sample size
        for variant_id, metrics in experiment.variant_metrics.items():
            if metrics.requests < self.min_samples_per_variant:
                return False, f"Need at least {self.min_samples_per_variant} samples per variant"
        
        # Run Bayesian analysis
        try:
            analysis_results = self.bayesian_analyzer.analyze_experiment(experiment)
            
            # Calculate weighted average probability for each variant
            weighted_probabilities = {}
            
            for metric, weights in analysis_results.winning_probability.items():
                metric_weight = self.metrics_weight.get(metric, 0.0)
                for variant, prob in weights.items():
                    if variant not in weighted_probabilities:
                        weighted_probabilities[variant] = 0.0
                    weighted_probabilities[variant] += prob * metric_weight
            
            # Find the maximum probability
            if weighted_probabilities:
                max_variant = max(weighted_probabilities.items(), key=lambda x: x[1])
                max_prob = max_variant[1]
                
                if max_prob >= self.probability_threshold:
                    return True, f"Variant '{max_variant[0]}' has {max_prob:.1%} probability of being best (threshold: {self.probability_threshold:.1%})"
            
            return False, "No variant has reached the probability threshold"
            
        except Exception as e:
            logger.error(f"Error running Bayesian analysis: {e}")
            return False, f"Error in Bayesian analysis: {str(e)}"


class ExpectedLossCriterion(StoppingCriterion):
    """Stopping criterion based on expected loss (regret) being below a threshold."""
    
    def __init__(
        self,
        loss_threshold: float = 0.005,
        min_samples_per_variant: int = 50,
        metrics_weight: Dict[str, float] = None
    ):
        """Initialize expected loss criterion.
        
        Args:
            loss_threshold: Maximum acceptable expected loss
            min_samples_per_variant: Minimum samples before applying criterion
            metrics_weight: Weights for different metrics in the decision
        """
        super().__init__(
            name="expected_loss",
            description=f"Stop when expected loss (regret) is below {loss_threshold:.3f}"
        )
        self.loss_threshold = loss_threshold
        self.min_samples_per_variant = min_samples_per_variant
        self.metrics_weight = metrics_weight or {
            "sentiment_accuracy": 0.4,
            "direction_accuracy": 0.3,
            "calibration_error": 0.2,
            "confidence_score": 0.1
        }
        # Normalize weights
        total_weight = sum(self.metrics_weight.values())
        self.metrics_weight = {k: v / total_weight for k, v in self.metrics_weight.items()}
        
        self.bayesian_analyzer = BayesianAnalyzer()
    
    def should_stop(self, experiment: Experiment) -> Tuple[bool, str]:
        """Check if expected loss is below the threshold.
        
        Args:
            experiment: The experiment to check
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if experiment.status != ExperimentStatus.ACTIVE:
            return False, "Experiment is not active"
        
        # Check minimum sample size
        for variant_id, metrics in experiment.variant_metrics.items():
            if metrics.requests < self.min_samples_per_variant:
                return False, f"Need at least {self.min_samples_per_variant} samples per variant"
        
        # Run Bayesian analysis
        try:
            analysis_results = self.bayesian_analyzer.analyze_experiment(experiment)
            
            # Find the variant with the minimum expected loss for each metric
            min_loss_variants = {}
            for metric, losses in analysis_results.expected_loss.items():
                metric_weight = self.metrics_weight.get(metric, 0.0)
                min_variant = min(losses.items(), key=lambda x: x[1])
                min_loss_variants[metric] = {
                    "variant": min_variant[0],
                    "loss": min_variant[1],
                    "weighted_loss": min_variant[1] * metric_weight
                }
            
            # Calculate total weighted loss for the best variant
            if min_loss_variants:
                total_weighted_loss = sum(info["weighted_loss"] for info in min_loss_variants.values())
                
                if total_weighted_loss <= self.loss_threshold:
                    best_variants = set(info["variant"] for info in min_loss_variants.values())
                    if len(best_variants) == 1:
                        best_variant = list(best_variants)[0]
                        return True, f"Variant '{best_variant}' has expected loss {total_weighted_loss:.4f} below threshold {self.loss_threshold:.4f}"
                    else:
                        return True, f"Multiple variants have low expected loss {total_weighted_loss:.4f} below threshold {self.loss_threshold:.4f}"
            
            return False, "Expected loss is above the threshold"
            
        except Exception as e:
            logger.error(f"Error running Bayesian analysis: {e}")
            return False, f"Error in Bayesian analysis: {str(e)}"


class ConfidenceIntervalCriterion(StoppingCriterion):
    """Stopping criterion based on confidence interval width."""
    
    def __init__(
        self,
        interval_width_threshold: float = 0.05,
        min_samples_per_variant: int = 50,
        metrics_to_check: List[str] = None
    ):
        """Initialize confidence interval criterion.
        
        Args:
            interval_width_threshold: Maximum acceptable interval width
            min_samples_per_variant: Minimum samples before applying criterion
            metrics_to_check: Which metrics to check intervals for
        """
        super().__init__(
            name="confidence_interval",
            description=f"Stop when confidence intervals are narrower than {interval_width_threshold:.1%}"
        )
        self.interval_width_threshold = interval_width_threshold
        self.min_samples_per_variant = min_samples_per_variant
        self.metrics_to_check = metrics_to_check or [
            "sentiment_accuracy", "direction_accuracy"
        ]
        
        self.bayesian_analyzer = BayesianAnalyzer()
    
    def should_stop(self, experiment: Experiment) -> Tuple[bool, str]:
        """Check if confidence intervals are narrow enough.
        
        Args:
            experiment: The experiment to check
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if experiment.status != ExperimentStatus.ACTIVE:
            return False, "Experiment is not active"
        
        # Check minimum sample size
        for variant_id, metrics in experiment.variant_metrics.items():
            if metrics.requests < self.min_samples_per_variant:
                return False, f"Need at least {self.min_samples_per_variant} samples per variant"
        
        # Run Bayesian analysis
        try:
            analysis_results = self.bayesian_analyzer.analyze_experiment(experiment)
            
            # Check interval width for each metric and variant
            narrow_enough = True
            widest_interval = 0.0
            
            for metric in self.metrics_to_check:
                if metric not in analysis_results.credible_intervals:
                    continue
                    
                for variant, intervals in analysis_results.credible_intervals[metric].items():
                    ci_95 = intervals.get("95%", [0, 1])
                    interval_width = ci_95[1] - ci_95[0]
                    widest_interval = max(widest_interval, interval_width)
                    
                    if interval_width > self.interval_width_threshold:
                        narrow_enough = False
            
            if narrow_enough:
                return True, f"All confidence intervals are narrower than {self.interval_width_threshold:.1%}"
            else:
                return False, f"Confidence intervals still too wide (widest: {widest_interval:.1%}, threshold: {self.interval_width_threshold:.1%})"
            
        except Exception as e:
            logger.error(f"Error running Bayesian analysis: {e}")
            return False, f"Error in Bayesian analysis: {str(e)}"


class TimeLimitCriterion(StoppingCriterion):
    """Stopping criterion based on experiment duration."""
    
    def __init__(self, max_days: int = 14):
        """Initialize time limit criterion.
        
        Args:
            max_days: Maximum experiment duration in days
        """
        super().__init__(
            name="time_limit",
            description=f"Stop when the experiment has been running for {max_days} days"
        )
        self.max_days = max_days
    
    def should_stop(self, experiment: Experiment) -> Tuple[bool, str]:
        """Check if the experiment has reached its time limit.
        
        Args:
            experiment: The experiment to check
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if experiment.status != ExperimentStatus.ACTIVE:
            return False, "Experiment is not active"
        
        if not experiment.start_time:
            return False, "Experiment has no start time"
        
        # Calculate experiment duration
        now = datetime.utcnow()
        duration = now - experiment.start_time
        duration_days = duration.total_seconds() / (60 * 60 * 24)
        
        if duration_days >= self.max_days:
            return True, f"Experiment has been running for {duration_days:.1f} days (limit: {self.max_days} days)"
        
        return False, f"Experiment has been running for {duration_days:.1f} days (limit: {self.max_days} days)"


class StoppingCriteriaManager:
    """Manager for experiment stopping criteria."""
    
    def __init__(self):
        """Initialize stopping criteria manager."""
        self.logger = get_logger("analysis_agents", "stopping_criteria")
        self.criteria = []
        
        # Add default criteria
        self.add_criterion(SampleSizeCriterion(min_samples_per_variant=100))
        self.add_criterion(BayesianProbabilityThresholdCriterion(probability_threshold=0.95))
        self.add_criterion(ExpectedLossCriterion(loss_threshold=0.005))
        self.add_criterion(ConfidenceIntervalCriterion(interval_width_threshold=0.05))
        self.add_criterion(TimeLimitCriterion(max_days=14))
    
    def add_criterion(self, criterion: StoppingCriterion) -> None:
        """Add a stopping criterion.
        
        Args:
            criterion: The criterion to add
        """
        self.criteria.append(criterion)
    
    def remove_criterion(self, criterion_name: str) -> bool:
        """Remove a stopping criterion by name.
        
        Args:
            criterion_name: Name of the criterion to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, criterion in enumerate(self.criteria):
            if criterion.name == criterion_name:
                del self.criteria[i]
                return True
        return False
    
    def clear_criteria(self) -> None:
        """Remove all stopping criteria."""
        self.criteria = []
    
    def evaluate_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Evaluate all stopping criteria for an experiment.
        
        Args:
            experiment: The experiment to evaluate
            
        Returns:
            Evaluation results
        """
        results = {
            "experiment_id": experiment.id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "should_stop": False,
            "stopping_reasons": [],
            "criteria_results": {}
        }
        
        # Skip if not active
        if experiment.status != ExperimentStatus.ACTIVE:
            return results
        
        # Evaluate each criterion
        for criterion in self.criteria:
            try:
                should_stop, reason = criterion.should_stop(experiment)
                
                results["criteria_results"][criterion.name] = {
                    "should_stop": should_stop,
                    "reason": reason
                }
                
                if should_stop:
                    results["should_stop"] = True
                    results["stopping_reasons"].append({
                        "criterion": criterion.name,
                        "reason": reason
                    })
            except Exception as e:
                self.logger.error(f"Error evaluating criterion {criterion.name}: {e}")
                results["criteria_results"][criterion.name] = {
                    "should_stop": False,
                    "reason": f"Error: {str(e)}"
                }
        
        return results


# Create singleton instance
stopping_criteria_manager = StoppingCriteriaManager()