"""Multi-Variant Testing Framework for sentiment analysis.

This module extends the A/B testing framework to support advanced multi-variant
experiments for testing multiple treatments simultaneously.
"""

import asyncio
import json
import logging
import random
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import f_oneway, tukey_hsd

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event
from src.analysis_agents.sentiment.ab_testing import (
    ExperimentType, ExperimentStatus, TargetingCriteria, 
    VariantAssignmentStrategy, ExperimentVariant, ExperimentMetrics, Experiment
)


class MultiVariantExperiment(Experiment):
    """Extended experiment class supporting multiple treatment variants."""
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results with multi-variant support.
        
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
        
        # Try to use Rust-optimized implementation first
        try:
            from src.rust_bridge import analyze_multi_variant_results
            
            # Prepare experiment data for Rust analysis
            experiment_data = {
                "id": self.id,
                "name": self.name,
                "experiment_type": self.experiment_type.value,
                "variants": []
            }
            
            # Convert variants to format expected by Rust
            for variant in self.variants:
                metrics = self.variant_metrics[variant.id].to_dict()
                variant_data = {
                    "id": variant.id,
                    "name": variant.name,
                    "control": variant.control,
                    "metrics": metrics
                }
                experiment_data["variants"].append(variant_data)
            
            # Call Rust-optimized function for multi-variant analysis
            rust_results = analyze_multi_variant_results(
                experiment_data, 
                self.min_confidence, 
                0.05  # Default improvement threshold
            )
            
            if rust_results:
                # Update timestamps
                rust_results["timestamp"] = datetime.utcnow().isoformat()
                
                # Add experiment metadata
                rust_results["experiment_id"] = self.id
                rust_results["control_variant"] = control_variant.name
                rust_results["total_traffic"] = sum(m.requests for m in self.variant_metrics.values())
                rust_results["variants_analyzed"] = len(self.variants)
                
                # Store the analysis and update timestamp
                self.results = rust_results
                self.updated_at = datetime.utcnow()
                
                return rust_results
            
        except Exception as e:
            # Log error and fall back to Python implementation
            logger = get_logger("analysis_agents", "multi_variant_testing")
            logger.warning(f"Error using Rust-optimized multi-variant analysis: {e}")
            logger.info("Falling back to Python implementation for multi-variant analysis")
        
        # Fallback to Python implementation if Rust optimization fails
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
        
        # Run ANOVA for multi-variant comparison
        anova_results = self._run_anova_analysis()
        analysis["anova_results"] = anova_results
        
        # Perform pairwise comparisons
        treatment_variants = [v for v in self.variants if not v.control]
        
        # Skip analysis if not enough data
        min_requests = min(self.variant_metrics[v.id].requests for v in self.variants)
        if min_requests < 10:
            analysis["status"] = "insufficient_data"
            analysis["message"] = f"Not enough data for statistical analysis (min: {min_requests} requests)"
            return analysis
        
        # For each treatment variant, compare with control
        for variant in treatment_variants:
            variant_metrics = self.variant_metrics[variant.id]
            
            # Analyze each metric
            metrics_analysis = {}
            
            for metric in analysis["metrics_analyzed"]:
                # Get values for control and variant
                control_value = self._get_metric_value(control_metrics, metric)
                variant_value = self._get_metric_value(variant_metrics, metric)
                
                # Calculate differences
                absolute_diff = variant_value - control_value
                percent_change = (variant_value - control_value) / max(0.0001, control_value) * 100
                
                # Get p-value from t-test
                p_value = self._calculate_pairwise_p_value(control_metrics, variant_metrics, metric)
                
                # Store results
                metrics_analysis[metric] = {
                    "control_value": control_value,
                    "variant_value": variant_value,
                    "absolute_difference": absolute_diff,
                    "percent_change": percent_change,
                    "p_value": p_value,
                    "is_significant": p_value < (1 - self.min_confidence),
                    "sample_size": {
                        "control": control_metrics.requests,
                        "variant": variant_metrics.requests
                    }
                }
            
            # Track significant metrics
            significant_metrics = [
                name for name, analysis in metrics_analysis.items()
                if analysis.get("is_significant", False)
            ]
            
            # Track metrics with positive/negative change
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
        
        # If we found significant results, perform Tukey HSD 
        # for pairwise comparisons to find the best variant
        if analysis["has_significant_results"]:
            tukey_results = self._run_tukey_analysis()
            analysis["tukey_results"] = tukey_results
            
            # Calculate overall variant scores
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
    
    def _get_metric_value(self, metrics: ExperimentMetrics, metric_name: str) -> float:
        """Get a specific metric value from experiment metrics.
        
        Args:
            metrics: The metrics object
            metric_name: Name of the metric to get
            
        Returns:
            Metric value
        """
        if metric_name == "success_rate":
            return metrics.get_success_rate()
        elif metric_name == "average_latency":
            return metrics.get_average_latency()
        elif metric_name == "sentiment_accuracy":
            return metrics.sentiment_accuracy
        elif metric_name == "calibration_error":
            return metrics.calibration_error
        elif metric_name == "direction_accuracy":
            return metrics.direction_accuracy
        elif metric_name == "confidence_score":
            return metrics.confidence_score
        else:
            return 0.0
    
    def _calculate_pairwise_p_value(
        self, 
        control_metrics: ExperimentMetrics, 
        variant_metrics: ExperimentMetrics,
        metric_name: str
    ) -> float:
        """Calculate p-value for comparing a variant with control.
        
        Args:
            control_metrics: Control variant metrics
            variant_metrics: Treatment variant metrics
            metric_name: Name of the metric to compare
            
        Returns:
            P-value for the comparison
        """
        control_value = self._get_metric_value(control_metrics, metric_name)
        variant_value = self._get_metric_value(variant_metrics, metric_name)
        
        # For proportions (e.g., success rate)
        if metric_name == "success_rate":
            return self._calculate_proportion_p_value(
                control_metrics.successes, control_metrics.requests,
                variant_metrics.successes, variant_metrics.requests
            )
        
        # For continuous metrics, use a simplified t-test
        # Note: In a real-world scenario, we'd have the actual samples for a proper t-test
        control_sample_size = control_metrics.requests
        variant_sample_size = variant_metrics.requests
        
        # Simplified std error estimate
        std_error = 0.1
        
        # Calculate t-statistic
        diff = variant_value - control_value
        t_statistic = diff / (std_error * np.sqrt(1/control_sample_size + 1/variant_sample_size))
        
        # Degrees of freedom
        df = control_sample_size + variant_sample_size - 2
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
        
        return p_value
    
    def _run_anova_analysis(self) -> Dict[str, Any]:
        """Run ANOVA to determine if there are significant differences between variants.
        
        Returns:
            ANOVA results
        """
        anova_results = {}
        
        # Extract data for each metric
        for metric_name in ["sentiment_accuracy", "direction_accuracy", "calibration_error", "confidence_score"]:
            # Get values for each variant
            values_by_variant = []
            variant_names = []
            
            for variant in self.variants:
                metrics = self.variant_metrics[variant.id]
                value = self._get_metric_value(metrics, metric_name)
                
                # Since we don't have the raw samples, we simulate them
                # This is a simplification for demonstration purposes
                # In a real-world implementation, we would use the actual samples
                sample_size = metrics.requests
                if sample_size >= 10:
                    # Create a simulated sample with mean=value and std=0.1
                    simulated_samples = np.random.normal(value, 0.1, size=sample_size)
                    values_by_variant.append(simulated_samples)
                    variant_names.append(variant.name)
            
            # Skip if not enough data
            if len(values_by_variant) < 2:
                anova_results[metric_name] = {
                    "status": "insufficient_data",
                    "message": "Not enough data for ANOVA"
                }
                continue
            
            # Run one-way ANOVA
            try:
                f_stat, p_value = f_oneway(*values_by_variant)
                
                # Store results
                anova_results[metric_name] = {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "is_significant": p_value < (1 - self.min_confidence),
                    "variants": variant_names,
                    "sample_sizes": [len(values) for values in values_by_variant]
                }
            except Exception as e:
                logger = get_logger("analysis_agents", "multi_variant_testing")
                logger.error(f"Error running ANOVA for {metric_name}: {e}")
                anova_results[metric_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        return anova_results
    
    def _run_tukey_analysis(self) -> Dict[str, Any]:
        """Run Tukey's HSD test for all pairwise comparisons.
        
        Returns:
            Tukey HSD results
        """
        tukey_results = {}
        
        # Extract data for each metric
        for metric_name in ["sentiment_accuracy", "direction_accuracy", "calibration_error", "confidence_score"]:
            # Get values for each variant
            values_by_variant = []
            variant_names = []
            
            for variant in self.variants:
                metrics = self.variant_metrics[variant.id]
                value = self._get_metric_value(metrics, metric_name)
                
                # Simulate samples as in ANOVA
                sample_size = metrics.requests
                if sample_size >= 10:
                    # Create a simulated sample with mean=value and std=0.1
                    simulated_samples = np.random.normal(value, 0.1, size=sample_size)
                    values_by_variant.append(simulated_samples)
                    variant_names.append(variant.name)
            
            # Skip if not enough data
            if len(values_by_variant) < 2:
                tukey_results[metric_name] = {
                    "status": "insufficient_data",
                    "message": "Not enough data for Tukey HSD"
                }
                continue
            
            # Run Tukey HSD
            try:
                # Create a single array of all values
                all_values = np.concatenate(values_by_variant)
                
                # Create a group identifier for each value
                groups = np.concatenate([np.full(len(values), i) for i, values in enumerate(values_by_variant)])
                
                # Run Tukey HSD
                result = tukey_hsd(all_values, groups, alpha=1-self.min_confidence)
                
                # Convert results to a more usable format
                pairwise_results = {}
                for i in range(len(variant_names)):
                    for j in range(i+1, len(variant_names)):
                        pair_key = f"{variant_names[i]} vs {variant_names[j]}"
                        pair_index = (i, j)
                        
                        # Get the Tukey result for this pair
                        mean_diff = result.meandiffs[pair_index]
                        p_value = result.pvalues[pair_index]
                        confidence_interval = (
                            result.confint[pair_index][0],
                            result.confint[pair_index][1]
                        )
                        
                        pairwise_results[pair_key] = {
                            "mean_difference": float(mean_diff),
                            "p_value": float(p_value),
                            "confidence_interval": [float(ci) for ci in confidence_interval],
                            "is_significant": p_value < (1 - self.min_confidence),
                            "better_variant": variant_names[i] if (metric_name != "calibration_error" and mean_diff > 0) or 
                                              (metric_name == "calibration_error" and mean_diff < 0) else variant_names[j]
                        }
                
                # Store results
                tukey_results[metric_name] = {
                    "pairwise_comparisons": pairwise_results,
                    "variants": variant_names,
                    "sample_sizes": [len(values) for values in values_by_variant]
                }
            except Exception as e:
                logger = get_logger("analysis_agents", "multi_variant_testing")
                logger.error(f"Error running Tukey HSD for {metric_name}: {e}")
                tukey_results[metric_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        return tukey_results


class MultiVariantExperimentFactory:
    """Factory for creating multi-variant experiments."""
    
    @staticmethod
    def create_experiment(
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
    ) -> MultiVariantExperiment:
        """Create a new multi-variant experiment.
        
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
        experiment_id = f"{experiment_type.value}_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
        
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
        experiment = MultiVariantExperiment(
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
        
        return experiment


# Create factory instance
multi_variant_experiment_factory = MultiVariantExperimentFactory()