"""
Python wrapper for the Rust implementation of continuous improvement algorithms.
This module provides high-performance implementations of the most computationally 
intensive parts of the continuous improvement system, including support for
multi-variant experiments with advanced statistical analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

# Try to import the Rust module
try:
    from crypto_trading_engine.sentiment.continuous_improvement import (
        analyze_experiment_results as _analyze_experiment_results,
        identify_improvement_opportunities as _identify_improvement_opportunities,
        analyze_multi_variant_results as _analyze_multi_variant_results
    )
    RUST_AVAILABLE = True
except ImportError:
    logging.warning("Rust sentiment module not available, falling back to Python implementation")
    RUST_AVAILABLE = False

class ContinuousImprovementRust:
    """
    Interface to the Rust-based continuous improvement optimization functions.
    Provides fallback to Python implementations when the Rust module is not available.
    """
    
    @staticmethod
    def analyze_experiment_results(
        experiment_data: Dict[str, Any], 
        significance_threshold: float = 0.9, 
        improvement_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Analyze experiment results using the Rust-optimized implementation
        
        Args:
            experiment_data: Dictionary containing experiment data including variants and metrics
            significance_threshold: Statistical significance threshold (0.0-1.0)
            improvement_threshold: Minimum improvement to consider significant
            
        Returns:
            Dictionary containing analysis results
        """
        if not RUST_AVAILABLE:
            # Fall back to Python implementation
            return ContinuousImprovementRust._analyze_experiment_results_py(
                experiment_data, significance_threshold, improvement_threshold
            )
            
        # Call the Rust implementation
        return _analyze_experiment_results(
            experiment_data, significance_threshold, improvement_threshold
        )
    
    @staticmethod
    def analyze_multi_variant_results(
        experiment_data: Dict[str, Any], 
        significance_threshold: float = 0.9, 
        improvement_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Analyze multi-variant experiment results using the Rust-optimized implementation
        with advanced statistical methods including ANOVA and Tukey HSD
        
        Args:
            experiment_data: Dictionary containing experiment data including variants and metrics
            significance_threshold: Statistical significance threshold (0.0-1.0)
            improvement_threshold: Minimum improvement to consider significant
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        if not RUST_AVAILABLE:
            # Fall back to Python implementation
            return ContinuousImprovementRust._analyze_multi_variant_results_py(
                experiment_data, significance_threshold, improvement_threshold
            )
            
        # Call the Rust implementation
        return _analyze_multi_variant_results(
            experiment_data, significance_threshold, improvement_threshold
        )
    
    @staticmethod
    def identify_improvement_opportunities(metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify improvement opportunities using the Rust-optimized implementation
        
        Args:
            metrics_data: Dictionary containing performance metrics
            
        Returns:
            List of dictionaries containing identified opportunities
        """
        if not RUST_AVAILABLE:
            # Fall back to Python implementation
            return ContinuousImprovementRust._identify_improvement_opportunities_py(metrics_data)
            
        # Call the Rust implementation
        return _identify_improvement_opportunities(metrics_data)
    
    @staticmethod
    def _analyze_experiment_results_py(
        experiment_data: Dict[str, Any], 
        significance_threshold: float, 
        improvement_threshold: float
    ) -> Dict[str, Any]:
        """
        Python fallback implementation for experiment analysis
        
        This is a simplified version of the Rust implementation for fallback purposes.
        """
        logging.warning("Using Python fallback for analyze_experiment_results")
        
        # Extract control variant
        control_variant = None
        treatment_variants = []
        
        for variant in experiment_data.get("variants", []):
            if variant.get("control", False):
                control_variant = variant
            else:
                treatment_variants.append(variant)
                
        if not control_variant:
            raise ValueError("No control variant found in experiment data")
            
        # Calculate metric differences for each variant compared to control
        metrics_differences = {}
        p_values = {}
        
        for variant in treatment_variants:
            variant_name = variant.get("name", "unknown")
            variant_diffs = {}
            variant_p_values = {}
            
            # Compare each metric with the control variant
            for metric, control_value in control_variant.get("metrics", {}).items():
                if metric in variant.get("metrics", {}):
                    variant_value = variant["metrics"][metric]
                    
                    # Calculate difference
                    diff = variant_value - control_value
                    variant_diffs[metric] = diff
                    
                    # Calculate p-value (simplified t-test approximation)
                    # In a real implementation, we'd use a proper statistical test
                    sample_size = 100.0  # Assume fixed sample size for this example
                    std_error = 0.1  # Simplified std error estimate
                    t_statistic = diff / (std_error / (sample_size ** 0.5))
                    degrees_freedom = sample_size * 2.0 - 2.0
                    
                    # Simplified p-value calculation
                    p_value = 1.0 - (abs(t_statistic) / (degrees_freedom ** 0.5 + abs(t_statistic)))
                    variant_p_values[metric] = p_value
            
            metrics_differences[variant_name] = variant_diffs
            p_values[variant_name] = variant_p_values
        
        # Determine if there's a clear winner
        has_significant_results = False
        has_clear_winner = False
        winning_variant = None
        best_score = 0.0
        
        for variant_name, diffs in metrics_differences.items():
            variant_p_values = p_values[variant_name]
            
            # Check if the variant has significant improvements
            significant_improvements = 0
            variant_score = 0.0
            
            for metric, diff in diffs.items():
                p_value = variant_p_values[metric]
                
                # Check for statistical significance and meaningful improvement
                if p_value < (1.0 - significance_threshold) and diff > improvement_threshold:
                    significant_improvements += 1
                    variant_score += diff
            
            if significant_improvements > 0:
                has_significant_results = True
                
                # If this variant has a better score than the current winner, update
                if variant_score > best_score:
                    best_score = variant_score
                    winning_variant = variant_name
                    has_clear_winner = True
        
        return {
            "has_significant_results": has_significant_results,
            "has_clear_winner": has_clear_winner,
            "winning_variant": winning_variant,
            "confidence_level": significance_threshold,
            "metrics_differences": metrics_differences,
            "p_values": p_values,
        }
    
    @staticmethod
    def _analyze_multi_variant_results_py(
        experiment_data: Dict[str, Any], 
        significance_threshold: float, 
        improvement_threshold: float
    ) -> Dict[str, Any]:
        """
        Python fallback implementation for multi-variant experiment analysis
        
        This is a simplified version of the Rust implementation for fallback purposes
        with support for ANOVA and Tukey HSD analysis.
        """
        logging.warning("Using Python fallback for analyze_multi_variant_results")
        
        try:
            import numpy as np
            import scipy.stats as stats
            from scipy.stats import f_oneway
        except ImportError:
            logging.error("scipy and numpy are required for multi-variant analysis")
            raise ImportError("scipy and numpy are required for multi-variant analysis")
        
        # Extract control variant
        control_variant = None
        treatment_variants = []
        
        for variant in experiment_data.get("variants", []):
            if variant.get("control", False):
                control_variant = variant
            else:
                treatment_variants.append(variant)
                
        if not control_variant:
            raise ValueError("No control variant found in experiment data")
            
        # Basic analysis (similar to regular A/B testing)
        metrics_differences = {}
        p_values = {}
        
        for variant in treatment_variants:
            variant_name = variant.get("name", "unknown")
            variant_diffs = {}
            variant_p_values = {}
            
            # Compare each metric with the control variant
            for metric, control_value in control_variant.get("metrics", {}).items():
                if metric in variant.get("metrics", {}):
                    variant_value = variant["metrics"][metric]
                    
                    # Calculate difference
                    diff = variant_value - control_value
                    variant_diffs[metric] = diff
                    
                    # Calculate p-value (simplified t-test approximation)
                    sample_size = 100.0  # Assume fixed sample size for this example
                    std_error = 0.1  # Simplified std error estimate
                    t_statistic = diff / (std_error / (sample_size ** 0.5))
                    degrees_freedom = sample_size * 2.0 - 2.0
                    
                    # Simplified p-value calculation
                    p_value = 1.0 - (abs(t_statistic) / (degrees_freedom ** 0.5 + abs(t_statistic)))
                    variant_p_values[metric] = p_value
            
            metrics_differences[variant_name] = variant_diffs
            p_values[variant_name] = variant_p_values
        
        # ANOVA analysis for multi-variant comparison
        anova_results = {}
        key_metrics = ["sentiment_accuracy", "direction_accuracy", "calibration_error", "confidence_score"]
        
        for metric in key_metrics:
            # Get values for each variant
            values = []
            variant_names = []
            
            for variant in experiment_data.get("variants", []):
                if metric in variant.get("metrics", {}):
                    value = variant["metrics"][metric]
                    
                    # Create simulated samples (since we don't have actual distributions)
                    # This is a simplification for demonstration purposes
                    np.random.seed(hash(variant.get("name", "") + metric) % 10000)
                    samples = np.random.normal(value, 0.1, size=100)
                    values.append(samples)
                    variant_names.append(variant.get("name", "unknown"))
            
            # Skip if not enough data
            if len(values) < 2:
                anova_results[metric] = {
                    "status": "insufficient_data",
                    "message": "Not enough data for ANOVA"
                }
                continue
            
            # Run one-way ANOVA
            try:
                f_stat, p_value = f_oneway(*values)
                
                anova_results[metric] = {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "is_significant": p_value < (1 - significance_threshold),
                    "variants": variant_names
                }
            except Exception as e:
                logging.error(f"Error running ANOVA for {metric}: {e}")
                anova_results[metric] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Run Tukey HSD for pairwise comparisons
        tukey_results = {}
        
        for metric in key_metrics:
            # Skip if ANOVA was not significant
            if metric not in anova_results or not anova_results[metric].get("is_significant", False):
                continue
                
            # Get values for each variant
            values = []
            variant_names = []
            
            for variant in experiment_data.get("variants", []):
                if metric in variant.get("metrics", {}):
                    value = variant["metrics"][metric]
                    values.append(value)
                    variant_names.append(variant.get("name", "unknown"))
            
            # Run all pairwise comparisons (simplified Tukey HSD)
            pairwise_comparisons = {}
            
            for i in range(len(variant_names)):
                for j in range(i+1, len(variant_names)):
                    # Calculate mean difference
                    mean_diff = values[i] - values[j]
                    
                    # Simplified p-value calculation
                    # In a real implementation we would use proper Tukey HSD
                    std_error = 0.1 / np.sqrt(100)  # Simplified
                    t_stat = mean_diff / std_error
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), 2 * 100 - 2))
                    
                    # Determine better variant
                    better_variant = None
                    if metric == "calibration_error":  # Lower is better
                        better_variant = variant_names[i] if mean_diff < 0 else variant_names[j]
                    else:  # Higher is better
                        better_variant = variant_names[i] if mean_diff > 0 else variant_names[j]
                    
                    pair_key = f"{variant_names[i]} vs {variant_names[j]}"
                    pairwise_comparisons[pair_key] = {
                        "mean_difference": mean_diff,
                        "p_value": p_value,
                        "is_significant": p_value < (1 - significance_threshold),
                        "better_variant": better_variant
                    }
            
            tukey_results[metric] = {
                "pairwise_comparisons": pairwise_comparisons
            }
        
        # Determine if there's a clear winner
        has_significant_results = False
        has_clear_winner = False
        winning_variant = None
        
        # Calculate scores based on pairwise comparisons
        variant_scores = {}
        
        # First check for significant improvements over control
        for variant_name, diffs in metrics_differences.items():
            variant_p_values = p_values[variant_name]
            
            # Check if the variant has significant improvements
            significant_improvements = 0
            variant_score = 0.0
            
            for metric, diff in diffs.items():
                p_value = variant_p_values[metric]
                
                # Check for statistical significance and meaningful improvement
                if p_value < (1.0 - significance_threshold) and diff > improvement_threshold:
                    significant_improvements += 1
                    variant_score += diff
            
            if significant_improvements > 0:
                has_significant_results = True
                variant_scores[variant_name] = variant_score
        
        # Add Tukey HSD results to scores
        for metric, result in tukey_results.items():
            for pair, comparison in result.get("pairwise_comparisons", {}).items():
                if comparison.get("is_significant", False):
                    better_variant = comparison.get("better_variant")
                    if better_variant:
                        variant_scores[better_variant] = variant_scores.get(better_variant, 0) + 0.5
        
        # Find the variant with the highest score
        if variant_scores:
            max_score = max(variant_scores.values())
            if max_score > 0:
                winning_variants = [name for name, score in variant_scores.items() if score == max_score]
                if len(winning_variants) == 1:
                    has_clear_winner = True
                    winning_variant = winning_variants[0]
        
        result = {
            "has_significant_results": has_significant_results,
            "has_clear_winner": has_clear_winner,
            "winning_variant": winning_variant,
            "confidence_level": significance_threshold,
            "metrics_differences": metrics_differences,
            "p_values": p_values,
            "anova_results": anova_results,
            "tukey_results": tukey_results
        }
        
        return result
    
    @staticmethod
    def _identify_improvement_opportunities_py(metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Python fallback implementation for identifying improvement opportunities
        
        This is a simplified version of the Rust implementation for fallback purposes.
        """
        logging.warning("Using Python fallback for identify_improvement_opportunities")
        
        opportunities = []
        
        # Extract needed metrics
        sentiment_accuracy = metrics_data.get("sentiment_accuracy", 0.8)
        direction_accuracy = metrics_data.get("direction_accuracy", 0.7)
        calibration_error = metrics_data.get("calibration_error", 0.1)
        confidence_score = metrics_data.get("confidence_score", 0.7)
        
        # Check for prompt template opportunity
        if sentiment_accuracy < 0.85 or direction_accuracy < 0.8:
            opportunities.append({
                "type": "PROMPT_TEMPLATE",
                "reason": "Sentiment accuracy or direction accuracy is below target",
                "metrics": {
                    "sentiment_accuracy": sentiment_accuracy,
                    "direction_accuracy": direction_accuracy
                },
                "potential_impact": 0.8 * (1.0 - min(sentiment_accuracy, direction_accuracy))
            })
        
        # Check for model selection opportunity
        if calibration_error > 0.08 or confidence_score < 0.75:
            opportunities.append({
                "type": "MODEL_SELECTION",
                "reason": "Calibration error is high or confidence score is low",
                "metrics": {
                    "calibration_error": calibration_error,
                    "confidence_score": confidence_score
                },
                "potential_impact": 0.7 * (calibration_error + (1.0 - confidence_score))
            })
            
        # Check for temperature parameter opportunity
        if calibration_error > 0.05:
            opportunities.append({
                "type": "TEMPERATURE",
                "reason": "High calibration error suggests temperature tuning needed",
                "metrics": {
                    "calibration_error": calibration_error
                },
                "potential_impact": 0.6 * calibration_error
            })
            
        # Simplified version - we'd add more opportunities in the real implementation
            
        return opportunities