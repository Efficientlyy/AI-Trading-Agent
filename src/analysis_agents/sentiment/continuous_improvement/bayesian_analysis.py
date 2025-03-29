"""Bayesian analysis for experiments in the continuous improvement system.

This module provides Bayesian analysis methods for experiment evaluation,
offering probabilistic interpretations of results and handling uncertainty
more robustly than traditional frequentist methods.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from src.common.logging import get_logger
from src.analysis_agents.sentiment.ab_testing import Experiment, ExperimentVariant, ExperimentMetrics

# Initialize logger
logger = get_logger("analysis_agents", "bayesian_analysis")

try:
    import pymc as pm
    import arviz as az
    from scipy import stats
    BAYESIAN_AVAILABLE = True
except ImportError:
    logger.warning("PyMC and/or Arviz not available, Bayesian analysis will be limited")
    BAYESIAN_AVAILABLE = False


class BayesianAnalysisResults:
    """Container for Bayesian analysis results."""
    
    def __init__(
        self,
        winning_probability: Dict[str, Dict[str, float]],
        lift_estimation: Dict[str, Dict[str, Dict[str, float]]],
        posterior_samples: Dict[str, Dict[str, np.ndarray]],
        expected_loss: Dict[str, Dict[str, float]],
        credible_intervals: Dict[str, Dict[str, Dict[str, List[float]]]],
        metrics_analyzed: List[str],
        experiment_name: str,
        plots: Dict[str, str] = None  # base64 encoded plots
    ):
        """Initialize Bayesian analysis results.
        
        Args:
            winning_probability: Probability each variant is best for each metric
            lift_estimation: Estimated lift over control for each variant and metric
            posterior_samples: Posterior samples for each variant and metric
            expected_loss: Expected loss for choosing each variant
            credible_intervals: Credible intervals for each variant and metric
            metrics_analyzed: List of metrics that were analyzed
            experiment_name: Name of the analyzed experiment
            plots: Dictionary of base64 encoded plots
        """
        self.winning_probability = winning_probability
        self.lift_estimation = lift_estimation
        self.posterior_samples = posterior_samples
        self.expected_loss = expected_loss
        self.credible_intervals = credible_intervals
        self.metrics_analyzed = metrics_analyzed
        self.experiment_name = experiment_name
        self.plots = plots or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary.
        
        Returns:
            Dictionary representation of results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_posterior_samples = {}
        for metric, variants in self.posterior_samples.items():
            serializable_posterior_samples[metric] = {}
            for variant, samples in variants.items():
                if isinstance(samples, np.ndarray):
                    serializable_posterior_samples[metric][variant] = samples.tolist()
                else:
                    serializable_posterior_samples[metric][variant] = samples
        
        return {
            "winning_probability": self.winning_probability,
            "lift_estimation": self.lift_estimation,
            "posterior_samples": serializable_posterior_samples,
            "expected_loss": self.expected_loss,
            "credible_intervals": self.credible_intervals,
            "metrics_analyzed": self.metrics_analyzed,
            "experiment_name": self.experiment_name,
            "plots": self.plots,
            "has_clear_winner": self.has_clear_winner(),
            "winning_variant": self.get_winning_variant()
        }
    
    def has_clear_winner(self) -> bool:
        """Check if there's a clear winner across all metrics.
        
        Returns:
            True if there's a clear winner, False otherwise
        """
        if not self.winning_probability:
            return False
        
        # Calculate average winning probability across all metrics
        avg_probs = {}
        for metric, probs in self.winning_probability.items():
            for variant, prob in probs.items():
                if variant not in avg_probs:
                    avg_probs[variant] = []
                avg_probs[variant].append(prob)
        
        avg_winning_probs = {
            variant: sum(probs) / len(probs) 
            for variant, probs in avg_probs.items() if probs
        }
        
        # Find the variant with the highest average winning probability
        if avg_winning_probs:
            max_prob = max(avg_winning_probs.values())
            # Consider it a clear winner if its probability is above 0.8
            if max_prob > 0.8:
                return True
        
        return False
    
    def get_winning_variant(self) -> Optional[str]:
        """Get the winning variant across all metrics.
        
        Returns:
            Name of the winning variant or None if no clear winner
        """
        if not self.has_clear_winner():
            return None
        
        # Calculate average winning probability across all metrics
        avg_probs = {}
        for metric, probs in self.winning_probability.items():
            for variant, prob in probs.items():
                if variant not in avg_probs:
                    avg_probs[variant] = []
                avg_probs[variant].append(prob)
        
        avg_winning_probs = {
            variant: sum(probs) / len(probs) 
            for variant, probs in avg_probs.items() if probs
        }
        
        # Find the variant with the highest average winning probability
        return max(avg_winning_probs.items(), key=lambda x: x[1])[0]
    
    def get_summary(self) -> str:
        """Get a textual summary of the Bayesian analysis results.
        
        Returns:
            Summary text
        """
        summary = []
        summary.append(f"Bayesian Analysis Summary for {self.experiment_name}")
        summary.append("=" * 60)
        
        winning_variant = self.get_winning_variant()
        if winning_variant:
            summary.append(f"Clear Winner: {winning_variant}")
        else:
            summary.append("No Clear Winner")
        
        summary.append("\nWinning Probabilities:")
        for metric in self.metrics_analyzed:
            if metric in self.winning_probability:
                summary.append(f"  {metric}:")
                for variant, prob in sorted(
                    self.winning_probability[metric].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    summary.append(f"    {variant}: {prob:.2%}")
        
        summary.append("\nEstimated Lift Over Control:")
        for metric in self.metrics_analyzed:
            if metric in self.lift_estimation:
                summary.append(f"  {metric}:")
                for variant, lift in self.lift_estimation[metric].items():
                    if "mean" in lift:
                        mean = lift["mean"]
                        summary.append(f"    {variant}: {mean:.2%}")
        
        summary.append("\nRecommendation:")
        if winning_variant:
            summary.append(f"  Implement variant '{winning_variant}' as it shows the ")
            summary.append(f"  highest probability of being the best variant across all metrics.")
        else:
            summary.append("  Consider collecting more data or running a new experiment.")
            # Find the variant with highest probability for most metrics
            variant_count = {}
            for metric, probs in self.winning_probability.items():
                best_variant = max(probs.items(), key=lambda x: x[1])[0]
                variant_count[best_variant] = variant_count.get(best_variant, 0) + 1
            
            if variant_count:
                most_wins = max(variant_count.items(), key=lambda x: x[1])
                summary.append(f"  '{most_wins[0]}' is the best variant for {most_wins[1]} out of {len(self.metrics_analyzed)} metrics.")
        
        return "\n".join(summary)
    
    def get_detailed_report(self) -> str:
        """Get a detailed report of the Bayesian analysis results.
        
        Returns:
            Detailed report
        """
        report = []
        report.append(f"Detailed Bayesian Analysis Report for {self.experiment_name}")
        report.append("=" * 80)
        
        # Summary section
        winning_variant = self.get_winning_variant()
        if winning_variant:
            report.append(f"Clear Winner: {winning_variant}")
        else:
            report.append("No Clear Winner")
        
        # Winning probabilities
        report.append("\nWinning Probabilities (Probability of Being Best):")
        for metric in self.metrics_analyzed:
            if metric in self.winning_probability:
                report.append(f"  {metric}:")
                for variant, prob in sorted(
                    self.winning_probability[metric].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    report.append(f"    {variant}: {prob:.4f} ({prob:.2%})")
        
        # Lift estimation
        report.append("\nEstimated Lift Over Control:")
        for metric in self.metrics_analyzed:
            if metric in self.lift_estimation:
                report.append(f"  {metric}:")
                for variant, lift in self.lift_estimation[metric].items():
                    if all(k in lift for k in ["mean", "std", "credible_interval"]):
                        mean = lift["mean"]
                        std = lift["std"]
                        ci = lift["credible_interval"]
                        report.append(f"    {variant}: {mean:.4f} ± {std:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        # Expected loss
        report.append("\nExpected Loss (Regret for Choosing Each Variant):")
        for metric in self.metrics_analyzed:
            if metric in self.expected_loss:
                report.append(f"  {metric}:")
                for variant, loss in sorted(
                    self.expected_loss[metric].items(),
                    key=lambda x: x[1]
                ):
                    report.append(f"    {variant}: {loss:.6f}")
        
        # Credible intervals
        report.append("\nCredible Intervals (95%):")
        for metric in self.metrics_analyzed:
            if metric in self.credible_intervals:
                report.append(f"  {metric}:")
                for variant, intervals in self.credible_intervals[metric].items():
                    if "95%" in intervals:
                        ci = intervals["95%"]
                        report.append(f"    {variant}: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        # Recommendation
        report.append("\nDetailed Recommendation:")
        if winning_variant:
            report.append(f"  Implement variant '{winning_variant}' as it shows the highest probability")
            report.append(f"  of being the best variant across all metrics. This assessment is based on")
            report.append(f"  Bayesian analysis which accounts for uncertainty in the experimental data.")
            
            # Add metrics details
            winning_metrics = []
            for metric, probs in self.winning_probability.items():
                if winning_variant in probs and probs[winning_variant] > 0.5:
                    winning_metrics.append(metric)
            
            if winning_metrics:
                report.append(f"\n  '{winning_variant}' is particularly strong in these metrics:")
                for metric in winning_metrics:
                    prob = self.winning_probability[metric][winning_variant]
                    report.append(f"    - {metric}: {prob:.2%} probability of being best")
        else:
            report.append("  The Bayesian analysis does not indicate a clear winner across all metrics.")
            report.append("  Consider collecting more data to reduce uncertainty or running a new")
            report.append("  experiment with more distinctive variants.")
        
        return "\n".join(report)


class BayesianAnalyzer:
    """Analyzer for Bayesian analysis of experiments."""
    
    def __init__(self, mcmc_samples: int = 10000, credible_interval: float = 0.95):
        """Initialize the Bayesian analyzer.
        
        Args:
            mcmc_samples: Number of MCMC samples to draw
            credible_interval: Credible interval (0-1)
        """
        self.mcmc_samples = mcmc_samples
        self.credible_interval = credible_interval
        self.logger = get_logger("analysis_agents", "bayesian_analyzer")
    
    def analyze_experiment(self, experiment: Experiment) -> BayesianAnalysisResults:
        """Analyze an experiment using Bayesian methods.
        
        Args:
            experiment: The experiment to analyze
            
        Returns:
            Bayesian analysis results
        """
        if not BAYESIAN_AVAILABLE:
            self.logger.warning("PyMC not available, falling back to simplified Bayesian analysis")
            return self._analyze_experiment_simple(experiment)
        
        # Find control variant
        control_variant = None
        for variant in experiment.variants:
            if variant.control:
                control_variant = variant
                break
        
        if not control_variant:
            # Use first variant as control if none specified
            control_variant = experiment.variants[0]
        
        # Get metrics to analyze
        metrics = ["sentiment_accuracy", "direction_accuracy", "calibration_error", "confidence_score"]
        
        # Results containers
        winning_probability = {}
        lift_estimation = {}
        posterior_samples = {}
        expected_loss = {}
        credible_intervals = {}
        plots = {}
        
        # Analyze each metric
        for metric in metrics:
            try:
                # Get metric values for each variant
                variant_values = {}
                
                for variant in experiment.variants:
                    variant_metrics = experiment.variant_metrics[variant.id]
                    
                    # Get the metric value
                    if metric == "sentiment_accuracy":
                        value = variant_metrics.sentiment_accuracy
                    elif metric == "direction_accuracy":
                        value = variant_metrics.direction_accuracy
                    elif metric == "calibration_error":
                        value = variant_metrics.calibration_error
                    elif metric == "confidence_score":
                        value = variant_metrics.confidence_score
                    else:
                        continue
                    
                    # Only include if valid value
                    if value > 0:
                        variant_values[variant.name] = value
                
                # Skip if not enough data
                if len(variant_values) < 2:
                    self.logger.info(f"Skipping Bayesian analysis for {metric}: not enough data")
                    continue
                
                # For metrics where lower is better (like calibration_error), invert the values
                if metric == "calibration_error":
                    variant_values = {k: 1 - v for k, v in variant_values.items()}
                
                # Run Bayesian analysis for this metric
                results = self._bayesian_analysis_for_metric(variant_values, experiment.variant_metrics, metric)
                
                # Store results
                winning_probability[metric] = results["winning_probability"]
                lift_estimation[metric] = results["lift_estimation"]
                posterior_samples[metric] = results["posterior_samples"]
                expected_loss[metric] = results["expected_loss"]
                credible_intervals[metric] = results["credible_intervals"]
                
                # Generate and store plot
                plot_data = self._generate_posterior_plot(results["posterior_samples"], metric)
                if plot_data:
                    plots[f"{metric}_posterior"] = plot_data
            
            except Exception as e:
                self.logger.error(f"Error in Bayesian analysis for {metric}: {e}")
                # Continue with other metrics
        
        # Create results object
        results = BayesianAnalysisResults(
            winning_probability=winning_probability,
            lift_estimation=lift_estimation,
            posterior_samples=posterior_samples,
            expected_loss=expected_loss,
            credible_intervals=credible_intervals,
            metrics_analyzed=metrics,
            experiment_name=experiment.name,
            plots=plots
        )
        
        return results
    
    def _bayesian_analysis_for_metric(
        self, 
        variant_values: Dict[str, float],
        variant_metrics: Dict[str, ExperimentMetrics],
        metric: str
    ) -> Dict[str, Any]:
        """Run Bayesian analysis for a specific metric.
        
        Args:
            variant_values: Values for each variant
            variant_metrics: Metrics for each variant
            metric: Metric name
            
        Returns:
            Analysis results for this metric
        """
        # Find control variant
        control_variant = None
        control_value = None
        for name, value in variant_values.items():
            # Assume variant with name containing "control" or "current" is the control
            if "control" in name.lower() or "current" in name.lower():
                control_variant = name
                control_value = value
                break
        
        # If no control found, use the first variant
        if control_variant is None:
            control_variant = list(variant_values.keys())[0]
            control_value = variant_values[control_variant]
        
        # Prepare data for PyMC model
        variant_names = list(variant_values.keys())
        values = list(variant_values.values())
        
        # Counts and sample sizes for each variant
        success_counts = {}
        sample_sizes = {}
        
        # Get the variant ID for each variant
        variant_id_map = {}
        for var_name in variant_names:
            for v in variant_metrics:
                if variant_metrics[v].requests > 0:
                    if var_name in variant_metrics[v].to_dict()["id"]:
                        variant_id_map[var_name] = v
                        break
        
        # Calculate success counts based on the metric
        for variant_name in variant_names:
            if variant_name in variant_id_map:
                variant_id = variant_id_map[variant_name]
                sample_size = variant_metrics[variant_id].requests
                
                # Adjust calculation based on metric type
                if metric in ["sentiment_accuracy", "direction_accuracy"]:
                    success_count = int(variant_values[variant_name] * sample_size)
                elif metric in ["calibration_error"]:
                    # For error metrics, lower is better, so we invert
                    # But we've already inverted the value, so use it directly
                    success_count = int(variant_values[variant_name] * sample_size)
                else:
                    # For other metrics like confidence_score, use the value as a proportion
                    success_count = int(variant_values[variant_name] * sample_size)
                
                success_counts[variant_name] = success_count
                sample_sizes[variant_name] = sample_size
            else:
                # Fallback if variant ID not found
                self.logger.warning(f"Variant ID not found for {variant_name}")
                success_counts[variant_name] = int(variant_values[variant_name] * 100)
                sample_sizes[variant_name] = 100
        
        # Create PyMC model
        with pm.Model() as model:
            # Priors for each variant
            # We use Beta(1, 1) which is a uniform prior
            alpha_prior = 1
            beta_prior = 1
            
            # Parameters for each variant
            variant_params = {}
            for variant_name in variant_names:
                variant_params[variant_name] = pm.Beta(
                    variant_name, 
                    alpha=alpha_prior,
                    beta=beta_prior
                )
            
            # Likelihood for each variant
            for variant_name in variant_names:
                pm.Binomial(
                    f"obs_{variant_name}",
                    n=sample_sizes[variant_name],
                    p=variant_params[variant_name],
                    observed=success_counts[variant_name]
                )
            
            # Compute the difference from control for each variant
            diffs = {}
            for variant_name in variant_names:
                if variant_name != control_variant:
                    diffs[variant_name] = pm.Deterministic(
                        f"diff_{variant_name}",
                        variant_params[variant_name] - variant_params[control_variant]
                    )
            
            # Compute relative lift
            lifts = {}
            for variant_name in variant_names:
                if variant_name != control_variant:
                    lifts[variant_name] = pm.Deterministic(
                        f"lift_{variant_name}",
                        (variant_params[variant_name] - variant_params[control_variant]) / variant_params[control_variant]
                    )
            
            # Sample from the posterior
            trace = pm.sample(
                self.mcmc_samples, 
                tune=1000,
                progressbar=False,
                chains=2,
                cores=1  # Use 1 core to avoid multiprocessing issues
            )
        
        # Extract posterior samples
        posterior_samples = {}
        for variant_name in variant_names:
            posterior_samples[variant_name] = trace.posterior[variant_name].values.flatten()
        
        # Calculate probability that each variant is the best
        best_variant_counts = {variant_name: 0 for variant_name in variant_names}
        
        # For each MCMC sample, find which variant has the highest value
        for i in range(len(posterior_samples[variant_names[0]])):
            values = {
                variant_name: posterior_samples[variant_name][i]
                for variant_name in variant_names
            }
            best_variant = max(values.items(), key=lambda x: x[1])[0]
            best_variant_counts[best_variant] += 1
        
        # Convert to probabilities
        winning_probability = {
            variant_name: count / self.mcmc_samples
            for variant_name, count in best_variant_counts.items()
        }
        
        # Calculate lift estimation
        lift_estimation = {}
        for variant_name in variant_names:
            if variant_name != control_variant:
                # Get lift samples
                lift_samples = trace.posterior[f"lift_{variant_name}"].values.flatten()
                
                # Calculate statistics
                lift_mean = lift_samples.mean()
                lift_std = lift_samples.std()
                lift_ci = np.percentile(lift_samples, [2.5, 97.5])
                
                lift_estimation[variant_name] = {
                    "mean": float(lift_mean),
                    "std": float(lift_std),
                    "credible_interval": [float(lift_ci[0]), float(lift_ci[1])]
                }
        
        # Calculate expected loss for each variant
        expected_loss = {}
        
        # For each variant, calculate the expected loss if we choose it
        for variant_name in variant_names:
            loss = 0
            for i in range(len(posterior_samples[variant_names[0]])):
                # Get the value for this variant in this sample
                variant_value = posterior_samples[variant_name][i]
                
                # Find the maximum value across all variants in this sample
                max_value = max(posterior_samples[v][i] for v in variant_names)
                
                # The loss is the difference between the max value and this variant's value
                loss += max_value - variant_value
            
            # Average loss across all samples
            expected_loss[variant_name] = loss / len(posterior_samples[variant_names[0]])
        
        # Calculate credible intervals
        credible_intervals = {}
        for variant_name in variant_names:
            samples = posterior_samples[variant_name]
            
            credible_intervals[variant_name] = {
                "95%": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
                "90%": [float(np.percentile(samples, 5)), float(np.percentile(samples, 95))],
                "80%": [float(np.percentile(samples, 10)), float(np.percentile(samples, 90))]
            }
        
        return {
            "winning_probability": winning_probability,
            "lift_estimation": lift_estimation,
            "posterior_samples": posterior_samples,
            "expected_loss": expected_loss,
            "credible_intervals": credible_intervals
        }
    
    def _generate_posterior_plot(self, posterior_samples: Dict[str, np.ndarray], metric: str) -> Optional[str]:
        """Generate a posterior distribution plot.
        
        Args:
            posterior_samples: Posterior samples for each variant
            metric: Metric name
            
        Returns:
            Base64 encoded image or None on error
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot posterior distributions for each variant
            for variant_name, samples in posterior_samples.items():
                # Use kernel density estimation to smooth the distribution
                density = stats.gaussian_kde(samples)
                x = np.linspace(min(samples), max(samples), 1000)
                plt.plot(x, density(x), label=variant_name)
            
            # Add details
            title = f"Posterior Distributions for {metric.replace('_', ' ').title()}"
            if metric == "calibration_error":
                title += " (Lower is Better, Inverted for Comparison)"
                
            plt.title(title)
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot to a bytes buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return img_str
        
        except Exception as e:
            self.logger.error(f"Error generating posterior plot: {e}")
            return None
    
    def _analyze_experiment_simple(self, experiment: Experiment) -> BayesianAnalysisResults:
        """Run a simplified Bayesian analysis when PyMC is not available.
        
        Args:
            experiment: The experiment to analyze
            
        Returns:
            Simplified Bayesian analysis results
        """
        self.logger.info("Running simplified Bayesian analysis")
        
        # Find control variant
        control_variant = None
        for variant in experiment.variants:
            if variant.control:
                control_variant = variant
                break
        
        if not control_variant:
            # Use first variant as control if none specified
            control_variant = experiment.variants[0]
        
        # Get metrics to analyze
        metrics = ["sentiment_accuracy", "direction_accuracy", "calibration_error", "confidence_score"]
        
        # Results containers
        winning_probability = {}
        lift_estimation = {}
        posterior_samples = {}
        expected_loss = {}
        credible_intervals = {}
        
        # Analyze each metric
        for metric in metrics:
            try:
                # Get metric values for each variant
                variant_values = {}
                variant_sample_sizes = {}
                
                for variant in experiment.variants:
                    variant_metrics = experiment.variant_metrics[variant.id]
                    sample_size = variant_metrics.requests
                    
                    # Skip if not enough data
                    if sample_size < 10:
                        continue
                    
                    # Get the metric value
                    if metric == "sentiment_accuracy":
                        value = variant_metrics.sentiment_accuracy
                    elif metric == "direction_accuracy":
                        value = variant_metrics.direction_accuracy
                    elif metric == "calibration_error":
                        value = variant_metrics.calibration_error
                    elif metric == "confidence_score":
                        value = variant_metrics.confidence_score
                    else:
                        continue
                    
                    # Only include if valid value
                    if value > 0:
                        variant_values[variant.name] = value
                        variant_sample_sizes[variant.name] = sample_size
                
                # Skip if not enough data
                if len(variant_values) < 2:
                    continue
                
                # For metrics where lower is better (like calibration_error), invert the values
                inverted = False
                if metric == "calibration_error":
                    variant_values = {k: 1 - v for k, v in variant_values.items()}
                    inverted = True
                
                # Run simplified Bayesian analysis for this metric
                results = self._simplified_bayesian_analysis(
                    variant_values, 
                    variant_sample_sizes, 
                    control_variant.name,
                    inverted
                )
                
                # Store results
                winning_probability[metric] = results["winning_probability"]
                lift_estimation[metric] = results["lift_estimation"]
                posterior_samples[metric] = results["posterior_samples"]
                expected_loss[metric] = results["expected_loss"]
                credible_intervals[metric] = results["credible_intervals"]
            
            except Exception as e:
                self.logger.error(f"Error in simplified Bayesian analysis for {metric}: {e}")
                # Continue with other metrics
        
        # Create results object
        results = BayesianAnalysisResults(
            winning_probability=winning_probability,
            lift_estimation=lift_estimation,
            posterior_samples=posterior_samples,
            expected_loss=expected_loss,
            credible_intervals=credible_intervals,
            metrics_analyzed=metrics,
            experiment_name=experiment.name
        )
        
        return results
    
    def _simplified_bayesian_analysis(
        self,
        variant_values: Dict[str, float],
        variant_sample_sizes: Dict[str, int],
        control_name: str,
        inverted: bool = False
    ) -> Dict[str, Any]:
        """Run a simplified Bayesian analysis using Beta distributions.
        
        Args:
            variant_values: Values for each variant
            variant_sample_sizes: Sample sizes for each variant
            control_name: Name of the control variant
            inverted: Whether the metric values have been inverted (for metrics where lower is better)
            
        Returns:
            Analysis results
        """
        # Results containers
        winning_probability = {}
        lift_estimation = {}
        posterior_samples = {}
        expected_loss = {}
        credible_intervals = {}
        
        # Parameter conversion (converting metric values to Beta distribution parameters)
        variant_params = {}
        
        for variant_name, value in variant_values.items():
            sample_size = variant_sample_sizes.get(variant_name, 100)
            
            # Convert to Beta parameters using method of moments
            # For a Beta distribution, mean = alpha / (alpha + beta)
            # We'll set alpha + beta = sample_size (effective sample size)
            mean = value
            alpha = mean * sample_size
            beta = sample_size - alpha
            
            # Ensure parameters are positive
            alpha = max(0.1, alpha)
            beta = max(0.1, beta)
            
            variant_params[variant_name] = (alpha, beta)
        
        # Generate posterior samples for each variant
        for variant_name, (alpha, beta) in variant_params.items():
            # Generate samples from Beta distribution
            samples = np.random.beta(alpha, beta, self.mcmc_samples)
            posterior_samples[variant_name] = samples
        
        # Calculate probability that each variant is the best
        best_variant_counts = {variant_name: 0 for variant_name in variant_values.keys()}
        
        # For each sample, find which variant has the highest value
        for i in range(self.mcmc_samples):
            values = {
                variant_name: posterior_samples[variant_name][i]
                for variant_name in variant_values.keys()
            }
            best_variant = max(values.items(), key=lambda x: x[1])[0]
            best_variant_counts[best_variant] += 1
        
        # Convert to probabilities
        winning_probability = {
            variant_name: count / self.mcmc_samples
            for variant_name, count in best_variant_counts.items()
        }
        
        # Calculate lift estimation relative to control
        for variant_name in variant_values.keys():
            if variant_name != control_name:
                # Calculate relative lift samples
                lift_samples = (
                    posterior_samples[variant_name] - posterior_samples[control_name]
                ) / posterior_samples[control_name]
                
                # Calculate statistics
                lift_mean = lift_samples.mean()
                lift_std = lift_samples.std()
                lift_ci = np.percentile(lift_samples, [2.5, 97.5])
                
                lift_estimation[variant_name] = {
                    "mean": float(lift_mean),
                    "std": float(lift_std),
                    "credible_interval": [float(lift_ci[0]), float(lift_ci[1])]
                }
        
        # Calculate expected loss for each variant
        for variant_name in variant_values.keys():
            loss = 0
            for i in range(self.mcmc_samples):
                # Get the value for this variant in this sample
                variant_value = posterior_samples[variant_name][i]
                
                # Find the maximum value across all variants in this sample
                max_value = max(posterior_samples[v][i] for v in variant_values.keys())
                
                # The loss is the difference between the max value and this variant's value
                loss += max_value - variant_value
            
            # Average loss across all samples
            expected_loss[variant_name] = loss / self.mcmc_samples
        
        # Calculate credible intervals
        for variant_name in variant_values.keys():
            samples = posterior_samples[variant_name]
            
            credible_intervals[variant_name] = {
                "95%": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
                "90%": [float(np.percentile(samples, 5)), float(np.percentile(samples, 95))],
                "80%": [float(np.percentile(samples, 10)), float(np.percentile(samples, 90))]
            }
        
        return {
            "winning_probability": winning_probability,
            "lift_estimation": lift_estimation,
            "posterior_samples": posterior_samples,
            "expected_loss": expected_loss,
            "credible_intervals": credible_intervals
        }


# Create singleton instance
bayesian_analyzer = BayesianAnalyzer()