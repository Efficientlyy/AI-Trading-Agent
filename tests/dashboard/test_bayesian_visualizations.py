"""
Unit tests for Bayesian analysis visualization utilities.

Tests the creation of various visualization components for Bayesian analysis results
and experiment data in the continuous improvement system.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.analysis_agents.sentiment.ab_testing import (
    ExperimentStatus, Experiment, ExperimentVariant, ExperimentMetrics
)
from src.dashboard.bayesian_visualizations import (
    create_posterior_distribution_plot,
    create_winning_probability_chart,
    create_lift_estimation_chart,
    create_experiment_progress_chart,
    create_expected_loss_chart,
    create_credible_interval_chart,
    create_multi_variant_comparison_chart,
    generate_experiment_visualizations
)


class TestBayesianVisualizations(unittest.TestCase):
    """Test case for Bayesian visualization functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock experiment
        self.experiment = MagicMock(spec=Experiment)
        self.experiment.id = "test-experiment-1"
        self.experiment.name = "Test Experiment"
        self.experiment.status = ExperimentStatus.ACTIVE
        self.experiment.sample_size = 200
        
        # Create variants
        self.control_variant = MagicMock(spec=ExperimentVariant)
        self.control_variant.id = "control-variant"
        self.control_variant.name = "Control"
        self.control_variant.control = True
        
        self.treatment_variant = MagicMock(spec=ExperimentVariant)
        self.treatment_variant.id = "treatment-variant"
        self.treatment_variant.name = "Treatment"
        self.treatment_variant.control = False
        
        self.experiment.variants = [self.control_variant, self.treatment_variant]
        
        # Create metrics
        self.control_metrics = MagicMock(spec=ExperimentMetrics)
        self.control_metrics.requests = 150
        self.control_metrics.sentiment_accuracy = 0.75
        self.control_metrics.direction_accuracy = 0.70
        self.control_metrics.calibration_error = 0.15
        self.control_metrics.confidence_score = 0.65
        
        self.treatment_metrics = MagicMock(spec=ExperimentMetrics)
        self.treatment_metrics.requests = 160
        self.treatment_metrics.sentiment_accuracy = 0.82
        self.treatment_metrics.direction_accuracy = 0.78
        self.treatment_metrics.calibration_error = 0.10
        self.treatment_metrics.confidence_score = 0.72
        
        self.experiment.variant_metrics = {
            self.control_variant.id: self.control_metrics,
            self.treatment_variant.id: self.treatment_metrics
        }
        
        # Set start time
        self.experiment.start_time = datetime.utcnow() - timedelta(days=5)
        
        # Mock Bayesian analysis data
        self.posterior_samples = {
            "Control": np.random.normal(0.75, 0.05, 1000),
            "Treatment": np.random.normal(0.82, 0.03, 1000)
        }
        
        self.winning_probability = {
            "sentiment_accuracy": {
                "Control": 0.15,
                "Treatment": 0.85
            },
            "direction_accuracy": {
                "Control": 0.20,
                "Treatment": 0.80
            }
        }
        
        self.expected_loss = {
            "sentiment_accuracy": {
                "Control": 0.07,
                "Treatment": 0.02
            },
            "direction_accuracy": {
                "Control": 0.08,
                "Treatment": 0.03
            }
        }
        
        self.lift_estimation = {
            "sentiment_accuracy": {
                "Control": {
                    "mean": 0.0,
                    "credible_interval": [0.0, 0.0]
                },
                "Treatment": {
                    "mean": 0.09,
                    "credible_interval": [0.05, 0.13]
                }
            },
            "direction_accuracy": {
                "Control": {
                    "mean": 0.0,
                    "credible_interval": [0.0, 0.0]
                },
                "Treatment": {
                    "mean": 0.11,
                    "credible_interval": [0.06, 0.16]
                }
            }
        }
        
        self.credible_intervals = {
            "sentiment_accuracy": {
                "Control": {
                    "95%": [0.70, 0.80]
                },
                "Treatment": {
                    "95%": [0.78, 0.86]
                }
            },
            "direction_accuracy": {
                "Control": {
                    "95%": [0.65, 0.75]
                },
                "Treatment": {
                    "95%": [0.73, 0.83]
                }
            }
        }
        
        # Mock stopping criteria results
        self.stopping_criteria_results = {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "status": "ACTIVE",
            "should_stop": True,
            "stopping_reasons": [
                {
                    "criterion": "bayesian_probability",
                    "reason": "Variant 'Treatment' has 85.0% probability of being best (threshold: 80.0%)"
                }
            ],
            "criteria_results": {
                "sample_size": {
                    "should_stop": True,
                    "reason": "All variants have at least 100 samples"
                },
                "bayesian_probability": {
                    "should_stop": True,
                    "reason": "Variant 'Treatment' has 85.0% probability of being best (threshold: 80.0%)"
                },
                "expected_loss": {
                    "should_stop": False,
                    "reason": "Expected loss is above the threshold"
                },
                "confidence_interval": {
                    "should_stop": False,
                    "reason": "Confidence intervals still too wide (widest: 10.0%, threshold: 5.0%)"
                },
                "time_limit": {
                    "should_stop": False,
                    "reason": "Experiment has been running for 5.0 days (limit: 14 days)"
                }
            }
        }

    def test_create_posterior_distribution_plot(self):
        """Test creating posterior distribution plot."""
        # Create plot
        fig = create_posterior_distribution_plot(
            self.posterior_samples,
            "sentiment_accuracy"
        )
        
        # Basic validation
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(4, len(fig.data))  # 2 distributions and 2 mean lines
        
        # Test with empty data
        fig = create_posterior_distribution_plot({}, "sentiment_accuracy")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(0, len(fig.data))
        
        # Test with custom title
        fig = create_posterior_distribution_plot(
            self.posterior_samples,
            "sentiment_accuracy",
            title="Custom Title"
        )
        self.assertEqual("Custom Title", fig.layout.title.text)

    def test_create_winning_probability_chart(self):
        """Test creating winning probability chart."""
        # Create chart
        fig = create_winning_probability_chart(
            self.winning_probability,
            self.experiment.name
        )
        
        # Basic validation
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(1, len(fig.data))  # Bar chart
        
        # Test with empty data
        fig = create_winning_probability_chart({}, self.experiment.name)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(0, len(fig.data))

    def test_create_lift_estimation_chart(self):
        """Test creating lift estimation chart."""
        # Create chart
        fig = create_lift_estimation_chart(
            self.lift_estimation,
            self.experiment.name
        )
        
        # Basic validation
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty data
        fig = create_lift_estimation_chart({}, self.experiment.name)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(0, len(fig.data))

    def test_create_experiment_progress_chart(self):
        """Test creating experiment progress chart."""
        # Create chart
        fig = create_experiment_progress_chart(
            self.experiment,
            self.stopping_criteria_results
        )
        
        # Basic validation
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty data
        fig = create_experiment_progress_chart(None, {})
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(0, len(fig.data))

    def test_create_expected_loss_chart(self):
        """Test creating expected loss chart."""
        # Create chart
        fig = create_expected_loss_chart(
            self.expected_loss,
            self.experiment.name
        )
        
        # Basic validation
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(1, len(fig.data))  # Bar chart
        
        # Test with empty data
        fig = create_expected_loss_chart({}, self.experiment.name)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(0, len(fig.data))

    def test_create_credible_interval_chart(self):
        """Test creating credible interval chart."""
        # Create chart
        fig = create_credible_interval_chart(
            self.credible_intervals,
            self.experiment.name
        )
        
        # Basic validation
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty data
        fig = create_credible_interval_chart({}, self.experiment.name)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(0, len(fig.data))

    def test_create_multi_variant_comparison_chart(self):
        """Test creating multi-variant comparison chart."""
        # Add results to experiment
        self.experiment.results = {
            "anova_results": {
                "sentiment_accuracy": {
                    "p_value": 0.01,
                    "f_statistic": 8.5,
                    "is_significant": True
                },
                "direction_accuracy": {
                    "p_value": 0.08,
                    "f_statistic": 3.2,
                    "is_significant": False
                }
            },
            "tukey_results": {
                "sentiment_accuracy": {
                    "pairwise_comparisons": {
                        "Control vs Treatment": {
                            "is_significant": True,
                            "better_variant": "Treatment"
                        }
                    }
                }
            }
        }
        
        # Create chart
        fig = create_multi_variant_comparison_chart(
            self.experiment,
            self.experiment.results
        )
        
        # Basic validation
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty data
        fig = create_multi_variant_comparison_chart(None, {})
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(0, len(fig.data))

    @patch("src.dashboard.bayesian_visualizations.BayesianAnalyzer")
    def test_generate_experiment_visualizations(self, mock_analyzer_class):
        """Test generating all experiment visualizations."""
        # Setup mock
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create mock analysis results
        mock_results = MagicMock()
        mock_results.posterior_samples = {
            "sentiment_accuracy": self.posterior_samples,
            "direction_accuracy": self.posterior_samples
        }
        mock_results.winning_probability = self.winning_probability
        mock_results.expected_loss = self.expected_loss
        mock_results.lift_estimation = self.lift_estimation
        mock_results.credible_intervals = self.credible_intervals
        mock_analyzer.analyze_experiment.return_value = mock_results
        
        # Generate visualizations
        visualizations = generate_experiment_visualizations(
            self.experiment,
            self.stopping_criteria_results
        )
        
        # Check that all visualizations were created
        self.assertIsInstance(visualizations, dict)
        self.assertIn("winning_probability", visualizations)
        self.assertIn("lift_estimation", visualizations)
        self.assertIn("posterior_sentiment_accuracy", visualizations)
        self.assertIn("posterior_direction_accuracy", visualizations)
        self.assertIn("expected_loss", visualizations)
        self.assertIn("credible_intervals", visualizations)
        self.assertIn("experiment_progress", visualizations)
        
        # Verify all are Plotly figures
        for viz_name, viz in visualizations.items():
            self.assertIsInstance(viz, go.Figure, f"{viz_name} is not a Figure")


if __name__ == "__main__":
    unittest.main()