"""
Unit tests for automatic stopping criteria for continuous improvement experiments.

Tests the implementation of various stopping criteria classes and the stopping criteria manager.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from datetime import datetime, timedelta

from src.analysis_agents.sentiment.ab_testing import (
    ExperimentStatus, Experiment, ExperimentVariant, ExperimentMetrics
)
from src.analysis_agents.sentiment.continuous_improvement.stopping_criteria import (
    StoppingCriterion, SampleSizeCriterion, BayesianProbabilityThresholdCriterion,
    ExpectedLossCriterion, ConfidenceIntervalCriterion, TimeLimitCriterion,
    StoppingCriteriaManager
)


class TestStoppingCriteria(unittest.TestCase):
    """Test case for stopping criteria classes."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock experiment
        self.experiment = MagicMock(spec=Experiment)
        self.experiment.status = ExperimentStatus.ACTIVE
        self.experiment.id = "test-experiment-1"
        self.experiment.name = "Test Experiment"
        
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
        self.control_metrics.requests = 0
        self.control_metrics.sentiment_accuracy = 0.75
        self.control_metrics.direction_accuracy = 0.70
        self.control_metrics.calibration_error = 0.15
        self.control_metrics.confidence_score = 0.65
        
        self.treatment_metrics = MagicMock(spec=ExperimentMetrics)
        self.treatment_metrics.requests = 0
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

    def test_sample_size_criterion(self):
        """Test SampleSizeCriterion."""
        # Create criterion with 100 samples required
        criterion = SampleSizeCriterion(min_samples_per_variant=100)
        
        # Test with insufficient samples
        self.control_metrics.requests = 50
        self.treatment_metrics.requests = 80
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertFalse(should_stop)
        self.assertIn("Need more samples", reason)
        
        # Test with exactly the required samples
        self.control_metrics.requests = 100
        self.treatment_metrics.requests = 120
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertTrue(should_stop)
        self.assertIn("All variants have at least 100 samples", reason)
        
        # Test with more than required samples
        self.control_metrics.requests = 150
        self.treatment_metrics.requests = 160
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertTrue(should_stop)
        self.assertIn("All variants have at least 100 samples", reason)
    
    @patch("src.analysis_agents.sentiment.continuous_improvement.stopping_criteria.BayesianAnalyzer")
    def test_bayesian_probability_threshold_criterion(self, mock_analyzer_class):
        """Test BayesianProbabilityThresholdCriterion."""
        # Setup mock
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create mock analysis results
        mock_results = MagicMock()
        mock_results.winning_probability = {
            "sentiment_accuracy": {
                "Control": 0.3,
                "Treatment": 0.7
            },
            "direction_accuracy": {
                "Control": 0.25,
                "Treatment": 0.75
            },
            "calibration_error": {
                "Control": 0.2,
                "Treatment": 0.8
            },
            "confidence_score": {
                "Control": 0.35,
                "Treatment": 0.65
            }
        }
        mock_analyzer.analyze_experiment.return_value = mock_results
        
        # Create criterion with 95% threshold
        criterion = BayesianProbabilityThresholdCriterion(
            probability_threshold=0.95,
            min_samples_per_variant=50
        )
        
        # Test with insufficient samples
        self.control_metrics.requests = 30
        self.treatment_metrics.requests = 30
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertFalse(should_stop)
        self.assertIn("Need at least 50 samples per variant", reason)
        
        # Test with sufficient samples but no variant reaches threshold
        self.control_metrics.requests = 100
        self.treatment_metrics.requests = 100
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertFalse(should_stop)
        self.assertIn("No variant has reached the probability threshold", reason)
        
        # Test with a variant reaching threshold
        mock_results.winning_probability = {
            "sentiment_accuracy": {
                "Control": 0.05,
                "Treatment": 0.95
            },
            "direction_accuracy": {
                "Control": 0.02,
                "Treatment": 0.98
            },
            "calibration_error": {
                "Control": 0.03,
                "Treatment": 0.97
            },
            "confidence_score": {
                "Control": 0.04,
                "Treatment": 0.96
            }
        }
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertTrue(should_stop)
        self.assertIn("Treatment has", reason)
        self.assertIn("probability of being best", reason)
    
    @patch("src.analysis_agents.sentiment.continuous_improvement.stopping_criteria.BayesianAnalyzer")
    def test_expected_loss_criterion(self, mock_analyzer_class):
        """Test ExpectedLossCriterion."""
        # Setup mock
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create mock analysis results
        mock_results = MagicMock()
        mock_results.expected_loss = {
            "sentiment_accuracy": {
                "Control": 0.03,
                "Treatment": 0.01
            },
            "direction_accuracy": {
                "Control": 0.04,
                "Treatment": 0.02
            },
            "calibration_error": {
                "Control": 0.05,
                "Treatment": 0.015
            },
            "confidence_score": {
                "Control": 0.03,
                "Treatment": 0.02
            }
        }
        mock_analyzer.analyze_experiment.return_value = mock_results
        
        # Create criterion with 0.005 threshold
        criterion = ExpectedLossCriterion(
            loss_threshold=0.005,
            min_samples_per_variant=50
        )
        
        # Test with insufficient samples
        self.control_metrics.requests = 30
        self.treatment_metrics.requests = 30
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertFalse(should_stop)
        self.assertIn("Need at least 50 samples per variant", reason)
        
        # Test with sufficient samples but no variant below threshold
        self.control_metrics.requests = 100
        self.treatment_metrics.requests = 100
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertFalse(should_stop)
        self.assertIn("Expected loss is above the threshold", reason)
        
        # Test with a variant below threshold
        mock_results.expected_loss = {
            "sentiment_accuracy": {
                "Control": 0.01,
                "Treatment": 0.001
            },
            "direction_accuracy": {
                "Control": 0.015,
                "Treatment": 0.002
            },
            "calibration_error": {
                "Control": 0.02,
                "Treatment": 0.003
            },
            "confidence_score": {
                "Control": 0.01,
                "Treatment": 0.001
            }
        }
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertTrue(should_stop)
        self.assertIn("expected loss", reason)
        self.assertIn("below threshold", reason)
    
    @patch("src.analysis_agents.sentiment.continuous_improvement.stopping_criteria.BayesianAnalyzer")
    def test_confidence_interval_criterion(self, mock_analyzer_class):
        """Test ConfidenceIntervalCriterion."""
        # Setup mock
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Create mock analysis results
        mock_results = MagicMock()
        mock_results.credible_intervals = {
            "sentiment_accuracy": {
                "Control": {
                    "95%": [0.65, 0.85]  # Width: 0.20
                },
                "Treatment": {
                    "95%": [0.75, 0.89]  # Width: 0.14
                }
            },
            "direction_accuracy": {
                "Control": {
                    "95%": [0.60, 0.80]  # Width: 0.20
                },
                "Treatment": {
                    "95%": [0.70, 0.86]  # Width: 0.16
                }
            }
        }
        mock_analyzer.analyze_experiment.return_value = mock_results
        
        # Create criterion with 0.05 threshold
        criterion = ConfidenceIntervalCriterion(
            interval_width_threshold=0.05,
            min_samples_per_variant=50,
            metrics_to_check=["sentiment_accuracy", "direction_accuracy"]
        )
        
        # Test with insufficient samples
        self.control_metrics.requests = 30
        self.treatment_metrics.requests = 30
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertFalse(should_stop)
        self.assertIn("Need at least 50 samples per variant", reason)
        
        # Test with sufficient samples but intervals too wide
        self.control_metrics.requests = 100
        self.treatment_metrics.requests = 100
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertFalse(should_stop)
        self.assertIn("Confidence intervals still too wide", reason)
        
        # Test with narrow enough intervals
        mock_results.credible_intervals = {
            "sentiment_accuracy": {
                "Control": {
                    "95%": [0.73, 0.77]  # Width: 0.04
                },
                "Treatment": {
                    "95%": [0.80, 0.84]  # Width: 0.04
                }
            },
            "direction_accuracy": {
                "Control": {
                    "95%": [0.68, 0.72]  # Width: 0.04
                },
                "Treatment": {
                    "95%": [0.76, 0.80]  # Width: 0.04
                }
            }
        }
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertTrue(should_stop)
        self.assertIn("All confidence intervals are narrower than", reason)
    
    def test_time_limit_criterion(self):
        """Test TimeLimitCriterion."""
        # Create criterion with 7 days limit
        criterion = TimeLimitCriterion(max_days=7)
        
        # Test with experiment running for 5 days (under limit)
        self.experiment.start_time = datetime.utcnow() - timedelta(days=5)
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertFalse(should_stop)
        self.assertIn("Experiment has been running for", reason)
        
        # Test with experiment running for exactly 7 days
        self.experiment.start_time = datetime.utcnow() - timedelta(days=7)
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertTrue(should_stop)
        self.assertIn("Experiment has been running for", reason)
        
        # Test with experiment running for more than 7 days
        self.experiment.start_time = datetime.utcnow() - timedelta(days=10)
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertTrue(should_stop)
        self.assertIn("Experiment has been running for", reason)
        
        # Test with no start time
        self.experiment.start_time = None
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertFalse(should_stop)
        self.assertEqual("Experiment has no start time", reason)
        
        # Test with inactive experiment
        self.experiment.status = ExperimentStatus.COMPLETED
        self.experiment.start_time = datetime.utcnow() - timedelta(days=10)
        should_stop, reason = criterion.should_stop(self.experiment)
        self.assertFalse(should_stop)
        self.assertEqual("Experiment is not active", reason)


class TestStoppingCriteriaManager(unittest.TestCase):
    """Test case for StoppingCriteriaManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock experiment
        self.experiment = MagicMock(spec=Experiment)
        self.experiment.status = ExperimentStatus.ACTIVE
        self.experiment.id = "test-experiment-1"
        self.experiment.name = "Test Experiment"
        
        # Create variants
        self.control_variant = MagicMock(spec=ExperimentVariant)
        self.control_variant.id = "control-variant"
        self.control_variant.name = "Control"
        
        self.treatment_variant = MagicMock(spec=ExperimentVariant)
        self.treatment_variant.id = "treatment-variant"
        self.treatment_variant.name = "Treatment"
        
        self.experiment.variants = [self.control_variant, self.treatment_variant]
        
        # Create metrics
        self.control_metrics = MagicMock(spec=ExperimentMetrics)
        self.control_metrics.requests = 100
        
        self.treatment_metrics = MagicMock(spec=ExperimentMetrics)
        self.treatment_metrics.requests = 100
        
        self.experiment.variant_metrics = {
            self.control_variant.id: self.control_metrics,
            self.treatment_variant.id: self.treatment_metrics
        }
        
        # Create stopping criteria manager
        self.manager = StoppingCriteriaManager()
        self.manager.clear_criteria()  # Clear default criteria

    def test_add_criterion(self):
        """Test adding criteria to manager."""
        # Add a sample size criterion
        criterion = SampleSizeCriterion(min_samples_per_variant=50)
        self.manager.add_criterion(criterion)
        
        # Check that criterion was added
        self.assertEqual(1, len(self.manager.criteria))
        self.assertEqual("sample_size", self.manager.criteria[0].name)

    def test_remove_criterion(self):
        """Test removing criteria from manager."""
        # Add two criteria
        self.manager.add_criterion(SampleSizeCriterion(min_samples_per_variant=50))
        self.manager.add_criterion(TimeLimitCriterion(max_days=7))
        
        # Check both were added
        self.assertEqual(2, len(self.manager.criteria))
        
        # Remove one criterion
        result = self.manager.remove_criterion("sample_size")
        self.assertTrue(result)
        
        # Check it was removed
        self.assertEqual(1, len(self.manager.criteria))
        self.assertEqual("time_limit", self.manager.criteria[0].name)
        
        # Try to remove non-existent criterion
        result = self.manager.remove_criterion("nonexistent")
        self.assertFalse(result)

    def test_clear_criteria(self):
        """Test clearing all criteria."""
        # Add criteria
        self.manager.add_criterion(SampleSizeCriterion(min_samples_per_variant=50))
        self.manager.add_criterion(TimeLimitCriterion(max_days=7))
        
        # Clear criteria
        self.manager.clear_criteria()
        
        # Check they were cleared
        self.assertEqual(0, len(self.manager.criteria))

    def test_evaluate_experiment(self):
        """Test evaluating experiment with multiple criteria."""
        # Add mock criteria that return different results
        criterion1 = MagicMock(spec=StoppingCriterion)
        criterion1.name = "test_criterion_1"
        criterion1.should_stop.return_value = (False, "Not yet")
        
        criterion2 = MagicMock(spec=StoppingCriterion)
        criterion2.name = "test_criterion_2"
        criterion2.should_stop.return_value = (True, "Threshold reached")
        
        self.manager.add_criterion(criterion1)
        self.manager.add_criterion(criterion2)
        
        # Evaluate experiment
        result = self.manager.evaluate_experiment(self.experiment)
        
        # Check the result
        self.assertTrue(result["should_stop"])
        self.assertEqual(2, len(result["criteria_results"]))
        self.assertFalse(result["criteria_results"]["test_criterion_1"]["should_stop"])
        self.assertTrue(result["criteria_results"]["test_criterion_2"]["should_stop"])
        self.assertEqual(1, len(result["stopping_reasons"]))
        self.assertEqual("test_criterion_2", result["stopping_reasons"][0]["criterion"])
        
        # Test with inactive experiment
        self.experiment.status = ExperimentStatus.COMPLETED
        result = self.manager.evaluate_experiment(self.experiment)
        self.assertFalse(result["should_stop"])

    def test_error_handling(self):
        """Test error handling during evaluation."""
        # Add a criterion that raises an exception
        criterion = MagicMock(spec=StoppingCriterion)
        criterion.name = "error_criterion"
        criterion.should_stop.side_effect = Exception("Test error")
        
        self.manager.add_criterion(criterion)
        
        # Evaluate experiment
        result = self.manager.evaluate_experiment(self.experiment)
        
        # Check the result
        self.assertFalse(result["should_stop"])
        self.assertIn("error_criterion", result["criteria_results"])
        self.assertFalse(result["criteria_results"]["error_criterion"]["should_stop"])
        self.assertIn("Error:", result["criteria_results"]["error_criterion"]["reason"])


if __name__ == "__main__":
    unittest.main()