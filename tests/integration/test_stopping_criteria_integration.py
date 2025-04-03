"""
Integration tests for the automatic stopping criteria system.

Tests the interaction between stopping criteria, the continuous improvement manager,
AB testing framework, and Bayesian analysis tools.
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock
import os
import sys
from datetime import datetime, timedelta

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis_agents.sentiment.ab_testing import (
    ExperimentStatus, ExperimentType, VariantAssignmentStrategy,
    Experiment, ExperimentVariant, ExperimentMetrics
)
from src.analysis_agents.sentiment.continuous_improvement.stopping_criteria import (
    StoppingCriterion, SampleSizeCriterion, BayesianProbabilityThresholdCriterion,
    ExpectedLossCriterion, ConfidenceIntervalCriterion, TimeLimitCriterion,
    StoppingCriteriaManager, stopping_criteria_manager
)
from src.analysis_agents.sentiment.continuous_improvement.improvement_manager import (
    ContinuousImprovementManager, continuous_improvement_manager
)
from src.analysis_agents.sentiment.continuous_improvement.bayesian_analysis import (
    BayesianAnalyzer, BayesianAnalysisResults
)


class TestStoppingCriteriaIntegration(unittest.TestCase):
    """Integration tests for the stopping criteria system."""

    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        # Silence loggers
        # import logging
        # logging.getLogger('analysis_agents').setLevel(logging.ERROR)

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock AB testing framework
        self.ab_testing_framework = MagicMock()
        self.improvement_manager = MagicMock(spec=ContinuousImprovementManager)
        
        # Create a test experiment
        self.experiment = MagicMock(spec=Experiment)
        self.experiment.id = "test-experiment-1"
        self.experiment.name = "Test Experiment"
        self.experiment.status = ExperimentStatus.ACTIVE
        self.experiment.experiment_type = ExperimentType.PROMPT_TEMPLATE
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

    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.ab_testing_framework")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.stopping_criteria_manager")
    async def test_check_experiments_with_stopping_criteria(self, mock_stopping_manager, mock_ab_framework):
        """Test that improvement manager checks stopping criteria for experiments."""
        # Setup mocks
        mock_ab_framework.list_experiments.return_value = [
            {"id": self.experiment.id, "name": self.experiment.name, "status": "ACTIVE"}
        ]
        mock_ab_framework.get_experiment.return_value = self.experiment
        
        # Configure stopping criteria evaluation
        mock_stopping_manager.evaluate_experiment.return_value = {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "status": "ACTIVE",
            "should_stop": True,
            "stopping_reasons": [
                {
                    "criterion": "bayesian_probability",
                    "reason": "Variant 'Treatment' has 95.0% probability of being best (threshold: 90.0%)"
                }
            ],
            "criteria_results": {
                "bayesian_probability": {
                    "should_stop": True,
                    "reason": "Variant 'Treatment' has 95.0% probability of being best (threshold: 90.0%)"
                }
            }
        }
        
        # Configure analysis
        mock_analyzer = MagicMock()
        mock_results = MagicMock()
        mock_results.has_clear_winner.return_value = True
        mock_results.get_winning_variant.return_value = "Treatment"
        mock_analyzer.analyze_experiment.return_value = mock_results
        
        # Create the improvement manager
        manager = ContinuousImprovementManager()
        manager._analyzer = mock_analyzer
        
        # Patch dependencies
        with patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.ab_testing_framework", mock_ab_framework):
            with patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.stopping_criteria_manager", mock_stopping_manager):
                # Run the check experiments method
                manager.check_experiments()
                
                # Verify that stopping criteria were evaluated
                mock_stopping_manager.evaluate_experiment.assert_called_once_with(self.experiment)
                
                # Verify that the experiment was completed and analyzed
                mock_ab_framework.complete_experiment.assert_called_once_with(self.experiment.id)
                mock_analyzer.analyze_experiment.assert_called_once_with(self.experiment)
                
                # Verify implementation check
                mock_ab_framework.implement_experiment.assert_called_once()

    def test_stopping_criteria_integration(self):
        """Test that all stopping criteria work together properly."""
        # Setup criteria manager with custom criteria
        manager = StoppingCriteriaManager()
        manager.clear_criteria()
        
        # Add criteria with different thresholds
        manager.add_criterion(SampleSizeCriterion(min_samples_per_variant=100))
        manager.add_criterion(BayesianProbabilityThresholdCriterion(
            probability_threshold=0.90,
            min_samples_per_variant=100
        ))
        manager.add_criterion(ExpectedLossCriterion(
            loss_threshold=0.01,
            min_samples_per_variant=100
        ))
        manager.add_criterion(ConfidenceIntervalCriterion(
            interval_width_threshold=0.10,
            min_samples_per_variant=100
        ))
        manager.add_criterion(TimeLimitCriterion(max_days=10))
        
        # Mock the Bayesian analyzer
        with patch("src.analysis_agents.sentiment.continuous_improvement.stopping_criteria.BayesianAnalyzer") as mock_class:
            mock_analyzer = MagicMock()
            mock_class.return_value = mock_analyzer
            
            # Create mock results
            mock_results = MagicMock()
            
            # Setup winning probability (above threshold)
            mock_results.winning_probability = {
                "sentiment_accuracy": {
                    "Control": 0.05,
                    "Treatment": 0.95
                },
                "direction_accuracy": {
                    "Control": 0.08,
                    "Treatment": 0.92
                }
            }
            
            # Setup expected loss (still above threshold)
            mock_results.expected_loss = {
                "sentiment_accuracy": {
                    "Control": 0.05,
                    "Treatment": 0.02
                },
                "direction_accuracy": {
                    "Control": 0.04,
                    "Treatment": 0.02
                }
            }
            
            # Setup credible intervals (within threshold)
            mock_results.credible_intervals = {
                "sentiment_accuracy": {
                    "Control": {
                        "95%": [0.70, 0.80]  # Width: 0.10
                    },
                    "Treatment": {
                        "95%": [0.77, 0.87]  # Width: 0.10
                    }
                },
                "direction_accuracy": {
                    "Control": {
                        "95%": [0.65, 0.75]  # Width: 0.10
                    },
                    "Treatment": {
                        "95%": [0.73, 0.83]  # Width: 0.10
                    }
                }
            }
            
            mock_analyzer.analyze_experiment.return_value = mock_results
            
            # Evaluate experiment
            result = manager.evaluate_experiment(self.experiment)
            
            # Verify results
            self.assertTrue(result["should_stop"])
            self.assertEqual(2, len(result["stopping_reasons"]))
            
            # Verify each criterion was evaluated correctly
            self.assertTrue(result["criteria_results"]["sample_size"]["should_stop"])
            self.assertTrue(result["criteria_results"]["bayesian_probability"]["should_stop"])
            self.assertFalse(result["criteria_results"]["expected_loss"]["should_stop"])
            self.assertTrue(result["criteria_results"]["confidence_interval"]["should_stop"])
            self.assertFalse(result["criteria_results"]["time_limit"]["should_stop"])


if __name__ == "__main__":
    unittest.main()