"""
Unit tests for the continuous improvement manager's integration with stopping criteria.

Tests the improvement manager's ability to check experiments, evaluate stopping criteria,
and implement results based on automatic stopping decisions.
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import os
import sys
from datetime import datetime, timedelta

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.analysis_agents.sentiment.ab_testing import (
    ExperimentStatus, ExperimentType, VariantAssignmentStrategy,
    Experiment, ExperimentVariant, ExperimentMetrics
)
from src.analysis_agents.sentiment.continuous_improvement.stopping_criteria import (
    StoppingCriteriaManager, SampleSizeCriterion
)
from src.analysis_agents.sentiment.continuous_improvement.improvement_manager import (
    ContinuousImprovementManager
)
from src.common.events import Event


class TestImprovementManagerStopping(unittest.TestCase):
    """Test case for improvement manager with stopping criteria."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock configuration
        self.config = {
            "enabled": True,
            "experiment_check_interval": 300,
            "generate_experiments": True,
            "auto_implement": True,
            "min_confidence": 0.95,
            "stopping_criteria": {
                "sample_size": {
                    "enabled": True,
                    "min_samples_per_variant": 100
                },
                "bayesian_probability": {
                    "enabled": True,
                    "probability_threshold": 0.95,
                    "min_samples_per_variant": 50
                },
                "expected_loss": {
                    "enabled": False,
                    "loss_threshold": 0.005,
                    "min_samples_per_variant": 30
                },
                "confidence_interval": {
                    "enabled": True,
                    "interval_width_threshold": 0.05,
                    "min_samples_per_variant": 40
                },
                "time_limit": {
                    "enabled": True,
                    "max_days": 7
                }
            }
        }
        
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
        
        # Create mock stopping criteria evaluation result
        self.stopping_evaluation = {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "status": "ACTIVE",
            "should_stop": True,
            "stopping_reasons": [
                {
                    "criterion": "sample_size",
                    "reason": "All variants have at least 100 samples"
                },
                {
                    "criterion": "bayesian_probability",
                    "reason": "Variant 'Treatment' has 95.0% probability of being best (threshold: 95.0%)"
                }
            ],
            "criteria_results": {
                "sample_size": {
                    "should_stop": True,
                    "reason": "All variants have at least 100 samples"
                },
                "bayesian_probability": {
                    "should_stop": True,
                    "reason": "Variant 'Treatment' has 95.0% probability of being best (threshold: 95.0%)"
                },
                "confidence_interval": {
                    "should_stop": False,
                    "reason": "Confidence intervals still too wide (widest: 12.0%, threshold: 5.0%)"
                },
                "time_limit": {
                    "should_stop": False,
                    "reason": "Experiment has been running for 5.0 days (limit: 7 days)"
                }
            }
        }

    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.config")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.ab_testing_framework")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.stopping_criteria_manager")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.event_bus")
    async def test_configure_stopping_criteria(self, mock_event_bus, mock_stopping_manager, mock_ab_framework, mock_config):
        """Test configuring stopping criteria from configuration."""
        # Setup mocks
        mock_config.get.return_value = self.config
        mock_stopping_manager.clear_criteria = MagicMock()
        mock_stopping_manager.add_criterion = MagicMock()
        
        # Create improvement manager
        manager = ContinuousImprovementManager()
        
        # Initialize the manager
        await manager.initialize()
        
        # Verify that stopping criteria were configured
        mock_stopping_manager.clear_criteria.assert_called_once()
        
        # Verify criteria were added based on config
        self.assertEqual(4, mock_stopping_manager.add_criterion.call_count)
        
        # Verify the first call (sample_size)
        criterion_args = mock_stopping_manager.add_criterion.call_args_list[0][0][0]
        self.assertIsInstance(criterion_args, SampleSizeCriterion)
        self.assertEqual(100, criterion_args.min_samples_per_variant)

    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.config")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.ab_testing_framework")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.stopping_criteria_manager")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.event_bus")
    async def test_check_experiments_with_stopping(self, mock_event_bus, mock_stopping_manager, mock_ab_framework, mock_config):
        """Test checking experiments with stopping criteria that indicate stopping."""
        # Setup mocks
        mock_config.get.return_value = self.config
        
        # Configure AB testing framework
        mock_ab_framework.list_experiments.return_value = [
            {"id": self.experiment.id, "name": self.experiment.name, "status": "ACTIVE"}
        ]
        mock_ab_framework.get_experiment.return_value = self.experiment
        mock_ab_framework.complete_experiment = AsyncMock()
        mock_ab_framework.implement_experiment = AsyncMock()
        
        # Configure stopping criteria manager
        mock_stopping_manager.evaluate_experiment.return_value = self.stopping_evaluation
        
        # Configure analyzer
        mock_analyzer = MagicMock()
        mock_results = MagicMock()
        mock_results.has_clear_winner.return_value = True
        mock_results.get_winning_variant.return_value = "Treatment"
        mock_analyzer.analyze_experiment.return_value = mock_results
        
        # Create the improvement manager
        manager = ContinuousImprovementManager()
        manager._analyzer = mock_analyzer
        manager._last_check = datetime.utcnow() - timedelta(minutes=10)
        
        # Initialize the manager
        await manager.initialize()
        
        # Patch dependencies for the check
        with patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.ab_testing_framework", mock_ab_framework):
            with patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.stopping_criteria_manager", mock_stopping_manager):
                # Run the check experiments method
                await manager.check_experiments()
                
                # Verify stopping criteria were evaluated
                mock_stopping_manager.evaluate_experiment.assert_called_once_with(self.experiment)
                
                # Verify experiment was completed
                mock_ab_framework.complete_experiment.assert_called_once_with(self.experiment.id)
                
                # Verify analysis was performed
                mock_analyzer.analyze_experiment.assert_called_once_with(self.experiment)
                
                # Verify event was published
                mock_event_bus.publish.assert_called()
                event_calls = mock_event_bus.publish.call_args_list
                self.assertTrue(any(isinstance(call[0][0], Event) and 
                                   call[0][0].event_type == "experiment.automatically_stopped" 
                                   for call in event_calls))
                
                # Verify implementation
                mock_ab_framework.implement_experiment.assert_called_once()

    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.config")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.ab_testing_framework")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.stopping_criteria_manager")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.event_bus")
    async def test_check_experiments_no_stopping(self, mock_event_bus, mock_stopping_manager, mock_ab_framework, mock_config):
        """Test checking experiments with stopping criteria that don't indicate stopping."""
        # Setup mocks
        mock_config.get.return_value = self.config
        
        # Configure AB testing framework
        mock_ab_framework.list_experiments.return_value = [
            {"id": self.experiment.id, "name": self.experiment.name, "status": "ACTIVE"}
        ]
        mock_ab_framework.get_experiment.return_value = self.experiment
        mock_ab_framework.complete_experiment = AsyncMock()
        
        # Configure stopping criteria manager - experiment should continue
        stopping_evaluation = dict(self.stopping_evaluation)
        stopping_evaluation["should_stop"] = False
        stopping_evaluation["stopping_reasons"] = []
        for key in stopping_evaluation["criteria_results"]:
            stopping_evaluation["criteria_results"][key]["should_stop"] = False
        
        mock_stopping_manager.evaluate_experiment.return_value = stopping_evaluation
        
        # Create the improvement manager
        manager = ContinuousImprovementManager()
        manager._last_check = datetime.utcnow() - timedelta(minutes=10)
        
        # Initialize the manager
        await manager.initialize()
        
        # Patch dependencies for the check
        with patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.ab_testing_framework", mock_ab_framework):
            with patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.stopping_criteria_manager", mock_stopping_manager):
                # Run the check experiments method
                await manager.check_experiments()
                
                # Verify stopping criteria were evaluated
                mock_stopping_manager.evaluate_experiment.assert_called_once_with(self.experiment)
                
                # Verify experiment was NOT completed
                mock_ab_framework.complete_experiment.assert_not_called()
                
                # Verify no events about stopping
                for call in mock_event_bus.publish.call_args_list:
                    if isinstance(call[0][0], Event):
                        self.assertNotEqual("experiment.automatically_stopped", call[0][0].event_type)

    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.config")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.ab_testing_framework")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.stopping_criteria_manager")
    @patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.event_bus")
    async def test_stopping_with_auto_implement_disabled(self, mock_event_bus, mock_stopping_manager, mock_ab_framework, mock_config):
        """Test stopping with auto-implement disabled."""
        # Modify config to disable auto-implement
        config_no_auto = dict(self.config)
        config_no_auto["auto_implement"] = False
        
        # Setup mocks
        mock_config.get.return_value = config_no_auto
        
        # Configure AB testing framework
        mock_ab_framework.list_experiments.return_value = [
            {"id": self.experiment.id, "name": self.experiment.name, "status": "ACTIVE"}
        ]
        mock_ab_framework.get_experiment.return_value = self.experiment
        mock_ab_framework.complete_experiment = AsyncMock()
        mock_ab_framework.implement_experiment = AsyncMock()
        
        # Configure stopping criteria manager
        mock_stopping_manager.evaluate_experiment.return_value = self.stopping_evaluation
        
        # Configure analyzer
        mock_analyzer = MagicMock()
        mock_results = MagicMock()
        mock_results.has_clear_winner.return_value = True
        mock_results.get_winning_variant.return_value = "Treatment"
        mock_analyzer.analyze_experiment.return_value = mock_results
        
        # Create the improvement manager
        manager = ContinuousImprovementManager()
        manager._analyzer = mock_analyzer
        manager._last_check = datetime.utcnow() - timedelta(minutes=10)
        
        # Initialize the manager
        await manager.initialize()
        
        # Patch dependencies for the check
        with patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.ab_testing_framework", mock_ab_framework):
            with patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.stopping_criteria_manager", mock_stopping_manager):
                # Run the check experiments method
                await manager.check_experiments()
                
                # Verify experiment was completed
                mock_ab_framework.complete_experiment.assert_called_once_with(self.experiment.id)
                
                # Verify analysis was performed
                mock_analyzer.analyze_experiment.assert_called_once_with(self.experiment)
                
                # Verify implementation was NOT called
                mock_ab_framework.implement_experiment.assert_not_called()


if __name__ == "__main__":
    unittest.main()