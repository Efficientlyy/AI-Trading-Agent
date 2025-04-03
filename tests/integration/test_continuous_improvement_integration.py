"""
Integration tests for the Continuous Improvement System.

This module provides end-to-end tests for the continuous improvement system,
verifying that all components work together correctly.
"""

import asyncio
import pytest
import unittest.mock as mock
from datetime import datetime, timedelta
import json

from src.analysis_agents.sentiment.continuous_improvement import continuous_improvement_manager
from src.analysis_agents.sentiment.ab_testing import (
    ab_testing_framework, ExperimentType, ExperimentStatus,
    TargetingCriteria, VariantAssignmentStrategy
)
from src.analysis_agents.sentiment.performance_tracker import performance_tracker
from src.analysis_agents.sentiment.llm_service import LLMService
from src.dashboard.components.continuous_improvement_dashboard import (
    create_layout, update_improvement_history, update_experiment_status, 
    create_visualization, get_improvement_metrics
)


@pytest.fixture
def setup_test_environment():
    """Set up a test environment for integration tests."""
    # Store original methods to restore later
    original_methods = {
        'update_config': continuous_improvement_manager._update_config,
        'ab_create_experiment': ab_testing_framework.create_experiment,
        'ab_start_experiment': ab_testing_framework.start_experiment,
        'ab_complete_experiment': ab_testing_framework.complete_experiment,
    }
    
    # Mock methods that require external resources
    continuous_improvement_manager._update_config = mock.MagicMock()
    ab_testing_framework.create_experiment = mock.MagicMock()
    ab_testing_framework.start_experiment = mock.MagicMock()
    ab_testing_framework.complete_experiment = mock.MagicMock()
    
    # Set up test data
    continuous_improvement_manager.enabled = True
    continuous_improvement_manager.auto_implement = True
    continuous_improvement_manager.results_history = []
    
    yield
    
    # Restore original methods
    continuous_improvement_manager._update_config = original_methods['update_config']
    ab_testing_framework.create_experiment = original_methods['ab_create_experiment']
    ab_testing_framework.start_experiment = original_methods['ab_start_experiment']
    ab_testing_framework.complete_experiment = original_methods['ab_complete_experiment']


class TestContinuousImprovementIntegration:
    """Integration tests for the continuous improvement system."""

    @pytest.mark.asyncio
    async def test_full_improvement_cycle(self, setup_test_environment):
        """Test a full improvement cycle from opportunity detection through implementation."""
        # Mock performance metrics
        mock_metrics = {
            "sentiment_accuracy": 0.78,
            "direction_accuracy": 0.72,
            "confidence_score": 0.65,
            "calibration_error": 0.15,
            "success_rate": 0.92,
            "average_latency": 500,
            "by_source": {
                "social_media": {"sentiment_accuracy": 0.75},
                "news": {"sentiment_accuracy": 0.85}
            },
            "by_market_condition": {
                "bullish": {"sentiment_accuracy": 0.85},
                "bearish": {"sentiment_accuracy": 0.75}
            }
        }
        
        # Mock performance tracker to return our test metrics
        with mock.patch.object(performance_tracker, 'get_recent_metrics', 
                              return_value=mock_metrics):
            # Mock A/B testing framework to handle our test experiment
            with mock.patch.object(ab_testing_framework, 'get_experiment') as mock_get_experiment:
                # Create a mock experiment
                mock_experiment = mock.MagicMock()
                mock_experiment.id = "test_exp_001"
                mock_experiment.name = "Test Prompt Template Experiment"
                mock_experiment.experiment_type = ExperimentType.PROMPT_TEMPLATE
                mock_experiment.status = ExperimentStatus.ANALYZED
                mock_experiment.variants = [
                    mock.MagicMock(name="Control Variant", id="control_variant"),
                    mock.MagicMock(name="Enhanced Template", id="enhanced_variant")
                ]
                mock_experiment.metadata = {"auto_generated": True}
                
                # Set up results showing that the experiment has a clear winner
                mock_experiment.results = {
                    "has_clear_winner": True,
                    "winning_variant": "Enhanced Template",
                    "metrics_improvement": {
                        "sentiment_accuracy": 0.08,
                        "direction_accuracy": 0.06
                    }
                }
                
                # Mock the get_experiment to return our test experiment
                mock_get_experiment.return_value = mock_experiment
                
                # Mock the list_experiments to include our test experiment
                ab_testing_framework.list_experiments = mock.MagicMock(
                    return_value=[{
                        "id": "test_exp_001",
                        "name": "Test Prompt Template Experiment",
                        "status": "analyzed",
                        "type": "prompt_template",
                        "auto_generated": True
                    }]
                )
                
                # Set active experiment IDs to include our test experiment
                ab_testing_framework.active_experiment_ids = ["test_exp_001"]
                
                # Step 1: Generate experiments from opportunities
                await continuous_improvement_manager.generate_experiments()
                
                # Step 2: Check and implement experiments
                await continuous_improvement_manager.check_experiments()
                
                # Step 3: Verify that the implementation was added to history
                assert len(continuous_improvement_manager.results_history) > 0
                
                # Step 4: Test dashboard components with new data
                history_data = update_improvement_history()
                assert history_data is not None
                assert len(history_data.get("data", [])) > 0
                
                exp_status = update_experiment_status()
                assert exp_status is not None
                
                visualizations = create_visualization()
                assert visualizations is not None
                
                metrics = get_improvement_metrics()
                assert metrics is not None
                assert metrics.get("improvement_count") > 0
                
                # Step 5: Verify that the dashboard layout can be created
                layout = create_layout()
                assert layout is not None
                
                # Verify that the auto-implementation occurred
                assert mock_experiment.implement.called

    @pytest.mark.asyncio
    async def test_multiple_experiment_types(self, setup_test_environment):
        """Test handling of multiple experiment types simultaneously."""
        # Create different types of opportunities and experiments
        opportunities = [
            {
                "type": ExperimentType.PROMPT_TEMPLATE,
                "reason": "Low sentiment accuracy",
                "metrics": {"sentiment_accuracy": 0.75},
                "potential_impact": 0.3
            },
            {
                "type": ExperimentType.MODEL_SELECTION,
                "reason": "High calibration error",
                "metrics": {"calibration_error": 0.15},
                "potential_impact": 0.2
            },
            {
                "type": ExperimentType.TEMPERATURE,
                "reason": "Confidence calibration issues",
                "metrics": {"confidence_score": 0.65},
                "potential_impact": 0.1
            }
        ]
        
        # Mock the opportunity identification method
        with mock.patch.object(continuous_improvement_manager, 
                             '_identify_improvement_opportunities',
                             return_value=opportunities):
            
            # Mock experiment creation
            with mock.patch.object(continuous_improvement_manager, 
                                  '_create_experiment_from_opportunity') as mock_create_exp:
                # Create mock experiments for each opportunity
                experiments = []
                for i, opp in enumerate(opportunities):
                    mock_exp = mock.MagicMock()
                    mock_exp.id = f"test_exp_00{i+1}"
                    mock_exp.name = f"Test {opp['type'].value} Experiment"
                    mock_exp.experiment_type = opp['type']
                    mock_exp.status = ExperimentStatus.DRAFT
                    experiments.append(mock_exp)
                
                # Set up the mock to return a different experiment for each opportunity
                mock_create_exp.side_effect = experiments
                
                # Generate experiments
                await continuous_improvement_manager.generate_experiments()
                
                # Verify that experiments were created for each opportunity
                assert mock_create_exp.call_count == len(opportunities)
                assert ab_testing_framework.start_experiment.call_count == len(opportunities)

    @pytest.mark.asyncio
    async def test_dashboard_data_integration(self, setup_test_environment):
        """Test that dashboard components receive and process data correctly."""
        # Add some test history data
        continuous_improvement_manager.results_history = [
            {
                "timestamp": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                "experiment_id": "test_exp_001",
                "experiment_name": "Test Prompt Template Experiment",
                "experiment_type": "prompt_template",
                "winning_variant": "Enhanced Template",
                "variant_config": {"template": "Test template content"},
                "metrics_improvement": {
                    "sentiment_accuracy": 0.08,
                    "direction_accuracy": 0.06
                }
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "experiment_id": "test_exp_002",
                "experiment_name": "Test Model Selection Experiment",
                "experiment_type": "model_selection",
                "winning_variant": "Alternative Model",
                "variant_config": {"model": "different-model"},
                "metrics_improvement": {
                    "sentiment_accuracy": 0.05,
                    "calibration_error": -0.03
                }
            }
        ]
        
        # Add improvement history events
        continuous_improvement_manager.improvement_history = [
            {
                "timestamp": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                "action": "experiment_generated",
                "experiment_id": "test_exp_001",
                "experiment_name": "Test Prompt Template Experiment",
                "opportunity": {
                    "type": "PROMPT_TEMPLATE",
                    "reason": "Low sentiment accuracy"
                }
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(days=4)).isoformat(),
                "action": "improvement_implemented",
                "details": continuous_improvement_manager.results_history[0]
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                "action": "experiment_generated",
                "experiment_id": "test_exp_002",
                "experiment_name": "Test Model Selection Experiment",
                "opportunity": {
                    "type": "MODEL_SELECTION",
                    "reason": "High calibration error"
                }
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "action": "improvement_implemented",
                "details": continuous_improvement_manager.results_history[1]
            }
        ]
        
        # Mock the active experiments
        with mock.patch.object(ab_testing_framework, 'active_experiment_ids', ["test_exp_003"]):
            # Mock a currently active experiment
            with mock.patch.object(ab_testing_framework, 'get_experiment') as mock_get_exp:
                mock_exp = mock.MagicMock()
                mock_exp.id = "test_exp_003"
                mock_exp.name = "Test Temperature Experiment"
                mock_exp.experiment_type = ExperimentType.TEMPERATURE
                mock_exp.status = ExperimentStatus.ACTIVE
                mock_exp.metadata = {"auto_generated": True}
                mock_exp.variants = [
                    mock.MagicMock(id="control_variant", name="Current Temperature"),
                    mock.MagicMock(id="test_variant", name="Alternative Temperature")
                ]
                mock_exp.variant_metrics = {
                    "control_variant": mock.MagicMock(requests=50),
                    "test_variant": mock.MagicMock(requests=50)
                }
                
                mock_get_exp.return_value = mock_exp
                
                # Test dashboard data components
                history_data = update_improvement_history()
                assert len(history_data.get("data", [])) == 4  # All history events
                
                status_data = update_experiment_status()
                assert len(status_data.get("data", [])) == 1  # One active experiment
                
                # Test visualizations 
                visualizations = create_visualization()
                assert len(visualizations) >= 2  # At least timeline and metrics
                
                # Test metrics calculations
                metrics = get_improvement_metrics()
                assert metrics.get("improvement_count") == 2
                assert metrics.get("experiments_generated") == 2
                assert metrics.get("implementations") == 2
                assert metrics.get("avg_sentiment_improvement") > 0
                
                # Test status data
                status = continuous_improvement_manager.get_status()
                assert status.get("improvements_count") == 2
                assert status.get("active_experiments") == 1