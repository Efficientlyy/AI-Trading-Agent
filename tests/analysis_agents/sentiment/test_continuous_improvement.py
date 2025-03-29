"""
Tests for the Continuous Improvement Manager.

This module tests the core functionality of the continuous improvement system 
for the sentiment analysis pipeline.
"""

import asyncio
import json
import os
import pytest
import unittest.mock as mock
from datetime import datetime, timedelta

from src.analysis_agents.sentiment.continuous_improvement.improvement_manager import (
    ContinuousImprovementManager, continuous_improvement_manager
)
from src.analysis_agents.sentiment.ab_testing import (
    ab_testing_framework, ExperimentType, ExperimentStatus, Experiment, ExperimentVariant
)
from src.analysis_agents.sentiment.performance_tracker import performance_tracker
from src.common.events import Event


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    with mock.patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.config") as mock_config:
        # Set default configuration values for testing
        mock_config.get.side_effect = lambda key, default=None: {
            "sentiment_analysis.continuous_improvement.enabled": True,
            "sentiment_analysis.continuous_improvement.check_interval": 60,
            "sentiment_analysis.continuous_improvement.experiment_generation_interval": 120,
            "sentiment_analysis.continuous_improvement.max_concurrent_experiments": 3,
            "sentiment_analysis.continuous_improvement.auto_implement": True,
            "sentiment_analysis.continuous_improvement.significance_threshold": 0.95,
            "sentiment_analysis.continuous_improvement.improvement_threshold": 0.05,
            "sentiment_analysis.continuous_improvement.results_history_file": "data/test_continuous_improvement_history.json",
        }.get(key, default)
        yield mock_config


@pytest.fixture
def mock_ab_testing(monkeypatch):
    """Mock A/B testing framework for tests."""
    with mock.patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.ab_testing_framework") as mock_ab:
        # Mock the framework's methods
        mock_ab.get_active_experiments_by_type.return_value = []
        mock_ab.create_experiment.return_value = Experiment(
            id="test_exp_001",
            name="Test Experiment",
            description="Test description",
            experiment_type=ExperimentType.PROMPT_TEMPLATE,
            variants=[
                ExperimentVariant(
                    id="control_variant",
                    name="Control Variant",
                    description="Control variant description",
                    control=True,
                    config={"template": "Original template"}
                ),
                ExperimentVariant(
                    id="test_variant",
                    name="Test Variant",
                    description="Test variant description",
                    config={"template": "Improved template"}
                )
            ]
        )
        yield mock_ab


@pytest.fixture
def mock_event_bus(monkeypatch):
    """Mock event bus for tests."""
    with mock.patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.event_bus") as mock_bus:
        yield mock_bus


@pytest.fixture
def mock_performance_tracker(monkeypatch):
    """Mock performance tracker for tests."""
    with mock.patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.performance_tracker") as mock_tracker:
        mock_tracker.get_recent_metrics.return_value = {
            "sentiment_accuracy": 0.75,
            "direction_accuracy": 0.65,
            "confidence_score": 0.6,
            "calibration_error": 0.12,
            "success_rate": 0.95,
            "average_latency": 450,
            "by_source": {
                "news": {"sentiment_accuracy": 0.65},
                "social_media": {"sentiment_accuracy": 0.70},
                "market": {"sentiment_accuracy": 0.85}
            },
            "by_market_condition": {
                "bullish": {"sentiment_accuracy": 0.80},
                "bearish": {"sentiment_accuracy": 0.60}
            }
        }
        yield mock_tracker


@pytest.fixture
def mock_prompt_tuning(monkeypatch):
    """Mock prompt tuning system for tests."""
    with mock.patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.prompt_tuning_system") as mock_prompt:
        mock_prompt.get_current_templates.return_value = {
            "sentiment_analysis": "Original template text for sentiment analysis"
        }
        yield mock_prompt


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for tests."""
    with mock.patch("src.analysis_agents.sentiment.continuous_improvement.improvement_manager.LLMService") as mock_llm:
        # Mock instance methods
        instance = mock_llm.return_value
        instance.initialize = mock.AsyncMock()
        instance.analyze_sentiment = mock.AsyncMock(return_value={
            "explanation": "Improved template text"
        })
        instance.close = mock.AsyncMock()
        yield mock_llm


@pytest.fixture
def test_manager(mock_config, mock_event_bus, mock_ab_testing, mock_performance_tracker, mock_prompt_tuning, mock_llm_service):
    """Create a test instance of ContinuousImprovementManager."""
    manager = ContinuousImprovementManager()
    # Set a different results file path for testing
    manager.results_history_file = "data/test_continuous_improvement_history.json"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(manager.results_history_file), exist_ok=True)
    
    # Mock internal methods to avoid side effects
    manager._save_results_history = mock.MagicMock()
    
    return manager


class TestContinuousImprovementManager:
    """Tests for the ContinuousImprovementManager class."""

    @pytest.mark.asyncio
    async def test_initialization(self, test_manager):
        """Test manager initialization."""
        test_manager.initialize()
        assert test_manager.enabled == True
        assert test_manager.check_interval == 60
        assert test_manager.auto_implement == True

    @pytest.mark.asyncio
    async def test_identify_improvement_opportunities(self, test_manager):
        """Test identification of improvement opportunities."""
        metrics = {
            "sentiment_accuracy": 0.75,
            "direction_accuracy": 0.65,
            "confidence_score": 0.6,
            "calibration_error": 0.12,
            "success_rate": 0.95,
            "average_latency": 450,
            "by_source": {
                "news": {"sentiment_accuracy": 0.65},
                "social_media": {"sentiment_accuracy": 0.70},
                "market": {"sentiment_accuracy": 0.85}
            },
            "by_market_condition": {
                "bullish": {"sentiment_accuracy": 0.80},
                "bearish": {"sentiment_accuracy": 0.60}
            }
        }

        opportunities = test_manager._identify_improvement_opportunities(metrics)
        
        # There should be multiple opportunities identified
        assert len(opportunities) > 0
        
        # Check for specific opportunities
        prompt_opportunity = next((o for o in opportunities if o["type"] = = ExperimentType.PROMPT_TEMPLATE), None)
        assert prompt_opportunity is not None
        assert "sentiment_accuracy" in prompt_opportunity["metrics"]
        
        model_opportunity = next((o for o in opportunities if o["type"] = = ExperimentType.MODEL_SELECTION), None)
        assert model_opportunity is not None
        assert "calibration_error" in model_opportunity["metrics"]
        
        # Check for source variance opportunity
        context_opportunity = next((o for o in opportunities if o["type"] = = ExperimentType.CONTEXT_STRATEGY), None)
        assert context_opportunity is not None
        assert "source_variances" in context_opportunity["metrics"]
        
        # Check for market condition opportunity
        aggregation_opportunity = next((o for o in opportunities if o["type"] = = ExperimentType.AGGREGATION_WEIGHTS), None)
        assert aggregation_opportunity is not None
        assert "condition_variances" in aggregation_opportunity["metrics"]

    @pytest.mark.asyncio
    async def test_generate_prompt_template_variants(self, test_manager, mock_prompt_tuning, mock_llm_service):
        """Test generation of prompt template variants."""
        variants = test_manager._generate_prompt_template_variants()
        
        # Should have two variants: control and treatment
        assert len(variants) == 2
        
        # Check control variant
        assert variants[0]["name"] == "Current Template"
        assert variants[0]["control"] == True
        assert "template" in variants[0]["config"]
        
        # Check treatment variant
        assert variants[1]["name"] == "AI-Enhanced Template"
        assert variants[1]["control"] == False
        assert "template" in variants[1]["config"]
        
        # Template should be different between variants
        assert variants[0]["config"]["template"] != variants[1]["config"]["template"]

    @pytest.mark.asyncio
    async def test_generate_experiments(self, test_manager, mock_ab_testing, mock_performance_tracker):
        """Test experiment generation."""
        # Mock active experiments to ensure we can generate new ones
        mock_ab_testing.get_active_experiments_by_type.return_value = []
        
        test_manager.generate_experiments()
        
        # Check that create_experiment was called
        assert mock_ab_testing.create_experiment.called
        
        # Check that start_experiment was called
        assert mock_ab_testing.start_experiment.called
        
        # Check that improvement history was updated
        assert len(test_manager.improvement_history) > 0
        assert test_manager.improvement_history[0]["action"] == "experiment_generated"

    @pytest.mark.asyncio
    async def test_implement_experiment(self, test_manager, mock_ab_testing, mock_prompt_tuning):
        """Test implementation of a successful experiment."""
        # Create a mock experiment with results
        experiment = Experiment(
            id="test_exp_001",
            name="Test Experiment",
            description="Test description",
            experiment_type=ExperimentType.PROMPT_TEMPLATE,
            variants=[
                ExperimentVariant(
                    id="control_variant",
                    name="Control Variant",
                    description="Control variant description",
                    control=True,
                    config={"template": "Original template"}
                ),
                ExperimentVariant(
                    id="test_variant",
                    name="Test Variant",
                    description="Test variant description",
                    config={"template": "Improved template"}
                )
            ],
            status=ExperimentStatus.ANALYZED
        )
        
        # Add results to the experiment
        experiment.results = {
            "has_clear_winner": True,
            "winning_variant": "Test Variant",
            "metrics_improvement": {
                "sentiment_accuracy": 0.05,
                "direction_accuracy": 0.07
            }
        }
        
        # Mock the implementation function
        test_manager._implement_prompt_template = mock.AsyncMock()
        
        # Implement the experiment
        await test_manager._implement_experiment(experiment)
        
        # Check that the implementation function was called
        assert test_manager._implement_prompt_template.called
        
        # Check that improvement history was updated
        assert len(test_manager.improvement_history) > 0
        last_entry = test_manager.improvement_history[-1]
        assert last_entry["action"] = = "improvement_implemented"
        assert last_entry["details"]["experiment_id"] == "test_exp_001"
        assert last_entry["details"]["winning_variant"] == "Test Variant"

    @pytest.mark.asyncio
    async def test_update_config(self, test_manager, mock_config):
        """Test configuration updates."""
        updates = {
            "llm.temperature": 0.2,
            "sentiment_analysis.confidence_threshold": 0.75
        }
        
        test_manager._update_config(updates)
        
        # Check that config.set was called for each update
        assert mock_config.set.call_count == 2
        mock_config.set.assert_any_call("llm.temperature", 0.2)
        mock_config.set.assert_any_call("sentiment_analysis.confidence_threshold", 0.75)
        
        # Check that config.save was called
        assert mock_config.save.called

    @pytest.mark.asyncio
    async def test_run_maintenance(self, test_manager):
        """Test maintenance task execution."""
        # Set last experiment generation to a time that would trigger generation
        test_manager.last_experiment_generation = datetime.utcnow() - timedelta(days=1)
        
        # Mock generate_experiments and check_experiments
        test_manager.generate_experiments = mock.AsyncMock()
        test_manager.check_experiments = mock.AsyncMock()
        
        test_manager.run_maintenance()
        
        # Check that maintenance tasks were called
        assert test_manager.generate_experiments.called
        assert test_manager.check_experiments.called
        
        # Check that timestamps were updated
        assert test_manager.last_experiment_generation > datetime.utcnow() - timedelta(minutes=1)
        assert test_manager.last_check > datetime.utcnow() - timedelta(minutes=1)

    @pytest.mark.asyncio
    async def test_handle_experiment_analyzed(self, test_manager, mock_ab_testing):
        """Test handling of experiment analyzed events."""
        # Mock _implement_experiment
        test_manager._implement_experiment = mock.AsyncMock()
        
        # Create a mock experiment with clear winner
        experiment = Experiment(
            id="test_exp_002",
            name="Test Experiment 2",
            description="Test description",
            experiment_type=ExperimentType.TEMPERATURE,
            variants=[],
            status=ExperimentStatus.ANALYZED
        )
        experiment.results = {"has_clear_winner": True, "winning_variant": "Test Variant"}
        
        # Mock get_experiment to return our experiment
        mock_ab_testing.get_experiment.return_value = experiment
        
        # Create event with experiment ID
        event = Event(
            event_type="experiment_analyzed",
            data={"experiment_id": "test_exp_002"}
        )
        
        # Handle the event
        await test_manager.handle_experiment_analyzed(event)
        
        # Check that _implement_experiment was called with the right experiment
        test_manager._implement_experiment.assert_called_once_with(experiment)


class TestContinuousImprovementIntegration:
    """Integration tests for the Continuous Improvement System."""

    @pytest.mark.asyncio
    async def test_full_improvement_cycle(self, test_manager, mock_ab_testing, mock_performance_tracker, mock_prompt_tuning):
        """Test a full improvement cycle from opportunity identification to implementation."""
        # Initialize the manager
        test_manager.initialize()
        
        # 1. Generate experiments based on identified opportunities
        test_manager.generate_experiments()
        
        # Verify experiment was created and started
        assert mock_ab_testing.create_experiment.called
        assert mock_ab_testing.start_experiment.called
        
        # Get the created experiment ID from history
        experiment_id = test_manager.improvement_history[0]["experiment_id"]
        
        # 2. Create a mock analyzed experiment with a clear winner
        experiment = Experiment(
            id=experiment_id,
            name="Auto Experiment",
            description="Automatically generated experiment",
            experiment_type=ExperimentType.PROMPT_TEMPLATE,
            variants=[
                ExperimentVariant(
                    id="control_variant",
                    name="Current Template",
                    description="Control variant",
                    control=True,
                    config={"template": "Original template"}
                ),
                ExperimentVariant(
                    id="test_variant",
                    name="Improved Template",
                    description="Improved variant",
                    config={"template": "Improved template"}
                )
            ],
            status=ExperimentStatus.ANALYZED
        )
        
        # Add results to the experiment
        experiment.results = {
            "has_clear_winner": True,
            "winning_variant": "Improved Template",
            "metrics_improvement": {
                "sentiment_accuracy": 0.08,
                "direction_accuracy": 0.06
            }
        }
        
        # Mock get_experiment to return our experiment
        mock_ab_testing.get_experiment.return_value = experiment
        
        # 3. Check experiments and implement if needed
        test_manager.check_experiments()
        
        # Verify implementation
        assert len(test_manager.improvement_history) >= 2
        assert any(entry["action"] = = "improvement_implemented" for entry in test_manager.improvement_history)
        
        # 4. Verify that results history was updated
        assert len(test_manager.results_history) > 0
        last_result = test_manager.results_history[-1]
        assert last_result["experiment_id"] = = experiment_id
        assert last_result["winning_variant"] = = "Improved Template"