"""
Tests for the Continuous Improvement Demo script.

This module tests the demonstration script for the continuous improvement system.
"""

import asyncio
import pytest
import unittest.mock as mock

from examples.continuous_improvement_demo import (
    setup_demo, demonstrate_auto_improvement, generate_opportunities,
    simulate_sentiment_analysis, trigger_maintenance, main
)


@pytest.fixture
def mock_continuous_improvement_manager():
    """Mock for the continuous improvement manager."""
    with mock.patch("examples.continuous_improvement_demo.continuous_improvement_manager") as mock_manager:
        # Mock methods
        mock_manager.initialize = mock.AsyncMock()
        mock_manager.generate_experiments = mock.AsyncMock()
        mock_manager.check_experiments = mock.AsyncMock()
        mock_manager.run_maintenance = mock.AsyncMock()
        
        # Mock simulation methods
        mock_manager._identify_improvement_opportunities = mock.MagicMock(return_value=[
            {
                "type": "PROMPT_TEMPLATE",
                "reason": "Sentiment accuracy below target",
                "metrics": {"sentiment_accuracy": 0.75},
                "potential_impact": 0.2
            },
            {
                "type": "MODEL_SELECTION",
                "reason": "High calibration error",
                "metrics": {"calibration_error": 0.12},
                "potential_impact": 0.15
            }
        ])
        
        yield mock_manager


@pytest.fixture
def mock_ab_testing_framework():
    """Mock for the A/B testing framework."""
    with mock.patch("examples.continuous_improvement_demo.ab_testing_framework") as mock_framework:
        # Mock methods
        mock_framework.initialize = mock.AsyncMock()
        mock_framework.list_experiments = mock.MagicMock(return_value=[
            {
                "id": "test_exp_001",
                "name": "Test Experiment",
                "status": "active",
                "type": "prompt_template"
            }
        ])
        
        yield mock_framework


@pytest.fixture
def mock_logger():
    """Mock for the logger."""
    with mock.patch("examples.continuous_improvement_demo.logger") as mock_logger:
        # Mock methods
        mock_logger.info = mock.MagicMock()
        mock_logger.error = mock.MagicMock()
        
        yield mock_logger


@pytest.fixture
def mock_print():
    """Mock print function for cleaner test output."""
    with mock.patch("builtins.print") as mock_print:
        yield mock_print


class TestContinuousImprovementDemo:
    """Tests for the continuous improvement demo."""

    @pytest.mark.asyncio
    async def test_setup_demo(self, mock_continuous_improvement_manager, mock_ab_testing_framework, mock_print):
        """Test demo setup."""
        # Mock LLMService
        with mock.patch("examples.continuous_improvement_demo.LLMService") as mock_llm_service_class:
            mock_llm_service = mock.MagicMock()
            mock_llm_service.initialize = mock.AsyncMock()
            mock_llm_service_class.return_value = mock_llm_service
            
            # Call setup_demo
            result = await setup_demo()
            
            # Check that all required components were initialized
            assert mock_continuous_improvement_manager.initialize.called
            assert mock_ab_testing_framework.initialize.called
            assert mock_llm_service.initialize.called
            
            # Check that LLM service was returned
            assert result == mock_llm_service

    @pytest.mark.asyncio
    async def test_demonstrate_auto_improvement(self, mock_continuous_improvement_manager, mock_ab_testing_framework):
        """Test auto improvement demonstration."""
        # Mock LLMService
        with mock.patch("examples.continuous_improvement_demo.LLMService") as mock_llm_service_class:
            mock_llm_service = mock.MagicMock()
            mock_llm_service.analyze_sentiment = mock.AsyncMock(return_value={
                "sentiment_value": 0.7,
                "direction": "bullish",
                "confidence": 0.8,
                "explanation": "Test explanation"
            })
            mock_llm_service.initialize = mock.AsyncMock()
            mock_llm_service.close = mock.AsyncMock()
            mock_llm_service_class.return_value = mock_llm_service
            
            # Mock experiment functions
            with mock.patch("examples.continuous_improvement_demo.generate_demo_experiment") as mock_generate:
                mock_generate.return_value = "test_exp_id"
                
                with mock.patch("examples.continuous_improvement_demo.check_experiment_status") as mock_check:
                    # Mock experiment object
                    mock_experiment = mock.MagicMock()
                    mock_experiment.variants = [mock.MagicMock(), mock.MagicMock()]
                    mock_experiment.analyze_results = mock.MagicMock()
                    mock_ab_testing_framework.get_experiment.return_value = mock_experiment
                    
                    # Mock get_improvement_history
                    mock_continuous_improvement_manager.get_improvement_history.return_value = [{
                        "action": "improvement_implemented",
                        "timestamp": "2025-03-24T12:00:00",
                        "details": {
                            "experiment_name": "Test Experiment",
                            "experiment_id": "test_exp_id",
                            "winning_variant": "Enhanced Template"
                        }
                    }]
                    
                    # Call the function
                    await demonstrate_auto_improvement()
                    
                    # Check that required methods were called
                    assert mock_continuous_improvement_manager.run_maintenance.called
                    assert mock_ab_testing_framework.complete_experiment.called
                    assert mock_experiment.analyze_results.called
                    assert mock_continuous_improvement_manager.get_improvement_history.called

    @pytest.mark.asyncio
    async def test_generate_opportunities(self, mock_continuous_improvement_manager, mock_ab_testing_framework):
        """Test opportunity generation function."""
        # Setup needed mocks
        opportunities = [
            {
                "type": mock.MagicMock(value="prompt_template"),
                "reason": "Sentiment accuracy is below target",
                "metrics": {"sentiment_accuracy": 0.75},
                "potential_impact": 0.8
            },
            {
                "type": mock.MagicMock(value="model_selection"),
                "reason": "Calibration error is high",
                "metrics": {"calibration_error": 0.12},
                "potential_impact": 0.7
            }
        ]
        mock_continuous_improvement_manager._identify_improvement_opportunities.return_value = opportunities
        
        # Mock experiment objects
        mock_experiments = [mock.MagicMock()]
        mock_ab_testing_framework.active_experiment_ids = ["test_exp_id"]
        mock_ab_testing_framework.get_experiment.return_value = mock_experiments[0]
        mock_experiments[0].metadata = {"auto_generated": True}
        mock_experiments[0].name = "Test Auto Experiment"
        mock_experiments[0].experiment_type = mock.MagicMock(value="prompt_template")
        
        # Test the function
        await generate_opportunities()
        
        # Check that required methods were called
        assert mock_continuous_improvement_manager._identify_improvement_opportunities.called
        
        if mock_continuous_improvement_manager.enabled:
            assert mock_continuous_improvement_manager.generate_experiments.called

    @pytest.mark.asyncio
    async def test_simulate_sentiment_analysis(self):
        """Test sentiment analysis simulation."""
        # Mock LLMService
        with mock.patch("examples.continuous_improvement_demo.LLMService") as mock_llm_service_class:
            mock_llm_service = mock.MagicMock()
            mock_llm_service.analyze_sentiment = mock.AsyncMock(return_value={
                "sentiment_value": 0.7,
                "direction": "bullish",
                "confidence": 0.8,
                "explanation": "Test explanation"
            })
            
            # Call function with small iteration count to speed up test
            await simulate_sentiment_analysis(mock_llm_service, iterations=2)
            
            # Verify sentiment analysis was called
            assert mock_llm_service.analyze_sentiment.call_count == 2

    @pytest.mark.asyncio
    async def test_trigger_maintenance(self, mock_continuous_improvement_manager):
        """Test trigger maintenance function."""
        await trigger_maintenance()
        
        # Check that maintenance was triggered
        assert mock_continuous_improvement_manager.run_maintenance.called

    @pytest.mark.asyncio
    async def test_main(self, mock_continuous_improvement_manager, mock_ab_testing_framework, monkeypatch):
        """Test the main function."""
        # Mock input to exit immediately
        monkeypatch.setattr('builtins.input', lambda _: '6')
        
        # Mock LLMService
        with mock.patch("examples.continuous_improvement_demo.LLMService") as mock_llm_service_class:
            mock_llm_service = mock.MagicMock()
            mock_llm_service.initialize = mock.AsyncMock()
            mock_llm_service.close = mock.AsyncMock()
            mock_llm_service_class.return_value = mock_llm_service
            
            # Run the main function
            await main()
            
            # Check that initialization was performed
            assert mock_continuous_improvement_manager.initialize.called
            assert mock_ab_testing_framework.initialize.called