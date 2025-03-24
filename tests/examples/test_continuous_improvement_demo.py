"""
Tests for the Continuous Improvement Demo script.

This module tests the demonstration script for the continuous improvement system.
"""

import asyncio
import pytest
import unittest.mock as mock

from examples.continuous_improvement_demo import (
    initialize_system, demonstrate_auto_improvement, demonstrate_opportunity_detection,
    demonstrate_experiment_tracking, demonstrate_metrics_monitoring, run_demo
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
def mock_performance_tracker():
    """Mock for the performance tracker."""
    with mock.patch("examples.continuous_improvement_demo.performance_tracker") as mock_tracker:
        # Mock methods
        mock_tracker.initialize = mock.AsyncMock()
        mock_tracker.get_recent_metrics = mock.MagicMock(return_value={
            "sentiment_accuracy": 0.75,
            "direction_accuracy": 0.65
        })
        
        yield mock_tracker


@pytest.fixture
def mock_print():
    """Mock print function for cleaner test output."""
    with mock.patch("examples.continuous_improvement_demo.print") as mock_print:
        yield mock_print


class TestContinuousImprovementDemo:
    """Tests for the continuous improvement demo."""

    @pytest.mark.asyncio
    async def test_initialize_system(self, mock_continuous_improvement_manager, mock_ab_testing_framework, mock_performance_tracker, mock_print):
        """Test system initialization."""
        await initialize_system()
        
        # Check that all required components were initialized
        assert mock_continuous_improvement_manager.initialize.called
        assert mock_ab_testing_framework.initialize.called
        assert mock_performance_tracker.initialize.called
        
        # Check output
        assert mock_print.called
        initialization_output = " ".join(str(args[0]) for args, _ in mock_print.call_args_list)
        assert "initialized" in initialization_output.lower()

    @pytest.mark.asyncio
    async def test_demonstrate_auto_improvement(self, mock_continuous_improvement_manager, mock_print):
        """Test auto improvement demonstration."""
        await demonstrate_auto_improvement()
        
        # Check that the required methods were called
        assert mock_continuous_improvement_manager.run_maintenance.called
        
        # Check output
        assert mock_print.called
        improvement_output = " ".join(str(args[0]) for args, _ in mock_print.call_args_list)
        assert "auto improvement" in improvement_output.lower()

    @pytest.mark.asyncio
    async def test_demonstrate_opportunity_detection(self, mock_continuous_improvement_manager, mock_performance_tracker, mock_print):
        """Test opportunity detection demonstration."""
        await demonstrate_opportunity_detection()
        
        # Check that the required methods were called
        assert mock_continuous_improvement_manager._identify_improvement_opportunities.called
        
        # Check output
        assert mock_print.called
        opportunity_output = " ".join(str(args[0]) for args, _ in mock_print.call_args_list)
        assert "opportunity" in opportunity_output.lower()
        assert "prompt_template" in opportunity_output.lower()
        assert "model_selection" in opportunity_output.lower()

    @pytest.mark.asyncio
    async def test_demonstrate_experiment_tracking(self, mock_continuous_improvement_manager, mock_ab_testing_framework, mock_print):
        """Test experiment tracking demonstration."""
        await demonstrate_experiment_tracking()
        
        # Check that the required methods were called
        assert mock_ab_testing_framework.list_experiments.called
        
        # Check output
        assert mock_print.called
        tracking_output = " ".join(str(args[0]) for args, _ in mock_print.call_args_list)
        assert "experiment" in tracking_output.lower()
        assert "test_exp_001" in tracking_output.lower()

    @pytest.mark.asyncio
    async def test_demonstrate_metrics_monitoring(self, mock_continuous_improvement_manager, mock_performance_tracker, mock_print):
        """Test metrics monitoring demonstration."""
        await demonstrate_metrics_monitoring()
        
        # Check that the required methods were called
        assert mock_performance_tracker.get_recent_metrics.called
        
        # Check output
        assert mock_print.called
        metrics_output = " ".join(str(args[0]) for args, _ in mock_print.call_args_list)
        assert "metrics" in metrics_output.lower()
        assert "sentiment_accuracy" in metrics_output.lower()
        assert "direction_accuracy" in metrics_output.lower()

    @pytest.mark.asyncio
    async def test_run_demo(self, mock_continuous_improvement_manager, mock_ab_testing_framework, mock_performance_tracker, mock_print, monkeypatch):
        """Test the full demo run with mocked user input."""
        # Mock input function to return '5' (exit option)
        monkeypatch.setattr('builtins.input', lambda _: '5')
        
        # Run the demo
        await run_demo()
        
        # Check that initialization was called
        assert mock_continuous_improvement_manager.initialize.called
        assert mock_ab_testing_framework.initialize.called
        assert mock_performance_tracker.initialize.called
        
        # Check output
        assert mock_print.called
        demo_output = " ".join(str(args[0]) for args, _ in mock_print.call_args_list)
        assert "continuous improvement" in demo_output.lower()
        assert "demo" in demo_output.lower()
        assert "exiting" in demo_output.lower()