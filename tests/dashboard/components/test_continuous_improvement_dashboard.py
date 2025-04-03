"""
Tests for the Continuous Improvement Dashboard component.

This module tests the dashboard functionality for the continuous improvement system.
"""

import pytest
import unittest.mock as mock
from datetime import datetime, timedelta

from src.dashboard.components.continuous_improvement_dashboard import (
    create_layout, update_improvement_history, update_experiment_status,
    create_visualization, get_improvement_metrics
)


@pytest.fixture
def mock_continuous_improvement_manager():
    """Mock for the continuous improvement manager."""
    with mock.patch("src.dashboard.components.continuous_improvement_dashboard.continuous_improvement_manager") as mock_manager:
        # Mock improvement history
        mock_manager.get_improvement_history.return_value = [
            {
                "timestamp": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                "action": "experiment_generated",
                "experiment_id": "prompt_template_001",
                "experiment_name": "Auto Prompt Template Test 1",
                "opportunity": {
                    "type": "PROMPT_TEMPLATE",
                    "reason": "Sentiment accuracy below target",
                    "metrics": {"sentiment_accuracy": 0.75},
                    "potential_impact": 0.2
                }
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                "action": "improvement_implemented",
                "details": {
                    "timestamp": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                    "experiment_id": "prompt_template_001",
                    "experiment_name": "Auto Prompt Template Test 1",
                    "experiment_type": "prompt_template",
                    "winning_variant": "AI-Enhanced Template",
                    "variant_config": {"template": "Improved template content"},
                    "metrics_improvement": {
                        "sentiment_accuracy": 0.06,
                        "direction_accuracy": 0.04
                    }
                }
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "action": "experiment_generated",
                "experiment_id": "model_selection_001",
                "experiment_name": "Auto Model Selection Test",
                "opportunity": {
                    "type": "MODEL_SELECTION",
                    "reason": "High calibration error",
                    "metrics": {"calibration_error": 0.12},
                    "potential_impact": 0.15
                }
            }
        ]
        
        # Mock status
        mock_manager.get_status.return_value = {
            "enabled": True,
            "last_check": datetime.utcnow().isoformat(),
            "last_experiment_generation": (datetime.utcnow() - timedelta(hours=5)).isoformat(),
            "improvements_count": 3,
            "auto_implement": True,
            "active_experiments": 1
        }
        
        # Mock results history
        mock_manager.results_history = [
            {
                "timestamp": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                "experiment_id": "prompt_template_001",
                "experiment_name": "Auto Prompt Template Test 1",
                "experiment_type": "prompt_template",
                "winning_variant": "AI-Enhanced Template",
                "variant_config": {"template": "Improved template content"},
                "metrics_improvement": {
                    "sentiment_accuracy": 0.06,
                    "direction_accuracy": 0.04
                }
            }
        ]
        
        yield mock_manager


@pytest.fixture
def mock_ab_testing_framework():
    """Mock for the A/B testing framework."""
    with mock.patch("src.dashboard.components.continuous_improvement_dashboard.ab_testing_framework") as mock_framework:
        # Mock active experiments
        mock_framework.list_experiments.return_value = [
            {
                "id": "model_selection_001",
                "name": "Auto Model Selection Test",
                "status": "active",
                "type": "model_selection",
                "variants": 2,
                "start_time": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "end_time": None,
                "total_traffic": 125,
                "owner": "continuous_improvement",
                "has_results": False,
                "has_winner": False,
                "created_at": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        ]
        
        yield mock_framework


class TestContinuousImprovementDashboard:
    """Tests for the continuous improvement dashboard component."""

    def test_create_layout(self, mock_continuous_improvement_manager, mock_ab_testing_framework):
        """Test layout creation."""
        layout = create_layout()
        
        # Layout should have tabs
        assert "tabs" in str(layout)
        
        # Should include both history tab and settings tab
        assert "history-tab" in str(layout)
        assert "settings-tab" in str(layout)
        assert "experiments-tab" in str(layout)
        assert "visualization-tab" in str(layout)

    def test_update_improvement_history(self, mock_continuous_improvement_manager):
        """Test improvement history update."""
        history_table = update_improvement_history()
        
        # History should include at least two entries
        assert history_table is not None
        assert len(history_table["data"]) >= 2
        
        # Check data structure
        first_row = history_table["data"][0]
        assert "Date" in first_row
        assert "Action" in first_row
        assert "Experiment" in first_row
        
        # Check content
        assert "experiment_generated" in str(history_table["data"])
        assert "improvement_implemented" in str(history_table["data"])
        assert "Auto Prompt Template Test 1" in str(history_table["data"])

    def test_update_experiment_status(self, mock_continuous_improvement_manager, mock_ab_testing_framework):
        """Test experiment status update."""
        status_table = update_experiment_status()
        
        # Status should include active experiments
        assert status_table is not None
        assert len(status_table["data"]) >= 1
        
        # Check data structure and content
        assert "Auto Model Selection Test" in str(status_table["data"])
        assert "model_selection" in str(status_table["data"])
        assert "active" in str(status_table["data"])

    def test_create_visualization(self, mock_continuous_improvement_manager):
        """Test visualization creation."""
        figures = create_visualization()
        
        # Should return a list of figures
        assert isinstance(figures, list)
        assert len(figures) >= 2  # At least timeline and metrics figures
        
        # Check that figures have data
        for fig in figures:
            assert fig is not None
            assert "data" in fig
            assert len(fig["data"]) > 0

    def test_get_improvement_metrics(self, mock_continuous_improvement_manager):
        """Test improvement metrics calculation."""
        metrics = get_improvement_metrics()
        
        # Should return metrics dictionary
        assert isinstance(metrics, dict)
        
        # Check required keys
        assert "improvement_count" in metrics
        assert "avg_sentiment_improvement" in metrics
        assert "avg_direction_improvement" in metrics
        assert "experiments_generated" in metrics
        assert "implementations" in metrics
        
        # Verify values
        assert metrics["improvement_count"] >= 1
        assert metrics["avg_sentiment_improvement"] > 0
        assert metrics["experiments_generated"] >= 2