"""Tests for Rust optimization in the continuous improvement system."""

import os
import sys
import pytest
import json
import time
from unittest import mock
import numpy as np

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Try to import Rust optimization functions
try:
    from src.rust_bridge import analyze_experiment_results, identify_improvement_opportunities
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def generate_test_experiment_data(num_variants=3, num_metrics=4, control_advantage=True):
    """Generate test experiment data.
    
    Args:
        num_variants: Number of variants to generate
        num_metrics: Number of metrics per variant
        control_advantage: If True, control performs worse than treatments
        
    Returns:
        Dictionary with experiment data
    """
    experiment_data = {
        "id": "test_experiment",
        "name": "Test Experiment",
        "experiment_type": "prompt_template",
        "variants": []
    }
    
    # Create control variant with base metrics
    control_metrics = {}
    for i in range(num_metrics):
        metric_name = f"metric_{i}"
        control_metrics[metric_name] = 0.7 if control_advantage else 0.8
    
    control_variant = {
        "id": "control",
        "name": "Control",
        "control": True,
        "metrics": control_metrics
    }
    experiment_data["variants"].append(control_variant)
    
    # Create treatment variants
    for i in range(1, num_variants):
        variant_metrics = {}
        for j in range(num_metrics):
            metric_name = f"metric_{j}"
            # Make treatment better or worse than control depending on control_advantage
            if control_advantage:
                # Treatment is better (higher value)
                variant_metrics[metric_name] = 0.7 + (0.1 * i)
            else:
                # Control is better (higher value)
                variant_metrics[metric_name] = 0.8 - (0.05 * i)
        
        variant = {
            "id": f"variant_{i}",
            "name": f"Variant {i}",
            "control": False,
            "metrics": variant_metrics
        }
        experiment_data["variants"].append(variant)
    
    return experiment_data


def generate_test_metrics_data(num_metrics=5, include_nested=True):
    """Generate test metrics data.
    
    Args:
        num_metrics: Number of base metrics to generate
        include_nested: Whether to include nested metrics
        
    Returns:
        Dictionary with metrics data
    """
    metrics_data = {}
    
    # Generate base metrics
    for i in range(num_metrics):
        metric_name = f"metric_{i}"
        metrics_data[metric_name] = 0.7 + (0.05 * i)
    
    # Add specific required metrics
    metrics_data["sentiment_accuracy"] = 0.82
    metrics_data["direction_accuracy"] = 0.75
    metrics_data["confidence_score"] = 0.78
    metrics_data["calibration_error"] = 0.09
    
    # Add nested metrics if requested
    if include_nested:
        # Add source metrics
        metrics_data["by_source"] = {
            "twitter": {"sentiment_accuracy": 0.85, "direction_accuracy": 0.78},
            "news": {"sentiment_accuracy": 0.75, "direction_accuracy": 0.70},
            "reddit": {"sentiment_accuracy": 0.80, "direction_accuracy": 0.73}
        }
        
        # Add market condition metrics
        metrics_data["by_market_condition"] = {
            "bull": {"sentiment_accuracy": 0.88, "direction_accuracy": 0.82},
            "bear": {"sentiment_accuracy": 0.76, "direction_accuracy": 0.68},
            "neutral": {"sentiment_accuracy": 0.81, "direction_accuracy": 0.74}
        }
        
        # Add update frequency metrics
        metrics_data["by_update_frequency"] = {
            "hourly": 0.83,
            "daily": 0.81,
            "weekly": 0.75
        }
    
    return metrics_data


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust optimization not available")
def test_analyze_experiment_results():
    """Test the Rust-optimized analyze_experiment_results function."""
    # Generate test data with control worse than treatments
    experiment_data = generate_test_experiment_data(num_variants=3, control_advantage=True)
    
    # Call the Rust-optimized function
    result = analyze_experiment_results(
        experiment_data,
        significance_threshold=0.95,
        improvement_threshold=0.05
    )
    
    # Check that the result contains expected fields
    assert "has_significant_results" in result
    assert "has_clear_winner" in result
    assert "confidence_level" in result
    
    # Check that metrics_differences and p_values are present
    assert "metrics_differences" in result
    assert "p_values" in result
    
    # In our test case the treatment variants should be better than control
    assert result["has_significant_results"] is True
    
    # With our values, we should have a clear winner
    assert result["has_clear_winner"] is True
    
    # Winner should be "Variant 2" (the better variant)
    assert result["winning_variant"] == "Variant 2"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust optimization not available")
def test_identify_improvement_opportunities():
    """Test the Rust-optimized identify_improvement_opportunities function."""
    # Generate test metrics data
    metrics_data = generate_test_metrics_data()
    
    # Call the Rust-optimized function
    results = identify_improvement_opportunities(metrics_data)
    
    # Check that we got a list of opportunities
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Each opportunity should have the expected fields
    for opportunity in results:
        assert "type" in opportunity
        assert "reason" in opportunity
        assert "metrics" in opportunity
        assert "potential_impact" in opportunity
        
        # Impact should be between 0 and 1
        assert 0 <= opportunity["potential_impact"] <= 1
        
        # Type should be a valid experiment type
        assert opportunity["type"] in [
            "PROMPT_TEMPLATE", "MODEL_SELECTION", "TEMPERATURE", 
            "CONTEXT_STRATEGY", "AGGREGATION_WEIGHTS", 
            "UPDATE_FREQUENCY", "CONFIDENCE_THRESHOLD"
        ]


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust optimization not available")
def test_large_experiment_performance():
    """Test performance of Rust optimization with large experiment data."""
    # Generate large test data (10 variants, 8 metrics each)
    experiment_data = generate_test_experiment_data(num_variants=10, num_metrics=8)
    
    # Time the Rust-optimized function
    start_time = time.time()
    result_rust = analyze_experiment_results(
        experiment_data,
        significance_threshold=0.95,
        improvement_threshold=0.05
    )
    rust_time = time.time() - start_time
    
    # Print performance results
    print(f"Rust analyze_experiment_results time: {rust_time:.4f} seconds")
    
    # We expect this to run in under 0.5 seconds even on slower hardware
    assert rust_time < 0.5, f"Rust optimization is slower than expected: {rust_time:.4f} seconds"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust optimization not available")
def test_large_metrics_performance():
    """Test performance of Rust optimization with large metrics data."""
    # Generate large test metrics data
    metrics_data = generate_test_metrics_data(num_metrics=20)
    
    # Add more nested data to make it complex
    for i in range(5):
        source_name = f"source_{i}"
        metrics_data["by_source"][source_name] = {
            "sentiment_accuracy": 0.75 + (0.02 * i),
            "direction_accuracy": 0.70 + (0.02 * i),
            "calibration_error": 0.10 - (0.01 * i)
        }
    
    for i in range(5):
        condition_name = f"condition_{i}"
        metrics_data["by_market_condition"][condition_name] = {
            "sentiment_accuracy": 0.80 + (0.01 * i),
            "direction_accuracy": 0.75 + (0.01 * i)
        }
    
    # Time the Rust-optimized function
    start_time = time.time()
    results_rust = identify_improvement_opportunities(metrics_data)
    rust_time = time.time() - start_time
    
    # Print performance results
    print(f"Rust identify_improvement_opportunities time: {rust_time:.4f} seconds")
    
    # We expect this to run in under 0.3 seconds
    assert rust_time < 0.3, f"Rust optimization is slower than expected: {rust_time:.4f} seconds"


def test_fallback_behavior():
    """Test fallback behavior when Rust optimization is not available."""
    # Mock import error to simulate Rust unavailability
    with mock.patch('src.rust_bridge.analyze_experiment_results', side_effect=ImportError):
        # Import the wrapper that should handle the fallback
        from src.rust_bridge.sentiment_py import ContinuousImprovementRust
        
        # Generate test data
        experiment_data = generate_test_experiment_data()
        
        # Call the function that should handle fallback
        result = ContinuousImprovementRust.analyze_experiment_results(
            experiment_data,
            significance_threshold=0.95,
            improvement_threshold=0.05
        )
        
        # Check that we got a valid result even without Rust
        assert isinstance(result, dict)
        assert "has_significant_results" in result
        
    # Similarly test the opportunity identification fallback
    with mock.patch('src.rust_bridge.identify_improvement_opportunities', side_effect=ImportError):
        # Import the wrapper that should handle the fallback
        from src.rust_bridge.sentiment_py import ContinuousImprovementRust
        
        # Generate test metrics data
        metrics_data = generate_test_metrics_data()
        
        # Call the function that should handle fallback
        results = ContinuousImprovementRust.identify_improvement_opportunities(metrics_data)
        
        # Check that we got a valid result even without Rust
        assert isinstance(results, list)
        assert len(results) > 0


if __name__ == "__main__":
    # Run tests if executed directly
    if RUST_AVAILABLE:
        print("Rust optimization is available, running all tests...")
        test_analyze_experiment_results()
        test_identify_improvement_opportunities()
        test_large_experiment_performance()
        test_large_metrics_performance()
    else:
        print("Rust optimization is not available, testing fallback behavior...")
    
    test_fallback_behavior()
    print("All tests passed!")