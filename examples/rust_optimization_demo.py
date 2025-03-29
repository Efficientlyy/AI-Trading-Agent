#!/usr/bin/env python
"""
Demonstration of Rust optimization for continuous improvement system.

This script demonstrates the performance improvements from using
Rust-optimized functions for experiment analysis and opportunity
identification in the continuous improvement system.
"""

import os
import sys
import time
import json
import random
import argparse
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Try to import Rust optimization functions
try:
    from src.rust_bridge import analyze_experiment_results, identify_improvement_opportunities
    from src.rust_bridge.sentiment_py import ContinuousImprovementRust
    RUST_AVAILABLE = True
    print("✅ Rust optimization available")
except ImportError:
    from src.rust_bridge.sentiment_py import ContinuousImprovementRust
    RUST_AVAILABLE = False
    print("⚠️  Rust optimization not available, using Python fallback")


def generate_test_experiment_data(
    num_variants: int = 3,
    num_metrics: int = 4,
    control_advantage: bool = True,
    samples_per_variant: int = 1000
) -> Dict[str, Any]:
    """Generate test experiment data.
    
    Args:
        num_variants: Number of variants to generate
        num_metrics: Number of metrics per variant
        control_advantage: If True, control performs worse than treatments
        samples_per_variant: Number of samples per variant
        
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
        # Base value for control
        base_value = 0.7 if control_advantage else 0.8
        # Add some randomness
        control_metrics[metric_name] = base_value + random.uniform(-0.05, 0.05)
    
    # Add standard metrics
    control_metrics["sentiment_accuracy"] = 0.82 if not control_advantage else 0.75
    control_metrics["direction_accuracy"] = 0.78 if not control_advantage else 0.70
    control_metrics["confidence_score"] = 0.88 if not control_advantage else 0.80
    control_metrics["calibration_error"] = 0.08 if not control_advantage else 0.12
    
    # Add counts
    control_metrics["requests"] = samples_per_variant
    control_metrics["successes"] = int(samples_per_variant * 0.95)
    control_metrics["errors"] = samples_per_variant - control_metrics["successes"]
    
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
                variant_metrics[metric_name] = 0.7 + (0.05 * i) + random.uniform(-0.03, 0.03)
            else:
                # Control is better (higher value)
                variant_metrics[metric_name] = 0.8 - (0.03 * i) + random.uniform(-0.03, 0.03)
        
        # Add standard metrics
        improvement = 0.05 * i if control_advantage else -0.03 * i
        variant_metrics["sentiment_accuracy"] = (0.82 if not control_advantage else 0.75) + improvement
        variant_metrics["direction_accuracy"] = (0.78 if not control_advantage else 0.70) + improvement
        variant_metrics["confidence_score"] = (0.88 if not control_advantage else 0.80) + improvement
        variant_metrics["calibration_error"] = (0.08 if not control_advantage else 0.12) - improvement
        
        # Add counts
        variant_metrics["requests"] = samples_per_variant
        variant_metrics["successes"] = int(samples_per_variant * (0.95 + improvement/10))
        variant_metrics["errors"] = samples_per_variant - variant_metrics["successes"]
        
        variant = {
            "id": f"variant_{i}",
            "name": f"Variant {i}",
            "control": False,
            "metrics": variant_metrics
        }
        experiment_data["variants"].append(variant)
    
    return experiment_data


def generate_test_metrics_data(num_metrics: int = 5, num_sources: int = 3, num_conditions: int = 3) -> Dict[str, Any]:
    """Generate test metrics data.
    
    Args:
        num_metrics: Number of base metrics to generate
        num_sources: Number of data sources to include
        num_conditions: Number of market conditions to include
        
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
    
    # Add source metrics
    metrics_data["by_source"] = {}
    for i in range(num_sources):
        source_name = f"source_{i}"
        source_metrics = {
            "sentiment_accuracy": 0.75 + (0.05 * i) + random.uniform(-0.05, 0.05),
            "direction_accuracy": 0.70 + (0.03 * i) + random.uniform(-0.03, 0.03),
            "confidence_score": 0.80 + (0.02 * i) + random.uniform(-0.02, 0.02),
            "calibration_error": 0.10 - (0.01 * i) + random.uniform(-0.01, 0.01)
        }
        metrics_data["by_source"][source_name] = source_metrics
    
    # Add market condition metrics
    metrics_data["by_market_condition"] = {}
    conditions = ["bull", "bear", "neutral", "volatile", "sideways"]
    for i in range(min(num_conditions, len(conditions))):
        condition_name = conditions[i]
        condition_metrics = {
            "sentiment_accuracy": 0.80 + (0.03 * i) + random.uniform(-0.03, 0.03),
            "direction_accuracy": 0.75 + (0.02 * i) + random.uniform(-0.02, 0.02),
            "confidence_score": 0.85 + (0.01 * i) + random.uniform(-0.01, 0.01),
            "calibration_error": 0.08 - (0.005 * i) + random.uniform(-0.005, 0.005)
        }
        metrics_data["by_market_condition"][condition_name] = condition_metrics
    
    # Add update frequency metrics
    metrics_data["by_update_frequency"] = {
        "hourly": 0.83,
        "daily": 0.81,
        "weekly": 0.75
    }
    
    return metrics_data


def benchmark_experiment_analysis(
    num_variants: int = 5, 
    num_metrics: int = 6, 
    num_runs: int = 5,
    samples_per_variant: int = 1000,
    control_advantage: bool = True
) -> Dict[str, Any]:
    """Benchmark experiment analysis performance.
    
    Args:
        num_variants: Number of variants to include
        num_metrics: Number of metrics per variant
        num_runs: Number of benchmark runs
        samples_per_variant: Number of samples per variant
        control_advantage: If True, control worse than treatments
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n🔍 Benchmarking experiment analysis with {num_variants} variants, {num_metrics} metrics")
    
    # Generate test data
    experiment_data = generate_test_experiment_data(
        num_variants=num_variants,
        num_metrics=num_metrics,
        control_advantage=control_advantage,
        samples_per_variant=samples_per_variant
    )
    
    # Benchmark Rust implementation
    if RUST_AVAILABLE:
        rust_times = []
        for i in range(num_runs):
            start_time = time.time()
            rust_result = analyze_experiment_results(
                experiment_data,
                significance_threshold=0.95,
                improvement_threshold=0.05
            )
            end_time = time.time()
            rust_times.append(end_time - start_time)
        
        avg_rust_time = sum(rust_times) / len(rust_times)
        print(f"  ⚡ Rust implementation: {avg_rust_time:.6f} seconds (avg of {num_runs} runs)")
    else:
        avg_rust_time = None
        print("  ⚠️  Rust implementation not available")
    
    # Benchmark Python implementation
    python_times = []
    for i in range(num_runs):
        start_time = time.time()
        python_result = ContinuousImprovementRust._analyze_experiment_results_py(
            experiment_data,
            significance_threshold=0.95,
            improvement_threshold=0.05
        )
        end_time = time.time()
        python_times.append(end_time - start_time)
    
    avg_python_time = sum(python_times) / len(python_times)
    print(f"  🐍 Python implementation: {avg_python_time:.6f} seconds (avg of {num_runs} runs)")
    
    # Calculate speedup
    if avg_rust_time:
        speedup = avg_python_time / avg_rust_time
        print(f"  🚀 Speedup: {speedup:.2f}x")
    else:
        speedup = None
    
    return {
        "configuration": {
            "num_variants": num_variants,
            "num_metrics": num_metrics,
            "samples_per_variant": samples_per_variant,
            "num_runs": num_runs
        },
        "rust_time": avg_rust_time,
        "python_time": avg_python_time,
        "speedup": speedup
    }


def benchmark_opportunity_identification(
    num_metrics: int = 10, 
    num_sources: int = 5, 
    num_conditions: int = 5,
    num_runs: int = 5
) -> Dict[str, Any]:
    """Benchmark opportunity identification performance.
    
    Args:
        num_metrics: Number of base metrics to include
        num_sources: Number of data sources to include
        num_conditions: Number of market conditions to include
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n🔍 Benchmarking opportunity identification with {num_metrics} metrics, {num_sources} sources")
    
    # Generate test data
    metrics_data = generate_test_metrics_data(
        num_metrics=num_metrics,
        num_sources=num_sources,
        num_conditions=num_conditions
    )
    
    # Benchmark Rust implementation
    if RUST_AVAILABLE:
        rust_times = []
        for i in range(num_runs):
            start_time = time.time()
            rust_result = identify_improvement_opportunities(metrics_data)
            end_time = time.time()
            rust_times.append(end_time - start_time)
        
        avg_rust_time = sum(rust_times) / len(rust_times)
        print(f"  ⚡ Rust implementation: {avg_rust_time:.6f} seconds (avg of {num_runs} runs)")
    else:
        avg_rust_time = None
        print("  ⚠️  Rust implementation not available")
    
    # Benchmark Python implementation
    python_times = []
    for i in range(num_runs):
        start_time = time.time()
        python_result = ContinuousImprovementRust._identify_improvement_opportunities_py(metrics_data)
        end_time = time.time()
        python_times.append(end_time - start_time)
    
    avg_python_time = sum(python_times) / len(python_times)
    print(f"  🐍 Python implementation: {avg_python_time:.6f} seconds (avg of {num_runs} runs)")
    
    # Calculate speedup
    if avg_rust_time:
        speedup = avg_python_time / avg_rust_time
        print(f"  🚀 Speedup: {speedup:.2f}x")
    else:
        speedup = None
    
    return {
        "configuration": {
            "num_metrics": num_metrics,
            "num_sources": num_sources,
            "num_conditions": num_conditions,
            "num_runs": num_runs
        },
        "rust_time": avg_rust_time,
        "python_time": avg_python_time,
        "speedup": speedup
    }


def run_scaling_benchmark(max_variants: int = 20, step: int = 5) -> None:
    """Run a scaling benchmark to show how performance changes with problem size.
    
    Args:
        max_variants: Maximum number of variants to test
        step: Step size for increasing variants
    """
    print("\n📊 Scaling Benchmark: Performance vs. Number of Variants")
    print("=" * 70)
    
    results = []
    variant_counts = list(range(2, max_variants + 1, step))
    
    for variant_count in variant_counts:
        print(f"\nRunning benchmark with {variant_count} variants...")
        result = benchmark_experiment_analysis(num_variants=variant_count, num_runs=3)
        results.append(result)
    
    # Print summary
    print("\n📋 Scaling Results Summary")
    print("=" * 70)
    print(f"{'Variants':<10} {'Python Time':<15} {'Rust Time':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for i, result in enumerate(results):
        variants = variant_counts[i]
        python_time = result["python_time"]
        rust_time = result["rust_time"] if result["rust_time"] else "N/A"
        speedup = f"{result['speedup']:.2f}x" if result["speedup"] else "N/A"
        
        print(f"{variants:<10} {python_time:<15.6f} ", end="")
        if isinstance(rust_time, float):
            print(f"{rust_time:<15.6f} {speedup:<10}")
        else:
            print(f"{rust_time:<15} {speedup:<10}")
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "output", "rust_optimization")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "scaling_benchmark.json")
    
    with open(output_file, "w") as f:
        json.dump({
            "variant_counts": variant_counts,
            "results": results
        }, f, indent=2)
    
    print(f"\nBenchmark results saved to {output_file}")


def compare_implementation_results() -> None:
    """Compare the results from Rust and Python implementations to verify correctness."""
    print("\n🔍 Comparing Implementation Results")
    print("=" * 70)
    
    # Generate test data
    experiment_data = generate_test_experiment_data(
        num_variants=5,
        num_metrics=8,
        control_advantage=True
    )
    
    # Run both implementations
    if RUST_AVAILABLE:
        rust_result = analyze_experiment_results(
            experiment_data,
            significance_threshold=0.95,
            improvement_threshold=0.05
        )
    else:
        rust_result = None
    
    python_result = ContinuousImprovementRust._analyze_experiment_results_py(
        experiment_data,
        significance_threshold=0.95,
        improvement_threshold=0.05
    )
    
    # Compare key fields
    if rust_result:
        print("  Comparing key fields between implementations:")
        
        # Check significant results
        rust_significant = rust_result.get("has_significant_results", False)
        python_significant = python_result.get("has_significant_results", False)
        print(f"  - has_significant_results: {'✅ Match' if rust_significant == python_significant else '❌ Mismatch'}")
        
        # Check clear winner
        rust_clear_winner = rust_result.get("has_clear_winner", False)
        python_clear_winner = python_result.get("has_clear_winner", False)
        print(f"  - has_clear_winner: {'✅ Match' if rust_clear_winner == python_clear_winner else '❌ Mismatch'}")
        
        # Check winning variant
        rust_winner = rust_result.get("winning_variant")
        python_winner = python_result.get("winning_variant")
        print(f"  - winning_variant: {'✅ Match' if rust_winner == python_winner else '❌ Mismatch'}")
        
        print(f"\n  Rust result: {rust_winner if rust_winner else 'No winner'}")
        print(f"  Python result: {python_winner if python_winner else 'No winner'}")
    else:
        print("  ⚠️ Rust implementation not available for comparison")


def main() -> None:
    """Run the demonstration script."""
    parser = argparse.ArgumentParser(description="Demonstrate Rust optimization for continuous improvement system")
    parser.add_argument("--full-benchmark", action="store_true", help="Run full benchmarks including scaling tests")
    parser.add_argument("--variants", type=int, default=10, help="Number of variants for benchmarks")
    parser.add_argument("--metrics", type=int, default=8, help="Number of metrics per variant")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--compare", action="store_true", help="Compare implementation results")
    args = parser.parse_args()
    
    print("\n🚀 Rust Optimization Demo for Continuous Improvement System")
    print("=" * 70)
    
    # Basic benchmarks
    benchmark_experiment_analysis(
        num_variants=args.variants,
        num_metrics=args.metrics,
        num_runs=args.runs
    )
    
    benchmark_opportunity_identification(
        num_metrics=args.metrics,
        num_sources=5,
        num_conditions=5,
        num_runs=args.runs
    )
    
    # Result comparison
    if args.compare:
        compare_implementation_results()
    
    # Full benchmark with scaling tests
    if args.full_benchmark:
        run_scaling_benchmark(max_variants=30, step=5)


if __name__ == "__main__":
    main()