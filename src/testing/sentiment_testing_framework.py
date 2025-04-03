"""Comprehensive testing framework for sentiment analysis.

This module provides a robust testing framework for sentiment analysis components,
including integration tests, performance tests, and regression tests.
"""

import asyncio
import time
import logging
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set, Callable
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sentiment_testing")


class TestResult:
    """Container for test results."""
    
    def __init__(self, test_name: str, component: str, test_type: str):
        """Initialize test result.
        
        Args:
            test_name: Name of the test
            component: Component being tested
            test_type: Type of test (unit, integration, performance, regression)
        """
        self.test_name = test_name
        self.component = component
        self.test_type = test_type
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.duration: Optional[float] = None
        self.success = False
        self.error: Optional[str] = None
        self.metrics: Dict[str, Any] = {}
        self.details: Dict[str, Any] = {}
        
    def complete(self, success: bool, error: Optional[str] = None):
        """Mark test as complete.
        
        Args:
            success: Whether the test passed
            error: Optional error message
        """
        self.end_time = datetime.utcnow()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error = error
        
    def add_metric(self, name: str, value: Any):
        """Add a metric to the test result.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value
        
    def add_detail(self, name: str, value: Any):
        """Add a detail to the test result.
        
        Args:
            name: Detail name
            value: Detail value
        """
        self.details[name] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary.
        
        Returns:
            Dictionary representation of test result
        """
        return {
            "test_name": self.test_name,
            "component": self.component,
            "test_type": self.test_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics,
            "details": self.details
        }


class TestSuite:
    """Test suite for running multiple tests."""
    
    def __init__(self, name: str, output_dir: str = "/tmp/sentiment_test_results"):
        """Initialize test suite.
        
        Args:
            name: Name of the test suite
            output_dir: Directory for test results
        """
        self.name = name
        self.output_dir = output_dir
        self.tests: List[Callable[[], TestResult]] = []
        self.results: List[TestResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def add_test(self, test_func: Callable[[], TestResult]):
        """Add a test to the suite.
        
        Args:
            test_func: Test function that returns a TestResult
        """
        self.tests.append(test_func)
        
    async def run_async(self):
        """Run all tests asynchronously."""
        self.start_time = datetime.utcnow()
        self.results = []
        
        # Create tasks for all tests
        tasks = []
        for test_func in self.tests:
            if asyncio.iscoroutinefunction(test_func):
                # Async function
                tasks.append(asyncio.create_task(test_func()))
            else:
                # Sync function, run in executor
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(test_func)
                ))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                # Create a failed test result
                error_result = TestResult(
                    test_name="unknown",
                    component="unknown",
                    test_type="unknown"
                )
                error_result.complete(False, str(result))
                self.results.append(error_result)
            else:
                self.results.append(result)
        
        self.end_time = datetime.utcnow()
        
        # Save results
        self.save_results()
        
        return self.results
        
    def run(self):
        """Run all tests synchronously."""
        self.start_time = datetime.utcnow()
        self.results = []
        
        for test_func in self.tests:
            try:
                result = test_func()
                self.results.append(result)
            except Exception as e:
                # Create a failed test result
                error_result = TestResult(
                    test_name="unknown",
                    component="unknown",
                    test_type="unknown"
                )
                error_result.complete(False, str(e))
                self.results.append(error_result)
        
        self.end_time = datetime.utcnow()
        
        # Save results
        self.save_results()
        
        return self.results
        
    def save_results(self):
        """Save test results to file."""
        if not self.results:
            return
            
        # Create timestamp for filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert results to dictionaries
        results_dict = {
            "suite_name": self.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None,
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.success),
            "failed_tests": sum(1 for r in self.results if not r.success),
            "results": [r.to_dict() for r in self.results]
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        logger.info(f"Test results saved to {filepath}")
        
        # Generate report
        self.generate_report(filepath)
        
    def generate_report(self, results_file: str):
        """Generate HTML report from test results.
        
        Args:
            results_file: Path to JSON results file
        """
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        # Create report filename
        report_file = results_file.replace('.json', '.html')
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{results['suite_name']} Test Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
                .test-result {{ margin-bottom: 10px; padding: 10px; border-radius: 5px; }}
                .success {{ background-color: #dff0d8; }}
                .failure {{ background-color: #f2dede; }}
                .metrics {{ margin-top: 10px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{results['suite_name']} Test Results</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Start Time: {results['start_time']}</p>
                <p>End Time: {results['end_time']}</p>
                <p>Duration: {results['duration']:.2f} seconds</p>
                <p>Total Tests: {results['total_tests']}</p>
                <p>Passed Tests: {results['passed_tests']}</p>
                <p>Failed Tests: {results['failed_tests']}</p>
            </div>
            
            <h2>Test Results</h2>
        """
        
        # Add test results
        for result in results['results']:
            result_class = "success" if result['success'] else "failure"
            html += f"""
            <div class="test-result {result_class}">
                <h3>{result['test_name']}</h3>
                <p>Component: {result['component']}</p>
                <p>Type: {result['test_type']}</p>
                <p>Duration: {result['duration']:.2f} seconds</p>
                <p>Status: {"Passed" if result['success'] else "Failed"}</p>
            """
            
            if result['error']:
                html += f"<p>Error: {result['error']}</p>"
                
            if result['metrics']:
                html += """
                <div class="metrics">
                    <h4>Metrics</h4>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                """
                
                for metric, value in result['metrics'].items():
                    html += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{value}</td>
                    </tr>
                    """
                    
                html += """
                    </table>
                </div>
                """
                
            html += "</div>"
            
        html += """
        </body>
        </html>
        """
        
        # Save HTML report
        with open(report_file, 'w') as f:
            f.write(html)
            
        logger.info(f"Test report generated at {report_file}")
        
    def summary(self) -> Dict[str, Any]:
        """Get summary of test results.
        
        Returns:
            Summary dictionary
        """
        if not self.results:
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0
            }
            
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        
        return {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed,
            "success_rate": passed / total if total > 0 else 0.0
        }


class PerformanceTest:
    """Base class for performance tests."""
    
    def __init__(self, 
                component_name: str, 
                test_name: str,
                iterations: int = 100,
                warmup_iterations: int = 10):
        """Initialize performance test.
        
        Args:
            component_name: Name of the component being tested
            test_name: Name of the test
            iterations: Number of test iterations
            warmup_iterations: Number of warmup iterations
        """
        self.component_name = component_name
        self.test_name = test_name
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        
    def setup(self):
        """Set up the test environment."""
        pass
        
    def teardown(self):
        """Clean up after the test."""
        pass
        
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single test iteration.
        
        Returns:
            Dictionary of metrics for this iteration
        """
        raise NotImplementedError("Subclasses must implement run_iteration")
        
    def run(self) -> TestResult:
        """Run the performance test.
        
        Returns:
            Test result
        """
        # Create test result
        result = TestResult(
            test_name=self.test_name,
            component=self.component_name,
            test_type="performance"
        )
        
        try:
            # Set up test
            self.setup()
            
            # Run warmup iterations
            for _ in range(self.warmup_iterations):
                self.run_iteration()
                
            # Run test iterations
            iteration_metrics = []
            for _ in range(self.iterations):
                metrics = self.run_iteration()
                iteration_metrics.append(metrics)
                
            # Calculate aggregate metrics
            aggregated_metrics = self._aggregate_metrics(iteration_metrics)
            
            # Add metrics to result
            for metric, value in aggregated_metrics.items():
                result.add_metric(metric, value)
                
            # Add raw iteration data as detail
            result.add_detail("iterations", iteration_metrics)
            
            # Mark as successful
            result.complete(True)
            
        except Exception as e:
            # Mark as failed
            result.complete(False, str(e))
            
        finally:
            # Clean up
            self.teardown()
            
        return result
        
    def _aggregate_metrics(self, iteration_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple iterations.
        
        Args:
            iteration_metrics: List of metrics from each iteration
            
        Returns:
            Aggregated metrics
        """
        if not iteration_metrics:
            return {}
            
        # Get all metric names
        metric_names = set()
        for metrics in iteration_metrics:
            metric_names.update(metrics.keys())
            
        # Aggregate each metric
        aggregated = {}
        for name in metric_names:
            # Extract values for this metric
            values = [
                metrics[name] for metrics in iteration_metrics
                if name in metrics and isinstance(metrics[name], (int, float))
            ]
            
            if values:
                # Calculate statistics
                aggregated[f"{name}_mean"] = float(np.mean(values))
                aggregated[f"{name}_median"] = float(np.median(values))
                aggregated[f"{name}_min"] = float(np.min(values))
                aggregated[f"{name}_max"] = float(np.max(values))
                aggregated[f"{name}_std"] = float(np.std(values))
                
        return aggregated


class IntegrationTest:
    """Base class for integration tests."""
    
    def __init__(self, component_name: str, test_name: str):
        """Initialize integration test.
        
        Args:
            component_name: Name of the component being tested
            test_name: Name of the test
        """
        self.component_name = component_name
        self.test_name = test_name
        
    def setup(self):
        """Set up the test environment."""
        pass
        
    def teardown(self):
        """Clean up after the test."""
        pass
        
    def run_test(self) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Run the integration test.
        
        Returns:
            Tuple of (success, error_message, metrics)
        """
        raise NotImplementedError("Subclasses must implement run_test")
        
    def run(self) -> TestResult:
        """Run the integration test.
        
        Returns:
            Test result
        """
        # Create test result
        result = TestResult(
            test_name=self.test_name,
            component=self.component_name,
            test_type="integration"
        )
        
        try:
            # Set up test
            self.setup()
            
            # Run test
            success, error, metrics = self.run_test()
            
            # Add metrics to result
            for metric, value in metrics.items():
                result.add_metric(metric, value)
                
            # Mark as successful or failed
            result.complete(success, error)
            
        except Exception as e:
            # Mark as failed
            result.complete(False, str(e))
            
        finally:
            # Clean up
            self.teardown()
            
        return result


class RegressionTest:
    """Base class for regression tests."""
    
    def __init__(self, 
                component_name: str, 
                test_name: str,
                baseline_file: Optional[str] = None):
        """Initialize regression test.
        
        Args:
            component_name: Name of the component being tested
            test_name: Name of the test
            baseline_file: Optional path to baseline results file
        """
        self.component_name = component_name
        self.test_name = test_name
        self.baseline_file = baseline_file
        self.baseline_metrics: Dict[str, Any] = {}
        
        # Load baseline if provided
        if baseline_file and os.path.exists(baseline_file):
            self._load_baseline()
        
    def setup(self):
        """Set up the test environment."""
        pass
        
    def teardown(self):
        """Clean up after the test."""
        pass
        
    def run_test(self) -> Dict[str, Any]:
        """Run the regression test.
        
        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError("Subclasses must implement run_test")
        
    def _load_baseline(self):
        """Load baseline metrics from file."""
        try:
            with open(self.baseline_file, 'r') as f:
                self.baseline_metrics = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            self.baseline_metrics = {}
        
    def _compare_with_baseline(self, current_metrics: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Compare current metrics with baseline.
        
        Args:
            current_metrics: Current test metrics
            
        Returns:
            Tuple of (passed, comparison_results)
        """
        if not self.baseline_metrics:
            return True, {}
            
        # Compare metrics
        comparisons = {}
        all_passed = True
        
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                
                # Skip non-numeric values
                if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                    continue
                    
                # Calculate difference
                abs_diff = abs(current_value - baseline_value)
                rel_diff = abs_diff / abs(baseline_value) if baseline_value != 0 else float('inf')
                
                # Determine if this metric passed
                # Default threshold: 5% relative difference
                threshold = 0.05
                passed = rel_diff <= threshold
                
                comparisons[metric] = {
                    "current": current_value,
                    "baseline": baseline_value,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "passed": passed
                }
                
                if not passed:
                    all_passed = False
        
        return all_passed, comparisons
        
    def run(self) -> TestResult:
        """Run the regression test.
        
        Returns:
            Test result
        """
        # Create test result
        result = TestResult(
            test_name=self.test_name,
            component=self.component_name,
            test_type="regression"
        )
        
        try:
            # Set up test
            self.setup()
            
            # Run test
            current_metrics = self.run_test()
            
            # Add current metrics to result
            for metric, value in current_metrics.items():
                result.add_metric(metric, value)
                
            # Compare with baseline if available
            if self.baseline_metrics:
                passed, comparisons = self._compare_with_baseline(current_metrics)
                result.add_detail("baseline_comparisons", comparisons)
                result.complete(passed)
            else:
                # No baseline to compare with
                result.add_detail("baseline_comparisons", {})
                result.complete(True)
            
        except Exception as e:
            # Mark as failed
            result.complete(False, str(e))
            
        finally:
            # Clean up
            self.teardown()
            
        return result
        
    def save_as_baseline(self, metrics: Dict[str, Any], output_file: Optional[str] = None):
        """Save metrics as new baseline.
        
        Args:
            metrics: Metrics to save
            output_file: Optional output file path
        """
        if output_file is None:
            if self.baseline_file:
                output_file = self.baseline_file
            else:
                # Generate default filename
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_file = f"{self.component_name}_{self.test_name}_baseline_{timestamp}.json"
                
        # Save metrics
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Baseline saved to {output_file}")


def time_execution(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # If result is a dictionary, add execution time
        if isinstance(result, dict):
            result["execution_time"] = execution_time
            
        return result
    return wrapper


class SentimentTestingFramework:
    """Main testing framework for sentiment analysis."""
    
    def __init__(self, output_dir: str = "/tmp/sentiment_test_results"):
        """Initialize testing framework.
        
        Args:
            output_dir: Directory for test results
        """
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def create_test_suite(self, name: str) -> TestSuite:
        """Create a new test suite.
        
        Args:
            name: Name of the test suite
            
        Returns:
            New test suite
        """
        return TestSuite(name, self.output_dir)
        
    def load_test_results(self, results_file: str) -> Dict[str, Any]:
        """Load test results from file.
        
        Args:
            results_file: Path to results file
            
        Returns:
            Test results dictionary
        """
        with open(results_file, 'r') as f:
            return json.load(f)
            
    def compare_results(self, 
                       results1: Dict[str, Any], 
                       results2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two test result sets.
        
        Args:
            results1: First results set
            results2: Second results set
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            "suite1": results1.get("suite_name", "unknown"),
            "suite2": results2.get("suite_name", "unknown"),
            "total_tests1": results1.get("total_tests", 0),
            "total_tests2": results2.get("total_tests", 0),
            "passed_tests1": results1.get("passed_tests", 0),
            "passed_tests2": results2.get("passed_tests", 0),
            "success_rate1": results1.get("passed_tests", 0) / results1.get("total_tests", 1),
            "success_rate2": results2.get("passed_tests", 0) / results2.get("total_tests", 1),
            "test_comparisons": []
        }
        
        # Create lookup for results2 tests
        results2_lookup = {
            r["test_name"]: r for r in results2.get("results", [])
        }
        
        # Compare individual tests
        for test1 in results1.get("results", []):
            test_name = test1["test_name"]
            
            if test_name in results2_lookup:
                test2 = results2_lookup[test_name]
                
                # Compare metrics
                metric_comparisons = {}
                for metric, value1 in test1.get("metrics", {}).items():
                    if metric in test2.get("metrics", {}):
                        value2 = test2["metrics"][metric]
                        
                        # Skip non-numeric values
                        if not isinstance(value1, (int, float)) or not isinstance(value2, (int, float)):
                            continue
                            
                        # Calculate difference
                        abs_diff = abs(value1 - value2)
                        rel_diff = abs_diff / abs(value1) if value1 != 0 else float('inf')
                        
                        metric_comparisons[metric] = {
                            "value1": value1,
                            "value2": value2,
                            "abs_diff": abs_diff,
                            "rel_diff": rel_diff
                        }
                
                comparison["test_comparisons"].append({
                    "test_name": test_name,
                    "success1": test1["success"],
                    "success2": test2["success"],
                    "duration1": test1["duration"],
                    "duration2": test2["duration"],
                    "metric_comparisons": metric_comparisons
                })
        
        return comparison
        
    def visualize_comparison(self, 
                            comparison: Dict[str, Any], 
                            output_file: Optional[str] = None) -> str:
        """Visualize comparison between two test result sets.
        
        Args:
            comparison: Comparison dictionary
            output_file: Optional output file path
            
        Returns:
            Path to visualization file
        """
        if output_file is None:
            # Generate default filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.output_dir, 
                f"comparison_{comparison['suite1']}_{comparison['suite2']}_{timestamp}.png"
            )
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        
        # Plot success rates
        ax1 = fig.add_subplot(2, 1, 1)
        labels = [comparison['suite1'], comparison['suite2']]
        success_rates = [comparison['success_rate1'], comparison['success_rate2']]
        
        bars = ax1.bar(labels, success_rates)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
            
        ax1.set_title("Success Rate Comparison")
        ax1.set_ylabel("Success Rate")
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y')
        
        # Plot metric comparisons
        if comparison["test_comparisons"]:
            # Find common metrics across tests
            common_metrics = set()
            for test_comp in comparison["test_comparisons"]:
                common_metrics.update(test_comp["metric_comparisons"].keys())
            
            # Select a few important metrics to visualize
            metrics_to_plot = list(common_metrics)[:5]  # Limit to 5 metrics
            
            if metrics_to_plot:
                ax2 = fig.add_subplot(2, 1, 2)
                
                # Prepare data
                test_names = []
                metric_diffs = {metric: [] for metric in metrics_to_plot}
                
                for test_comp in comparison["test_comparisons"]:
                    test_names.append(test_comp["test_name"])
                    
                    for metric in metrics_to_plot:
                        if metric in test_comp["metric_comparisons"]:
                            # Use relative difference
                            rel_diff = test_comp["metric_comparisons"][metric]["rel_diff"]
                            metric_diffs[metric].append(rel_diff)
                        else:
                            metric_diffs[metric].append(0)
                
                # Plot relative differences
                x = np.arange(len(test_names))
                width = 0.8 / len(metrics_to_plot)
                
                for i, metric in enumerate(metrics_to_plot):
                    offset = (i - len(metrics_to_plot)/2 + 0.5) * width
                    ax2.bar(x + offset, metric_diffs[metric], width, label=metric)
                
                ax2.set_title("Relative Metric Differences")
                ax2.set_xlabel("Test")
                ax2.set_ylabel("Relative Difference")
                ax2.set_xticks(x)
                ax2.set_xticklabels(test_names, rotation=45, ha="right")
                ax2.legend()
                ax2.grid(axis='y')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file)
        plt.close()
        
        return output_file
