"""Performance benchmarks for market regime detection."""

import numpy as np
from numpy import floating
import time
from typing import Dict, List, Optional, Tuple, Union, Any, cast
from numpy.typing import NDArray
from dataclasses import dataclass
from ..features.market_regime import MarketRegimeDetector

@dataclass
class BenchmarkResults:
    """Results from benchmarking regime detection methods."""
    method_name: str
    execution_time: float  # Store as Python float
    memory_usage: float   # Store as Python float
    accuracy: float       # Store as Python float
    stability: float      # Store as Python float
    regime_transitions: int

class RegimeBenchmarker:
    """Benchmark regime detection methods."""
    
    def __init__(self, detector: MarketRegimeDetector):
        """Initialize benchmarker.
        
        Args:
            detector: MarketRegimeDetector instance
        """
        self.detector = detector
        
    def benchmark_method(
        self,
        method_name: str,
        data: Dict[str, NDArray],
        n_runs: int = 10
    ) -> BenchmarkResults:
        """Benchmark a single regime detection method.
        
        Args:
            method_name: Name of the method to benchmark
            data: Dictionary of input data arrays
            n_runs: Number of benchmark runs
        
        Returns:
            BenchmarkResults object
        """
        execution_times = []
        memory_usages = []
        regime_labels = []
        
        for _ in range(n_runs):
            # Measure execution time
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Call the appropriate method
            if method_name == "hmm":
                labels, _ = self.detector.detect_regimes_hmm(
                    data['returns'],
                    data['volumes']
                )
            elif method_name == "gmm":
                labels, _ = self.detector.detect_regimes_gmm(
                    data['returns'],
                    data['volumes']
                )
            elif method_name == "volatility":
                labels = self.detector.detect_volatility_regimes(
                    data['returns']
                )
            elif method_name == "momentum":
                labels = self.detector.detect_momentum_regimes(
                    data['returns']
                )
            elif method_name == "liquidity":
                labels = self.detector.detect_liquidity_regimes(
                    data['volumes'],
                    data['spreads']
                )
            elif method_name == "sentiment":
                labels = self.detector.detect_sentiment_regimes(
                    data['returns'],
                    data['volumes']
                )
            elif method_name == "microstructure":
                labels = self.detector.detect_microstructure_regimes(
                    data['returns'],
                    data['volumes'],
                    data['spreads']
                )
            elif method_name == "tail_risk":
                labels = self.detector.detect_tail_risk_regimes(
                    data['returns']
                )
            elif method_name == "dispersion":
                labels = self.detector.detect_dispersion_regimes(
                    data['returns_matrix']
                )
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_times.append(end_time - start_time)
            memory_usages.append(end_memory - start_memory)
            regime_labels.append(labels)
        
        # Calculate metrics and ensure they are Python floats
        avg_execution_time = float(np.mean(execution_times))
        avg_memory_usage = float(np.mean(memory_usages))
        
        # Calculate regime stability
        stability = float(self._calculate_stability(regime_labels))
        
        # Calculate accuracy (if ground truth available)
        accuracy = float(self._calculate_accuracy(
            regime_labels[0],
            data.get('true_labels', None)
        ))
        
        # Count regime transitions
        transitions = self._count_transitions(regime_labels[0])
        
        return BenchmarkResults(
            method_name=method_name,
            execution_time=avg_execution_time,
            memory_usage=avg_memory_usage,
            accuracy=accuracy,
            stability=stability,
            regime_transitions=transitions
        )
    
    def benchmark_all_methods(
        self,
        data: Dict[str, NDArray],
        methods: Optional[List[str]] = None
    ) -> Dict[str, BenchmarkResults]:
        """Benchmark multiple regime detection methods.
        
        Args:
            data: Dictionary of input data arrays
            methods: Optional list of methods to benchmark
        
        Returns:
            Dictionary of benchmark results
        """
        if methods is None:
            methods = [
                "hmm", "gmm", "volatility", "momentum",
                "liquidity", "sentiment", "microstructure",
                "tail_risk", "dispersion"
            ]
        
        results = {}
        for method in methods:
            try:
                results[method] = self.benchmark_method(method, data)
            except Exception as e:
                print(f"Error benchmarking {method}: {str(e)}")
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return float(process.memory_info().rss / 1024 / 1024)  # Explicit cast to float
    
    def _calculate_stability(
        self,
        regime_labels: List[NDArray[np.int64]]
    ) -> float:
        """Calculate stability of regime detection.
        
        Args:
            regime_labels: List of regime label arrays from multiple runs
        
        Returns:
            Stability score between 0 and 1
        """
        n_runs = len(regime_labels)
        if n_runs < 2:
            return 1.0
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                agreement = float(np.mean(regime_labels[i] == regime_labels[j]))
                agreements.append(agreement)
        
        return float(np.mean(agreements))
    
    def _calculate_accuracy(
        self,
        predicted_labels: NDArray[np.int64],
        true_labels: Optional[NDArray[np.int64]]
    ) -> float:
        """Calculate accuracy if ground truth is available.
        
        Args:
            predicted_labels: Predicted regime labels
            true_labels: Optional ground truth labels
        
        Returns:
            Accuracy score between 0 and 1
        """
        if true_labels is None:
            return float('nan')
        
        return float(np.mean(predicted_labels == true_labels))
    
    def _count_transitions(
        self,
        labels: NDArray[np.int64]
    ) -> int:
        """Count number of regime transitions.
        
        Args:
            labels: Array of regime labels
        
        Returns:
            Number of regime transitions
        """
        return np.sum(np.diff(labels) != 0)
    
    def generate_report(
        self,
        results: Dict[str, BenchmarkResults]
    ) -> str:
        """Generate benchmark report.
        
        Args:
            results: Dictionary of benchmark results
        
        Returns:
            Formatted report string
        """
        report = "Market Regime Detection Benchmark Report\n"
        report += "=" * 40 + "\n\n"
        
        # Summary table
        report += "Performance Summary:\n"
        report += "-" * 80 + "\n"
        report += f"{'Method':<15} {'Time (s)':<10} {'Memory (MB)':<12} "
        report += f"{'Stability':<10} {'Transitions':<12} {'Accuracy':<10}\n"
        report += "-" * 80 + "\n"
        
        for method, result in results.items():
            report += (
                f"{method:<15} "
                f"{result.execution_time:>9.3f} "
                f"{result.memory_usage:>11.1f} "
                f"{result.stability:>9.2f} "
                f"{result.regime_transitions:>11d} "
                f"{result.accuracy:>9.2f}\n"
            )
        
        report += "-" * 80 + "\n\n"
        
        # Detailed analysis
        report += "Detailed Analysis:\n"
        report += "-" * 40 + "\n"
        
        # Find best performing methods using explicit type comparisons
        fastest = min(
            results.items(),
            key=lambda x: float(x[1].execution_time)
        )
        most_stable = max(
            results.items(),
            key=lambda x: float(x[1].stability)
        )
        most_accurate = max(
            results.items(),
            key=lambda x: float(x[1].accuracy) if not np.isnan(x[1].accuracy) else -1.0
        )
        
        report += f"Fastest method: {fastest[0]} ({fastest[1].execution_time:.3f}s)\n"
        report += f"Most stable: {most_stable[0]} ({most_stable[1].stability:.2f})\n"
        if not np.isnan(most_accurate[1].accuracy):
            report += f"Most accurate: {most_accurate[0]} "
            report += f"({most_accurate[1].accuracy:.2f})\n"
        
        return report 