"""
Performance Profiling Utility

This module provides tools for profiling and optimizing the performance of the trading agent,
particularly focusing on the indicator calculation and signal generation components.
"""

import time
import functools
import logging
from typing import Dict, Any, Callable, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import psutil
import gc
import tracemalloc

# Configure logger
logger = logging.getLogger("PerformanceProfiler")

class PerformanceProfiler:
    """
    A utility for profiling and analyzing the performance of trading agent components.
    
    Provides:
    - Function execution time tracking
    - Memory usage monitoring
    - Bottleneck identification
    - Performance reporting
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the performance profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.profiling_data = {}
        self.reset()
        
    def reset(self):
        """Reset all profiling data."""
        self.profiling_data = {
            "function_calls": {},
            "memory_snapshots": [],
            "execution_times": {},
            "bottlenecks": [],
        }
        
    def profile(self, component_name: str = None):
        """
        Decorator for profiling function execution time and call count.
        
        Args:
            component_name: Optional name for categorizing the function
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Record start time and memory
                start_time = time.time()
                
                # Call the function
                result = func(*args, **kwargs)
                
                # Record execution time
                execution_time = time.time() - start_time
                
                # Get function info
                func_name = func.__name__
                func_module = func.__module__
                
                # Create full identifier
                if component_name:
                    identifier = f"{component_name}.{func_name}"
                else:
                    identifier = f"{func_module}.{func_name}"
                
                # Update profiling data
                if identifier not in self.profiling_data["function_calls"]:
                    self.profiling_data["function_calls"][identifier] = {
                        "count": 0,
                        "total_time": 0,
                        "avg_time": 0,
                        "min_time": float('inf'),
                        "max_time": 0,
                        "last_execution": datetime.now().isoformat(),
                    }
                
                self.profiling_data["function_calls"][identifier]["count"] += 1
                self.profiling_data["function_calls"][identifier]["total_time"] += execution_time
                self.profiling_data["function_calls"][identifier]["avg_time"] = (
                    self.profiling_data["function_calls"][identifier]["total_time"] / 
                    self.profiling_data["function_calls"][identifier]["count"]
                )
                self.profiling_data["function_calls"][identifier]["min_time"] = min(
                    self.profiling_data["function_calls"][identifier]["min_time"],
                    execution_time
                )
                self.profiling_data["function_calls"][identifier]["max_time"] = max(
                    self.profiling_data["function_calls"][identifier]["max_time"],
                    execution_time
                )
                self.profiling_data["function_calls"][identifier]["last_execution"] = datetime.now().isoformat()
                
                # Check if this is a bottleneck (execution time > 100ms)
                if execution_time > 0.1:  # 100ms threshold
                    self.profiling_data["bottlenecks"].append({
                        "function": identifier,
                        "execution_time": execution_time,
                        "timestamp": datetime.now().isoformat(),
                        "args_summary": str(args)[:100] if args else "No args",
                    })
                
                return result
            return wrapper
        return decorator
    
    def start_memory_tracking(self):
        """Start memory usage tracking."""
        if not self.enabled:
            return
            
        tracemalloc.start()
        
    def take_memory_snapshot(self, label: str = None):
        """
        Take a snapshot of current memory usage.
        
        Args:
            label: Optional label for the snapshot
        """
        if not self.enabled or not tracemalloc.is_tracing():
            return
            
        snapshot = tracemalloc.take_snapshot()
        
        # Get process memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.profiling_data["memory_snapshots"].append({
            "timestamp": datetime.now().isoformat(),
            "label": label or f"Snapshot-{len(self.profiling_data['memory_snapshots'])}",
            "process_memory_mb": memory_info.rss / (1024 * 1024),  # Convert to MB
            "snapshot": snapshot,
        })
        
    def stop_memory_tracking(self):
        """Stop memory usage tracking."""
        if not self.enabled or not tracemalloc.is_tracing():
            return
            
        tracemalloc.stop()
        
    def compare_memory_snapshots(self, snapshot1_label: str, snapshot2_label: str, top_n: int = 10) -> List[Dict]:
        """
        Compare two memory snapshots to identify memory leaks or high usage.
        
        Args:
            snapshot1_label: Label of first snapshot
            snapshot2_label: Label of second snapshot
            top_n: Number of top differences to return
            
        Returns:
            List of memory usage differences by object
        """
        if not self.enabled:
            return []
            
        # Find snapshots
        snapshot1 = None
        snapshot2 = None
        
        for snapshot_data in self.profiling_data["memory_snapshots"]:
            if snapshot_data["label"] == snapshot1_label:
                snapshot1 = snapshot_data["snapshot"]
            if snapshot_data["label"] == snapshot2_label:
                snapshot2 = snapshot_data["snapshot"]
                
        if not snapshot1 or not snapshot2:
            logger.warning(f"Snapshots not found: {snapshot1_label} or {snapshot2_label}")
            return []
        
        # Compare snapshots
        comparison = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Format results
        results = []
        for stat in comparison[:top_n]:
            results.append({
                "file": stat.traceback[0].filename,
                "line": stat.traceback[0].lineno,
                "size_diff": stat.size_diff,
                "count_diff": stat.count_diff,
                "size_current": stat.size,
                "count_current": stat.count,
            })
            
        return results
    
    def identify_bottlenecks(self, threshold_ms: float = 100, top_n: int = 5) -> List[Dict]:
        """
        Identify performance bottlenecks based on execution time.
        
        Args:
            threshold_ms: Minimum execution time to be considered a bottleneck (in ms)
            top_n: Number of top bottlenecks to return
            
        Returns:
            List of bottleneck functions sorted by total time
        """
        if not self.enabled:
            return []
            
        bottlenecks = []
        threshold_sec = threshold_ms / 1000  # Convert to seconds
        
        for func_name, stats in self.profiling_data["function_calls"].items():
            if stats["avg_time"] > threshold_sec:
                bottlenecks.append({
                    "function": func_name,
                    "avg_time_ms": stats["avg_time"] * 1000,  # Convert to ms
                    "total_time_ms": stats["total_time"] * 1000,  # Convert to ms
                    "call_count": stats["count"],
                    "max_time_ms": stats["max_time"] * 1000,  # Convert to ms
                })
                
        # Sort by total execution time
        bottlenecks.sort(key=lambda x: x["total_time_ms"], reverse=True)
        
        return bottlenecks[:top_n]
    
    def get_function_stats(self, function_name: str = None) -> Dict:
        """
        Get execution statistics for a specific function or all functions.
        
        Args:
            function_name: Name of function to get stats for, or None for all
            
        Returns:
            Dictionary of function statistics
        """
        if not self.enabled:
            return {}
            
        if function_name:
            matching_funcs = {k: v for k, v in self.profiling_data["function_calls"].items() 
                             if function_name in k}
            return matching_funcs
        else:
            return self.profiling_data["function_calls"]
            
    def get_memory_usage(self) -> Dict:
        """
        Get current memory usage of the process.
        
        Returns:
            Dictionary with memory usage statistics
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident set size in MB
            "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual memory size in MB
            "percent": process.memory_percent(),
            "timestamp": datetime.now().isoformat(),
        }
        
    def generate_report(self, include_memory: bool = True) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Args:
            include_memory: Whether to include memory usage details
            
        Returns:
            Dictionary with full performance report
        """
        if not self.enabled:
            return {"error": "Profiling not enabled"}
            
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_functions_profiled": len(self.profiling_data["function_calls"]),
            "total_bottlenecks": len(self.profiling_data["bottlenecks"]),
            "top_bottlenecks": self.identify_bottlenecks(top_n=5),
            "execution_summary": {},
        }
        
        # Get top 10 functions by total execution time
        functions_by_time = sorted(
            self.profiling_data["function_calls"].items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )[:10]
        
        report["execution_summary"] = {
            "top_by_total_time": [{k: v} for k, v in functions_by_time],
            "top_by_avg_time": sorted(
                self.profiling_data["function_calls"].items(),
                key=lambda x: x[1]["avg_time"],
                reverse=True
            )[:5],
            "most_called": sorted(
                self.profiling_data["function_calls"].items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:5],
        }
        
        if include_memory:
            report["current_memory"] = self.get_memory_usage()
            
            # Memory growth between first and last snapshot if available
            if len(self.profiling_data["memory_snapshots"]) >= 2:
                first = self.profiling_data["memory_snapshots"][0]
                last = self.profiling_data["memory_snapshots"][-1]
                
                report["memory_growth"] = {
                    "start_mb": first["process_memory_mb"],
                    "end_mb": last["process_memory_mb"],
                    "growth_mb": last["process_memory_mb"] - first["process_memory_mb"],
                    "growth_percent": (
                        (last["process_memory_mb"] - first["process_memory_mb"]) / 
                        first["process_memory_mb"] * 100
                    ) if first["process_memory_mb"] > 0 else 0,
                }
        
        return report
    
    def optimize_memory(self):
        """
        Attempt to optimize memory usage by forcing garbage collection.
        
        Returns:
            Dictionary with memory before and after optimization
        """
        before = self.get_memory_usage()
        
        # Force garbage collection
        gc.collect()
        
        after = self.get_memory_usage()
        
        return {
            "before_mb": before["rss_mb"],
            "after_mb": after["rss_mb"],
            "saved_mb": before["rss_mb"] - after["rss_mb"],
            "percent_reduction": (
                (before["rss_mb"] - after["rss_mb"]) / before["rss_mb"] * 100
            ) if before["rss_mb"] > 0 else 0,
        }


# Singleton instance for easy access
profiler = PerformanceProfiler()

def profile_execution(component: str = None):
    """
    Decorator for profiling function execution.
    
    Args:
        component: Optional component name for categorization
        
    Returns:
        Decorated function with profiling
    """
    return profiler.profile(component)

class PerformanceContext:
    """Context manager for timing blocks of code."""
    
    def __init__(self, label: str, component: str = None):
        """
        Initialize context manager.
        
        Args:
            label: Label for this timing context
            component: Optional component name for categorization
        """
        self.label = label
        self.component = component
        self.start_time = None
        
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record results."""
        if not profiler.enabled or not self.start_time:
            return
            
        execution_time = time.time() - self.start_time
        
        # Create identifier
        if self.component:
            identifier = f"{self.component}.{self.label}"
        else:
            identifier = self.label
            
        # Record timing
        if "blocks" not in profiler.profiling_data:
            profiler.profiling_data["blocks"] = {}
            
        if identifier not in profiler.profiling_data["blocks"]:
            profiler.profiling_data["blocks"][identifier] = {
                "count": 0,
                "total_time": 0,
                "avg_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "last_execution": datetime.now().isoformat(),
            }
            
        profiler.profiling_data["blocks"][identifier]["count"] += 1
        profiler.profiling_data["blocks"][identifier]["total_time"] += execution_time
        profiler.profiling_data["blocks"][identifier]["avg_time"] = (
            profiler.profiling_data["blocks"][identifier]["total_time"] / 
            profiler.profiling_data["blocks"][identifier]["count"]
        )
        profiler.profiling_data["blocks"][identifier]["min_time"] = min(
            profiler.profiling_data["blocks"][identifier]["min_time"],
            execution_time
        )
        profiler.profiling_data["blocks"][identifier]["max_time"] = max(
            profiler.profiling_data["blocks"][identifier]["max_time"],
            execution_time
        )
        profiler.profiling_data["blocks"][identifier]["last_execution"] = datetime.now().isoformat()


def time_block(label: str, component: str = None):
    """
    Create a context manager for timing a block of code.
    
    Args:
        label: Label for this timing context
        component: Optional component name for categorization
        
    Returns:
        PerformanceContext context manager
    """
    return PerformanceContext(label, component)
