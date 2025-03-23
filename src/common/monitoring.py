"""Monitoring module for sentiment analysis system.

This module provides functionality for collecting and reporting system metrics
to monitor performance, health, and usage patterns.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

class MetricsCollector:
    """Collects and reports system metrics."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.gauges: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.histograms: Dict[str, List[float]] = {}
        
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        key = self._get_key(name, tags)
        self.gauges[key] = value
        
    def counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.
        
        Args:
            name: Metric name
            value: Increment value
            tags: Optional tags
        """
        key = self._get_key(name, tags)
        if key not in self.counters:
            self.counters[key] = 0
        self.counters[key] += value
        
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        key = self._get_key(name, tags)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
        
        # Limit histogram size
        max_size = 1000
        if len(self.histograms[key]) > max_size:
            self.histograms[key] = self.histograms[key][-max_size:]
            
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> Callable:
        """Create a timer context manager.
        
        Args:
            name: Metric name
            tags: Optional tags
            
        Returns:
            A context manager that times the enclosed code
        """
        start_time = time.time()
        
        def _timer_end():
            duration = time.time() - start_time
            self.histogram(name, duration, tags)
            return duration
            
        return _timer_end
        
    def _get_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Get a unique key for a metric.
        
        Args:
            name: Metric name
            tags: Optional tags
            
        Returns:
            A unique key
        """
        if not tags:
            return name
            
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics.
        
        Returns:
            Dictionary of all metrics
        """
        result = {
            "gauges": self.gauges.copy(),
            "counters": self.counters.copy(),
            "histograms": {
                k: {
                    "count": len(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "avg": sum(v) / len(v) if v else 0
                }
                for k, v in self.histograms.items()
            }
        }
        
        return result


# Global metrics collector
metrics = MetricsCollector()
