"""Unit tests for the monitoring module.

This module contains tests for the metrics collection system.
"""

import pytest
import time
from unittest.mock import patch

from src.common.monitoring import MetricsCollector


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""
    
    def test_gauge_metric(self):
        """Test recording and retrieving gauge metrics."""
        # Create metrics collector
        metrics = MetricsCollector()
        
        # Record gauge metrics
        metrics.gauge("cpu_usage", 45.2)
        metrics.gauge("memory_usage", 78.6, tags={"host": "server1"})
        
        # Get all metrics
        all_metrics = metrics.get_metrics()
        
        # Verify gauges
        assert "cpu_usage" in all_metrics["gauges"]
        assert all_metrics["gauges"]["cpu_usage"] == 45.2
        assert "memory_usage[host=server1]" in all_metrics["gauges"]
        assert all_metrics["gauges"]["memory_usage[host=server1]"] == 78.6
    
    def test_counter_metric(self):
        """Test incrementing and retrieving counter metrics."""
        # Create metrics collector
        metrics = MetricsCollector()
        
        # Increment counters
        metrics.counter("api_calls")  # Default increment by 1
        metrics.counter("api_calls")  # Another increment
        metrics.counter("errors", 5)  # Increment by 5
        metrics.counter("requests", 3, tags={"endpoint": "/users"})
        
        # Get all metrics
        all_metrics = metrics.get_metrics()
        
        # Verify counters
        assert "api_calls" in all_metrics["counters"]
        assert all_metrics["counters"]["api_calls"] == 2
        assert "errors" in all_metrics["counters"]
        assert all_metrics["counters"]["errors"] == 5
        assert "requests[endpoint=/users]" in all_metrics["counters"]
        assert all_metrics["counters"]["requests[endpoint=/users]"] == 3
    
    def test_histogram_metric(self):
        """Test recording and retrieving histogram metrics."""
        # Create metrics collector
        metrics = MetricsCollector()
        
        # Record histogram values
        metrics.histogram("response_time", 0.2)
        metrics.histogram("response_time", 0.3)
        metrics.histogram("response_time", 0.1)
        metrics.histogram("db_query_time", 0.05, tags={"query": "select"})
        metrics.histogram("db_query_time", 0.08, tags={"query": "select"})
        
        # Get all metrics
        all_metrics = metrics.get_metrics()
        
        # Verify histograms
        assert "response_time" in all_metrics["histograms"]
        assert all_metrics["histograms"]["response_time"]["count"] == 3
        assert all_metrics["histograms"]["response_time"]["min"] == 0.1
        assert all_metrics["histograms"]["response_time"]["max"] == 0.3
        assert all_metrics["histograms"]["response_time"]["avg"] == 0.2
        
        assert "db_query_time[query=select]" in all_metrics["histograms"]
        assert all_metrics["histograms"]["db_query_time[query=select]"]["count"] == 2
        assert all_metrics["histograms"]["db_query_time[query=select]"]["min"] == 0.05
        assert all_metrics["histograms"]["db_query_time[query=select]"]["max"] == 0.08
        assert all_metrics["histograms"]["db_query_time[query=select]"]["avg"] == 0.065
    
    def test_timer_function(self):
        """Test the timer function for measuring durations."""
        # Create metrics collector
        metrics = MetricsCollector()
        
        # Use timer
        timer_end = metrics.timer("operation_duration")
        time.sleep(0.1)  # Simulate operation
        duration = timer_end()
        
        # Get all metrics
        all_metrics = metrics.get_metrics()
        
        # Verify timer recorded a histogram
        assert "operation_duration" in all_metrics["histograms"]
        assert all_metrics["histograms"]["operation_duration"]["count"] == 1
        assert all_metrics["histograms"]["operation_duration"]["min"] > 0.05  # Should be at least 0.05s
        assert all_metrics["histograms"]["operation_duration"]["max"] > 0.05
        
        # Verify returned duration
        assert duration > 0.05
    
    def test_histogram_limit(self):
        """Test that histograms limit the number of values they store."""
        # Create metrics collector
        metrics = MetricsCollector()
        
        # Add many values to a histogram
        for i in range(1100):  # More than the 1000 limit
            metrics.histogram("test_histogram", i)
        
        # Get all metrics
        all_metrics = metrics.get_metrics()
        
        # Verify histogram was limited
        assert all_metrics["histograms"]["test_histogram"]["count"] == 1000
        assert all_metrics["histograms"]["test_histogram"]["min"] == 100  # Should have dropped the first 100
        assert all_metrics["histograms"]["test_histogram"]["max"] == 1099
    
    def test_key_generation(self):
        """Test key generation with different tag combinations."""
        # Create metrics collector
        metrics = MetricsCollector()
        
        # Record metrics with different tag combinations
        metrics.gauge("metric", 1)
        metrics.gauge("metric", 2, tags={"a": "1"})
        metrics.gauge("metric", 3, tags={"a": "1", "b": "2"})
        metrics.gauge("metric", 4, tags={"b": "2", "a": "1"})  # Same tags, different order
        
        # Get all metrics
        all_metrics = metrics.get_metrics()
        
        # Verify keys
        assert "metric" in all_metrics["gauges"]
        assert "metric[a=1]" in all_metrics["gauges"]
        assert "metric[a=1,b=2]" in all_metrics["gauges"]
        assert all_metrics["gauges"]["metric[a=1,b=2]"] == 4  # Last value with these tags
