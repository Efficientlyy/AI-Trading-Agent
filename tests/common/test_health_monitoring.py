"""Unit tests for the health monitoring system."""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.common.health_monitoring import (
    HealthCheck,
    HealthMetric,
    HealthMonitor,
    HealthStatus,
    MetricType
)


@pytest.fixture
def health_monitor():
    """Create a health monitor instance for testing."""
    monitor = HealthMonitor()
    # Don't start the monitor thread
    monitor.running = False
    return monitor


def test_health_check():
    """Test basic health check functionality."""
    check_count = 0
    
    def check_func():
        nonlocal check_count
        check_count += 1
        return check_count < 3  # Fail after 2 successful checks
        
    check = HealthCheck(
        name="test_check",
        check_func=check_func,
        description="Test check",
        interval=1
    )
    
    # First check should pass
    assert check.run() == HealthStatus.HEALTHY
    assert check.consecutive_failures == 0
    
    # Second check should pass
    assert check.run() == HealthStatus.HEALTHY
    assert check.consecutive_failures == 0
    
    # Third check should fail
    assert check.run() == HealthStatus.UNHEALTHY
    assert check.consecutive_failures == 1


def test_health_check_timeout():
    """Test health check timeout."""
    def slow_check():
        time.sleep(2)
        return True
        
    check = HealthCheck(
        name="slow_check",
        check_func=slow_check,
        timeout=1
    )
    
    assert check.run() == HealthStatus.UNHEALTHY
    assert "timed out" in check.last_error.lower()


def test_health_check_exception():
    """Test health check exception handling."""
    def failing_check():
        raise ValueError("Test error")
        
    check = HealthCheck(
        name="failing_check",
        check_func=failing_check
    )
    
    assert check.run() == HealthStatus.UNHEALTHY
    assert check.last_error == "Test error"


def test_counter_metric():
    """Test counter metric type."""
    metric = HealthMetric(
        name="test_counter",
        metric_type=MetricType.COUNTER,
        description="Test counter"
    )
    
    metric.update(5)
    assert metric.get_value() == 5
    
    metric.update(3)
    assert metric.get_value() == 8


def test_gauge_metric():
    """Test gauge metric type."""
    metric = HealthMetric(
        name="test_gauge",
        metric_type=MetricType.GAUGE,
        description="Test gauge"
    )
    
    metric.update(5)
    assert metric.get_value() == 5
    
    metric.update(3)
    assert metric.get_value() == 3


def test_histogram_metric():
    """Test histogram metric type."""
    metric = HealthMetric(
        name="test_histogram",
        metric_type=MetricType.HISTOGRAM,
        description="Test histogram"
    )
    
    metric.update(1)
    metric.update(2)
    metric.update(3)
    
    stats = metric.get_value()
    assert stats["count"] == 3
    assert stats["min"] == 1
    assert stats["max"] == 3
    assert stats["avg"] == 2


def test_rate_metric():
    """Test rate metric type."""
    metric = HealthMetric(
        name="test_rate",
        metric_type=MetricType.RATE,
        description="Test rate"
    )
    
    # Simulate events over time
    now = datetime.now()
    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = now
        metric.update(0)
        
        mock_datetime.now.return_value = now + timedelta(seconds=10)
        metric.update(50)
        
    # Should be 5 events per second (50 events / 10 seconds)
    assert abs(metric.get_value() - 5.0) < 0.1


def test_metric_thresholds(health_monitor):
    """Test metric threshold alerts."""
    # Mock the alert manager
    mock_alert = MagicMock()
    with patch("src.common.alerting.alert_manager", mock_alert):
        metric = HealthMetric(
            name="test_metric",
            metric_type=MetricType.GAUGE,
            thresholds={
                "warning": 80,
                "critical": 90
            }
        )
        
        # No alert
        metric.update(70)
        mock_alert.process_metric.assert_not_called()
        
        # Warning alert
        metric.update(85)
        mock_alert.process_metric.assert_called_with(
            "test_metric", 85, level="warning", threshold=80
        )
        
        # Critical alert
        metric.update(95)
        mock_alert.process_metric.assert_called_with(
            "test_metric", 95, level="critical", threshold=90
        )


def test_health_monitor_status(health_monitor):
    """Test overall health monitor status."""
    def check1():
        return True
        
    def check2():
        return False
        
    health_monitor.add_check(
        HealthCheck("check1", check1, description="Always passes")
    )
    health_monitor.add_check(
        HealthCheck("check2", check2, description="Always fails")
    )
    
    # Run checks
    health_monitor._run_checks()
    
    # Should be unhealthy because one check failed
    assert health_monitor.status == HealthStatus.UNHEALTHY
    
    # Get full status
    status = health_monitor.get_status()
    assert status["status"] == "unhealthy"
    assert len(status["checks"]) == 2
    assert status["checks"]["check1"]["status"] == "healthy"
    assert status["checks"]["check2"]["status"] == "unhealthy"


def test_system_metrics(health_monitor):
    """Test system metrics collection."""
    # Mock psutil functions
    with patch("psutil.cpu_percent") as mock_cpu, \
         patch("psutil.virtual_memory") as mock_memory, \
         patch("psutil.disk_usage") as mock_disk:
        
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 60.0
        mock_disk.return_value.percent = 70.0
        
        # Update system metrics
        health_monitor._update_system_metrics()
        
        # Check metric values
        status = health_monitor.get_status()
        metrics = status["metrics"]
        
        assert metrics["system.cpu.usage"]["value"] == 50.0
        assert metrics["system.memory.usage"]["value"] == 60.0
        assert metrics["system.disk.usage"]["value"] == 70.0


def test_health_check_dependencies(health_monitor):
    """Test health check dependencies."""
    def check1():
        return True
        
    def check2():
        return True
        
    # Add checks with dependencies
    health_monitor.add_check(
        HealthCheck("check1", check1, description="First check")
    )
    health_monitor.add_check(
        HealthCheck(
            "check2",
            check2,
            description="Second check",
            dependencies=["check1"]
        )
    )
    
    # Run checks
    health_monitor._run_checks()
    
    # Both should pass
    status = health_monitor.get_status()
    assert status["checks"]["check1"]["status"] == "healthy"
    assert status["checks"]["check2"]["status"] == "healthy"
    
    # Make first check fail
    health_monitor.checks["check1"].check_func = lambda: False
    
    # Run checks again
    health_monitor._run_checks()
    
    # Second check should be unknown due to failed dependency
    status = health_monitor.get_status()
    assert status["checks"]["check1"]["status"] == "unhealthy"
    assert status["checks"]["check2"]["status"] == "unknown"
