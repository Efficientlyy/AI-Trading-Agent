"""Integration tests for the logging system components."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.common.health_monitoring import HealthCheck, HealthMonitor, HealthStatus
from src.common.log_query import LogQuery
from src.common.log_replay import LogReplay
from src.common.logging import get_logger


@pytest.fixture
def setup_test_environment():
    """Set up the test environment with log files and configuration."""
    # Create temp directory for logs
    log_dir = Path(tempfile.mkdtemp())
    
    # Generate sample logs
    logger = get_logger("test")
    
    # Request 1 logs
    logger.info(
        "Request started",
        request_id="req1",
        path="/api/data",
        method="GET"
    )
    logger.info(
        "Database query executed",
        request_id="req1",
        query_time=150,
        rows=10
    )
    logger.error(
        "Request failed",
        request_id="req1",
        error="Database timeout",
        status_code=500
    )
    
    # Request 2 logs
    logger.info(
        "Request started",
        request_id="req2",
        path="/api/users",
        method="POST"
    )
    logger.info(
        "Database query executed",
        request_id="req2",
        query_time=50,
        rows=1
    )
    logger.info(
        "Request completed",
        request_id="req2",
        response_time=120,
        status_code=200
    )
    
    # System logs
    system_logger = get_logger("system")
    system_logger.info(
        "System startup",
        version="1.0.0",
        environment="test"
    )
    system_logger.warning(
        "High CPU usage",
        cpu_percent=85.5,
        process_count=120
    )
    
    # Find log files
    log_files = list(Path("logs").glob("*.log"))
    
    yield {
        "log_dir": Path("logs"),
        "log_files": log_files
    }


def test_query_and_replay_integration(setup_test_environment):
    """Test integration between log query and replay systems."""
    log_dir = setup_test_environment["log_dir"]
    
    # Create a query to find error logs
    query = 'level = "error" AND component = "test"'
    log_query = LogQuery(query)
    
    # Search for errors
    error_logs = log_query.search_directory(
        directory=str(log_dir),
        pattern="*.log"
    )
    
    # Should find at least one error
    assert len(error_logs) >= 1
    assert error_logs[0]["level"] == "error"
    assert error_logs[0]["request_id"] == "req1"
    
    # Now replay just that request using the request ID
    replayed_entries = []
    
    def capture_entry(entry):
        replayed_entries.append(entry)
    
    replay = LogReplay(handlers={"default": capture_entry})
    replay.replay_by_request_id("req1")
    
    # Should replay all entries for request 1
    assert len(replayed_entries) >= 3
    assert all(entry["request_id"] == "req1" for entry in replayed_entries)
    
    # Find the error entry
    error_entries = [e for e in replayed_entries if e.get("level") == "error"]
    assert len(error_entries) == 1
    assert error_entries[0]["error"] == "Database timeout"


def test_health_monitoring_with_logs(setup_test_environment):
    """Test health monitoring integration with logging."""
    log_dir = setup_test_environment["log_dir"]
    
    # Set up a health check that looks for error logs
    def check_for_errors():
        query = 'level = "error"'
        log_query = LogQuery(query)
        
        # Search for recent errors (last minute)
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        errors = log_query.search_directory(
            directory=str(log_dir),
            pattern="*.log",
            start_time=one_minute_ago
        )
        
        # If we find recent errors, health check fails
        if errors:
            return False
        return True
    
    # Create health monitor
    monitor = HealthMonitor()
    monitor.add_check(
        HealthCheck(
            name="recent_errors",
            check_func=check_for_errors,
            description="Check for recent error logs",
            interval=30
        )
    )
    
    # Run health checks
    monitor._run_checks()
    
    # Should be unhealthy due to error logs
    status = monitor.get_status()
    assert status["status"] == "unhealthy"
    assert status["checks"]["recent_errors"]["status"] == "unhealthy"


def test_query_performance_monitoring():
    """Test monitoring query performance with health metrics."""
    # Create a health monitor
    monitor = HealthMonitor()
    
    # Add performance metrics
    monitor.add_metric(
        name="query.duration",
        metric_type="histogram",
        description="Query execution time"
    )
    
    # Create a query
    query = 'level = "info" OR level = "error"'
    log_query = LogQuery(query)
    
    # Run queries and track performance
    for _ in range(10):
        start_time = datetime.now()
        
        # Create a temporary log file with random entries
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            for i in range(100):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "info" if i % 5 != 0 else "error",
                    "message": f"Test message {i}",
                    "component": "test"
                }
                f.write(json.dumps(entry) + "\n")
        
        # Run query
        results = log_query.search_file(f.name)
        
        # Record duration
        duration = (datetime.now() - start_time).total_seconds() * 1000  # ms
        monitor.update_metric("query.duration", duration)
        
        # Clean up
        os.unlink(f.name)
    
    # Check metrics
    status = monitor.get_status()
    query_stats = status["metrics"]["query.duration"]["value"]
    
    # Verify we have performance data
    assert query_stats["count"] == 10
    assert query_stats["min"] > 0
    assert query_stats["max"] > 0
    assert query_stats["avg"] > 0


def test_end_to_end_workflow(setup_test_environment):
    """Test end-to-end workflow with all components."""
    log_dir = setup_test_environment["log_dir"]
    
    # 1. Generate new log entries
    logger = get_logger("workflow_test")
    for i in range(5):
        logger.info(
            f"Processing item {i}",
            request_id="workflow1",
            item_id=i
        )
    
    # Add an error
    logger.error(
        "Processing failed",
        request_id="workflow1",
        item_id=3,
        error="Validation error"
    )
    
    # 2. Query for the error
    query = 'level = "error" AND request_id = "workflow1"'
    log_query = LogQuery(query)
    
    error_logs = log_query.search_directory(
        directory=str(log_dir),
        pattern="*.log"
    )
    
    assert len(error_logs) == 1
    assert error_logs[0]["item_id"] == 3
    
    # 3. Set up health monitoring
    monitor = HealthMonitor()
    
    # Add check for workflow errors
    def check_workflow_errors():
        query = 'level = "error" AND request_id ~ "workflow"'
        log_query = LogQuery(query)
        
        errors = log_query.search_directory(
            directory=str(log_dir),
            pattern="*.log"
        )
        
        return len(errors) == 0
    
    monitor.add_check(
        HealthCheck(
            name="workflow_errors",
            check_func=check_workflow_errors,
            description="Check for workflow errors"
        )
    )
    
    # Run health checks
    monitor._run_checks()
    
    # Should be unhealthy due to workflow error
    status = monitor.get_status()
    assert status["status"] == "unhealthy"
    assert status["checks"]["workflow_errors"]["status"] == "unhealthy"
    
    # 4. Replay the workflow
    replayed_entries = []
    
    def capture_entry(entry):
        replayed_entries.append(entry)
    
    replay = LogReplay(handlers={"default": capture_entry})
    replay.replay_by_request_id("workflow1")
    
    # Should replay all workflow entries
    assert len(replayed_entries) >= 6  # 5 info + 1 error
    assert all(entry["request_id"] == "workflow1" for entry in replayed_entries)
