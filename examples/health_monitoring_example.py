"""Example usage of the health monitoring system."""

import random
import time
from datetime import datetime

from src.common.health_monitoring import (
    HealthCheck,
    HealthStatus,
    MetricType,
    health_monitor
)


def check_api_health() -> bool:
    """Example health check for API service."""
    # Simulate API check
    return random.random() > 0.1  # 90% success rate


def check_database_health() -> bool:
    """Example health check for database."""
    # Simulate database check
    return random.random() > 0.05  # 95% success rate


def main():
    """Run health monitoring example."""
    # Add some health checks
    health_monitor.add_check(
        HealthCheck(
            name="api_health",
            check_func=check_api_health,
            description="Check if API service is responding",
            interval=30  # Check every 30 seconds
        )
    )
    
    health_monitor.add_check(
        HealthCheck(
            name="database_health",
            check_func=check_database_health,
            description="Check if database is accessible",
            interval=45  # Check every 45 seconds
        )
    )
    
    # Add a custom metric
    health_monitor.add_metric(
        name="requests.latency",
        metric_type=MetricType.HISTOGRAM,
        description="API request latency",
        unit="ms",
        thresholds={
            "warning": 1000,  # 1 second
            "critical": 5000  # 5 seconds
        }
    )
    
    # Simulate some activity
    try:
        while True:
            # Simulate request latency
            latency = random.gauss(500, 200)  # Mean 500ms, stddev 200ms
            health_monitor.update_metric("requests.latency", latency)
            
            # Get current health status
            status = health_monitor.get_status()
            print(f"\nHealth Status: {status['status']}")
            print("\nChecks:")
            for name, check in status['checks'].items():
                print(f"  {name}: {check['status']}")
            print("\nMetrics:")
            for name, metric in status['metrics'].items():
                print(f"  {name}: {metric['value']}")
                
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping health monitor...")
        health_monitor.stop()


if __name__ == "__main__":
    main()
