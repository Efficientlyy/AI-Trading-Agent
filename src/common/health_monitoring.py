"""Health monitoring system based on logging metrics.

This module provides health monitoring capabilities by analyzing log metrics
and system performance indicators to detect and report system health issues.
"""

import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

import psutil
import structlog

from src.common.alerting import alert_manager
from src.common.config import config
from src.common.logging import format_iso


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics that can be monitored."""
    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    RATE = "rate"  # Rate of events over time


class HealthCheck:
    """Individual health check that monitors a specific aspect of the system."""
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        description: str = "",
        interval: int = 60,
        timeout: int = 10,
        dependencies: List[str] = None
    ):
        """
        Initialize a health check.
        
        Args:
            name: Name of the health check
            check_func: Function that performs the check and returns True if healthy
            description: Description of what this check monitors
            interval: How often to run the check in seconds
            timeout: Maximum time in seconds for the check to complete
            dependencies: List of other health check names this check depends on
        """
        self.name = name
        self.check_func = check_func
        self.description = description
        self.interval = interval
        self.timeout = timeout
        self.dependencies = dependencies or []
        self.last_check_time = None
        self.last_status = HealthStatus.UNKNOWN
        self.last_error = None
        self.consecutive_failures = 0
        
    def run(self) -> HealthStatus:
        """
        Run the health check.
        
        Returns:
            Current health status
        """
        self.last_check_time = datetime.now()
        
        try:
            # Run the check with timeout
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.check_func)
                is_healthy = future.result(timeout=self.timeout)
                
            if is_healthy:
                self.last_status = HealthStatus.HEALTHY
                self.last_error = None
                self.consecutive_failures = 0
            else:
                self.last_status = HealthStatus.UNHEALTHY
                self.last_error = "Check returned False"
                self.consecutive_failures += 1
        except concurrent.futures.TimeoutError:
            self.last_status = HealthStatus.UNHEALTHY
            self.last_error = f"Check timed out after {self.timeout} seconds"
            self.consecutive_failures += 1
        except Exception as e:
            self.last_status = HealthStatus.UNHEALTHY
            self.last_error = str(e)
            self.consecutive_failures += 1
            
        return self.last_status
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert health check status to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "status": self.last_status.value if self.last_status else "unknown",
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures,
            "dependencies": self.dependencies
        }


class HealthMetric:
    """Metric that tracks a specific health indicator."""
    
    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        unit: str = "",
        thresholds: Dict[str, float] = None
    ):
        """
        Initialize a health metric.
        
        Args:
            name: Name of the metric
            metric_type: Type of metric
            description: Description of what this metric measures
            unit: Unit of measurement
            thresholds: Dict of threshold levels to values
        """
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self.unit = unit
        self.thresholds = thresholds or {}
        
        # Metric storage
        self.value = 0
        self.values = deque(maxlen=1000)  # For histogram
        self.last_update = None
        self.start_time = datetime.now()
        
    def update(self, value: float) -> None:
        """
        Update the metric value.
        
        Args:
            value: New value for the metric
        """
        self.last_update = datetime.now()
        
        if self.metric_type == MetricType.COUNTER:
            self.value += value
        elif self.metric_type == MetricType.GAUGE:
            self.value = value
        elif self.metric_type == MetricType.HISTOGRAM:
            self.values.append(value)
        elif self.metric_type == MetricType.RATE:
            self.values.append((self.last_update, value))
            
        # Check thresholds
        self._check_thresholds(value)
        
    def get_value(self) -> Union[float, Dict[str, float]]:
        """
        Get the current metric value.
        
        Returns:
            Current value or statistics for histogram/rate
        """
        if self.metric_type == MetricType.COUNTER:
            return self.value
        elif self.metric_type == MetricType.GAUGE:
            return self.value
        elif self.metric_type == MetricType.HISTOGRAM:
            if not self.values:
                return {"count": 0}
            return {
                "count": len(self.values),
                "min": min(self.values),
                "max": max(self.values),
                "avg": sum(self.values) / len(self.values)
            }
        elif self.metric_type == MetricType.RATE:
            if len(self.values) < 2:
                return 0
            first = self.values[0]
            last = self.values[-1]
            time_diff = (last[0] - first[0]).total_seconds()
            if time_diff == 0:
                return 0
            value_diff = last[1] - first[1]
            return value_diff / time_diff
            
    def _check_thresholds(self, value: float) -> None:
        """
        Check if value crosses any thresholds.
        
        Args:
            value: Value to check
        """
        if not self.thresholds:
            return
            
        # Get current value based on metric type
        if self.metric_type == MetricType.RATE:
            check_value = self.get_value()
        else:
            check_value = value
            
        # Check each threshold
        for level, threshold in self.thresholds.items():
            if check_value > threshold:
                alert_manager.process_metric(
                    self.name,
                    check_value,
                    level=level,
                    threshold=threshold
                )
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "description": self.description,
            "unit": self.unit,
            "value": self.get_value(),
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "thresholds": self.thresholds
        }


class HealthMonitor:
    """System health monitor that tracks various health indicators."""
    
    def __init__(self):
        """Initialize the health monitor."""
        self.logger = structlog.get_logger("health_monitor")
        self.checks = {}
        self.metrics = {}
        self.status = HealthStatus.UNKNOWN
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread = None
        
        # Initialize system metrics
        self._init_system_metrics()
        
    def _init_system_metrics(self):
        """Initialize built-in system metrics."""
        # CPU usage
        self.add_metric(
            name="system.cpu.usage",
            metric_type=MetricType.GAUGE,
            description="CPU usage percentage",
            unit="%",
            thresholds={
                "warning": 80,
                "critical": 90
            }
        )
        
        # Memory usage
        self.add_metric(
            name="system.memory.usage",
            metric_type=MetricType.GAUGE,
            description="Memory usage percentage",
            unit="%",
            thresholds={
                "warning": 80,
                "critical": 90
            }
        )
        
        # Disk usage
        self.add_metric(
            name="system.disk.usage",
            metric_type=MetricType.GAUGE,
            description="Disk usage percentage",
            unit="%",
            thresholds={
                "warning": 80,
                "critical": 90
            }
        )
        
        # Error rate
        self.add_metric(
            name="system.errors.rate",
            metric_type=MetricType.RATE,
            description="Rate of error log messages",
            unit="errors/sec",
            thresholds={
                "warning": 1,
                "critical": 5
            }
        )
        
    def add_check(self, check: HealthCheck) -> None:
        """
        Add a health check.
        
        Args:
            check: Health check to add
        """
        with self.lock:
            self.checks[check.name] = check
            self.logger.info(f"Added health check: {check.name}")
            
    def add_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        unit: str = "",
        thresholds: Dict[str, float] = None
    ) -> None:
        """
        Add a health metric.
        
        Args:
            name: Name of the metric
            metric_type: Type of metric
            description: Description of what this metric measures
            unit: Unit of measurement
            thresholds: Dict of threshold levels to values
        """
        with self.lock:
            metric = HealthMetric(name, metric_type, description, unit, thresholds)
            self.metrics[name] = metric
            self.logger.info(f"Added health metric: {name}")
            
    def update_metric(self, name: str, value: float) -> None:
        """
        Update a metric value.
        
        Args:
            name: Name of the metric
            value: New value
        """
        with self.lock:
            if name in self.metrics:
                self.metrics[name].update(value)
            else:
                self.logger.warning(f"Unknown metric: {name}")
                
    def get_status(self) -> Dict[str, Any]:
        """
        Get current health status.
        
        Returns:
            Dict with current health status
        """
        with self.lock:
            return {
                "status": self.status.value,
                "timestamp": datetime.now().isoformat(),
                "checks": {
                    name: check.to_dict()
                    for name, check in self.checks.items()
                },
                "metrics": {
                    name: metric.to_dict()
                    for name, metric in self.metrics.items()
                }
            }
            
    def _update_system_metrics(self):
        """Update built-in system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.update_metric("system.cpu.usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.update_metric("system.memory.usage", memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.update_metric("system.disk.usage", disk.percent)
        except Exception as e:
            self.logger.error("Error updating system metrics", error=str(e))
            
    def _run_checks(self):
        """Run all health checks."""
        with self.lock:
            # Track check results
            results = []
            
            # Run each check
            for check in self.checks.values():
                # Skip if not time for next check
                if (check.last_check_time and 
                    (datetime.now() - check.last_check_time).total_seconds() < check.interval):
                    results.append(check.last_status)
                    continue
                    
                # Skip if dependencies are not healthy
                skip = False
                for dep in check.dependencies:
                    if dep in self.checks:
                        dep_check = self.checks[dep]
                        if dep_check.last_status != HealthStatus.HEALTHY:
                            skip = True
                            break
                            
                if skip:
                    results.append(HealthStatus.UNKNOWN)
                    continue
                    
                # Run the check
                status = check.run()
                results.append(status)
                
            # Update overall status
            if not results:
                self.status = HealthStatus.UNKNOWN
            elif all(r == HealthStatus.HEALTHY for r in results):
                self.status = HealthStatus.HEALTHY
            elif any(r == HealthStatus.UNHEALTHY for r in results):
                self.status = HealthStatus.UNHEALTHY
            else:
                self.status = HealthStatus.DEGRADED
                
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Run health checks
                self._run_checks()
                
                # Log current status
                self.logger.info(
                    "Health status updated",
                    status=self.status.value,
                    metrics=len(self.metrics),
                    checks=len(self.checks)
                )
                
            except Exception as e:
                self.logger.error("Error in health monitor loop", error=str(e))
                
            # Sleep until next update
            time.sleep(config.get("system.health.update_interval", 60))
            
    def start(self):
        """Start the health monitor."""
        with self.lock:
            if self.running:
                return
                
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
            self.logger.info("Health monitor started")
            
    def stop(self):
        """Stop the health monitor."""
        with self.lock:
            if not self.running:
                return
                
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5.0)
            if self.monitor_thread.is_alive():
                self.logger.warning("Health monitor thread failed to stop gracefully")
            else:
                self.monitor_thread = None
                self.logger.info("Health monitor stopped")
    
    # Dashboard-specific helper methods
    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage as a percentage.
        
        Returns:
            CPU usage percentage (0-100)
        """
        try:
            return psutil.cpu_percent()
        except Exception as e:
            self.logger.error("Error getting CPU usage", error=str(e))
            return 0.0
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage as a percentage.
        
        Returns:
            Memory usage percentage (0-100)
        """
        try:
            return psutil.virtual_memory().percent
        except Exception as e:
            self.logger.error("Error getting memory usage", error=str(e))
            return 0.0
    
    def get_disk_usage(self) -> float:
        """
        Get current disk usage as a percentage.
        
        Returns:
            Disk usage percentage (0-100)
        """
        try:
            return psutil.disk_usage('/').percent
        except Exception as e:
            self.logger.error("Error getting disk usage", error=str(e))
            return 0.0
    
    def has_anomalies(self) -> bool:
        """
        Check if any anomalies are detected in the system.
        
        Returns:
            True if anomalies are present, False otherwise
        """
        # Check for unhealthy status in any checks
        for check_name, check in self.checks.items():
            if check.last_status == HealthStatus.UNHEALTHY:
                return True
        
        # Check for metrics exceeding thresholds
        for metric_name, metric in self.metrics.items():
            if metric.metric_type != MetricType.HISTOGRAM:
                value = metric.get_value()
                for level, threshold in metric.thresholds.items():
                    if level == "warning" or level == "critical":
                        if value >= threshold:
                            return True
        
        return False


# Global health monitor instance
health_monitor = HealthMonitor()

# Start monitoring when module is imported
health_monitor.start()
