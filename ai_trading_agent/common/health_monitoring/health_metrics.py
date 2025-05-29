"""
Health Metrics for Health Monitoring System.

This module implements performance metric tracking and threshold monitoring
for detecting performance degradation and potential issues in the trading system.
"""

import threading
import time
import logging
import uuid
import statistics
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Deque

# Import from core definitions to avoid circular dependencies
from .core_definitions import ThresholdType, AlertSeverity
from .health_status import AlertData

# Set up logger
logger = logging.getLogger(__name__)


class MetricThreshold:
    """
    Threshold configuration for metric monitoring.
    
    Defines when alerts should be triggered based on metric values.
    """
    
    def __init__(
        self,
        metric_name: str,
        threshold_type: ThresholdType,
        warning_threshold: float,
        critical_threshold: float,
        duration: float = 0.0,
        component_id: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Initialize a metric threshold.
        
        Args:
            metric_name: Name of the metric to monitor
            threshold_type: Type of threshold comparison
            warning_threshold: Threshold for warning alerts
            critical_threshold: Threshold for critical alerts
            duration: Duration threshold must be violated before alerting (seconds)
            component_id: Optional component ID if specific to a component
            description: Optional description of the threshold
        """
        self.metric_name = metric_name
        self.threshold_type = threshold_type
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.duration = duration
        self.component_id = component_id
        self.description = description or f"Threshold for {metric_name}"
        
        # Internal state
        self.violation_start_time = None
        self.current_severity = None
    
    def check_threshold(self, value: float, timestamp: float) -> Optional[AlertSeverity]:
        """
        Check if a metric value violates the threshold.
        
        Args:
            value: Current metric value
            timestamp: Current timestamp
            
        Returns:
            AlertSeverity if threshold is violated, None otherwise
        """
        # Check if value violates threshold
        severity = self._check_value(value)
        
        # If no violation, reset tracking
        if severity is None:
            self.violation_start_time = None
            self.current_severity = None
            return None
            
        # If this is the start of a violation, record the time
        if self.violation_start_time is None:
            self.violation_start_time = timestamp
            self.current_severity = severity
            
        # If we need duration, check if enough time has elapsed
        if self.duration > 0:
            elapsed = timestamp - self.violation_start_time
            if elapsed < self.duration:
                return None
        
        # Return the current severity if the duration requirement is met
        return self.current_severity
    
    def _check_value(self, value: float) -> Optional[AlertSeverity]:
        """
        Check if a value violates the threshold.
        
        Args:
            value: Value to check
            
        Returns:
            AlertSeverity if threshold is violated, None otherwise
        """
        if self.threshold_type == ThresholdType.UPPER:
            if value >= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= self.warning_threshold:
                return AlertSeverity.WARNING
                
        elif self.threshold_type == ThresholdType.LOWER:
            if value <= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= self.warning_threshold:
                return AlertSeverity.WARNING
                
        elif self.threshold_type == ThresholdType.EQUALITY:
            if value == self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value == self.warning_threshold:
                return AlertSeverity.WARNING
                
        return None


class MetricValue:
    """
    Time-series data for a single metric.
    
    Stores historical values and provides statistical analysis.
    """
    
    def __init__(
        self,
        name: str,
        max_history: int = 100,
        reservoir_size: int = 1000
    ):
        """
        Initialize metric value storage.
        
        Args:
            name: Name of the metric
            max_history: Maximum number of recent values to store in time-order
            reservoir_size: Maximum number of values to store for statistics
        """
        self.name = name
        self.max_history = max_history
        self.reservoir_size = reservoir_size
        
        # Recent values in time order (for trend analysis)
        self.recent_values: Deque[Tuple[float, float]] = deque(maxlen=max_history)
        
        # Statistical reservoir (for accurate statistics with long history)
        self.reservoir: List[float] = []
        
        # Current statistics
        self.count = 0
        self.sum = 0.0
        self.min = None
        self.max = None
        self.last_value = None
        self.last_timestamp = None
    
    def add_value(self, value: float, timestamp: float = None) -> None:
        """
        Add a new metric value.
        
        Args:
            value: Metric value to add
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Update recent values
        self.recent_values.append((timestamp, value))
        
        # Update reservoir (with reservoir sampling for large datasets)
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(value)
        else:
            # Reservoir sampling - randomly replace with decreasing probability
            self.count += 1
            idx = int(self.count * random.random())
            if idx < self.reservoir_size:
                self.reservoir[idx] = value
        
        # Update statistics
        self.sum += value
        self.last_value = value
        self.last_timestamp = timestamp
        
        if self.min is None or value < self.min:
            self.min = value
            
        if self.max is None or value > self.max:
            self.max = value
    
    def get_average(self) -> Optional[float]:
        """
        Calculate the average of all values.
        
        Returns:
            Average value or None if no values
        """
        if not self.reservoir:
            return None
            
        return self.sum / len(self.reservoir)
    
    def get_median(self) -> Optional[float]:
        """
        Calculate the median value.
        
        Returns:
            Median value or None if no values
        """
        if not self.reservoir:
            return None
            
        return statistics.median(self.reservoir)
    
    def get_percentile(self, percentile: float) -> Optional[float]:
        """
        Calculate a percentile value.
        
        Args:
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value or None if no values
        """
        if not self.reservoir:
            return None
            
        return statistics.quantiles(self.reservoir, n=100)[int(percentile)-1]
    
    def get_stddev(self) -> Optional[float]:
        """
        Calculate the standard deviation.
        
        Returns:
            Standard deviation or None if not enough values
        """
        if len(self.reservoir) < 2:
            return None
            
        return statistics.stdev(self.reservoir)
    
    def get_change_rate(self) -> Optional[float]:
        """
        Calculate the rate of change.
        
        Returns:
            Rate of change per second or None if not enough values
        """
        if len(self.recent_values) < 2:
            return None
            
        first_time, first_value = self.recent_values[0]
        last_time, last_value = self.recent_values[-1]
        
        time_diff = last_time - first_time
        if time_diff <= 0:
            return 0.0
            
        value_diff = last_value - first_value
        return value_diff / time_diff
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get all statistics for the metric.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "name": self.name,
            "count": len(self.reservoir),
            "last_value": self.last_value,
            "last_timestamp": self.last_timestamp,
            "min": self.min,
            "max": self.max,
            "avg": self.get_average(),
            "median": self.get_median(),
            "stddev": self.get_stddev(),
            "change_rate": self.get_change_rate(),
            "p90": self.get_percentile(90) if len(self.reservoir) >= 10 else None,
            "p95": self.get_percentile(95) if len(self.reservoir) >= 20 else None,
            "p99": self.get_percentile(99) if len(self.reservoir) >= 100 else None
        }


class HealthMetrics:
    """
    Collects and analyzes performance metrics for health monitoring.
    
    Tracks metrics, applies thresholds, and generates alerts when
    metrics indicate potential performance problems.
    """
    
    def __init__(
        self,
        alert_callback: Optional[Callable[[AlertData], None]] = None,
        max_history: int = 100,
        check_interval: float = 10.0
    ):
        """
        Initialize health metrics monitoring.
        
        Args:
            alert_callback: Callback function for alerts
            max_history: Maximum history size for each metric
            check_interval: Interval for checking thresholds in seconds
        """
        self.alert_callback = alert_callback
        self.max_history = max_history
        self.check_interval = check_interval
        
        self.metrics = {}  # Dict[str, Dict[str, MetricValue]]
        self.thresholds = []  # List[MetricThreshold]
        
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
    
    def add_metric(
        self,
        component_id: str,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Add a metric value for a component.
        
        Args:
            component_id: ID of the component
            metric_name: Name of the metric
            value: Current metric value
            timestamp: Optional timestamp (defaults to current time)
        """
        with self._lock:
            # Create component dict if it doesn't exist
            if component_id not in self.metrics:
                self.metrics[component_id] = {}
                
            # Create metric if it doesn't exist
            if metric_name not in self.metrics[component_id]:
                self.metrics[component_id][metric_name] = MetricValue(
                    name=metric_name,
                    max_history=self.max_history
                )
                
            # Add value to metric
            self.metrics[component_id][metric_name].add_value(
                value=value,
                timestamp=timestamp
            )
    
    def add_threshold(self, threshold: MetricThreshold) -> None:
        """
        Add a threshold for metric monitoring.
        
        Args:
            threshold: Threshold configuration
        """
        with self._lock:
            self.thresholds.append(threshold)
    
    def remove_threshold(
        self,
        metric_name: str,
        component_id: Optional[str] = None
    ) -> bool:
        """
        Remove a threshold.
        
        Args:
            metric_name: Name of the metric
            component_id: Optional component ID
            
        Returns:
            True if threshold was removed, False if not found
        """
        with self._lock:
            for i, threshold in enumerate(self.thresholds):
                if (threshold.metric_name == metric_name and
                    (component_id is None or threshold.component_id == component_id)):
                    self.thresholds.pop(i)
                    return True
                    
            return False
    
    def get_metrics(
        self,
        component_id: Optional[str] = None,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metrics with optional filtering.
        
        Args:
            component_id: Optional component ID to filter by
            metric_name: Optional metric name to filter by
            
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            result = {}
            
            # Filter by component if specified
            if component_id is not None:
                if component_id not in self.metrics:
                    return {}
                
                components = {component_id: self.metrics[component_id]}
            else:
                components = self.metrics
                
            # Build result
            for comp_id, comp_metrics in components.items():
                result[comp_id] = {}
                
                # Filter by metric name if specified
                if metric_name is not None:
                    if metric_name not in comp_metrics:
                        continue
                        
                    result[comp_id][metric_name] = comp_metrics[metric_name].get_statistics()
                else:
                    # Include all metrics
                    for metric_id, metric in comp_metrics.items():
                        result[comp_id][metric_id] = metric.get_statistics()
                        
            return result
    
    def start(self) -> None:
        """Start the metrics monitoring thread."""
        with self._lock:
            if self._running:
                logger.warning("Health metrics monitoring already running")
                return
                
            self._running = True
            self._thread = threading.Thread(
                target=self._monitor_metrics,
                name="HealthMetricsMonitor",
                daemon=True
            )
            self._thread.start()
            
            logger.info("Health metrics monitoring started")
    
    def stop(self) -> None:
        """Stop the metrics monitoring thread."""
        with self._lock:
            if not self._running:
                logger.warning("Health metrics monitoring already stopped")
                return
                
            self._running = False
            if self._thread:
                self._thread.join(timeout=5.0)
                self._thread = None
                
            logger.info("Health metrics monitoring stopped")
    
    def _monitor_metrics(self) -> None:
        """Background thread for monitoring metric thresholds."""
        logger.info("Metrics monitoring thread started")
        
        while self._running:
            try:
                self._check_thresholds()
            except Exception as e:
                logger.error(f"Error in metrics monitoring: {str(e)}")
                
            # Sleep for check interval
            time.sleep(self.check_interval)
            
        logger.info("Metrics monitoring thread stopped")
    
    def _check_thresholds(self) -> None:
        """Check all metrics against thresholds."""
        with self._lock:
            current_time = time.time()
            
            for threshold in self.thresholds:
                component_id = threshold.component_id
                metric_name = threshold.metric_name
                
                # If component-specific, check only that component
                if component_id is not None:
                    if component_id not in self.metrics:
                        continue
                        
                    if metric_name not in self.metrics[component_id]:
                        continue
                        
                    metric = self.metrics[component_id][metric_name]
                    self._check_metric_threshold(
                        metric=metric,
                        threshold=threshold,
                        component_id=component_id,
                        current_time=current_time
                    )
                else:
                    # Check all components
                    for comp_id, comp_metrics in self.metrics.items():
                        if metric_name not in comp_metrics:
                            continue
                            
                        metric = comp_metrics[metric_name]
                        self._check_metric_threshold(
                            metric=metric,
                            threshold=threshold,
                            component_id=comp_id,
                            current_time=current_time
                        )
    
    def _check_metric_threshold(
        self,
        metric: MetricValue,
        threshold: MetricThreshold,
        component_id: str,
        current_time: float
    ) -> None:
        """
        Check a specific metric against a threshold.
        
        Args:
            metric: Metric to check
            threshold: Threshold to apply
            component_id: ID of the component
            current_time: Current timestamp
        """
        # Get the value to check based on threshold type
        if threshold.threshold_type == ThresholdType.CHANGE_RATE:
            value = metric.get_change_rate()
        elif threshold.threshold_type == ThresholdType.STANDARD_DEVIATION:
            stddev = metric.get_stddev()
            avg = metric.get_average()
            if stddev is None or avg is None or metric.last_value is None:
                return
                
            # Calculate z-score
            value = abs((metric.last_value - avg) / stddev)
        else:
            # Use last value for other threshold types
            value = metric.last_value
            
        # Skip if no value
        if value is None:
            return
            
        # Check threshold
        severity = threshold.check_threshold(value, current_time)
        
        # Generate alert if threshold violated
        if severity is not None and self.alert_callback:
            alert_id = f"metric_{component_id}_{metric.name}_{uuid.uuid4().hex[:8]}"
            
            if severity == AlertSeverity.WARNING:
                threshold_value = threshold.warning_threshold
            else:
                threshold_value = threshold.critical_threshold
                
            message = (f"Metric {metric.name} for component {component_id} "
                      f"violated {threshold.threshold_type.value} threshold. "
                      f"Current value: {value:.4f}, Threshold: {threshold_value:.4f}")
                      
            details = {
                "component_id": component_id,
                "metric_name": metric.name,
                "current_value": value,
                "threshold_type": threshold.threshold_type.value,
                "threshold_value": threshold_value,
                "statistics": metric.get_statistics()
            }
            
            alert = AlertData(
                alert_id=alert_id,
                component_id=component_id,
                severity=severity,
                message=message,
                details=details
            )
            
            self.alert_callback(alert)
