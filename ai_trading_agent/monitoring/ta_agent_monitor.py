"""
Technical Analysis Agent Monitoring Module

This module provides production monitoring and alerting for the Technical Analysis Agent,
including performance metrics, error tracking, and health checks.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
import threading
import json
import os
import pandas as pd
import numpy as np

from ..common.utils import get_logger
from ..agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ..config.data_source_config import get_data_source_config
from ..data.data_source_factory import get_data_source_factory

# Setup logging
logger = get_logger("TAAgentMonitor")

class PerformanceMetric:
    """Class for tracking performance metrics."""
    
    def __init__(self, name: str, description: str, unit: str = ""):
        self.name = name
        self.description = description
        self.unit = unit
        self.values = []
        self.timestamps = []
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        """Add a value to the metric."""
        self.values.append(value)
        self.timestamps.append(timestamp or datetime.now())
    
    def get_latest(self) -> Optional[float]:
        """Get the latest value."""
        if not self.values:
            return None
        return self.values[-1]
    
    def get_average(self, window: int = None) -> Optional[float]:
        """Get the average value over a window."""
        if not self.values:
            return None
        
        if window is not None and window < len(self.values):
            return sum(self.values[-window:]) / window
        
        return sum(self.values) / len(self.values)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "latest": self.get_latest(),
            "average": self.get_average(),
            "count": len(self.values)
        }

class HealthStatus:
    """Class for tracking health status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    
    def __init__(self, component: str):
        self.component = component
        self.status = self.HEALTHY
        self.last_check = datetime.now()
        self.message = "OK"
        self.error_count = 0
        self.recovery_count = 0
    
    def update(self, status: str, message: str = ""):
        """Update the health status."""
        old_status = self.status
        self.status = status
        self.last_check = datetime.now()
        self.message = message
        
        if old_status != self.UNHEALTHY and status == self.UNHEALTHY:
            self.error_count += 1
        
        if old_status == self.UNHEALTHY and status != self.UNHEALTHY:
            self.recovery_count += 1
    
    def is_healthy(self) -> bool:
        """Check if the component is healthy."""
        return self.status == self.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "status": self.status,
            "last_check": self.last_check.isoformat(),
            "message": self.message,
            "error_count": self.error_count,
            "recovery_count": self.recovery_count
        }

class TAAgentMonitor:
    """
    Monitoring system for the Technical Analysis Agent.
    """
    
    def __init__(
        self,
        ta_agent: Optional[AdvancedTechnicalAnalysisAgent] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the monitoring system.
        
        Args:
            ta_agent: Technical Analysis Agent instance
            config: Configuration dictionary
        """
        self.ta_agent = ta_agent or AdvancedTechnicalAnalysisAgent()
        self.config = config or {
            "monitoring_interval": 60,  # seconds
            "health_check_interval": 300,  # seconds
            "metrics_history": 1000,  # number of data points to keep
            "alert_thresholds": {
                "error_rate": 0.05,  # 5% of operations
                "latency": 2000,  # milliseconds
                "memory_usage": 1024 * 1024 * 1024,  # 1 GB
                "signal_confidence": 0.6  # minimum confidence for signals
            },
            "export_metrics": True,
            "metrics_dir": "metrics"
        }
        
        # Initialize metrics
        self.metrics = {
            "operation_count": PerformanceMetric("operation_count", "Number of operations performed", "count"),
            "error_count": PerformanceMetric("error_count", "Number of errors encountered", "count"),
            "average_latency": PerformanceMetric("average_latency", "Average operation latency", "ms"),
            "signal_count": PerformanceMetric("signal_count", "Number of signals generated", "count"),
            "pattern_count": PerformanceMetric("pattern_count", "Number of patterns detected", "count"),
            "memory_usage": PerformanceMetric("memory_usage", "Memory usage of the agent", "bytes")
        }
        
        # Initialize health status
        self.health_status = {
            "agent": HealthStatus("TechnicalAnalysisAgent"),
            "data_source": HealthStatus("DataSource"),
            "pattern_detection": HealthStatus("PatternDetection"),
            "indicator_engine": HealthStatus("IndicatorEngine"),
            "signal_generation": HealthStatus("SignalGeneration")
        }
        
        # Track active alerts
        self.active_alerts = []
        
        # Monitoring state
        self.running = False
        self.monitor_thread = None
        self.health_check_thread = None
        
        # Registered alert handlers
        self.alert_handlers = []
        
        # Create metrics directory if needed
        if self.config["export_metrics"] and not os.path.exists(self.config["metrics_dir"]):
            os.makedirs(self.config["metrics_dir"])
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Start health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        logger.info("Started Technical Analysis Agent monitoring")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.running:
            logger.warning("Monitoring not running")
            return
        
        self.running = False
        logger.info("Stopping Technical Analysis Agent monitoring")
    
    def register_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Register a handler for alerts.
        
        Args:
            handler: Callback function to handle alerts
        """
        self.alert_handlers.append(handler)
        logger.info("Registered alert handler")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                start_time = time.time()
                
                # Collect metrics
                self._collect_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Export metrics if enabled
                if self.config["export_metrics"]:
                    self._export_metrics()
                
                # Calculate sleep time to maintain consistent interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config["monitoring_interval"] - elapsed)
                
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.config["monitoring_interval"])
    
    def _health_check_loop(self):
        """Health check loop."""
        while self.running:
            try:
                # Perform health checks
                self._check_agent_health()
                self._check_data_source_health()
                self._check_pattern_detection_health()
                self._check_indicator_engine_health()
                self._check_signal_generation_health()
                
                # Sleep until next check
                time.sleep(self.config["health_check_interval"])
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                time.sleep(self.config["health_check_interval"])
    
    def _collect_metrics(self):
        """Collect performance metrics."""
        try:
            # Get agent statistics
            agent_stats = self.ta_agent.get_statistics()
            
            # Update metrics
            self.metrics["operation_count"].add_value(agent_stats.get("operation_count", 0))
            self.metrics["error_count"].add_value(agent_stats.get("error_count", 0))
            self.metrics["average_latency"].add_value(agent_stats.get("average_latency_ms", 0))
            self.metrics["signal_count"].add_value(agent_stats.get("signal_count", 0))
            self.metrics["pattern_count"].add_value(agent_stats.get("pattern_count", 0))
            
            # Get memory usage
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            self.metrics["memory_usage"].add_value(memory_info.rss)
            
            logger.debug("Collected metrics successfully")
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
    
    def _check_alerts(self):
        """Check for alert conditions."""
        alerts = []
        
        # Check error rate
        error_rate_threshold = self.config["alert_thresholds"]["error_rate"]
        if len(self.metrics["operation_count"].values) > 0:
            error_rate = self.metrics["error_count"].get_latest() / self.metrics["operation_count"].get_latest()
            if error_rate > error_rate_threshold:
                alerts.append({
                    "type": "error_rate",
                    "severity": "high",
                    "message": f"Error rate ({error_rate:.2%}) exceeds threshold ({error_rate_threshold:.2%})",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Check latency
        latency_threshold = self.config["alert_thresholds"]["latency"]
        latest_latency = self.metrics["average_latency"].get_latest()
        if latest_latency is not None and latest_latency > latency_threshold:
            alerts.append({
                "type": "latency",
                "severity": "medium",
                "message": f"Latency ({latest_latency:.2f} ms) exceeds threshold ({latency_threshold} ms)",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check memory usage
        memory_threshold = self.config["alert_thresholds"]["memory_usage"]
        latest_memory = self.metrics["memory_usage"].get_latest()
        if latest_memory is not None and latest_memory > memory_threshold:
            alerts.append({
                "type": "memory_usage",
                "severity": "high",
                "message": f"Memory usage ({latest_memory / (1024*1024):.2f} MB) exceeds threshold ({memory_threshold / (1024*1024):.2f} MB)",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check health status
        for component, status in self.health_status.items():
            if not status.is_healthy():
                alerts.append({
                    "type": "health",
                    "severity": "high" if status.status == HealthStatus.UNHEALTHY else "medium",
                    "component": component,
                    "message": f"{component} health check failed: {status.message}",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Process alerts
        for alert in alerts:
            self._handle_alert(alert)
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """
        Handle an alert.
        
        Args:
            alert: Alert information
        """
        # Add to active alerts if not already present
        alert_key = f"{alert['type']}:{alert.get('component', '')}"
        existing_alerts = [a for a in self.active_alerts if f"{a['type']}:{a.get('component', '')}" == alert_key]
        
        if not existing_alerts:
            self.active_alerts.append(alert)
            logger.warning(f"New alert: {alert['message']}")
            
            # Call registered handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {str(e)}")
    
    def _export_metrics(self):
        """Export metrics to files."""
        try:
            timestamp = datetime.now()
            metrics_file = os.path.join(self.config["metrics_dir"], f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}.json")
            
            # Prepare metrics data
            metrics_data = {
                "timestamp": timestamp.isoformat(),
                "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
                "health": {name: status.to_dict() for name, status in self.health_status.items()},
                "alerts": self.active_alerts
            }
            
            # Write to file
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.debug(f"Exported metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
    
    def _check_agent_health(self):
        """Check Technical Analysis Agent health."""
        try:
            # Get agent status
            agent_status = self.ta_agent.get_status()
            
            if agent_status.get("error_count", 0) > 10:
                self.health_status["agent"].update(
                    HealthStatus.DEGRADED if agent_status.get("error_count", 0) < 50 else HealthStatus.UNHEALTHY,
                    f"Agent has {agent_status.get('error_count', 0)} errors"
                )
            else:
                self.health_status["agent"].update(HealthStatus.HEALTHY, "Agent is functioning normally")
        except Exception as e:
            self.health_status["agent"].update(HealthStatus.UNHEALTHY, f"Failed to check agent health: {str(e)}")
    
    def _check_data_source_health(self):
        """Check data source health."""
        try:
            # Get data source and check if it's working
            data_source_factory = get_data_source_factory()
            data_provider = data_source_factory.get_data_provider()
            
            # Try to get some test data
            if get_data_source_config().use_mock_data:
                test_data = data_provider.generate_data(symbols=["BTC/USD"], timeframes=["1h"], periods=10)
            else:
                test_data = data_provider.get_historical_data(symbols=["BTC/USD"], timeframes=["1h"], periods=10)
            
            if test_data and any(test_data.values()):
                self.health_status["data_source"].update(HealthStatus.HEALTHY, "Data source is providing data")
            else:
                self.health_status["data_source"].update(HealthStatus.DEGRADED, "Data source returned empty data")
        except Exception as e:
            self.health_status["data_source"].update(HealthStatus.UNHEALTHY, f"Failed to check data source health: {str(e)}")
    
    def _check_pattern_detection_health(self):
        """Check pattern detection health."""
        try:
            # Create test data
            test_data = pd.DataFrame({
                'open': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
                'high': [105, 106, 107, 108, 109, 110, 109, 108, 107, 106],
                'low': [95, 96, 97, 98, 99, 100, 99, 98, 97, 96],
                'close': [101, 102, 103, 104, 105, 104, 103, 102, 101, 100]
            })
            
            # Test pattern detection
            patterns = self.ta_agent.detect_patterns(test_data)
            
            if patterns is not None:
                self.health_status["pattern_detection"].update(HealthStatus.HEALTHY, "Pattern detection is working")
            else:
                self.health_status["pattern_detection"].update(HealthStatus.DEGRADED, "Pattern detection returned no results")
        except Exception as e:
            self.health_status["pattern_detection"].update(HealthStatus.UNHEALTHY, f"Failed to check pattern detection health: {str(e)}")
    
    def _check_indicator_engine_health(self):
        """Check indicator engine health."""
        try:
            # Create test data
            test_data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101]
            })
            
            # Test indicator calculation
            indicators = self.ta_agent.calculate_indicators(test_data, ["sma", "rsi"])
            
            if indicators and all(i in indicators for i in ["sma", "rsi"]):
                self.health_status["indicator_engine"].update(HealthStatus.HEALTHY, "Indicator engine is working")
            else:
                self.health_status["indicator_engine"].update(HealthStatus.DEGRADED, "Indicator engine returned incomplete results")
        except Exception as e:
            self.health_status["indicator_engine"].update(HealthStatus.UNHEALTHY, f"Failed to check indicator engine health: {str(e)}")
    
    def _check_signal_generation_health(self):
        """Check signal generation health."""
        try:
            # Create test data
            test_data = {
                'BTC/USD': {
                    '1h': pd.DataFrame({
                        'open': [100, 101, 102, 103, 104, 105, 104, 103, 102, 101],
                        'high': [105, 106, 107, 108, 109, 110, 109, 108, 107, 106],
                        'low': [95, 96, 97, 98, 99, 100, 99, 98, 97, 96],
                        'close': [101, 102, 103, 104, 105, 104, 103, 102, 101, 100]
                    })
                }
            }
            
            # Test signal generation
            signals = self.ta_agent.generate_signals("BTC/USD", "1h", test_data)
            
            if signals is not None:
                self.health_status["signal_generation"].update(HealthStatus.HEALTHY, "Signal generation is working")
            else:
                self.health_status["signal_generation"].update(HealthStatus.DEGRADED, "Signal generation returned no results")
        except Exception as e:
            self.health_status["signal_generation"].update(HealthStatus.UNHEALTHY, f"Failed to check signal generation health: {str(e)}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status.
        
        Returns:
            Status dictionary with metrics and health information
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "health": {name: status.to_dict() for name, status in self.health_status.items()},
            "alerts": self.active_alerts,
            "config": self.config
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the system health.
        
        Returns:
            Health summary dictionary
        """
        # Count components by status
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0
        }
        
        for status in self.health_status.values():
            status_counts[status.status] += 1
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            overall_status = HealthStatus.DEGRADED
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "component_status": {name: status.status for name, status in self.health_status.items()},
            "alert_count": len(self.active_alerts),
            "status_counts": status_counts
        }

def setup_production_monitoring(ta_agent: AdvancedTechnicalAnalysisAgent) -> TAAgentMonitor:
    """
    Set up production monitoring for the Technical Analysis Agent.
    
    Args:
        ta_agent: Technical Analysis Agent instance
        
    Returns:
        Monitoring instance
    """
    # Create monitoring instance
    monitor = TAAgentMonitor(ta_agent)
    
    # Register alert handler
    def log_alert(alert):
        logger.critical(f"ALERT: {alert['severity'].upper()} - {alert['message']}")
    
    monitor.register_alert_handler(log_alert)
    
    # Start monitoring
    monitor.start_monitoring()
    
    logger.info("Production monitoring set up for Technical Analysis Agent")
    
    return monitor
