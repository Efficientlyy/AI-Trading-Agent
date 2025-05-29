"""
Health Monitor for AI Trading Agent.

This module implements the main health monitoring system that integrates
component health tracking, heartbeat monitoring, performance metrics,
and alert management for comprehensive system health monitoring.
"""

import threading
import time
import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, Set, Union

# Import core definitions to avoid circular dependencies
from .core_definitions import HealthStatus, AlertSeverity, ThresholdType

# Import components with proper dependency order
from .health_status import AlertData
from .component_health import ComponentHealth
from .heartbeat_manager import HeartbeatManager, HeartbeatConfig
from .alert_manager import AlertManager, AlertChannel
from .health_metrics import HealthMetrics, MetricThreshold

# Set up logger
logger = logging.getLogger(__name__)


class RecoveryAction:
    """
    Represents a recovery action that can be taken in response to health issues.
    """
    
    def __init__(
        self,
        action_id: str,
        description: str,
        action_func: Callable[..., bool],
        component_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        severity_threshold: AlertSeverity = AlertSeverity.ERROR
    ):
        """
        Initialize a recovery action.
        
        Args:
            action_id: Unique identifier for the action
            description: Human-readable description of the action
            action_func: Function to call when action is executed
            component_id: Optional component ID this action applies to
            parameters: Optional parameters to pass to action function
            severity_threshold: Only execute for alerts of this severity or higher
        """
        self.action_id = action_id
        self.description = description
        self.action_func = action_func
        self.component_id = component_id
        self.parameters = parameters or {}
        self.severity_threshold = severity_threshold
        self.last_executed = None
        self.success_count = 0
        self.failure_count = 0
    
    def execute(self) -> bool:
        """
        Execute the recovery action.
        
        Returns:
            True if action was successful, False otherwise
        """
        self.last_executed = time.time()
        
        try:
            result = self.action_func(**self.parameters)
            
            if result:
                self.success_count += 1
            else:
                self.failure_count += 1
                
            return result
        except Exception as e:
            logger.error(f"Error executing recovery action {self.action_id}: {str(e)}")
            self.failure_count += 1
            return False


class RecoveryCoordinator:
    """
    Coordinates recovery actions for system components.
    
    Manages the selection and execution of recovery actions in response to health issues,
    provides logging and tracking of recovery attempts, and prevents action conflicts.
    """
    
    def __init__(self):
        """Initialize the recovery coordinator."""
        self.actions = {}  # Dict[str, RecoveryAction]
        self.component_actions = {}  # Dict[str, List[str]]
        self.recovery_history = []  # List of recovery attempt records
        self.executing_actions = set()  # Set of action IDs currently executing
        self._lock = threading.RLock()
    
    def register_action(self, action: RecoveryAction) -> None:
        """
        Register a recovery action.
        
        Args:
            action: The recovery action to register
        """
        with self._lock:
            if action.action_id in self.actions:
                logger.warning(f"Recovery action {action.action_id} already registered")
                return
                
            self.actions[action.action_id] = action
            
            # Index by component if specified
            if action.component_id:
                if action.component_id not in self.component_actions:
                    self.component_actions[action.component_id] = []
                    
                self.component_actions[action.component_id].append(action.action_id)
                
            logger.info(f"Registered recovery action {action.action_id}: {action.description}")
    
    def unregister_action(self, action_id: str) -> bool:
        """
        Unregister a recovery action.
        
        Args:
            action_id: ID of the action to unregister
            
        Returns:
            True if action was unregistered, False if not found
        """
        with self._lock:
            if action_id not in self.actions:
                logger.warning(f"Recovery action {action_id} not found for unregistration")
                return False
                
            action = self.actions[action_id]
            del self.actions[action_id]
            
            # Remove from component index if applicable
            if action.component_id and action.component_id in self.component_actions:
                if action_id in self.component_actions[action.component_id]:
                    self.component_actions[action.component_id].remove(action_id)
                    
            logger.info(f"Unregistered recovery action {action_id}")
            return True
    
    def handle_alert(self, alert: AlertData) -> bool:
        """
        Handle an alert by finding and executing appropriate recovery actions.
        
        Args:
            alert: The alert to handle
            
        Returns:
            True if recovery action was taken, False otherwise
        """
        with self._lock:
            component_id = alert.component_id
            actions_to_execute = []
            
            # Find applicable actions
            if component_id in self.component_actions:
                for action_id in self.component_actions[component_id]:
                    action = self.actions[action_id]
                    
                    # Check if severity threshold is met
                    severity_values = {s: i for i, s in enumerate(AlertSeverity)}
                    if severity_values[alert.severity] >= severity_values[action.severity_threshold]:
                        actions_to_execute.append(action)
            
            # Execute actions
            if not actions_to_execute:
                logger.info(f"No recovery actions available for alert {alert.alert_id}")
                return False
                
            # Execute first applicable action
            action = actions_to_execute[0]
            
            # Check if already executing
            if action.action_id in self.executing_actions:
                logger.info(f"Recovery action {action.action_id} already executing")
                return False
                
            # Mark as executing
            self.executing_actions.add(action.action_id)
            
            try:
                logger.info(f"Executing recovery action {action.action_id} for alert {alert.alert_id}")
                success = action.execute()
                
                # Record recovery attempt
                recovery_record = {
                    "timestamp": time.time(),
                    "alert_id": alert.alert_id,
                    "action_id": action.action_id,
                    "component_id": component_id,
                    "success": success
                }
                
                self.recovery_history.append(recovery_record)
                
                # Keep recovery history size reasonable
                if len(self.recovery_history) > 1000:
                    self.recovery_history = self.recovery_history[-1000:]
                    
                return success
            finally:
                # Remove from executing set
                self.executing_actions.remove(action.action_id)
    
    def get_actions_for_component(self, component_id: str) -> List[RecoveryAction]:
        """
        Get all recovery actions for a component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            List of recovery actions
        """
        with self._lock:
            if component_id not in self.component_actions:
                return []
                
            return [self.actions[action_id] for action_id in self.component_actions[component_id]]
    
    def get_recovery_history(
        self,
        component_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recovery history with optional filtering.
        
        Args:
            component_id: Optional component ID to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of recovery attempt records
        """
        with self._lock:
            if component_id is None:
                filtered_history = self.recovery_history
            else:
                filtered_history = [
                    record for record in self.recovery_history
                    if record["component_id"] == component_id
                ]
                
            # Sort by timestamp (newest first)
            sorted_history = sorted(
                filtered_history,
                key=lambda r: r["timestamp"],
                reverse=True
            )
            
            return sorted_history[:limit]


class HealthMonitor:
    """
    Central health monitoring system for AI Trading Agent.
    
    Integrates component health tracking, heartbeat monitoring, performance metrics,
    and alert management for comprehensive system health monitoring.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        dashboard_callback: Optional[Callable[[AlertData], None]] = None
    ):
        """
        Initialize the health monitoring system.
        
        Args:
            config_path: Optional path to configuration file
            log_dir: Optional directory for alert logs
            dashboard_callback: Optional callback for dashboard integration
        """
        # Set up log directory
        self.log_dir = log_dir
        if self.log_dir:
            self.log_dir = Path(self.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.alert_log_path = str(self.log_dir / "health_alerts.log")
        else:
            self.alert_log_path = None
            
        # Load configuration if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Set up alert channels
        alert_channels = self._get_alert_channels_from_config()
        
        # Initialize component health tracking
        self.component_health = {}  # Dict[str, ComponentHealth]
        
        # Initialize subsystems
        self.alert_manager = AlertManager(
            channels=alert_channels,
            alert_log_path=self.alert_log_path,
            dashboard_callback=dashboard_callback
        )
        
        self.heartbeat_manager = HeartbeatManager(
            alert_callback=self._handle_heartbeat_alert,
            status_callback=self._update_component_status
        )
        
        self.health_metrics = HealthMetrics(
            alert_callback=self._handle_metric_alert
        )
        
        self.recovery_coordinator = RecoveryCoordinator()
        
        # Set up locks for thread safety
        self._lock = threading.RLock()
        
        # Set up background task management
        self._running = False
        self._thread = None
        
        logger.info("Health monitoring system initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                else:
                    # Assume YAML
                    import yaml
                    config = yaml.safe_load(f)
                    
            logger.info(f"Loaded health monitoring configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _get_alert_channels_from_config(self) -> List[AlertChannel]:
        """
        Get alert channels from configuration.
        
        Returns:
            List of enabled alert channels
        """
        channels = [AlertChannel.CONSOLE]  # Console always enabled
        
        if not self.config.get('alerts'):
            return channels
            
        alert_config = self.config['alerts']
        
        # Add file channel if log dir provided
        if self.alert_log_path:
            channels.append(AlertChannel.FILE)
            
        # Add other channels from config
        if alert_config.get('dashboard', False):
            channels.append(AlertChannel.DASHBOARD)
            
        if alert_config.get('email', False):
            channels.append(AlertChannel.EMAIL)
            
        if alert_config.get('slack', False):
            channels.append(AlertChannel.SLACK)
            
        return channels
    
    def _handle_heartbeat_alert(self, alert: AlertData) -> None:
        """
        Handle an alert from the heartbeat manager.
        
        Args:
            alert: The alert to handle
        """
        # Forward to alert manager
        self.alert_manager.add_alert(alert)
        
        # Try recovery
        self.recovery_coordinator.handle_alert(alert)
    
    def _handle_metric_alert(self, alert: AlertData) -> None:
        """
        Handle an alert from the metrics manager.
        
        Args:
            alert: The alert to handle
        """
        # Forward to alert manager
        self.alert_manager.add_alert(alert)
        
        # Try recovery
        self.recovery_coordinator.handle_alert(alert)
    
    def _update_component_status(
        self,
        component_id: str,
        status: HealthStatus
    ) -> None:
        """
        Update a component's health status.
        
        Args:
            component_id: ID of the component
            status: New health status
        """
        with self._lock:
            if component_id not in self.component_health:
                self.register_component(component_id)
                
            component = self.component_health[component_id]
            old_status = component.status
            
            # Update status if changed
            if old_status != status:
                component.status = status
                component.last_status_change = time.time()
                
                logger.info(f"Component {component_id} status changed from "
                          f"{old_status.value} to {status.value}")
                
                # Generate alert for status change
                if status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    self._generate_status_change_alert(component_id, old_status, status)
    
    def _generate_status_change_alert(
        self,
        component_id: str,
        old_status: HealthStatus,
        new_status: HealthStatus
    ) -> None:
        """
        Generate an alert for a component status change.
        
        Args:
            component_id: ID of the component
            old_status: Previous health status
            new_status: New health status
        """
        # Determine severity based on new status
        severity = AlertSeverity.INFO
        if new_status == HealthStatus.DEGRADED:
            severity = AlertSeverity.WARNING
        elif new_status == HealthStatus.UNHEALTHY:
            severity = AlertSeverity.ERROR
        elif new_status == HealthStatus.CRITICAL:
            severity = AlertSeverity.CRITICAL
            
        alert_id = f"status_{component_id}_{int(time.time())}"
        
        message = (f"Component {component_id} status changed from "
                 f"{old_status.value} to {new_status.value}")
                 
        details = {
            "component_id": component_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "component": self.component_health[component_id].to_dict()
        }
        
        alert = AlertData(
            alert_id=alert_id,
            component_id=component_id,
            severity=severity,
            message=message,
            details=details
        )
        
        self.alert_manager.add_alert(alert)
    
    def register_component(
        self,
        component_id: str,
        description: Optional[str] = None,
        heartbeat_config: Optional[HeartbeatConfig] = None,
        monitors: Optional[List[str]] = None
    ) -> None:
        """
        Register a component for health monitoring.
        
        Args:
            component_id: ID of the component to register
            description: Optional description of the component
            heartbeat_config: Optional custom heartbeat configuration
            monitors: Optional list of monitors to enable
        """
        with self._lock:
            # Create component health record if not exists
            if component_id not in self.component_health:
                self.component_health[component_id] = ComponentHealth(
                    component_id=component_id,
                    description=description or f"Component {component_id}"
                )
                
            # Register for heartbeat monitoring
            if monitors is None or "heartbeat" in monitors:
                self.heartbeat_manager.register_component(
                    component_id=component_id,
                    config=heartbeat_config
                )
                
            logger.info(f"Registered component {component_id} for health monitoring")
    
    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a component from health monitoring.
        
        Args:
            component_id: ID of the component to unregister
            
        Returns:
            True if component was unregistered, False if not found
        """
        with self._lock:
            if component_id not in self.component_health:
                logger.warning(f"Component {component_id} not found for unregistration")
                return False
            
            # Unregister from heartbeat monitoring
            self.heartbeat_manager.unregister_component(component_id)
            
            # Remove component health record
            del self.component_health[component_id]
            
            logger.info(f"Unregistered component {component_id} from health monitoring")
            return True
    
    def record_heartbeat(
        self,
        component_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a heartbeat from a component.
        
        Args:
            component_id: ID of the component sending heartbeat
            data: Optional data provided with the heartbeat
            
        Returns:
            True if heartbeat was recorded, False if component not registered
        """
        return self.heartbeat_manager.record_heartbeat(
            component_id=component_id,
            data=data
        )
    
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
        # Register component if not already registered
        if component_id not in self.component_health:
            self.register_component(component_id)
            
        self.health_metrics.add_metric(
            component_id=component_id,
            metric_name=metric_name,
            value=value,
            timestamp=timestamp
        )
    
    def add_metric_threshold(
        self,
        metric_name: str,
        threshold_type: ThresholdType,
        warning_threshold: float,
        critical_threshold: float,
        component_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Add a threshold for metric monitoring.
        
        Args:
            metric_name: Name of the metric to monitor
            threshold_type: Type of threshold comparison
            warning_threshold: Threshold for warning alerts
            critical_threshold: Threshold for critical alerts
            component_id: Optional component ID if specific to a component
            description: Optional description of the threshold
        """
        threshold = MetricThreshold(
            metric_name=metric_name,
            threshold_type=threshold_type,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            component_id=component_id,
            description=description
        )
        
        self.health_metrics.add_threshold(threshold)
    
    def register_recovery_action(
        self,
        action_id: str,
        description: str,
        action_func: Callable[..., bool],
        component_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        severity_threshold: AlertSeverity = AlertSeverity.ERROR
    ) -> None:
        """
        Register a recovery action.
        
        Args:
            action_id: Unique identifier for the action
            description: Human-readable description of the action
            action_func: Function to call when action is executed
            component_id: Optional component ID this action applies to
            parameters: Optional parameters to pass to action function
            severity_threshold: Only execute for alerts of this severity or higher
        """
        action = RecoveryAction(
            action_id=action_id,
            description=description,
            action_func=action_func,
            component_id=component_id,
            parameters=parameters,
            severity_threshold=severity_threshold
        )
        
        self.recovery_coordinator.register_action(action)
    
    def start(self) -> None:
        """Start all health monitoring subsystems."""
        with self._lock:
            if self._running:
                logger.warning("Health monitoring already running")
                return
                
            # Start subsystems
            self.heartbeat_manager.start()
            self.health_metrics.start()
            
            self._running = True
            
            logger.info("Health monitoring started")
    
    def stop(self) -> None:
        """Stop all health monitoring subsystems."""
        with self._lock:
            if not self._running:
                logger.warning("Health monitoring already stopped")
                return
                
            # Stop subsystems
            self.heartbeat_manager.stop()
            self.health_metrics.stop()
            
            self._running = False
            
            logger.info("Health monitoring stopped")
    
    def get_component_health(
        self,
        component_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get health status of components.
        
        Args:
            component_id: Optional component ID to filter by
            
        Returns:
            Dictionary of component health status
        """
        with self._lock:
            result = {}
            
            if component_id is not None:
                # Return specific component
                if component_id not in self.component_health:
                    return {}
                    
                component = self.component_health[component_id]
                result[component_id] = component.to_dict()
            else:
                # Return all components
                for comp_id, component in self.component_health.items():
                    result[comp_id] = component.to_dict()
                    
            return result
    
    def get_active_alerts(
        self,
        component_id: Optional[str] = None,
        min_severity: Optional[AlertSeverity] = None
    ) -> List[AlertData]:
        """
        Get active alerts with optional filtering.
        
        Args:
            component_id: Optional component ID to filter by
            min_severity: Optional minimum severity to filter by
            
        Returns:
            List of active alerts matching filters
        """
        return self.alert_manager.get_active_alerts(
            component_id=component_id,
            min_severity=min_severity
        )
    
    def get_alert_history(
        self,
        limit: int = 100,
        component_id: Optional[str] = None,
        min_severity: Optional[AlertSeverity] = None
    ) -> List[AlertData]:
        """
        Get alert history with optional filtering.
        
        Args:
            limit: Maximum number of alerts to return
            component_id: Optional component ID to filter by
            min_severity: Optional minimum severity to filter by
            
        Returns:
            List of historical alerts matching filters
        """
        return self.alert_manager.get_alert_history(
            limit=limit,
            component_id=component_id,
            min_severity=min_severity
        )
    
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
        return self.health_metrics.get_metrics(
            component_id=component_id,
            metric_name=metric_name
        )
    
    def get_recovery_history(
        self,
        component_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recovery history with optional filtering.
        
        Args:
            component_id: Optional component ID to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of recovery attempt records
        """
        return self.recovery_coordinator.get_recovery_history(
            component_id=component_id,
            limit=limit
        )
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with system health information
        """
        with self._lock:
            # Count components by status
            status_counts = {status.value: 0 for status in HealthStatus}
            for component in self.component_health.values():
                status_counts[component.status.value] += 1
                
            # Count active alerts by severity
            alerts = self.get_active_alerts()
            alert_counts = {severity.value: 0 for severity in AlertSeverity}
            for alert in alerts:
                alert_counts[alert.severity.value] += 1
                
            # Determine overall status
            if status_counts[HealthStatus.CRITICAL.value] > 0:
                overall_status = HealthStatus.CRITICAL
            elif status_counts[HealthStatus.UNHEALTHY.value] > 0:
                overall_status = HealthStatus.UNHEALTHY
            elif status_counts[HealthStatus.DEGRADED.value] > 0:
                overall_status = HealthStatus.DEGRADED
            elif status_counts[HealthStatus.RECOVERING.value] > 0:
                overall_status = HealthStatus.RECOVERING
            else:
                overall_status = HealthStatus.HEALTHY
                
            return {
                "overall_status": overall_status.value,
                "component_count": len(self.component_health),
                "status_counts": status_counts,
                "active_alerts": len(alerts),
                "alert_counts": alert_counts,
                "timestamp": time.time()
            }
