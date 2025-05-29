"""
Health status definitions for the Health Monitoring System.

This module defines the data structures used to represent health status and alerts
for system components. Core enums are imported from core_definitions.
"""

import time
from typing import Any, Dict, List, Optional, Union

# Import enums from core_definitions to avoid circular dependencies
from .core_definitions import HealthStatus, AlertSeverity


class AlertData:
    """
    Data structure for health alerts.
    
    Represents an alert generated due to a health issue in a component 
    or the system as a whole.
    """
    
    def __init__(
        self,
        alert_id: str,
        component_id: str,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an alert.
        
        Args:
            alert_id: Unique identifier for the alert
            component_id: ID of the component that generated the alert
            severity: Severity level of the alert
            message: Alert message
            details: Additional details about the alert
        """
        self.alert_id = alert_id
        self.component_id = component_id
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()
        self.acknowledged = False
        self.resolved = False
        self.resolution_time = None
    
    def acknowledge(self) -> None:
        """Mark the alert as acknowledged."""
        self.acknowledged = True
    
    def resolve(self) -> None:
        """Mark the alert as resolved."""
        self.resolved = True
        self.resolution_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert to a dictionary representation."""
        return {
            "alert_id": self.alert_id,
            "component_id": self.component_id,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time
        }


class ComponentHealthData:
    """
    Data structure for component health status.
    
    Represents the health status and metrics of an individual component.
    """
    
    def __init__(
        self,
        component_id: str,
        component_type: str,
        status: HealthStatus = HealthStatus.UNKNOWN
    ):
        """
        Initialize component health data.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of the component
            status: Initial health status
        """
        self.component_id = component_id
        self.component_type = component_type
        self.status = status
        self.last_heartbeat = None
        self.first_heartbeat = None
        self.heartbeat_count = 0
        self.metrics = {}
        self.alerts = []
        self.recovery_attempts = 0
        self.last_recovery_time = None
        self.diagnostics = {}
        self.last_updated = time.time()
    
    @property
    def uptime(self) -> Optional[float]:
        """Calculate component uptime in seconds."""
        if self.first_heartbeat is None:
            return None
        return time.time() - self.first_heartbeat
    
    def update_status(self, status: HealthStatus) -> None:
        """
        Update the component status.
        
        Args:
            status: New health status
        """
        self.status = status
        self.last_updated = time.time()
    
    def record_heartbeat(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a heartbeat from the component.
        
        Args:
            data: Optional data provided with the heartbeat
        """
        current_time = time.time()
        self.last_heartbeat = current_time
        
        if self.first_heartbeat is None:
            self.first_heartbeat = current_time
            
        self.heartbeat_count += 1
        
        if data:
            # Update diagnostics with heartbeat data
            self.diagnostics.update(data)
            
        self.last_updated = current_time
    
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record metrics from the component.
        
        Args:
            metrics: Dictionary of metrics from the component
        """
        self.metrics.update(metrics)
        self.last_updated = time.time()
    
    def add_alert(self, alert: AlertData) -> None:
        """
        Add an alert to the component.
        
        Args:
            alert: The alert to add
        """
        self.alerts.append(alert)
        self.last_updated = time.time()
    
    def record_recovery_attempt(self) -> None:
        """Record a recovery attempt for this component."""
        self.recovery_attempts += 1
        self.last_recovery_time = time.time()
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the component health data to a dictionary representation."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat,
            "first_heartbeat": self.first_heartbeat,
            "heartbeat_count": self.heartbeat_count,
            "uptime": self.uptime,
            "metrics": self.metrics,
            "alerts": [alert.to_dict() for alert in self.alerts],
            "recovery_attempts": self.recovery_attempts,
            "last_recovery_time": self.last_recovery_time,
            "diagnostics": self.diagnostics,
            "last_updated": self.last_updated
        }


class SystemHealthData:
    """
    Data structure for system-wide health status.
    
    Represents the aggregated health status of the entire system.
    """
    
    def __init__(self):
        """Initialize system health data."""
        self.overall_status = HealthStatus.UNKNOWN
        self.component_statuses = {}  # Dict[str, ComponentHealthData]
        self.system_metrics = {}
        self.active_alerts = []  # List[AlertData]
        self.recent_recoveries = []  # List of recent recovery events
        self.started_at = time.time()
        self.last_updated = time.time()
    
    def update_overall_status(self) -> None:
        """Update the overall system status based on component statuses."""
        if not self.component_statuses:
            self.overall_status = HealthStatus.UNKNOWN
            return
            
        # Determine worst status among components
        status_priority = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.UNKNOWN: 1,
            HealthStatus.RECOVERING: 2,
            HealthStatus.DEGRADED: 3,
            HealthStatus.UNHEALTHY: 4,
            HealthStatus.CRITICAL: 5
        }
        
        worst_status = HealthStatus.HEALTHY
        
        for component in self.component_statuses.values():
            if status_priority[component.status] > status_priority[worst_status]:
                worst_status = component.status
                
        self.overall_status = worst_status
        self.last_updated = time.time()
    
    def add_component_status(self, component_data: ComponentHealthData) -> None:
        """
        Add or update a component status.
        
        Args:
            component_data: Component health data to add/update
        """
        self.component_statuses[component_data.component_id] = component_data
        self.update_overall_status()
    
    def add_alert(self, alert: AlertData) -> None:
        """
        Add an active alert.
        
        Args:
            alert: The alert to add
        """
        self.active_alerts.append(alert)
        self.last_updated = time.time()
    
    def update_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update system-wide metrics.
        
        Args:
            metrics: Dictionary of system metrics
        """
        self.system_metrics.update(metrics)
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the system health data to a dictionary representation."""
        return {
            "overall_status": self.overall_status.value,
            "component_statuses": {
                component_id: component.to_dict() 
                for component_id, component in self.component_statuses.items()
            },
            "system_metrics": self.system_metrics,
            "active_alerts": [alert.to_dict() for alert in self.active_alerts],
            "recent_recoveries": self.recent_recoveries,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "uptime": time.time() - self.started_at
        }
