"""
Alert Manager for Health Monitoring System.

This module implements alert processing, routing, and management for
the health monitoring system, providing a centralized way to handle
health alerts from various components.
"""

import threading
import time
import logging
import json
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum
from pathlib import Path

# Import from core definitions to avoid circular dependencies
from .core_definitions import AlertSeverity
from .health_status import AlertData

# Set up logger
logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Available alert notification channels."""
    CONSOLE = "console"  # Log to console
    FILE = "file"        # Write to alert log file
    DASHBOARD = "dashboard"  # Send to dashboard
    EMAIL = "email"      # Send email notification
    SLACK = "slack"      # Send Slack notification


class AlertGroup:
    """
    Group of related alerts for aggregation and throttling.
    
    Used to prevent alert storms by grouping similar alerts together.
    """
    
    def __init__(
        self,
        group_id: str,
        description: str,
        throttling_period: float = 300.0  # 5 minutes
    ):
        """
        Initialize an alert group.
        
        Args:
            group_id: Unique identifier for the group
            description: Description of the alert group
            throttling_period: Time in seconds to throttle similar alerts
        """
        self.group_id = group_id
        self.description = description
        self.throttling_period = throttling_period
        self.alerts = []  # List[AlertData]
        self.last_notification_time = 0.0
        self.count_since_last_notification = 0
    
    def add_alert(self, alert: AlertData) -> bool:
        """
        Add an alert to the group.
        
        Args:
            alert: The alert to add
            
        Returns:
            True if notification should be sent, False if throttled
        """
        current_time = time.time()
        self.alerts.append(alert)
        self.count_since_last_notification += 1
        
        # Check if we should send notification
        elapsed = current_time - self.last_notification_time
        
        if elapsed >= self.throttling_period or self.last_notification_time == 0:
            self.last_notification_time = current_time
            notify_count = self.count_since_last_notification
            self.count_since_last_notification = 0
            return True, notify_count
            
        return False, self.count_since_last_notification
    
    def get_latest_alerts(self, limit: int = 10) -> List[AlertData]:
        """
        Get the most recent alerts in the group.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of most recent alerts
        """
        return sorted(
            self.alerts,
            key=lambda a: a.timestamp,
            reverse=True
        )[:limit]
    
    def clear_resolved(self) -> int:
        """
        Remove resolved alerts from the group.
        
        Returns:
            Number of alerts removed
        """
        before_count = len(self.alerts)
        self.alerts = [a for a in self.alerts if not a.resolved]
        return before_count - len(self.alerts)


class AlertManager:
    """
    Manager for processing and routing health alerts.
    
    Handles alert generation, routing to appropriate channels, and
    alert aggregation to prevent alert storms.
    """
    
    def __init__(
        self,
        channels: Optional[List[AlertChannel]] = None,
        throttling_period: float = 300.0,
        alert_log_path: Optional[str] = None,
        max_history: int = 1000,
        dashboard_callback: Optional[Callable[[AlertData], None]] = None
    ):
        """
        Initialize the alert manager.
        
        Args:
            channels: List of alert channels to enable
            throttling_period: Default throttling period for alert groups
            alert_log_path: Path to write alert log file
            max_history: Maximum number of alerts to keep in history
            dashboard_callback: Callback function for dashboard alerts
        """
        self.channels = channels or [AlertChannel.CONSOLE]
        self.throttling_period = throttling_period
        self.alert_log_path = alert_log_path
        self.max_history = max_history
        self.dashboard_callback = dashboard_callback
        
        self.alert_groups = {}  # Dict[str, AlertGroup]
        self.alert_history = []  # List[AlertData]
        self.active_alerts = {}  # Dict[str, AlertData]
        
        self._lock = threading.RLock()
        
        # Set up alert log file if needed
        if AlertChannel.FILE in self.channels and self.alert_log_path:
            log_dir = Path(self.alert_log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
    
    def add_alert(self, alert: AlertData) -> None:
        """
        Add an alert and route to appropriate channels.
        
        Args:
            alert: The alert to add
        """
        with self._lock:
            # Add to history
            self.alert_history.append(alert)
            
            # Trim history if needed
            if len(self.alert_history) > self.max_history:
                self.alert_history = self.alert_history[-self.max_history:]
            
            # Add to active alerts if not resolved
            if not alert.resolved:
                self.active_alerts[alert.alert_id] = alert
            
            # Determine alert group
            group_id = self._get_group_id(alert)
            
            # Create group if it doesn't exist
            if group_id not in self.alert_groups:
                description = f"Alerts for component {alert.component_id}"
                self.alert_groups[group_id] = AlertGroup(
                    group_id=group_id,
                    description=description,
                    throttling_period=self.throttling_period
                )
            
            # Add to group and check if notification should be sent
            should_notify, notify_count = self.alert_groups[group_id].add_alert(alert)
            
            if should_notify:
                if notify_count > 1:
                    # Modify message to include count of similar alerts
                    alert.message = f"{alert.message} ({notify_count} similar alerts)"
                
                # Route to enabled channels
                self._route_alert(alert)
    
    def _get_group_id(self, alert: AlertData) -> str:
        """
        Get the group ID for an alert.
        
        Determines how alerts are grouped for throttling.
        
        Args:
            alert: The alert to get group ID for
            
        Returns:
            Group ID string
        """
        # Group by component and alert type
        alert_type = alert.alert_id.split('_')[0] if '_' in alert.alert_id else "default"
        return f"{alert.component_id}_{alert_type}"
    
    def _route_alert(self, alert: AlertData) -> None:
        """
        Route an alert to all enabled channels.
        
        Args:
            alert: The alert to route
        """
        for channel in self.channels:
            try:
                if channel == AlertChannel.CONSOLE:
                    self._log_alert(alert)
                elif channel == AlertChannel.FILE:
                    self._write_alert_to_file(alert)
                elif channel == AlertChannel.DASHBOARD and self.dashboard_callback:
                    self.dashboard_callback(alert)
                elif channel == AlertChannel.EMAIL:
                    self._send_email_alert(alert)
                elif channel == AlertChannel.SLACK:
                    self._send_slack_alert(alert)
            except Exception as e:
                logger.error(f"Error routing alert to {channel.value}: {str(e)}")
    
    def _log_alert(self, alert: AlertData) -> None:
        """
        Log an alert to the console.
        
        Args:
            alert: The alert to log
        """
        # Determine log level based on severity
        if alert.severity == AlertSeverity.INFO:
            logger.info(f"ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(f"ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(f"ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.CRITICAL:
            logger.critical(f"ALERT: {alert.message}")
        else:
            logger.warning(f"ALERT ({alert.severity.value}): {alert.message}")
    
    def _write_alert_to_file(self, alert: AlertData) -> None:
        """
        Write an alert to the alert log file.
        
        Args:
            alert: The alert to write
        """
        if not self.alert_log_path:
            return
            
        try:
            log_file = Path(self.alert_log_path)
            
            # Convert alert to JSON
            alert_json = json.dumps(alert.to_dict(), indent=2)
            
            # Append to log file
            with open(log_file, 'a') as f:
                f.write(f"{alert_json}\n")
                
        except Exception as e:
            logger.error(f"Error writing alert to file: {str(e)}")
    
    def _send_email_alert(self, alert: AlertData) -> None:
        """
        Send an alert via email.
        
        Args:
            alert: The alert to send
        """
        # This is a placeholder for actual email implementation
        logger.info(f"Would send email alert: {alert.message}")
    
    def _send_slack_alert(self, alert: AlertData) -> None:
        """
        Send an alert via Slack.
        
        Args:
            alert: The alert to send
        """
        # This is a placeholder for actual Slack implementation
        logger.info(f"Would send Slack alert: {alert.message}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if alert was acknowledged, False if not found
        """
        with self._lock:
            if alert_id not in self.active_alerts:
                return False
                
            alert = self.active_alerts[alert_id]
            alert.acknowledge()
            
            return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if alert was resolved, False if not found
        """
        with self._lock:
            if alert_id not in self.active_alerts:
                return False
                
            alert = self.active_alerts[alert_id]
            alert.resolve()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            return True
    
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
        with self._lock:
            alerts = list(self.active_alerts.values())
            
            # Filter by component if specified
            if component_id is not None:
                alerts = [a for a in alerts if a.component_id == component_id]
                
            # Filter by severity if specified
            if min_severity is not None:
                severity_values = {s: i for i, s in enumerate(AlertSeverity)}
                min_value = severity_values[min_severity]
                alerts = [a for a in alerts if severity_values[a.severity] >= min_value]
                
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda a: a.timestamp, reverse=True)
            
            return alerts
    
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
        with self._lock:
            alerts = list(self.alert_history)
            
            # Filter by component if specified
            if component_id is not None:
                alerts = [a for a in alerts if a.component_id == component_id]
                
            # Filter by severity if specified
            if min_severity is not None:
                severity_values = {s: i for i, s in enumerate(AlertSeverity)}
                min_value = severity_values[min_severity]
                alerts = [a for a in alerts if severity_values[a.severity] >= min_value]
                
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda a: a.timestamp, reverse=True)
            
            return alerts[:limit]
    
    def clear_resolved_alerts(self) -> int:
        """
        Clear resolved alerts from groups to free memory.
        
        Returns:
            Number of alerts cleared
        """
        with self._lock:
            total_cleared = 0
            
            for group in self.alert_groups.values():
                total_cleared += group.clear_resolved()
                
            return total_cleared
