"""
Alert management system for the trading platform.

This module provides functionality for generating, tracking, and distributing
alerts about system conditions and trading activities.
"""

import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import uuid
import json
from pathlib import Path

# Configure logger
logger = logging.getLogger("monitoring.alerts")


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"  # Informational events
    WARNING = "warning"  # Concerning events that don't impact operations
    ERROR = "error"  # Issues affecting some operations 
    CRITICAL = "critical"  # Severe issues requiring immediate attention


class AlertCategory(str, Enum):
    """Categories for different types of alerts."""
    SYSTEM = "system"  # System health and component status
    EXCHANGE = "exchange"  # Exchange connectivity and API issues
    ORDER = "order"  # Order execution issues
    POSITION = "position"  # Position management alerts
    STRATEGY = "strategy"  # Strategy performance and behavior
    RISK = "risk"  # Risk limit violations
    SECURITY = "security"  # Security-related alerts


class Alert:
    """
    Class representing a single alert in the system.
    """
    def __init__(
        self,
        message: str,
        level: AlertLevel,
        category: AlertCategory,
        source: str,
        details: Optional[Dict[str, Any]] = None,
        expiry: Optional[datetime] = None,
        is_active: bool = True,
        id: Optional[str] = None
    ):
        """
        Initialize a new alert.
        
        Args:
            message: Human-readable description of the alert
            level: Severity level of the alert
            category: Category the alert belongs to
            source: Component or process that generated the alert
            details: Additional structured data relevant to the alert
            expiry: When the alert should automatically resolve (None for no auto-expiry)
            is_active: Whether the alert is currently active
            id: Unique identifier for the alert (generated if not provided)
        """
        self.id = id or str(uuid.uuid4())
        self.message = message
        self.level = level
        self.category = category
        self.source = source
        self.details = details or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.expiry = expiry
        self.is_active = is_active
        self.resolved_at = None
        self.resolution_message = None
    
    def resolve(self, resolution_message: Optional[str] = None) -> None:
        """
        Mark an alert as resolved.
        
        Args:
            resolution_message: Optional message describing the resolution
        """
        if self.is_active:
            self.is_active = False
            self.resolved_at = datetime.now()
            self.updated_at = self.resolved_at
            self.resolution_message = resolution_message
            logger.info(f"Alert {self.id} resolved: {self.message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert alert to a dictionary for serialization.
        
        Returns:
            Dict representation of the alert
        """
        return {
            "id": self.id,
            "message": self.message,
            "level": self.level,
            "category": self.category,
            "source": self.source,
            "details": self.details,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "is_active": self.is_active,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_message": self.resolution_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """
        Create an alert instance from a dictionary.
        
        Args:
            data: Dictionary containing alert data
            
        Returns:
            Alert instance
        """
        # Convert ISO format strings back to datetime objects
        created_at = datetime.fromisoformat(data.pop("created_at"))
        updated_at = datetime.fromisoformat(data.pop("updated_at"))
        
        expiry = data.pop("expiry")
        if expiry:
            expiry = datetime.fromisoformat(expiry)
            
        resolved_at = data.pop("resolved_at")
        if resolved_at:
            resolved_at = datetime.fromisoformat(resolved_at)
        
        # Create the alert instance
        alert = cls(**data)
        
        # Set the timestamp fields
        alert.created_at = created_at
        alert.updated_at = updated_at
        alert.expiry = expiry
        alert.resolved_at = resolved_at
        
        return alert


class AlertHandler:
    """
    Base class for alert handlers.
    
    Alert handlers are responsible for processing alerts when they are
    triggered or updated, such as sending notifications or logging.
    """
    def __init__(self, name: str):
        """
        Initialize the alert handler.
        
        Args:
            name: Unique name for this handler
        """
        self.name = name
    
    def handle_alert(self, alert: Alert) -> None:
        """
        Process an alert. Must be implemented by subclasses.
        
        Args:
            alert: The alert to process
        """
        raise NotImplementedError("Subclasses must implement handle_alert()")


class LoggingAlertHandler(AlertHandler):
    """
    Alert handler that logs alerts to the standard logging system.
    """
    def __init__(self, name: str = "logging_handler"):
        """
        Initialize the logging alert handler.
        
        Args:
            name: Unique name for this handler
        """
        super().__init__(name)
        self.logger = logging.getLogger("monitoring.alerts")
    
    def handle_alert(self, alert: Alert) -> None:
        """
        Log an alert with the appropriate severity level.
        
        Args:
            alert: The alert to log
        """
        message = f"ALERT [{alert.category}] {alert.message}"
        if alert.details:
            message += f" - Details: {json.dumps(alert.details)}"
        
        if alert.level == AlertLevel.INFO:
            self.logger.info(message)
        elif alert.level == AlertLevel.WARNING:
            self.logger.warning(message)
        elif alert.level == AlertLevel.ERROR:
            self.logger.error(message)
        elif alert.level == AlertLevel.CRITICAL:
            self.logger.critical(message)


class FileAlertHandler(AlertHandler):
    """
    Alert handler that writes alerts to a JSON file.
    """
    def __init__(
        self, 
        alerts_file_path: Path,
        name: str = "file_handler"
    ):
        """
        Initialize the file alert handler.
        
        Args:
            alerts_file_path: Path to the alerts file
            name: Unique name for this handler
        """
        super().__init__(name)
        self.file_path = alerts_file_path
    
    def handle_alert(self, alert: Alert) -> None:
        """
        Write an alert to the alerts file.
        
        Args:
            alert: The alert to write
        """
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing alerts if the file exists
        alerts = []
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    alerts = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse alerts file: {self.file_path}")
        
        # Add or update this alert
        alert_dict = alert.to_dict()
        for i, existing in enumerate(alerts):
            if existing.get('id') == alert.id:
                alerts[i] = alert_dict
                break
        else:
            alerts.append(alert_dict)
        
        # Write back to file
        with open(self.file_path, 'w') as f:
            json.dump(alerts, f, indent=2)


class AlertManager:
    """
    Central manager for handling system alerts.
    
    This class provides methods for creating, tracking, and managing alerts.
    It applies alert handling logic and notifies registered handlers.
    """
    def __init__(self, max_alerts: int = 1000):
        """
        Initialize the alert manager.
        
        Args:
            max_alerts: Maximum number of alerts to keep in memory
        """
        self.alerts: Dict[str, Alert] = {}  # All alerts by ID
        self.active_alerts: Dict[str, Alert] = {}  # Only active alerts
        self.handlers: List[AlertHandler] = []
        self.max_alerts = max_alerts
        
        # Add default logging handler
        self.add_handler(LoggingAlertHandler())
    
    def add_handler(self, handler: AlertHandler) -> None:
        """
        Register a new alert handler.
        
        Args:
            handler: The handler to register
        """
        self.handlers.append(handler)
        logger.info(f"Added alert handler: {handler.name}")
    
    def create_alert(
        self,
        message: str,
        level: AlertLevel,
        category: AlertCategory,
        source: str,
        details: Optional[Dict[str, Any]] = None,
        expiry: Optional[datetime] = None
    ) -> Alert:
        """
        Create and process a new alert.
        
        Args:
            message: Human-readable description of the alert
            level: Severity level of the alert
            category: Category the alert belongs to
            source: Component or process that generated the alert
            details: Additional structured data relevant to the alert
            expiry: When the alert should automatically resolve (None for no auto-expiry)
            
        Returns:
            The created Alert object
        """
        alert = Alert(
            message=message,
            level=level,
            category=category,
            source=source,
            details=details,
            expiry=expiry
        )
        
        # Store the alert
        self.alerts[alert.id] = alert
        if alert.is_active:
            self.active_alerts[alert.id] = alert
        
        # Process the alert through all handlers
        for handler in self.handlers:
            try:
                handler.handle_alert(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.name}: {str(e)}")
        
        # Enforce the maximum alert limit
        self._enforce_max_alerts()
        
        return alert
    
    def update_alert(
        self,
        alert_id: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        level: Optional[AlertLevel] = None
    ) -> Optional[Alert]:
        """
        Update an existing alert.
        
        Args:
            alert_id: ID of the alert to update
            message: New message (if changed)
            details: New details to merge with existing ones
            level: New alert level (if changed)
            
        Returns:
            The updated Alert object, or None if not found
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            logger.warning(f"Attempted to update nonexistent alert: {alert_id}")
            return None
        
        # Update the alert
        if message is not None:
            alert.message = message
            
        if details is not None:
            if isinstance(details, dict):
                alert.details.update(details)
            else:
                logger.warning(f"Invalid details format for alert {alert_id}: {details}")
                
        if level is not None:
            alert.level = level
            
        alert.updated_at = datetime.now()
        
        # Notify handlers
        for handler in self.handlers:
            try:
                handler.handle_alert(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.name}: {str(e)}")
        
        return alert
    
    def resolve_alert(
        self,
        alert_id: str,
        resolution_message: Optional[str] = None
    ) -> Optional[Alert]:
        """
        Mark an alert as resolved.
        
        Args:
            alert_id: ID of the alert to resolve
            resolution_message: Message describing how the alert was resolved
            
        Returns:
            The resolved Alert object, or None if not found
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            logger.warning(f"Attempted to resolve nonexistent alert: {alert_id}")
            return None
        
        if not alert.is_active:
            logger.debug(f"Alert already resolved: {alert_id}")
            return alert
        
        # Resolve the alert
        alert.resolve(resolution_message)
        
        # Remove from active alerts
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
        
        # Notify handlers
        for handler in self.handlers:
            try:
                handler.handle_alert(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.name}: {str(e)}")
        
        return alert
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """
        Get an alert by ID.
        
        Args:
            alert_id: ID of the alert to retrieve
            
        Returns:
            The Alert object, or None if not found
        """
        return self.alerts.get(alert_id)
    
    def get_alerts(
        self,
        active_only: bool = False,
        level: Optional[AlertLevel] = None,
        category: Optional[AlertCategory] = None,
        source: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Alert]:
        """
        Get alerts matching the specified criteria.
        
        Args:
            active_only: Whether to include only active alerts
            level: Filter by alert level
            category: Filter by alert category
            source: Filter by alert source
            since: Only include alerts created after this time
            limit: Maximum number of alerts to return
            
        Returns:
            List of Alert objects matching the criteria
        """
        # Start with all alerts or active alerts
        if active_only:
            alerts = list(self.active_alerts.values())
        else:
            alerts = list(self.alerts.values())
        
        # Apply filters
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
            
        if category is not None:
            alerts = [a for a in alerts if a.category == category]
            
        if source is not None:
            alerts = [a for a in alerts if a.source == source]
            
        if since is not None:
            alerts = [a for a in alerts if a.created_at >= since]
        
        # Sort by creation time (newest first) and apply limit
        alerts.sort(key=lambda a: a.created_at, reverse=True)
        return alerts[:limit]
    
    def _enforce_max_alerts(self) -> None:
        """
        Ensure the total number of alerts doesn't exceed the maximum.
        Removes the oldest resolved alerts first, then oldest active alerts if necessary.
        """
        if len(self.alerts) <= self.max_alerts:
            return
        
        # Get all alerts sorted by creation time (oldest first)
        all_alerts = sorted(
            self.alerts.values(),
            key=lambda a: a.created_at
        )
        
        # Split into active and resolved alerts
        resolved_alerts = [a for a in all_alerts if not a.is_active]
        active_alerts = [a for a in all_alerts if a.is_active]
        
        # Calculate how many alerts to remove
        excess = len(self.alerts) - self.max_alerts
        
        # Remove resolved alerts first
        for alert in resolved_alerts[:excess]:
            if alert.id in self.alerts:
                del self.alerts[alert.id]
                excess -= 1
            
            if excess <= 0:
                break
        
        # If we still have excess, remove oldest active alerts
        if excess > 0:
            for alert in active_alerts[:excess]:
                if alert.id in self.alerts:
                    del self.alerts[alert.id]
                    if alert.id in self.active_alerts:
                        del self.active_alerts[alert.id]
    
    def check_expired_alerts(self) -> None:
        """
        Check for and resolve any alerts that have passed their expiry time.
        """
        now = datetime.now()
        for alert_id, alert in list(self.active_alerts.items()):
            if alert.expiry and now >= alert.expiry:
                self.resolve_alert(
                    alert_id,
                    resolution_message="Alert automatically resolved due to expiry"
                )
    
    def get_active_alert_counts(self) -> Dict[str, int]:
        """
        Get counts of active alerts by level.
        
        Returns:
            Dictionary mapping level names to counts
        """
        counts = {level.value: 0 for level in AlertLevel}
        
        for alert in self.active_alerts.values():
            counts[alert.level] += 1
            
        return counts


# Global instance for alert management
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """
    Get the global AlertManager instance.
    
    Returns:
        The global AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def reset_alert_manager() -> None:
    """
    Reset the global AlertManager instance.
    
    This is primarily used for testing.
    """
    global _alert_manager
    _alert_manager = None 