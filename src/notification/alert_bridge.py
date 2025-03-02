"""
Bridge between the alert system and the notification system.

This module provides functionality to convert alerts to notifications
and handle alert events by sending appropriate notifications.
"""

import logging
from typing import Dict, Any, Optional

try:
    from src.monitoring.alerts import (
        AlertLevel, 
        AlertCategory, 
        Alert,
        AlertHandler
    )
except ImportError:
    # For standalone mode or when alerts module isn't available
    class AlertLevel:
        INFO = "info"
        WARNING = "warning"
        ERROR = "error" 
        CRITICAL = "critical"
    
    class AlertCategory:
        SYSTEM = "system"
        EXCHANGE = "exchange"
        ORDER = "order"
        POSITION = "position"
        STRATEGY = "strategy"
        RISK = "risk"
        SECURITY = "security"
    
    class Alert:
        def __init__(
            self, 
            message: str,
            level: str,
            category: str,
            source: str,
            details: Optional[Dict[str, Any]] = None
        ):
            self.message = message
            self.level = level
            self.category = category
            self.source = source
            self.details = details or {}
    
    class AlertHandler:
        def __init__(self, name: str):
            self.name = name
        
        def handle_alert(self, alert: Alert) -> None:
            pass

from src.notification.models import (
    NotificationChannel,
    NotificationPriority,
    Notification
)
from src.notification.service import get_notification_service

# Configure logger
logger = logging.getLogger("notification.alert_bridge")


# Mapping from alert levels to notification priorities
LEVEL_TO_PRIORITY = {
    AlertLevel.INFO: NotificationPriority.LOW,
    AlertLevel.WARNING: NotificationPriority.MEDIUM,
    AlertLevel.ERROR: NotificationPriority.HIGH,
    AlertLevel.CRITICAL: NotificationPriority.URGENT
}

# Mapping from alert categories to more human-readable titles
CATEGORY_TITLES = {
    AlertCategory.SYSTEM: "System Alert",
    AlertCategory.EXCHANGE: "Exchange Alert",
    AlertCategory.ORDER: "Order Alert",
    AlertCategory.POSITION: "Position Alert",
    AlertCategory.STRATEGY: "Strategy Alert",
    AlertCategory.RISK: "Risk Alert",
    AlertCategory.SECURITY: "Security Alert"
}


class NotificationAlertHandler(AlertHandler):
    """
    Alert handler that sends notifications for alerts.
    
    This handler converts alerts to notifications and sends them
    through the notification system.
    """
    
    def __init__(
        self,
        channel: NotificationChannel = NotificationChannel.CONSOLE,
        recipient: str = "",
        name: str = "notification_handler"
    ):
        """
        Initialize the notification alert handler.
        
        Args:
            channel: The notification channel to use
            recipient: The recipient for notifications (e.g., email address)
            name: Unique name for this handler
        """
        super().__init__(name)
        self.channel = channel
        self.recipient = recipient
        self.notification_service = get_notification_service()
    
    def handle_alert(self, alert: Alert) -> None:
        """
        Handle an alert by converting it to a notification and sending it.
        
        Args:
            alert: The alert to handle
        """
        # Skip already-resolved alerts
        if hasattr(alert, 'is_active') and not alert.is_active:
            return
        
        # Map alert level to notification priority
        priority = LEVEL_TO_PRIORITY.get(
            alert.level, 
            NotificationPriority.MEDIUM
        )
        
        # Create a title based on category
        title = CATEGORY_TITLES.get(
            alert.category, 
            f"{alert.category.capitalize()} Alert"
        )
        
        # Include source in title if available
        if hasattr(alert, 'source') and alert.source:
            title += f" - {alert.source}"
        
        # Send notification
        notification = self.notification_service.send_notification(
            title=title,
            message=alert.message,
            channel=self.channel,
            priority=priority,
            recipient=self.recipient,
            metadata=alert.details
        )
        
        if notification and notification.sent:
            logger.info(f"Sent notification for alert: {alert.message}")
        else:
            logger.warning(f"Failed to send notification for alert: {alert.message}")


def get_notification_handler(
    channel: NotificationChannel = NotificationChannel.CONSOLE,
    recipient: str = ""
) -> NotificationAlertHandler:
    """
    Create a notification alert handler.
    
    Args:
        channel: The notification channel to use
        recipient: The recipient for notifications
        
    Returns:
        A NotificationAlertHandler instance
    """
    return NotificationAlertHandler(
        channel=channel,
        recipient=recipient
    ) 