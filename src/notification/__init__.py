"""
Notification module for the AI Trading System.

This module provides functionality for sending notifications about
important events through various channels (email, SMS, Slack, etc.).
"""

# Import key classes for easy access
from src.notification.service import (
    NotificationService,
    get_notification_service
)
from src.notification.models import (
    NotificationChannel,
    NotificationPriority,
    Notification
) 