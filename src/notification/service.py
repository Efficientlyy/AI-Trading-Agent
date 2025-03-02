"""
Notification service for the AI Trading System.

This module provides a service for sending notifications through various channels.
It integrates with the alert system to notify users of important events.
"""

import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable, Type

from src.notification.models import (
    NotificationChannel,
    NotificationPriority,
    Notification
)

# Configure logger
logger = logging.getLogger("notification.service")


# Provider interfaces
class NotificationProvider(ABC):
    """Interface for notification providers."""
    
    @abstractmethod
    def send_notification(self, notification: Notification) -> bool:
        """Send a notification through this provider."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this provider."""
        pass
    
    @abstractmethod
    def get_supported_channel(self) -> NotificationChannel:
        """Get the channel this provider supports."""
        pass


class ConsoleNotificationProvider(NotificationProvider):
    """Provider that outputs notifications to the console."""
    
    def send_notification(self, notification: Notification) -> bool:
        """Send a notification to the console."""
        # Format based on priority
        if notification.priority == NotificationPriority.URGENT:
            prefix = "🚨 URGENT"
        elif notification.priority == NotificationPriority.HIGH:
            prefix = "🔴 HIGH"
        elif notification.priority == NotificationPriority.MEDIUM:
            prefix = "🟠 MEDIUM"
        else:
            prefix = "🟢 LOW"
            
        print(f"\n{prefix} NOTIFICATION: {notification.title}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"{notification.message}")
        
        if notification.metadata:
            print(f"\nAdditional Information:")
            for key, value in notification.metadata.items():
                print(f"  - {key}: {value}")
        
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return True
    
    def get_name(self) -> str:
        """Get the name of this provider."""
        return "console"
    
    def get_supported_channel(self) -> NotificationChannel:
        """Get the channel this provider supports."""
        return NotificationChannel.CONSOLE


class FileNotificationProvider(NotificationProvider):
    """Provider that saves notifications to a file."""
    
    def __init__(self, notification_dir: Optional[Path] = None):
        """Initialize the file notification provider."""
        if notification_dir is None:
            # Default to a 'notifications' directory in the data directory
            self.notification_dir = Path("data/notifications")
        else:
            self.notification_dir = notification_dir
            
        # Ensure directory exists
        self.notification_dir.mkdir(parents=True, exist_ok=True)
        
        # Path for storing all notifications
        self.notification_file = self.notification_dir / "notifications.json"
        
        # Initialize empty notifications file if it doesn't exist
        if not self.notification_file.exists():
            with open(self.notification_file, 'w') as f:
                json.dump([], f)
    
    def send_notification(self, notification: Notification) -> bool:
        """Save a notification to the file."""
        try:
            # Read existing notifications
            notifications = []
            if self.notification_file.exists():
                with open(self.notification_file, 'r') as f:
                    try:
                        notifications = json.load(f)
                    except json.JSONDecodeError:
                        # If file is corrupted, start with an empty list
                        notifications = []
            
            # Add the new notification
            notifications.append(notification.to_dict())
            
            # Write back to file
            with open(self.notification_file, 'w') as f:
                json.dump(notifications, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error saving notification to file: {e}")
            return False
    
    def get_name(self) -> str:
        """Get the name of this provider."""
        return "file"
    
    def get_supported_channel(self) -> NotificationChannel:
        """Get the channel this provider supports."""
        return NotificationChannel.WEBHOOK  # Using webhook as a generic channel


# NotificationService singleton instance
_notification_service = None


class NotificationService:
    """
    Service for sending notifications through various channels.
    
    This service manages notification providers and handles the routing
    of notifications to appropriate providers based on channel.
    """
    
    def __init__(self):
        """Initialize the notification service."""
        self.providers: Dict[NotificationChannel, List[NotificationProvider]] = {}
        self.notifications: List[Notification] = []
        self.max_notifications = 1000
        
        # Register default providers
        self.register_provider(ConsoleNotificationProvider())
        self.register_provider(FileNotificationProvider())
        
    def register_provider(self, provider: NotificationProvider) -> None:
        """
        Register a notification provider.
        
        Args:
            provider: The provider to register
        """
        channel = provider.get_supported_channel()
        if channel not in self.providers:
            self.providers[channel] = []
            
        self.providers[channel].append(provider)
        logger.info(f"Registered notification provider: {provider.get_name()} for channel {channel.value}")
    
    def send_notification(
        self,
        title: str,
        message: str,
        channel: NotificationChannel,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        recipient: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Notification]:
        """
        Send a notification through the specified channel.
        
        Args:
            title: Title of the notification
            message: Body of the notification
            channel: Channel to send the notification through
            priority: Priority level of the notification
            recipient: Recipient of the notification (e.g., email address)
            metadata: Additional data to include with the notification
            
        Returns:
            The created Notification object, or None if sending failed
        """
        # Create notification object
        notification = Notification(
            id=str(uuid.uuid4()),
            title=title,
            message=message,
            channel=channel,
            priority=priority,
            recipient=recipient,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store the notification
        self.notifications.append(notification)
        
        # Enforce max notifications limit
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications:]
        
        # Check if we have providers for this channel
        if channel not in self.providers or not self.providers[channel]:
            logger.warning(f"No providers registered for channel: {channel.value}")
            notification.error = f"No providers available for channel: {channel.value}"
            return notification
        
        # Send through all providers for this channel
        success = False
        for provider in self.providers[channel]:
            try:
                if provider.send_notification(notification):
                    success = True
                    notification.sent = True
                    logger.info(f"Notification sent via {provider.get_name()}: {title}")
            except Exception as e:
                logger.error(f"Error sending notification via {provider.get_name()}: {e}")
                notification.error = f"Provider {provider.get_name()} error: {str(e)}"
        
        if success:
            notification.delivered = True
            
        return notification
    
    def get_notifications(
        self,
        limit: int = 100,
        channel: Optional[NotificationChannel] = None,
        priority: Optional[NotificationPriority] = None,
        since: Optional[datetime] = None
    ) -> List[Notification]:
        """
        Get notifications filtered by criteria.
        
        Args:
            limit: Maximum number of notifications to return
            channel: Filter by notification channel
            priority: Filter by priority level
            since: Only include notifications after this time
            
        Returns:
            List of notifications matching the criteria
        """
        filtered = self.notifications
        
        if channel:
            filtered = [n for n in filtered if n.channel == channel]
            
        if priority:
            filtered = [n for n in filtered if n.priority == priority]
            
        if since:
            filtered = [n for n in filtered if n.timestamp >= since]
            
        # Return most recent first
        filtered.sort(key=lambda n: n.timestamp, reverse=True)
        
        return filtered[:limit]


def get_notification_service() -> NotificationService:
    """
    Get the singleton instance of the notification service.
    
    Returns:
        The NotificationService instance
    """
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


def reset_notification_service() -> None:
    """Reset the notification service singleton for testing."""
    global _notification_service
    _notification_service = None 