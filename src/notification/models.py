"""
Data models for the notification system.

This module defines the data structures and types used by the notification system.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


class NotificationChannel(str, Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"
    CONSOLE = "console"


class NotificationPriority(str, Enum):
    """Priority levels for notifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Data model representing a notification."""
    id: str
    title: str
    message: str
    channel: NotificationChannel
    priority: NotificationPriority
    recipient: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent: bool = False
    delivered: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to a dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "channel": self.channel.value,
            "priority": self.priority.value,
            "recipient": self.recipient,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "sent": self.sent,
            "delivered": self.delivered,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Notification':
        """Create a notification from a dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            message=data["message"],
            channel=NotificationChannel(data["channel"]),
            priority=NotificationPriority(data["priority"]),
            recipient=data["recipient"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            sent=data.get("sent", False),
            delivered=data.get("delivered", False),
            error=data.get("error")
        ) 