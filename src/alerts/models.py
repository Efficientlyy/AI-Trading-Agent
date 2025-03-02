"""
Alert models for the AI Crypto Trading Agent.

This module defines the data structures for the alert system.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, Optional, Any


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
    FEE = "fee"  # Fee-related alerts
    ROUTING = "routing"  # Order routing alerts


class Alert:
    """
    Class representing a single alert in the system.
    
    An alert captures information about events that may require attention
    from users or other system components.
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
        Initialize a new Alert.
        
        Args:
            message: Human-readable alert message
            level: Severity level of the alert
            category: Category the alert belongs to
            source: Component or system that generated the alert
            details: Additional structured data related to the alert
            expiry: Optional datetime when the alert should auto-resolve
            is_active: Whether the alert is currently active
            id: Optional unique identifier for the alert (generated if None)
        """
        import uuid
        
        self.id = id if id is not None else str(uuid.uuid4())
        self.message = message
        self.level = level
        self.category = category
        self.source = source
        self.details = details or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.is_active = is_active
        self.expiry = expiry
        self.resolved_at = None
        self.resolution_message = None
    
    def resolve(self, resolution_message: Optional[str] = None) -> None:
        """
        Mark the alert as resolved.
        
        Args:
            resolution_message: Optional message explaining how the alert was resolved
        """
        if self.is_active:
            self.is_active = False
            self.resolved_at = datetime.now()
            self.updated_at = self.resolved_at
            self.resolution_message = resolution_message
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the alert to a dictionary for serialization.
        
        Returns:
            A dictionary representation of the alert
        """
        return {
            "id": self.id,
            "message": self.message,
            "level": self.level.value,
            "category": self.category.value,
            "source": self.source,
            "details": self.details,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_message": self.resolution_message
        } 