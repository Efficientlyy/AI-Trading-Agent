#!/usr/bin/env python
"""
Example demonstrating the integration between alerts and notifications.

This script shows how to bridge the alert system with the notification system,
allowing alerts to be automatically sent as notifications.
"""

import sys
import os
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any

# Add the project root to the Python path to make imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger for this example
logger = logging.getLogger("alerts_notification_bridge")


# Define standalone alert classes to avoid dependency issues
class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    """Categories for different types of alerts."""
    SYSTEM = "system"
    EXCHANGE = "exchange"
    ORDER = "order"
    POSITION = "position"
    STRATEGY = "strategy"
    RISK = "risk"
    SECURITY = "security"


class Alert:
    """Class representing a single alert in the system."""
    
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
        """Initialize the alert."""
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
        """Resolve this alert."""
        if self.is_active:
            self.is_active = False
            self.resolved_at = datetime.now()
            self.resolution_message = resolution_message
            self.updated_at = self.resolved_at


class AlertHandler:
    """Interface for alert handlers."""
    
    def __init__(self, name: str):
        """Initialize the alert handler."""
        self.name = name
    
    def handle_alert(self, alert: Alert) -> None:
        """Handle an alert."""
        pass


class AlertManager:
    """Central manager for handling system alerts."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.handlers: List[AlertHandler] = []
    
    def add_handler(self, handler: AlertHandler) -> None:
        """Register a new alert handler."""
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
        """Create and process a new alert."""
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
        
        # Notify handlers
        for handler in self.handlers:
            try:
                handler.handle_alert(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.name}: {e}")
        
        return alert
    
    def resolve_alert(
        self,
        alert_id: str,
        resolution_message: Optional[str] = None
    ) -> Optional[Alert]:
        """Resolve an active alert."""
        if alert_id not in self.alerts:
            logger.warning(f"Cannot resolve alert {alert_id}: Alert not found")
            return None
        
        alert = self.alerts[alert_id]
        if not alert.is_active:
            logger.info(f"Alert {alert_id} is already resolved")
            return alert
        
        # Resolve the alert
        alert.resolve(resolution_message)
        logger.info(f"Alert {alert_id} resolved: {alert.message}")
        
        # Remove from active alerts
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
        
        # Notify handlers of the resolution
        for handler in self.handlers:
            try:
                handler.handle_alert(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.name}: {e}")
        
        return alert


# Define the notification bridge
class NotificationBridge:
    """Bridge between alerts and notifications."""
    
    def __init__(self):
        """Initialize the notification bridge."""
        try:
            from src.notification.models import (
                NotificationChannel,
                NotificationPriority,
                Notification
            )
            from src.notification.service import get_notification_service
            
            self.NotificationChannel = NotificationChannel
            self.NotificationPriority = NotificationPriority
            self.notification_service = get_notification_service()
            self.available = True
        except ImportError as e:
            logger.error(f"Could not import notification classes: {e}")
            self.available = False
    
    def alert_to_notification(
        self,
        alert: Alert,
        channel: str = "console"  # Using string to avoid direct enum reference
    ) -> None:
        """Convert an alert to a notification and send it."""
        if not self.available:
            logger.warning("Notification system not available")
            return
        
        # Map alert level to notification priority
        priority_map = {
            AlertLevel.INFO: self.NotificationPriority.LOW,
            AlertLevel.WARNING: self.NotificationPriority.MEDIUM,
            AlertLevel.ERROR: self.NotificationPriority.HIGH,
            AlertLevel.CRITICAL: self.NotificationPriority.URGENT
        }
        priority = priority_map.get(alert.level, self.NotificationPriority.MEDIUM)
        
        # Create a title based on category
        title_map = {
            AlertCategory.SYSTEM: "System Alert",
            AlertCategory.EXCHANGE: "Exchange Alert",
            AlertCategory.ORDER: "Order Alert",
            AlertCategory.POSITION: "Position Alert",
            AlertCategory.STRATEGY: "Strategy Alert",
            AlertCategory.RISK: "Risk Alert",
            AlertCategory.SECURITY: "Security Alert"
        }
        title = title_map.get(alert.category, f"{alert.category.capitalize()} Alert")
        
        # Include source in title if available
        if alert.source:
            title += f" - {alert.source}"
        
        # Add resolution info if resolved
        message = alert.message
        if not alert.is_active and alert.resolution_message:
            message += f"\nResolution: {alert.resolution_message}"
        
        # Send notification
        try:
            notification_channel = getattr(self.NotificationChannel, channel.upper())
            notification = self.notification_service.send_notification(
                title=title,
                message=message,
                channel=notification_channel,
                priority=priority,
                metadata=alert.details
            )
            
            if notification and notification.sent:
                logger.info(f"Sent notification for alert: {alert.id}")
            else:
                logger.warning(f"Failed to send notification for alert: {alert.id}")
        except AttributeError:
            logger.error(f"Invalid notification channel: {channel}")
        except Exception as e:
            logger.error(f"Error sending notification: {e}")


class NotificationAlertHandler(AlertHandler):
    """Alert handler that sends notifications."""
    
    def __init__(self, name: str = "notification_handler"):
        """Initialize the notification alert handler."""
        super().__init__(name)
        self.bridge = NotificationBridge()
    
    def handle_alert(self, alert: Alert) -> None:
        """Handle an alert by converting it to a notification and sending it."""
        self.bridge.alert_to_notification(alert)


async def generate_alerts_with_notifications():
    """Generate alerts that trigger notifications."""
    # Create an alert manager
    alert_manager = AlertManager()
    logger.info("Alert manager initialized")
    
    # Add a notification handler
    notification_handler = NotificationAlertHandler()
    alert_manager.add_handler(notification_handler)
    logger.info("Added notification handler to alert manager")
    
    # Ensure the alert directory exists
    alert_dir = Path("examples/data")
    alert_dir.mkdir(parents=True, exist_ok=True)
    
    # Create system startup alert
    logger.info("Creating system startup alert...")
    alert = alert_manager.create_alert(
        message="AI Trading System started",
        level=AlertLevel.INFO,
        category=AlertCategory.SYSTEM,
        source="system.startup",
        details={
            "version": "1.2.0",
            "startup_time": datetime.now().isoformat()
        }
    )
    logger.info(f"Created alert: {alert.id}")
    await asyncio.sleep(2)
    
    # Create exchange connection alerts
    exchanges = ["Binance", "Coinbase", "Kraken"]
    for exchange in exchanges:
        logger.info(f"Creating connection alert for {exchange}...")
        # 80% chance of successful connection
        if random.random() < 0.8:
            alert = alert_manager.create_alert(
                message=f"Connected to {exchange} exchange",
                level=AlertLevel.INFO,
                category=AlertCategory.EXCHANGE,
                source=f"exchange.{exchange.lower()}",
                details={
                    "exchange": exchange,
                    "latency_ms": round(random.uniform(50, 200), 2),
                    "connection_time": datetime.now().isoformat()
                }
            )
        else:
            alert = alert_manager.create_alert(
                message=f"Failed to connect to {exchange} exchange",
                level=AlertLevel.ERROR,
                category=AlertCategory.EXCHANGE,
                source=f"exchange.{exchange.lower()}",
                details={
                    "exchange": exchange,
                    "error": "Connection timeout",
                    "attempt_time": datetime.now().isoformat()
                }
            )
        logger.info(f"Created alert: {alert.id}")
        await asyncio.sleep(1)
    
    # Create order execution alert
    logger.info("Creating order execution alert...")
    order_id = f"order-{random.randint(10000, 99999)}"
    symbol = random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    side = random.choice(["buy", "sell"])
    quantity = round(random.uniform(0.1, 2.0), 4)
    price = round(random.uniform(40000, 60000), 2) if "BTC" in symbol else round(random.uniform(2000, 3000), 2)
    
    alert = alert_manager.create_alert(
        message=f"Order {order_id} executed successfully",
        level=AlertLevel.INFO,
        category=AlertCategory.ORDER,
        source=f"order.execution",
        details={
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "exchange": random.choice(exchanges),
            "execution_time": datetime.now().isoformat()
        }
    )
    logger.info(f"Created alert: {alert.id}")
    await asyncio.sleep(2)
    
    # Create risk warning alert
    logger.info("Creating risk warning alert...")
    portfolio_id = "main"
    exposure = round(random.uniform(65, 85), 2)
    threshold = 70.0
    
    if exposure > threshold:
        alert = alert_manager.create_alert(
            message=f"High exposure in portfolio {portfolio_id}: {exposure}%",
            level=AlertLevel.WARNING if exposure < 80 else AlertLevel.ERROR,
            category=AlertCategory.RISK,
            source="risk.exposure",
            details={
                "portfolio_id": portfolio_id,
                "exposure_percent": exposure,
                "threshold": threshold,
                "timestamp": datetime.now().isoformat()
            }
        )
        logger.info(f"Created alert: {alert.id}")
    
    # Wait a moment for all notifications to be processed
    await asyncio.sleep(2)
    
    logger.info("Bridge example completed")


async def main():
    """Run the alerts notification bridge example."""
    logger.info("Starting alerts notification bridge example")
    await generate_alerts_with_notifications()


if __name__ == "__main__":
    asyncio.run(main()) 