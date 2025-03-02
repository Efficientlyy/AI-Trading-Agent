#!/usr/bin/env python
"""
Standalone example demonstrating the notification system.

This script shows how to use the notification system to send notifications
through various channels and with different priority levels.
"""

import sys
import os
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path to make imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger for this example
logger = logging.getLogger("notification_example")


def main():
    """Run the notification example."""
    logger.info("Starting notification example")
    
    try:
        from src.notification.models import (
            NotificationChannel,
            NotificationPriority,
            Notification
        )
        from src.notification.service import get_notification_service
    except ImportError as e:
        logger.error(f"Could not import notification classes: {e}")
        sys.exit(1)
    
    # Get the notification service
    notification_service = get_notification_service()
    logger.info("Notification service initialized")
    
    # Ensure the notification directory exists
    notification_dir = Path("examples/data/notifications")
    notification_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Notifications will be saved to {notification_dir.resolve()}")
    
    # Create some sample notifications
    logger.info("Sending sample notifications...")
    
    # System startup notification (low priority)
    notification_service.send_notification(
        title="System Started",
        message="The trading system has started successfully.",
        channel=NotificationChannel.CONSOLE,
        priority=NotificationPriority.LOW,
        metadata={
            "startup_time": datetime.now().isoformat(),
            "version": "1.2.0"
        }
    )
    time.sleep(1)
    
    # Exchange connectivity notification (medium priority)
    exchanges = ["Binance", "Coinbase", "Kraken", "OKX", "Bybit"]
    for exchange in random.sample(exchanges, 3):
        connected = random.random() > 0.2  # 80% chance of success
        
        if connected:
            notification_service.send_notification(
                title=f"Connected to {exchange}",
                message=f"Successfully established connection to {exchange} exchange.",
                channel=NotificationChannel.CONSOLE,
                priority=NotificationPriority.MEDIUM,
                metadata={
                    "exchange": exchange,
                    "latency_ms": round(random.uniform(50, 200), 2),
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            notification_service.send_notification(
                title=f"Failed to connect to {exchange}",
                message=f"Could not establish connection to {exchange} exchange. Retrying in 60 seconds.",
                channel=NotificationChannel.CONSOLE,
                priority=NotificationPriority.HIGH,
                metadata={
                    "exchange": exchange,
                    "error": "Connection timeout",
                    "retry_count": random.randint(1, 3),
                    "timestamp": datetime.now().isoformat()
                }
            )
        time.sleep(1)
    
    # Order execution notification (high priority)
    order_id = f"ord-{random.randint(10000, 99999)}"
    symbol = random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    exchange = random.choice(exchanges)
    side = random.choice(["BUY", "SELL"])
    quantity = round(random.uniform(0.1, 2.0), 4)
    price = round(random.uniform(10000, 60000), 2) if "BTC" in symbol else round(random.uniform(1000, 3000), 2)
    
    notification_service.send_notification(
        title=f"Order Executed: {order_id}",
        message=f"Order {order_id} ({side} {quantity} {symbol}) was executed on {exchange} at ${price}.",
        channel=NotificationChannel.CONSOLE,
        priority=NotificationPriority.HIGH,
        metadata={
            "order_id": order_id,
            "symbol": symbol,
            "exchange": exchange,
            "side": side,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now().isoformat()
        }
    )
    time.sleep(1)
    
    # Risk alert notification (urgent priority)
    if random.random() > 0.5:  # 50% chance
        portfolio_id = "main"
        margin_used = round(random.uniform(80, 95), 2)
        
        notification_service.send_notification(
            title="RISK ALERT: High Margin Usage",
            message=f"Portfolio {portfolio_id} has reached {margin_used}% margin usage, exceeding the 80% threshold!",
            channel=NotificationChannel.CONSOLE,
            priority=NotificationPriority.URGENT,
            metadata={
                "portfolio_id": portfolio_id,
                "margin_used_percent": margin_used,
                "margin_threshold": 80.0,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # Report on sent notifications
    logger.info("\n--- NOTIFICATION SUMMARY ---")
    
    priorities = {
        NotificationPriority.URGENT: "URGENT",
        NotificationPriority.HIGH: "HIGH",
        NotificationPriority.MEDIUM: "MEDIUM",
        NotificationPriority.LOW: "LOW"
    }
    
    for priority in NotificationPriority:
        notifications = notification_service.get_notifications(
            priority=priority,
            limit=100
        )
        if notifications:
            logger.info(f"\n{priorities[priority]} NOTIFICATIONS ({len(notifications)}):")
            for i, notif in enumerate(notifications, 1):
                logger.info(f"  {i}. {notif.title}")
                logger.info(f"     Message: {notif.message}")
                logger.info(f"     Sent: {'Yes' if notif.sent else 'No'}")
                if notif.metadata:
                    meta_str = ", ".join(f"{k}: {v}" for k, v in notif.metadata.items())
                    logger.info(f"     Details: {meta_str}")
    
    logger.info("\nNotification example completed")


if __name__ == "__main__":
    main() 