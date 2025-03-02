"""
Example script demonstrating the alert system.

This script shows how to create, manage, and handle alerts in the trading system.
"""

import asyncio
import sys
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path to make imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.monitoring.alerts import (
        AlertLevel,
        AlertCategory,
        get_alert_manager,
        FileAlertHandler,
        LoggingAlertHandler
    )
except ImportError:
    print("Could not import alert classes. Please run this script from the project root.")
    sys.exit(1)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger for this example
logger = logging.getLogger("alerts_example")


async def generate_exchange_alerts():
    """Generate alerts related to exchange connectivity."""
    alert_manager = get_alert_manager()
    
    # Exchange connection established
    exchange_id = random.choice(["binance", "coinbase", "kraken"])
    logger.info(f"Connecting to {exchange_id}...")
    
    # Create info alert for successful connection
    connect_alert = alert_manager.create_alert(
        message=f"Connected to {exchange_id} exchange",
        level=AlertLevel.INFO,
        category=AlertCategory.EXCHANGE,
        source=f"exchange.{exchange_id}",
        expiry=datetime.now() + timedelta(minutes=10)
    )
    
    logger.info(f"Created alert {connect_alert.id}")
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Simulate high latency
    logger.info(f"Simulating high latency with {exchange_id}...")
    latency = random.uniform(350, 500)
    
    # Create warning alert for high latency
    latency_alert = alert_manager.create_alert(
        message=f"High latency detected on {exchange_id} exchange",
        level=AlertLevel.WARNING,
        category=AlertCategory.EXCHANGE,
        source=f"exchange.{exchange_id}",
        details={
            "latency_ms": latency,
            "threshold_ms": 300,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    logger.info(f"Created alert {latency_alert.id}")
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Simulate latency recovery
    logger.info(f"Simulating latency recovery for {exchange_id}...")
    
    # Resolve the latency alert
    alert_manager.resolve_alert(
        latency_alert.id,
        resolution_message=f"Latency recovered to normal levels: 150ms"
    )
    
    logger.info(f"Resolved alert {latency_alert.id}")
    
    # Simulate API rate limit exceeded (50% chance)
    if random.random() < 0.5:
        logger.info(f"Simulating API rate limit exceeded for {exchange_id}...")
        
        # Create error alert for rate limit
        rate_limit_alert = alert_manager.create_alert(
            message=f"API rate limit exceeded on {exchange_id} exchange",
            level=AlertLevel.ERROR,
            category=AlertCategory.EXCHANGE,
            source=f"exchange.{exchange_id}",
            details={
                "retry_after": 30,
                "request_path": "/api/v3/trades",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Created alert {rate_limit_alert.id}")
        
        # Wait a bit
        await asyncio.sleep(1)
        
        # Update the alert with more information
        alert_manager.update_alert(
            rate_limit_alert.id,
            details={
                "requests_count": random.randint(10, 50),
                "limit": 10,
                "window_seconds": 60
            }
        )
        
        logger.info(f"Updated alert {rate_limit_alert.id}")


async def generate_order_alerts():
    """Generate alerts related to order execution."""
    alert_manager = get_alert_manager()
    
    # Create simulated order IDs
    order_id = f"order-{random.randint(10000, 99999)}"
    symbol = random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
    exchange = random.choice(["binance", "coinbase", "kraken"])
    logger.info(f"Simulating order {order_id} for {symbol} on {exchange}")
    
    # Order submitted successfully
    submit_alert = alert_manager.create_alert(
        message=f"Order {order_id} submitted to {exchange}",
        level=AlertLevel.INFO,
        category=AlertCategory.ORDER,
        source="execution.service",
        details={
            "order_id": order_id,
            "symbol": symbol,
            "exchange": exchange,
            "side": "BUY",
            "type": "LIMIT",
            "quantity": 0.1,
            "price": 50000.0,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    logger.info(f"Created alert {submit_alert.id}")
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Simulate order outcome (filled, partially filled, or rejected)
    outcome = random.choices(
        ["filled", "partial", "rejected", "timeout"],
        weights=[0.7, 0.1, 0.1, 0.1]
    )[0]
    
    if outcome == "filled":
        # Order filled successfully
        logger.info(f"Simulating successful fill for order {order_id}")
        
        fill_alert = alert_manager.create_alert(
            message=f"Order {order_id} filled successfully",
            level=AlertLevel.INFO,
            category=AlertCategory.ORDER,
            source="execution.service",
            details={
                "order_id": order_id,
                "symbol": symbol,
                "exchange": exchange,
                "fill_price": round(random.uniform(49800, 50200), 2),
                "fill_quantity": 0.1,
                "fill_time": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Created alert {fill_alert.id}")
        
    elif outcome == "partial":
        # Order partially filled
        logger.info(f"Simulating partial fill for order {order_id}")
        
        partial_fill_alert = alert_manager.create_alert(
            message=f"Order {order_id} partially filled",
            level=AlertLevel.WARNING,
            category=AlertCategory.ORDER,
            source="execution.service",
            details={
                "order_id": order_id,
                "symbol": symbol,
                "exchange": exchange,
                "fill_price": round(random.uniform(49800, 50200), 2),
                "fill_quantity": 0.05,
                "total_quantity": 0.1,
                "fill_time": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Created alert {partial_fill_alert.id}")
        
    elif outcome == "rejected":
        # Order rejected
        logger.info(f"Simulating rejection for order {order_id}")
        
        reject_alert = alert_manager.create_alert(
            message=f"Order {order_id} rejected by {exchange}",
            level=AlertLevel.ERROR,
            category=AlertCategory.ORDER,
            source="execution.service",
            details={
                "order_id": order_id,
                "symbol": symbol,
                "exchange": exchange,
                "reason": random.choice([
                    "Insufficient funds",
                    "Price outside allowed bounds",
                    "Invalid quantity",
                    "Market closed"
                ]),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Created alert {reject_alert.id}")
        
    elif outcome == "timeout":
        # Order timed out
        logger.info(f"Simulating timeout for order {order_id}")
        
        timeout_alert = alert_manager.create_alert(
            message=f"Order {order_id} timed out",
            level=AlertLevel.CRITICAL,
            category=AlertCategory.ORDER,
            source="execution.service",
            details={
                "order_id": order_id,
                "symbol": symbol,
                "exchange": exchange,
                "timeout_seconds": 30,
                "last_status": "PENDING",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Created alert {timeout_alert.id}")


async def generate_risk_alerts():
    """Generate alerts related to risk management."""
    alert_manager = get_alert_manager()
    
    # Simulate portfolio risk alerts
    portfolio_id = "main"
    logger.info(f"Simulating risk alerts for portfolio {portfolio_id}")
    
    # Calculate some mock risk metrics
    margin_used_pct = random.uniform(70, 95)
    exposure_pct = random.uniform(60, 90)
    concentration_pct = random.uniform(30, 60)
    
    # Create risk alert if margin usage is high
    if margin_used_pct > 90:
        level = AlertLevel.CRITICAL
        message = f"Critical margin usage for portfolio {portfolio_id}: {margin_used_pct:.2f}%"
    elif margin_used_pct > 80:
        level = AlertLevel.ERROR
        message = f"High margin usage for portfolio {portfolio_id}: {margin_used_pct:.2f}%"
    elif margin_used_pct > 70:
        level = AlertLevel.WARNING
        message = f"Elevated margin usage for portfolio {portfolio_id}: {margin_used_pct:.2f}%"
    else:
        level = AlertLevel.INFO
        message = f"Normal margin usage for portfolio {portfolio_id}: {margin_used_pct:.2f}%"
    
    margin_alert = alert_manager.create_alert(
        message=message,
        level=level,
        category=AlertCategory.RISK,
        source="risk.manager",
        details={
            "portfolio_id": portfolio_id,
            "margin_used_percent": margin_used_pct,
            "margin_threshold": 80.0,
            "exposure_percent": exposure_pct,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    logger.info(f"Created alert {margin_alert.id}")
    
    # Create concentration alert if a single position is too large
    if concentration_pct > 50:
        conc_alert = alert_manager.create_alert(
            message=f"High position concentration detected in portfolio {portfolio_id}",
            level=AlertLevel.WARNING,
            category=AlertCategory.RISK,
            source="risk.manager",
            details={
                "portfolio_id": portfolio_id,
                "asset": random.choice(["BTC", "ETH", "SOL"]),
                "concentration_percent": concentration_pct,
                "max_allowed_percent": 50.0,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Created alert {conc_alert.id}")


async def display_alerts():
    """Display all alerts in the system."""
    alert_manager = get_alert_manager()
    
    logger.info("\n--- ACTIVE ALERTS ---")
    active_alerts = alert_manager.get_alerts(active_only=True)
    
    for level in [AlertLevel.CRITICAL, AlertLevel.ERROR, AlertLevel.WARNING, AlertLevel.INFO]:
        level_alerts = [a for a in active_alerts if a.level == level]
        if level_alerts:
            logger.info(f"\n{level.upper()} ALERTS ({len(level_alerts)}):")
            for alert in level_alerts:
                logger.info(f"  - [{alert.category}] {alert.message}")
                if alert.details:
                    logger.info(f"    Details: {alert.details}")
    
    logger.info("\n--- RECENTLY RESOLVED ALERTS ---")
    resolved_alerts = [a for a in alert_manager.get_alerts(active_only=False) 
                      if not a.is_active][:5]
    
    for alert in resolved_alerts:
        logger.info(f"  - [{alert.category}] {alert.message}")
        if alert.resolution_message:
            logger.info(f"    Resolution: {alert.resolution_message}")


async def main():
    """Run the alerts example."""
    logger.info("Starting alerts example")
    
    # Set up a file handler to save alerts to a file
    alert_manager = get_alert_manager()
    
    alert_file = Path("./examples/data/alerts.json")
    alert_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = FileAlertHandler(alert_file)
    alert_manager.add_handler(file_handler)
    
    logger.info(f"Alerts will be saved to {alert_file}")
    
    # Create system startup alert
    startup_alert = alert_manager.create_alert(
        message="Alert example script started",
        level=AlertLevel.INFO,
        category=AlertCategory.SYSTEM,
        source="alerts_example",
        expiry=datetime.now() + timedelta(minutes=5)
    )
    
    # Generate various types of alerts
    await generate_exchange_alerts()
    await asyncio.sleep(1)
    
    await generate_order_alerts()
    await asyncio.sleep(1)
    
    await generate_risk_alerts()
    await asyncio.sleep(1)
    
    # Display all active alerts
    await display_alerts()
    
    # Resolve the startup alert
    alert_manager.resolve_alert(
        startup_alert.id,
        resolution_message="Alert example script completed"
    )
    
    logger.info(f"\nAlerts have been saved to {alert_file}")
    logger.info("End of alerts example")


if __name__ == "__main__":
    asyncio.run(main()) 