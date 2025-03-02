"""
Simplified example script demonstrating how to use the monitoring service.

This example is a modified version that doesn't depend on the config system.
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from decimal import Decimal

# Mock event bus class to avoid config dependency
class SimpleEventBus:
    async def publish(self, event):
        print(f"Published event: {event.__class__.__name__}")

# Mock monitoring service
class SimpleMonitoringService:
    def __init__(self):
        self.system_status = {
            "system_stats": {
                "uptime_seconds": 3600,
                "error_count": 2
            },
            "component_status": {
                "execution": {"status": "ok", "message": "Running normally"},
                "data": {"status": "warning", "message": "Experiencing delays"},
                "strategy": {"status": "error", "message": "Encountered an error"}
            }
        }
        self.active_orders = [
            {"order_id": "test-order-123", "symbol": "BTC/USDT", "side": "BUY", "status": "FILLED"}
        ]
        self.recent_trades = [
            {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "symbol": "BTC/USDT", "side": "BUY", "price": "49950.00"}
        ]
        self.performance_metrics = {
            "performance_metrics": {
                "execution_metrics": {
                    "orders_submitted": 45,
                    "orders_filled": 40,
                    "orders_cancelled": 3,
                    "orders_rejected": 2,
                    "avg_fill_time_ms": 250
                }
            }
        }
    
    async def initialize(self):
        print("Monitoring service initialized")
    
    async def start(self):
        print("Monitoring service started")
    
    async def stop(self):
        print("Monitoring service stopped")
    
    def get_system_status(self):
        return self.system_status
    
    def get_active_orders(self):
        return self.active_orders
    
    def get_recent_trades(self):
        return self.recent_trades
    
    def get_performance_metrics(self):
        return self.performance_metrics

# Mock event classes
class SystemStatusEvent:
    def __init__(self, component_id, status, message, details):
        self.component_id = component_id
        self.status = status
        self.message = message
        self.details = details

class OrderSide:
    BUY = "BUY"
    SELL = "SELL"

class OrderType:
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderStatus:
    CREATED = "CREATED"
    OPEN = "OPEN"
    FILLED = "FILLED"

class TimeInForce:
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"

class Order:
    def __init__(self, symbol, side, order_type, quantity, price, exchange, time_in_force, id, client_order_id, strategy_id, status):
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.exchange = exchange
        self.time_in_force = time_in_force
        self.id = id
        self.client_order_id = client_order_id
        self.strategy_id = strategy_id
        self.status = status
        self.submitted_at = None
        self.updated_at = None
    
    def update_status(self, status, filled_qty=None, avg_price=None):
        self.status = status
        if filled_qty is not None:
            self.filled_quantity = filled_qty
        if avg_price is not None:
            self.avg_fill_price = avg_price

class OrderEvent:
    def __init__(self, order_id, order_event_type, order):
        self.order_id = order_id
        self.order_event_type = order_event_type
        self.order = order

class PerformanceEvent:
    def __init__(self, metric_type, strategy_id=None, portfolio_id=None, metrics=None):
        self.metric_type = metric_type
        self.strategy_id = strategy_id
        self.portfolio_id = portfolio_id
        self.metrics = metrics or {}

# Global instances
event_bus = SimpleEventBus()
monitoring_service = SimpleMonitoringService()

async def report_system_status():
    """Report system status events."""
    # Send OK status for execution service
    event = SystemStatusEvent(
        component_id="execution",
        status="ok",
        message="Execution service running normally",
        details={
            "active_connections": 2,
            "pending_orders": 3
        }
    )
    await event_bus.publish(event)
    print(f"Published system status event for execution service: status={event.status}")
    
    # Send warning status for data service
    event = SystemStatusEvent(
        component_id="data",
        status="warning",
        message="Data service experiencing delays",
        details={
            "latency_ms": 500,
            "data_sources": ["binance", "coinbase"]
        }
    )
    await event_bus.publish(event)
    print(f"Published system status event for data service: status={event.status}")
    
    # Send error status for strategy service
    event = SystemStatusEvent(
        component_id="strategy",
        status="error",
        message="Strategy service encountered an error",
        details={
            "error_type": "RuntimeError",
            "error_details": "Failed to calculate indicator: division by zero"
        }
    )
    await event_bus.publish(event)
    print(f"Published system status event for strategy service: status={event.status}")


async def simulate_order_lifecycle():
    """Simulate an order lifecycle with events."""
    # Create a test order
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.01"),
        price=Decimal("50000"),
        exchange="binance",
        time_in_force=TimeInForce.GTC,
        id="test-order-123",
        client_order_id="client-order-123",
        strategy_id="test-strategy",
        status=OrderStatus.CREATED
    )
    
    # Order submitted event
    event = OrderEvent(
        order_id=order.id,
        order_event_type="SUBMITTED",
        order=order
    )
    await event_bus.publish(event)
    print(f"Published order event: order_id={order.id}, type={event.order_event_type}")
    
    # Wait a bit for the order to be "processed"
    await asyncio.sleep(1)
    
    # Update order status to "open"
    order.update_status(OrderStatus.OPEN)
    order.submitted_at = datetime.now()
    
    # Wait a bit for the order to be "filled"
    await asyncio.sleep(2)
    
    # Update order fill information
    order.update_status(OrderStatus.FILLED, Decimal("0.01"), Decimal("49950"))
    order.updated_at = datetime.now()
    
    # Order filled event
    event = OrderEvent(
        order_id=order.id,
        order_event_type="FILLED",
        order=order
    )
    await event_bus.publish(event)
    print(f"Published order event: order_id={order.id}, type={event.order_event_type}")


async def report_performance_metrics():
    """Report performance metrics events."""
    # Strategy performance event
    strategy_event = PerformanceEvent(
        metric_type="strategy",
        strategy_id="trend-following-btc",
        metrics={
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.12,
            "total_trades": 45,
            "winning_trades": 30,
            "losing_trades": 15,
            "avg_win": 250.75,
            "avg_loss": 150.25,
            "expectancy": 95.50
        }
    )
    await event_bus.publish(strategy_event)
    print(f"Published strategy performance event for strategy_id={strategy_event.strategy_id}")
    
    # Portfolio performance event
    portfolio_event = PerformanceEvent(
        metric_type="portfolio",
        portfolio_id="main",
        metrics={
            "total_value": 25000.50,
            "starting_value": 20000.00,
            "pnl": 5000.50,
            "pnl_percent": 25.0,
            "exposure": 0.75,
            "margin_used": 15000.00,
            "free_margin": 10000.50,
            "positions": 5,
            "daily_pnl": 250.75,
            "weekly_pnl": 1250.50,
            "monthly_pnl": 3500.25
        }
    )
    await event_bus.publish(portfolio_event)
    print(f"Published portfolio performance event for portfolio_id={portfolio_event.portfolio_id}")


async def access_monitoring_data():
    """Access and display monitoring data."""
    global monitoring_service
    
    # Get system status
    status = monitoring_service.get_system_status()
    print("\nSystem Status:")
    print(f"  Uptime: {status['system_stats']['uptime_seconds']} seconds")
    print(f"  Error count: {status['system_stats']['error_count']}")
    print("  Components:")
    for component_id, component in status['component_status'].items():
        print(f"    {component_id}: {component['status']}")
    
    # Get active orders
    active_orders = monitoring_service.get_active_orders()
    print("\nActive Orders:")
    if active_orders:
        for order in active_orders:
            print(f"  {order['order_id']} - {order['symbol']} - {order['side']} - {order['status']}")
    else:
        print("  No active orders")
    
    # Get recent trades
    recent_trades = monitoring_service.get_recent_trades()
    print("\nRecent Trades:")
    if recent_trades:
        for trade in recent_trades:
            print(f"  {trade['time']} - {trade['symbol']} - {trade['side']} - {trade['price']}")
    else:
        print("  No recent trades")
    
    # Get performance metrics
    metrics = monitoring_service.get_performance_metrics()
    print("\nExecution Metrics:")
    execution = metrics['performance_metrics']['execution_metrics']
    print(f"  Orders submitted: {execution['orders_submitted']}")
    print(f"  Orders filled: {execution['orders_filled']}")
    print(f"  Average fill time: {execution.get('avg_fill_time_ms', 'N/A')} ms")


async def main():
    """Run the monitoring example."""
    print("Starting monitoring example...")
    
    # Initialize monitoring service
    global monitoring_service
    await monitoring_service.initialize()
    await monitoring_service.start()
    print("Monitoring service initialized and started")
    
    # Run example functions
    await report_system_status()
    await simulate_order_lifecycle()
    await report_performance_metrics()
    
    # Wait a bit for events to be processed
    await asyncio.sleep(2)
    
    # Access and display monitoring data
    await access_monitoring_data()
    
    # Stop monitoring service
    await monitoring_service.stop()
    print("Monitoring service stopped")


if __name__ == "__main__":
    asyncio.run(main()) 