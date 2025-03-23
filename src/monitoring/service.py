"""
Monitoring Service.

This service provides a centralized monitoring system for the trading application,
collecting statistics, health information, and performance metrics from all components.
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from src.common.component import Component
from src.common.config import config
from src.common.events import event_bus
from src.common.logging import get_logger
from src.models.events import SystemStatusEvent, OrderEvent, PerformanceEvent
from src.models.order import Order, OrderStatus
from src.monitoring.alerts import (
    AlertLevel, 
    AlertCategory,
    Alert,
    AlertManager,
    FileAlertHandler,
    get_alert_manager
)


class MonitoringService(Component):
    """Service for monitoring the trading system.
    
    This component collects and tracks:
    - System health and component status
    - Order history and execution statistics
    - Performance metrics
    - Trading activity
    - Error rates and system alerts
    """
    
    def __init__(self):
        """Initialize the monitoring service."""
        super().__init__("monitoring")
        self.logger = get_logger("monitoring", "service")
        
        # Health status of components
        self.component_status: Dict[str, Dict[str, Any]] = {}
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Dict[str, Any]] = []
        self.recent_trades: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.performance_metrics: Dict[str, Any] = {
            "strategy_performance": {},
            "portfolio_performance": {},
            "execution_metrics": {
                "orders_submitted": 0,
                "orders_filled": 0,
                "orders_cancelled": 0,
                "orders_rejected": 0,
                "avg_fill_time_ms": 0,
                "total_fill_time_ms": 0
            }
        }
        
        # System statistics
        self.system_stats: Dict[str, Any] = {
            "start_time": None,
            "uptime_seconds": 0,
            "error_count": 0,
            "last_error_time": None,
            "last_error_message": None
        }
        
        # Data collection configuration
        self.data_retention_days = 7
        self.snapshot_interval_seconds = 300  # 5 minutes
        self.snapshot_dir = None
        
        # Status summary flags
        self.has_critical_errors = False
        self.has_warnings = False
        
        # Set up alert manager
        self.alert_manager = get_alert_manager()
        
        # Add file alert handler
        alert_file_path = Path(self.get_config("alert_file", "data/monitoring/alerts.json"))
        alert_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_manager.add_handler(FileAlertHandler(alert_file_path))
        
        # Start time for uptime calculation
        self.start_time = datetime.now()
        
        # Track registered components
        self.registered_components: Set[str] = set()
        
        # Track if the service is running
        self.running = False
        self.snapshot_task = None
        
    async def _initialize(self) -> None:
        """Initialize the monitoring service."""
        self.logger.info("Initializing monitoring service")
        
        # Load configuration
        self.data_retention_days = self.get_config("data_retention.days", 7)
        self.snapshot_interval_seconds = self.get_config("snapshot_interval_seconds", 300)
        
        # Setup snapshot directory
        snapshot_dir = self.get_config("snapshot_dir", "logs/monitoring")
        if not os.path.isabs(snapshot_dir):
            # Make it relative to project root
            from src.common.utils import get_project_root
            snapshot_dir = os.path.join(get_project_root(), snapshot_dir)
        
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component status for known components
        for component_id in ["execution", "portfolio", "strategy", "data"]:
            self.component_status[component_id] = {
                "status": "unknown",
                "last_heartbeat": None,
                "details": {},
                "errors": []
            }
        
        # Initialize system stats
        self.system_stats["start_time"] = datetime.now().isoformat()
        
        self.logger.info("Monitoring service initialized",
                      data_retention_days=self.data_retention_days,
                      snapshot_interval=self.snapshot_interval_seconds,
                      snapshot_dir=str(self.snapshot_dir))
        
        # Load any existing alerts
        alerts_file = Path(self.get_config("alert_file", "data/monitoring/alerts.json"))
        if alerts_file.exists():
            try:
                with open(alerts_file, 'r') as f:
                    alerts_data = json.load(f)
                    for alert_data in alerts_data:
                        if alert_data.get("is_active", False):
                            self.logger.info(f"Restoring active alert: {alert_data.get('id')}")
            except Exception as e:
                self.logger.error(f"Error loading alerts: {str(e)}")
        
        # Create system startup alert
        self.alert_manager.create_alert(
            message="Monitoring service started",
            level=AlertLevel.INFO,
            category=AlertCategory.SYSTEM,
            source="monitoring.service",
            expiry=datetime.now() + timedelta(hours=1)
        )
    
    async def _start(self) -> None:
        """Start the monitoring service."""
        self.logger.info("Starting monitoring service")
        
        # Register event handlers
        event_bus.subscribe("SystemStatusEvent", self._handle_system_status_event)
        event_bus.subscribe("OrderEvent", self._handle_order_event)
        event_bus.subscribe("PerformanceEvent", self._handle_performance_event)
        
        # Start background tasks
        self.create_task(self._cleanup_old_data())
        self.create_task(self._save_periodic_snapshots())
        
        self.running = True
        self.snapshot_task = asyncio.create_task(self._periodic_snapshot())
        
        self.logger.info("Monitoring service started")
    
    async def _stop(self) -> None:
        """Stop the monitoring service."""
        self.logger.info("Stopping monitoring service")
        
        # Unregister event handlers
        event_bus.unsubscribe("SystemStatusEvent", self._handle_system_status_event)
        event_bus.unsubscribe("OrderEvent", self._handle_order_event)
        event_bus.unsubscribe("PerformanceEvent", self._handle_performance_event)
        
        # Save final snapshot before shutting down
        await self._save_snapshot("shutdown")
        
        self.running = False
        if self.snapshot_task:
            self.snapshot_task.cancel()
            try:
                await self.snapshot_task
            except asyncio.CancelledError:
                pass
        
        # Create system shutdown alert
        self.alert_manager.create_alert(
            message="Monitoring service stopped",
            level=AlertLevel.INFO,
            category=AlertCategory.SYSTEM,
            source="monitoring.service"
        )
        
        self.logger.info("Monitoring service stopped")
    
    async def _handle_system_status_event(self, event: 'SystemStatusEvent') -> None:
        """Handle system status events from components."""
        component_id = event.component_id
        status = event.status
        
        # Initialize component status if not exists
        if component_id not in self.component_status:
            self.component_status[component_id] = {
                "status": "unknown",
                "last_heartbeat": None,
                "details": {},
                "errors": []
            }
        
        # Update component status
        self.component_status[component_id]["status"] = status
        self.component_status[component_id]["last_heartbeat"] = datetime.now().isoformat()
        
        if hasattr(event, "details") and event.details:
            self.component_status[component_id]["details"].update(event.details)
        
        # Handle errors
        if status == "error":
            error_info = {
                "time": datetime.now().isoformat(),
                "message": event.message if hasattr(event, "message") else "Unknown error",
                "details": event.details if hasattr(event, "details") else {}
            }
            self.component_status[component_id]["errors"].append(error_info)
            self.system_stats["error_count"] += 1
            self.system_stats["last_error_time"] = error_info["time"]
            self.system_stats["last_error_message"] = error_info["message"]
            self.has_critical_errors = True
        
        elif status == "warning":
            self.has_warnings = True
        
        self.logger.debug(f"Updated status for component {component_id}: {status}")
        
        # Update status summary
        self._update_status_summary()
    
    async def _handle_order_event(self, event: 'OrderEvent') -> None:
        """Handle order events."""
        order_id = event.order_id
        event_type = event.event_type
        order = event.order
        
        # Track execution metrics
        if event_type == "SUBMITTED":
            self.performance_metrics["execution_metrics"]["orders_submitted"] += 1
            # Add to active orders
            if order:
                self.active_orders[order_id] = order
        
        elif event_type == "FILLED":
            self.performance_metrics["execution_metrics"]["orders_filled"] += 1
            # Calculate fill time if we have the original order
            if order and order_id in self.active_orders:
                submitted_time = self.active_orders[order_id].submitted_time
                filled_time = order.filled_time
                if submitted_time and filled_time:
                    fill_time_ms = (filled_time - submitted_time).total_seconds() * 1000
                    self.performance_metrics["execution_metrics"]["total_fill_time_ms"] += fill_time_ms
                    total_fills = self.performance_metrics["execution_metrics"]["orders_filled"]
                    total_time = self.performance_metrics["execution_metrics"]["total_fill_time_ms"]
                    self.performance_metrics["execution_metrics"]["avg_fill_time_ms"] = total_time / total_fills
            
            # Add to recent trades
            if order:
                trade_info = {
                    "time": datetime.now().isoformat(),
                    "order_id": order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "type": order.type.value,
                    "price": order.filled_price,
                    "quantity": order.filled_quantity,
                    "exchange": order.exchange
                }
                self.recent_trades.append(trade_info)
                
                # Remove from active orders
                if order_id in self.active_orders:
                    # Add to order history
                    self.order_history.append(self._order_to_dict(order))
                    del self.active_orders[order_id]
        
        elif event_type == "CANCELLED":
            self.performance_metrics["execution_metrics"]["orders_cancelled"] += 1
            # Remove from active orders
            if order_id in self.active_orders:
                # Add to order history
                if order:
                    self.order_history.append(self._order_to_dict(order))
                del self.active_orders[order_id]
        
        elif event_type == "REJECTED":
            self.performance_metrics["execution_metrics"]["orders_rejected"] += 1
            # Remove from active orders
            if order_id in self.active_orders:
                # Add to order history
                if order:
                    self.order_history.append(self._order_to_dict(order))
                del self.active_orders[order_id]
            
        # Limit the size of recent trades list
        max_recent_trades = 100
        if len(self.recent_trades) > max_recent_trades:
            self.recent_trades = self.recent_trades[-max_recent_trades:]
    
    async def _handle_performance_event(self, event: 'PerformanceEvent') -> None:
        """Handle performance metric events."""
        if event.metric_type == "strategy":
            # Update strategy performance
            self.performance_metrics["strategy_performance"][event.strategy_id] = {
                "time": datetime.now().isoformat(),
                "metrics": event.metrics
            }
        
        elif event.metric_type == "portfolio":
            # Update portfolio performance
            self.performance_metrics["portfolio_performance"] = {
                "time": datetime.now().isoformat(),
                "metrics": event.metrics
            }
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old data based on retention policy."""
        while True:
            try:
                # Sleep first to avoid cleaning up data right after startup
                await asyncio.sleep(3600)  # Check every hour
                
                # Calculate cutoff date
                cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
                cutoff_str = cutoff_date.isoformat()
                
                # Clean up order history
                self.order_history = [
                    order for order in self.order_history 
                    if order.get("time", datetime.max.isoformat()) > cutoff_str
                ]
                
                # Clean up error logs in component status
                for component_id in self.component_status:
                    self.component_status[component_id]["errors"] = [
                        error for error in self.component_status[component_id]["errors"]
                        if error.get("time", datetime.max.isoformat()) > cutoff_str
                    ]
                
                # Clean up old snapshot files
                if self.snapshot_dir and self.snapshot_dir.exists():
                    for snapshot_file in self.snapshot_dir.glob("*.json"):
                        file_time_str = snapshot_file.stem.split("_")[0]
                        try:
                            file_time = datetime.fromisoformat(file_time_str)
                            if file_time < cutoff_date:
                                snapshot_file.unlink()
                                self.logger.debug(f"Deleted old snapshot file: {snapshot_file}")
                        except (ValueError, IndexError):
                            self.logger.warning(f"Could not parse date from snapshot filename: {snapshot_file}")
                
                self.logger.info("Cleaned up old monitoring data",
                              cutoff_date=cutoff_str,
                              order_history_size=len(self.order_history))
            
            except Exception as e:
                self.logger.error(f"Error cleaning up old data: {str(e)}")
    
    async def _save_periodic_snapshots(self) -> None:
        """Save periodic snapshots of the monitoring data."""
        while True:
            try:
                # Sleep first to avoid saving snapshot right after startup
                await asyncio.sleep(self.snapshot_interval_seconds)
                
                # Save snapshot
                await self._save_snapshot("periodic")
                
            except Exception as e:
                self.logger.error(f"Error saving periodic snapshot: {str(e)}")
    
    async def _save_snapshot(self, snapshot_type: str) -> None:
        """Save a snapshot of the current monitoring data."""
        # Generate a timestamp for the filename
        timestamp = datetime.now().isoformat().replace(":", "-")
        filename = f"{timestamp}_{snapshot_type}.json"
        filepath = self.snapshot_dir / filename
        
        # Update system stats
        self.system_stats["uptime_seconds"] = (
            datetime.now() - datetime.fromisoformat(self.system_stats["start_time"])
        ).total_seconds()
        
        # Prepare data for snapshot
        snapshot_data = {
            "timestamp": timestamp,
            "type": snapshot_type,
            "system_stats": self.system_stats,
            "component_status": self.component_status,
            "performance_metrics": self.performance_metrics,
            "active_orders_count": len(self.active_orders),
            "order_history_count": len(self.order_history),
            "recent_trades_count": len(self.recent_trades),
            "status_summary": {
                "has_critical_errors": self.has_critical_errors,
                "has_warnings": self.has_warnings
            }
        }
        
        # Save to file
        try:
            with open(filepath, "w") as f:
                json.dump(snapshot_data, f, indent=2)
            
            self.logger.debug(f"Saved monitoring snapshot: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving snapshot to {filepath}: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current system status.
        
        Returns:
            Dictionary with system status information
        """
        # Update uptime
        self.system_stats["uptime_seconds"] = (
            datetime.now() - datetime.fromisoformat(self.system_stats["start_time"])
        ).total_seconds()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_stats": self.system_stats,
            "component_status": self.component_status,
            "status_summary": {
                "has_critical_errors": self.has_critical_errors,
                "has_warnings": self.has_warnings
            }
        }
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get the current active orders.
        
        Returns:
            List of active orders as dictionaries
        """
        return [self._order_to_dict(order) for order in self.active_orders.values()]
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of recent trades
        """
        return self.recent_trades[-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": self.performance_metrics
        }
    
    def _order_to_dict(self, order: Order) -> Dict[str, Any]:
        """Convert an Order object to a dictionary.
        
        Args:
            order: The Order object to convert
            
        Returns:
            Dictionary representation of the order
        """
        return {
            "order_id": order.order_id,
            "time": datetime.now().isoformat(),
            "symbol": order.symbol,
            "exchange": order.exchange,
            "side": order.side.value,
            "type": order.type.value,
            "status": order.status.value,
            "quantity": order.quantity,
            "price": order.price,
            "filled_quantity": order.filled_quantity,
            "filled_price": order.filled_price,
            "submitted_time": order.submitted_time.isoformat() if order.submitted_time else None,
            "filled_time": order.filled_time.isoformat() if order.filled_time else None,
            "cancelled_time": order.cancelled_time.isoformat() if order.cancelled_time else None,
            "strategy_id": order.strategy_id,
            "position_id": order.position_id
        }
    
    def _update_status_summary(self) -> None:
        """Update the system status summary based on component statuses."""
        has_warnings = False
        has_errors = False
        has_critical_errors = False
        
        for component_id, status in self.component_status.items():
            if status["status"] == "warning":
                has_warnings = True
            elif status["status"] == "error":
                has_errors = True
            elif status["status"] == "critical":
                has_critical_errors = True
        
        self.system_stats["status_summary"] = {
            "all_ok": not (has_warnings or has_errors or has_critical_errors),
            "has_warnings": has_warnings,
            "has_errors": has_errors,
            "has_critical_errors": has_critical_errors
        }
    
    async def _periodic_snapshot(self) -> None:
        """
        Periodically save monitoring data snapshots.
        """
        self.logger.info(f"Starting periodic snapshot task (interval: {self.snapshot_interval_seconds} seconds)")
        
        while self.running:
            try:
                # Wait for the next snapshot interval
                await asyncio.sleep(self.snapshot_interval_seconds)
                
                if not self.running:
                    break
                
                # Save snapshot
                await self._save_snapshot("periodic")
                
                # Check for expired alerts
                self.alert_manager.check_expired_alerts()
                
            except asyncio.CancelledError:
                self.logger.info("Periodic snapshot task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in periodic snapshot task: {str(e)}")
                await asyncio.sleep(60)  # Shorter retry interval


# Singleton instance
_MONITORING_SERVICE: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Get the singleton monitoring service instance.
    
    Returns:
        The monitoring service instance
    """
    global _MONITORING_SERVICE
    if _MONITORING_SERVICE is None:
        _MONITORING_SERVICE = MonitoringService()
    return _MONITORING_SERVICE 