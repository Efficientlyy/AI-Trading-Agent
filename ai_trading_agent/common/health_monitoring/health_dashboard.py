"""
Health Monitoring Dashboard.

This module provides a web-based dashboard for visualizing the health monitoring
data in a user-friendly and interactive way using a modular structure.
"""

import os
import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

# For web dashboard
from flask import Flask, render_template, jsonify, request

# Health monitoring imports
from ai_trading_agent.common.health_monitoring import (
    HealthMonitor,
    HealthStatus,
    AlertSeverity
)

# Dashboard template loader
from ai_trading_agent.common.health_monitoring.dashboard.template_loader import TemplateLoader

# Set up logger
logger = logging.getLogger(__name__)


class HealthDashboard:
    """
    Web-based dashboard for visualizing health monitoring data.
    
    Provides real-time visualization of system health, component status,
    performance metrics, and alerts.
    """
    
    def __init__(
        self,
        health_monitor: HealthMonitor,
        host: str = "127.0.0.1",
        port: int = 5000,
        dashboard_dir: Optional[str] = None,
        update_interval: float = 5.0
    ):
        """
        Initialize the health dashboard.
        
        Args:
            health_monitor: Health monitoring system instance
            host: Host address for the web server
            port: Port for the web server
            dashboard_dir: Directory for dashboard templates and static files
            update_interval: Interval for data updates in seconds
        """
        self.health_monitor = health_monitor
        self.host = host
        self.port = port
        self.update_interval = update_interval
        
        # Set up dashboard directory
        if dashboard_dir:
            self.dashboard_dir = Path(dashboard_dir)
        else:
            # Default to package directory
            current_file = Path(__file__).resolve()
            self.dashboard_dir = current_file.parent / "dashboard"
            
        # Ensure dashboard directory exists
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up templates and static directories
        self.templates_dir = self.dashboard_dir / "templates"
        self.static_dir = self.dashboard_dir / "static"
        
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Flask app
        self.app = Flask(
            __name__,
            template_folder=str(self.templates_dir),
            static_folder=str(self.static_dir)
        )
        
        # Set up data storage
        self.dashboard_data = {}
        self.historical_data = {
            "system_health": [],
            "component_status": {},
            "metrics": {},
            "alerts": []
        }
        self.last_update_time = 0
        
        # Set up routes
        self._setup_routes()
        
        # Thread for running the web server
        self._server_thread = None
        self._running = False
        
        # Thread for updating data
        self._update_thread = None
        
        logger.info(f"Health dashboard initialized on {host}:{port}")
    
    def _setup_routes(self) -> None:
        """Set up Flask routes for the dashboard."""
        # Main dashboard page
        @self.app.route("/")
        def index():
            return render_template("index.html")
        
        # API endpoint for dashboard data
        @self.app.route("/api/dashboard-data")
        def dashboard_data():
            return jsonify(self.dashboard_data)
        
        # API endpoint for historical data
        @self.app.route("/api/historical-data")
        def historical_data():
            return jsonify(self.historical_data)
        
        # API endpoint for component details
        @self.app.route("/api/component/<component_id>")
        def component_details(component_id):
            component_health = self.health_monitor.get_component_health(component_id)
            component_metrics = self.health_monitor.get_metrics(component_id)
            
            if not component_health:
                return jsonify({"error": "Component not found"}), 404
                
            return jsonify({
                "health": component_health,
                "metrics": component_metrics
            })
            
        # API endpoint for starting an agent
        @self.app.route("/api/start-agent/<agent_id>", methods=["POST"])
        def start_agent(agent_id):
            try:
                # Send start command to the agent through health monitor
                success = self.health_monitor.recovery_coordinator.start_component(agent_id)
                if success:
                    return jsonify({"success": True})
                else:
                    return jsonify({"success": False, "error": "Failed to start agent"}), 400
            except Exception as e:
                logger.error(f"Error starting agent {agent_id}: {str(e)}")
                return jsonify({"success": False, "error": str(e)}), 500
        
        # API endpoint for stopping an agent
        @self.app.route("/api/stop-agent/<agent_id>", methods=["POST"])
        def stop_agent(agent_id):
            try:
                # Send stop command to the agent through health monitor
                success = self.health_monitor.recovery_coordinator.stop_component(agent_id)
                if success:
                    return jsonify({"success": True})
                else:
                    return jsonify({"success": False, "error": "Failed to stop agent"}), 400
            except Exception as e:
                logger.error(f"Error stopping agent {agent_id}: {str(e)}")
                return jsonify({"success": False, "error": str(e)}), 500
                
        # API endpoint for starting all agents
        @self.app.route("/api/start-all-agents", methods=["POST"])
        def start_all_agents():
            try:
                # Start all components through health monitor
                success = self.health_monitor.recovery_coordinator.start_all_components()
                if success:
                    return jsonify({"success": True})
                else:
                    return jsonify({"success": False, "error": "Failed to start all agents"}), 400
            except Exception as e:
                logger.error(f"Error starting all agents: {str(e)}")
                return jsonify({"success": False, "error": str(e)}), 500
                
        # API endpoint for stopping all agents
        @self.app.route("/api/stop-all-agents", methods=["POST"])
        def stop_all_agents():
            try:
                # Stop all components through health monitor
                success = self.health_monitor.recovery_coordinator.stop_all_components()
                if success:
                    return jsonify({"success": True})
                else:
                    return jsonify({"success": False, "error": "Failed to stop all agents"}), 400
            except Exception as e:
                logger.error(f"Error stopping all agents: {str(e)}")
                return jsonify({"success": False, "error": str(e)}), 500
                
        # API endpoint for toggling auto recovery
        @self.app.route("/api/toggle-auto-recovery", methods=["POST"])
        def toggle_auto_recovery():
            try:
                data = request.get_json()
                enabled = data.get("enabled", False)
                
                self.health_monitor.recovery_coordinator.set_auto_recovery(enabled)
                return jsonify({"success": True})
            except Exception as e:
                logger.error(f"Error toggling auto recovery: {str(e)}")
                return jsonify({"success": False, "error": str(e)}), 500
                
        # API endpoint for alert details
        @self.app.route("/api/alert-details/<alert_id>")
        def alert_details(alert_id):
            alert = self.health_monitor.alert_manager.get_alert(alert_id)
            if not alert:
                return jsonify({"error": "Alert not found"}), 404
                
            return jsonify({"alert": alert})
        
        # API endpoint for acknowledging alerts
        @self.app.route("/api/alerts/acknowledge", methods=["POST"])
        def acknowledge_alert():
            data = request.json
            alert_id = data.get("alert_id")
            
            if not alert_id:
                return jsonify({"success": False, "error": "Missing alert_id"})
                
            result = self.health_monitor.alert_manager.acknowledge_alert(alert_id)
            return jsonify({"success": result})
        
        # API endpoint for resolving alerts
        @self.app.route("/api/alerts/resolve", methods=["POST"])
        def resolve_alert():
            data = request.json
            alert_id = data.get("alert_id")
            
            if not alert_id:
                return jsonify({"success": False, "error": "Missing alert_id"})
                
            result = self.health_monitor.alert_manager.resolve_alert(alert_id)
            return jsonify({"success": result})
    
    def _update_dashboard_data(self) -> None:
        """Update dashboard data from health monitor."""
        # Get system health
        system_health = self.health_monitor.get_system_health()
        
        # Get component health
        component_health = self.health_monitor.get_component_health()
        
        # Get active alerts
        active_alerts = self.health_monitor.get_active_alerts()
        
        # Get all metrics
        metrics = self.health_monitor.get_metrics()
        
        # Get recovery history
        recovery_history = self.health_monitor.get_recovery_history(limit=10)
        
        # Update dashboard data
        self.dashboard_data = {
            "system_health": system_health,
            "component_health": component_health,
            "active_alerts": [alert.to_dict() for alert in active_alerts],
            "metrics": metrics,
            "recovery_history": recovery_history,
            "last_update": time.time()
        }
        
        # Update historical data
        timestamp = datetime.now().isoformat()
        
        # Store system health history
        self.historical_data["system_health"].append({
            "timestamp": timestamp,
            "overall_status": system_health["overall_status"],
            "component_count": system_health["component_count"],
            "status_counts": system_health["status_counts"]
        })
        
        # Keep reasonable history size
        if len(self.historical_data["system_health"]) > 1000:
            self.historical_data["system_health"] = self.historical_data["system_health"][-1000:]
        
        # Store component status history
        for component_id, component in component_health.items():
            if component_id not in self.historical_data["component_status"]:
                self.historical_data["component_status"][component_id] = []
                
            self.historical_data["component_status"][component_id].append({
                "timestamp": timestamp,
                "status": component["status"]
            })
            
            # Keep reasonable history size
            if len(self.historical_data["component_status"][component_id]) > 1000:
                self.historical_data["component_status"][component_id] = (
                    self.historical_data["component_status"][component_id][-1000:]
                )
        
        # Store metrics history
        for component_id, comp_metrics in metrics.items():
            if component_id not in self.historical_data["metrics"]:
                self.historical_data["metrics"][component_id] = {}
                
            for metric_name, metric_data in comp_metrics.items():
                metric_key = f"{component_id}_{metric_name}"
                
                if metric_key not in self.historical_data["metrics"][component_id]:
                    self.historical_data["metrics"][component_id][metric_key] = []
                    
                self.historical_data["metrics"][component_id][metric_key].append({
                    "timestamp": timestamp,
                    "value": metric_data["last_value"]
                })
                
                # Keep reasonable history size
                if len(self.historical_data["metrics"][component_id][metric_key]) > 1000:
                    self.historical_data["metrics"][component_id][metric_key] = (
                        self.historical_data["metrics"][component_id][metric_key][-1000:]
                    )
        
        # Store alerts history
        for alert in active_alerts:
            if alert.alert_id not in [a["alert_id"] for a in self.historical_data["alerts"]]:
                self.historical_data["alerts"].append({
                    "alert_id": alert.alert_id,
                    "component_id": alert.component_id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved
                })
        
        # Keep reasonable alerts history size
        if len(self.historical_data["alerts"]) > 1000:
            self.historical_data["alerts"] = self.historical_data["alerts"][-1000:]
            
        self.last_update_time = time.time()
    
    def _update_loop(self) -> None:
        """Background thread for updating dashboard data."""
        logger.info("Dashboard update thread started")
        
        while self._running:
            try:
                self._update_dashboard_data()
            except Exception as e:
                logger.error(f"Error updating dashboard data: {str(e)}")
                
            # Sleep until next update
            time.sleep(self.update_interval)
            
        logger.info("Dashboard update thread stopped")
    
    def generate_default_templates(self) -> None:
        """Generate default dashboard templates if they don't exist."""
        logger.info("Loading dashboard templates using template loader")
        
        # Use the template loader to handle template management
        template_loader = TemplateLoader(str(self.dashboard_dir))
        
        # Ensure all required templates exist
        template_loader.ensure_all_templates_exist()
    
    def start(self) -> None:
        """Start the dashboard web server and update thread."""
        if self._running:
            logger.warning("Health dashboard already running")
            return
            
        # Generate default templates
        self.generate_default_templates()
            
        # Start update thread
        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_loop,
            name="DashboardUpdate",
            daemon=True
        )
        self._update_thread.start()
        
        # Start web server
        self._server_thread = threading.Thread(
            target=self._run_server,
            name="DashboardServer",
            daemon=True
        )
        self._server_thread.start()
        
        logger.info(f"Health dashboard started at http://{self.host}:{self.port}")
        
        # Print URL for user
        print(f"\nHealth Dashboard available at: http://{self.host}:{self.port}\n")
    
    def _run_server(self) -> None:
        """Run the Flask web server."""
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    
    def stop(self) -> None:
        """Stop the dashboard web server and update thread."""
        if not self._running:
            logger.warning("Health dashboard already stopped")
            return
            
        # Stop update thread
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5.0)
            self._update_thread = None
            
        # Note: We don't stop the Flask server here as it's not easily stoppable
        # in a clean way from another thread
        
        logger.info("Health dashboard stopped")


def run_standalone_dashboard(
    health_monitor: Optional[HealthMonitor] = None,
    host: str = "127.0.0.1",
    port: int = 5000
) -> HealthDashboard:
    """
    Run a standalone health dashboard application.
    
    Args:
        health_monitor: Health monitor instance, creates one if not provided
        host: Host address for the web server
        port: Port for the web server
        
    Returns:
        The HealthDashboard instance
    """
    if health_monitor is None:
        health_monitor = HealthMonitor()
        health_monitor.start()
        
    dashboard = HealthDashboard(
        health_monitor=health_monitor,
        host=host,
        port=port
    )
    
    dashboard.start()
    return dashboard


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run standalone dashboard
    dashboard = run_standalone_dashboard()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard")
        dashboard.stop()
