"""Alerts dashboard for sentiment analysis system.

This module provides a dashboard for visualizing alerts and system health status,
including active alerts, historical trends, and real-time monitoring.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event
from src.analysis_agents.sentiment.monitoring_alerts import alert_manager, AlertStatus, AlertSeverity, AlertType


class AlertsDashboard:
    """Dashboard for visualizing alerts and system health."""
    
    def __init__(self):
        """Initialize the alerts dashboard."""
        self.logger = get_logger("dashboard", "alerts_dashboard")
        
        # Configuration
        self.refresh_interval = config.get("dashboard.alerts.refresh_interval", 10)  # seconds
        self.max_alerts = config.get("dashboard.alerts.max_alerts", 100)
        self.default_timeframe = config.get("dashboard.alerts.default_timeframe", 24)  # hours
        
        # Dashboard state
        self.last_update = datetime.min
        self.health_status = "unknown"
        self.active_alerts = []
        self.recent_alerts = []
        self.alert_counts = {}
        self.alert_severity_counts = {}
        self.health_checks = []
    
    async def initialize(self) -> None:
        """Initialize the dashboard."""
        self.logger.info("Initializing alerts dashboard")
        
        # Subscribe to events
        event_bus.subscribe("sentiment_health_status", self.handle_health_status)
        event_bus.subscribe("sentiment_alert", self.handle_alert)
    
    def handle_health_status(self, event: Event) -> None:
        """Handle health status events.
        
        Args:
            event: Health status event
        """
        data = event.data
        self.health_status = data.get("status", "unknown")
        self.health_checks = data.get("checks", [])
    
    def handle_alert(self, event: Event) -> None:
        """Handle alert events.
        
        Args:
            event: Alert event
        """
        # Just mark dashboard as needing refresh
        self.last_update = datetime.min
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts.
        
        Returns:
            List of active alerts
        """
        active_alerts = alert_manager.get_active_alerts()
        return [alert.to_dict() for alert in active_alerts]
    
    async def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        recent_alerts = await alert_manager.get_recent_alerts(hours)
        return [alert.to_dict() for alert in recent_alerts]
    
    async def update_data(self) -> None:
        """Update dashboard data."""
        # Check if update is needed
        now = datetime.now()
        if (now - self.last_update).total_seconds() < self.refresh_interval:
            return
        
        self.last_update = now
        
        try:
            # Get active alerts
            self.active_alerts = self.get_active_alerts()
            
            # Get recent alerts
            self.recent_alerts = await self.get_recent_alerts(self.default_timeframe)
            
            # Calculate alert counts
            self.alert_counts = {}
            self.alert_severity_counts = {}
            
            for alert in self.recent_alerts:
                # Count by type
                alert_type = alert["alert_type"]
                if alert_type not in self.alert_counts:
                    self.alert_counts[alert_type] = 0
                self.alert_counts[alert_type] += 1
                
                # Count by severity
                severity = alert["severity"]
                if severity not in self.alert_severity_counts:
                    self.alert_severity_counts[severity] = 0
                self.alert_severity_counts[severity] += 1
            
            self.logger.info(f"Updated alerts dashboard data: {len(self.active_alerts)} active, {len(self.recent_alerts)} recent")
            
        except Exception as e:
            self.logger.error(f"Error updating alerts dashboard data: {str(e)}")
    
    def get_alert_class(self, severity: str) -> str:
        """Get CSS class for alert severity.
        
        Args:
            severity: Alert severity
            
        Returns:
            CSS class
        """
        if severity == "CRITICAL":
            return "critical-alert"
        elif severity == "ERROR":
            return "error-alert"
        elif severity == "WARNING":
            return "warning-alert"
        else:
            return "info-alert"
    
    def get_health_status_class(self, status: str) -> str:
        """Get CSS class for health status.
        
        Args:
            status: Health status
            
        Returns:
            CSS class
        """
        if status == "healthy":
            return "status-healthy"
        elif status == "degraded":
            return "status-degraded"
        elif status == "unhealthy":
            return "status-unhealthy"
        else:
            return "status-unknown"
    
    def get_health_checks_component(self) -> html.Div:
        """Generate health checks component.
        
        Returns:
            Health checks div
        """
        # Group checks by status
        checks_by_status = {"healthy": [], "degraded": [], "unhealthy": [], "unknown": []}
        
        for check in self.health_checks:
            status = check.get("status", "unknown")
            if status not in checks_by_status:
                status = "unknown"
                
            checks_by_status[status].append(check)
        
        # Create components for each status
        components = []
        
        # First show unhealthy checks
        if checks_by_status["unhealthy"]:
            components.append(html.H4("Unhealthy Checks", className="health-status-heading unhealthy"))
            for check in checks_by_status["unhealthy"]:
                components.append(self._create_health_check_card(check))
        
        # Then show degraded checks
        if checks_by_status["degraded"]:
            components.append(html.H4("Degraded Checks", className="health-status-heading degraded"))
            for check in checks_by_status["degraded"]:
                components.append(self._create_health_check_card(check))
        
        # Then show healthy checks
        if checks_by_status["healthy"]:
            components.append(html.H4("Healthy Checks", className="health-status-heading healthy"))
            for check in checks_by_status["healthy"]:
                components.append(self._create_health_check_card(check))
        
        # Finally show unknown checks
        if checks_by_status["unknown"]:
            components.append(html.H4("Unknown Checks", className="health-status-heading unknown"))
            for check in checks_by_status["unknown"]:
                components.append(self._create_health_check_card(check))
        
        return html.Div(components, className="health-checks-container")
    
    def _create_health_check_card(self, check: Dict[str, Any]) -> html.Div:
        """Create a health check card.
        
        Args:
            check: Health check data
            
        Returns:
            Health check card
        """
        name = check.get("name", "Unknown")
        status = check.get("status", "unknown")
        details = check.get("details", {})
        timestamp = check.get("timestamp", "")
        
        # Format timestamp
        try:
            ts = datetime.fromisoformat(timestamp)
            timestamp_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            timestamp_str = timestamp
        
        # Format details
        details_components = []
        for key, value in details.items():
            if key == "error" and value:
                details_components.append(
                    html.Div([
                        html.Span("Error: ", className="detail-label"),
                        html.Span(str(value), className="detail-value error-text")
                    ], className="check-detail")
                )
            else:
                details_components.append(
                    html.Div([
                        html.Span(f"{key}: ", className="detail-label"),
                        html.Span(str(value), className="detail-value")
                    ], className="check-detail")
                )
        
        return html.Div([
            html.Div([
                html.Span(name, className="check-name"),
                html.Span(status.upper(), className=f"check-status {status}")
            ], className="check-header"),
            html.Div(timestamp_str, className="check-timestamp"),
            html.Div(details_components, className="check-details")
        ], className=f"health-check-card {status}")
    
    def create_app(self) -> dash.Dash:
        """Create the Dash application.
        
        Returns:
            Dash application
        """
        # Create Dash app
        app = dash.Dash(
            __name__,
            title="Sentiment Analysis Alerts Dashboard",
            update_title=None,
            suppress_callback_exceptions=True,
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
        )
        
        # App layout
        app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Sentiment Analysis Alerts Dashboard", className="header-title"),
                html.Div([
                    html.Span("System Status: ", className="status-label"),
                    html.Span(id="health-status", className="status-value")
                ], className="health-status"),
                html.Button("Refresh", id="refresh-button", className="refresh-button"),
            ], className="header"),
            
            # Main content
            html.Div([
                # Alert summary
                html.Div([
                    html.Div([
                        html.H3("Active Alerts"),
                        html.Div(id="active-alert-count", className="summary-value")
                    ], className="summary-card"),
                    
                    html.Div([
                        html.H3("Recent Alerts (24h)"),
                        html.Div(id="recent-alert-count", className="summary-value")
                    ], className="summary-card"),
                    
                    html.Div([
                        html.H3("Critical Alerts"),
                        html.Div(id="critical-alert-count", className="summary-value critical")
                    ], className="summary-card"),
                    
                    html.Div([
                        html.H3("Error Alerts"),
                        html.Div(id="error-alert-count", className="summary-value error")
                    ], className="summary-card"),
                ], className="summary-container"),
                
                # Charts
                html.Div([
                    # Alert types chart
                    html.Div([
                        html.H3("Alert Types"),
                        dcc.Graph(id="alert-types-chart")
                    ], className="chart-container"),
                    
                    # Alert timeline chart
                    html.Div([
                        html.H3("Alert Timeline"),
                        dcc.Graph(id="alert-timeline-chart")
                    ], className="chart-container"),
                ], className="charts-row"),
                
                # Tab container
                html.Div([
                    dcc.Tabs(id="tabs", value="active-alerts", children=[
                        dcc.Tab(label="Active Alerts", value="active-alerts"),
                        dcc.Tab(label="Recent Alerts", value="recent-alerts"),
                        dcc.Tab(label="Health Checks", value="health-checks"),
                    ]),
                    html.Div(id="tab-content")
                ], className="tabs-container"),
                
                # Store for alert data
                dcc.Store(id="alert-data-store"),
                
                # Interval for periodic refresh
                dcc.Interval(
                    id="refresh-interval",
                    interval=self.refresh_interval * 1000,  # ms
                    n_intervals=0
                ),
            ], className="main-content"),
            
            # Footer
            html.Div([
                html.P("Sentiment Analysis Alerts Dashboard"),
                html.P(id="last-update-time")
            ], className="footer"),
        ], className="app-container")
        
        # Register callbacks
        self._register_callbacks(app)
        
        return app
    
    def _register_callbacks(self, app: dash.Dash) -> None:
        """Register Dash callbacks.
        
        Args:
            app: Dash application
        """
        # Update data store
        @app.callback(
            Output("alert-data-store", "data"),
            [Input("refresh-interval", "n_intervals"),
             Input("refresh-button", "n_clicks")]
        )
        async def update_data_store(n_intervals, n_clicks):
            self.update_data()
            
            return {
                "active_alerts": self.active_alerts,
                "recent_alerts": self.recent_alerts,
                "health_status": self.health_status,
                "health_checks": self.health_checks,
                "timestamp": datetime.now().isoformat()
            }
        
        # Update health status
        @app.callback(
            Output("health-status", "children"),
            Output("health-status", "className"),
            Input("alert-data-store", "data")
        )
        def update_health_status(data):
            if not data:
                return "Unknown", "status-value status-unknown"
            
            status = data.get("health_status", "unknown").upper()
            status_class = f"status-value {self.get_health_status_class(data.get('health_status', 'unknown'))}"
            
            return status, status_class
        
        # Update summary counts
        @app.callback(
            [Output("active-alert-count", "children"),
             Output("recent-alert-count", "children"),
             Output("critical-alert-count", "children"),
             Output("error-alert-count", "children")],
            Input("alert-data-store", "data")
        )
        def update_summary_counts(data):
            if not data:
                return "0", "0", "0", "0"
            
            active_count = len(data.get("active_alerts", []))
            recent_count = len(data.get("recent_alerts", []))
            
            # Count by severity
            critical_count = sum(1 for alert in data.get("recent_alerts", []) if alert["severity"] = = "CRITICAL")
            error_count = sum(1 for alert in data.get("recent_alerts", []) if alert["severity"] = = "ERROR")
            
            return str(active_count), str(recent_count), str(critical_count), str(error_count)
        
        # Update alert types chart
        @app.callback(
            Output("alert-types-chart", "figure"),
            Input("alert-data-store", "data")
        )
        def update_alert_types_chart(data):
            if not data or not data.get("recent_alerts"):
                # Empty chart
                return go.Figure().update_layout(title="No alerts data available")
            
            # Count alerts by type
            alert_counts = {}
            for alert in data["recent_alerts"]:
                alert_type = alert["alert_type"]
                if alert_type not in alert_counts:
                    alert_counts[alert_type] = 0
                alert_counts[alert_type] += 1
            
            # Convert to pandas DataFrame
            df = pd.DataFrame([
                {"type": alert_type, "count": count}
                for alert_type, count in alert_counts.items()
            ])
            
            if df.empty:
                return go.Figure().update_layout(title="No alerts data available")
            
            # Sort by count
            df = df.sort_values("count", ascending=False)
            
            # Create bar chart
            fig = px.bar(
                df,
                x="type",
                y="count",
                title="Alert Types (last 24 hours)",
                labels={"type": "Alert Type", "count": "Count"},
                color="type"
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Alert Type",
                yaxis_title="Count",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        
        # Update alert timeline chart
        @app.callback(
            Output("alert-timeline-chart", "figure"),
            Input("alert-data-store", "data")
        )
        def update_alert_timeline_chart(data):
            if not data or not data.get("recent_alerts"):
                # Empty chart
                return go.Figure().update_layout(title="No alerts data available")
            
            # Sort alerts by timestamp
            alerts = sorted(
                data["recent_alerts"],
                key=lambda a: datetime.fromisoformat(a["timestamp"])
            )
            
            # Group by hour and severity
            hourly_counts = {}
            try:
                for alert in alerts:
                    timestamp = datetime.fromisoformat(alert["timestamp"])
                    hour = timestamp.replace(minute=0, second=0, microsecond=0)
                    severity = alert["severity"]
                    
                    if hour not in hourly_counts:
                        hourly_counts[hour] = {"CRITICAL": 0, "ERROR": 0, "WARNING": 0, "INFO": 0}
                    
                    hourly_counts[hour][severity] += 1
            except (ValueError, KeyError) as e:
                return go.Figure().update_layout(title=f"Error processing alert data: {str(e)}")
            
            # Sort hours
            hours = sorted(hourly_counts.keys())
            
            if not hours:
                return go.Figure().update_layout(title="No alerts data available")
            
            # Create traces for each severity
            traces = []
            for severity in ["CRITICAL", "ERROR", "WARNING", "INFO"]:
                traces.append(
                    go.Scatter(
                        x=hours,
                        y=[hourly_counts[hour][severity] for hour in hours],
                        mode="lines+markers",
                        name=severity,
                        marker=dict(size=8)
                    )
                )
            
            # Create figure
            fig = go.Figure(data=traces)
            
            # Update layout
            fig.update_layout(
                title="Alert Timeline (by Hour)",
                xaxis_title="Time",
                yaxis_title="Alert Count",
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(x=0, y=1, orientation="h")
            )
            
            return fig
        
        # Update tab content
        @app.callback(
            Output("tab-content", "children"),
            [Input("tabs", "value"),
             Input("alert-data-store", "data")]
        )
        def update_tab_content(tab, data):
            if not data:
                return html.Div("Loading data...", className="loading-message")
            
            if tab == "active-alerts":
                return self._create_active_alerts_tab(data.get("active_alerts", []))
            elif tab == "recent-alerts":
                return self._create_recent_alerts_tab(data.get("recent_alerts", []))
            elif tab == "health-checks":
                return self._create_health_checks_tab(data.get("health_checks", []))
            else:
                return html.Div("Unknown tab")
        
        # Update last update time
        @app.callback(
            Output("last-update-time", "children"),
            Input("alert-data-store", "data")
        )
        def update_last_update_time(data):
            if not data or "timestamp" not in data:
                return "Last update: Never"
            
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
                return f"Last update: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            except (ValueError, TypeError):
                return f"Last update: {data['timestamp']}"
    
    def _create_active_alerts_tab(self, alerts: List[Dict[str, Any]]) -> html.Div:
        """Create active alerts tab content.
        
        Args:
            alerts: List of active alerts
            
        Returns:
            Tab content
        """
        if not alerts:
            return html.Div("No active alerts", className="no-alerts-message")
        
        # Sort alerts by severity and timestamp
        alerts = sorted(
            alerts,
            key=lambda a: (
                ["CRITICAL", "ERROR", "WARNING", "INFO"].index(a["severity"]),
                datetime.fromisoformat(a["timestamp"])
            )
        )
        
        # Create alert cards
        alert_cards = [self._create_alert_card(alert) for alert in alerts]
        
        return html.Div(alert_cards, className="alerts-container")
    
    def _create_recent_alerts_tab(self, alerts: List[Dict[str, Any]]) -> html.Div:
        """Create recent alerts tab content.
        
        Args:
            alerts: List of recent alerts
            
        Returns:
            Tab content
        """
        if not alerts:
            return html.Div("No recent alerts", className="no-alerts-message")
        
        # Sort alerts by timestamp (newest first)
        alerts = sorted(
            alerts,
            key=lambda a: datetime.fromisoformat(a["timestamp"]),
            reverse=True
        )
        
        # Create alert cards
        alert_cards = [self._create_alert_card(alert) for alert in alerts]
        
        return html.Div(alert_cards, className="alerts-container")
    
    def _create_health_checks_tab(self, checks: List[Dict[str, Any]]) -> html.Div:
        """Create health checks tab content.
        
        Args:
            checks: List of health checks
            
        Returns:
            Tab content
        """
        self.health_checks = checks  # Update checks
        return self.get_health_checks_component()
    
    def _create_alert_card(self, alert: Dict[str, Any]) -> html.Div:
        """Create an alert card.
        
        Args:
            alert: Alert data
            
        Returns:
            Alert card
        """
        try:
            # Get alert data
            alert_id = alert.get("id", "")
            severity = alert.get("severity", "INFO")
            source = alert.get("source", "")
            message = alert.get("message", "")
            timestamp = alert.get("timestamp", "")
            status = alert.get("status", "active")
            details = alert.get("details", {})
            related_entities = alert.get("related_entities", [])
            
            # Format timestamp
            try:
                ts = datetime.fromisoformat(timestamp)
                timestamp_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                timestamp_str = timestamp
            
            # Format details
            details_components = []
            for key, value in details.items():
                details_components.append(
                    html.Div([
                        html.Span(f"{key}: ", className="detail-label"),
                        html.Span(str(value), className="detail-value")
                    ], className="alert-detail")
                )
            
            # Create related entities component
            related_component = None
            if related_entities:
                related_component = html.Div([
                    html.Span("Related: ", className="related-label"),
                    html.Span(", ".join(related_entities), className="related-value")
                ], className="alert-related")
            
            # Create status component
            status_component = html.Div(
                status.upper(),
                className=f"alert-status {status}"
            )
            
            return html.Div([
                html.Div([
                    html.Span(severity, className=f"alert-severity {severity.lower()}"),
                    html.Span(source, className="alert-source"),
                    status_component
                ], className="alert-header"),
                html.Div(message, className="alert-message"),
                html.Div([
                    html.Span(timestamp_str, className="alert-timestamp"),
                    related_component
                ], className="alert-metadata"),
                html.Div(details_components, className="alert-details")
            ], className=f"alert-card {self.get_alert_class(severity)}")
            
        except Exception as e:
            # Return error card for this alert
            return html.Div([
                html.Div("Error rendering alert", className="alert-header"),
                html.Div(str(e), className="alert-message error-text")
            ], className="alert-card error-alert")
    
    async def run_server(self, port: int = 8052, debug: bool = False) -> None:
        """Run the dashboard server.
        
        Args:
            port: Port to run the server on
            debug: Whether to run in debug mode
        """
        # Initialize
        self.initialize()
        
        # Create app
        app = self.create_app()
        
        # Run server
        self.logger.info(f"Starting alerts dashboard on port {port}")
        app.run_server(debug=debug, port=port)


async def main():
    """Run the alerts dashboard."""
    # Initialize alert manager
    from src.analysis_agents.sentiment.monitoring_alerts import alert_manager
    alert_manager.initialize()
    
    # Create and run dashboard
    dashboard = AlertsDashboard()
    await dashboard.run_server(port=8052, debug=True)


if __name__ == "__main__":
    asyncio.run(main())