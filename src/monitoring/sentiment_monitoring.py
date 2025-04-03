"""Documentation and monitoring system for sentiment analysis.

This module provides comprehensive documentation generation and monitoring capabilities
for the sentiment analysis system, including metrics collection, alerting, and visualization.
"""

import os
import json
import logging
import time
import threading
import queue
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set, Callable, Union
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sentiment_monitoring")


class MetricsCollector:
    """Collects and stores metrics for monitoring."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of data points to keep per metric
        """
        self.max_history = max_history
        self.metrics: Dict[str, deque] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.last_update = datetime.utcnow()
        self._lock = threading.Lock()
        
    def register_metric(self, 
                       name: str, 
                       description: str, 
                       unit: str = "",
                       metric_type: str = "gauge"):
        """Register a new metric.
        
        Args:
            name: Metric name
            description: Metric description
            unit: Unit of measurement
            metric_type: Type of metric (gauge, counter, histogram)
        """
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.max_history)
                self.metadata[name] = {
                    "description": description,
                    "unit": unit,
                    "type": metric_type,
                    "created_at": datetime.utcnow()
                }
                logger.info(f"Registered metric: {name}")
    
    def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        with self._lock:
            # Register metric if it doesn't exist
            if name not in self.metrics:
                self.register_metric(name, f"Auto-registered metric: {name}")
                
            # Record value
            self.metrics[name].append((timestamp, value))
            self.last_update = datetime.utcnow()
    
    def increment_counter(self, name: str, increment: float = 1.0):
        """Increment a counter metric.
        
        Args:
            name: Metric name
            increment: Amount to increment by
        """
        with self._lock:
            # Register metric if it doesn't exist
            if name not in self.metrics:
                self.register_metric(name, f"Auto-registered counter: {name}", metric_type="counter")
                self.metrics[name].append((datetime.utcnow(), 0.0))
                
            # Get last value
            if self.metrics[name]:
                last_timestamp, last_value = self.metrics[name][-1]
                new_value = last_value + increment
            else:
                new_value = increment
                
            # Record new value
            self.record_metric(name, new_value)
    
    def record_histogram(self, name: str, value: float):
        """Record a value for a histogram metric.
        
        Args:
            name: Metric name
            value: Value to record
        """
        with self._lock:
            # Register metric if it doesn't exist
            if name not in self.metrics:
                self.register_metric(name, f"Auto-registered histogram: {name}", metric_type="histogram")
                
            # Record value
            self.record_metric(name, value)
    
    def get_metric_values(self, 
                         name: str, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Tuple[datetime, float]]:
        """Get values for a metric.
        
        Args:
            name: Metric name
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of (timestamp, value) tuples
        """
        with self._lock:
            if name not in self.metrics:
                return []
                
            values = list(self.metrics[name])
            
            # Apply time filters
            if start_time or end_time:
                filtered_values = []
                for timestamp, value in values:
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                    filtered_values.append((timestamp, value))
                return filtered_values
            else:
                return values
    
    def get_metric_statistics(self, 
                            name: str,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get statistics for a metric.
        
        Args:
            name: Metric name
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary of statistics
        """
        values = self.get_metric_values(name, start_time, end_time)
        
        if not values:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "last": None,
                "first": None
            }
            
        # Extract just the values
        numeric_values = [v for _, v in values]
        
        return {
            "count": len(numeric_values),
            "min": float(np.min(numeric_values)),
            "max": float(np.max(numeric_values)),
            "mean": float(np.mean(numeric_values)),
            "median": float(np.median(numeric_values)),
            "std": float(np.std(numeric_values)) if len(numeric_values) > 1 else 0.0,
            "last": numeric_values[-1],
            "first": numeric_values[0]
        }
    
    def get_all_metrics(self) -> List[str]:
        """Get list of all registered metrics.
        
        Returns:
            List of metric names
        """
        with self._lock:
            return list(self.metrics.keys())
    
    def get_metric_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Metric metadata
        """
        with self._lock:
            if name not in self.metadata:
                return {}
            return self.metadata[name].copy()
    
    def export_metrics(self, 
                      output_file: Optional[str] = None,
                      format: str = "json") -> str:
        """Export metrics to file.
        
        Args:
            output_file: Optional output file path
            format: Export format (json or csv)
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = f"metrics_export_{timestamp}.{format}"
            
        with self._lock:
            if format == "json":
                # Export to JSON
                export_data = {
                    "metrics": {},
                    "metadata": self.metadata,
                    "export_time": datetime.utcnow().isoformat()
                }
                
                for name, values in self.metrics.items():
                    export_data["metrics"][name] = [
                        {"timestamp": ts.isoformat(), "value": val}
                        for ts, val in values
                    ]
                    
                with open(output_file, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif format == "csv":
                # Export to CSV
                all_data = []
                
                for name, values in self.metrics.items():
                    for timestamp, value in values:
                        all_data.append({
                            "metric_name": name,
                            "timestamp": timestamp.isoformat(),
                            "value": value
                        })
                        
                if all_data:
                    df = pd.DataFrame(all_data)
                    df.to_csv(output_file, index=False)
                else:
                    # Create empty CSV
                    with open(output_file, 'w') as f:
                        f.write("metric_name,timestamp,value\n")
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        return output_file


class AlertManager:
    """Manages alerts based on metric thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.
        
        Args:
            metrics_collector: MetricsCollector instance
        """
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self._lock = threading.Lock()
        
    def add_alert(self, 
                 name: str,
                 metric_name: str,
                 condition: str,
                 threshold: float,
                 description: str = "",
                 severity: str = "warning"):
        """Add a new alert.
        
        Args:
            name: Alert name
            metric_name: Metric to monitor
            condition: Condition (>, <, >=, <=, ==, !=)
            threshold: Threshold value
            description: Alert description
            severity: Alert severity (info, warning, error, critical)
        """
        with self._lock:
            self.alerts[name] = {
                "metric_name": metric_name,
                "condition": condition,
                "threshold": threshold,
                "description": description,
                "severity": severity,
                "created_at": datetime.utcnow(),
                "last_triggered": None,
                "active": True
            }
            logger.info(f"Added alert: {name}")
    
    def remove_alert(self, name: str):
        """Remove an alert.
        
        Args:
            name: Alert name
        """
        with self._lock:
            if name in self.alerts:
                del self.alerts[name]
                logger.info(f"Removed alert: {name}")
    
    def enable_alert(self, name: str):
        """Enable an alert.
        
        Args:
            name: Alert name
        """
        with self._lock:
            if name in self.alerts:
                self.alerts[name]["active"] = True
                logger.info(f"Enabled alert: {name}")
    
    def disable_alert(self, name: str):
        """Disable an alert.
        
        Args:
            name: Alert name
        """
        with self._lock:
            if name in self.alerts:
                self.alerts[name]["active"] = False
                logger.info(f"Disabled alert: {name}")
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all alerts against current metric values.
        
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        with self._lock:
            for name, alert in self.alerts.items():
                if not alert["active"]:
                    continue
                    
                # Get latest metric value
                metric_values = self.metrics_collector.get_metric_values(alert["metric_name"])
                if not metric_values:
                    continue
                    
                latest_timestamp, latest_value = metric_values[-1]
                
                # Check condition
                triggered = False
                
                if alert["condition"] = = ">":
                    triggered = latest_value > alert["threshold"]
                elif alert["condition"] = = "<":
                    triggered = latest_value < alert["threshold"]
                elif alert["condition"] = = ">=":
                    triggered = latest_value >= alert["threshold"]
                elif alert["condition"] = = "<=":
                    triggered = latest_value <= alert["threshold"]
                elif alert["condition"] = = "==":
                    triggered = latest_value == alert["threshold"]
                elif alert["condition"] = = "!=":
                    triggered = latest_value != alert["threshold"]
                    
                if triggered:
                    # Update last triggered time
                    alert["last_triggered"] = datetime.utcnow()
                    
                    # Create alert event
                    alert_event = {
                        "name": name,
                        "metric_name": alert["metric_name"],
                        "condition": alert["condition"],
                        "threshold": alert["threshold"],
                        "actual_value": latest_value,
                        "description": alert["description"],
                        "severity": alert["severity"],
                        "timestamp": datetime.utcnow()
                    }
                    
                    # Add to history
                    self.alert_history.append(alert_event)
                    
                    # Trim history if needed
                    if len(self.alert_history) > self.max_history:
                        self.alert_history = self.alert_history[-self.max_history:]
                        
                    # Add to triggered alerts
                    triggered_alerts.append(alert_event)
                    
                    logger.warning(f"Alert triggered: {name} - {alert['description']}")
                    
        return triggered_alerts
    
    def get_alert_history(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alert history.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            severity: Optional severity filter
            
        Returns:
            List of alert events
        """
        with self._lock:
            # Apply filters
            filtered_history = []
            
            for event in self.alert_history:
                # Apply time filters
                if start_time and event["timestamp"] < start_time:
                    continue
                if end_time and event["timestamp"] > end_time:
                    continue
                    
                # Apply severity filter
                if severity and event["severity"] != severity:
                    continue
                    
                filtered_history.append(event)
                
            return filtered_history
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts.
        
        Returns:
            List of active alerts
        """
        with self._lock:
            return [
                {**alert, "name": name}
                for name, alert in self.alerts.items()
                if alert["active"]
            ]


class DocumentationGenerator:
    """Generates documentation for the sentiment analysis system."""
    
    def __init__(self, output_dir: str = "/tmp/sentiment_documentation"):
        """Initialize documentation generator.
        
        Args:
            output_dir: Output directory for documentation
        """
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_component_documentation(self, 
                                       component_name: str,
                                       description: str,
                                       usage_examples: List[str],
                                       configuration_options: Dict[str, Dict[str, Any]],
                                       dependencies: List[str] = None,
                                       output_format: str = "markdown") -> str:
        """Generate documentation for a component.
        
        Args:
            component_name: Name of the component
            description: Component description
            usage_examples: List of usage examples
            configuration_options: Dictionary of configuration options
            dependencies: List of dependencies
            output_format: Output format (markdown or html)
            
        Returns:
            Path to generated documentation file
        """
        if dependencies is None:
            dependencies = []
            
        # Create filename
        filename = f"{component_name.lower().replace(' ', '_')}"
        
        if output_format == "markdown":
            filename += ".md"
            content = self._generate_markdown_documentation(
                component_name, description, usage_examples,
                configuration_options, dependencies
            )
        elif output_format == "html":
            filename += ".html"
            content = self._generate_html_documentation(
                component_name, description, usage_examples,
                configuration_options, dependencies
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        # Write to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(content)
            
        return output_path
    
    def _generate_markdown_documentation(self,
                                       component_name: str,
                                       description: str,
                                       usage_examples: List[str],
                                       configuration_options: Dict[str, Dict[str, Any]],
                                       dependencies: List[str]) -> str:
        """Generate markdown documentation.
        
        Args:
            component_name: Name of the component
            description: Component description
            usage_examples: List of usage examples
            configuration_options: Dictionary of configuration options
            dependencies: List of dependencies
            
        Returns:
            Markdown content
        """
        content = f"# {component_name}\n\n"
        
        # Add description
        content += f"{description}\n\n"
        
        # Add dependencies
        if dependencies:
            content += "## Dependencies\n\n"
            for dep in dependencies:
                content += f"- {dep}\n"
            content += "\n"
            
        # Add configuration options
        if configuration_options:
            content += "## Configuration Options\n\n"
            content += "| Option | Type | Default | Description |\n"
            content += "| ------ | ---- | ------- | ----------- |\n"
            
            for option, details in configuration_options.items():
                option_type = details.get("type", "")
                default = details.get("default", "")
                description = details.get("description", "")
                
                content += f"| {option} | {option_type} | {default} | {description} |\n"
                
            content += "\n"
            
        # Add usage examples
        if usage_examples:
            content += "## Usage Examples\n\n"
            
            for i, example in enumerate(usage_examples):
                content += f"### Example {i+1}\n\n"
                content += "```python\n"
                content += f"{example}\n"
                content += "```\n\n"
                
        return content
    
    def _generate_html_documentation(self,
                                   component_name: str,
                                   description: str,
                                   usage_examples: List[str],
                                   configuration_options: Dict[str, Dict[str, Any]],
                                   dependencies: List[str]) -> str:
        """Generate HTML documentation.
        
        Args:
            component_name: Name of the component
            description: Component description
            usage_examples: List of usage examples
            configuration_options: Dictionary of configuration options
            dependencies: List of dependencies
            
        Returns:
            HTML content
        """
        content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{component_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{component_name}</h1>
            
            <p>{description}</p>
        """
        
        # Add dependencies
        if dependencies:
            content += "<h2>Dependencies</h2>\n<ul>\n"
            for dep in dependencies:
                content += f"<li>{dep}</li>\n"
            content += "</ul>\n"
            
        # Add configuration options
        if configuration_options:
            content += """
            <h2>Configuration Options</h2>
            <table>
                <tr>
                    <th>Option</th>
                    <th>Type</th>
                    <th>Default</th>
                    <th>Description</th>
                </tr>
            """
            
            for option, details in configuration_options.items():
                option_type = details.get("type", "")
                default = details.get("default", "")
                description = details.get("description", "")
                
                content += f"""
                <tr>
                    <td>{option}</td>
                    <td>{option_type}</td>
                    <td>{default}</td>
                    <td>{description}</td>
                </tr>
                """
                
            content += "</table>\n"
            
        # Add usage examples
        if usage_examples:
            content += "<h2>Usage Examples</h2>\n"
            
            for i, example in enumerate(usage_examples):
                content += f"<h3>Example {i+1}</h3>\n"
                content += f"<pre><code>{example}</code></pre>\n"
                
        content += """
        </body>
        </html>
        """
        
        return content
    
    def generate_system_documentation(self,
                                    title: str,
                                    overview: str,
                                    components: List[Dict[str, Any]],
                                    architecture_diagram: Optional[str] = None,
                                    output_format: str = "markdown") -> str:
        """Generate system-level documentation.
        
        Args:
            title: Documentation title
            overview: System overview
            components: List of component descriptions
            architecture_diagram: Optional path to architecture diagram
            output_format: Output format (markdown or html)
            
        Returns:
            Path to generated documentation file
        """
        # Create filename
        filename = "system_documentation"
        
        if output_format == "markdown":
            filename += ".md"
            content = self._generate_markdown_system_doc(
                title, overview, components, architecture_diagram
            )
        elif output_format == "html":
            filename += ".html"
            content = self._generate_html_system_doc(
                title, overview, components, architecture_diagram
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        # Write to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(content)
            
        return output_path
    
    def _generate_markdown_system_doc(self,
                                    title: str,
                                    overview: str,
                                    components: List[Dict[str, Any]],
                                    architecture_diagram: Optional[str]) -> str:
        """Generate markdown system documentation.
        
        Args:
            title: Documentation title
            overview: System overview
            components: List of component descriptions
            architecture_diagram: Optional path to architecture diagram
            
        Returns:
            Markdown content
        """
        content = f"# {title}\n\n"
        
        # Add overview
        content += f"{overview}\n\n"
        
        # Add architecture diagram
        if architecture_diagram:
            content += "## System Architecture\n\n"
            content += f"![System Architecture]({architecture_diagram})\n\n"
            
        # Add components
        if components:
            content += "## Components\n\n"
            
            for component in components:
                name = component.get("name", "")
                description = component.get("description", "")
                responsibility = component.get("responsibility", "")
                
                content += f"### {name}\n\n"
                content += f"{description}\n\n"
                content += f"**Responsibility**: {responsibility}\n\n"
                
                # Add interfaces if available
                interfaces = component.get("interfaces", [])
                if interfaces:
                    content += "#### Interfaces\n\n"
                    for interface in interfaces:
                        content += f"- {interface}\n"
                    content += "\n"
                    
        return content
    
    def _generate_html_system_doc(self,
                                title: str,
                                overview: str,
                                components: List[Dict[str, Any]],
                                architecture_diagram: Optional[str]) -> str:
        """Generate HTML system documentation.
        
        Args:
            title: Documentation title
            overview: System overview
            components: List of component descriptions
            architecture_diagram: Optional path to architecture diagram
            
        Returns:
            HTML content
        """
        content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                h3 {{ color: #999; }}
                .component {{ background-color: #f9f9f9; padding: 10px; margin-bottom: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            
            <p>{overview}</p>
        """
        
        # Add architecture diagram
        if architecture_diagram:
            content += f"""
            <h2>System Architecture</h2>
            <img src="{architecture_diagram}" alt="System Architecture">
            """
            
        # Add components
        if components:
            content += "<h2>Components</h2>\n"
            
            for component in components:
                name = component.get("name", "")
                description = component.get("description", "")
                responsibility = component.get("responsibility", "")
                
                content += f"""
                <div class="component">
                    <h3>{name}</h3>
                    <p>{description}</p>
                    <p><strong>Responsibility</strong>: {responsibility}</p>
                """
                
                # Add interfaces if available
                interfaces = component.get("interfaces", [])
                if interfaces:
                    content += "<h4>Interfaces</h4>\n<ul>\n"
                    for interface in interfaces:
                        content += f"<li>{interface}</li>\n"
                    content += "</ul>\n"
                    
                content += "</div>\n"
                
        content += """
        </body>
        </html>
        """
        
        return content
    
    def generate_api_documentation(self,
                                 api_name: str,
                                 description: str,
                                 endpoints: List[Dict[str, Any]],
                                 output_format: str = "markdown") -> str:
        """Generate API documentation.
        
        Args:
            api_name: Name of the API
            description: API description
            endpoints: List of endpoint descriptions
            output_format: Output format (markdown or html)
            
        Returns:
            Path to generated documentation file
        """
        # Create filename
        filename = f"{api_name.lower().replace(' ', '_')}_api"
        
        if output_format == "markdown":
            filename += ".md"
            content = self._generate_markdown_api_doc(
                api_name, description, endpoints
            )
        elif output_format == "html":
            filename += ".html"
            content = self._generate_html_api_doc(
                api_name, description, endpoints
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        # Write to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(content)
            
        return output_path
    
    def _generate_markdown_api_doc(self,
                                 api_name: str,
                                 description: str,
                                 endpoints: List[Dict[str, Any]]) -> str:
        """Generate markdown API documentation.
        
        Args:
            api_name: Name of the API
            description: API description
            endpoints: List of endpoint descriptions
            
        Returns:
            Markdown content
        """
        content = f"# {api_name} API\n\n"
        
        # Add description
        content += f"{description}\n\n"
        
        # Add endpoints
        if endpoints:
            content += "## Endpoints\n\n"
            
            for endpoint in endpoints:
                path = endpoint.get("path", "")
                method = endpoint.get("method", "GET")
                description = endpoint.get("description", "")
                
                content += f"### {method} {path}\n\n"
                content += f"{description}\n\n"
                
                # Add parameters if available
                parameters = endpoint.get("parameters", [])
                if parameters:
                    content += "#### Parameters\n\n"
                    content += "| Name | Type | Required | Description |\n"
                    content += "| ---- | ---- | -------- | ----------- |\n"
                    
                    for param in parameters:
                        name = param.get("name", "")
                        param_type = param.get("type", "")
                        required = "Yes" if param.get("required", False) else "No"
                        param_description = param.get("description", "")
                        
                        content += f"| {name} | {param_type} | {required} | {param_description} |\n"
                        
                    content += "\n"
                    
                # Add response if available
                response = endpoint.get("response", {})
                if response:
                    content += "#### Response\n\n"
                    
                    # Add response schema
                    schema = response.get("schema", {})
                    if schema:
                        content += "```json\n"
                        content += json.dumps(schema, indent=2)
                        content += "\n```\n\n"
                        
                    # Add example
                    example = response.get("example", {})
                    if example:
                        content += "Example:\n\n"
                        content += "```json\n"
                        content += json.dumps(example, indent=2)
                        content += "\n```\n\n"
                        
        return content
    
    def _generate_html_api_doc(self,
                             api_name: str,
                             description: str,
                             endpoints: List[Dict[str, Any]]) -> str:
        """Generate HTML API documentation.
        
        Args:
            api_name: Name of the API
            description: API description
            endpoints: List of endpoint descriptions
            
        Returns:
            HTML content
        """
        content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{api_name} API</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                h3 {{ color: #999; }}
                .endpoint {{ background-color: #f9f9f9; padding: 10px; margin-bottom: 20px; border-radius: 5px; }}
                .method {{ display: inline-block; padding: 5px; border-radius: 3px; color: white; }}
                .get {{ background-color: #61affe; }}
                .post {{ background-color: #49cc90; }}
                .put {{ background-color: #fca130; }}
                .delete {{ background-color: #f93e3e; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{api_name} API</h1>
            
            <p>{description}</p>
        """
        
        # Add endpoints
        if endpoints:
            content += "<h2>Endpoints</h2>\n"
            
            for endpoint in endpoints:
                path = endpoint.get("path", "")
                method = endpoint.get("method", "GET").upper()
                description = endpoint.get("description", "")
                
                method_class = method.lower()
                
                content += f"""
                <div class="endpoint">
                    <h3>
                        <span class="method {method_class}">{method}</span>
                        {path}
                    </h3>
                    <p>{description}</p>
                """
                
                # Add parameters if available
                parameters = endpoint.get("parameters", [])
                if parameters:
                    content += """
                    <h4>Parameters</h4>
                    <table>
                        <tr>
                            <th>Name</th>
                            <th>Type</th>
                            <th>Required</th>
                            <th>Description</th>
                        </tr>
                    """
                    
                    for param in parameters:
                        name = param.get("name", "")
                        param_type = param.get("type", "")
                        required = "Yes" if param.get("required", False) else "No"
                        param_description = param.get("description", "")
                        
                        content += f"""
                        <tr>
                            <td>{name}</td>
                            <td>{param_type}</td>
                            <td>{required}</td>
                            <td>{param_description}</td>
                        </tr>
                        """
                        
                    content += "</table>\n"
                    
                # Add response if available
                response = endpoint.get("response", {})
                if response:
                    content += "<h4>Response</h4>\n"
                    
                    # Add response schema
                    schema = response.get("schema", {})
                    if schema:
                        content += "<pre><code>"
                        content += json.dumps(schema, indent=2)
                        content += "</code></pre>\n"
                        
                    # Add example
                    example = response.get("example", {})
                    if example:
                        content += "<p>Example:</p>\n"
                        content += "<pre><code>"
                        content += json.dumps(example, indent=2)
                        content += "</code></pre>\n"
                        
                content += "</div>\n"
                
        content += """
        </body>
        </html>
        """
        
        return content


class SentimentMonitoringDashboard:
    """Dashboard for monitoring sentiment analysis system."""
    
    def __init__(self, 
                metrics_collector: MetricsCollector,
                alert_manager: AlertManager,
                output_dir: str = "/tmp/sentiment_dashboard"):
        """Initialize monitoring dashboard.
        
        Args:
            metrics_collector: MetricsCollector instance
            alert_manager: AlertManager instance
            output_dir: Output directory for dashboard files
        """
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_dashboard(self, 
                         title: str = "Sentiment Analysis Monitoring",
                         metrics_to_display: Optional[List[str]] = None) -> str:
        """Generate monitoring dashboard.
        
        Args:
            title: Dashboard title
            metrics_to_display: Optional list of metrics to display
            
        Returns:
            Path to generated dashboard file
        """
        # Get metrics to display
        if metrics_to_display is None:
            metrics_to_display = self.metrics_collector.get_all_metrics()
            
        # Create dashboard HTML
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"dashboard_{timestamp}.html")
        
        # Generate metric charts
        chart_files = []
        for metric in metrics_to_display:
            chart_file = self._generate_metric_chart(metric)
            if chart_file:
                chart_files.append((metric, chart_file))
                
        # Get recent alerts
        recent_alerts = self.alert_manager.get_alert_history(
            start_time=datetime.utcnow() - timedelta(days=1)
        )
        
        # Generate HTML
        with open(output_file, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; }}
                    .dashboard {{ display: flex; flex-wrap: wrap; }}
                    .metric {{ width: 45%; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
                    .alerts {{ margin-top: 20px; }}
                    .alert {{ padding: 10px; margin-bottom: 10px; border-radius: 5px; }}
                    .info {{ background-color: #d1ecf1; }}
                    .warning {{ background-color: #fff3cd; }}
                    .error {{ background-color: #f8d7da; }}
                    .critical {{ background-color: #dc3545; color: white; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <p>Generated at: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>Metrics</h2>
                <div class="dashboard">
            """)
            
            # Add metric charts
            for metric, chart_file in chart_files:
                # Get relative path
                rel_path = os.path.relpath(chart_file, self.output_dir)
                
                # Get metadata
                metadata = self.metrics_collector.get_metric_metadata(metric)
                description = metadata.get("description", "")
                
                # Get statistics
                stats = self.metrics_collector.get_metric_statistics(metric)
                
                f.write(f"""
                <div class="metric">
                    <h3>{metric}</h3>
                    <p>{description}</p>
                    <img src="{rel_path}" alt="{metric} chart" width="100%">
                    <p>
                        Last value: {stats.get("last")}<br>
                        Mean: {stats.get("mean")}<br>
                        Min: {stats.get("min")}<br>
                        Max: {stats.get("max")}
                    </p>
                </div>
                """)
                
            f.write("</div>\n")
            
            # Add alerts
            f.write("""
            <h2>Recent Alerts</h2>
            <div class="alerts">
            """)
            
            if recent_alerts:
                for alert in recent_alerts:
                    severity = alert.get("severity", "info")
                    name = alert.get("name", "")
                    description = alert.get("description", "")
                    timestamp = alert.get("timestamp", datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S")
                    metric_name = alert.get("metric_name", "")
                    threshold = alert.get("threshold", "")
                    actual_value = alert.get("actual_value", "")
                    
                    f.write(f"""
                    <div class="alert {severity}">
                        <strong>{name}</strong> - {timestamp}<br>
                        {description}<br>
                        Metric: {metric_name}, Threshold: {threshold}, Actual: {actual_value}
                    </div>
                    """)
            else:
                f.write("<p>No alerts in the last 24 hours.</p>\n")
                
            f.write("</div>\n")
            
            f.write("""
            </body>
            </html>
            """)
            
        return output_file
    
    def _generate_metric_chart(self, metric_name: str) -> Optional[str]:
        """Generate chart for a metric.
        
        Args:
            metric_name: Metric name
            
        Returns:
            Path to generated chart file
        """
        # Get metric values
        values = self.metrics_collector.get_metric_values(metric_name)
        
        if not values:
            return None
            
        # Extract timestamps and values
        timestamps = [ts for ts, _ in values]
        metric_values = [val for _, val in values]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, metric_values)
        plt.title(f"{metric_name} Trend")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, f"{metric_name}_chart.png")
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def generate_alert_report(self, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> str:
        """Generate alert report.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Path to generated report file
        """
        # Default to last 7 days if not specified
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(days=7)
            
        if end_time is None:
            end_time = datetime.utcnow()
            
        # Get alerts
        alerts = self.alert_manager.get_alert_history(start_time, end_time)
        
        # Create report file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"alert_report_{timestamp}.html")
        
        # Generate HTML
        with open(output_file, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Alert Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; }}
                    .alert {{ padding: 10px; margin-bottom: 10px; border-radius: 5px; }}
                    .info {{ background-color: #d1ecf1; }}
                    .warning {{ background-color: #fff3cd; }}
                    .error {{ background-color: #f8d7da; }}
                    .critical {{ background-color: #dc3545; color: white; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Alert Report</h1>
                <p>Period: {start_time.strftime("%Y-%m-%d %H:%M:%S")} to {end_time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Generated at: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h2>Summary</h2>
                <p>Total alerts: {len(alerts)}</p>
            """)
            
            # Add severity breakdown
            severity_counts = {}
            for alert in alerts:
                severity = alert.get("severity", "info")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
            if severity_counts:
                f.write("<table>\n")
                f.write("<tr><th>Severity</th><th>Count</th></tr>\n")
                
                for severity, count in severity_counts.items():
                    f.write(f"<tr><td>{severity}</td><td>{count}</td></tr>\n")
                    
                f.write("</table>\n")
                
            # Add alert details
            f.write("<h2>Alert Details</h2>\n")
            
            if alerts:
                f.write("<table>\n")
                f.write("<tr><th>Time</th><th>Name</th><th>Severity</th><th>Metric</th><th>Threshold</th><th>Actual</th><th>Description</th></tr>\n")
                
                for alert in alerts:
                    timestamp = alert.get("timestamp", datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S")
                    name = alert.get("name", "")
                    severity = alert.get("severity", "info")
                    metric_name = alert.get("metric_name", "")
                    threshold = alert.get("threshold", "")
                    actual_value = alert.get("actual_value", "")
                    description = alert.get("description", "")
                    
                    f.write(f"<tr><td>{timestamp}</td><td>{name}</td><td>{severity}</td><td>{metric_name}</td><td>{threshold}</td><td>{actual_value}</td><td>{description}</td></tr>\n")
                    
                f.write("</table>\n")
            else:
                f.write("<p>No alerts found for the specified period.</p>\n")
                
            f.write("""
            </body>
            </html>
            """)
            
        return output_file
