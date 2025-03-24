"""Monitoring and alerting system for sentiment analysis components.

This module provides automated monitoring, alerting, and health tracking
for the sentiment analysis system, detecting performance degradation,
availability issues, and providing tiered alerting with configurable thresholds.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union

from src.common.config import config
from src.common.logging import get_logger
from src.common.events import event_bus, Event


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class AlertType(Enum):
    """Types of alerts that can be generated."""
    API_FAILURE = "api_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONFIDENCE_DEVIATION = "confidence_deviation"
    DATA_QUALITY = "data_quality"
    COST_THRESHOLD = "cost_threshold"
    RATE_LIMIT = "rate_limit"
    SYSTEM_ERROR = "system_error"
    LATENCY_THRESHOLD = "latency_threshold"
    COMPONENT_FAILURE = "component_failure"
    WEIGHT_DRIFT = "weight_drift"


class AlertStatus(Enum):
    """Status of an alert."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class Alert:
    """Represents a monitoring alert."""
    
    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        source: str,
        message: str,
        details: Dict[str, Any] = None,
        related_entities: List[str] = None
    ):
        """Initialize a new alert.
        
        Args:
            alert_type: Type of the alert
            severity: Severity level
            source: Component that generated the alert
            message: Alert message
            details: Additional alert details
            related_entities: Entities related to this alert (symbols, models, etc.)
        """
        self.id = f"{int(time.time())}_{source}_{alert_type.value}"
        self.timestamp = datetime.utcnow()
        self.alert_type = alert_type
        self.severity = severity
        self.source = source
        self.message = message
        self.details = details or {}
        self.related_entities = related_entities or []
        self.status = AlertStatus.ACTIVE
        self.acknowledged_time = None
        self.resolved_time = None
        self.resolution_message = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization.
        
        Returns:
            Dictionary representation of alert
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type.value,
            "severity": self.severity.name,
            "source": self.source,
            "message": self.message,
            "details": self.details,
            "related_entities": self.related_entities,
            "status": self.status.value,
            "acknowledged_time": self.acknowledged_time.isoformat() if self.acknowledged_time else None,
            "resolved_time": self.resolved_time.isoformat() if self.resolved_time else None,
            "resolution_message": self.resolution_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary.
        
        Args:
            data: Dictionary representation of alert
            
        Returns:
            Alert instance
        """
        alert = cls(
            alert_type=AlertType(data["alert_type"]),
            severity=AlertSeverity[data["severity"]],
            source=data["source"],
            message=data["message"],
            details=data.get("details", {}),
            related_entities=data.get("related_entities", [])
        )
        
        alert.id = data["id"]
        alert.timestamp = datetime.fromisoformat(data["timestamp"])
        alert.status = AlertStatus(data["status"])
        
        if data.get("acknowledged_time"):
            alert.acknowledged_time = datetime.fromisoformat(data["acknowledged_time"])
            
        if data.get("resolved_time"):
            alert.resolved_time = datetime.fromisoformat(data["resolved_time"])
            
        alert.resolution_message = data.get("resolution_message")
        
        return alert
    
    def acknowledge(self) -> None:
        """Mark the alert as acknowledged."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_time = datetime.utcnow()
    
    def resolve(self, message: str = None) -> None:
        """Mark the alert as resolved.
        
        Args:
            message: Optional resolution message
        """
        self.status = AlertStatus.RESOLVED
        self.resolved_time = datetime.utcnow()
        self.resolution_message = message
    
    def ignore(self) -> None:
        """Mark the alert as ignored."""
        self.status = AlertStatus.IGNORED


class AlertThresholds:
    """Configurable thresholds for generating alerts."""
    
    def __init__(self):
        """Initialize alert thresholds from configuration."""
        # API related thresholds
        self.api_error_count = config.get("monitoring.thresholds.api_error_count", 3)
        self.api_error_window = config.get("monitoring.thresholds.api_error_window", 300)  # seconds
        
        # Performance thresholds
        self.accuracy_drop_percent = config.get("monitoring.thresholds.accuracy_drop_percent", 10)
        self.confidence_deviation = config.get("monitoring.thresholds.confidence_deviation", 0.2)
        
        # Latency thresholds
        self.api_latency_threshold = config.get("monitoring.thresholds.api_latency_threshold", 2000)  # ms
        self.processing_latency_threshold = config.get("monitoring.thresholds.processing_latency_threshold", 5000)  # ms
        
        # Cost thresholds
        self.daily_cost_threshold = config.get("monitoring.thresholds.daily_cost_threshold", 50.0)  # dollars
        self.weekly_cost_threshold = config.get("monitoring.thresholds.weekly_cost_threshold", 250.0)  # dollars
        
        # Data quality thresholds
        self.data_quality_threshold = config.get("monitoring.thresholds.data_quality_threshold", 0.8)  # 0-1 scale
        self.missing_data_threshold = config.get("monitoring.thresholds.missing_data_threshold", 0.1)  # percent
        
        # Weight drift thresholds
        self.weight_drift_threshold = config.get("monitoring.thresholds.weight_drift_threshold", 0.15)  # max change
        
        # Rate limiting
        self.rate_limit_threshold = config.get("monitoring.thresholds.rate_limit_threshold", 0.8)  # percent of limit


class NotificationChannel:
    """Base class for alert notification channels."""
    
    def __init__(self, name: str):
        """Initialize notification channel.
        
        Args:
            name: Channel name
        """
        self.name = name
        self.logger = get_logger("monitoring", f"channel_{name}")
        self.min_severity = AlertSeverity.INFO
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for an alert.
        
        Args:
            alert: The alert to send
            
        Returns:
            True if notification was sent successfully
        """
        # Skip notifications below minimum severity
        if alert.severity.value < self.min_severity.value:
            return True
        
        return await self._send(alert)
    
    async def _send(self, alert: Alert) -> bool:
        """Implementation-specific notification sending.
        
        Args:
            alert: The alert to send
            
        Returns:
            True if notification was sent successfully
        """
        raise NotImplementedError("Subclasses must implement _send")


class LogNotificationChannel(NotificationChannel):
    """Notification channel that logs alerts."""
    
    def __init__(self):
        """Initialize log notification channel."""
        super().__init__("log")
    
    async def _send(self, alert: Alert) -> bool:
        """Log the alert.
        
        Args:
            alert: The alert to send
            
        Returns:
            True always
        """
        log_level = logging.INFO
        if alert.severity == AlertSeverity.WARNING:
            log_level = logging.WARNING
        elif alert.severity == AlertSeverity.ERROR:
            log_level = logging.ERROR
        elif alert.severity == AlertSeverity.CRITICAL:
            log_level = logging.CRITICAL
        
        self.logger.log(
            log_level,
            f"ALERT [{alert.severity.name}] - {alert.source}: {alert.message}"
        )
        return True


class FileNotificationChannel(NotificationChannel):
    """Notification channel that writes alerts to a file."""
    
    def __init__(self, file_path: str = None):
        """Initialize file notification channel.
        
        Args:
            file_path: Path to the alert log file
        """
        super().__init__("file")
        self.file_path = file_path or os.path.join("logs", "alerts.log")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
    
    async def _send(self, alert: Alert) -> bool:
        """Write alert to file.
        
        Args:
            alert: The alert to send
            
        Returns:
            True if alert was written successfully
        """
        try:
            with open(self.file_path, "a") as f:
                f.write(f"[{alert.timestamp.isoformat()}] [{alert.severity.name}] {alert.source}: {alert.message}\n")
                
                # Add details for higher severity alerts
                if alert.severity.value >= AlertSeverity.ERROR.value and alert.details:
                    f.write(f"  Details: {json.dumps(alert.details)}\n")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write alert to file: {str(e)}")
            return False


class EventBusNotificationChannel(NotificationChannel):
    """Notification channel that publishes alerts to the event bus."""
    
    def __init__(self):
        """Initialize event bus notification channel."""
        super().__init__("event_bus")
    
    async def _send(self, alert: Alert) -> bool:
        """Publish alert to event bus.
        
        Args:
            alert: The alert to send
            
        Returns:
            True always
        """
        # Create a dictionary representation of the alert
        alert_data = alert.to_dict()
        
        # Publish to event bus
        event_bus.publish(
            "sentiment_alert",
            alert_data
        )
        
        return True


class WebhookNotificationChannel(NotificationChannel):
    """Notification channel that sends alerts to a webhook."""
    
    def __init__(self, webhook_url: str = None):
        """Initialize webhook notification channel.
        
        Args:
            webhook_url: URL of the webhook
        """
        super().__init__("webhook")
        self.webhook_url = webhook_url or config.get("monitoring.webhook_url")
        
        # Set minimum severity to WARNING for webhook notifications
        self.min_severity = AlertSeverity.WARNING
    
    async def _send(self, alert: Alert) -> bool:
        """Send alert to webhook.
        
        Args:
            alert: The alert to send
            
        Returns:
            True if alert was sent successfully
        """
        if not self.webhook_url:
            self.logger.warning("Webhook URL not configured")
            return False
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=alert.to_dict(),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status < 300:
                        return True
                    else:
                        self.logger.error(
                            f"Failed to send webhook notification: {response.status} - {response.text()}"
                        )
                        return False
                        
        except ImportError:
            self.logger.error("aiohttp not installed, cannot send webhook notification")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {str(e)}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Notification channel that sends alerts to Slack."""
    
    def __init__(self, webhook_url: str = None):
        """Initialize Slack notification channel.
        
        Args:
            webhook_url: URL of the Slack webhook
        """
        super().__init__("slack")
        self.webhook_url = webhook_url or config.get("monitoring.slack_webhook_url")
        
        # Set minimum severity to WARNING for Slack notifications
        self.min_severity = AlertSeverity.WARNING
    
    async def _send(self, alert: Alert) -> bool:
        """Send alert to Slack.
        
        Args:
            alert: The alert to send
            
        Returns:
            True if alert was sent successfully
        """
        if not self.webhook_url:
            self.logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            import aiohttp
            
            # Format Slack message
            color = "#3498db"  # Info (blue)
            if alert.severity == AlertSeverity.WARNING:
                color = "#f39c12"  # Warning (orange)
            elif alert.severity == AlertSeverity.ERROR:
                color = "#e74c3c"  # Error (red)
            elif alert.severity == AlertSeverity.CRITICAL:
                color = "#9b59b6"  # Critical (purple)
            
            message = {
                "attachments": [
                    {
                        "color": color,
                        "pretext": f"*{alert.severity.name} Alert*: {alert.source}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Type",
                                "value": alert.alert_type.value,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            }
                        ],
                        "footer": "Sentiment Analysis Monitoring System",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Add details for higher severity alerts
            if alert.severity.value >= AlertSeverity.ERROR.value and alert.details:
                details_text = "\n".join([f"• *{k}*: {v}" for k, v in alert.details.items()])
                message["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": details_text,
                    "short": False
                })
            
            # Add related entities if any
            if alert.related_entities:
                message["attachments"][0]["fields"].append({
                    "title": "Related Entities",
                    "value": ", ".join(alert.related_entities),
                    "short": False
                })
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=message,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status < 300:
                        return True
                    else:
                        self.logger.error(
                            f"Failed to send Slack notification: {response.status} - {response.text()}"
                        )
                        return False
                        
        except ImportError:
            self.logger.error("aiohttp not installed, cannot send Slack notification")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {str(e)}")
            return False


class EmailNotificationChannel(NotificationChannel):
    """Notification channel that sends alerts via email."""
    
    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = None,
        smtp_username: str = None,
        smtp_password: str = None,
        from_email: str = None,
        to_emails: List[str] = None
    ):
        """Initialize email notification channel.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP port
            smtp_username: SMTP username
            smtp_password: SMTP password
            from_email: Sender email address
            to_emails: Recipient email addresses
        """
        super().__init__("email")
        self.smtp_server = smtp_server or config.get("monitoring.smtp_server")
        self.smtp_port = smtp_port or config.get("monitoring.smtp_port", 587)
        self.smtp_username = smtp_username or config.get("monitoring.smtp_username")
        self.smtp_password = smtp_password or config.get("monitoring.smtp_password")
        self.from_email = from_email or config.get("monitoring.from_email")
        self.to_emails = to_emails or config.get("monitoring.to_emails", [])
        
        # Set minimum severity to ERROR for email notifications
        self.min_severity = AlertSeverity.ERROR
    
    async def _send(self, alert: Alert) -> bool:
        """Send alert via email.
        
        Args:
            alert: The alert to send
            
        Returns:
            True if alert was sent successfully
        """
        if not all([self.smtp_server, self.smtp_username, self.smtp_password, self.from_email, self.to_emails]):
            self.logger.warning("Email notification channel not fully configured")
            return False
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            msg["Subject"] = f"[{alert.severity.name}] Sentiment Analysis Alert: {alert.source}"
            
            # Create the plain-text message body
            text = f"""
Alert: {alert.message}
Type: {alert.alert_type.value}
Severity: {alert.severity.name}
Source: {alert.source}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Add details for higher severity alerts
            if alert.details:
                text += "\nDetails:\n"
                for k, v in alert.details.items():
                    text += f"  {k}: {v}\n"
            
            # Add related entities if any
            if alert.related_entities:
                text += f"\nRelated Entities: {', '.join(alert.related_entities)}\n"
            
            msg.attach(MIMEText(text, "plain"))
            
            # Send the message via SMTP server
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
                
            return True
            
        except ImportError:
            self.logger.error("email modules not available, cannot send email notification")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")
            return False


class AlertManager:
    """Manages alerts, thresholds, and notifications."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.logger = get_logger("monitoring", "alert_manager")
        self.thresholds = AlertThresholds()
        self.alerts: List[Alert] = []
        self.notification_channels: List[NotificationChannel] = []
        self.deduplication_window = config.get("monitoring.deduplication_window", 300)  # 5 minutes
        self.alert_storage_path = os.path.join("data", "monitoring", "alerts.json")
        
        # Create directory for alert storage
        os.makedirs(os.path.dirname(self.alert_storage_path), exist_ok=True)
        
        # Initialize default notification channels
        self._initialize_default_channels()
        
        # API error tracking
        self.api_errors: Dict[str, List[datetime]] = {}
        
        # Performance tracking
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
    
    def _initialize_default_channels(self) -> None:
        """Initialize default notification channels."""
        # Always add log and event bus channels
        self.notification_channels.append(LogNotificationChannel())
        self.notification_channels.append(EventBusNotificationChannel())
        self.notification_channels.append(FileNotificationChannel())
        
        # Add Slack channel if configured
        if config.get("monitoring.slack_webhook_url"):
            self.notification_channels.append(SlackNotificationChannel())
        
        # Add webhook channel if configured
        if config.get("monitoring.webhook_url"):
            self.notification_channels.append(WebhookNotificationChannel())
        
        # Add email channel if configured
        if config.get("monitoring.smtp_server"):
            self.notification_channels.append(EmailNotificationChannel())
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel.
        
        Args:
            channel: The notification channel to add
        """
        self.notification_channels.append(channel)
        self.logger.info(f"Added notification channel: {channel.name}")
    
    async def initialize(self) -> None:
        """Initialize the alert manager."""
        self.logger.info("Initializing alert manager")
        
        # Load alert history
        self._load_alerts()
        
        # Load performance baselines
        self._load_performance_baselines()
        
        # Subscribe to relevant events
        event_bus.subscribe("api_error", self.handle_api_error)
        event_bus.subscribe("model_performance", self.handle_performance_event)
        event_bus.subscribe("sentiment_data_quality", self.handle_data_quality_event)
        event_bus.subscribe("sentiment_model_latency", self.handle_latency_event)
        event_bus.subscribe("sentiment_api_cost", self.handle_cost_event)
        event_bus.subscribe("weight_change", self.handle_weight_change_event)
        event_bus.subscribe("component_status", self.handle_component_status_event)
    
    async def _load_alerts(self) -> None:
        """Load alert history from storage."""
        if not os.path.exists(self.alert_storage_path):
            self.logger.info("No alert history found")
            return
        
        try:
            with open(self.alert_storage_path, "r") as f:
                alerts_data = json.load(f)
            
            self.alerts = [Alert.from_dict(alert_data) for alert_data in alerts_data]
            self.logger.info(f"Loaded {len(self.alerts)} alerts from storage")
            
        except Exception as e:
            self.logger.error(f"Failed to load alerts: {str(e)}")
    
    async def _save_alerts(self) -> None:
        """Save alerts to storage."""
        try:
            # Only keep the last 1000 alerts to prevent the file from growing too large
            alerts_to_save = self.alerts[-1000:] if len(self.alerts) > 1000 else self.alerts
            
            alerts_data = [alert.to_dict() for alert in alerts_to_save]
            
            with open(self.alert_storage_path, "w") as f:
                json.dump(alerts_data, f, indent=2)
                
            self.logger.debug(f"Saved {len(alerts_to_save)} alerts to storage")
            
        except Exception as e:
            self.logger.error(f"Failed to save alerts: {str(e)}")
    
    async def _load_performance_baselines(self) -> None:
        """Load performance baselines from storage."""
        baseline_path = os.path.join("data", "monitoring", "performance_baselines.json")
        
        if not os.path.exists(baseline_path):
            self.logger.info("No performance baselines found")
            return
        
        try:
            with open(baseline_path, "r") as f:
                self.performance_baselines = json.load(f)
                
            self.logger.info(f"Loaded performance baselines for {len(self.performance_baselines)} sources")
            
        except Exception as e:
            self.logger.error(f"Failed to load performance baselines: {str(e)}")
    
    async def _save_performance_baselines(self) -> None:
        """Save performance baselines to storage."""
        baseline_path = os.path.join("data", "monitoring", "performance_baselines.json")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
            
            with open(baseline_path, "w") as f:
                json.dump(self.performance_baselines, f, indent=2)
                
            self.logger.debug("Saved performance baselines to storage")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance baselines: {str(e)}")
    
    def _is_duplicate_alert(self, alert: Alert) -> bool:
        """Check if alert is a duplicate within deduplication window.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert is a duplicate
        """
        # Deduplication window start time
        window_start = datetime.utcnow() - timedelta(seconds=self.deduplication_window)
        
        # Check for similar alerts in the deduplication window
        for existing_alert in self.alerts:
            # Skip alerts outside the deduplication window
            if existing_alert.timestamp < window_start:
                continue
            
            # Skip resolved or ignored alerts
            if existing_alert.status in [AlertStatus.RESOLVED, AlertStatus.IGNORED]:
                continue
            
            # Check if alert is similar
            is_similar = (
                existing_alert.alert_type == alert.alert_type and
                existing_alert.source == alert.source and
                existing_alert.severity == alert.severity
            )
            
            # For some alert types, also check related entities
            if is_similar and alert.alert_type in [
                AlertType.API_FAILURE,
                AlertType.PERFORMANCE_DEGRADATION,
                AlertType.CONFIDENCE_DEVIATION
            ]:
                # Check if related entities overlap
                if set(existing_alert.related_entities).intersection(set(alert.related_entities)):
                    return True
            elif is_similar:
                return True
        
        return False
    
    async def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        source: str,
        message: str,
        details: Dict[str, Any] = None,
        related_entities: List[str] = None
    ) -> Optional[Alert]:
        """Create and process a new alert.
        
        Args:
            alert_type: Type of the alert
            severity: Alert severity
            source: Component that generated the alert
            message: Alert message
            details: Additional details
            related_entities: Related entities (symbols, models, etc.)
            
        Returns:
            Created alert or None if it was a duplicate
        """
        # Create alert
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            source=source,
            message=message,
            details=details,
            related_entities=related_entities
        )
        
        # Check for duplicates
        if self._is_duplicate_alert(alert):
            self.logger.debug(f"Suppressed duplicate alert: {message}")
            return None
        
        # Add alert to history
        self.alerts.append(alert)
        
        # Send notifications
        notification_tasks = [
            channel.send_notification(alert)
            for channel in self.notification_channels
        ]
        
        if notification_tasks:
            await asyncio.gather(*notification_tasks)
        
        # Save alerts to storage
        self._save_alerts()
        
        return alert
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts.
        
        Returns:
            List of active alerts
        """
        return [
            alert for alert in self.alerts
            if alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]
        ]
    
    async def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get list of recent alerts.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ]
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if alert was acknowledged
        """
        for alert in self.alerts:
            if alert.id == alert_id and alert.status == AlertStatus.ACTIVE:
                alert.acknowledge()
                self._save_alerts()
                return True
        
        return False
    
    async def resolve_alert(self, alert_id: str, message: str = None) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            message: Optional resolution message
            
        Returns:
            True if alert was resolved
        """
        for alert in self.alerts:
            if alert.id == alert_id and alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
                alert.resolve(message)
                self._save_alerts()
                return True
        
        return False
    
    async def handle_api_error(self, event: Event) -> None:
        """Handle API error events.
        
        Args:
            event: The API error event
        """
        data = event.data
        provider = data.get("provider", "unknown")
        error_message = data.get("error", "Unknown error")
        status_code = data.get("status_code")
        operation = data.get("operation", "API call")
        
        # Add error to history
        if provider not in self.api_errors:
            self.api_errors[provider] = []
        
        self.api_errors[provider].append(datetime.utcnow())
        
        # Check for error threshold
        error_count = 0
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.thresholds.api_error_window)
        
        for timestamp in self.api_errors[provider]:
            if timestamp >= cutoff_time:
                error_count += 1
        
        # Determine severity based on error count
        severity = AlertSeverity.WARNING
        if error_count >= self.thresholds.api_error_count * 2:
            severity = AlertSeverity.CRITICAL
        elif error_count >= self.thresholds.api_error_count:
            severity = AlertSeverity.ERROR
        
        # Only create alert if threshold is exceeded
        if error_count >= self.thresholds.api_error_count:
            await self.create_alert(
                alert_type=AlertType.API_FAILURE,
                severity=severity,
                source=f"api_{provider}",
                message=f"API errors detected for {provider}: {error_count} in the last {self.thresholds.api_error_window}s",
                details={
                    "provider": provider,
                    "error_count": error_count,
                    "window_seconds": self.thresholds.api_error_window,
                    "latest_error": error_message,
                    "status_code": status_code,
                    "operation": operation
                },
                related_entities=[provider]
            )
    
    async def handle_performance_event(self, event: Event) -> None:
        """Handle model performance events.
        
        Args:
            event: The performance event
        """
        data = event.data
        source = data.get("source", "unknown")
        metric = data.get("metric", "accuracy")
        value = data.get("value", 0.0)
        symbol = data.get("symbol")
        
        # If symbol is provided, use it in the source identifier
        source_id = f"{source}_{symbol}" if symbol else source
        
        # Initialize baseline if not exists
        if source_id not in self.performance_baselines:
            self.performance_baselines[source_id] = {}
        
        if metric not in self.performance_baselines[source_id]:
            # First time seeing this metric, establish baseline
            self.performance_baselines[source_id][metric] = value
            self._save_performance_baselines()
            return
        
        # Calculate percentage drop
        baseline = self.performance_baselines[source_id][metric]
        pct_change = ((value - baseline) / baseline) * 100 if baseline != 0 else 0
        
        # Check for performance degradation
        if pct_change <= -self.thresholds.accuracy_drop_percent:
            # Determine severity based on degradation amount
            severity = AlertSeverity.WARNING
            if pct_change <= -self.thresholds.accuracy_drop_percent * 3:
                severity = AlertSeverity.CRITICAL
            elif pct_change <= -self.thresholds.accuracy_drop_percent * 2:
                severity = AlertSeverity.ERROR
            
            await self.create_alert(
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=severity,
                source=source,
                message=f"Performance degradation detected for {source}" + 
                        (f" on {symbol}" if symbol else "") + 
                        f": {metric} dropped by {abs(pct_change):.1f}%",
                details={
                    "source": source,
                    "symbol": symbol,
                    "metric": metric,
                    "current_value": value,
                    "baseline_value": baseline,
                    "percent_change": pct_change
                },
                related_entities=[symbol] if symbol else []
            )
        elif pct_change >= self.thresholds.accuracy_drop_percent * 2:
            # Performance has improved significantly, update baseline
            self.performance_baselines[source_id][metric] = value
            self._save_performance_baselines()
            
            self.logger.info(
                f"Updated performance baseline for {source_id}/{metric}: {baseline:.4f} -> {value:.4f} (+{pct_change:.1f}%)"
            )
    
    async def handle_data_quality_event(self, event: Event) -> None:
        """Handle data quality events.
        
        Args:
            event: The data quality event
        """
        data = event.data
        source = data.get("source", "unknown")
        quality_score = data.get("quality_score", 1.0)
        missing_data_pct = data.get("missing_data_percent", 0.0)
        details = data.get("details", {})
        
        # Check quality score threshold
        if quality_score < self.thresholds.data_quality_threshold:
            severity = AlertSeverity.WARNING
            if quality_score < self.thresholds.data_quality_threshold / 2:
                severity = AlertSeverity.ERROR
            
            await self.create_alert(
                alert_type=AlertType.DATA_QUALITY,
                severity=severity,
                source=source,
                message=f"Data quality issue detected for {source}: quality score {quality_score:.2f}",
                details={
                    "source": source,
                    "quality_score": quality_score,
                    "threshold": self.thresholds.data_quality_threshold,
                    **details
                }
            )
        
        # Check missing data threshold
        if missing_data_pct > self.thresholds.missing_data_threshold:
            severity = AlertSeverity.WARNING
            if missing_data_pct > self.thresholds.missing_data_threshold * 2:
                severity = AlertSeverity.ERROR
            
            await self.create_alert(
                alert_type=AlertType.DATA_QUALITY,
                severity=severity,
                source=source,
                message=f"Missing data detected for {source}: {missing_data_pct:.1f}% missing",
                details={
                    "source": source,
                    "missing_data_percent": missing_data_pct,
                    "threshold": self.thresholds.missing_data_threshold,
                    **details
                }
            )
    
    async def handle_latency_event(self, event: Event) -> None:
        """Handle latency events.
        
        Args:
            event: The latency event
        """
        data = event.data
        source = data.get("source", "unknown")
        latency_ms = data.get("latency_ms", 0)
        operation_type = data.get("operation_type", "unknown")
        details = data.get("details", {})
        
        # Determine threshold based on operation type
        threshold = (
            self.thresholds.api_latency_threshold
            if operation_type == "api"
            else self.thresholds.processing_latency_threshold
        )
        
        # Check latency threshold
        if latency_ms > threshold:
            severity = AlertSeverity.WARNING
            if latency_ms > threshold * 2:
                severity = AlertSeverity.ERROR
            
            await self.create_alert(
                alert_type=AlertType.LATENCY_THRESHOLD,
                severity=severity,
                source=source,
                message=f"High latency detected for {source}: {latency_ms:.0f}ms ({operation_type})",
                details={
                    "source": source,
                    "latency_ms": latency_ms,
                    "threshold": threshold,
                    "operation_type": operation_type,
                    **details
                }
            )
    
    async def handle_cost_event(self, event: Event) -> None:
        """Handle cost events.
        
        Args:
            event: The cost event
        """
        data = event.data
        provider = data.get("provider", "unknown")
        daily_cost = data.get("daily_cost", 0.0)
        weekly_cost = data.get("weekly_cost", 0.0)
        details = data.get("details", {})
        
        # Check daily cost threshold
        if daily_cost > self.thresholds.daily_cost_threshold:
            severity = AlertSeverity.WARNING
            if daily_cost > self.thresholds.daily_cost_threshold * 1.5:
                severity = AlertSeverity.ERROR
            
            await self.create_alert(
                alert_type=AlertType.COST_THRESHOLD,
                severity=severity,
                source=f"cost_{provider}",
                message=f"Daily cost threshold exceeded for {provider}: ${daily_cost:.2f}",
                details={
                    "provider": provider,
                    "daily_cost": daily_cost,
                    "threshold": self.thresholds.daily_cost_threshold,
                    **details
                },
                related_entities=[provider]
            )
        
        # Check weekly cost threshold
        if weekly_cost > self.thresholds.weekly_cost_threshold:
            severity = AlertSeverity.WARNING
            if weekly_cost > self.thresholds.weekly_cost_threshold * 1.2:
                severity = AlertSeverity.ERROR
            
            await self.create_alert(
                alert_type=AlertType.COST_THRESHOLD,
                severity=severity,
                source=f"cost_{provider}",
                message=f"Weekly cost threshold exceeded for {provider}: ${weekly_cost:.2f}",
                details={
                    "provider": provider,
                    "weekly_cost": weekly_cost,
                    "threshold": self.thresholds.weekly_cost_threshold,
                    **details
                },
                related_entities=[provider]
            )
    
    async def handle_weight_change_event(self, event: Event) -> None:
        """Handle weight change events.
        
        Args:
            event: The weight change event
        """
        data = event.data
        source = data.get("source", "unknown")
        weight_changes = data.get("weight_changes", {})
        symbol = data.get("symbol")
        
        # Check for excessive weight changes
        excessive_changes = {}
        for model, change in weight_changes.items():
            if abs(change) > self.thresholds.weight_drift_threshold:
                excessive_changes[model] = change
        
        if excessive_changes:
            await self.create_alert(
                alert_type=AlertType.WEIGHT_DRIFT,
                severity=AlertSeverity.WARNING,
                source=source,
                message=f"Significant weight changes detected for {source}" + 
                        (f" on {symbol}" if symbol else ""),
                details={
                    "source": source,
                    "symbol": symbol,
                    "weight_changes": excessive_changes,
                    "threshold": self.thresholds.weight_drift_threshold
                },
                related_entities=[symbol] if symbol else []
            )
    
    async def handle_component_status_event(self, event: Event) -> None:
        """Handle component status events.
        
        Args:
            event: The component status event
        """
        data = event.data
        component = data.get("component", "unknown")
        status = data.get("status", "unknown")
        details = data.get("details", {})
        
        # Only alert on error statuses
        if status.lower() in ["error", "failed", "unhealthy"]:
            await self.create_alert(
                alert_type=AlertType.COMPONENT_FAILURE,
                severity=AlertSeverity.ERROR,
                source=component,
                message=f"Component failure detected: {component} is {status}",
                details={
                    "component": component,
                    "status": status,
                    **details
                }
            )


# Singleton instance
alert_manager = AlertManager()