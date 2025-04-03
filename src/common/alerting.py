"""Alerting system based on log patterns and metrics thresholds.

This module provides alerting capabilities triggered by log patterns or metrics
thresholds, with support for multiple notification channels.
"""

import json
import logging
import re
import smtplib
import threading
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

import requests
import structlog

from src.common.config import config

# Initialize logger
logger = structlog.get_logger("alerting")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Available alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    LOG = "log"


class AlertRule:
    """Rule for triggering alerts based on log patterns or metrics."""
    
    def __init__(
        self,
        name: str,
        description: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        channels: List[AlertChannel] = None,
        pattern: Optional[str] = None,
        threshold: Optional[Dict[str, Any]] = None,
        cooldown: int = 300,  # 5 minutes
        enabled: bool = True
    ):
        """
        Initialize a new alert rule.
        
        Args:
            name: Unique name for the rule
            description: Description of the alert
            severity: Alert severity level
            channels: List of notification channels
            pattern: Regex pattern to match in logs
            threshold: Dict with metric_name, operator, and threshold value
            cooldown: Minimum seconds between repeated alerts
            enabled: Whether the rule is active
        """
        self.name = name
        self.description = description
        self.severity = severity
        self.channels = channels or [AlertChannel.LOG]
        self.pattern = re.compile(pattern) if pattern else None
        self.threshold = threshold
        self.cooldown = cooldown
        self.enabled = enabled
        self.last_triggered = None
        
    def check_log_entry(self, log_entry: Dict[str, Any]) -> bool:
        """
        Check if a log entry matches this rule.
        
        Args:
            log_entry: Log entry to check
            
        Returns:
            True if the log entry matches and triggers the rule
        """
        if not self.enabled or not self.pattern:
            return False
            
        # Check if we're in cooldown period
        if self.last_triggered and (datetime.now() - self.last_triggered).total_seconds() < self.cooldown:
            return False
            
        # Check if the log message matches the pattern
        if 'event' in log_entry and isinstance(log_entry['event'], str):
            if self.pattern.search(log_entry['event']):
                self.last_triggered = datetime.now()
                return True
                
        # Also check other fields
        for key, value in log_entry.items():
            if isinstance(value, str) and self.pattern.search(value):
                self.last_triggered = datetime.now()
                return True
                
        return False
        
    def check_metric(self, metric_name: str, value: float) -> bool:
        """
        Check if a metric value triggers this rule.
        
        Args:
            metric_name: Name of the metric
            value: Current value of the metric
            
        Returns:
            True if the metric value triggers the rule
        """
        if not self.enabled or not self.threshold:
            return False
            
        # Check if we're in cooldown period
        if self.last_triggered and (datetime.now() - self.last_triggered).total_seconds() < self.cooldown:
            return False
            
        # Check if this rule applies to this metric
        if self.threshold.get('metric_name') != metric_name:
            return False
            
        # Get operator and threshold value
        operator = self.threshold.get('operator', '>')
        threshold_value = self.threshold.get('value', 0)
        
        # Check the condition
        triggered = False
        if operator == '>':
            triggered = value > threshold_value
        elif operator == '>=':
            triggered = value >= threshold_value
        elif operator == '<':
            triggered = value < threshold_value
        elif operator == '<=':
            triggered = value <= threshold_value
        elif operator == '==':
            triggered = value == threshold_value
        elif operator == '!=':
            triggered = value != threshold_value
            
        if triggered:
            self.last_triggered = datetime.now()
            
        return triggered
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'severity': self.severity.value,
            'channels': [channel.value for channel in self.channels],
            'pattern': self.pattern.pattern if self.pattern else None,
            'threshold': self.threshold,
            'cooldown': self.cooldown,
            'enabled': self.enabled,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None
        }


class AlertManager:
    """Manager for alert rules and notifications."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.rules = []
        self.lock = threading.Lock()
        self.notifiers = {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SLACK: self._send_slack_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.SMS: self._send_sms_alert,
            AlertChannel.PAGERDUTY: self._send_pagerduty_alert,
            AlertChannel.LOG: self._send_log_alert
        }
        
    def add_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.
        
        Args:
            rule: Alert rule to add
        """
        with self.lock:
            # Check if rule with this name already exists
            for i, existing_rule in enumerate(self.rules):
                if existing_rule.name == rule.name:
                    # Replace existing rule
                    self.rules[i] = rule
                    logger.info(f"Updated alert rule: {rule.name}")
                    return
                    
            # Add new rule
            self.rules.append(rule)
            logger.info(f"Added alert rule: {rule.name}")
            
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if the rule was removed, False if not found
        """
        with self.lock:
            for i, rule in enumerate(self.rules):
                if rule.name == rule_name:
                    del self.rules[i]
                    logger.info(f"Removed alert rule: {rule_name}")
                    return True
                    
            logger.warning(f"Failed to remove alert rule (not found): {rule_name}")
            return False
            
    def enable_rule(self, rule_name: str) -> bool:
        """Enable an alert rule."""
        with self.lock:
            for rule in self.rules:
                if rule.name == rule_name:
                    rule.enabled = True
                    logger.info(f"Enabled alert rule: {rule_name}")
                    return True
            return False
            
    def disable_rule(self, rule_name: str) -> bool:
        """Disable an alert rule."""
        with self.lock:
            for rule in self.rules:
                if rule.name == rule_name:
                    rule.enabled = False
                    logger.info(f"Disabled alert rule: {rule_name}")
                    return True
            return False
            
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all alert rules as dictionaries."""
        with self.lock:
            return [rule.to_dict() for rule in self.rules]
            
    def process_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """
        Process a log entry and trigger alerts if needed.
        
        Args:
            log_entry: Log entry to process
        """
        triggered_rules = []
        
        with self.lock:
            for rule in self.rules:
                if rule.check_log_entry(log_entry):
                    triggered_rules.append(rule)
                    
        # Trigger alerts for matched rules
        for rule in triggered_rules:
            self._trigger_alert(rule, log_entry)
            
    def process_metric(self, metric_name: str, value: float) -> None:
        """
        Process a metric update and trigger alerts if needed.
        
        Args:
            metric_name: Name of the metric
            value: Current value of the metric
        """
        triggered_rules = []
        
        with self.lock:
            for rule in self.rules:
                if rule.check_metric(metric_name, value):
                    triggered_rules.append(rule)
                    
        # Trigger alerts for matched rules
        for rule in triggered_rules:
            metric_data = {
                'metric_name': metric_name,
                'value': value,
                'timestamp': datetime.now().isoformat()
            }
            self._trigger_alert(rule, metric_data)
            
    def _trigger_alert(self, rule: AlertRule, data: Dict[str, Any]) -> None:
        """
        Trigger alerts for a rule.
        
        Args:
            rule: The rule that was triggered
            data: Data that triggered the rule
        """
        alert_data = {
            'rule': rule.name,
            'description': rule.description,
            'severity': rule.severity.value,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Send notifications through each channel
        for channel in rule.channels:
            if channel in self.notifiers:
                try:
                    self.notifiers[channel](alert_data)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel.value}", error=str(e))
                    
    def load_rules_from_config(self) -> None:
        """Load alert rules from configuration."""
        rules_config = config.get("system.alerting.rules", [])
        
        for rule_data in rules_config:
            try:
                # Convert string values to enums
                severity = AlertSeverity(rule_data.get('severity', 'warning'))
                channels = [AlertChannel(c) for c in rule_data.get('channels', ['log'])]
                
                # Create rule
                rule = AlertRule(
                    name=rule_data['name'],
                    description=rule_data.get('description', ''),
                    severity=severity,
                    channels=channels,
                    pattern=rule_data.get('pattern'),
                    threshold=rule_data.get('threshold'),
                    cooldown=rule_data.get('cooldown', 300),
                    enabled=rule_data.get('enabled', True)
                )
                
                self.add_rule(rule)
            except Exception as e:
                logger.error(f"Failed to load alert rule: {rule_data.get('name', 'unknown')}", error=str(e))
                
    def _send_email_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send an alert via email."""
        email_config = config.get("system.alerting.channels.email", {})
        if not email_config.get('enabled', False):
            return
            
        # Get email configuration
        smtp_server = email_config.get('smtp_server', 'localhost')
        smtp_port = email_config.get('smtp_port', 25)
        smtp_username = email_config.get('smtp_username')
        smtp_password = email_config.get('smtp_password')
        sender = email_config.get('sender', 'alerts@example.com')
        recipients = email_config.get('recipients', [])
        
        if not recipients:
            logger.warning("No email recipients configured for alerts")
            return
            
        # Create email message
        subject = f"{alert_data['severity'].upper()} Alert: {alert_data['rule']}"
        
        # Create message
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ', '.join(recipients)
        msg["Subject"] = subject
        
        # Add HTML body
        body = f"""
        <html>
        <body>
            <h2>{subject}</h2>
            <p><strong>Description:</strong> {alert_data['description']}</p>
            <p><strong>Time:</strong> {alert_data['timestamp']}</p>
            <p><strong>Severity:</strong> {alert_data['severity']}</p>
            <h3>Alert Data:</h3>
            <pre>{json.dumps(alert_data['data'], indent=2)}</pre>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            if smtp_username and smtp_password:
                server.starttls()
                server.login(smtp_username, smtp_password)
            server.send_message(msg)
            server.quit()
            logger.info(f"Sent email alert: {subject}")
        except Exception as e:
            logger.error("Failed to send email alert", error=str(e))
            
    def _send_slack_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send an alert to Slack."""
        slack_config = config.get("system.alerting.channels.slack", {})
        if not slack_config.get('enabled', False):
            return
            
        webhook_url = slack_config.get('webhook_url')
        if not webhook_url:
            logger.warning("No Slack webhook URL configured for alerts")
            return
            
        # Create message
        color = {
            'info': '#36a64f',  # green
            'warning': '#f2c744',  # yellow
            'error': '#d00000',  # red
            'critical': '#7d0000'  # dark red
        }.get(alert_data['severity'], '#36a64f')
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"{alert_data['severity'].upper()} Alert: {alert_data['rule']}",
                'text': alert_data['description'],
                'fields': [
                    {
                        'title': 'Time',
                        'value': alert_data['timestamp'],
                        'short': True
                    },
                    {
                        'title': 'Severity',
                        'value': alert_data['severity'].upper(),
                        'short': True
                    }
                ],
                'footer': 'AI Trading Agent Alerting System'
            }]
        }
        
        # Add alert data
        if 'event' in alert_data['data']:
            payload['attachments'][0]['fields'].append({
                'title': 'Event',
                'value': f"```{alert_data['data']['event']}```",
                'short': False
            })
        elif 'metric_name' in alert_data['data']:
            payload['attachments'][0]['fields'].append({
                'title': 'Metric',
                'value': f"{alert_data['data']['metric_name']} = {alert_data['data']['value']}",
                'short': False
            })
            
        # Send to Slack
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logger.info(f"Sent Slack alert: {alert_data['rule']}")
        except Exception as e:
            logger.error("Failed to send Slack alert", error=str(e))
            
    def _send_webhook_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send an alert to a webhook."""
        webhook_config = config.get("system.alerting.channels.webhook", {})
        if not webhook_config.get('enabled', False):
            return
            
        url = webhook_config.get('url')
        headers = webhook_config.get('headers', {})
        
        if not url:
            logger.warning("No webhook URL configured for alerts")
            return
            
        # Send to webhook
        try:
            response = requests.post(
                url,
                json=alert_data,
                headers=headers
            )
            response.raise_for_status()
            logger.info(f"Sent webhook alert: {alert_data['rule']}")
        except Exception as e:
            logger.error("Failed to send webhook alert", error=str(e))
            
    def _send_sms_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send an alert via SMS."""
        sms_config = config.get("system.alerting.channels.sms", {})
        if not sms_config.get('enabled', False):
            return
            
        # This would typically use a third-party SMS service
        # Implementation depends on the service used
        logger.info(f"SMS alerts not implemented: {alert_data['rule']}")
            
    def _send_pagerduty_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send an alert to PagerDuty."""
        pagerduty_config = config.get("system.alerting.channels.pagerduty", {})
        if not pagerduty_config.get('enabled', False):
            return
            
        integration_key = pagerduty_config.get('integration_key')
        if not integration_key:
            logger.warning("No PagerDuty integration key configured for alerts")
            return
            
        # Map severity to PagerDuty severity
        severity_map = {
            'info': 'info',
            'warning': 'warning',
            'error': 'error',
            'critical': 'critical'
        }
        
        # Create event payload
        payload = {
            'routing_key': integration_key,
            'event_action': 'trigger',
            'payload': {
                'summary': f"{alert_data['severity'].upper()} Alert: {alert_data['rule']}",
                'source': 'AI Trading Agent',
                'severity': severity_map.get(alert_data['severity'], 'warning'),
                'timestamp': alert_data['timestamp'],
                'custom_details': alert_data['data']
            }
        }
        
        # Send to PagerDuty
        try:
            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logger.info(f"Sent PagerDuty alert: {alert_data['rule']}")
        except Exception as e:
            logger.error("Failed to send PagerDuty alert", error=str(e))
            
    def _send_log_alert(self, alert_data: Dict[str, Any]) -> None:
        """Log the alert."""
        severity = alert_data['severity']
        if severity == 'critical':
            logger.critical(f"ALERT: {alert_data['rule']}", alert=alert_data)
        elif severity == 'error':
            logger.error(f"ALERT: {alert_data['rule']}", alert=alert_data)
        elif severity == 'warning':
            logger.warning(f"ALERT: {alert_data['rule']}", alert=alert_data)
        else:
            logger.info(f"ALERT: {alert_data['rule']}", alert=alert_data)


# Global alert manager instance
alert_manager = AlertManager()

# Initialize alert rules from config
alert_manager.load_rules_from_config()


# Hook into the logging system
def log_handler(log_entry):
    """
    Handler for processing log entries.
    
    Args:
        log_entry: Log entry to process
    """
    # Process log entry for alerts
    alert_manager.process_log_entry(log_entry)
    
    # Return the log entry unchanged
    return log_entry


def process_metric(metric_name, value):
    """
    Process a metric update for alerting.
    
    Args:
        metric_name: Name of the metric
        value: Current value of the metric
    """
    alert_manager.process_metric(metric_name, value)
