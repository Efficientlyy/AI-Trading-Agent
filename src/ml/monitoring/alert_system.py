"""Real-time alert notification system."""

import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass
import logging
import json
from .performance_monitor import Alert

@dataclass
class EmailConfig:
    """Email notification configuration."""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_address: str
    to_addresses: List[str]
    use_tls: bool = True

@dataclass
class SlackConfig:
    """Slack notification configuration."""
    webhook_url: str
    channel: str
    username: str = "Trading Bot"
    icon_emoji: str = ":chart_with_upwards_trend:"

@dataclass
class DiscordConfig:
    """Discord notification configuration."""
    webhook_url: str
    username: str = "Trading Bot"

@dataclass
class TelegramConfig:
    """Telegram notification configuration."""
    bot_token: str
    chat_ids: List[str]

class AlertNotifier:
    """Alert notification system supporting multiple channels."""
    
    def __init__(
        self,
        email_config: Optional[EmailConfig] = None,
        slack_config: Optional[SlackConfig] = None,
        discord_config: Optional[DiscordConfig] = None,
        telegram_config: Optional[TelegramConfig] = None
    ):
        """Initialize notifier with channel configurations."""
        self.email_config = email_config
        self.slack_config = slack_config
        self.discord_config = discord_config
        self.telegram_config = telegram_config
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def send_alert(
        self,
        alert: Alert,
        channels: Optional[Set[str]] = None
    ) -> Dict[str, bool]:
        """Send alert through specified channels.
        
        Args:
            alert: Alert to send
            channels: Set of channels to use ('email', 'slack', 'discord', 'telegram')
                     If None, use all configured channels
        
        Returns:
            Dictionary of channel -> success status
        """
        if channels is None:
            channels = {
                'email', 'slack', 'discord', 'telegram'
            }
        
        results = {}
        
        if 'email' in channels and self.email_config:
            results['email'] = self._send_email_alert(alert)
        
        if 'slack' in channels and self.slack_config:
            results['slack'] = self._send_slack_alert(alert)
        
        if 'discord' in channels and self.discord_config:
            results['discord'] = self._send_discord_alert(alert)
        
        if 'telegram' in channels and self.telegram_config:
            results['telegram'] = self._send_telegram_alert(alert)
        
        return results
    
    def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not self.email_config:
            self.logger.error("Email config is not set")
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config.from_address
            msg['To'] = ', '.join(self.email_config.to_addresses)
            msg['Subject'] = f"Trading Alert: {alert.category} - {alert.level}"
            
            # Create message body
            body = (
                f"Time: {alert.timestamp}\n"
                f"Level: {alert.level}\n"
                f"Category: {alert.category}\n"
                f"Message: {alert.message}\n\n"
                f"Metrics:\n"
                + "\n".join(f"  {k}: {v}" for k, v in alert.metrics.items())
            )
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(
                self.email_config.smtp_server,
                self.email_config.smtp_port
            )
            
            if self.email_config.use_tls:
                server.starttls()
            
            server.login(
                self.email_config.username,
                self.email_config.password
            )
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def _send_slack_alert(self, alert: Alert) -> bool:
        """Send alert via Slack."""
        if not self.slack_config:
            self.logger.error("Slack config is not set")
            return False
            
        try:
            # Create message blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Trading Alert: {alert.category}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{alert.timestamp}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Level:*\n{alert.level}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Message:*\n{alert.message}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Metrics:*\n" + "\n".join(
                            f"• {k}: {v}" for k, v in alert.metrics.items()
                        )
                    }
                }
            ]
            
            # Prepare payload
            payload = {
                "channel": self.slack_config.channel,
                "username": self.slack_config.username,
                "icon_emoji": self.slack_config.icon_emoji,
                "blocks": blocks
            }
            
            # Send message
            response = requests.post(
                self.slack_config.webhook_url,
                json=payload
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {str(e)}")
            return False
    
    def _send_discord_alert(self, alert: Alert) -> bool:
        """Send alert via Discord."""
        if not self.discord_config:
            self.logger.error("Discord config is not set")
            return False
            
        try:
            # Create embed
            embed = {
                "title": f"Trading Alert: {alert.category}",
                "description": alert.message,
                "color": {
                    "ERROR": 0xFF0000,
                    "WARNING": 0xFFAA00,
                    "INFO": 0x00AA00
                }.get(alert.level, 0x000000),
                "fields": [
                    {
                        "name": "Level",
                        "value": alert.level,
                        "inline": True
                    },
                    {
                        "name": "Time",
                        "value": str(alert.timestamp),
                        "inline": True
                    }
                ] + [
                    {
                        "name": k,
                        "value": str(v),
                        "inline": True
                    }
                    for k, v in alert.metrics.items()
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Prepare payload
            payload = {
                "username": self.discord_config.username,
                "embeds": [embed]
            }
            
            # Send message
            response = requests.post(
                self.discord_config.webhook_url,
                json=payload
            )
            
            return response.status_code == 204
            
        except Exception as e:
            self.logger.error(f"Failed to send Discord alert: {str(e)}")
            return False
    
    def _send_telegram_alert(self, alert: Alert) -> bool:
        """Send alert via Telegram."""
        if not self.telegram_config:
            self.logger.error("Telegram config is not set")
            return False
            
        try:
            # Create message text
            message = (
                f"🚨 *Trading Alert: {alert.category}*\n\n"
                f"*Level:* {alert.level}\n"
                f"*Time:* {alert.timestamp}\n"
                f"*Message:* {alert.message}\n\n"
                f"*Metrics:*\n"
                + "\n".join(f"• {k}: {v}" for k, v in alert.metrics.items())
            )
            
            success = True
            
            # Send to all chat IDs
            for chat_id in self.telegram_config.chat_ids:
                # Prepare API URL
                url = (
                    f"https://api.telegram.org/bot{self.telegram_config.bot_token}"
                    f"/sendMessage"
                )
                
                # Prepare payload
                payload = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                }
                
                # Send message
                response = requests.post(url, json=payload)
                
                if response.status_code != 200:
                    success = False
                    self.logger.error(
                        f"Failed to send Telegram alert to {chat_id}: "
                        f"{response.text}"
                    )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {str(e)}")
            return False 