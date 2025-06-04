#!/usr/bin/env python
"""
Enhanced Telegram Notification System

This module provides an improved integration of Telegram notifications
with the trading system, including proper logging, error handling, and
customizable notification templates.
"""

import os
import sys
import json
import time
import logging
import threading
import requests
from queue import Queue
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from enhanced_logging_fixed import EnhancedLogger

# Initialize enhanced logger
logger = EnhancedLogger("telegram_notifications")

class EnhancedTelegramNotifier:
    """Enhanced Telegram notification system with improved integration"""
    
    def __init__(self, config=None):
        """Initialize enhanced Telegram notifier
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logger
        
        # Get Telegram configuration
        self.bot_token = self.config.get('telegram_bot_token') or os.environ.get('TELEGRAM_BOT_TOKEN')
        self.user_id = self.config.get('telegram_user_id') or os.environ.get('TELEGRAM_USER_ID')
        
        # Notification queue
        self.notification_queue = Queue()
        
        # Running flag
        self.running = False
        
        # Notification templates
        self.templates = {
            'signal': '🔔 *Signal Detected*\n'
                      'Symbol: `{symbol}`\n'
                      'Type: `{type}`\n'
                      'Strength: `{strength:.2f}`\n'
                      'Price: `{price:.2f}`\n'
                      'Source: `{source}`\n'
                      'Time: `{time}`',
            
            'order_created': '📝 *Order Created*\n'
                            'Symbol: `{symbol}`\n'
                            'Side: `{side}`\n'
                            'Type: `{type}`\n'
                            'Quantity: `{quantity:.8f}`\n'
                            'Price: `{price:.2f}`\n'
                            'Order ID: `{order_id}`\n'
                            'Time: `{time}`',
            
            'order_filled': '✅ *Order Filled*\n'
                           'Symbol: `{symbol}`\n'
                           'Side: `{side}`\n'
                           'Type: `{type}`\n'
                           'Quantity: `{quantity:.8f}`\n'
                           'Price: `{price:.2f}`\n'
                           'Order ID: `{order_id}`\n'
                           'Time: `{time}`',
            
            'order_cancelled': '❌ *Order Cancelled*\n'
                              'Symbol: `{symbol}`\n'
                              'Side: `{side}`\n'
                              'Type: `{type}`\n'
                              'Quantity: `{quantity:.8f}`\n'
                              'Price: `{price:.2f}`\n'
                              'Order ID: `{order_id}`\n'
                              'Reason: `{reason}`\n'
                              'Time: `{time}`',
            
            'error': '⚠️ *Error*\n'
                    'Component: `{component}`\n'
                    'Message: `{message}`\n'
                    'Time: `{time}`',
            
            'system': '🖥️ *System Notification*\n'
                     'Component: `{component}`\n'
                     'Message: `{message}`\n'
                     'Time: `{time}`',
            
            'decision': '🧠 *Trading Decision*\n'
                       'Symbol: `{symbol}`\n'
                       'Action: `{action}`\n'
                       'Confidence: `{confidence:.2f}`\n'
                       'Reason: `{reason}`\n'
                       'Time: `{time}`',
            
            'performance': '📊 *Performance Metrics*\n'
                          'Metric: `{metric}`\n'
                          'Value: `{value}`\n'
                          'Time: `{time}`'
        }
        
        # Initialize
        self.initialize()
        
        self.logger.system.info("Enhanced Telegram notifier initialized")
    
    def initialize(self):
        """Initialize Telegram notifier"""
        if not self.bot_token:
            self.logger.system.warning("Telegram bot token not found, notifications will be logged only")
            self.mock_mode = True
        elif not self.user_id:
            self.logger.system.warning("Telegram user ID not found, notifications will be logged only")
            self.mock_mode = True
        else:
            self.mock_mode = False
            self.logger.system.info(f"Telegram notifier initialized for user ID: {self.user_id}")
    
    def start(self):
        """Start the notification system"""
        self.logger.system.info("Starting Telegram notification system")
        
        try:
            # Start notification processing thread
            self.running = True
            self.notification_thread = threading.Thread(target=self.process_notifications)
            self.notification_thread.daemon = True
            self.notification_thread.start()
            
            self.logger.system.info("Telegram notification system started")
        except Exception as e:
            self.logger.log_error("Error starting Telegram notification system", component="telegram")
            raise
    
    def stop(self):
        """Stop the notification system"""
        self.logger.system.info("Stopping Telegram notification system")
        
        try:
            # Stop notification processing
            self.running = False
            
            # Wait for thread to terminate
            if hasattr(self, 'notification_thread') and self.notification_thread.is_alive():
                self.notification_thread.join(timeout=5.0)
            
            self.logger.system.info("Telegram notification system stopped")
        except Exception as e:
            self.logger.log_error("Error stopping Telegram notification system", component="telegram")
            raise
    
    def process_notifications(self):
        """Process notifications from the queue"""
        self.logger.system.info("Notification processing thread started")
        last_heartbeat = time.time()
        
        while self.running:
            try:
                # Send heartbeat log periodically
                current_time = time.time()
                if current_time - last_heartbeat > 60:
                    self.logger.system.debug("Notification processing thread heartbeat")
                    last_heartbeat = current_time
                
                # Process notifications from queue
                if not self.notification_queue.empty():
                    notification = self.notification_queue.get(timeout=0.1)
                    self.send_notification(notification)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.logger.log_error("Error in notification processing thread", component="telegram")
                time.sleep(1)  # Prevent tight loop on persistent errors
        
        self.logger.system.info("Notification processing thread stopped")
    
    def send_notification(self, notification):
        """Send a notification
        
        Args:
            notification: Notification dictionary
        """
        try:
            # Extract notification data
            notification_type = notification.get('type', 'system')
            notification_data = notification.get('data', {})
            
            # Get template
            template = self.templates.get(notification_type, self.templates['system'])
            
            # Add timestamp if not present
            if 'time' not in notification_data:
                notification_data['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Format message
            message = template.format(**notification_data)
            
            # Send notification
            if self.mock_mode:
                self.logger.system.info(f"Mock notification: {message}")
            else:
                self.send_telegram_message(message)
                
            self.logger.system.info(f"Notification sent: {notification_type}")
        except Exception as e:
            self.logger.log_error(f"Error sending notification: {str(e)}", component="telegram")
    
    def send_telegram_message(self, message):
        """Send message to Telegram
        
        Args:
            message: Message text
        """
        try:
            # Prepare request
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.user_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            # Send request
            response = requests.post(url, data=data)
            
            # Check response
            if response.status_code == 200:
                self.logger.system.debug("Telegram message sent successfully")
            else:
                self.logger.system.error(f"Error sending Telegram message: {response.status_code} - {response.text}")
        except Exception as e:
            self.logger.log_error(f"Error sending Telegram message: {str(e)}", component="telegram")
    
    def notify_signal(self, signal):
        """Notify about a trading signal
        
        Args:
            signal: Signal dictionary
        """
        notification = {
            'type': 'signal',
            'data': {
                'symbol': signal.get('symbol', 'unknown'),
                'type': signal.get('type', 'unknown'),
                'strength': signal.get('strength', 0.0),
                'price': signal.get('price', 0.0),
                'source': signal.get('source', 'unknown'),
                'time': datetime.fromtimestamp(signal.get('timestamp', time.time() * 1000) / 1000).strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.notification_queue.put(notification)
    
    def notify_order_created(self, order):
        """Notify about an order creation
        
        Args:
            order: Order dictionary
        """
        notification = {
            'type': 'order_created',
            'data': {
                'symbol': order.get('symbol', 'unknown'),
                'side': order.get('side', 'unknown'),
                'type': order.get('type', 'unknown'),
                'quantity': order.get('quantity', 0.0),
                'price': order.get('price', 0.0),
                'order_id': order.get('orderId', 'unknown'),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.notification_queue.put(notification)
    
    def notify_order_filled(self, order):
        """Notify about an order fill
        
        Args:
            order: Order dictionary
        """
        notification = {
            'type': 'order_filled',
            'data': {
                'symbol': order.get('symbol', 'unknown'),
                'side': order.get('side', 'unknown'),
                'type': order.get('type', 'unknown'),
                'quantity': order.get('quantity', 0.0),
                'price': order.get('price', 0.0),
                'order_id': order.get('orderId', 'unknown'),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.notification_queue.put(notification)
    
    def notify_order_cancelled(self, order, reason="Unknown"):
        """Notify about an order cancellation
        
        Args:
            order: Order dictionary
            reason: Cancellation reason
        """
        notification = {
            'type': 'order_cancelled',
            'data': {
                'symbol': order.get('symbol', 'unknown'),
                'side': order.get('side', 'unknown'),
                'type': order.get('type', 'unknown'),
                'quantity': order.get('quantity', 0.0),
                'price': order.get('price', 0.0),
                'order_id': order.get('orderId', 'unknown'),
                'reason': reason,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.notification_queue.put(notification)
    
    def notify_error(self, component, message):
        """Notify about an error
        
        Args:
            component: Component name
            message: Error message
        """
        notification = {
            'type': 'error',
            'data': {
                'component': component,
                'message': message,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.notification_queue.put(notification)
    
    def notify_system(self, component, message):
        """Notify about a system event
        
        Args:
            component: Component name
            message: System message
        """
        notification = {
            'type': 'system',
            'data': {
                'component': component,
                'message': message,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.notification_queue.put(notification)
    
    def notify_decision(self, decision):
        """Notify about a trading decision
        
        Args:
            decision: Decision dictionary
        """
        notification = {
            'type': 'decision',
            'data': {
                'symbol': decision.get('symbol', 'unknown'),
                'action': decision.get('action', 'HOLD'),
                'confidence': decision.get('confidence', 0.0),
                'reason': decision.get('reason', 'No reason provided'),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.notification_queue.put(notification)
    
    def notify_performance(self, metric, value):
        """Notify about a performance metric
        
        Args:
            metric: Metric name
            value: Metric value
        """
        notification = {
            'type': 'performance',
            'data': {
                'metric': metric,
                'value': value,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.notification_queue.put(notification)
    
    # Add send_signal_notification method for integrated pipeline compatibility
    def send_signal_notification(self, signal):
        """Send signal notification for integrated pipeline compatibility
        
        Args:
            signal: Signal dictionary
        """
        self.logger.system.info(f"Sending signal notification for {signal.get('symbol', 'unknown')}")
        self.notify_signal(signal)
    
    # Add send_trade_notification method for integrated pipeline compatibility
    def send_trade_notification(self, trade):
        """Send trade notification for integrated pipeline compatibility
        
        Args:
            trade: Trade dictionary
        """
        self.logger.system.info(f"Sending trade notification for {trade.get('symbol', 'unknown')}")
        
        # Create order-like structure for notification
        order = {
            'symbol': trade.get('symbol', 'unknown'),
            'side': trade.get('action', 'unknown'),
            'type': 'MARKET',
            'quantity': trade.get('quantity', 0.0),
            'price': trade.get('price', 0.0),
            'orderId': trade.get('order_id', trade.get('id', 'unknown'))
        }
        
        self.notify_order_filled(order)
    
    # Add send_error_notification method for integrated pipeline compatibility
    def send_error_notification(self, message, component="system"):
        """Send error notification for integrated pipeline compatibility
        
        Args:
            message: Error message
            component: Component name
        """
        self.logger.system.info(f"Sending error notification: {message}")
        self.notify_error(component, message)
    
    # Add send_system_notification method for integrated pipeline compatibility
    def send_system_notification(self, message, details=None, component="system"):
        """Send system notification for integrated pipeline compatibility
        
        Args:
            message: System message
            details: Additional details (optional)
            component: Component name
        """
        self.logger.system.info(f"Sending system notification: {message}")
        
        # Add details to message if provided
        if details:
            if isinstance(details, dict):
                details_str = ", ".join(f"{k}: {v}" for k, v in details.items())
                message = f"{message}\n{details_str}"
            else:
                message = f"{message}\n{details}"
        
        self.notify_system(component, message)

# Alias EnhancedTelegramNotifier as TelegramNotifier for compatibility with existing code
TelegramNotifier = EnhancedTelegramNotifier
