#!/usr/bin/env python
"""
Market Monitor Plugin for System Overseer.
"""

import os
import json
import logging
import time
import threading
from datetime import datetime

logger = logging.getLogger("system_overseer.plugins.market_monitor")

class MarketMonitorPlugin:
    """Market Monitor Plugin for System Overseer."""
    
    def __init__(self):
        """Initialize Market Monitor Plugin."""
        self.id = "market_monitor"
        self.name = "Market Monitor"
        self.description = "Monitors market conditions and provides alerts"
        self.version = "1.0.0"
        self.system_core = None
        self.config = {
            "update_interval": 300,  # 5 minutes
            "pairs": ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
        }
        self.data_dir = None
        self.running = False
        self.monitor_thread = None
        self.last_update = 0
        self.market_data = {}
    
    def initialize(self, system_core):
        """Initialize plugin with system core.
        
        Args:
            system_core: System core instance
        """
        logger.info("Initializing Market Monitor Plugin")
        self.system_core = system_core
        self.data_dir = os.path.join(system_core.data_dir, "plugins", "market_monitor")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Get configuration from system core
        config_registry = system_core.get_service("config_registry")
        if config_registry:
            plugin_config = config_registry.get_config("plugins.market_monitor", {})
            self.config.update(plugin_config)
        
        logger.info("Market Monitor Plugin initialized")
        return True
    
    def start(self):
        """Start plugin."""
        logger.info("Starting Market Monitor Plugin")
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Market Monitor Plugin started")
        return True
    
    def stop(self):
        """Stop plugin."""
        logger.info("Stopping Market Monitor Plugin")
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Market Monitor Plugin stopped")
        return True
    
    def _monitor_loop(self):
        """Monitor loop."""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_update >= self.config["update_interval"]:
                    self._update_market_data()
                    self._check_alerts()
                    self.last_update = current_time
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
    
    def _update_market_data(self):
        """Update market data."""
        logger.info("Updating market data")
        
        # In a real implementation, this would fetch actual market data
        # For now, we'll generate mock data
        for pair in self.config["pairs"]:
            self.market_data[pair] = {
                "price": round(10000 + ((time.time() % 1000) - 500), 2),
                "volume": round(1000000 + ((time.time() % 1000000) - 500000), 2),
                "change_24h": round(((time.time() % 10) - 5), 2),
                "high_24h": round(10500 + ((time.time() % 500) - 250), 2),
                "low_24h": round(9500 + ((time.time() % 500) - 250), 2),
                "timestamp": datetime.now().isoformat()
            }
        
        # Save market data to file
        with open(os.path.join(self.data_dir, "market_data.json"), "w") as f:
            json.dump(self.market_data, f, indent=2)
        
        logger.info("Market data updated")
    
    def _check_alerts(self):
        """Check for alerts."""
        logger.info("Checking for alerts")
        
        # In a real implementation, this would check for actual alert conditions
        # For now, we'll generate mock alerts
        alerts = []
        
        for pair, data in self.market_data.items():
            if abs(data["change_24h"]) > 5:
                alerts.append({
                    "pair": pair,
                    "type": "price_change",
                    "level": "high",
                    "message": f"{pair} price changed by {data['change_24h']}% in the last 24 hours",
                    "timestamp": datetime.now().isoformat()
                })
        
        if alerts:
            # Save alerts to file
            alerts_file = os.path.join(self.data_dir, "alerts.json")
            existing_alerts = []
            
            if os.path.exists(alerts_file):
                with open(alerts_file, "r") as f:
                    try:
                        existing_alerts = json.load(f)
                    except json.JSONDecodeError:
                        existing_alerts = []
            
            existing_alerts.extend(alerts)
            
            with open(alerts_file, "w") as f:
                json.dump(existing_alerts, f, indent=2)
            
            # Notify system core
            event_bus = self.system_core.get_service("event_bus")
            if event_bus:
                for alert in alerts:
                    event_bus.publish("market_monitor.alert", alert)
        
        logger.info(f"Found {len(alerts)} alerts")
    
    def get_market_data(self, pair=None):
        """Get market data.
        
        Args:
            pair: Trading pair
            
        Returns:
            dict: Market data
        """
        if pair:
            return self.market_data.get(pair, {})
        else:
            return self.market_data
    
    def get_alerts(self, count=10):
        """Get recent alerts.
        
        Args:
            count: Number of alerts to return
            
        Returns:
            list: Recent alerts
        """
        alerts_file = os.path.join(self.data_dir, "alerts.json")
        
        if os.path.exists(alerts_file):
            with open(alerts_file, "r") as f:
                try:
                    alerts = json.load(f)
                    return alerts[-count:]
                except json.JSONDecodeError:
                    return []
        else:
            return []
