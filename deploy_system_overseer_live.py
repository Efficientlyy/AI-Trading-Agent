#!/usr/bin/env python
"""
System Overseer Deployment Script

This script deploys the System Overseer with real services and configurations.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("system_overseer_deployment.log")
    ]
)
logger = logging.getLogger("system_overseer_deployment")

# Load environment variables
load_dotenv('.env-secure/.env')

def create_directories():
    """Create necessary directories for the System Overseer."""
    logger.info("Creating necessary directories...")
    
    directories = [
        "config",
        "data",
        "logs",
        "plugins",
        "system_overseer/data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True

def create_config_files():
    """Create configuration files for the System Overseer."""
    logger.info("Creating configuration files...")
    
    # System configuration
    system_config = {
        "system": {
            "name": "Trading System Overseer",
            "version": "1.0.0",
            "environment": "production"
        },
        "trading": {
            "default_pairs": ["BTCUSDC", "ETHUSDC", "SOLUSDC"],
            "risk_level": "moderate",
            "exchange": "mexc"
        },
        "notifications": {
            "level": "all",
            "frequency": "medium"
        },
        "llm": {
            "provider": "openrouter",
            "model": "openai/gpt-4o",
            "temperature": 0.7,
            "max_tokens": 500
        }
    }
    
    # Personality configuration
    personality_config = {
        "traits": {
            "formality": 0.7,
            "verbosity": 0.6,
            "helpfulness": 0.9,
            "proactivity": 0.8
        },
        "memory": {
            "retention_period": 7,
            "importance_threshold": 0.5
        }
    }
    
    # Plugins configuration
    plugins_config = {
        "plugins": [
            {
                "id": "trading_analytics",
                "path": "plugins.trading_analytics.TradingAnalyticsPlugin",
                "enabled": True,
                "config": {}
            },
            {
                "id": "market_monitor",
                "path": "plugins.market_monitor.MarketMonitorPlugin",
                "enabled": True,
                "config": {
                    "update_interval": 300,
                    "pairs": ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
                }
            }
        ]
    }
    
    # Write configuration files
    with open("config/system.json", "w") as f:
        json.dump(system_config, f, indent=2)
    
    with open("config/personality.json", "w") as f:
        json.dump(personality_config, f, indent=2)
    
    with open("config/plugins.json", "w") as f:
        json.dump(plugins_config, f, indent=2)
    
    logger.info("Configuration files created successfully")
    return True

def create_requirements_file():
    """Create requirements file for the System Overseer."""
    logger.info("Creating requirements file...")
    
    requirements = [
        "python-dotenv",
        "requests",
        "python-telegram-bot",
        "tiktoken",
        "pandas",
        "numpy",
        "matplotlib",
        "pyyaml"
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    logger.info("Requirements file created successfully")
    return True

def install_dependencies():
    """Install dependencies for the System Overseer."""
    logger.info("Installing dependencies...")
    
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def create_plugin_files():
    """Create plugin files for the System Overseer."""
    logger.info("Creating plugin files...")
    
    # Create plugins directory
    os.makedirs("plugins", exist_ok=True)
    
    # Create __init__.py
    with open("plugins/__init__.py", "w") as f:
        f.write('"""System Overseer plugins package."""\n')
    
    # Create trading analytics plugin
    os.makedirs("plugins/trading_analytics", exist_ok=True)
    
    with open("plugins/trading_analytics/__init__.py", "w") as f:
        f.write('"""Trading Analytics plugin package."""\n')
    
    trading_analytics_plugin = """#!/usr/bin/env python
\"\"\"
Trading Analytics Plugin for System Overseer.
\"\"\"

import os
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger("system_overseer.plugins.trading_analytics")

class TradingAnalyticsPlugin:
    \"\"\"Trading Analytics Plugin for System Overseer.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize Trading Analytics Plugin.\"\"\"
        self.id = "trading_analytics"
        self.name = "Trading Analytics"
        self.description = "Provides trading analytics and insights"
        self.version = "1.0.0"
        self.system_core = None
        self.config = {}
        self.data_dir = None
        self.running = False
    
    def initialize(self, system_core):
        \"\"\"Initialize plugin with system core.
        
        Args:
            system_core: System core instance
        \"\"\"
        logger.info("Initializing Trading Analytics Plugin")
        self.system_core = system_core
        self.data_dir = os.path.join(system_core.data_dir, "plugins", "trading_analytics")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Get configuration from system core
        config_registry = system_core.get_service("config_registry")
        if config_registry:
            plugin_config = config_registry.get_config("plugins.trading_analytics", {})
            self.config.update(plugin_config)
        
        logger.info("Trading Analytics Plugin initialized")
        return True
    
    def start(self):
        \"\"\"Start plugin.\"\"\"
        logger.info("Starting Trading Analytics Plugin")
        self.running = True
        logger.info("Trading Analytics Plugin started")
        return True
    
    def stop(self):
        \"\"\"Stop plugin.\"\"\"
        logger.info("Stopping Trading Analytics Plugin")
        self.running = False
        logger.info("Trading Analytics Plugin stopped")
        return True
    
    def get_analytics(self, pair=None):
        \"\"\"Get trading analytics.
        
        Args:
            pair: Trading pair
            
        Returns:
            dict: Trading analytics
        \"\"\"
        # In a real implementation, this would fetch actual trading data
        # For now, we'll return mock data
        pairs = pair if pair else self.config.get("pairs", ["BTCUSDC", "ETHUSDC", "SOLUSDC"])
        
        if not isinstance(pairs, list):
            pairs = [pairs]
        
        analytics = {}
        
        for p in pairs:
            analytics[p] = {
                "performance": {
                    "daily": round(((time.time() % 10) - 5) * 2, 2),
                    "weekly": round(((time.time() % 20) - 10) * 1.5, 2),
                    "monthly": round(((time.time() % 30) - 15), 2)
                },
                "trades": {
                    "total": int(time.time() % 100),
                    "successful": int((time.time() % 100) * 0.7),
                    "failed": int((time.time() % 100) * 0.3)
                },
                "signals": {
                    "buy": int(time.time() % 50),
                    "sell": int((time.time() % 50) * 0.8)
                }
            }
        
        return analytics
    
    def get_insights(self, pair=None):
        \"\"\"Get trading insights.
        
        Args:
            pair: Trading pair
            
        Returns:
            dict: Trading insights
        \"\"\"
        # In a real implementation, this would generate actual insights
        # For now, we'll return mock insights
        analytics = self.get_analytics(pair)
        insights = {}
        
        for p, data in analytics.items():
            performance = data["performance"]
            trades = data["trades"]
            
            if performance["daily"] > 0:
                trend = "upward"
                sentiment = "positive"
            else:
                trend = "downward"
                sentiment = "negative"
            
            success_rate = trades["successful"] / trades["total"] if trades["total"] > 0 else 0
            
            insights[p] = {
                "trend": trend,
                "sentiment": sentiment,
                "success_rate": round(success_rate * 100, 2),
                "recommendation": "buy" if performance["daily"] > 0 and success_rate > 0.6 else "hold" if performance["daily"] > -2 else "sell"
            }
        
        return insights
"""
    
    with open("plugins/trading_analytics/trading_analytics_plugin.py", "w") as f:
        f.write(trading_analytics_plugin)
    
    # Create market monitor plugin
    os.makedirs("plugins/market_monitor", exist_ok=True)
    
    with open("plugins/market_monitor/__init__.py", "w") as f:
        f.write('"""Market Monitor plugin package."""\n')
    
    market_monitor_plugin = """#!/usr/bin/env python
\"\"\"
Market Monitor Plugin for System Overseer.
\"\"\"

import os
import json
import logging
import time
import threading
from datetime import datetime

logger = logging.getLogger("system_overseer.plugins.market_monitor")

class MarketMonitorPlugin:
    \"\"\"Market Monitor Plugin for System Overseer.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize Market Monitor Plugin.\"\"\"
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
        \"\"\"Initialize plugin with system core.
        
        Args:
            system_core: System core instance
        \"\"\"
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
        \"\"\"Start plugin.\"\"\"
        logger.info("Starting Market Monitor Plugin")
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Market Monitor Plugin started")
        return True
    
    def stop(self):
        \"\"\"Stop plugin.\"\"\"
        logger.info("Stopping Market Monitor Plugin")
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Market Monitor Plugin stopped")
        return True
    
    def _monitor_loop(self):
        \"\"\"Monitor loop.\"\"\"
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
        \"\"\"Update market data.\"\"\"
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
        \"\"\"Check for alerts.\"\"\"
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
        \"\"\"Get market data.
        
        Args:
            pair: Trading pair
            
        Returns:
            dict: Market data
        \"\"\"
        if pair:
            return self.market_data.get(pair, {})
        else:
            return self.market_data
    
    def get_alerts(self, count=10):
        \"\"\"Get recent alerts.
        
        Args:
            count: Number of alerts to return
            
        Returns:
            list: Recent alerts
        \"\"\"
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
"""
    
    with open("plugins/market_monitor/market_monitor_plugin.py", "w") as f:
        f.write(market_monitor_plugin)
    
    logger.info("Plugin files created successfully")
    return True

def create_service_script():
    """Create service script for the System Overseer."""
    logger.info("Creating service script...")
    
    service_script = """#!/usr/bin/env python
\"\"\"
System Overseer Service Script

This script runs the System Overseer as a background service,
connecting to Telegram and providing continuous monitoring and control.
\"\"\"

import os
import sys
import time
import signal
import logging
import argparse
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/system_overseer.log")
    ]
)
logger = logging.getLogger("system_overseer.service")

# Import System Overseer components
from system_overseer.module_registry import ModuleRegistry
from system_overseer.config_registry import ConfigRegistry
from system_overseer.event_bus import EventBus
from system_overseer.core import SystemCore
from system_overseer.llm_client import LLMClient
from system_overseer.dialogue_manager import DialogueManager
from system_overseer.personality_system import PersonalitySystem
from system_overseer.telegram_integration import TelegramIntegration
from system_overseer.plugin_manager import PluginManager

class SystemOverseerService:
    \"\"\"System Overseer Service.\"\"\"
    
    def __init__(self, config_dir: str = "./config", data_dir: str = "./data"):
        \"\"\"Initialize System Overseer Service.
        
        Args:
            config_dir: Configuration directory
            data_dir: Data directory
        \"\"\"
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.running = False
        
        # Create directories
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Create components
        self.module_registry = ModuleRegistry()
        self.config_registry = ConfigRegistry(
            config_dir=config_dir
        )
        self.event_bus = EventBus()
        
        # Create system core
        self.system_core = SystemCore(
            config_registry=self.config_registry,
            event_bus=self.event_bus,
            data_dir=os.path.join(data_dir, "system")
        )
        
        # Create LLM client
        self.llm_client = LLMClient()
        
        # Create dialogue manager
        self.dialogue_manager = DialogueManager(
            llm_client=self.llm_client,
            data_dir=os.path.join(data_dir, "dialogue")
        )
        
        # Create personality system
        self.personality_system = PersonalitySystem(
            data_dir=os.path.join(data_dir, "personality")
        )
        
        # Create plugin manager
        self.plugin_manager = PluginManager(
            system_core=self.system_core,
            data_dir=os.path.join(data_dir, "plugins")
        )
        
        # Create Telegram integration
        self.telegram_integration = TelegramIntegration(
            dialogue_manager=self.dialogue_manager,
            system_core=self.system_core
        )
        
        # Register components with module registry
        self.module_registry.register_service("config_registry", self.config_registry)
        self.module_registry.register_service("event_bus", self.event_bus)
        self.module_registry.register_service("llm_client", self.llm_client)
        self.module_registry.register_service("dialogue_manager", self.dialogue_manager)
        self.module_registry.register_service("personality_system", self.personality_system)
        self.module_registry.register_service("plugin_manager", self.plugin_manager)
        self.module_registry.register_service("telegram_integration", self.telegram_integration)
    
    def initialize(self):
        \"\"\"Initialize the service.\"\"\"
        logger.info("Initializing System Overseer Service")
        
        # Initialize system core
        self.system_core.initialize()
        
        # Initialize Telegram integration
        self.telegram_integration.initialize()
        
        # Load plugins
        self.plugin_manager.load_plugins()
        
        logger.info("System Overseer Service initialized")
    
    def start(self):
        \"\"\"Start the service.\"\"\"
        logger.info("Starting System Overseer Service")
        
        # Set running flag
        self.running = True
        
        # Start Telegram integration
        self.telegram_integration.start()
        
        # Start plugins
        self.plugin_manager.start_plugins()
        
        logger.info("System Overseer Service started")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        # Main loop
        try:
            while self.running:
                # Process events
                self.event_bus.process_events()
                
                # Sleep
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()
    
    def stop(self):
        \"\"\"Stop the service.\"\"\"
        logger.info("Stopping System Overseer Service")
        
        # Set running flag
        self.running = False
        
        # Stop plugins
        self.plugin_manager.stop_plugins()
        
        # Stop Telegram integration
        self.telegram_integration.stop()
        
        logger.info("System Overseer Service stopped")
    
    def handle_signal(self, sig, frame):
        \"\"\"Handle signal.
        
        Args:
            sig: Signal number
            frame: Frame
        \"\"\"
        logger.info(f"Signal received: {sig}")
        self.stop()


def main():
    \"\"\"Main function.\"\"\"
    # Parse arguments
    parser = argparse.ArgumentParser(description="System Overseer Service")
    parser.add_argument("--config-dir", default="./config", help="Configuration directory")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    args = parser.parse_args()
    
    # Create service
    service = SystemOverseerService(
        config_dir=args.config_dir,
        data_dir=args.data_dir
    )
    
    # Initialize service
    service.initialize()
    
    # Start service
    service.start()


if __name__ == "__main__":
    main()
"""
    
    with open("system_overseer_service.py", "w") as f:
        f.write(service_script)
    
    # Make executable
    os.chmod("system_overseer_service.py", 0o755)
    
    logger.info("Service script created successfully")
    return True

def deploy_system_overseer():
    """Deploy System Overseer with real services."""
    logger.info("Deploying System Overseer with real services...")
    
    # Check if service is already running
    try:
        output = subprocess.check_output(["pgrep", "-f", "python.*system_overseer_service.py"]).decode().strip()
        if output:
            logger.info("System Overseer service is already running. Stopping it...")
            subprocess.run(["pkill", "-f", "python.*system_overseer_service.py"])
            time.sleep(2)
    except subprocess.CalledProcessError:
        pass
    
    # Start service
    logger.info("Starting System Overseer service...")
    
    try:
        subprocess.Popen(
            ["python", "system_overseer_service.py", "--config-dir", "./config", "--data-dir", "./data"],
            stdout=open("logs/service_stdout.log", "w"),
            stderr=open("logs/service_stderr.log", "w")
        )
        
        # Wait for service to start
        time.sleep(5)
        
        # Check if service is running
        try:
            output = subprocess.check_output(["pgrep", "-f", "python.*system_overseer_service.py"]).decode().strip()
            if output:
                logger.info("System Overseer service started successfully!")
                return True
            else:
                logger.error("System Overseer service failed to start")
                return False
        except subprocess.CalledProcessError:
            logger.error("System Overseer service failed to start")
            return False
    except Exception as e:
        logger.error(f"Failed to start System Overseer service: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting System Overseer deployment...")
    
    # Create directories
    if not create_directories():
        logger.error("Failed to create directories")
        return False
    
    # Create configuration files
    if not create_config_files():
        logger.error("Failed to create configuration files")
        return False
    
    # Create requirements file
    if not create_requirements_file():
        logger.error("Failed to create requirements file")
        return False
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Create plugin files
    if not create_plugin_files():
        logger.error("Failed to create plugin files")
        return False
    
    # Create service script
    if not create_service_script():
        logger.error("Failed to create service script")
        return False
    
    # Deploy System Overseer
    if not deploy_system_overseer():
        logger.error("Failed to deploy System Overseer")
        return False
    
    logger.info("System Overseer deployment completed successfully!")
    return True

if __name__ == "__main__":
    main()
