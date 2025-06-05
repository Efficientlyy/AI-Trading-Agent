#!/usr/bin/env python
"""
System Overseer Service Script

This script runs the System Overseer as a background service,
connecting to Telegram and providing continuous monitoring and control.
"""

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
    """System Overseer Service."""
    
    def __init__(self, config_dir: str = "./config", data_dir: str = "./data"):
        """Initialize System Overseer Service.
        
        Args:
            config_dir: Configuration directory
            data_dir: Data directory
        """
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
        """Initialize the service."""
        logger.info("Initializing System Overseer Service")
        
        # Initialize system core
        self.system_core.initialize()
        
        # Initialize Telegram integration
        self.telegram_integration.initialize()
        
        # Load plugins
        self.plugin_manager.load_plugins()
        
        logger.info("System Overseer Service initialized")
    
    def start(self):
        """Start the service."""
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
        """Stop the service."""
        logger.info("Stopping System Overseer Service")
        
        # Set running flag
        self.running = False
        
        # Stop plugins
        self.plugin_manager.stop_plugins()
        
        # Stop Telegram integration
        self.telegram_integration.stop()
        
        logger.info("System Overseer Service stopped")
    
    def handle_signal(self, sig, frame):
        """Handle signal.
        
        Args:
            sig: Signal number
            frame: Frame
        """
        logger.info(f"Signal received: {sig}")
        self.stop()


def main():
    """Main function."""
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
