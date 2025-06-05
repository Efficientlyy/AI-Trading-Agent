#!/usr/bin/env python
"""
Integration of LLM-powered Conversation and Plugin Manager with Telegram Bot

This module integrates the dialogue manager, plugin system, and Telegram bot
to create a unified conversational interface for the System Overseer.
"""

import os
import sys
import json
import time
import logging
import threading
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime
import uuid

# Import core components
# In a real implementation, these would be proper imports from the project structure
# For demonstration purposes, we're assuming these modules exist
try:
    from telegram_bot_module import TelegramBotModule
    from dialogue_manager import DialogueManager, DialogueContext, UserProfile, DialoguePlugin
    from config_registry_design import ConfigRegistry
    from event_bus_design import EventBus
    from analytics_plugin_framework import AnalyticsPlugin
except ImportError:
    # Create placeholder classes for type hints if imports fail
    class TelegramBotModule: pass
    class DialogueManager: pass
    class DialogueContext: pass
    class UserProfile: pass
    class DialoguePlugin: pass
    class ConfigRegistry: pass
    class EventBus: pass
    class AnalyticsPlugin: pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_overseer")

class ServiceLocator:
    """Simple service locator for system components"""
    
    def __init__(self):
        """Initialize service locator"""
        self.services = {}
    
    def register_service(self, service_name: str, service: Any):
        """Register a service
        
        Args:
            service_name: Service name
            service: Service instance
        """
        self.services[service_name] = service
        logger.info(f"Service registered: {service_name}")
    
    def get_service(self, service_name: str) -> Any:
        """Get a service
        
        Args:
            service_name: Service name
            
        Returns:
            Any: Service instance
            
        Raises:
            KeyError: If service not found
        """
        if service_name not in self.services:
            raise KeyError(f"Service not found: {service_name}")
        return self.services[service_name]
    
    def has_service(self, service_name: str) -> bool:
        """Check if service exists
        
        Args:
            service_name: Service name
            
        Returns:
            bool: True if service exists
        """
        return service_name in self.services


class LLMClient:
    """Client for interacting with LLM services"""
    
    def __init__(
        self,
        config_registry: ConfigRegistry = None,
        default_model: str = "openai/gpt-3.5-turbo",
        default_temperature: float = 0.7
    ):
        """Initialize LLM client
        
        Args:
            config_registry: Configuration registry
            default_model: Default model to use
            default_temperature: Default temperature for generation
        """
        self.config_registry = config_registry
        self.default_model = default_model
        self.default_temperature = default_temperature
        
        # Initialize from config if provided
        if self.config_registry:
            self._init_from_config()
        
        logger.info("LLMClient initialized")
    
    def _init_from_config(self):
        """Initialize from configuration registry"""
        if not self.config_registry:
            return
        
        try:
            # Get LLM client configuration
            self.default_model = self.config_registry.get_parameter(
                "llm_client", "default_model", self.default_model)
            
            self.default_temperature = self.config_registry.get_parameter(
                "llm_client", "default_temperature", self.default_temperature)
            
            logger.info("LLMClient initialized from configuration")
        except Exception as e:
            logger.error(f"Error initializing LLMClient from configuration: {e}")
    
    def generate(
        self,
        prompt: str = None,
        messages: List[Dict[str, str]] = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate text from LLM
        
        Args:
            prompt: Text prompt (for single-turn)
            messages: Message list (for multi-turn)
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
            
        Raises:
            ValueError: If neither prompt nor messages provided
        """
        # Use default values if not provided
        model = model or self.default_model
        temperature = temperature or self.default_temperature
        
        # Check if we have a prompt or messages
        if not prompt and not messages:
            raise ValueError("Either prompt or messages must be provided")
        
        # In a real implementation, this would call an actual LLM API
        # For demonstration purposes, we'll just return a placeholder response
        if messages:
            # Get the last user message
            last_user_message = None
            for msg in reversed(messages):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break
            
            if "market" in last_user_message.lower():
                return "The market is currently showing mixed signals. Bitcoin is up 2.3% in the last 24 hours, while Ethereum is down 0.8%. Trading volume is above average today."
            elif "help" in last_user_message.lower():
                return "I'm your trading assistant. I can help you monitor the market, analyze trends, and manage your trading settings. What would you like to know?"
            else:
                return "I understand your message. How else can I assist you with your trading today?"
        else:
            # Single-turn prompt
            if "market" in prompt.lower():
                return "The market is currently showing mixed signals. Bitcoin is up 2.3% in the last 24 hours, while Ethereum is down 0.8%. Trading volume is above average today."
            elif "help" in prompt.lower():
                return "I'm your trading assistant. I can help you monitor the market, analyze trends, and manage your trading settings. What would you like to know?"
            else:
                return "I understand your request. How else can I assist you with your trading today?"
    
    async def generate_async(
        self,
        prompt: str = None,
        messages: List[Dict[str, str]] = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate text from LLM asynchronously
        
        Args:
            prompt: Text prompt (for single-turn)
            messages: Message list (for multi-turn)
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        # This is a simple wrapper around the synchronous method
        # In a real implementation, this would use an async API client
        return self.generate(
            prompt=prompt,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )


class PluginManager:
    """Manager for system plugins"""
    
    def __init__(
        self,
        config_registry: ConfigRegistry = None,
        event_bus: EventBus = None,
        service_locator: ServiceLocator = None,
        plugin_dirs: List[str] = None
    ):
        """Initialize plugin manager
        
        Args:
            config_registry: Configuration registry
            event_bus: Event bus
            service_locator: Service locator
            plugin_dirs: Plugin directories
        """
        self.config_registry = config_registry
        self.event_bus = event_bus
        self.service_locator = service_locator
        self.plugin_dirs = plugin_dirs or []
        
        # Plugin storage
        self.plugins = {}  # plugin_id -> plugin
        self.plugin_types = {}  # plugin_type -> [plugin_id]
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("PluginManager initialized")
    
    def register_plugin(self, plugin: Union[DialoguePlugin, AnalyticsPlugin]) -> bool:
        """Register a plugin
        
        Args:
            plugin: Plugin to register
            
        Returns:
            bool: True if registration successful
        """
        with self.lock:
            plugin_id = plugin.plugin_id
            
            # Check if plugin already exists
            if plugin_id in self.plugins:
                logger.warning(f"Plugin {plugin_id} already registered")
                return False
            
            # Determine plugin type
            if isinstance(plugin, DialoguePlugin):
                plugin_type = "dialogue"
            elif isinstance(plugin, AnalyticsPlugin):
                plugin_type = "analytics"
            else:
                plugin_type = "unknown"
            
            # Store plugin
            self.plugins[plugin_id] = plugin
            
            # Add to plugin type mapping
            if plugin_type not in self.plugin_types:
                self.plugin_types[plugin_type] = []
            self.plugin_types[plugin_type].append(plugin_id)
            
            logger.info(f"Plugin {plugin_id} registered as {plugin_type}")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="plugin_manager.plugin_registered",
                    data={
                        "plugin_id": plugin_id,
                        "plugin_type": plugin_type,
                        "name": getattr(plugin, "name", plugin_id),
                        "version": getattr(plugin, "version", "unknown")
                    }
                )
            
            return True
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister a plugin
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            bool: True if unregistration successful
        """
        with self.lock:
            # Check if plugin exists
            if plugin_id not in self.plugins:
                logger.warning(f"Plugin {plugin_id} not registered")
                return False
            
            # Get plugin
            plugin = self.plugins[plugin_id]
            
            # Remove from plugin type mapping
            for plugin_type, plugin_ids in self.plugin_types.items():
                if plugin_id in plugin_ids:
                    plugin_ids.remove(plugin_id)
            
            # Remove plugin
            del self.plugins[plugin_id]
            
            logger.info(f"Plugin {plugin_id} unregistered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="plugin_manager.plugin_unregistered",
                    data={
                        "plugin_id": plugin_id
                    }
                )
            
            return True
    
    def get_plugin(self, plugin_id: str) -> Optional[Union[DialoguePlugin, AnalyticsPlugin]]:
        """Get a plugin by ID
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            Union[DialoguePlugin, AnalyticsPlugin]: Plugin instance or None if not found
        """
        return self.plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[Union[DialoguePlugin, AnalyticsPlugin]]:
        """Get plugins by type
        
        Args:
            plugin_type: Plugin type
            
        Returns:
            list: List of plugin instances
        """
        plugin_ids = self.plugin_types.get(plugin_type, [])
        return [self.plugins[pid] for pid in plugin_ids if pid in self.plugins]
    
    def discover_plugins(self) -> int:
        """Discover plugins from plugin directories
        
        Returns:
            int: Number of plugins discovered
        """
        count = 0
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                logger.warning(f"Plugin directory not found: {plugin_dir}")
                continue
            
            # In a real implementation, this would dynamically load Python modules
            # For demonstration purposes, we'll just log that we're scanning
            logger.info(f"Scanning plugin directory: {plugin_dir}")
            
            # Placeholder for plugin discovery logic
            # This would typically involve:
            # 1. Finding Python modules in the directory
            # 2. Loading each module
            # 3. Looking for plugin classes
            # 4. Instantiating and registering plugins
            
            # For now, just increment count as a placeholder
            count += 1
        
        return count
    
    def initialize_plugins(self) -> int:
        """Initialize all registered plugins
        
        Returns:
            int: Number of plugins initialized
        """
        count = 0
        
        for plugin_id, plugin in self.plugins.items():
            try:
                # Initialize plugin based on its type
                if isinstance(plugin, DialoguePlugin) and hasattr(plugin, "initialize"):
                    # Initialize dialogue plugin
                    if self.service_locator and self.service_locator.has_service("dialogue_manager"):
                        dialogue_manager = self.service_locator.get_service("dialogue_manager")
                        plugin.initialize(
                            dialogue_manager=dialogue_manager,
                            config_registry=self.config_registry,
                            event_bus=self.event_bus,
                            service_locator=self.service_locator
                        )
                        count += 1
                
                elif isinstance(plugin, AnalyticsPlugin) and hasattr(plugin, "initialize"):
                    # Initialize analytics plugin
                    plugin.initialize(
                        config_registry=self.config_registry,
                        event_bus=self.event_bus,
                        service_locator=self.service_locator
                    )
                    count += 1
            
            except Exception as e:
                logger.error(f"Error initializing plugin {plugin_id}: {e}")
        
        logger.info(f"Initialized {count} plugins")
        return count
    
    def start_plugins(self) -> int:
        """Start all registered plugins
        
        Returns:
            int: Number of plugins started
        """
        count = 0
        
        for plugin_id, plugin in self.plugins.items():
            try:
                if hasattr(plugin, "start"):
                    plugin.start()
                    count += 1
            except Exception as e:
                logger.error(f"Error starting plugin {plugin_id}: {e}")
        
        logger.info(f"Started {count} plugins")
        return count
    
    def stop_plugins(self) -> int:
        """Stop all registered plugins
        
        Returns:
            int: Number of plugins stopped
        """
        count = 0
        
        for plugin_id, plugin in self.plugins.items():
            try:
                if hasattr(plugin, "stop"):
                    plugin.stop()
                    count += 1
            except Exception as e:
                logger.error(f"Error stopping plugin {plugin_id}: {e}")
        
        logger.info(f"Stopped {count} plugins")
        return count


class TelegramDialogueAdapter:
    """Adapter between Telegram bot and dialogue manager"""
    
    def __init__(
        self,
        telegram_bot: TelegramBotModule,
        dialogue_manager: DialogueManager,
        config_registry: ConfigRegistry = None,
        event_bus: EventBus = None
    ):
        """Initialize adapter
        
        Args:
            telegram_bot: Telegram bot module
            dialogue_manager: Dialogue manager
            config_registry: Configuration registry
            event_bus: Event bus
        """
        self.telegram_bot = telegram_bot
        self.dialogue_manager = dialogue_manager
        self.config_registry = config_registry
        self.event_bus = event_bus
        
        # User ID mapping (Telegram user ID -> dialogue user ID)
        self.user_mapping = {}
        
        # Session mapping (Telegram user ID -> dialogue session ID)
        self.session_mapping = {}
        
        logger.info("TelegramDialogueAdapter initialized")
    
    def register_handlers(self):
        """Register message handlers with Telegram bot"""
        # Register fallback message handler
        self.telegram_bot.register_intent(
            intent_id="dialogue_fallback",
            patterns=[r".*"],  # Match any message
            handler=self.handle_message,
            description="Fallback handler for dialogue manager",
            examples=["How is the market?", "What's the status of my trades?"],
            plugin_id="telegram_dialogue_adapter",
            enabled=True
        )
        
        # Register help command
        self.telegram_bot.register_command(
            command="help",
            handler=self.handle_help_command,
            description="Show help information",
            help_text="Show available commands and how to interact with the system",
            plugin_id="telegram_dialogue_adapter"
        )
        
        # Register status command
        self.telegram_bot.register_command(
            command="status",
            handler=self.handle_status_command,
            description="Show system status",
            help_text="Show current status of the trading system",
            plugin_id="telegram_dialogue_adapter"
        )
        
        # Register settings command
        self.telegram_bot.register_command(
            command="settings",
            handler=self.handle_settings_command,
            description="Manage settings",
            help_text="View and change system settings",
            plugin_id="telegram_dialogue_adapter"
        )
        
        # Register personality command
        self.telegram_bot.register_command(
            command="personality",
            handler=self.handle_personality_command,
            description="Change assistant personality",
            help_text="Change the personality of the assistant (professional, friendly, concise, technical, educational)",
            plugin_id="telegram_dialogue_adapter"
        )
        
        logger.info("TelegramDialogueAdapter handlers registered")
    
    async def handle_message(self, update, context, match):
        """Handle message via dialogue manager
        
        Args:
            update: Telegram update
            context: Telegram context
            match: Regex match
            
        Returns:
            str: Response message
        """
        # Get user and chat IDs
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # Get message text
        text = update.message.text
        
        # Map Telegram user ID to dialogue user ID
        dialogue_user_id = self.user_mapping.get(user_id, user_id)
        
        # Get dialogue session ID
        session_id = self.session_mapping.get(user_id)
        
        # Process message with dialogue manager
        response = self.dialogue_manager.process_message(
            user_id=dialogue_user_id,
            message=text,
            session_id=session_id,
            metadata={
                "telegram_user_id": user_id,
                "telegram_chat_id": chat_id,
                "source": "telegram"
            }
        )
        
        # Update session mapping
        context = self.dialogue_manager.get_or_create_context(dialogue_user_id, session_id)
        self.session_mapping[user_id] = context.session_id
        
        # Publish event if event bus is available
        if self.event_bus:
            self.event_bus.publish(
                event_type="telegram_dialogue.message_processed",
                data={
                    "telegram_user_id": user_id,
                    "dialogue_user_id": dialogue_user_id,
                    "message": text,
                    "response": response
                }
            )
        
        return response
    
    async def handle_help_command(self, update, context):
        """Handle help command
        
        Args:
            update: Telegram update
            context: Telegram context
        """
        # Get user ID
        user_id = update.effective_user.id
        
        # Map Telegram user ID to dialogue user ID
        dialogue_user_id = self.user_mapping.get(user_id, user_id)
        
        # Get user profile
        profile = self.dialogue_manager.get_or_create_user_profile(dialogue_user_id)
        
        # Generate help message based on user profile
        help_text = (
            "🤖 *Trading System Assistant*\n\n"
            "I'm your AI assistant for the trading system. You can talk to me naturally "
            "or use commands for specific functions.\n\n"
            "*Available Commands:*\n"
            "• /help - Show this help message\n"
            "• /status - Show system status\n"
            "• /settings - Manage system settings\n"
            "• /personality - Change my personality\n"
            "• /pairs - Show active trading pairs\n"
            "• /add_pair SYMBOL - Add trading pair\n"
            "• /remove_pair SYMBOL - Remove trading pair\n"
            "• /notifications LEVEL - Set notification level\n\n"
            "*Natural Language:*\n"
            "You can also just chat with me naturally. Ask about the market, "
            "your trades, or how to use the system."
        )
        
        # Send help message
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def handle_status_command(self, update, context):
        """Handle status command
        
        Args:
            update: Telegram update
            context: Telegram context
        """
        # Get user ID
        user_id = update.effective_user.id
        
        # Map Telegram user ID to dialogue user ID
        dialogue_user_id = self.user_mapping.get(user_id, user_id)
        
        # Get user profile
        profile = self.dialogue_manager.get_or_create_user_profile(dialogue_user_id)
        
        # Get active trading pairs
        trading_pairs = profile.get_preference("trading_pairs", ["BTCUSDC"])
        
        # Generate status message
        status_text = (
            "📊 *System Status*\n\n"
            "• System: 🟢 Online\n"
            "• Market Data: 🟢 Connected\n"
            "• Trading: 🟢 Active\n"
            f"• Active Pairs: {', '.join(trading_pairs)}\n"
            "• Notifications: 🟢 Enabled\n\n"
            "*Market Overview:*\n"
            "• BTC: $48,256 (+2.3%)\n"
            "• ETH: $3,150 (-0.8%)\n"
            "• SOL: $102 (+1.5%)\n\n"
            "*Recent Activity:*\n"
            "• Last signal: BTC buy signal (15 min ago)\n"
            "• Last trade: ETH buy at $3,120 (30 min ago)"
        )
        
        # Send status message
        await update.message.reply_text(status_text, parse_mode="Markdown")
    
    async def handle_settings_command(self, update, context):
        """Handle settings command
        
        Args:
            update: Telegram update
            context: Telegram context
        """
        # Get user ID
        user_id = update.effective_user.id
        
        # Map Telegram user ID to dialogue user ID
        dialogue_user_id = self.user_mapping.get(user_id, user_id)
        
        # Get user profile
        profile = self.dialogue_manager.get_or_create_user_profile(dialogue_user_id)
        
        # Get settings
        notification_level = profile.get_preference("notification_level", "important")
        trading_pairs = profile.get_preference("trading_pairs", ["BTCUSDC"])
        personality = profile.get_preference("personality_profile", "friendly")
        
        # Generate settings message
        settings_text = (
            "⚙️ *System Settings*\n\n"
            f"• Notification Level: {notification_level}\n"
            f"• Active Trading Pairs: {', '.join(trading_pairs)}\n"
            f"• Assistant Personality: {personality}\n\n"
            "*Change Settings:*\n"
            "• /notifications [all|important|trades|signals|none]\n"
            "• /add_pair SYMBOL\n"
            "• /remove_pair SYMBOL\n"
            "• /personality [professional|friendly|concise|technical|educational]"
        )
        
        # Send settings message
        await update.message.reply_text(settings_text, parse_mode="Markdown")
    
    async def handle_personality_command(self, update, context):
        """Handle personality command
        
        Args:
            update: Telegram update
            context: Telegram context
        """
        # Get user ID
        user_id = update.effective_user.id
        
        # Map Telegram user ID to dialogue user ID
        dialogue_user_id = self.user_mapping.get(user_id, user_id)
        
        # Get user profile
        profile = self.dialogue_manager.get_or_create_user_profile(dialogue_user_id)
        
        # Get current personality
        current_personality = profile.get_preference("personality_profile", "friendly")
        
        # Check if a personality was specified
        if context.args and len(context.args) > 0:
            # Get specified personality
            personality = context.args[0].lower()
            
            # Check if valid personality
            valid_personalities = ["professional", "friendly", "concise", "technical", "educational"]
            if personality in valid_personalities:
                # Update personality
                profile.update_preference("personality_profile", personality)
                
                # Set personality in dialogue manager
                self.dialogue_manager.set_default_personality(personality)
                
                # Send confirmation
                await update.message.reply_text(
                    f"Personality changed to *{personality}*. I'll adjust my communication style accordingly.",
                    parse_mode="Markdown"
                )
            else:
                # Send error message
                await update.message.reply_text(
                    f"Invalid personality: {personality}\n\n"
                    f"Valid options: {', '.join(valid_personalities)}",
                    parse_mode="Markdown"
                )
        else:
            # Send current personality and options
            await update.message.reply_text(
                f"Current personality: *{current_personality}*\n\n"
                "Available personalities:\n"
                "• *professional* - Formal, precise, and business-oriented\n"
                "• *friendly* - Warm, approachable, and conversational\n"
                "• *concise* - Brief, direct, and to-the-point\n"
                "• *technical* - Detailed, technical, and comprehensive\n"
                "• *educational* - Explanatory, patient, and informative\n\n"
                "To change, use: /personality [option]",
                parse_mode="Markdown"
            )


class SystemOverseer:
    """Main system overseer class integrating all components"""
    
    def __init__(
        self,
        config_file: str = None,
        persistence_dir: str = None,
        plugin_dirs: List[str] = None
    ):
        """Initialize system overseer
        
        Args:
            config_file: Configuration file path
            persistence_dir: Persistence directory path
            plugin_dirs: Plugin directories
        """
        self.config_file = config_file
        self.persistence_dir = persistence_dir or "./data"
        self.plugin_dirs = plugin_dirs or ["./plugins"]
        
        # Create persistence directory
        os.makedirs(self.persistence_dir, exist_ok=True)
        
        # Core components
        self.config_registry = None
        self.event_bus = None
        self.service_locator = None
        self.llm_client = None
        self.plugin_manager = None
        self.dialogue_manager = None
        self.telegram_bot = None
        self.telegram_adapter = None
        
        # Component threads
        self.telegram_thread = None
        
        # Running flag
        self.running = False
        
        logger.info("SystemOverseer initialized")
    
    def initialize(self) -> bool:
        """Initialize system components
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Create core components
            self.config_registry = self._create_config_registry()
            self.event_bus = self._create_event_bus()
            self.service_locator = self._create_service_locator()
            
            # Register core services
            self.service_locator.register_service("config_registry", self.config_registry)
            self.service_locator.register_service("event_bus", self.event_bus)
            
            # Create LLM client
            self.llm_client = self._create_llm_client()
            self.service_locator.register_service("llm_client", self.llm_client)
            
            # Create plugin manager
            self.plugin_manager = self._create_plugin_manager()
            self.service_locator.register_service("plugin_manager", self.plugin_manager)
            
            # Create dialogue manager
            self.dialogue_manager = self._create_dialogue_manager()
            self.service_locator.register_service("dialogue_manager", self.dialogue_manager)
            
            # Create Telegram bot
            self.telegram_bot = self._create_telegram_bot()
            self.service_locator.register_service("telegram_bot", self.telegram_bot)
            
            # Create Telegram adapter
            self.telegram_adapter = self._create_telegram_adapter()
            
            # Register handlers
            self.telegram_adapter.register_handlers()
            
            # Discover and initialize plugins
            self.plugin_manager.discover_plugins()
            self.plugin_manager.initialize_plugins()
            
            logger.info("SystemOverseer initialization complete")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing SystemOverseer: {e}")
            return False
    
    def start(self) -> bool:
        """Start system components
        
        Returns:
            bool: True if start successful
        """
        try:
            # Check if already running
            if self.running:
                logger.warning("SystemOverseer already running")
                return False
            
            # Start plugins
            self.plugin_manager.start_plugins()
            
            # Start Telegram bot in a separate thread
            self.telegram_thread = threading.Thread(
                target=self._run_telegram_bot,
                name="TelegramBot",
                daemon=True
            )
            self.telegram_thread.start()
            
            # Set running flag
            self.running = True
            
            logger.info("SystemOverseer started")
            
            # Publish event
            self.event_bus.publish(
                event_type="system.started",
                data={
                    "timestamp": time.time()
                }
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error starting SystemOverseer: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop system components
        
        Returns:
            bool: True if stop successful
        """
        try:
            # Check if running
            if not self.running:
                logger.warning("SystemOverseer not running")
                return False
            
            # Publish event
            self.event_bus.publish(
                event_type="system.stopping",
                data={
                    "timestamp": time.time()
                }
            )
            
            # Stop Telegram bot
            if self.telegram_bot:
                self.telegram_bot.stop()
            
            # Stop plugins
            if self.plugin_manager:
                self.plugin_manager.stop_plugins()
            
            # Stop dialogue manager
            if self.dialogue_manager:
                self.dialogue_manager.stop()
            
            # Clear running flag
            self.running = False
            
            logger.info("SystemOverseer stopped")
            
            return True
        
        except Exception as e:
            logger.error(f"Error stopping SystemOverseer: {e}")
            return False
    
    def _create_config_registry(self) -> ConfigRegistry:
        """Create configuration registry
        
        Returns:
            ConfigRegistry: Configuration registry instance
        """
        # In a real implementation, this would create a proper ConfigRegistry
        # For demonstration purposes, we'll create a placeholder
        logger.info("Creating ConfigRegistry")
        
        # Placeholder for ConfigRegistry
        class DummyConfigRegistry:
            def __init__(self):
                self.params = {}
            
            def register_parameter(self, module_id, param_id, default_value, **kwargs):
                key = f"{module_id}.{param_id}"
                if key not in self.params:
                    self.params[key] = default_value
                return True
            
            def get_parameter(self, module_id, param_id, default=None):
                key = f"{module_id}.{param_id}"
                return self.params.get(key, default)
            
            def set_parameter(self, module_id, param_id, value):
                key = f"{module_id}.{param_id}"
                self.params[key] = value
                return True
        
        return DummyConfigRegistry()
    
    def _create_event_bus(self) -> EventBus:
        """Create event bus
        
        Returns:
            EventBus: Event bus instance
        """
        # In a real implementation, this would create a proper EventBus
        # For demonstration purposes, we'll create a placeholder
        logger.info("Creating EventBus")
        
        # Placeholder for EventBus
        class DummyEventBus:
            def __init__(self):
                self.subscribers = {}
            
            def publish(self, event_type, data, publisher_id=None, priority=0):
                logger.info(f"Event published: {event_type}")
                
                # Call subscribers
                if event_type in self.subscribers:
                    for subscriber_id, callback in self.subscribers[event_type]:
                        try:
                            callback({
                                "event_type": event_type,
                                "data": data,
                                "publisher_id": publisher_id,
                                "timestamp": time.time()
                            })
                        except Exception as e:
                            logger.error(f"Error in subscriber {subscriber_id}: {e}")
                
                return str(uuid.uuid4())
            
            def subscribe(self, subscriber_id, event_type, callback, filter_func=None):
                if event_type not in self.subscribers:
                    self.subscribers[event_type] = []
                
                self.subscribers[event_type].append((subscriber_id, callback))
                logger.info(f"Subscriber {subscriber_id} subscribed to {event_type}")
                
                return str(uuid.uuid4())
            
            def unsubscribe(self, subscription_id):
                # In this simple implementation, we don't track subscription IDs
                return True
        
        return DummyEventBus()
    
    def _create_service_locator(self) -> ServiceLocator:
        """Create service locator
        
        Returns:
            ServiceLocator: Service locator instance
        """
        logger.info("Creating ServiceLocator")
        return ServiceLocator()
    
    def _create_llm_client(self) -> LLMClient:
        """Create LLM client
        
        Returns:
            LLMClient: LLM client instance
        """
        logger.info("Creating LLMClient")
        return LLMClient(config_registry=self.config_registry)
    
    def _create_plugin_manager(self) -> PluginManager:
        """Create plugin manager
        
        Returns:
            PluginManager: Plugin manager instance
        """
        logger.info("Creating PluginManager")
        return PluginManager(
            config_registry=self.config_registry,
            event_bus=self.event_bus,
            service_locator=self.service_locator,
            plugin_dirs=self.plugin_dirs
        )
    
    def _create_dialogue_manager(self) -> DialogueManager:
        """Create dialogue manager
        
        Returns:
            DialogueManager: Dialogue manager instance
        """
        logger.info("Creating DialogueManager")
        
        # In a real implementation, this would create a proper DialogueManager
        # For demonstration purposes, we'll create a placeholder
        dialogue_dir = os.path.join(self.persistence_dir, "dialogue")
        os.makedirs(dialogue_dir, exist_ok=True)
        
        # Placeholder for DialogueManager
        class DummyDialogueManager:
            def __init__(self):
                self.contexts = {}
                self.profiles = {}
                self.personality_profiles = {
                    "professional": {"name": "Professional"},
                    "friendly": {"name": "Friendly"},
                    "concise": {"name": "Concise"},
                    "technical": {"name": "Technical"},
                    "educational": {"name": "Educational"}
                }
                self.default_personality = "friendly"
            
            def get_or_create_context(self, user_id, session_id=None):
                if not session_id:
                    session_id = f"session_{user_id}"
                
                if session_id not in self.contexts:
                    self.contexts[session_id] = {
                        "user_id": user_id,
                        "session_id": session_id,
                        "history": []
                    }
                
                return type("DialogueContext", (), self.contexts[session_id])
            
            def get_or_create_user_profile(self, user_id):
                if user_id not in self.profiles:
                    self.profiles[user_id] = {
                        "user_id": user_id,
                        "preferences": {
                            "personality_profile": "friendly",
                            "notification_level": "important",
                            "trading_pairs": ["BTCUSDC"]
                        },
                        "get_preference": lambda k, d=None: self.profiles[user_id]["preferences"].get(k, d),
                        "update_preference": lambda k, v: self.profiles[user_id]["preferences"].update({k: v})
                    }
                
                return type("UserProfile", (), self.profiles[user_id])
            
            def process_message(self, user_id, message, session_id=None, metadata=None):
                # Get LLM client
                llm_client = service_locator.get_service("llm_client")
                
                # Generate response
                response = llm_client.generate(prompt=message)
                
                return response
            
            def set_default_personality(self, profile_id):
                if profile_id in self.personality_profiles:
                    self.default_personality = profile_id
                    return True
                return False
            
            def stop(self):
                pass
        
        # Create dummy dialogue manager
        service_locator = self.service_locator
        return DummyDialogueManager()
    
    def _create_telegram_bot(self) -> TelegramBotModule:
        """Create Telegram bot
        
        Returns:
            TelegramBotModule: Telegram bot instance
        """
        logger.info("Creating TelegramBotModule")
        
        # Get Telegram token from config
        token = self.config_registry.get_parameter("telegram_bot", "token")
        
        # Get admin and allowed user IDs from config
        admin_user_ids_str = self.config_registry.get_parameter("telegram_bot", "admin_user_ids", "")
        allowed_user_ids_str = self.config_registry.get_parameter("telegram_bot", "allowed_user_ids", "")
        
        # Parse user IDs
        admin_user_ids = [int(uid.strip()) for uid in admin_user_ids_str.split(",") if uid.strip()] if admin_user_ids_str else []
        allowed_user_ids = [int(uid.strip()) for uid in allowed_user_ids_str.split(",") if uid.strip()] if allowed_user_ids_str else []
        
        # Create Telegram bot
        return TelegramBotModule(
            config_registry=self.config_registry,
            event_bus=self.event_bus,
            service_locator=self.service_locator,
            token=token,
            admin_user_ids=admin_user_ids,
            allowed_user_ids=allowed_user_ids,
            persistence_file=os.path.join(self.persistence_dir, "telegram_states.json")
        )
    
    def _create_telegram_adapter(self) -> TelegramDialogueAdapter:
        """Create Telegram adapter
        
        Returns:
            TelegramDialogueAdapter: Telegram adapter instance
        """
        logger.info("Creating TelegramDialogueAdapter")
        return TelegramDialogueAdapter(
            telegram_bot=self.telegram_bot,
            dialogue_manager=self.dialogue_manager,
            config_registry=self.config_registry,
            event_bus=self.event_bus
        )
    
    def _run_telegram_bot(self):
        """Run Telegram bot in event loop"""
        try:
            # Create event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Start Telegram bot
            loop.run_until_complete(self.telegram_bot.start())
        
        except Exception as e:
            logger.error(f"Error running Telegram bot: {e}")


# Example usage
if __name__ == "__main__":
    # Create system overseer
    overseer = SystemOverseer(
        config_file="config.json",
        persistence_dir="./data",
        plugin_dirs=["./plugins"]
    )
    
    # Initialize system
    if overseer.initialize():
        # Start system
        overseer.start()
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            # Stop system
            overseer.stop()
    
    else:
        logger.error("Failed to initialize SystemOverseer")
