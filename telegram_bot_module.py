#!/usr/bin/env python
"""
Modular Telegram Bot Framework for System Overseer

This module implements a modular Telegram bot with support for plugin-based
command registration and natural language conversation handling.
"""

import os
import re
import json
import logging
import threading
import asyncio
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime
import uuid

# Import Telegram library
try:
    from telegram import Update, Bot, InlineKeyboardMarkup, InlineKeyboardButton
    from telegram.ext import (
        Application, CommandHandler, MessageHandler, CallbackQueryHandler,
        ContextTypes, filters, ConversationHandler
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    # Create placeholder classes for type hints
    class Update: pass
    class Bot: pass
    class InlineKeyboardMarkup: pass
    class InlineKeyboardButton: pass
    class Application: pass
    class CommandHandler: pass
    class MessageHandler: pass
    class CallbackQueryHandler: pass
    class ContextTypes: pass
    class filters: pass
    class ConversationHandler: pass
    TELEGRAM_AVAILABLE = False
    logging.warning("Telegram library not available. Install with: pip install python-telegram-bot")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("telegram_bot")

class CommandDefinition:
    """Definition of a Telegram bot command"""
    
    def __init__(
        self,
        command: str,
        handler: Callable,
        description: str,
        help_text: str = None,
        examples: List[str] = None,
        plugin_id: str = None,
        admin_only: bool = False,
        enabled: bool = True
    ):
        """Initialize command definition
        
        Args:
            command: Command name (without leading slash)
            handler: Function to handle the command
            description: Short description of the command
            help_text: Detailed help text for the command
            examples: List of example usages
            plugin_id: ID of the plugin that registered this command
            admin_only: Whether the command is restricted to admin users
            enabled: Whether the command is currently enabled
        """
        self.command = command
        self.handler = handler
        self.description = description
        self.help_text = help_text or description
        self.examples = examples or []
        self.plugin_id = plugin_id
        self.admin_only = admin_only
        self.enabled = enabled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert command definition to dictionary
        
        Returns:
            dict: Dictionary representation of command definition
        """
        return {
            "command": self.command,
            "description": self.description,
            "help_text": self.help_text,
            "examples": self.examples,
            "plugin_id": self.plugin_id,
            "admin_only": self.admin_only,
            "enabled": self.enabled
        }


class ConversationState:
    """State of a conversation with a user"""
    
    def __init__(
        self,
        user_id: int,
        chat_id: int,
        context: Dict[str, Any] = None,
        history: List[Dict[str, Any]] = None,
        current_handler: str = None,
        last_activity: float = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize conversation state
        
        Args:
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            context: Conversation context data
            history: Conversation history
            current_handler: ID of current conversation handler
            last_activity: Timestamp of last activity
            metadata: Additional metadata
        """
        self.user_id = user_id
        self.chat_id = chat_id
        self.context = context or {}
        self.history = history or []
        self.current_handler = current_handler
        self.last_activity = last_activity or datetime.now().timestamp()
        self.metadata = metadata or {}
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now().timestamp()
    
    def add_message(self, role: str, text: str, metadata: Dict[str, Any] = None):
        """Add message to conversation history
        
        Args:
            role: Role of the message sender ("user" or "bot")
            text: Message text
            metadata: Additional metadata
        """
        self.history.append({
            "role": role,
            "text": text,
            "timestamp": datetime.now().timestamp(),
            "metadata": metadata or {}
        })
        self.update_activity()
    
    def get_recent_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history
        
        Args:
            count: Maximum number of messages to return
            
        Returns:
            list: Recent conversation history
        """
        return self.history[-count:] if len(self.history) > count else self.history
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation state to dictionary
        
        Returns:
            dict: Dictionary representation of conversation state
        """
        return {
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "context": self.context,
            "history": self.history,
            "current_handler": self.current_handler,
            "last_activity": self.last_activity,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create conversation state from dictionary
        
        Args:
            data: Dictionary representation of conversation state
            
        Returns:
            ConversationState: Conversation state object
        """
        return cls(
            user_id=data["user_id"],
            chat_id=data["chat_id"],
            context=data.get("context", {}),
            history=data.get("history", []),
            current_handler=data.get("current_handler"),
            last_activity=data.get("last_activity"),
            metadata=data.get("metadata", {})
        )


class ConversationHandlerDefinition:
    """Definition of a conversation handler"""
    
    def __init__(
        self,
        handler_id: str,
        entry_points: List[Tuple[str, Callable]],
        states: Dict[str, List[Tuple[str, Callable]]],
        fallbacks: List[Tuple[str, Callable]],
        description: str,
        plugin_id: str = None,
        timeout: int = 300,
        enabled: bool = True
    ):
        """Initialize conversation handler definition
        
        Args:
            handler_id: Unique identifier for the handler
            entry_points: List of (trigger_type, handler_func) tuples for entry points
            states: Dictionary of state_name -> [(trigger_type, handler_func), ...] for states
            fallbacks: List of (trigger_type, handler_func) tuples for fallbacks
            description: Description of the conversation handler
            plugin_id: ID of the plugin that registered this handler
            timeout: Conversation timeout in seconds
            enabled: Whether the handler is currently enabled
        """
        self.handler_id = handler_id
        self.entry_points = entry_points
        self.states = states
        self.fallbacks = fallbacks
        self.description = description
        self.plugin_id = plugin_id
        self.timeout = timeout
        self.enabled = enabled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation handler definition to dictionary
        
        Returns:
            dict: Dictionary representation of conversation handler definition
        """
        # Cannot directly serialize handler functions
        return {
            "handler_id": self.handler_id,
            "description": self.description,
            "plugin_id": self.plugin_id,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "entry_points_count": len(self.entry_points),
            "states_count": {state: len(handlers) for state, handlers in self.states.items()},
            "fallbacks_count": len(self.fallbacks)
        }


class NLUIntent:
    """Natural Language Understanding Intent"""
    
    def __init__(
        self,
        intent_id: str,
        patterns: List[str],
        handler: Callable,
        description: str,
        examples: List[str] = None,
        plugin_id: str = None,
        enabled: bool = True
    ):
        """Initialize NLU intent
        
        Args:
            intent_id: Unique identifier for the intent
            patterns: List of regex patterns to match
            handler: Function to handle the intent
            description: Description of the intent
            examples: List of example phrases
            plugin_id: ID of the plugin that registered this intent
            enabled: Whether the intent is currently enabled
        """
        self.intent_id = intent_id
        self.patterns = patterns
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.handler = handler
        self.description = description
        self.examples = examples or []
        self.plugin_id = plugin_id
        self.enabled = enabled
    
    def match(self, text: str) -> Optional[re.Match]:
        """Check if text matches any pattern
        
        Args:
            text: Text to check
            
        Returns:
            re.Match: Match object if matched, None otherwise
        """
        if not self.enabled:
            return None
        
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                return match
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert NLU intent to dictionary
        
        Returns:
            dict: Dictionary representation of NLU intent
        """
        return {
            "intent_id": self.intent_id,
            "patterns": self.patterns,
            "description": self.description,
            "examples": self.examples,
            "plugin_id": self.plugin_id,
            "enabled": self.enabled
        }


class TelegramBotModule:
    """Modular Telegram bot with plugin support"""
    
    def __init__(
        self,
        config_registry=None,
        event_bus=None,
        service_locator=None,
        token: str = None,
        admin_user_ids: List[int] = None,
        allowed_user_ids: List[int] = None,
        conversation_timeout: int = 300,
        persistence_file: str = None
    ):
        """Initialize Telegram bot module
        
        Args:
            config_registry: Configuration registry
            event_bus: Event bus
            service_locator: Service locator
            token: Telegram bot token (optional, can be set via config)
            admin_user_ids: List of admin user IDs (optional, can be set via config)
            allowed_user_ids: List of allowed user IDs (optional, can be set via config)
            conversation_timeout: Conversation timeout in seconds
            persistence_file: Path to persistence file
        """
        self.config_registry = config_registry
        self.event_bus = event_bus
        self.service_locator = service_locator
        
        self.token = token
        self.admin_user_ids = admin_user_ids or []
        self.allowed_user_ids = allowed_user_ids or []
        self.conversation_timeout = conversation_timeout
        self.persistence_file = persistence_file
        
        # Command handlers
        self.commands = {}  # {command: CommandDefinition}
        
        # Conversation handlers
        self.conversation_handlers = {}  # {handler_id: ConversationHandlerDefinition}
        self.conversation_states = {}  # {user_id: ConversationState}
        
        # NLU intents
        self.intents = {}  # {intent_id: NLUIntent}
        
        # Telegram application
        self.application = None
        self.bot = None
        self.running = False
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize from config if provided
        if self.config_registry:
            self._init_from_config()
        
        logger.info("TelegramBotModule initialized")
    
    def _init_from_config(self):
        """Initialize from configuration registry"""
        if not self.config_registry:
            return
        
        try:
            # Get Telegram bot configuration
            self.token = self.config_registry.get_parameter(
                "telegram_bot", "token", self.token)
            
            admin_user_ids = self.config_registry.get_parameter(
                "telegram_bot", "admin_user_ids", None)
            if admin_user_ids:
                self.admin_user_ids = [int(uid) for uid in admin_user_ids.split(",")]
            
            allowed_user_ids = self.config_registry.get_parameter(
                "telegram_bot", "allowed_user_ids", None)
            if allowed_user_ids:
                self.allowed_user_ids = [int(uid) for uid in allowed_user_ids.split(",")]
            
            self.conversation_timeout = self.config_registry.get_parameter(
                "telegram_bot", "conversation_timeout", self.conversation_timeout)
            
            self.persistence_file = self.config_registry.get_parameter(
                "telegram_bot", "persistence_file", self.persistence_file)
            
            logger.info("TelegramBotModule initialized from configuration")
        except Exception as e:
            logger.error(f"Error initializing TelegramBotModule from configuration: {e}")
    
    async def start(self):
        """Start the Telegram bot"""
        with self.lock:
            if self.running:
                logger.warning("TelegramBotModule already running")
                return
            
            if not self.token:
                logger.error("Telegram bot token not set")
                return
            
            if not TELEGRAM_AVAILABLE:
                logger.error("Telegram library not available")
                return
            
            try:
                # Create application
                self.application = Application.builder().token(self.token).build()
                
                # Get bot instance
                self.bot = self.application.bot
                
                # Register command handlers
                for command, cmd_def in self.commands.items():
                    if cmd_def.enabled:
                        self.application.add_handler(
                            CommandHandler(command, self._wrap_command_handler(cmd_def))
                        )
                
                # Register conversation handlers
                for handler_id, handler_def in self.conversation_handlers.items():
                    if handler_def.enabled:
                        # Convert entry points, states, and fallbacks to proper handlers
                        entry_points = self._create_handlers(handler_def.entry_points)
                        states = {
                            state: self._create_handlers(handlers)
                            for state, handlers in handler_def.states.items()
                        }
                        fallbacks = self._create_handlers(handler_def.fallbacks)
                        
                        # Create conversation handler
                        conv_handler = ConversationHandler(
                            entry_points=entry_points,
                            states=states,
                            fallbacks=fallbacks,
                            name=handler_id,
                            persistent=False,
                            conversation_timeout=handler_def.timeout
                        )
                        
                        self.application.add_handler(conv_handler)
                
                # Register message handler for NLU intents
                self.application.add_handler(
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
                )
                
                # Register error handler
                self.application.add_error_handler(self._error_handler)
                
                # Load conversation states from persistence file
                if self.persistence_file and os.path.exists(self.persistence_file):
                    self._load_conversation_states()
                
                # Start the bot
                self.running = True
                await self.application.initialize()
                await self.application.start()
                await self.application.updater.start_polling()
                
                logger.info("TelegramBotModule started")
                
                # Publish event if event bus is available
                if self.event_bus:
                    self.event_bus.publish(
                        event_type="telegram_bot.started",
                        data={
                            "bot_username": (await self.bot.get_me()).username
                        }
                    )
                
                # Run the bot until stopped
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
            
            except Exception as e:
                logger.error(f"Error starting TelegramBotModule: {e}")
                self.running = False
    
    def stop(self):
        """Stop the Telegram bot"""
        with self.lock:
            if not self.running:
                logger.warning("TelegramBotModule not running")
                return
            
            try:
                # Save conversation states to persistence file
                if self.persistence_file:
                    self._save_conversation_states()
                
                # Stop the application
                asyncio.run_coroutine_threadsafe(
                    self.application.updater.stop(), 
                    self.application.updater.loop
                )
                asyncio.run_coroutine_threadsafe(
                    self.application.stop(), 
                    self.application.updater.loop
                )
                asyncio.run_coroutine_threadsafe(
                    self.application.shutdown(), 
                    self.application.updater.loop
                )
                
                self.running = False
                logger.info("TelegramBotModule stopped")
                
                # Publish event if event bus is available
                if self.event_bus:
                    self.event_bus.publish(
                        event_type="telegram_bot.stopped",
                        data={}
                    )
            
            except Exception as e:
                logger.error(f"Error stopping TelegramBotModule: {e}")
    
    def register_command(
        self,
        command: str,
        handler: Callable,
        description: str,
        help_text: str = None,
        examples: List[str] = None,
        plugin_id: str = None,
        admin_only: bool = False,
        enabled: bool = True
    ) -> bool:
        """Register a command handler
        
        Args:
            command: Command name (without leading slash)
            handler: Function to handle the command
            description: Short description of the command
            help_text: Detailed help text for the command
            examples: List of example usages
            plugin_id: ID of the plugin that registered this command
            admin_only: Whether the command is restricted to admin users
            enabled: Whether the command is currently enabled
            
        Returns:
            bool: True if command was registered, False otherwise
        """
        with self.lock:
            # Check if command already exists
            if command in self.commands:
                logger.warning(f"Command /{command} already registered")
                return False
            
            # Create command definition
            cmd_def = CommandDefinition(
                command=command,
                handler=handler,
                description=description,
                help_text=help_text,
                examples=examples,
                plugin_id=plugin_id,
                admin_only=admin_only,
                enabled=enabled
            )
            
            # Store command definition
            self.commands[command] = cmd_def
            
            # Register command handler if bot is running
            if self.running and cmd_def.enabled:
                self.application.add_handler(
                    CommandHandler(command, self._wrap_command_handler(cmd_def))
                )
            
            logger.info(f"Command /{command} registered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="telegram_bot.command_registered",
                    data={
                        "command": command,
                        "description": description,
                        "plugin_id": plugin_id
                    }
                )
            
            return True
    
    def unregister_command(self, command: str) -> bool:
        """Unregister a command handler
        
        Args:
            command: Command name (without leading slash)
            
        Returns:
            bool: True if command was unregistered, False otherwise
        """
        with self.lock:
            # Check if command exists
            if command not in self.commands:
                logger.warning(f"Command /{command} not registered")
                return False
            
            # Remove command definition
            cmd_def = self.commands.pop(command)
            
            # Remove command handler if bot is running
            if self.running:
                handlers = [h for h in self.application.handlers[0] 
                           if isinstance(h, CommandHandler) and h.command == [command]]
                for handler in handlers:
                    self.application.remove_handler(handler)
            
            logger.info(f"Command /{command} unregistered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="telegram_bot.command_unregistered",
                    data={
                        "command": command,
                        "plugin_id": cmd_def.plugin_id
                    }
                )
            
            return True
    
    def register_conversation_handler(
        self,
        handler_id: str,
        entry_points: List[Tuple[str, Callable]],
        states: Dict[str, List[Tuple[str, Callable]]],
        fallbacks: List[Tuple[str, Callable]],
        description: str,
        plugin_id: str = None,
        timeout: int = None,
        enabled: bool = True
    ) -> bool:
        """Register a conversation handler
        
        Args:
            handler_id: Unique identifier for the handler
            entry_points: List of (trigger_type, handler_func) tuples for entry points
            states: Dictionary of state_name -> [(trigger_type, handler_func), ...] for states
            fallbacks: List of (trigger_type, handler_func) tuples for fallbacks
            description: Description of the conversation handler
            plugin_id: ID of the plugin that registered this handler
            timeout: Conversation timeout in seconds (defaults to global timeout)
            enabled: Whether the handler is currently enabled
            
        Returns:
            bool: True if handler was registered, False otherwise
        """
        with self.lock:
            # Check if handler already exists
            if handler_id in self.conversation_handlers:
                logger.warning(f"Conversation handler {handler_id} already registered")
                return False
            
            # Use global timeout if not specified
            if timeout is None:
                timeout = self.conversation_timeout
            
            # Create conversation handler definition
            handler_def = ConversationHandlerDefinition(
                handler_id=handler_id,
                entry_points=entry_points,
                states=states,
                fallbacks=fallbacks,
                description=description,
                plugin_id=plugin_id,
                timeout=timeout,
                enabled=enabled
            )
            
            # Store conversation handler definition
            self.conversation_handlers[handler_id] = handler_def
            
            # Register conversation handler if bot is running
            if self.running and handler_def.enabled:
                # Convert entry points, states, and fallbacks to proper handlers
                entry_points = self._create_handlers(handler_def.entry_points)
                states = {
                    state: self._create_handlers(handlers)
                    for state, handlers in handler_def.states.items()
                }
                fallbacks = self._create_handlers(handler_def.fallbacks)
                
                # Create conversation handler
                conv_handler = ConversationHandler(
                    entry_points=entry_points,
                    states=states,
                    fallbacks=fallbacks,
                    name=handler_id,
                    persistent=False,
                    conversation_timeout=handler_def.timeout
                )
                
                self.application.add_handler(conv_handler)
            
            logger.info(f"Conversation handler {handler_id} registered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="telegram_bot.conversation_handler_registered",
                    data={
                        "handler_id": handler_id,
                        "description": description,
                        "plugin_id": plugin_id
                    }
                )
            
            return True
    
    def unregister_conversation_handler(self, handler_id: str) -> bool:
        """Unregister a conversation handler
        
        Args:
            handler_id: Unique identifier for the handler
            
        Returns:
            bool: True if handler was unregistered, False otherwise
        """
        with self.lock:
            # Check if handler exists
            if handler_id not in self.conversation_handlers:
                logger.warning(f"Conversation handler {handler_id} not registered")
                return False
            
            # Remove conversation handler definition
            handler_def = self.conversation_handlers.pop(handler_id)
            
            # Remove conversation handler if bot is running
            if self.running:
                handlers = [h for h in self.application.handlers[0] 
                           if isinstance(h, ConversationHandler) and h.name == handler_id]
                for handler in handlers:
                    self.application.remove_handler(handler)
            
            logger.info(f"Conversation handler {handler_id} unregistered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="telegram_bot.conversation_handler_unregistered",
                    data={
                        "handler_id": handler_id,
                        "plugin_id": handler_def.plugin_id
                    }
                )
            
            return True
    
    def register_intent(
        self,
        intent_id: str,
        patterns: List[str],
        handler: Callable,
        description: str,
        examples: List[str] = None,
        plugin_id: str = None,
        enabled: bool = True
    ) -> bool:
        """Register an NLU intent
        
        Args:
            intent_id: Unique identifier for the intent
            patterns: List of regex patterns to match
            handler: Function to handle the intent
            description: Description of the intent
            examples: List of example phrases
            plugin_id: ID of the plugin that registered this intent
            enabled: Whether the intent is currently enabled
            
        Returns:
            bool: True if intent was registered, False otherwise
        """
        with self.lock:
            # Check if intent already exists
            if intent_id in self.intents:
                logger.warning(f"Intent {intent_id} already registered")
                return False
            
            # Create NLU intent
            intent = NLUIntent(
                intent_id=intent_id,
                patterns=patterns,
                handler=handler,
                description=description,
                examples=examples,
                plugin_id=plugin_id,
                enabled=enabled
            )
            
            # Store NLU intent
            self.intents[intent_id] = intent
            
            logger.info(f"Intent {intent_id} registered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="telegram_bot.intent_registered",
                    data={
                        "intent_id": intent_id,
                        "description": description,
                        "plugin_id": plugin_id
                    }
                )
            
            return True
    
    def unregister_intent(self, intent_id: str) -> bool:
        """Unregister an NLU intent
        
        Args:
            intent_id: Unique identifier for the intent
            
        Returns:
            bool: True if intent was unregistered, False otherwise
        """
        with self.lock:
            # Check if intent exists
            if intent_id not in self.intents:
                logger.warning(f"Intent {intent_id} not registered")
                return False
            
            # Remove NLU intent
            intent = self.intents.pop(intent_id)
            
            logger.info(f"Intent {intent_id} unregistered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="telegram_bot.intent_unregistered",
                    data={
                        "intent_id": intent_id,
                        "plugin_id": intent.plugin_id
                    }
                )
            
            return True
    
    def get_conversation_state(self, user_id: int) -> Optional[ConversationState]:
        """Get conversation state for a user
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            ConversationState: Conversation state or None if not found
        """
        with self.lock:
            return self.conversation_states.get(user_id)
    
    def set_conversation_state(self, state: ConversationState) -> bool:
        """Set conversation state for a user
        
        Args:
            state: Conversation state
            
        Returns:
            bool: True if state was set
        """
        with self.lock:
            self.conversation_states[state.user_id] = state
            return True
    
    def clear_conversation_state(self, user_id: int) -> bool:
        """Clear conversation state for a user
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            bool: True if state was cleared, False if not found
        """
        with self.lock:
            if user_id in self.conversation_states:
                del self.conversation_states[user_id]
                return True
            return False
    
    def _wrap_command_handler(self, cmd_def: CommandDefinition) -> Callable:
        """Wrap command handler with permission checks
        
        Args:
            cmd_def: Command definition
            
        Returns:
            Callable: Wrapped handler function
        """
        async def wrapped_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            # Check if user is allowed
            if not self._is_user_allowed(update):
                await update.message.reply_text("You are not authorized to use this bot.")
                return
            
            # Check if admin-only command
            if cmd_def.admin_only and not self._is_user_admin(update):
                await update.message.reply_text("This command is restricted to administrators.")
                return
            
            # Get user and chat IDs
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            
            # Get or create conversation state
            state = self.get_conversation_state(user_id)
            if not state:
                state = ConversationState(user_id=user_id, chat_id=chat_id)
                self.set_conversation_state(state)
            else:
                state.update_activity()
            
            # Add message to conversation history
            state.add_message(
                role="user",
                text=update.message.text,
                metadata={"type": "command", "command": cmd_def.command}
            )
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="telegram_bot.command_received",
                    data={
                        "command": cmd_def.command,
                        "user_id": user_id,
                        "chat_id": chat_id,
                        "text": update.message.text
                    }
                )
            
            try:
                # Call original handler
                result = cmd_def.handler(update, context)
                
                # Handle coroutines
                if inspect.iscoroutine(result):
                    result = await result
                
                return result
            
            except Exception as e:
                logger.error(f"Error handling command /{cmd_def.command}: {e}")
                await update.message.reply_text(f"Error executing command: {str(e)}")
        
        return wrapped_handler
    
    def _create_handlers(self, handler_specs: List[Tuple[str, Callable]]) -> List:
        """Create handlers from specifications
        
        Args:
            handler_specs: List of (trigger_type, handler_func) tuples
            
        Returns:
            list: List of handler objects
        """
        handlers = []
        
        for trigger_type, handler_func in handler_specs:
            if trigger_type == "command":
                # Command trigger
                command = handler_func.__name__.replace("command_", "")
                handlers.append(CommandHandler(command, handler_func))
            
            elif trigger_type == "message":
                # Message trigger
                handlers.append(MessageHandler(filters.TEXT & ~filters.COMMAND, handler_func))
            
            elif trigger_type == "regex":
                # Regex trigger
                pattern = getattr(handler_func, "pattern", None)
                if pattern:
                    handlers.append(MessageHandler(filters.Regex(pattern), handler_func))
            
            elif trigger_type == "callback":
                # Callback query trigger
                pattern = getattr(handler_func, "pattern", None)
                if pattern:
                    handlers.append(CallbackQueryHandler(handler_func, pattern=pattern))
                else:
                    handlers.append(CallbackQueryHandler(handler_func))
        
        return handlers
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages for NLU intents
        
        Args:
            update: Telegram update
            context: Telegram context
        """
        # Check if user is allowed
        if not self._is_user_allowed(update):
            await update.message.reply_text("You are not authorized to use this bot.")
            return
        
        # Get user and chat IDs
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # Get message text
        text = update.message.text
        
        # Get or create conversation state
        state = self.get_conversation_state(user_id)
        if not state:
            state = ConversationState(user_id=user_id, chat_id=chat_id)
            self.set_conversation_state(state)
        else:
            state.update_activity()
        
        # Add message to conversation history
        state.add_message(
            role="user",
            text=text,
            metadata={"type": "message"}
        )
        
        # Publish event if event bus is available
        if self.event_bus:
            self.event_bus.publish(
                event_type="telegram_bot.message_received",
                data={
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "text": text
                }
            )
        
        # Check if message matches any intent
        matched_intent = None
        match_obj = None
        
        for intent_id, intent in self.intents.items():
            match_obj = intent.match(text)
            if match_obj:
                matched_intent = intent
                break
        
        if matched_intent:
            logger.info(f"Message matched intent: {matched_intent.intent_id}")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="telegram_bot.intent_matched",
                    data={
                        "intent_id": matched_intent.intent_id,
                        "user_id": user_id,
                        "chat_id": chat_id,
                        "text": text
                    }
                )
            
            try:
                # Call intent handler
                result = matched_intent.handler(update, context, match_obj)
                
                # Handle coroutines
                if inspect.iscoroutine(result):
                    result = await result
                
                return result
            
            except Exception as e:
                logger.error(f"Error handling intent {matched_intent.intent_id}: {e}")
                await update.message.reply_text(f"Error processing your message: {str(e)}")
        
        else:
            # No intent matched, use LLM for natural language understanding if available
            llm_client = self._get_llm_client()
            if llm_client:
                try:
                    # Process message with LLM
                    response = await self._process_message_with_llm(text, state)
                    
                    # Send response
                    await update.message.reply_text(response)
                    
                    # Add response to conversation history
                    state.add_message(
                        role="bot",
                        text=response,
                        metadata={"type": "llm_response"}
                    )
                    
                    return
                
                except Exception as e:
                    logger.error(f"Error processing message with LLM: {e}")
            
            # Default response if no intent matched and no LLM available
            await update.message.reply_text(
                "I'm not sure how to respond to that. Try using a command like /help."
            )
    
    async def _process_message_with_llm(self, text: str, state: ConversationState) -> str:
        """Process message with LLM
        
        Args:
            text: Message text
            state: Conversation state
            
        Returns:
            str: LLM response
        """
        llm_client = self._get_llm_client()
        if not llm_client:
            return "I'm not sure how to respond to that. Try using a command like /help."
        
        # Get conversation history
        history = state.get_recent_history(10)
        
        # Format conversation history for LLM
        formatted_history = []
        for msg in history:
            if msg["role"] == "user":
                formatted_history.append({"role": "user", "content": msg["text"]})
            else:
                formatted_history.append({"role": "assistant", "content": msg["text"]})
        
        # Remove the last user message (current message) as we'll add it separately
        if formatted_history and formatted_history[-1]["role"] == "user":
            formatted_history = formatted_history[:-1]
        
        # Add system message
        system_message = self._get_system_message()
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history
        messages.extend(formatted_history)
        
        # Add current message
        messages.append({"role": "user", "content": text})
        
        # Get LLM response
        response = await llm_client.generate_async(messages=messages)
        
        return response
    
    def _get_system_message(self) -> str:
        """Get system message for LLM
        
        Returns:
            str: System message
        """
        # Get system message from config if available
        if self.config_registry:
            system_message = self.config_registry.get_parameter(
                "telegram_bot", "llm_system_message", None)
            if system_message:
                return system_message
        
        # Default system message
        return (
            "You are an AI assistant for a trading system. "
            "You can help users with commands, answer questions about the system, "
            "and provide information about trading. "
            "Be concise, helpful, and informative."
        )
    
    def _get_llm_client(self):
        """Get LLM client from service locator
        
        Returns:
            LLMClient: LLM client or None if not available
        """
        if self.service_locator:
            try:
                return self.service_locator.get_service("llm_client")
            except Exception as e:
                logger.error(f"Error getting LLM client: {e}")
        return None
    
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in updates
        
        Args:
            update: Telegram update
            context: Telegram context
        """
        logger.error(f"Error handling update: {context.error}")
        
        # Publish event if event bus is available
        if self.event_bus:
            self.event_bus.publish(
                event_type="telegram_bot.error",
                data={
                    "error": str(context.error),
                    "update_id": update.update_id if update else None
                }
            )
        
        # Send error message to user if possible
        if update and update.effective_chat:
            await update.effective_chat.send_message(
                "An error occurred while processing your request. Please try again later."
            )
    
    def _is_user_allowed(self, update: Update) -> bool:
        """Check if user is allowed to use the bot
        
        Args:
            update: Telegram update
            
        Returns:
            bool: True if user is allowed, False otherwise
        """
        user_id = update.effective_user.id
        
        # Admin users are always allowed
        if user_id in self.admin_user_ids:
            return True
        
        # Check if user is in allowed list
        if self.allowed_user_ids and user_id not in self.allowed_user_ids:
            return False
        
        return True
    
    def _is_user_admin(self, update: Update) -> bool:
        """Check if user is an admin
        
        Args:
            update: Telegram update
            
        Returns:
            bool: True if user is an admin, False otherwise
        """
        user_id = update.effective_user.id
        return user_id in self.admin_user_ids
    
    def _save_conversation_states(self):
        """Save conversation states to persistence file"""
        if not self.persistence_file:
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.persistence_file)), exist_ok=True)
            
            # Convert conversation states to dictionaries
            states_dict = {
                str(user_id): state.to_dict()
                for user_id, state in self.conversation_states.items()
            }
            
            # Save to file
            with open(self.persistence_file, "w") as f:
                json.dump(states_dict, f)
            
            logger.info(f"Conversation states saved to {self.persistence_file}")
        
        except Exception as e:
            logger.error(f"Error saving conversation states: {e}")
    
    def _load_conversation_states(self):
        """Load conversation states from persistence file"""
        if not self.persistence_file or not os.path.exists(self.persistence_file):
            return
        
        try:
            # Load from file
            with open(self.persistence_file, "r") as f:
                states_dict = json.load(f)
            
            # Convert dictionaries to conversation states
            for user_id_str, state_dict in states_dict.items():
                try:
                    user_id = int(user_id_str)
                    state = ConversationState.from_dict(state_dict)
                    self.conversation_states[user_id] = state
                except Exception as e:
                    logger.error(f"Error loading conversation state for user {user_id_str}: {e}")
            
            logger.info(f"Conversation states loaded from {self.persistence_file}")
        
        except Exception as e:
            logger.error(f"Error loading conversation states: {e}")
    
    def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = None,
        reply_markup=None
    ) -> bool:
        """Send message to a chat
        
        Args:
            chat_id: Telegram chat ID
            text: Message text
            parse_mode: Parse mode (None, "Markdown", or "HTML")
            reply_markup: Reply markup
            
        Returns:
            bool: True if message was sent, False otherwise
        """
        if not self.running or not self.bot:
            logger.error("Bot not running")
            return False
        
        try:
            # Send message
            asyncio.run_coroutine_threadsafe(
                self.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup
                ),
                self.application.updater.loop
            )
            
            # Get user ID from conversation states
            user_id = None
            for uid, state in self.conversation_states.items():
                if state.chat_id == chat_id:
                    user_id = uid
                    break
            
            # Add message to conversation history if user ID found
            if user_id:
                state = self.conversation_states.get(user_id)
                if state:
                    state.add_message(
                        role="bot",
                        text=text,
                        metadata={"type": "sent_message"}
                    )
            
            return True
        
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def send_notification(
        self,
        text: str,
        level: str = "info",
        user_id: int = None,
        parse_mode: str = None
    ) -> bool:
        """Send notification to users
        
        Args:
            text: Notification text
            level: Notification level ("info", "warning", "error", "success")
            user_id: Specific user ID to notify (None for all users)
            parse_mode: Parse mode (None, "Markdown", or "HTML")
            
        Returns:
            bool: True if notification was sent, False otherwise
        """
        if not self.running or not self.bot:
            logger.error("Bot not running")
            return False
        
        # Add emoji based on level
        if level == "info":
            prefix = "ℹ️ "
        elif level == "warning":
            prefix = "⚠️ "
        elif level == "error":
            prefix = "❌ "
        elif level == "success":
            prefix = "✅ "
        else:
            prefix = ""
        
        # Format message
        message = f"{prefix}{text}"
        
        try:
            if user_id:
                # Send to specific user
                chat_id = None
                state = self.conversation_states.get(user_id)
                if state:
                    chat_id = state.chat_id
                
                if chat_id:
                    return self.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode=parse_mode
                    )
                else:
                    logger.warning(f"Chat ID not found for user {user_id}")
                    return False
            
            else:
                # Send to all users
                success = True
                for uid, state in self.conversation_states.items():
                    if not self.send_message(
                        chat_id=state.chat_id,
                        text=message,
                        parse_mode=parse_mode
                    ):
                        success = False
                
                return success
        
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
    
    def get_commands(self) -> Dict[str, Dict[str, Any]]:
        """Get registered commands
        
        Returns:
            dict: Dictionary of command definitions
        """
        with self.lock:
            return {cmd: cmd_def.to_dict() for cmd, cmd_def in self.commands.items()}
    
    def get_conversation_handlers(self) -> Dict[str, Dict[str, Any]]:
        """Get registered conversation handlers
        
        Returns:
            dict: Dictionary of conversation handler definitions
        """
        with self.lock:
            return {
                handler_id: handler_def.to_dict()
                for handler_id, handler_def in self.conversation_handlers.items()
            }
    
    def get_intents(self) -> Dict[str, Dict[str, Any]]:
        """Get registered intents
        
        Returns:
            dict: Dictionary of intent definitions
        """
        with self.lock:
            return {intent_id: intent.to_dict() for intent_id, intent in self.intents.items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics
        
        Returns:
            dict: Bot statistics
        """
        with self.lock:
            return {
                "commands": len(self.commands),
                "conversation_handlers": len(self.conversation_handlers),
                "intents": len(self.intents),
                "conversation_states": len(self.conversation_states),
                "running": self.running
            }


# Example usage
if __name__ == "__main__":
    # Create Telegram bot module
    bot_module = TelegramBotModule(
        token="YOUR_BOT_TOKEN",
        admin_user_ids=[123456789],
        conversation_timeout=300,
        persistence_file="telegram_bot_states.json"
    )
    
    # Define command handlers
    async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Welcome to the Trading System Bot! Use /help to see available commands."
        )
    
    async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        commands = bot_module.get_commands()
        help_text = "Available commands:\n\n"
        
        for cmd, cmd_def in commands.items():
            if cmd_def["admin_only"] and not bot_module._is_user_admin(update):
                continue
            help_text += f"/{cmd} - {cmd_def['description']}\n"
        
        await update.message.reply_text(help_text)
    
    async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("System is running normally.")
    
    # Register command handlers
    bot_module.register_command(
        command="start",
        handler=start_command,
        description="Start the bot"
    )
    
    bot_module.register_command(
        command="help",
        handler=help_command,
        description="Show available commands"
    )
    
    bot_module.register_command(
        command="status",
        handler=status_command,
        description="Show system status"
    )
    
    # Define NLU intents
    async def greeting_intent(update: Update, context: ContextTypes.DEFAULT_TYPE, match):
        await update.message.reply_text("Hello! How can I help you today?")
    
    async def farewell_intent(update: Update, context: ContextTypes.DEFAULT_TYPE, match):
        await update.message.reply_text("Goodbye! Have a great day!")
    
    # Register NLU intents
    bot_module.register_intent(
        intent_id="greeting",
        patterns=[r"^(hi|hello|hey|greetings).*$"],
        handler=greeting_intent,
        description="Greeting intent",
        examples=["hi", "hello", "hey there"]
    )
    
    bot_module.register_intent(
        intent_id="farewell",
        patterns=[r"^(bye|goodbye|see you|farewell).*$"],
        handler=farewell_intent,
        description="Farewell intent",
        examples=["bye", "goodbye", "see you later"]
    )
    
    # Start the bot
    asyncio.run(bot_module.start())
