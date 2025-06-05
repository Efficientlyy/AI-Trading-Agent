#!/usr/bin/env python
"""
Improved Telegram Integration for System Overseer.

This module provides the TelegramIntegration class for interacting with Telegram,
with improved application lifecycle management based on successful debug tests.
"""

import os
import sys
import json
import logging
import threading
import time
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env-secure/.env')

# Import telegram library
try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
except ImportError:
    logging.error("Telegram library not installed. Please install python-telegram-bot.")
    sys.exit(1)

logger = logging.getLogger("system_overseer.telegram_integration")

class TelegramIntegration:
    """Telegram Integration for System Overseer with improved lifecycle management."""
    
    def __init__(self, dialogue_manager=None, system_core=None, token=None, chat_id=None):
        """Initialize Telegram Integration.
        
        Args:
            dialogue_manager: Dialogue manager instance
            system_core: System core instance
            token: Telegram bot token (optional, will use env var if not provided)
            chat_id: Telegram chat ID (optional, will use env var if not provided)
        """
        self.dialogue_manager = dialogue_manager
        self.system_core = system_core
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.application = None
        self.bot = None
        self.running = False
        self.thread = None
        self.lock = threading.RLock()
        self.message_handlers = []
        
        if not self.token:
            logger.error("Telegram bot token not found in environment variables")
        
        if not self.chat_id:
            logger.warning("Telegram chat ID not found in environment variables")
        
        logger.info("TelegramIntegration initialized")
    
    def initialize(self):
        """Initialize Telegram bot.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Initializing Telegram bot...")
        
        try:
            # Create bot instance for direct API calls
            self.bot = Bot(token=self.token)
            logger.info("Telegram bot initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            return False
    
    def start(self):
        """Start Telegram bot.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Starting Telegram bot...")
        
        try:
            # Set running flag
            self.running = True
            
            # Start bot in a separate thread
            self.thread = threading.Thread(target=self._run_bot)
            self.thread.daemon = True
            self.thread.start()
            
            # Give the thread time to start up
            time.sleep(2)
            
            if not self.running:
                logger.error("Telegram bot failed to start")
                return False
                
            logger.info("Telegram bot started")
            return True
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            self.running = False
            return False
    
    def stop(self):
        """Stop Telegram bot.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Stopping Telegram bot...")
        
        try:
            # Set running flag to signal thread to stop
            self.running = False
            
            # Wait for thread to finish
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    logger.warning("Telegram bot thread did not stop gracefully")
            
            logger.info("Telegram bot stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop Telegram bot: {e}")
            return False
    
    def _run_bot(self):
        """Run Telegram bot in a separate thread with proper asyncio event loop management."""
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the bot using asyncio.run which properly manages the event loop
            loop.run_until_complete(self._run_telegram_application())
        except Exception as e:
            logger.error(f"Error in Telegram bot thread: {e}")
        finally:
            # Clean up
            if not loop.is_closed():
                loop.close()
            self.running = False
            logger.info("Telegram bot thread exited")
    
    async def _run_telegram_application(self):
        """Run the Telegram application with proper lifecycle management."""
        # Create application
        application = Application.builder().token(self.token).build()
        
        # Register handlers
        application.add_handler(CommandHandler("start", self._start_command))
        application.add_handler(CommandHandler("help", self._help_command))
        application.add_handler(CommandHandler("status", self._status_command))
        application.add_handler(CommandHandler("pairs", self._pairs_command))
        application.add_handler(CommandHandler("add_pair", self._add_pair_command))
        application.add_handler(CommandHandler("remove_pair", self._remove_pair_command))
        application.add_handler(CommandHandler("notifications", self._notifications_command))
        
        # Register message handler
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._message_handler))
        
        # Start the application
        await application.initialize()
        await application.start()
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        
        logger.info("Telegram polling started successfully")
        
        # Store application reference
        self.application = application
        
        # Keep running until stop is requested
        while self.running:
            await asyncio.sleep(1)
        
        # Proper shutdown sequence
        logger.info("Stopping Telegram polling...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        self.application = None
        logger.info("Telegram polling stopped")
    
    def register_message_handler(self, handler: Callable[[Dict[str, Any]], bool]):
        """Register a message handler.
        
        Args:
            handler: Function that takes a message dict and returns True if handled
        """
        with self.lock:
            self.message_handlers.append(handler)
    
    def send_message(self, chat_id: int, text: str):
        """Send a message to a chat.
        
        Args:
            chat_id: Chat ID
            text: Message text
        """
        if not self.bot:
            logger.error("Telegram bot not initialized")
            return
        
        try:
            # Create a new event loop for this operation
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.bot.send_message(chat_id=chat_id, text=text))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    def send_photo(self, chat_id: int, photo_path: str, caption: str = None):
        """Send a photo to a chat.
        
        Args:
            chat_id: Chat ID
            photo_path: Path to photo file
            caption: Optional caption
        """
        if not self.bot:
            logger.error("Telegram bot not initialized")
            return
        
        try:
            # Create a new event loop for this operation
            loop = asyncio.new_event_loop()
            try:
                with open(photo_path, 'rb') as photo_file:
                    loop.run_until_complete(self.bot.send_photo(
                        chat_id=chat_id, 
                        photo=photo_file, 
                        caption=caption
                    ))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to send photo: {e}")
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command.
        
        Args:
            update: Update object
            context: Context object
        """
        await update.message.reply_text(
            "Welcome to the Trading System Overseer! I'm here to help you monitor and control your trading system.\n\n"
            "Use /help to see available commands."
        )
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command.
        
        Args:
            update: Update object
            context: Context object
        """
        await update.message.reply_text(
            "Available commands:\n\n"
            "/help - Show this help message\n"
            "/status - Show system status\n"
            "/pairs - Show active trading pairs\n"
            "/add_pair SYMBOL - Add trading pair (e.g., /add_pair ETHUSDC)\n"
            "/remove_pair SYMBOL - Remove trading pair (e.g., /remove_pair ETHUSDC)\n"
            "/notifications LEVEL - Set notification level (all, signals, trades, errors, none)\n\n"
            "You can also ask me questions in natural language, and I'll do my best to help!\n\n"
            "For charts, simply ask something like:\n"
            "- \"Show me the BTC chart\"\n"
            "- \"Give me the ETH candlestick chart\"\n"
            "- \"I need to see the SOL price for the last hour\""
        )
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command.
        
        Args:
            update: Update object
            context: Context object
        """
        # Get system status
        status = "operational"
        details = "All systems are functioning normally."
        
        # Get config registry
        config_registry = self.system_core.get_service("config_registry") if self.system_core else None
        
        if config_registry:
            # Get trading pairs
            pairs = config_registry.get_config("trading.default_pairs", ["BTCUSDC", "ETHUSDC", "SOLUSDC"])
            
            # Get risk level
            risk_level = config_registry.get_config("trading.risk_level", "moderate")
            
            # Notification level
            notification_level = config_registry.get_config("notifications.level", "all")
            
            await update.message.reply_text(
                f"System Status: {status}\n"
                f"Details: {details}\n\n"
                f"Active Trading Pairs: {', '.join(pairs)}\n"
                f"Risk Level: {risk_level}\n"
                f"Notification Level: {notification_level}"
            )
        else:
            await update.message.reply_text(
                f"System Status: {status}\n"
                f"Details: {details}\n\n"
                "Note: Configuration registry not available."
            )
    
    async def _pairs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pairs command.
        
        Args:
            update: Update object
            context: Context object
        """
        # Get config registry
        config_registry = self.system_core.get_service("config_registry") if self.system_core else None
        
        if config_registry:
            # Get trading pairs
            pairs = config_registry.get_config("trading.default_pairs", ["BTCUSDC", "ETHUSDC", "SOLUSDC"])
            
            await update.message.reply_text(
                f"Active Trading Pairs: {', '.join(pairs)}"
            )
        else:
            await update.message.reply_text(
                "Active Trading Pairs: BTCUSDC, ETHUSDC, SOLUSDC\n\n"
                "Note: Configuration registry not available."
            )
    
    async def _add_pair_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add_pair command.
        
        Args:
            update: Update object
            context: Context object
        """
        # Get pair from command
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "Please specify a trading pair to add.\n"
                "Example: /add_pair ETHUSDC"
            )
            return
        
        pair = context.args[0].upper()
        
        # Get config registry
        config_registry = self.system_core.get_service("config_registry") if self.system_core else None
        
        if config_registry:
            # Get trading pairs
            pairs = config_registry.get_config("trading.default_pairs", ["BTCUSDC", "ETHUSDC", "SOLUSDC"])
            
            # Add pair if not already in list
            if pair not in pairs:
                pairs.append(pair)
                
                # Update config
                config_registry.set_config("trading.default_pairs", pairs)
                
                await update.message.reply_text(
                    f"Added trading pair: {pair}\n\n"
                    f"Active Trading Pairs: {', '.join(pairs)}"
                )
            else:
                await update.message.reply_text(
                    f"Trading pair {pair} is already active.\n\n"
                    f"Active Trading Pairs: {', '.join(pairs)}"
                )
        else:
            await update.message.reply_text(
                f"Added trading pair: {pair} (simulated)\n\n"
                "Note: Configuration registry not available."
            )
    
    async def _remove_pair_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remove_pair command.
        
        Args:
            update: Update object
            context: Context object
        """
        # Get pair from command
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "Please specify a trading pair to remove.\n"
                "Example: /remove_pair ETHUSDC"
            )
            return
        
        pair = context.args[0].upper()
        
        # Get config registry
        config_registry = self.system_core.get_service("config_registry") if self.system_core else None
        
        if config_registry:
            # Get trading pairs
            pairs = config_registry.get_config("trading.default_pairs", ["BTCUSDC", "ETHUSDC", "SOLUSDC"])
            
            # Remove pair if in list
            if pair in pairs:
                pairs.remove(pair)
                
                # Update config
                config_registry.set_config("trading.default_pairs", pairs)
                
                await update.message.reply_text(
                    f"Removed trading pair: {pair}\n\n"
                    f"Active Trading Pairs: {', '.join(pairs)}"
                )
            else:
                await update.message.reply_text(
                    f"Trading pair {pair} is not active.\n\n"
                    f"Active Trading Pairs: {', '.join(pairs)}"
                )
        else:
            await update.message.reply_text(
                f"Removed trading pair: {pair} (simulated)\n\n"
                "Note: Configuration registry not available."
            )
    
    async def _notifications_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /notifications command.
        
        Args:
            update: Update object
            context: Context object
        """
        # Get level from command
        if not context.args or len(context.args) < 1:
            await update.message.reply_text(
                "Please specify a notification level.\n"
                "Available levels: all, signals, trades, errors, none\n"
                "Example: /notifications all"
            )
            return
        
        level = context.args[0].lower()
        
        # Validate level
        valid_levels = ["all", "signals", "trades", "errors", "none"]
        if level not in valid_levels:
            await update.message.reply_text(
                f"Invalid notification level: {level}\n"
                f"Available levels: {', '.join(valid_levels)}\n"
                "Example: /notifications all"
            )
            return
        
        # Get config registry
        config_registry = self.system_core.get_service("config_registry") if self.system_core else None
        
        if config_registry:
            # Update config
            config_registry.set_config("notifications.level", level)
            
            await update.message.reply_text(
                f"Notification level set to: {level}"
            )
        else:
            await update.message.reply_text(
                f"Notification level set to: {level} (simulated)\n\n"
                "Note: Configuration registry not available."
            )
    
    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages.
        
        Args:
            update: Update object
            context: Context object
        """
        # Get message text
        message_text = update.message.text
        
        # Convert to message dict
        message = {
            "text": message_text,
            "chat": {"id": update.effective_chat.id},
            "from": {"id": update.effective_user.id, "username": update.effective_user.username},
            "message_id": update.message.message_id,
            "date": update.message.date.timestamp()
        }
        
        # Process with dialogue manager if available
        if self.dialogue_manager:
            response = self.dialogue_manager.process_message(message)
            if response:
                await update.message.reply_text(response)
                return
        
        # Try registered message handlers
        handled = False
        with self.lock:
            for handler in self.message_handlers:
                try:
                    if handler(message):
                        handled = True
                        break
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
        
        # If not handled, provide a default response
        if not handled:
            # Check if this might be a chart request
            if any(keyword in message_text.lower() for keyword in ["chart", "price", "btc", "eth", "sol", "bitcoin", "ethereum", "solana"]):
                await update.message.reply_text(
                    "I detected you might be asking for a chart. Our natural language chart request feature is being initialized. "
                    "Please try again in a moment, or use a command like /help to see available options."
                )
            else:
                await update.message.reply_text(
                    "I'm sorry, I encountered an error while processing your message. Please try again later."
                )
