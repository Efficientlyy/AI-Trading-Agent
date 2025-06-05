#!/usr/bin/env python
"""
Telegram Integration for System Overseer.

This module provides the TelegramIntegration class for interacting with Telegram.
"""

import os
import sys
import json
import logging
import threading
import time
import asyncio
from typing import Dict, Any, List, Optional
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
    """Telegram Integration for System Overseer."""
    
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
        self.loop = None
        
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
            # Create bot instance
            self.bot = Bot(token=self.token)
            
            # Create application
            self.application = Application.builder().token(self.token).build()
            
            # Register handlers
            self.application.add_handler(CommandHandler("start", self._start_command))
            self.application.add_handler(CommandHandler("help", self._help_command))
            self.application.add_handler(CommandHandler("status", self._status_command))
            self.application.add_handler(CommandHandler("pairs", self._pairs_command))
            self.application.add_handler(CommandHandler("add_pair", self._add_pair_command))
            self.application.add_handler(CommandHandler("remove_pair", self._remove_pair_command))
            self.application.add_handler(CommandHandler("notifications", self._notifications_command))
            
            # Register message handler
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._message_handler))
            
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
        
        if not self.application:
            logger.error("Telegram bot not initialized")
            return False
        
        try:
            # Set running flag
            self.running = True
            
            # Start bot in a separate thread
            self.thread = threading.Thread(target=self._run_bot)
            self.thread.daemon = True
            self.thread.start()
            
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
            # Set running flag
            self.running = False
            
            # Stop application using asyncio
            if self.application and self.loop:
                # Create a future to run stop in the event loop
                future = asyncio.run_coroutine_threadsafe(self.application.stop(), self.loop)
                # Wait for the future to complete with timeout
                try:
                    future.result(timeout=5)
                except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                    logger.warning("Timeout while stopping Telegram application")
            
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
        """Run Telegram bot."""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run the application
            self.loop.run_until_complete(self._start_polling())
            
            # Run the event loop
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"Error in Telegram bot thread: {e}")
        finally:
            # Clean up
            if self.loop and self.loop.is_running():
                self.loop.stop()
            if self.loop and not self.loop.is_closed():
                self.loop.close()
            self.loop = None
            self.running = False
    
    async def _start_polling(self):
        """Start polling for updates."""
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        # Monitor the running flag
        while self.running:
            await asyncio.sleep(1)
        
        # Stop polling when running flag is False
        await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()
    
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
            "You can also ask me questions in natural language, and I'll do my best to help!"
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
        
        # Process message with dialogue manager
        if self.dialogue_manager:
            try:
                # Get response from dialogue manager
                response = await self.dialogue_manager.process_message(message_text)
                
                # Send response
                await update.message.reply_text(response)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await update.message.reply_text(
                    "I'm sorry, I encountered an error while processing your message. Please try again later."
                )
        else:
            # Default response if dialogue manager not available
            await update.message.reply_text(
                "I understand you're trying to communicate with me, but my natural language processing capabilities are currently limited. "
                "Please use commands like /help, /status, etc. for now."
            )
    
    def send_message(self, message: str):
        """Send message to Telegram chat.
        
        Args:
            message: Message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.bot or not self.chat_id:
            logger.error("Telegram bot or chat ID not available")
            return False
        
        try:
            # Create a new event loop for this thread if needed
            if not asyncio.get_event_loop().is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._send_message_async(message))
            else:
                # Use existing event loop
                asyncio.create_task(self._send_message_async(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def _send_message_async(self, message: str):
        """Send message to Telegram chat asynchronously.
        
        Args:
            message: Message to send
        """
        await self.bot.send_message(chat_id=self.chat_id, text=message)
    
    def send_notification(self, notification_type: str, message: str):
        """Send notification to Telegram chat.
        
        Args:
            notification_type: Notification type (all, signals, trades, errors)
            message: Message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get config registry
        config_registry = self.system_core.get_service("config_registry") if self.system_core else None
        
        # Get notification level
        notification_level = "all"
        if config_registry:
            notification_level = config_registry.get_config("notifications.level", "all")
        
        # Check if notification should be sent
        if notification_level == "none":
            return True
        
        if notification_level != "all" and notification_type != notification_level:
            if notification_type != "errors" or notification_level != "errors":
                return True
        
        # Format notification
        notification = f"[{notification_type.upper()}] {message}"
        
        # Send notification
        return self.send_message(notification)
