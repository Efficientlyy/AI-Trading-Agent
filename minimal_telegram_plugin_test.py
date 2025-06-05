#!/usr/bin/env python
"""
Minimal Telegram Bot Plugin Test

This script tests the Telegram bot functionality in a minimal plugin context
to isolate and debug the application lifecycle issues.
"""

import os
import sys
import logging
import asyncio
import threading
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('minimal_telegram_plugin_test.log')
    ]
)

logger = logging.getLogger("minimal_telegram_plugin")

# Load environment variables
load_dotenv('.env-secure/.env')

# Import telegram library
try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
except ImportError:
    logger.error("Telegram library not installed. Please install python-telegram-bot.")
    sys.exit(1)

# Get Telegram token from environment
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
    sys.exit(1)

class MinimalTelegramPlugin:
    """A minimal Telegram bot plugin to test application lifecycle."""
    
    def __init__(self):
        """Initialize the plugin."""
        self.token = TELEGRAM_BOT_TOKEN
        self.bot = None
        self.application = None
        self.running = False
        self.thread = None
        self.loop = None
        
        logger.info("MinimalTelegramPlugin initialized")
    
    def start(self):
        """Start the plugin."""
        logger.info("Starting MinimalTelegramPlugin...")
        
        try:
            # Create bot instance for direct API calls
            self.bot = Bot(token=self.token)
            
            # Set running flag
            self.running = True
            
            # Start bot in a separate thread
            self.thread = threading.Thread(target=self._run_bot)
            self.thread.daemon = True
            self.thread.start()
            
            # Give the thread time to start up
            time.sleep(2)
            
            if not self.running:
                logger.error("MinimalTelegramPlugin failed to start")
                return False
                
            logger.info("MinimalTelegramPlugin started")
            return True
        except Exception as e:
            logger.error(f"Failed to start MinimalTelegramPlugin: {e}")
            self.running = False
            return False
    
    def stop(self):
        """Stop the plugin."""
        logger.info("Stopping MinimalTelegramPlugin...")
        
        try:
            # Set running flag to signal thread to stop
            self.running = False
            
            # Wait for thread to finish
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    logger.warning("MinimalTelegramPlugin thread did not stop gracefully")
            
            logger.info("MinimalTelegramPlugin stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop MinimalTelegramPlugin: {e}")
            return False
    
    def _run_bot(self):
        """Run the bot in a separate thread."""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run the bot
            self.loop.run_until_complete(self._run_telegram_application())
        except Exception as e:
            logger.error(f"Error in MinimalTelegramPlugin thread: {e}")
        finally:
            # Clean up
            if self.loop and not self.loop.is_closed():
                self.loop.close()
            self.loop = None
            self.running = False
            logger.info("MinimalTelegramPlugin thread exited")
    
    async def _run_telegram_application(self):
        """Run the Telegram application."""
        try:
            # Create application
            application = Application.builder().token(self.token).build()
            
            # Register handlers
            application.add_handler(CommandHandler("start", self._start_command))
            application.add_handler(CommandHandler("help", self._help_command))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._echo))
            
            # Start the application
            logger.info("Initializing application...")
            await application.initialize()
            logger.info("Starting application...")
            await application.start()
            logger.info("Starting polling...")
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
        except Exception as e:
            logger.error(f"Error in Telegram application: {e}")
            self.running = False
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        await update.message.reply_text("Hello! I'm a minimal test bot for debugging purposes.")
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await update.message.reply_text("This is a minimal test bot for debugging the application lifecycle.")
    
    async def _echo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Echo the user message."""
        await update.message.reply_text(f"You said: {update.message.text}")

def main():
    """Main function."""
    logger.info("Starting minimal Telegram plugin test...")
    
    # Create plugin
    plugin = MinimalTelegramPlugin()
    
    # Start plugin
    if not plugin.start():
        logger.error("Failed to start plugin")
        return
    
    try:
        # Keep running until user presses Ctrl-C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Test stopped by user")
    finally:
        # Stop plugin
        plugin.stop()
    
    logger.info("Minimal Telegram plugin test completed")

if __name__ == "__main__":
    main()
