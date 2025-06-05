#!/usr/bin/env python
"""
Improved Telegram Bot Service

This script runs the Telegram bot as a background service,
handling commands and notifications for the Trading-Agent system.
Added startup conflict detection to prevent multiple instances.
"""

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
import psutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("telegram_bot_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("telegram_bot_service")

# Import the improved Telegram settings command handler
try:
    from improved_telegram_settings_command import EnhancedTelegramNotifier
except ImportError:
    try:
        from telegram_settings_command import EnhancedTelegramNotifier
    except ImportError:
        logger.error("Failed to import EnhancedTelegramNotifier")
        sys.exit(1)

class TelegramBotService:
    """Telegram bot service"""
    
    def __init__(self, env_path=None):
        """Initialize Telegram bot service
        
        Args:
            env_path: Path to .env file (optional)
        """
        self.env_path = env_path
        self.notifier = None
        self.running = False
        self.pid_file = "telegram_bot_service.pid"
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
    
    def handle_signal(self, sig, frame):
        """Handle signals
        
        Args:
            sig: Signal number
            frame: Frame
        """
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
    
    def check_existing_instance(self):
        """Check for existing Telegram bot service instance
        
        Returns:
            bool: True if no conflict, False if conflict detected
        """
        # Method 1: Check for PID file
        if os.path.exists(self.pid_file):
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process with this PID exists
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    # Verify it's a Python process with telegram in the command line
                    if "python" in process.name().lower() and any("telegram" in cmd.lower() for cmd in process.cmdline()):
                        logger.warning(f"Existing Telegram bot service found with PID {pid}")
                        return False
            except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.info(f"PID file exists but process check failed: {str(e)}")
                # PID file exists but process doesn't, so we can remove the stale PID file
                os.remove(self.pid_file)
        
        # Method 2: Check for running processes with telegram_bot_service in command line
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Skip current process
                if proc.pid == os.getpid():
                    continue
                
                # Check if it's a Python process with telegram_bot_service in command line
                if "python" in proc.name().lower() and any("telegram_bot_service" in cmd.lower() for cmd in proc.cmdline()):
                    logger.warning(f"Existing Telegram bot service found with PID {proc.pid}")
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # No conflict detected
        return True
    
    def write_pid_file(self):
        """Write PID file"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            logger.info(f"PID file written: {self.pid_file}")
        except Exception as e:
            logger.error(f"Error writing PID file: {str(e)}")
    
    def remove_pid_file(self):
        """Remove PID file"""
        if os.path.exists(self.pid_file):
            try:
                os.remove(self.pid_file)
                logger.info(f"PID file removed: {self.pid_file}")
            except Exception as e:
                logger.error(f"Error removing PID file: {str(e)}")
    
    def start(self):
        """Start Telegram bot service"""
        if self.running:
            logger.warning("Telegram bot service already running")
            return
        
        logger.info("Starting Telegram bot service...")
        
        # Check for existing instance
        if not self.check_existing_instance():
            logger.error("Another instance of Telegram bot service is already running")
            logger.error("Use --force to terminate existing instance and start a new one")
            return
        
        try:
            # Write PID file
            self.write_pid_file()
            
            # Initialize notifier
            self.notifier = EnhancedTelegramNotifier(self.env_path)
            
            # Start notifier
            self.notifier.start()
            
            self.running = True
            
            logger.info("Telegram bot service started")
            
            # Send startup notification
            self.notifier.send_system_notification(f"Trading-Agent Telegram bot service started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Keep running until stopped
            while self.running:
                time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error starting Telegram bot service: {str(e)}")
            self.stop()
    
    def stop(self):
        """Stop Telegram bot service"""
        if not self.running:
            return
        
        logger.info("Stopping Telegram bot service...")
        
        try:
            # Stop notifier
            if self.notifier:
                self.notifier.stop()
            
            self.running = False
            
            # Remove PID file
            self.remove_pid_file()
            
            logger.info("Telegram bot service stopped")
        
        except Exception as e:
            logger.error(f"Error stopping Telegram bot service: {str(e)}")

def terminate_existing_instance():
    """Terminate existing Telegram bot service instance
    
    Returns:
        bool: True if terminated, False otherwise
    """
    terminated = False
    
    # Check for running processes with telegram_bot_service in command line
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Skip current process
            if proc.pid == os.getpid():
                continue
            
            # Check if it's a Python process with telegram_bot_service in command line
            if "python" in proc.name().lower() and any("telegram_bot_service" in cmd.lower() for cmd in proc.cmdline()):
                logger.info(f"Terminating existing Telegram bot service with PID {proc.pid}")
                proc.terminate()
                terminated = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return terminated

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Telegram Bot Service")
    parser.add_argument("--env", help="Path to .env file")
    parser.add_argument("--force", action="store_true", help="Force start by terminating existing instance")
    args = parser.parse_args()
    
    # Find .env file
    env_path = args.env
    if not env_path:
        possible_paths = [
            '.env-secure/.env',
            '.env',
            '../.env-secure/.env',
            '../.env'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                env_path = path
                logger.info(f"Found .env file at: {os.path.abspath(path)}")
                break
    
    # Force start if requested
    if args.force:
        if terminate_existing_instance():
            logger.info("Existing instance terminated")
            # Wait for termination to complete
            time.sleep(2)
    
    # Start service
    service = TelegramBotService(env_path)
    service.start()

if __name__ == "__main__":
    main()
