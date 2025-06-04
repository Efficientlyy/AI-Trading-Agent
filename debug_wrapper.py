#!/usr/bin/env python
'''
Debug wrapper for flash_trading.py
'''

import os
import sys
import time
import logging
import argparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_runtime.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug_wrapper")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Flash Trading System Debug Wrapper")
parser.add_argument("--env", help="Path to .env file")
parser.add_argument("--config", help="Path to configuration file")
parser.add_argument("--duration", type=int, default=0, help="Run duration in seconds (0 for indefinite)")
parser.add_argument("--reset", action="store_true", help="Reset paper trading state")
args = parser.parse_args()

# Log command line arguments
logger.info(f"Command line arguments: {args}")

# Check environment before loading
api_key_before = os.environ.get('MEXC_API_KEY')
api_secret_before = os.environ.get('MEXC_API_SECRET')
logger.info(f"Before env loading - MEXC_API_KEY: {api_key_before}")
logger.info(f"Before env loading - MEXC_API_SECRET: {api_secret_before}")

# Load environment if specified
if args.env:
    logger.info(f"Loading environment from: {args.env}")
    try:
        # Check if file exists
        if not os.path.exists(args.env):
            logger.error(f"Environment file not found: {args.env}")
        else:
            # Read file content
            with open(args.env, 'r') as f:
                content = f.read()
                logger.info(f"Environment file content: {content}")
            
            # Load environment
            load_dotenv(args.env)
            logger.info("load_dotenv called successfully")
    except Exception as e:
        logger.error(f"Error in load_dotenv: {str(e)}")

# Check environment after loading
api_key_after = os.environ.get('MEXC_API_KEY')
api_secret_after = os.environ.get('MEXC_API_SECRET')
logger.info(f"After env loading - MEXC_API_KEY: {api_key_after}")
logger.info(f"After env loading - MEXC_API_SECRET: {api_secret_after}")

# Monkey patch OptimizedMEXCClient.__init__ to log API key
try:
    from optimized_mexc_client import OptimizedMEXCClient
    original_init = OptimizedMEXCClient.__init__
    
    def patched_init(self, api_key=None, secret_key=None, env_path=None):
        logger.info(f"OptimizedMEXCClient.__init__ called with api_key={api_key}, secret_key={secret_key}, env_path={env_path}")
        
        # Call original init
        original_init(self, api_key, secret_key, env_path)
        
        # Log API key after initialization
        logger.info(f"OptimizedMEXCClient API key type: {type(self.api_key)}")
        logger.info(f"OptimizedMEXCClient API key: {self.api_key}")
    
    OptimizedMEXCClient.__init__ = patched_init
    logger.info("Successfully monkey patched OptimizedMEXCClient.__init__")
except Exception as e:
    logger.error(f"Error monkey patching OptimizedMEXCClient.__init__: {str(e)}")

# Import flash_trading module
try:
    from flash_trading import FlashTradingSystem
    logger.info("Successfully imported FlashTradingSystem")
except Exception as e:
    logger.error(f"Error importing FlashTradingSystem: {str(e)}")
    sys.exit(1)

# Create flash trading system
logger.info(f"Creating FlashTradingSystem with env_path={args.env}, config_path={args.config}")
flash_trading = FlashTradingSystem(env_path=args.env, config_path=args.config)

# Debug API key in flash_trading.client
logger.info(f"FlashTradingSystem client API key type: {type(flash_trading.client.api_key)}")
logger.info(f"FlashTradingSystem client API key: {flash_trading.client.api_key}")

# Run for specified duration
if args.duration > 0:
    print(f"Running flash trading system for {args.duration} seconds...")
    flash_trading.run_for_duration(args.duration)
else:
    print("Running flash trading system indefinitely (Ctrl+C to stop)...")
    try:
        flash_trading.start()
        while True:
            flash_trading.process_signals_and_execute()
            time.sleep(1)
    except KeyboardInterrupt:
        flash_trading.stop()
