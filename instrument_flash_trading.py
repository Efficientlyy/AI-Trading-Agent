#!/usr/bin/env python
"""
Instrumented Flash Trading Module for Runtime Debugging

This script adds instrumentation to flash_trading.py to debug API key propagation issues.
"""

import os
import sys
import logging
import time
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
logger = logging.getLogger("runtime_debug")

def instrument_flash_trading():
    """Instrument flash_trading.py with debug logging"""
    
    # Path to the flash_trading.py file
    script_path = "flash_trading.py"
    
    # Check if file exists
    if not os.path.exists(script_path):
        logger.error(f"Script file not found: {script_path}")
        return False
    
    # Read the file
    try:
        with open(script_path, 'r') as f:
            content = f.read()
            logger.info(f"Successfully read script file. Content length: {len(content)}")
    except Exception as e:
        logger.error(f"Error reading script file: {str(e)}")
        return False
    
    # Add instrumentation to main function
    if "if __name__ == \"__main__\":" in content:
        # Find the main block
        main_idx = content.find("if __name__ == \"__main__\":")
        if main_idx == -1:
            logger.error("Could not find main block")
            return False
        
        # Create instrumented main block
        instrumented_main = """
if __name__ == "__main__":
    # Debug instrumentation
    import logging
    debug_logger = logging.getLogger("runtime_debug")
    debug_logger.setLevel(logging.DEBUG)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flash Trading System")
    parser.add_argument("--env", help="Path to .env file")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--duration", type=int, default=0, help="Run duration in seconds (0 for indefinite)")
    parser.add_argument("--reset", action="store_true", help="Reset paper trading state")
    args = parser.parse_args()
    
    # Debug environment variables
    debug_logger.info(f"Command line arguments: {args}")
    
    # Check environment before loading
    api_key_before = os.environ.get('MEXC_API_KEY')
    api_secret_before = os.environ.get('MEXC_API_SECRET')
    debug_logger.info(f"Before env loading - MEXC_API_KEY: {api_key_before}")
    debug_logger.info(f"Before env loading - MEXC_API_SECRET: {api_secret_before}")
    
    # Load environment if specified
    if args.env:
        debug_logger.info(f"Loading environment from: {args.env}")
        try:
            load_dotenv(args.env)
            debug_logger.info("load_dotenv called successfully")
        except Exception as e:
            debug_logger.error(f"Error in load_dotenv: {str(e)}")
    
    # Check environment after loading
    api_key_after = os.environ.get('MEXC_API_KEY')
    api_secret_after = os.environ.get('MEXC_API_SECRET')
    debug_logger.info(f"After env loading - MEXC_API_KEY: {api_key_after}")
    debug_logger.info(f"After env loading - MEXC_API_SECRET: {api_secret_after}")
    
    # Reset paper trading state if requested
    if args.reset:
        debug_logger.info("Resetting paper trading state")
        if os.path.exists("paper_trading_state.json"):
            os.remove("paper_trading_state.json")
    
    # Create flash trading system
    debug_logger.info(f"Creating FlashTradingSystem with env_path={args.env}, config_path={args.config}")
    flash_trading = FlashTradingSystem(env_path=args.env, config_path=args.config)
    
    # Debug API key in flash_trading.client
    debug_logger.info(f"FlashTradingSystem client API key type: {type(flash_trading.client.api_key)}")
    debug_logger.info(f"FlashTradingSystem client API key: {flash_trading.client.api_key}")
    
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
"""
        
        # Replace the main block
        end_idx = content.find("\n", main_idx + 1)
        while content[end_idx+1] == ' ' or content[end_idx+1] == '\t':
            end_idx = content.find("\n", end_idx + 1)
            if end_idx == -1:
                end_idx = len(content)
                break
        
        new_content = content[:main_idx] + instrumented_main
        
        # Write the instrumented file
        try:
            with open("instrumented_flash_trading.py", 'w') as f:
                f.write(new_content)
            logger.info("Successfully wrote instrumented script file")
        except Exception as e:
            logger.error(f"Error writing instrumented script file: {str(e)}")
            return False
    
    # Add instrumentation to FlashTradingSystem.__init__
    if "def __init__(self, env_path=None, config_path=None):" in content:
        # Find the __init__ method
        init_idx = content.find("def __init__(self, env_path=None, config_path=None):")
        if init_idx == -1:
            logger.error("Could not find __init__ method")
            return False
        
        # Create instrumented __init__ method
        instrumented_init = """    def __init__(self, env_path=None, config_path=None):
        \"\"\"Initialize the flash trading system\"\"\"
        # Debug instrumentation
        import logging
        debug_logger = logging.getLogger("runtime_debug")
        debug_logger.info(f"FlashTradingSystem.__init__ called with env_path={env_path}, config_path={config_path}")
        
        # Check environment variables
        api_key = os.environ.get('MEXC_API_KEY')
        api_secret = os.environ.get('MEXC_API_SECRET')
        debug_logger.info(f"In __init__ - MEXC_API_KEY: {api_key}")
        debug_logger.info(f"In __init__ - MEXC_API_SECRET: {api_secret}")
        
        # Load configuration
        self.config = FlashTradingConfig(config_path)
        
        # Create API client with direct credentials if available
        if api_key and api_secret:
            debug_logger.info("Creating OptimizedMEXCClient with direct credentials")
            self.client = OptimizedMEXCClient(api_key=api_key, secret_key=api_secret)
        else:
            debug_logger.info(f"Creating OptimizedMEXCClient with env_path={env_path}")
            self.client = OptimizedMEXCClient(env_path=env_path)
        
        # Debug client API key
        debug_logger.info(f"Created client API key type: {type(self.client.api_key)}")
        debug_logger.info(f"Created client API key: {self.client.api_key}")
        
        # Create paper trading system
        self.paper_trading = PaperTradingSystem(self.client, self.config)
        
        # Create signal generator
        self.signal_generator = SignalGenerator(self.client, env_path)
"""
        
        # Replace the __init__ method
        end_idx = content.find("    def ", init_idx + 1)
        if end_idx == -1:
            end_idx = len(content)
        
        new_content = content[:init_idx] + instrumented_init + content[end_idx:]
        
        # Write the instrumented file
        try:
            with open("instrumented_flash_trading.py", 'w') as f:
                f.write(new_content)
            logger.info("Successfully wrote instrumented script file with __init__ changes")
        except Exception as e:
            logger.error(f"Error writing instrumented script file: {str(e)}")
            return False
    
    return True

if __name__ == "__main__":
    success = instrument_flash_trading()
    
    if success:
        logger.info("Successfully instrumented flash_trading.py")
        sys.exit(0)
    else:
        logger.error("Failed to instrument flash_trading.py")
        sys.exit(1)
