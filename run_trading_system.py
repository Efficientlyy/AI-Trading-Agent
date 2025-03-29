#!/usr/bin/env python3
"""
Production launcher for the AI Crypto Trading System.

This script provides a robust entry point for running the trading system
in a production environment, with proper error handling, logging, and
configuration management.
"""

import asyncio
import os
import sys
import signal
import argparse
import logging
from pathlib import Path
import traceback
import yaml
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.main import Application
from src.common.logging import setup_logging, get_logger, system_logger
from src.common.config import config
from src.common.security.api_keys import APIKeyManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Crypto Trading System")
    parser.add_argument(
        "--config-dir", 
        type=str, 
        default="config",
        help="Path to configuration directory"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        choices=["development", "testing", "production"],
        default="development",
        help="Environment to run in"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override logging level"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Validate configuration and components without executing trades"
    )
    parser.add_argument(
        "--component", 
        type=str, 
        action="append",
        help="Specific components to enable (can be used multiple times)"
    )
    parser.add_argument(
        "--skip-component", 
        type=str, 
        action="append",
        help="Specific components to disable (can be used multiple times)"
    )
    parser.add_argument(
        "--validate-keys", 
        action="store_true",
        help="Validate API keys before starting"
    )
    
    return parser.parse_args()


async def check_api_keys(args):
    """Check that all required API keys are available and valid."""
    system_logger.info("Validating API keys")
    try:
        # Initialize the API key manager
        key_manager = APIKeyManager()
        await key_manager.initialize()
        
        # Get all required exchanges
        exchanges_config = config.get("execution.exchanges", {})
        required_exchanges = [
            name for name, cfg in exchanges_config.items() 
            if not cfg.get("type") == "mock"
        ]
        
        # Check if we have keys for all required exchanges
        missing_keys = []
        for exchange in required_exchanges:
            if not key_manager.has_keys(exchange):
                missing_keys.append(exchange)
        
        if missing_keys:
            system_logger.error(
                "Missing API keys for exchanges", 
                exchanges=missing_keys
            )
            return False
        
        # Validate each key if requested
        if args.validate_keys:
            validation_results = {}
            for exchange in required_exchanges:
                try:
                    valid = await key_manager.validate_keys(exchange)
                    validation_results[exchange] = valid
                    if not valid:
                        system_logger.error(
                            "API key validation failed",
                            exchange=exchange
                        )
                except Exception as e:
                    system_logger.error(
                        "Error validating API keys",
                        exchange=exchange,
                        error=str(e)
                    )
                    validation_results[exchange] = False
            
            # Check if any validations failed
            if not all(validation_results.values()):
                system_logger.error(
                    "Some API key validations failed",
                    results=validation_results
                )
                return False
        
        system_logger.info("API key validation successful")
        return True
    
    except Exception as e:
        system_logger.exception("API key validation error", error=str(e))
        return False


def load_environment_config(args):
    """Load environment-specific configuration."""
    # Base system config
    system_config_path = Path(args.config_dir) / "system.yaml"
    if not system_config_path.exists():
        system_logger.error("System configuration file not found", path=str(system_config_path))
        return False
    
    try:
        # Load the system config first
        config.load_config_file(system_config_path)
        
        # Set environment in config
        config.set("system.environment", args.env)
        
        # Override logging level if specified
        if args.log_level:
            config.set("logging.level", args.log_level)

        # Set dry run mode if specified
        if args.dry_run:
            config.set("system.dry_run", True)
            # Disable execution in dry run mode
            config.set("execution.enabled", False)
            # Make sure we use mock exchange in dry run mode
            for exchange_name, exchange_config in config.get("execution.exchanges", {}).items():
                config.set(f"execution.exchanges.{exchange_name}.type", "mock")
                config.set(f"execution.exchanges.{exchange_name}.paper_trading", True)
        
        # Apply component-specific overrides
        if args.component:
            # First disable all components
            for component in ["data_collection", "analysis_agents", "strategies", "portfolio", "execution"]:
                config.set(f"{component}.enabled", False)
            
            # Then enable only the specified ones
            for component in args.component:
                config.set(f"{component}.enabled", True)
        
        # Skip specific components if requested
        if args.skip_component:
            for component in args.skip_component:
                config.set(f"{component}.enabled", False)
        
        # Load environment-specific config file if it exists
        env_config_path = Path(args.config_dir) / f"{args.env}.yaml"
        if env_config_path.exists():
            config.load_config_file(env_config_path)
            system_logger.info("Loaded environment-specific configuration", env=args.env)
        
        # Apply any additional environment variables with prefix TRADING_
        for key, value in os.environ.items():
            if key.startswith("TRADING_"):
                config_key = key[8:].lower().replace("__", ".").replace("_", ".")
                config.set(config_key, value)
                system_logger.debug("Applied environment variable override", key=config_key)
        
        return True
    
    except Exception as e:
        system_logger.exception("Error loading configuration", error=str(e))
        return False


def validate_system_dependencies():
    """Validate system dependencies and requirements."""
    system_logger.info("Validating system dependencies")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        system_logger.error(
            "Unsupported Python version. Python 3.8+ required",
            version=f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        )
        return False
    
    # Check if required modules are installed
    required_modules = [
        "numpy", "pandas", "plotly", "dash", "sqlalchemy", 
        "aiohttp", "websockets", "pydantic"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        system_logger.error("Missing required Python modules", modules=missing_modules)
        return False
    
    # Check if Rust extensions are available (if needed)
    use_rust = config.get("system.use_rust_extensions", False)
    if use_rust:
        try:
            from src.rust_bridge import market_data_py
            system_logger.info("Rust extensions available")
        except ImportError:
            system_logger.warning(
                "Rust extensions not available, will use Python fallbacks",
                extensions=["market_data_py"]
            )
            # Don't fail, just warn - we have Python fallbacks
    
    # All checks passed
    system_logger.info("System dependency validation passed")
    return True


def configure_logging(args):
    """Configure logging based on environment and arguments."""
    # Get log level from config
    log_level = config.get("logging.level", "INFO")
    
    # Override from command line if specified
    if args.log_level:
        log_level = args.log_level
    
    # Convert string to logging level
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    level = log_level_map.get(log_level, logging.INFO)
    
    # Configure structured logging
    setup_logging(level=level)
    
    # Log to file if in production
    if args.env == "production":
        # Ensure log directory exists
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create a log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"trading_system_{timestamp}.log"
        
        # Update config to enable file logging
        config.set("logging.output.file", True)
        config.set("logging.file_path", str(log_file))
    
    system_logger.info(
        "Logging configured", 
        level=log_level, 
        env=args.env,
        file_logging=config.get("logging.output.file", False)
    )


async def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Load configuration
        if not load_environment_config(args):
            system_logger.error("Failed to load configuration")
            return 1
        
        # Configure logging
        configure_logging(args)
        
        # Validate system dependencies
        if not validate_system_dependencies():
            system_logger.error("System dependency validation failed")
            return 1
        
        # Check API keys
        if not await check_api_keys(args):
            system_logger.error("API key validation failed")
            if args.env == "production":
                return 1
            else:
                system_logger.warning("Continuing despite API key validation failure (non-production environment)")
        
        # Create and start the application
        system_logger.info("Starting AI Trading System", environment=args.env)
        app = Application()
        await app.start()
        
        return 0
    
    except KeyboardInterrupt:
        system_logger.info("Application interrupted by user")
        return 0
    
    except Exception as e:
        system_logger.exception("Unhandled exception in main", error=str(e))
        return 1


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)