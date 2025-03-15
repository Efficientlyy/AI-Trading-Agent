"""Example of using configuration schema validation."""

import logging
from pathlib import Path

from src.common.config import Config, ConfigurationError
from src.common.config_schema import create_example_schema


def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_with_valid_config():
    """Example with a valid configuration."""
    # Create a new Config instance
    config = Config()
    
    # Set the schema
    schema = create_example_schema()
    config.set_schema(schema)
    
    # Add some valid configuration
    config.set("system.logging.level", "INFO")
    config.set("exchanges.binance.enabled", True)
    config.set("exchanges.binance.api_key", "sample_key")
    config.set("exchanges.binance.api_secret", "sample_secret")
    config.set("strategies.ma_crossover.enabled", True)
    config.set("strategies.ma_crossover.symbols", ["BTC/USDT", "ETH/USDT"])
    config.set("strategies.ma_crossover.fast_period", 15)
    config.set("strategies.ma_crossover.slow_period", 50)
    
    # Validate
    result = config.validate_config()
    
    if result.is_valid:
        logging.info("Configuration is valid!")
        logging.info("Configured exchanges: %s", list(config.get("exchanges", {}).keys()))
        logging.info("Configured strategies: %s", list(config.get("strategies", {}).keys()))
        
        # Access some configuration values
        binance_config = config.get("exchanges.binance")
        logging.info("Binance configuration: %s", binance_config)
        
        ma_strategy = config.get("strategies.ma_crossover")
        logging.info("MA Crossover strategy configuration: %s", ma_strategy)
    else:
        logging.error("Configuration is invalid!")
        for error in result.errors:
            logging.error("  - %s", error)


def example_with_invalid_config():
    """Example with an invalid configuration."""
    # Create a new Config instance
    config = Config()
    
    # Set the schema
    schema = create_example_schema()
    config.set_schema(schema)
    
    # Add some invalid configuration
    config.set("exchanges.binance.enabled", "not_a_boolean")  # Should be a boolean
    config.set("exchanges.binance.api_key", 12345)  # Should be a string
    # Missing required api_secret
    config.set("strategies.ma_crossover.fast_period", 1)  # Below minimum
    config.set("strategies.ma_crossover.slow_period", 1000)  # Above maximum
    
    # Validate
    try:
        # This will raise an exception when loading a config file
        dummy_path = Path("dummy_config.yaml")
        config._loaded_files.add(dummy_path)  # Mark as loaded to avoid file not found
        
        # Manual validation
        result = config.validate_config()
        
        if result.is_valid:
            logging.info("Configuration is valid!")
        else:
            logging.error("Configuration is invalid!")
            for error in result.errors:
                logging.error("  - %s", error)
    except ConfigurationError as e:
        logging.error("Configuration validation failed: %s", str(e))


def main():
    """Run the examples."""
    setup_logging()
    
    logging.info("Running example with valid configuration...")
    example_with_valid_config()
    
    logging.info("\nRunning example with invalid configuration...")
    example_with_invalid_config()


if __name__ == "__main__":
    main()
