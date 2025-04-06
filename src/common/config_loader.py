"""
Configuration loader for the AI Trading Agent.
Provides functionality to load and validate configuration from YAML files.
"""
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from src.common.logging_config import logger # Use relative import

# Load environment variables from .env file
load_dotenv()

class ConfigLoader:
    """
    Handles loading and validating configuration from YAML files.
    """

    def __init__(self, config_path=None):
        """
        Initialize the ConfigLoader.

        Args:
            config_path (str, optional): Path to the configuration file.
                If not provided, uses the default config path.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = None

    def _get_default_config_path(self):
        """
        Get the default configuration path.

        Returns:
            str: Path to the default configuration file.
        """
        # Start with the directory of this file
        current_dir = Path(__file__).parent.absolute()

        # Navigate to the project root (assuming src/common/config_loader.py structure)
        project_root = current_dir.parent.parent

        # Default config path
        return os.path.join(project_root, 'config', 'config.yaml')

    def load_config(self):
        """
        Load the configuration from the YAML file.

        Returns:
            dict: The loaded configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            yaml.YAMLError: If the YAML file is invalid.
        """
        try:
            logger.info(f"Loading configuration from {self.config_path}")
            with open(self.config_path, 'r') as config_file:
                self.config = yaml.safe_load(config_file)
            # Pass the loaded config to the logger setup
            from src.common.logging_config import setup_logging # Re-import to use the loaded config
            setup_logging(self.config)
            logger.info("Configuration loaded successfully")
            return self.config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise

    def get_config(self):
        """
        Get the loaded configuration. Loads it first if not already loaded.

        Returns:
            dict: The loaded configuration.
        """
        if self.config is None:
            return self.load_config()
        return self.config

    def get_value(self, key_path, default=None):
        """
        Get a specific value from the configuration using a dot-notation path.

        Args:
            key_path (str): Dot-notation path to the configuration value (e.g., 'system.log_level').
            default: Value to return if the key doesn't exist.

        Returns:
            The value at the specified path, or the default if not found.
        """
        if self.config is None:
            self.load_config()

        # Navigate through the nested dictionary using the key path
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Configuration key '{key_path}' not found, using default: {default}")
            return default


# Create a singleton instance for global use
config_loader = ConfigLoader()

# Function to get the global configuration
def get_config():
    """
    Get the global configuration.

    Returns:
        dict: The loaded configuration.
    """
    return config_loader.get_config()

# Function to get a specific configuration value
def get_config_value(key_path, default=None):
    """
    Get a specific value from the global configuration.

    Args:
        key_path (str): Dot-notation path to the configuration value.
        default: Value to return if the key doesn't exist.

    Returns:
        The value at the specified path, or the default if not found.
    """
    return config_loader.get_value(key_path, default)
