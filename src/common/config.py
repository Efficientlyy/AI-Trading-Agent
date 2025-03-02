"""Configuration management system.

This module handles loading, validating, and accessing configuration from YAML files
and environment variables.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, create_model, validator

# Load environment variables from .env file
load_dotenv()

# Define configuration paths
CONFIG_DIR = Path("config")
DEFAULT_CONFIG_FILE = CONFIG_DIR / "system.yaml"


class ConfigurationError(Exception):
    """Raised when there's an issue with configuration loading or validation."""

    pass


class ConfigValidator:
    """Validates configuration against schemas."""

    @staticmethod
    def validate_config(config: Dict[str, Any], schema_model: type) -> Dict[str, Any]:
        """
        Validate configuration against a Pydantic schema.

        Args:
            config: Configuration dictionary to validate
            schema_model: Pydantic model class for validation

        Returns:
            Validated configuration dictionary

        Raises:
            ConfigurationError: If validation fails
        """
        try:
            validated = schema_model(**config)
            return validated.dict()
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}") from e


class ConfigLoader:
    """Loads configuration from files and environment variables."""

    @staticmethod
    def load_yaml_file(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            filepath: Path to the YAML configuration file

        Returns:
            Dictionary containing the configuration

        Raises:
            ConfigurationError: If the file cannot be loaded
        """
        try:
            with open(filepath, "r") as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration file {filepath}: {str(e)}") from e

    @staticmethod
    def override_from_env(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override configuration values from environment variables.

        Environment variables should follow the pattern:
        COMPONENT_SECTION_KEY=value

        Args:
            config: Configuration dictionary to override

        Returns:
            Updated configuration dictionary with environment overrides
        """
        result = config.copy()

        # Get all environment variables
        for env_key, env_value in os.environ.items():
            # Check if this is a configuration override (all uppercase)
            if not env_key.isupper() or "_" not in env_key:
                continue

            # Convert from ENV_VAR format to nested dict format
            parts = env_key.lower().split("_")
            
            # Need at least 2 parts (component_key)
            if len(parts) < 2:
                continue

            # Navigate to the correct part of the config
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value, converting to the appropriate type
            key = parts[-1]
            try:
                # Try to infer the type from the existing value if it exists
                if key in current and current[key] is not None:
                    # Convert the string to the same type as the existing value
                    value_type = type(current[key])
                    if value_type == bool:
                        # Special handling for boolean values
                        current[key] = env_value.lower() in ("true", "1", "yes")
                    elif value_type == int:
                        current[key] = int(env_value)
                    elif value_type == float:
                        current[key] = float(env_value)
                    elif value_type == list:
                        # Split by commas for lists
                        current[key] = [item.strip() for item in env_value.split(",")]
                    else:
                        current[key] = env_value
                else:
                    # If the key doesn't exist yet, just set it as a string
                    current[key] = env_value
            except ValueError:
                # If type conversion fails, just use the string value
                current[key] = env_value

        return result


class Config:
    """
    Central configuration manager.
    
    This class provides access to configuration from all sources,
    with proper validation and overrides.
    """

    _instance = None
    _config: Dict[str, Any] = {}
    _loaded_files: set = set()
    _initialized = False

    def __new__(cls):
        """Implement the singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the configuration system."""
        if not self._initialized:
            self._load_system_config()
            self._initialized = True

    def _load_system_config(self) -> None:
        """Load the main system configuration file."""
        self.load_config_file(DEFAULT_CONFIG_FILE)

    def load_config_file(self, filepath: Union[str, Path]) -> None:
        """
        Load a configuration file and merge it with the existing configuration.

        Args:
            filepath: Path to the configuration file

        Raises:
            ConfigurationError: If the file cannot be loaded or is invalid
        """
        filepath = Path(filepath)
        if filepath in self._loaded_files:
            return

        # Load the file
        config = ConfigLoader.load_yaml_file(filepath)
        
        # Override with environment variables
        config = ConfigLoader.override_from_env(config)
        
        # Merge with existing configuration
        self._merge_config(config)
        
        # Mark this file as loaded
        self._loaded_files.add(filepath)

    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """
        Merge new configuration with the existing configuration.

        Args:
            new_config: New configuration to merge
        """
        self._config = self._merge_dicts(self._config, new_config)

    @staticmethod
    def _merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries.

        Args:
            dict1: First dictionary
            dict2: Second dictionary (overrides dict1)

        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = Config._merge_dicts(result[key], value)
            else:
                # Override or add the key
                result[key] = value
                
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a dot-notation path.
        
        Args:
            key_path: Dot-notation path to the configuration value (e.g., "system.logging.level")
            default: Default value to return if the key doesn't exist
            
        Returns:
            The configuration value or the default if not found
        """
        parts = key_path.split(".")
        value = self._config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            Dictionary containing all configuration
        """
        return self._config.copy()


# Create a singleton instance
config = Config() 