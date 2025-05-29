"""
Configuration module for LLM Oversight in AI Trading Agent.

This module provides utilities for loading, validating, and managing
oversight configuration settings from config files or environment variables.
"""

import os
import yaml
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum

from ai_trading_agent.oversight.llm_oversight import OversightLevel, LLMProvider

# Set up logger
logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Sources for configuration settings."""
    DEFAULT = "default"
    CONFIG_FILE = "config_file"
    ENV_VAR = "environment_variable"
    RUNTIME = "runtime"


class OversightConfig:
    """
    Configuration manager for LLM oversight system.
    
    Handles loading configuration from files, environment variables,
    and provides validated settings for the oversight components.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        "llm": {
            "provider": "openai",
            "model_name": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 1000,
            "api_base": None,
            "timeout": 30.0,
            "retry_attempts": 3,
            "system_prompt": None
        },
        "oversight": {
            "level": "advise",
            "enable_market_analysis": True,
            "enable_signal_validation": True,
            "enable_order_validation": True,
            "enable_error_analysis": True,
            "enable_autonomous_recovery": True,
            "decision_log_path": "logs/oversight_decisions",
            "max_decision_cache_size": 100
        },
        "circuit_breaker": {
            "warning_threshold": 3,
            "failure_threshold": 5,
            "recovery_time_base": 30.0,
            "max_recovery_time": 600.0,
            "exponential_factor": 2.0,
            "reset_timeout": 300.0
        },
        "integrations": {
            "slack_notifications": False,
            "email_alerts": False,
            "dashboard_metrics": True
        }
    }
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        env_prefix: str = "TRADING_LLM_",
        validate: bool = True
    ):
        """
        Initialize the oversight configuration.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            env_prefix: Prefix for environment variables
            validate: Whether to validate the configuration on load
        """
        self.config_path = config_path
        self.env_prefix = env_prefix
        
        # Initialize configuration
        self.config: Dict[str, Any] = {}
        self.config_sources: Dict[str, Dict[str, ConfigSource]] = {}
        
        # Load configuration
        self._load_default_config()
        
        if config_path:
            self._load_config_file()
            
        self._load_environment_variables()
        
        # Validate configuration if requested
        if validate:
            self._validate_config()
            
        logger.info("Oversight configuration initialized")
    
    def _load_default_config(self) -> None:
        """Load default configuration."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Initialize sources tracking
        for section, section_config in self.config.items():
            self.config_sources[section] = {}
            for key in section_config:
                self.config_sources[section][key] = ConfigSource.DEFAULT
    
    def _load_config_file(self) -> None:
        """Load configuration from file."""
        if not self.config_path:
            return
            
        try:
            config_file = Path(self.config_path)
            
            if not config_file.exists():
                logger.warning(f"Configuration file not found: {self.config_path}")
                return
                
            # Load based on file extension
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
            else:
                logger.warning(f"Unsupported configuration file format: {config_file.suffix}")
                return
                
            # Update configuration
            self._update_config_recursive(self.config, file_config, ConfigSource.CONFIG_FILE)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
    
    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        for env_name, env_value in os.environ.items():
            # Check if variable starts with our prefix
            if not env_name.startswith(self.env_prefix):
                continue
                
            # Parse the variable name to determine config path
            config_path = env_name[len(self.env_prefix):].lower().split('_')
            
            if len(config_path) < 2:
                logger.warning(f"Skipping environment variable with invalid format: {env_name}")
                continue
                
            # The first element is the section, the rest form the key path
            section = config_path[0]
            key_path = config_path[1:]
            
            # Skip if section is not valid
            if section not in self.config:
                logger.warning(f"Skipping environment variable with unknown section: {env_name}")
                continue
                
            # Update the configuration
            self._update_config_from_env(section, key_path, env_value)
    
    def _update_config_from_env(self, section: str, key_path: List[str], value: str) -> None:
        """
        Update configuration from environment variable.
        
        Args:
            section: Configuration section
            key_path: Path to configuration key
            value: Value to set
        """
        # Start at the section level
        current = self.config[section]
        current_path = [section]
        
        # Navigate to the target node
        for i, key in enumerate(key_path[:-1]):
            current_path.append(key)
            
            # If key doesn't exist in current level, add it
            if key not in current:
                logger.warning(
                    f"Creating missing configuration path: {'.'.join(current_path)}"
                )
                current[key] = {}
                
            # If value is not a dict, we can't go deeper
            if not isinstance(current[key], dict):
                logger.warning(
                    f"Cannot set {'.'.join(current_path + key_path[i+1:])} because "
                    f"{'.'.join(current_path)} is not a dictionary"
                )
                return
                
            current = current[key]
            
        # Set the final value
        final_key = key_path[-1]
        current_path.append(final_key)
        
        # Convert string value to appropriate type based on default's type or key name
        try:
            # Try to determine expected type from existing value
            if final_key in current and current[final_key] is not None:
                target_type = type(current[final_key])
                
                if target_type == bool:
                    typed_value = value.lower() in ['true', 'yes', '1', 'y']
                elif target_type == int:
                    typed_value = int(value)
                elif target_type == float:
                    typed_value = float(value)
                elif target_type == list:
                    # Assume comma-separated list
                    typed_value = [item.strip() for item in value.split(',')]
                else:
                    typed_value = value
            else:
                # Guess type based on value format
                if value.lower() in ['true', 'false', 'yes', 'no', 'y', 'n', '1', '0']:
                    typed_value = value.lower() in ['true', 'yes', '1', 'y']
                elif value.isdigit():
                    typed_value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    typed_value = float(value)
                else:
                    typed_value = value
                    
            # Update the configuration
            current[final_key] = typed_value
            
            # Update the source
            path_str = '.'.join(current_path)
            section_sources = self.config_sources.setdefault(section, {})
            section_sources[path_str[len(section)+1:]] = ConfigSource.ENV_VAR
            
            logger.debug(f"Set configuration {path_str} = {typed_value} from environment")
            
        except Exception as e:
            logger.warning(f"Error setting configuration {'.'.join(current_path)}: {str(e)}")
    
    def _update_config_recursive(
        self, 
        target: Dict[str, Any], 
        source: Dict[str, Any], 
        source_type: ConfigSource,
        path: List[str] = []
    ) -> None:
        """
        Recursively update configuration from source.
        
        Args:
            target: Target configuration dict
            source: Source configuration dict
            source_type: Source type for tracking
            path: Current path in configuration
        """
        for key, value in source.items():
            current_path = path + [key]
            
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recurse into nested dictionaries
                self._update_config_recursive(target[key], value, source_type, current_path)
            else:
                # Set the value
                target[key] = value
                
                # Update the source
                if path:
                    section = path[0]
                    section_path = '.'.join(current_path[1:])
                    
                    section_sources = self.config_sources.setdefault(section, {})
                    section_sources[section_path] = source_type
    
    def _validate_config(self) -> None:
        """Validate the configuration."""
        # Validate LLM provider
        llm_config = self.config.get('llm', {})
        provider = llm_config.get('provider', '').lower()
        
        valid_providers = [p.value for p in LLMProvider]
        if provider not in valid_providers:
            logger.warning(
                f"Invalid LLM provider '{provider}', must be one of: {', '.join(valid_providers)}. "
                f"Falling back to default: {self.DEFAULT_CONFIG['llm']['provider']}"
            )
            llm_config['provider'] = self.DEFAULT_CONFIG['llm']['provider']
            
        # Validate oversight level
        oversight_config = self.config.get('oversight', {})
        level = oversight_config.get('level', '').lower()
        
        valid_levels = [l.value for l in OversightLevel]
        if level not in valid_levels:
            logger.warning(
                f"Invalid oversight level '{level}', must be one of: {', '.join(valid_levels)}. "
                f"Falling back to default: {self.DEFAULT_CONFIG['oversight']['level']}"
            )
            oversight_config['level'] = self.DEFAULT_CONFIG['oversight']['level']
            
        # Validate numeric values in circuit breaker config
        cb_config = self.config.get('circuit_breaker', {})
        for key, default_value in self.DEFAULT_CONFIG['circuit_breaker'].items():
            if not isinstance(cb_config.get(key), (int, float)) or cb_config.get(key) <= 0:
                logger.warning(
                    f"Invalid circuit breaker parameter '{key}' = {cb_config.get(key)}, "
                    f"must be a positive number. Falling back to default: {default_value}"
                )
                cb_config[key] = default_value
    
    def set_config_value(self, path: str, value: Any) -> bool:
        """
        Set a configuration value at runtime.
        
        Args:
            path: Dot-separated path to the configuration setting
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split the path
            parts = path.split('.')
            
            if len(parts) < 2:
                logger.warning(f"Invalid configuration path: {path}")
                return False
                
            section = parts[0]
            key_path = parts[1:]
            
            # Ensure section exists
            if section not in self.config:
                logger.warning(f"Configuration section not found: {section}")
                return False
                
            # Navigate to the target node
            current = self.config[section]
            current_path = [section]
            
            for i, key in enumerate(key_path[:-1]):
                current_path.append(key)
                
                # If key doesn't exist in current level, add it
                if key not in current:
                    logger.info(f"Creating missing configuration path: {'.'.join(current_path)}")
                    current[key] = {}
                    
                # If value is not a dict, we can't go deeper
                if not isinstance(current[key], dict):
                    logger.warning(
                        f"Cannot set {path} because {'.'.join(current_path)} is not a dictionary"
                    )
                    return False
                    
                current = current[key]
                
            # Set the final value
            final_key = key_path[-1]
            current_path.append(final_key)
            
            # Update the configuration
            current[final_key] = value
            
            # Update the source
            path_str = '.'.join(current_path)
            section_sources = self.config_sources.setdefault(section, {})
            section_sources[path_str[len(section)+1:]] = ConfigSource.RUNTIME
            
            logger.info(f"Set configuration {path_str} = {value} at runtime")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration {path}: {str(e)}")
            return False
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            path: Dot-separated path to the configuration setting
            default: Default value to return if setting not found
            
        Returns:
            The configuration value or default if not found
        """
        try:
            # Split the path
            parts = path.split('.')
            
            if len(parts) < 1:
                return default
                
            # Start with the full config
            current = self.config
            
            # Navigate to the target node
            for part in parts:
                if part not in current:
                    return default
                current = current[part]
                
            return current
            
        except Exception:
            return default
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get the LLM configuration section.
        
        Returns:
            LLM configuration dictionary
        """
        return self.config.get('llm', {}).copy()
    
    def get_oversight_config(self) -> Dict[str, Any]:
        """
        Get the oversight configuration section.
        
        Returns:
            Oversight configuration dictionary
        """
        return self.config.get('oversight', {}).copy()
    
    def get_circuit_breaker_config(self) -> Dict[str, Any]:
        """
        Get the circuit breaker configuration section.
        
        Returns:
            Circuit breaker configuration dictionary
        """
        return self.config.get('circuit_breaker', {}).copy()
    
    def get_oversight_level(self) -> OversightLevel:
        """
        Get the configured oversight level.
        
        Returns:
            OversightLevel enum value
        """
        level_str = self.config.get('oversight', {}).get('level', 'advise').lower()
        
        try:
            return OversightLevel(level_str)
        except ValueError:
            logger.warning(f"Invalid oversight level '{level_str}', using ADVISE")
            return OversightLevel.ADVISE
    
    def get_llm_provider(self) -> LLMProvider:
        """
        Get the configured LLM provider.
        
        Returns:
            LLMProvider enum value
        """
        provider_str = self.config.get('llm', {}).get('provider', 'openai').lower()
        
        try:
            return LLMProvider(provider_str)
        except ValueError:
            logger.warning(f"Invalid LLM provider '{provider_str}', using OPENAI")
            return LLMProvider.OPENAI
    
    def save_config(self, output_path: Optional[str] = None) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            output_path: Path to save configuration (uses config_path if None)
            
        Returns:
            True if successful, False otherwise
        """
        path = output_path or self.config_path
        
        if not path:
            logger.warning("No output path specified for saving configuration")
            return False
            
        try:
            output_file = Path(path)
            
            # Ensure directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on file extension
            if output_file.suffix.lower() in ['.yaml', '.yml']:
                with open(output_file, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif output_file.suffix.lower() == '.json':
                with open(output_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                logger.warning(f"Unsupported configuration file format: {output_file.suffix}")
                return False
                
            logger.info(f"Saved configuration to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get_config_sources(self) -> Dict[str, Dict[str, str]]:
        """
        Get the sources of configuration values.
        
        Returns:
            Dictionary mapping configuration paths to their sources
        """
        # Convert enum values to strings
        result = {}
        for section, sources in self.config_sources.items():
            result[section] = {}
            for key, source in sources.items():
                result[section][key] = source.value
                
        return result
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return json.dumps(self.config, indent=2)


# Create default instance
default_config = OversightConfig()


class OversightServiceConfig:
    """
    Configuration for the LLM Oversight Service.
    
    This class manages the configuration settings specific to the oversight service
    including host, port, data storage, and metrics collection.
    """
    
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 data_dir: str = "./data/oversight",
                 metrics_enabled: bool = True,
                 metrics_multiproc_dir: Optional[str] = None,
                 debug: bool = False):
        """
        Initialize the oversight service configuration.
        
        Args:
            host: Host to bind the service to
            port: Port to bind the service to
            data_dir: Directory for storing oversight data
            metrics_enabled: Whether to enable Prometheus metrics
            metrics_multiproc_dir: Directory for multiprocess metrics collection
            debug: Whether to enable debug mode
        """
        self.host = host
        self.port = port
        self.data_dir = data_dir
        self.metrics_enabled = metrics_enabled
        self.metrics_multiproc_dir = metrics_multiproc_dir
        self.debug = debug
        
        # Create data directory if it doesn't exist
        if self.data_dir:
            os.makedirs(self.data_dir, exist_ok=True)
            
        # Initialize oversight config
        self.oversight_config = default_config
        
    @classmethod
    def from_env(cls) -> 'OversightServiceConfig':
        """
        Create a configuration from environment variables.
        
        Returns:
            OversightServiceConfig instance
        """
        return cls(
            host=os.environ.get("OVERSIGHT_HOST", "0.0.0.0"),
            port=int(os.environ.get("OVERSIGHT_PORT", "8080")),
            data_dir=os.environ.get("OVERSIGHT_DATA_DIR", "./data/oversight"),
            metrics_enabled=os.environ.get("OVERSIGHT_METRICS_ENABLED", "true").lower() == "true",
            metrics_multiproc_dir=os.environ.get("PROMETHEUS_MULTIPROC_DIR"),
            debug=os.environ.get("OVERSIGHT_DEBUG", "false").lower() == "true"
        )


def validate_config(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Validate the oversight configuration.
    
    Args:
        config: Configuration dictionary to validate (uses default_config if None)
        
    Returns:
        True if valid, False otherwise
    """
    cfg = config or default_config.config
    
    # Check required sections
    for section in ["llm", "oversight", "circuit_breaker"]:
        if section not in cfg:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate LLM provider
    provider = cfg.get("llm", {}).get("provider")
    if provider and provider.lower() not in [p.value for p in LLMProvider]:
        logger.error(f"Invalid LLM provider: {provider}")
        return False
    
    # Validate oversight level
    level = cfg.get("oversight", {}).get("level")
    if level and level.lower() not in [l.value for l in OversightLevel]:
        logger.error(f"Invalid oversight level: {level}")
        return False
    
    return True


def get_config(config_path: Optional[str] = None) -> OversightConfig:
    """
    Get the oversight configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        OversightConfig instance
    """
    global default_config
    
    if config_path and config_path != default_config.config_path:
        default_config = OversightConfig(config_path)
        
    return default_config


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = OversightConfig()
    
    # Display configuration
    print("LLM Oversight Configuration:")
    print(config)
    
    # Get specific values
    provider = config.get_llm_provider()
    print(f"\nConfigured LLM Provider: {provider}")
    
    level = config.get_oversight_level()
    print(f"Configured Oversight Level: {level}")
    
    # Set a configuration value
    config.set_config_value("llm.temperature", 0.4)
    
    # Save configuration
    config.save_config("oversight_config.yaml")
