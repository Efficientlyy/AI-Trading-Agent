"""
Settings Manager

This module provides a class for managing dashboard settings, including
real data configuration. It handles loading, saving, and validating settings.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("settings_manager")

class SettingsManager:
    """
    Manages dashboard settings, including real data configuration.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the settings manager.
        
        Args:
            config_dir: Directory where configuration files are stored
        """
        self.config_dir = Path(config_dir)
        self.real_data_config_path = self.config_dir / "real_data_config.json"
        self.settings_path = self.config_dir / "dashboard_settings.json"
        
        # Default settings
        self.default_settings = {
            "general": {
                "theme": "system",
                "autoRefresh": True,
                "refreshInterval": 30
            },
            "dataSources": {
                "realDataEnabled": False,
                "fallbackStrategy": "cache_then_mock",
                "cacheDuration": 3600
            },
            "display": {
                "chartStyle": "modern",
                "defaultTimeRange": "1w",
                "decimalPlaces": 2
            },
            "notifications": {
                "desktopNotifications": True,
                "notificationLevel": "warning",
                "soundAlerts": False
            }
        }
        
        # Ensure config directory exists
        self._ensure_config_dir()
        
        # Load settings
        self.settings = self._load_settings()
        
    def _ensure_config_dir(self) -> None:
        """
        Ensure the configuration directory exists.
        """
        if not self.config_dir.exists():
            try:
                self.config_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created configuration directory: {self.config_dir}")
            except Exception as e:
                logger.error(f"Error creating configuration directory: {e}")
                
    def _load_settings(self) -> Dict[str, Any]:
        """
        Load settings from the settings file.
        
        Returns:
            Dict containing the settings
        """
        # Check if settings file exists
        if not self.settings_path.exists():
            # Create default settings file
            self._save_settings(self.default_settings)
            return self.default_settings.copy()
        
        try:
            with open(self.settings_path, 'r') as f:
                settings = json.load(f)
                
            # Validate and update settings with any missing defaults
            settings = self._validate_settings(settings)
            return settings
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return self.default_settings.copy()
            
    def _save_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Save settings to the settings file.
        
        Args:
            settings: Settings to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(settings, f, indent=4)
            logger.info(f"Settings saved to {self.settings_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
            
    def _validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate settings and fill in any missing values with defaults.
        
        Args:
            settings: Settings to validate
            
        Returns:
            Validated settings
        """
        validated = self.default_settings.copy()
        
        # Update with provided settings
        for section, section_settings in settings.items():
            if section in validated:
                if isinstance(section_settings, dict):
                    for key, value in section_settings.items():
                        if key in validated[section]:
                            # Type checking
                            default_type = type(validated[section][key])
                            if isinstance(value, default_type):
                                validated[section][key] = value
                            else:
                                logger.warning(f"Invalid type for setting {section}.{key}: expected {default_type.__name__}, got {type(value).__name__}")
                else:
                    logger.warning(f"Invalid section settings for {section}: expected dict, got {type(section_settings).__name__}")
        
        return validated
        
    def get_settings(self) -> Dict[str, Any]:
        """
        Get the current settings.
        
        Returns:
            Dict containing the settings
        """
        return self.settings.copy()
        
    def update_settings(self, new_settings: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Update settings with new values.
        
        Args:
            new_settings: New settings to apply
            
        Returns:
            Tuple of (success, reload_required)
        """
        # Check if real data setting is changing
        current_real_data = self.settings.get("dataSources", {}).get("realDataEnabled", False)
        new_real_data = new_settings.get("dataSources", {}).get("realDataEnabled", False)
        reload_required = current_real_data != new_real_data
        
        # Validate new settings
        validated_settings = self._validate_settings(new_settings)
        
        # Save settings
        success = self._save_settings(validated_settings)
        
        if success:
            # Update real data configuration if needed
            if reload_required:
                self._update_real_data_config(new_real_data)
                
            # Update current settings
            self.settings = validated_settings
            
        return success, reload_required
        
    def reset_settings(self) -> Dict[str, Any]:
        """
        Reset settings to defaults.
        
        Returns:
            Dict containing the default settings
        """
        # Save default settings
        self._save_settings(self.default_settings)
        
        # Update real data configuration
        self._update_real_data_config(self.default_settings.get("dataSources", {}).get("realDataEnabled", False))
        
        # Update current settings
        self.settings = self.default_settings.copy()
        
        return self.settings
        
    def _update_real_data_config(self, enabled: bool) -> bool:
        """
        Update the real data configuration file.
        
        Args:
            enabled: Whether real data should be enabled
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if real data config file exists
            if not self.real_data_config_path.exists():
                # Create default config
                default_config = {
                    "enabled": enabled,
                    "connections": {
                        "exchange_api": {
                            "enabled": True,
                            "retry_attempts": 3,
                            "timeout_seconds": 10,
                            "cache_duration_seconds": 60
                        },
                        "market_data": {
                            "enabled": True,
                            "retry_attempts": 3,
                            "timeout_seconds": 15,
                            "cache_duration_seconds": 30
                        },
                        "sentiment_api": {
                            "enabled": True,
                            "retry_attempts": 2,
                            "timeout_seconds": 20,
                            "cache_duration_seconds": 300
                        },
                        "news_api": {
                            "enabled": True,
                            "retry_attempts": 2,
                            "timeout_seconds": 20,
                            "cache_duration_seconds": 600
                        }
                    },
                    "fallback_strategy": {
                        "use_cached_data": True,
                        "cache_expiry_seconds": 3600,
                        "use_mock_data_on_failure": True
                    },
                    "error_tracking": {
                        "log_errors": True,
                        "max_consecutive_errors": 5,
                        "error_cooldown_seconds": 300
                    },
                    "data_validation": {
                        "validate_schema": True,
                        "validate_types": True,
                        "validate_required_fields": True
                    }
                }
                
                with open(self.real_data_config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                    
                logger.info(f"Created default real data config with enabled={enabled}")
                return True
            
            # Load existing config
            with open(self.real_data_config_path, 'r') as f:
                config = json.load(f)
                
            # Update enabled status
            config["enabled"] = enabled
            
            # Save updated config
            with open(self.real_data_config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            logger.info(f"Updated real data config with enabled={enabled}")
            return True
        except Exception as e:
            logger.error(f"Error updating real data config: {e}")
            return False
            
    def is_real_data_enabled(self) -> bool:
        """
        Check if real data is enabled.
        
        Returns:
            True if real data is enabled, False otherwise
        """
        return self.settings.get("dataSources", {}).get("realDataEnabled", False)
        
    def get_real_data_config(self) -> Dict[str, Any]:
        """
        Get the real data configuration.
        
        Returns:
            Dict containing the real data configuration
        """
        try:
            if not self.real_data_config_path.exists():
                return {"enabled": False}
                
            with open(self.real_data_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading real data config: {e}")
            return {"enabled": False}
            
    def update_real_data_config(self, config: Dict[str, Any]) -> bool:
        """
        Update the real data configuration.
        
        Args:
            config: New configuration to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure enabled status matches settings
            config["enabled"] = self.is_real_data_enabled()
            
            # Save updated config
            with open(self.real_data_config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            logger.info("Updated real data configuration")
            return True
        except Exception as e:
            logger.error(f"Error updating real data config: {e}")
            return False
            
    def reset_real_data_config(self) -> Dict[str, Any]:
        """
        Reset real data configuration to defaults.
        
        Returns:
            Dict containing the default real data configuration
        """
        try:
            # Create default config
            default_config = {
                "enabled": self.is_real_data_enabled(),
                "connections": {
                    "exchange_api": {
                        "enabled": True,
                        "retry_attempts": 3,
                        "timeout_seconds": 10,
                        "cache_duration_seconds": 60
                    },
                    "market_data": {
                        "enabled": True,
                        "retry_attempts": 3,
                        "timeout_seconds": 15,
                        "cache_duration_seconds": 30
                    },
                    "sentiment_api": {
                        "enabled": True,
                        "retry_attempts": 2,
                        "timeout_seconds": 20,
                        "cache_duration_seconds": 300
                    },
                    "news_api": {
                        "enabled": True,
                        "retry_attempts": 2,
                        "timeout_seconds": 20,
                        "cache_duration_seconds": 600
                    }
                },
                "fallback_strategy": {
                    "strategy": "cache_then_mock",
                    "use_cached_data": True,
                    "cache_expiry_seconds": 3600,
                    "use_mock_data_on_failure": True
                },
                "auto_recovery": True,
                "recovery_attempts": 3,
                "health_check_interval": 60,
                "advanced": {
                    "logging_level": "info",
                    "collect_performance_metrics": True,
                    "metrics_retention_days": 7,
                    "max_concurrent_requests": 5,
                    "default_timeout_seconds": 10
                }
            }
            
            # Save default config
            with open(self.real_data_config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
                
            logger.info("Reset real data configuration to defaults")
            return default_config
        except Exception as e:
            logger.error(f"Error resetting real data config: {e}")
            return {"enabled": self.is_real_data_enabled()}