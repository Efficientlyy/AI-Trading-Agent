"""Tests for the configuration management system."""

import os
from pathlib import Path

import pytest
import yaml

from src.common.config import Config, config


def test_singleton_pattern():
    """Test that Config implements the singleton pattern."""
    config1 = Config()
    config2 = Config()
    
    # Both instances should be the same object
    assert config1 is config2
    
    # The global config instance should be the same as well
    assert config is config1


def test_load_config_file(tmp_path):
    """Test loading configuration from a file."""
    # Create a test config file
    test_config = {
        "test": {
            "value1": "test value",
            "value2": 123,
            "nested": {
                "value3": True,
            }
        }
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
    
    # Create a fresh config instance (reset the singleton)
    Config._instance = None
    Config._initialized = False
    Config._config = {}
    Config._loaded_files = set()
    test_config_instance = Config()
    
    # Load the config file
    test_config_instance.load_config_file(config_path)
    
    # Check that values were loaded correctly
    assert test_config_instance.get("test.value1") == "test value"
    assert test_config_instance.get("test.value2") == 123
    assert test_config_instance.get("test.nested.value3") is True
    
    # Check default values
    assert test_config_instance.get("non_existent", "default") == "default"


def test_override_from_env(monkeypatch):
    """Test overriding configuration from environment variables."""
    # Create a fresh config instance (reset the singleton)
    Config._instance = None
    Config._initialized = False
    Config._config = {}
    Config._loaded_files = set()
    test_config_instance = Config()
    
    # Set environment variables
    monkeypatch.setenv("TEST_VALUE1", "env value")
    monkeypatch.setenv("TEST_NESTED_VALUE3", "false")
    monkeypatch.setenv("TEST_NEW_VALUE", "brand new")
    
    # Create initial config
    initial_config = {
        "test": {
            "value1": "test value",
            "value2": 123,
            "nested": {
                "value3": True,
            }
        }
    }
    
    # Override with environment
    from src.common.config import ConfigLoader
    result = ConfigLoader.override_from_env(initial_config)
    
    # Check that values were overridden correctly
    assert result["test"]["value1"] == "env value"
    assert result["test"]["value2"] == 123  # Unchanged
    assert result["test"]["nested"]["value3"] is False  # Properly converted to boolean False
    assert "new" in result["test"]  # New section added
    assert "value" in result["test"]["new"]  # New value key added
    assert result["test"]["new"]["value"] == "brand new"  # New value has correct content


def test_merge_dicts():
    """Test merging dictionaries."""
    # Create a fresh config instance (reset the singleton)
    Config._instance = None
    Config._initialized = False
    Config._config = {}
    Config._loaded_files = set()
    test_config_instance = Config()
    
    dict1 = {
        "a": 1,
        "b": {
            "c": 2,
            "d": 3
        }
    }
    
    dict2 = {
        "b": {
            "d": 4,
            "e": 5
        },
        "f": 6
    }
    
    # Call _merge_dicts as an instance method, not a static method
    result = test_config_instance._merge_dicts(dict1, dict2)
    
    # Check that dictionaries were merged correctly
    assert result["a"] == 1
    assert result["b"]["c"] == 2
    assert result["b"]["d"] == 4  # Overridden by dict2
    assert result["b"]["e"] == 5  # Added from dict2
    assert result["f"] == 6  # Added from dict2