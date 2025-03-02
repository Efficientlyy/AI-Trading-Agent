"""Pytest configuration file for the AI Crypto Trading System."""

import asyncio
import os
import shutil
from pathlib import Path

import pytest

# Ensure we're using the test configuration
os.environ["ENVIRONMENT"] = "testing"


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config_dir():
    """Create a temporary configuration directory for tests."""
    # Create a temporary test config directory
    test_dir = Path("tests/test_config")
    test_dir.mkdir(exist_ok=True)
    
    # Yield the directory path
    yield test_dir
    
    # Clean up after tests if the directory exists
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture(scope="session")
def test_logs_dir():
    """Create a temporary logs directory for tests."""
    # Create a temporary test logs directory
    test_dir = Path("tests/test_logs")
    test_dir.mkdir(exist_ok=True)
    
    # Yield the directory path
    yield test_dir
    
    # Clean up after tests if the directory exists
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture(scope="session", autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # This helps ensure tests don't interfere with each other
    from src.common.config import Config
    from src.common.events import EventBus
    
    # Reset Config singleton
    Config._instance = None
    Config._initialized = False
    Config._config = {}
    Config._loaded_files = set()
    
    # Reset EventBus singleton
    EventBus._instance = None
    EventBus._initialized = False
    EventBus._subscribers = {}
    EventBus._registered_event_types = set()
    
    yield
    
    # Also reset after all tests
    Config._instance = None
    Config._initialized = False
    Config._config = {}
    Config._loaded_files = set()
    
    EventBus._instance = None
    EventBus._initialized = False
    EventBus._subscribers = {}
    EventBus._registered_event_types = set() 