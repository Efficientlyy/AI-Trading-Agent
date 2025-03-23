"""Pytest configuration file for the AI Crypto Trading System."""

import asyncio
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

# Ensure we're using the test configuration
os.environ["ENVIRONMENT"] = "testing"


@pytest.fixture(scope="session")
def event_loop_policy():
    """Create an event loop policy for tests."""
    return asyncio.DefaultEventLoopPolicy()


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


@pytest.fixture(scope="session")
def mock_dependencies():
    """Mock dependencies for sentiment strategy tests."""
    # Store original modules
    original_modules = {}
    for mod in ['src.common.logging', 'src.common.events', 'src.common.config']:
        if mod in sys.modules:
            original_modules[mod] = sys.modules[mod]
        sys.modules[mod] = MagicMock()
    
    yield
    
    # Restore original modules
    for mod, orig in original_modules.items():
        sys.modules[mod] = orig


@pytest.fixture(scope="session", autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Check if we need to mock dependencies first
    if 'src.common.config' not in sys.modules:
        from unittest.mock import MagicMock
        sys.modules['src.common.config'] = MagicMock()
        sys.modules['src.common.events'] = MagicMock()
    
    # Let the test run
    yield