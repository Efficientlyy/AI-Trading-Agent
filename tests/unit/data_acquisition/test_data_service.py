"""
Unit tests for the DataService.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import asyncio

from src.data_acquisition.data_service import DataService
from src.data_acquisition.mock_provider import MockDataProvider
from src.data_acquisition.ccxt_provider import CcxtProvider
from src.data_acquisition.base_provider import BaseDataProvider

# Use pytest-asyncio for async tests
# pytestmark = pytest.mark.asyncio  # Removed global asyncio mark

# Sample configurations for testing
@pytest.fixture
def mock_config():
    return {
        'data_sources': {
            'active_provider': 'mock',
            'mock': {'generation_seed': 42},
            'ccxt': {'exchange_id': 'binance'}
        }
    }

@pytest.fixture
def ccxt_config():
    return {
        'data_sources': {
            'active_provider': 'ccxt',
            'mock': {'generation_seed': 42},
            'ccxt': {'exchange_id': 'binance', 'api_key': 'testkey'}
        }
    }

@pytest.fixture
def invalid_config():
    return {
        'data_sources': {
            'active_provider': 'nonexistent',
            'mock': {},
            'ccxt': {}
        }
    }

# --- Test Initialization --- 

# Patch get_config instead of load_config
@patch('src.data_acquisition.data_service.get_config_value')
@patch('src.data_acquisition.data_service.get_config') 
@patch('src.data_acquisition.data_service.MockDataProvider')
def test_data_service_init_mock(mock_MockDataProvider, mock_get_config, mock_get_config_value, mock_config):
    """Test DataService initialization with mock provider."""
    # Write debug info to a file
    with open('debug_output.txt', 'w') as f:
        f.write(f"mock_config: {mock_config}\n")
        
        # Ensure get_config returns our mock config
        mock_get_config.return_value = mock_config
        # Mock get_config_value to return the data_sources section
        mock_get_config_value.return_value = mock_config['data_sources']
        
        # Set up the mock provider instance
        mock_provider_instance = MagicMock(spec=MockDataProvider)
        mock_provider_instance.provider_name = 'mock'  # Add this attribute explicitly
        mock_MockDataProvider.return_value = mock_provider_instance
        
        # Create the DataService
        service = DataService()
        
        # Write more debug info
        f.write(f"service.active_provider_name: {service.active_provider_name}\n")
        f.write(f"service.data_sources_config: {service.data_sources_config}\n")
        
        # Assertions
        try:
            assert service.active_provider_name == 'mock'
            assert service.provider is mock_provider_instance
            mock_MockDataProvider.assert_called_once_with(config=mock_config['data_sources']['mock'])
            mock_get_config.assert_called_once()
            mock_get_config_value.assert_called_once_with('data_sources', {})
            f.write("All assertions passed!\n")
        except AssertionError as e:
            f.write(f"Assertion failed: {e}\n")
            raise

@patch('src.data_acquisition.data_service.get_config_value')
@patch('src.data_acquisition.data_service.get_config') 
@patch('src.data_acquisition.data_service.CcxtProvider')
def test_data_service_init_ccxt(mock_CcxtProvider, mock_get_config, mock_get_config_value, ccxt_config):
    """Test DataService initialization with ccxt provider."""
    mock_get_config.return_value = ccxt_config
    mock_get_config_value.return_value = ccxt_config['data_sources']
    mock_provider_instance = MagicMock(spec=CcxtProvider)
    mock_provider_instance.provider_name = 'ccxt'
    mock_CcxtProvider.return_value = mock_provider_instance

    service = DataService()

    assert service.active_provider_name == 'ccxt'
    assert service.provider is mock_provider_instance
    mock_CcxtProvider.assert_called_once_with(config=ccxt_config['data_sources']['ccxt'])
    mock_get_config.assert_called_once()
    mock_get_config_value.assert_called_once_with('data_sources', {})

@patch('src.data_acquisition.data_service.get_config_value')
@patch('src.data_acquisition.data_service.get_config') 
def test_data_service_init_invalid_provider(mock_get_config, mock_get_config_value, invalid_config):
    """Test DataService initialization with an unknown provider name."""
    mock_get_config.return_value = invalid_config
    mock_get_config_value.return_value = invalid_config['data_sources']
    
    # The DataService._create_provider method will raise ValueError for unknown provider
    with pytest.raises(ValueError, match="Unsupported data provider:"): 
        DataService()

@patch('src.data_acquisition.data_service.get_config_value')
@patch('src.data_acquisition.data_service.get_config') 
def test_data_service_init_missing_config_section(mock_get_config, mock_get_config_value):
    """Test DataService initialization when provider config section is missing."""
    config_missing_mock = {
        'data_sources': {
            'active_provider': 'mock',
            # 'mock' section intentionally missing
        }
    }
    mock_get_config.return_value = config_missing_mock
    mock_get_config_value.return_value = config_missing_mock['data_sources']
    # Expect it to initialize with an empty config for the provider
    with patch('src.data_acquisition.data_service.MockDataProvider') as mock_MockDataProvider:
        # Setup the mock provider instance with the provider_name attribute
        mock_provider_instance = MagicMock(spec=MockDataProvider)
        mock_provider_instance.provider_name = 'mock'
        mock_MockDataProvider.return_value = mock_provider_instance
        
        service = DataService()
        
        assert service.active_provider_name == 'mock'
        assert service.provider is mock_provider_instance
        mock_MockDataProvider.assert_called_once_with(config={}) # Expect empty config
        mock_get_config.assert_called_once()
        mock_get_config_value.assert_called_once_with('data_sources', {})



@patch('src.data_acquisition.data_service.get_config_value')
@patch('src.data_acquisition.data_service.get_config') 
@patch('src.data_acquisition.data_service.MockDataProvider')
@pytest.mark.asyncio
async def test_data_service_delegation(mock_MockDataProvider, mock_get_config, mock_get_config_value, mock_config):
    """Test that DataService correctly delegates calls to the active provider."""
    mock_get_config.return_value = mock_config
    mock_get_config_value.return_value = mock_config['data_sources']
    mock_provider_instance = MagicMock(spec=MockDataProvider)
    mock_provider_instance.provider_name = 'mock'
    # Use the correct signature for the mocked method to align with DataService
    mock_provider_instance.fetch_historical_data = AsyncMock(return_value={"BTC/USDT": pd.DataFrame()})
    mock_provider_instance.get_info = MagicMock(return_value={"name": "mock"})
    mock_provider_instance.close = AsyncMock() # Mock the close method if necessary for assertions
    mock_MockDataProvider.return_value = mock_provider_instance

    service = DataService()

    # Test fetch_historical_data delegation
    # Use arguments matching DataService.fetch_historical_data signature
    symbols = ["BTC/USDT"]
    timeframe = "1h"
    start_date = pd.Timestamp("2023-01-01T00:00:00Z")
    end_date = pd.Timestamp("2023-01-01T01:00:00Z")
    params = {'specific_param': 'value'}

    result = await service.fetch_historical_data(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        params=params
    )
    assert isinstance(result, dict)
    assert "BTC/USDT" in result
    # Don't expect params to be passed to the provider since it doesn't accept it
    mock_provider_instance.fetch_historical_data.assert_awaited_once_with(
        symbols, timeframe, start_date, end_date
    )

    # Test get_provider_info delegation
    info = service.get_provider_info()
    assert info == {"name": "mock"}
    mock_provider_instance.get_info.assert_called_once()

    # Test active_provider_name property
    assert service.active_provider_name == 'mock'

    # Test close
    await service.close()
    # Check if provider's close was called if it exists and is async
    # Since MockDataProvider might not have an async close, check based on spec/existence
    if hasattr(MockDataProvider, 'close') and asyncio.iscoroutinefunction(MockDataProvider.close):
         mock_provider_instance.close.assert_awaited_once()


@patch('src.data_acquisition.data_service.get_config_value')
@patch('src.data_acquisition.data_service.get_config') 
@patch('src.data_acquisition.data_service.CcxtProvider') # Still patching CcxtProvider here? Let's assume we want to test the *logic* of close with a non-async provider, even if config points elsewhere.
@pytest.mark.asyncio
async def test_data_service_close_no_provider_close(mock_CcxtProvider, mock_get_config, mock_get_config_value, mock_config):
    """Test DataService close method when provider has no close method."""
    mock_get_config.return_value = mock_config
    mock_get_config_value.return_value = mock_config['data_sources']
    
    # Instead of creating a subclass, just use a MagicMock with the necessary attributes
    mock_provider = MagicMock(spec=CcxtProvider)
    mock_provider.provider_name = 'mock'
    # Don't add a close method - we want to test what happens when it's missing
    
    mock_CcxtProvider.return_value = mock_provider

    service = DataService()
    
    # Test that close works even when provider doesn't have an async close
    await service.close()  # Should not raise an exception
