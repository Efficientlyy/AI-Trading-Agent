"""Unit tests for the RetryableAPIClient and CircuitBreaker.

This module contains tests for the API client with retry logic and circuit breaker pattern.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import time

from src.common.api_client import RetryableAPIClient, CircuitBreaker, CircuitOpenError


@pytest.fixture
def api_client():
    """Create a RetryableAPIClient for testing."""
    logger = MagicMock()
    return RetryableAPIClient(max_retries=3, backoff_factor=0.1, logger=logger)


class TestRetryableAPIClient:
    """Tests for the RetryableAPIClient class."""
    
    @pytest.mark.asyncio
    async def test_successful_call(self, api_client):
        """Test a successful API call with no retries."""
        # Mock API function that succeeds
        async def mock_api_func():
            return "success"
            
        # Call with retry
        result = await api_client.call_with_retry(mock_api_func)
        
        # Verify result
        assert result == "success"
        
        # Verify logger was not called for warnings or errors
        api_client.logger.warning.assert_not_called()
        api_client.logger.error.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_retry_then_success(self, api_client):
        """Test an API call that fails once then succeeds."""
        # Counter for number of calls
        call_count = 0
        
        # Mock API function that fails once then succeeds
        async def mock_api_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary error")
            return "success after retry"
            
        # Call with retry
        result = await api_client.call_with_retry(mock_api_func)
        
        # Verify result
        assert result == "success after retry"
        assert call_count == 2
        
        # Verify logger was called for warning but not error
        api_client.logger.warning.assert_called_once()
        api_client.logger.error.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_all_retries_fail(self, api_client):
        """Test an API call that fails all retry attempts."""
        # Mock API function that always fails
        async def mock_api_func():
            raise Exception("Persistent error")
            
        # Call with retry and expect exception
        with pytest.raises(Exception) as excinfo:
            await api_client.call_with_retry(mock_api_func)
            
        # Verify exception
        assert "Persistent error" in str(excinfo.value)
        
        # Verify logger was called for both warnings and error
        assert api_client.logger.warning.call_count == 3  # Once for each retry
        api_client.logger.error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self, api_client):
        """Test circuit breaker with successful calls."""
        # Reset circuit breaker state
        CircuitBreaker._circuits = {}
        
        # Mock API function that succeeds
        async def mock_api_func():
            return "success"
            
        # Call with circuit breaker
        result = await api_client.call_with_circuit_breaker(
            mock_api_func, "test_circuit"
        )
        
        # Verify result
        assert result == "success"
        
        # Verify circuit state
        assert "test_circuit" in CircuitBreaker._circuits
        assert CircuitBreaker._circuits["test_circuit"]["state"] == "closed"
        assert CircuitBreaker._circuits["test_circuit"]["failures"] == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, api_client):
        """Test circuit breaker opening after failures."""
        # Reset circuit breaker state
        CircuitBreaker._circuits = {}
        CircuitBreaker._failure_threshold = 2  # Lower threshold for testing
        
        # Mock API function that always fails
        async def mock_api_func():
            raise Exception("Service unavailable")
            
        # Call with circuit breaker until it opens
        for _ in range(2):
            try:
                await api_client.call_with_circuit_breaker(
                    mock_api_func, "test_circuit"
                )
            except Exception:
                pass
                
        # Verify circuit is open
        assert CircuitBreaker.is_open("test_circuit")
        
        # Next call should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await api_client.call_with_circuit_breaker(
                mock_api_func, "test_circuit"
            )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open(self, api_client):
        """Test circuit breaker transitioning to half-open state."""
        # Reset circuit breaker state
        CircuitBreaker._circuits = {}
        CircuitBreaker._failure_threshold = 2
        CircuitBreaker._reset_timeout = 0.1  # Short timeout for testing
        
        # Set up an open circuit
        circuit = {
            "failures": 2,
            "state": "open",
            "last_failure": time.time() - 0.2  # Past the reset timeout
        }
        CircuitBreaker._circuits["test_circuit"] = circuit
        
        # Mock API function that succeeds
        async def mock_api_func():
            return "success"
            
        # Wait for reset timeout
        await asyncio.sleep(0.1)
        
        # Circuit should be half-open now
        assert CircuitBreaker.is_open("test_circuit") is False
        assert CircuitBreaker._circuits["test_circuit"]["state"] == "half-open"
        
        # Successful call should close the circuit
        result = await api_client.call_with_circuit_breaker(
            mock_api_func, "test_circuit"
        )
        
        # Verify result and circuit state
        assert result == "success"
        assert CircuitBreaker._circuits["test_circuit"]["state"] == "closed"
        assert CircuitBreaker._circuits["test_circuit"]["failures"] == 0
