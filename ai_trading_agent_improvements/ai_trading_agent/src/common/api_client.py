"""Retryable API client with circuit breaker pattern.

This module provides a client for making API calls with retry logic and
circuit breaker pattern to improve resilience when dealing with external services.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union

T = TypeVar('T')

class CircuitOpenError(Exception):
    """Exception raised when a circuit is open."""
    pass


class CircuitBreaker:
    """Static circuit breaker implementation."""
    
    _circuits: Dict[str, Dict[str, Any]] = {}
    _failure_threshold = 5
    _reset_timeout = 60  # seconds
    
    @classmethod
    def is_open(cls, circuit_key: str) -> bool:
        """Check if a circuit is open.
        
        Args:
            circuit_key: The circuit identifier
            
        Returns:
            True if the circuit is open, False otherwise
        """
        if circuit_key not in cls._circuits:
            cls._circuits[circuit_key] = {
                "failures": 0,
                "state": "closed",
                "last_failure": 0
            }
            return False
            
        circuit = cls._circuits[circuit_key]
        
        # Check if it's time to try resetting the circuit
        if (circuit["state"] == "open" and 
            (time.time() - circuit["last_failure"]) > cls._reset_timeout):
            circuit["state"] = "half-open"
            
        return circuit["state"] == "open"
    
    @classmethod
    def record_success(cls, circuit_key: str) -> None:
        """Record a successful API call.
        
        Args:
            circuit_key: The circuit identifier
        """
        if circuit_key not in cls._circuits:
            return
            
        circuit = cls._circuits[circuit_key]
        
        if circuit["state"] == "half-open":
            circuit["state"] = "closed"
            
        circuit["failures"] = 0
    
    @classmethod
    def record_failure(cls, circuit_key: str) -> None:
        """Record a failed API call.
        
        Args:
            circuit_key: The circuit identifier
        """
        if circuit_key not in cls._circuits:
            cls._circuits[circuit_key] = {
                "failures": 0,
                "state": "closed",
                "last_failure": 0
            }
            
        circuit = cls._circuits[circuit_key]
        circuit["failures"] += 1
        circuit["last_failure"] = time.time()
        
        if circuit["failures"] >= cls._failure_threshold:
            circuit["state"] = "open"
    
    @classmethod
    def configure(cls, failure_threshold: int = 5, reset_timeout: int = 60) -> None:
        """Configure the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            reset_timeout: Seconds to wait before trying to reset the circuit
        """
        cls._failure_threshold = failure_threshold
        cls._reset_timeout = reset_timeout


class RetryableAPIClient:
    """Client for making API calls with retry logic."""
    
    def __init__(self, 
                 max_retries: int = 3, 
                 backoff_factor: float = 1.5,
                 logger: Optional[logging.Logger] = None):
        """Initialize the retryable API client.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Multiplier for exponential backoff
            logger: Logger instance for logging errors
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logger or logging.getLogger(__name__)
        
    async def call_with_retry(self, 
                             api_func: Callable[..., T], 
                             *args: Any, 
                             **kwargs: Any) -> T:
        """Call an API function with retry logic.
        
        Args:
            api_func: Async function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the API call
            
        Raises:
            Exception: If all retry attempts fail
        """
        retries = 0
        last_exception = None
        
        while retries < self.max_retries:
            try:
                return await api_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                wait_time = self.backoff_factor ** retries
                self.logger.warning(
                    f"API call failed, retrying in {wait_time:.2f}s", 
                    extra={"error": str(e), "retry_count": retries+1}
                )
                await asyncio.sleep(wait_time)
                retries += 1
                
        self.logger.error(
            f"API call failed after {self.max_retries} retries", 
            extra={"error": str(last_exception)}
        )
        raise last_exception
        
    async def call_with_circuit_breaker(self,
                                       api_func: Callable[..., T],
                                       circuit_key: str,
                                       *args: Any,
                                       **kwargs: Any) -> T:
        """Call an API function with circuit breaker pattern.
        
        Args:
            api_func: Async function to call
            circuit_key: Unique identifier for this circuit
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the API call
            
        Raises:
            CircuitOpenError: If the circuit is open
            Exception: If the API call fails
        """
        # Check if circuit is open
        if CircuitBreaker.is_open(circuit_key):
            self.logger.warning(
                f"Circuit {circuit_key} is open, skipping API call"
            )
            raise CircuitOpenError(f"Circuit {circuit_key} is open")
            
        try:
            result = await self.call_with_retry(api_func, *args, **kwargs)
            CircuitBreaker.record_success(circuit_key)
            return result
        except Exception as e:
            CircuitBreaker.record_failure(circuit_key)
            raise e
