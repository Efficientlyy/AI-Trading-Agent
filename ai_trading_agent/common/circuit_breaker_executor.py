"""
Circuit breaker executor for the AI Trading Agent.

This module provides a circuit breaker executor for handling failures
in the paper trading system, with automatic recovery mechanisms.
"""

import time
import logging
import asyncio
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from .error_handling import (
    TradingAgentError,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity,
    circuit_breaker,
    retry,
    log_error
)

# Set up logger
logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')
AsyncT = TypeVar('AsyncT')


class CircuitBreakerExecutor:
    """Circuit breaker executor for handling failures with automatic recovery."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        reset_timeout: float = 60.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        on_circuit_open: Optional[Callable[[], None]] = None,
        on_circuit_close: Optional[Callable[[], None]] = None,
        on_retry: Optional[Callable[[Exception, int], None]] = None
    ):
        """Initialize the circuit breaker executor.
        
        Args:
            name: Name of the executor for logging purposes.
            failure_threshold: Number of consecutive failures before opening the circuit.
            reset_timeout: Time in seconds before attempting to half-open the circuit.
            retry_attempts: Maximum number of retry attempts.
            retry_delay: Initial delay between retries in seconds.
            retry_backoff_factor: Multiplicative factor for exponential backoff.
            on_circuit_open: Optional callback function to call when the circuit opens.
            on_circuit_close: Optional callback function to call when the circuit closes.
            on_retry: Optional callback function to call on each retry.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.on_circuit_open = on_circuit_open
        self.on_circuit_close = on_circuit_close
        self.on_retry = on_retry
        
        # Circuit state
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = 0.0
        
        logger.info(f"Initialized CircuitBreakerExecutor '{name}' with failure_threshold={failure_threshold}, reset_timeout={reset_timeout}")
    
    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        retry_on_exceptions: Union[List[Exception], Exception] = Exception,
        **kwargs: Any
    ) -> T:
        """Execute a function with circuit breaker and retry logic.
        
        Args:
            func: The function to execute.
            *args: Positional arguments to pass to the function.
            retry_on_exceptions: Exception type(s) to catch and retry on.
            **kwargs: Keyword arguments to pass to the function.
        
        Returns:
            The result of the function.
        
        Raises:
            TradingAgentError: If the function fails after all retries or if the circuit is open.
        """
        current_time = time.time()
        
        # Check if the circuit is OPEN
        if self.state == "OPEN":
            if current_time - self.last_failure_time >= self.reset_timeout:
                # Transition to HALF_OPEN
                logger.info(f"Circuit '{self.name}' transitioning from OPEN to HALF_OPEN")
                self.state = "HALF_OPEN"
            else:
                # Circuit is still OPEN
                error_message = f"Circuit '{self.name}' is OPEN. Retry after {self.reset_timeout - (current_time - self.last_failure_time):.2f} seconds"
                logger.warning(error_message)
                raise TradingAgentError(
                    message=error_message,
                    error_code=ErrorCode.SYSTEM_RESOURCE_ERROR,
                    error_category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.WARNING,
                    details={"circuit_state": self.state, "failures": self.failures}
                )
        
        # Execute with retry logic
        last_exception = None
        current_delay = self.retry_delay
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                result = func(*args, **kwargs)
                
                # If we're in HALF_OPEN and the call succeeded, close the circuit
                if self.state == "HALF_OPEN":
                    logger.info(f"Circuit '{self.name}' transitioning from HALF_OPEN to CLOSED")
                    self.state = "CLOSED"
                    self.failures = 0
                    
                    if self.on_circuit_close:
                        self.on_circuit_close()
                
                # If we're in CLOSED, reset the failure count on success
                if self.state == "CLOSED":
                    self.failures = 0
                
                return result
                
            except retry_on_exceptions as e:
                last_exception = e
                
                # Increment failure count
                self.failures += 1
                self.last_failure_time = current_time
                
                # Check if we need to open the circuit
                if self.state == "CLOSED" and self.failures >= self.failure_threshold:
                    logger.warning(f"Circuit '{self.name}' transitioning from CLOSED to OPEN after {self.failures} consecutive failures")
                    self.state = "OPEN"
                    
                    if self.on_circuit_open:
                        self.on_circuit_open()
                
                # If we're in HALF_OPEN and failed, go back to OPEN
                if self.state == "HALF_OPEN":
                    logger.warning(f"Circuit '{self.name}' transitioning from HALF_OPEN back to OPEN after failure")
                    self.state = "OPEN"
                
                if attempt < self.retry_attempts:
                    if self.on_retry:
                        self.on_retry(e, attempt)
                    
                    logger.warning(
                        f"Attempt {attempt}/{self.retry_attempts} for '{self.name}' failed: {str(e)}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= self.retry_backoff_factor
                else:
                    logger.error(
                        f"All {self.retry_attempts} attempts for '{self.name}' failed. "
                        f"Last error: {str(e)}"
                    )
        
        if last_exception:
            if isinstance(last_exception, TradingAgentError):
                raise last_exception
            else:
                raise TradingAgentError.from_exception(
                    last_exception,
                    error_code=ErrorCode.UNKNOWN_ERROR,
                    error_category=ErrorCategory.UNKNOWN,
                    severity=ErrorSeverity.ERROR,
                    details={"executor": self.name, "max_attempts": self.retry_attempts}
                )
        
        # This should never happen, but to satisfy the type checker
        raise RuntimeError("Unexpected error in CircuitBreakerExecutor.execute")
    
    async def execute_async(
        self,
        func: Callable[..., AsyncT],
        *args: Any,
        retry_on_exceptions: Union[List[Exception], Exception] = Exception,
        **kwargs: Any
    ) -> AsyncT:
        """Execute an async function with circuit breaker and retry logic.
        
        Args:
            func: The async function to execute.
            *args: Positional arguments to pass to the function.
            retry_on_exceptions: Exception type(s) to catch and retry on.
            **kwargs: Keyword arguments to pass to the function.
        
        Returns:
            The result of the function.
        
        Raises:
            TradingAgentError: If the function fails after all retries or if the circuit is open.
        """
        current_time = time.time()
        
        # Check if the circuit is OPEN
        if self.state == "OPEN":
            if current_time - self.last_failure_time >= self.reset_timeout:
                # Transition to HALF_OPEN
                logger.info(f"Circuit '{self.name}' transitioning from OPEN to HALF_OPEN")
                self.state = "HALF_OPEN"
            else:
                # Circuit is still OPEN
                error_message = f"Circuit '{self.name}' is OPEN. Retry after {self.reset_timeout - (current_time - self.last_failure_time):.2f} seconds"
                logger.warning(error_message)
                raise TradingAgentError(
                    message=error_message,
                    error_code=ErrorCode.SYSTEM_RESOURCE_ERROR,
                    error_category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.WARNING,
                    details={"circuit_state": self.state, "failures": self.failures}
                )
        
        # Execute with retry logic
        last_exception = None
        current_delay = self.retry_delay
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                result = await func(*args, **kwargs)
                
                # If we're in HALF_OPEN and the call succeeded, close the circuit
                if self.state == "HALF_OPEN":
                    logger.info(f"Circuit '{self.name}' transitioning from HALF_OPEN to CLOSED")
                    self.state = "CLOSED"
                    self.failures = 0
                    
                    if self.on_circuit_close:
                        self.on_circuit_close()
                
                # If we're in CLOSED, reset the failure count on success
                if self.state == "CLOSED":
                    self.failures = 0
                
                return result
                
            except retry_on_exceptions as e:
                last_exception = e
                
                # Increment failure count
                self.failures += 1
                self.last_failure_time = current_time
                
                # Check if we need to open the circuit
                if self.state == "CLOSED" and self.failures >= self.failure_threshold:
                    logger.warning(f"Circuit '{self.name}' transitioning from CLOSED to OPEN after {self.failures} consecutive failures")
                    self.state = "OPEN"
                    
                    if self.on_circuit_open:
                        self.on_circuit_open()
                
                # If we're in HALF_OPEN and failed, go back to OPEN
                if self.state == "HALF_OPEN":
                    logger.warning(f"Circuit '{self.name}' transitioning from HALF_OPEN back to OPEN after failure")
                    self.state = "OPEN"
                
                if attempt < self.retry_attempts:
                    if self.on_retry:
                        self.on_retry(e, attempt)
                    
                    logger.warning(
                        f"Attempt {attempt}/{self.retry_attempts} for '{self.name}' failed: {str(e)}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= self.retry_backoff_factor
                else:
                    logger.error(
                        f"All {self.retry_attempts} attempts for '{self.name}' failed. "
                        f"Last error: {str(e)}"
                    )
        
        if last_exception:
            if isinstance(last_exception, TradingAgentError):
                raise last_exception
            else:
                raise TradingAgentError.from_exception(
                    last_exception,
                    error_code=ErrorCode.UNKNOWN_ERROR,
                    error_category=ErrorCategory.UNKNOWN,
                    severity=ErrorSeverity.ERROR,
                    details={"executor": self.name, "max_attempts": self.retry_attempts}
                )
        
        # This should never happen, but to satisfy the type checker
        raise RuntimeError("Unexpected error in CircuitBreakerExecutor.execute_async")
    
    def reset(self) -> None:
        """Reset the circuit breaker to its initial state."""
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = 0.0
        logger.info(f"Circuit '{self.name}' has been reset to CLOSED state")
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the circuit breaker.
        
        Returns:
            A dictionary containing the current state.
        """
        return {
            "name": self.name,
            "state": self.state,
            "failures": self.failures,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "reset_timeout": self.reset_timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "retry_backoff_factor": self.retry_backoff_factor
        }
