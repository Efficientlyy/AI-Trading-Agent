"""
Enhanced Circuit Breaker implementation for AI Trading Agent.

This module provides an advanced circuit breaker pattern implementation
with tiered failure thresholds, exponential backoff, and autonomous recovery
capabilities for building self-healing components.
"""

import time
import logging
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Tuple
from enum import Enum

from ai_trading_agent.common.error_handling import (
    TradingAgentError, 
    ErrorCode, 
    ErrorCategory,
    ErrorSeverity
)

# Set up logger
logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')

class CircuitState(Enum):
    """Enum for circuit breaker states."""
    CLOSED = "closed"  # Normal operation - requests allowed
    OPEN = "open"      # Circuit is open - requests blocked
    HALF_OPEN = "half_open"  # Testing if service is recovered
    WARNING = "warning"  # Elevated failure rate but still allowing requests


class EnhancedCircuitBreaker:
    """
    Advanced circuit breaker implementation with tiered failure thresholds,
    exponential backoff, and self-recovery capabilities.
    """
    
    def __init__(
        self,
        name: str,
        warning_threshold: int = 3,
        failure_threshold: int = 5,
        recovery_time_base: float = 5.0,
        max_recovery_time: float = 300.0,
        exponential_factor: float = 2.0,
        half_open_max_tries: int = 3,
        reset_timeout: float = 60.0
    ):
        """
        Initialize the enhanced circuit breaker.
        
        Args:
            name: Name/identifier for this circuit breaker
            warning_threshold: Number of failures before entering WARNING state
            failure_threshold: Number of failures before fully opening circuit
            recovery_time_base: Base time (seconds) to wait before recovery attempts
            max_recovery_time: Maximum recovery time regardless of backoff calculation
            exponential_factor: Factor to multiply recovery time by after each failure
            half_open_max_tries: Maximum number of failed recovery attempts before re-opening
            reset_timeout: Time in seconds after which failure counts are reset if no new failures
        """
        self.name = name
        self.warning_threshold = warning_threshold
        self.failure_threshold = failure_threshold
        self.recovery_time_base = recovery_time_base
        self.max_recovery_time = max_recovery_time
        self.exponential_factor = exponential_factor
        self.half_open_max_tries = half_open_max_tries
        self.reset_timeout = reset_timeout
        
        # Internal state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.consecutive_failures = 0
        self.half_open_failures = 0
        self.last_failure_time = None
        self.last_success_time = time.time()
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_calls = 0
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Enhanced Circuit Breaker '{name}' initialized with warning threshold={warning_threshold}, "
                   f"failure threshold={failure_threshold}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics for this circuit breaker.
        
        Returns:
            Dictionary with current metrics
        """
        with self._lock:
            success_rate = 0 if self.total_calls == 0 else (self.successful_calls / self.total_calls) * 100
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "consecutive_failures": self.consecutive_failures,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "total_calls": self.total_calls,
                "success_rate": success_rate,
                "last_failure_time": self.last_failure_time,
                "last_success_time": self.last_success_time
            }
            
    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self.successful_calls += 1
            self.total_calls += 1
            self.consecutive_failures = 0
            self.last_success_time = time.time()
            
            # If we're in HALF_OPEN and succeed, close the circuit
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit breaker '{self.name}' - successful recovery, transitioning from HALF_OPEN to CLOSED")
                self.reset()
            
            # If we're in WARNING and have enough successes, return to CLOSED
            if self.state == CircuitState.WARNING and self.consecutive_failures == 0:
                # Reset after 2x warning threshold of successful calls
                if self.successful_calls - self.failed_calls > self.warning_threshold * 2:
                    logger.info(f"Circuit breaker '{self.name}' - recovered from WARNING state, transitioning to CLOSED")
                    self.reset()
    
    def record_failure(self) -> Tuple[bool, float]:
        """
        Record a failed operation and determine if circuit should open.
        
        Returns:
            Tuple of (allowed, wait_time):
                - allowed: Whether requests are still allowed
                - wait_time: How long to wait before attempting recovery (if not allowed)
        """
        with self._lock:
            self.failure_count += 1
            self.consecutive_failures += 1
            self.failed_calls += 1
            self.total_calls += 1
            self.last_failure_time = time.time()
            
            # Logic for different states
            if self.state == CircuitState.CLOSED:
                if self.consecutive_failures >= self.failure_threshold:
                    # Transition to OPEN
                    self.state = CircuitState.OPEN
                    recovery_time = self._calculate_recovery_time()
                    logger.warning(f"Circuit breaker '{self.name}' transitioning to OPEN after {self.consecutive_failures} "
                                  f"consecutive failures. Recovery attempt in {recovery_time:.2f}s")
                    return False, recovery_time
                elif self.consecutive_failures >= self.warning_threshold:
                    # Transition to WARNING
                    self.state = CircuitState.WARNING
                    logger.warning(f"Circuit breaker '{self.name}' transitioning to WARNING after {self.consecutive_failures} "
                                  f"consecutive failures")
                    return True, 0
                    
            elif self.state == CircuitState.WARNING:
                if self.consecutive_failures >= self.failure_threshold:
                    # Transition from WARNING to OPEN
                    self.state = CircuitState.OPEN
                    recovery_time = self._calculate_recovery_time()
                    logger.warning(f"Circuit breaker '{self.name}' transitioning from WARNING to OPEN "
                                  f"after {self.consecutive_failures} consecutive failures. "
                                  f"Recovery attempt in {recovery_time:.2f}s")
                    return False, recovery_time
                
            elif self.state == CircuitState.HALF_OPEN:
                # If we fail during HALF_OPEN, go back to OPEN with increased backoff
                self.half_open_failures += 1
                self.state = CircuitState.OPEN
                # Use more aggressive backoff for repeated recovery failures
                recovery_time = self._calculate_recovery_time(base_multiplier=self.half_open_failures)
                logger.warning(f"Circuit breaker '{self.name}' failed during recovery attempt, "
                              f"transitioning back to OPEN. Next recovery attempt in {recovery_time:.2f}s")
                return False, recovery_time
                
            return self.state != CircuitState.OPEN, 0
    
    def attempt_reset(self) -> bool:
        """
        Attempt to reset the circuit from OPEN to HALF_OPEN state.
        
        Returns:
            True if circuit is now in HALF_OPEN state, False otherwise
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                self.state = CircuitState.HALF_OPEN
                self.half_open_failures = 0
                logger.info(f"Circuit breaker '{self.name}' transitioning from OPEN to HALF_OPEN for recovery attempt")
                return True
            return False
    
    def reset(self) -> None:
        """Fully reset the circuit breaker to initial state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.consecutive_failures = 0
            self.half_open_failures = 0
            logger.info(f"Circuit breaker '{self.name}' reset to CLOSED state")
    
    def is_allowed(self) -> bool:
        """
        Check if a request is allowed based on current circuit state.
        
        Returns:
            True if the request is allowed, False otherwise
        """
        with self._lock:
            # Auto-reset logic - if it's been long enough since last failure
            if self.state != CircuitState.CLOSED and self.last_failure_time is not None:
                elapsed = time.time() - self.last_failure_time
                
                # Check for timeout-based reset
                if elapsed > self.reset_timeout and self.state != CircuitState.OPEN:
                    logger.info(f"Circuit breaker '{self.name}' auto-reset after {elapsed:.2f}s with no failures")
                    self.reset()
                
                # Check for recovery attempt
                if self.state == CircuitState.OPEN:
                    recovery_time = self._calculate_recovery_time()
                    if elapsed > recovery_time:
                        self.attempt_reset()
            
            return self.state != CircuitState.OPEN
    
    def _calculate_recovery_time(self, base_multiplier: int = 1) -> float:
        """
        Calculate recovery time using exponential backoff.
        
        Args:
            base_multiplier: Additional multiplier for the base time
            
        Returns:
            Time to wait before recovery attempt in seconds
        """
        # Start with base time
        time_to_wait = self.recovery_time_base * base_multiplier
        
        # Apply exponential factor based on consecutive failures beyond threshold
        excess_failures = max(0, self.consecutive_failures - self.failure_threshold)
        time_to_wait *= (self.exponential_factor ** excess_failures)
        
        # Cap at maximum recovery time
        return min(time_to_wait, self.max_recovery_time)


# Global registry for circuit breakers
_circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}


def get_circuit_breaker(name: str) -> Optional[EnhancedCircuitBreaker]:
    """
    Get a registered circuit breaker by name.
    
    Args:
        name: Name of the circuit breaker
        
    Returns:
        The circuit breaker instance or None if not found
    """
    return _circuit_breakers.get(name)


def register_circuit_breaker(circuit_breaker: EnhancedCircuitBreaker) -> None:
    """
    Register a circuit breaker in the global registry.
    
    Args:
        circuit_breaker: The circuit breaker to register
    """
    _circuit_breakers[circuit_breaker.name] = circuit_breaker
    logger.info(f"Registered circuit breaker '{circuit_breaker.name}'")


def enhanced_circuit_breaker(
    name: str,
    warning_threshold: int = 3,
    failure_threshold: int = 5,
    recovery_time_base: float = 5.0,
    max_recovery_time: float = 300.0,
    exponential_factor: float = 2.0,
    half_open_max_tries: int = 3,
    reset_timeout: float = 60.0,
    fallback_result: Any = None,
    on_open: Optional[Callable[[], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that applies the enhanced circuit breaker pattern to a function.
    
    Args:
        name: Name for this circuit breaker instance
        warning_threshold: Number of failures before entering WARNING state
        failure_threshold: Number of failures before fully opening circuit
        recovery_time_base: Base time (seconds) to wait before recovery attempts
        max_recovery_time: Maximum recovery time regardless of backoff calculation
        exponential_factor: Factor to multiply recovery time by after each failure
        half_open_max_tries: Maximum number of failed recovery attempts before re-opening
        reset_timeout: Time in seconds after which failure counts are reset if no new failures 
        fallback_result: Result to return when circuit is open (None means raise exception)
        on_open: Callback function to execute when circuit opens
        
    Returns:
        The decorated function
    """
    # Create or get circuit breaker with given name
    circuit_breaker = get_circuit_breaker(name)
    if circuit_breaker is None:
        circuit_breaker = EnhancedCircuitBreaker(
            name=name,
            warning_threshold=warning_threshold,
            failure_threshold=failure_threshold,
            recovery_time_base=recovery_time_base,
            max_recovery_time=max_recovery_time,
            exponential_factor=exponential_factor,
            half_open_max_tries=half_open_max_tries,
            reset_timeout=reset_timeout
        )
        register_circuit_breaker(circuit_breaker)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Check if circuit is open
            if not circuit_breaker.is_allowed():
                if fallback_result is not None:
                    logger.warning(f"Circuit '{name}' is OPEN - using fallback result for {func.__name__}")
                    return fallback_result
                else:
                    error_message = f"Circuit '{name}' is OPEN - request blocked for {func.__name__}"
                    logger.error(error_message)
                    raise TradingAgentError(
                        message=error_message,
                        error_code=ErrorCode.SYSTEM_DEPENDENCY_ERROR,
                        error_category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.ERROR,
                        details={
                            "circuit_name": name,
                            "function": func.__name__,
                            "metrics": circuit_breaker.get_metrics()
                        }
                    )
            
            try:
                # Circuit is closed or half-open, attempt the call
                result = func(*args, **kwargs)
                # Record success
                circuit_breaker.record_success()
                return result
            except Exception as e:
                # Record failure
                allowed, wait_time = circuit_breaker.record_failure()
                
                # If transitioning to open state, call the callback
                if not allowed and on_open:
                    try:
                        on_open()
                    except Exception as callback_error:
                        logger.error(f"Error in circuit breaker '{name}' on_open callback: {callback_error}")
                
                # Re-raise the exception
                if isinstance(e, TradingAgentError):
                    # Enhance with circuit breaker info
                    if e.details is None:
                        e.details = {}
                    e.details.update({
                        "circuit_breaker": name,
                        "circuit_state": circuit_breaker.state.value,
                        "failure_count": circuit_breaker.failure_count,
                        "consecutive_failures": circuit_breaker.consecutive_failures
                    })
                    raise e
                else:
                    # Wrap other exceptions
                    raise TradingAgentError.from_exception(
                        e,
                        error_code=ErrorCode.SYSTEM_DEPENDENCY_ERROR, 
                        error_category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.ERROR,
                        details={
                            "circuit_breaker": name,
                            "circuit_state": circuit_breaker.state.value,
                            "function": func.__name__,
                            "failure_count": circuit_breaker.failure_count,
                            "consecutive_failures": circuit_breaker.consecutive_failures,
                            "wait_time": wait_time
                        }
                    )
        
        return wrapper
    
    return decorator


# Central error registry with metrics tracking
class ErrorRegistry:
    """
    Centralized registry for tracking errors and their patterns across the system.
    This enables sophisticated error pattern detection and analysis for autonomous recovery.
    """
    
    def __init__(self):
        """Initialize the error registry."""
        self._errors: List[Dict[str, Any]] = []
        self._error_counts: Dict[str, Dict[str, int]] = {}  # category -> code -> count
        self._lock = threading.RLock()
        
    def register_error(self, error: TradingAgentError) -> None:
        """
        Register an error with the registry.
        
        Args:
            error: The error to register
        """
        with self._lock:
            # Add to list of errors
            error_dict = error.to_dict()
            self._errors.append(error_dict)
            
            # Update counts
            category = error_dict["error_category"]["value"]
            code = error_dict["error_code"]["value"]
            
            if category not in self._error_counts:
                self._error_counts[category] = {}
                
            if code not in self._error_counts[category]:
                self._error_counts[category][code] = 0
                
            self._error_counts[category][code] += 1
            
            # Log high-frequency errors
            count = self._error_counts[category][code]
            if count > 0 and count % 10 == 0:  # Log every 10 occurrences
                logger.warning(f"High frequency error detected: {category}.{code} has occurred {count} times")
    
    def get_error_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Get current error counts by category and code.
        
        Returns:
            Dictionary mapping error categories to dictionaries of error codes and counts
        """
        with self._lock:
            return self._error_counts.copy()
    
    def get_recent_errors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the most recent errors.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent errors
        """
        with self._lock:
            return self._errors[-limit:]
    
    def get_error_frequency(self, category: Optional[str] = None, 
                          code: Optional[str] = None, 
                          window_seconds: float = 300) -> int:
        """
        Get the frequency of errors in a specific time window.
        
        Args:
            category: Optional category to filter by
            code: Optional code to filter by
            window_seconds: Time window in seconds
            
        Returns:
            Number of errors in the specified time window
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - window_seconds
            
            count = 0
            for error in self._errors:
                # Skip if timestamp is too old
                if error["timestamp"] < window_start:
                    continue
                    
                # Skip if category doesn't match (if specified)
                if category is not None and error["error_category"]["value"] != category:
                    continue
                    
                # Skip if code doesn't match (if specified)
                if code is not None and error["error_code"]["value"] != code:
                    continue
                    
                count += 1
                
            return count


# Global error registry
error_registry = ErrorRegistry()


def register_error(error: Union[TradingAgentError, Exception]) -> None:
    """
    Register an error with the global registry.
    
    Args:
        error: The error to register
    """
    if isinstance(error, TradingAgentError):
        error_registry.register_error(error)
    else:
        # Convert to TradingAgentError first
        trading_error = TradingAgentError.from_exception(
            error,
            error_code=ErrorCode.UNKNOWN_ERROR,
            error_category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR
        )
        error_registry.register_error(trading_error)


# Example usage of enhanced circuit breaker
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    @enhanced_circuit_breaker(
        name="example_api", 
        warning_threshold=2,
        failure_threshold=3,
        recovery_time_base=2.0,
        max_recovery_time=10.0
    )
    def example_api_call(should_fail: bool = False):
        """Example function protected by circuit breaker."""
        if should_fail:
            raise ValueError("API call failed")
        return "API call succeeded"
    
    # Demo the circuit breaker
    print("Testing Enhanced Circuit Breaker:")
    
    # Successful calls
    for i in range(3):
        try:
            result = example_api_call(should_fail=False)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1} failed: {e}")
    
    # Failing calls to trigger WARNING and then OPEN states
    for i in range(4):
        try:
            result = example_api_call(should_fail=True)
            print(f"Failing call {i+1}: {result}")
        except Exception as e:
            print(f"Failing call {i+1} error: {str(e)}")
    
    # This call should be blocked by the circuit breaker
    try:
        result = example_api_call(should_fail=False)
        print(f"After circuit opens: {result}")
    except Exception as e:
        print(f"After circuit opens error: {str(e)}")
    
    # Wait for circuit to go to HALF_OPEN
    print("Waiting for recovery attempt...")
    time.sleep(3)
    
    # This call should work as we're now in HALF_OPEN and not failing
    try:
        result = example_api_call(should_fail=False)
        print(f"Recovery attempt: {result}")
    except Exception as e:
        print(f"Recovery attempt error: {str(e)}")
    
    # Show circuit breaker metrics
    cb = get_circuit_breaker("example_api")
    if cb:
        print(f"Circuit Breaker Metrics: {cb.get_metrics()}")
