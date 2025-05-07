"""
Error handling module for the AI Trading Agent.

This module provides a comprehensive error classification system,
detailed error messages with troubleshooting guidance, and
automatic recovery mechanisms for common failures.
"""

import enum
import time
import logging
import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

# Set up logger
logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')


class ErrorSeverity(enum.Enum):
    """Enum for error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(enum.Enum):
    """Enum for error categories."""
    DATA_PROVIDER = "data_provider"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    PORTFOLIO = "portfolio"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    DATABASE = "database"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorCode(enum.Enum):
    """Enum for specific error codes."""
    # Data provider errors (1000-1999)
    DATA_PROVIDER_CONNECTION_ERROR = 1000
    DATA_PROVIDER_AUTHENTICATION_ERROR = 1001
    DATA_PROVIDER_RATE_LIMIT_ERROR = 1002
    DATA_PROVIDER_TIMEOUT_ERROR = 1003
    DATA_PROVIDER_DATA_ERROR = 1004
    DATA_PROVIDER_SYMBOL_ERROR = 1005
    
    # Strategy errors (2000-2999)
    STRATEGY_CALCULATION_ERROR = 2000
    STRATEGY_PARAMETER_ERROR = 2001
    STRATEGY_SIGNAL_ERROR = 2002
    STRATEGY_VALIDATION_ERROR = 2003
    
    # Execution errors (3000-3999)
    EXECUTION_CONNECTION_ERROR = 3000
    EXECUTION_AUTHENTICATION_ERROR = 3001
    EXECUTION_ORDER_ERROR = 3002
    EXECUTION_RATE_LIMIT_ERROR = 3003
    EXECUTION_TIMEOUT_ERROR = 3004
    EXECUTION_INSUFFICIENT_FUNDS_ERROR = 3005
    
    # Portfolio errors (4000-4999)
    PORTFOLIO_CALCULATION_ERROR = 4000
    PORTFOLIO_POSITION_ERROR = 4001
    PORTFOLIO_VALUATION_ERROR = 4002
    
    # Configuration errors (5000-5999)
    CONFIGURATION_FILE_ERROR = 5000
    CONFIGURATION_PARAMETER_ERROR = 5001
    CONFIGURATION_VALIDATION_ERROR = 5002
    
    # Authentication errors (6000-6999)
    AUTHENTICATION_CREDENTIALS_ERROR = 6000
    AUTHENTICATION_TOKEN_ERROR = 6001
    AUTHENTICATION_PERMISSION_ERROR = 6002
    
    # Network errors (7000-7999)
    NETWORK_CONNECTION_ERROR = 7000
    NETWORK_TIMEOUT_ERROR = 7001
    NETWORK_DNS_ERROR = 7002
    
    # Database errors (8000-8999)
    DATABASE_CONNECTION_ERROR = 8000
    DATABASE_QUERY_ERROR = 8001
    DATABASE_TRANSACTION_ERROR = 8002
    
    # System errors (9000-9999)
    SYSTEM_RESOURCE_ERROR = 9000
    SYSTEM_DEPENDENCY_ERROR = 9001
    SYSTEM_ENVIRONMENT_ERROR = 9002
    
    # Unknown errors (10000+)
    UNKNOWN_ERROR = 10000


class TradingAgentError(Exception):
    """Base exception class for all AI Trading Agent errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        troubleshooting: Optional[List[str]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: A human-readable error message.
            error_code: The specific error code.
            error_category: The category of the error.
            severity: The severity level of the error.
            details: Additional details about the error.
            cause: The original exception that caused this error.
            troubleshooting: List of troubleshooting steps.
        """
        self.message = message
        self.error_code = error_code
        self.error_category = error_category
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.troubleshooting = troubleshooting or []
        self.timestamp = time.time()
        
        # Add stack trace if there's a cause
        if cause:
            self.details["cause"] = str(cause)
            self.details["traceback"] = traceback.format_exc()
        
        # Construct the full error message
        full_message = f"[{error_code.name}] {message}"
        if troubleshooting:
            full_message += "\nTroubleshooting steps:\n" + "\n".join(f"- {step}" for step in troubleshooting)
        
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary representation."""
        return {
            "message": self.message,
            "error_code": {
                "code": self.error_code.value,
                "name": self.error_code.name
            },
            "error_category": self.error_category.value,
            "severity": self.severity.value,
            "details": self.details,
            "troubleshooting": self.troubleshooting,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        troubleshooting: Optional[List[str]] = None
    ) -> 'TradingAgentError':
        """Create a TradingAgentError from another exception."""
        return cls(
            message=str(exception),
            error_code=error_code,
            error_category=error_category,
            severity=severity,
            details=details,
            cause=exception,
            troubleshooting=troubleshooting
        )


# Specific error classes for different categories
class DataProviderError(TradingAgentError):
    """Exception raised for errors related to data providers."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATA_PROVIDER_CONNECTION_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        troubleshooting: Optional[List[str]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            error_category=ErrorCategory.DATA_PROVIDER,
            severity=severity,
            details=details,
            cause=cause,
            troubleshooting=troubleshooting or [
                "Check your internet connection",
                "Verify API credentials for the data provider",
                "Check if the data provider service is operational",
                "Ensure you're not exceeding API rate limits"
            ]
        )


class StrategyError(TradingAgentError):
    """Exception raised for errors related to trading strategies."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.STRATEGY_CALCULATION_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        troubleshooting: Optional[List[str]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            error_category=ErrorCategory.STRATEGY,
            severity=severity,
            details=details,
            cause=cause,
            troubleshooting=troubleshooting or [
                "Check strategy parameters for validity",
                "Ensure input data is properly formatted",
                "Verify there's enough historical data for calculations",
                "Check for division by zero or other mathematical errors"
            ]
        )


class ExecutionError(TradingAgentError):
    """Exception raised for errors related to order execution."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.EXECUTION_CONNECTION_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        troubleshooting: Optional[List[str]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            error_category=ErrorCategory.EXECUTION,
            severity=severity,
            details=details,
            cause=cause,
            troubleshooting=troubleshooting or [
                "Check your internet connection",
                "Verify API credentials for the exchange",
                "Ensure you have sufficient funds for the order",
                "Check if the exchange is operational",
                "Verify the order parameters are valid"
            ]
        )


class PortfolioError(TradingAgentError):
    """Exception raised for errors related to portfolio management."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.PORTFOLIO_CALCULATION_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        troubleshooting: Optional[List[str]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            error_category=ErrorCategory.PORTFOLIO,
            severity=severity,
            details=details,
            cause=cause,
            troubleshooting=troubleshooting or [
                "Check portfolio data for consistency",
                "Verify position calculations",
                "Ensure all required portfolio data is available",
                "Check for mathematical errors in calculations"
            ]
        )


class ConfigurationError(TradingAgentError):
    """Exception raised for errors related to configuration."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.CONFIGURATION_FILE_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        troubleshooting: Optional[List[str]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            error_category=ErrorCategory.CONFIGURATION,
            severity=severity,
            details=details,
            cause=cause,
            troubleshooting=troubleshooting or [
                "Check if the configuration file exists and is readable",
                "Verify the configuration file format (YAML, JSON, etc.)",
                "Ensure all required configuration parameters are present",
                "Check for syntax errors in the configuration file"
            ]
        )


# Error handling decorators
def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying a function on specified exceptions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        backoff_factor: Multiplicative factor for exponential backoff.
        exceptions: Exception type(s) to catch and retry on.
        on_retry: Optional callback function to call on each retry.
    
    Returns:
        The decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        if on_retry:
                            on_retry(e, attempt)
                        
                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {str(e)}. "
                            f"Retrying in {current_delay:.2f} seconds..."
                        )
                        
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_attempts} attempts for {func.__name__} failed. "
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
                        details={"function": func.__name__, "max_attempts": max_attempts}
                    )
            
            # This should never happen, but to satisfy the type checker
            raise RuntimeError("Unexpected error in retry decorator")
        
        return wrapper
    
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    half_open_timeout: float = 30.0,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    on_open: Optional[Callable[[], None]] = None,
    on_close: Optional[Callable[[], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator implementing the circuit breaker pattern.
    
    Args:
        failure_threshold: Number of consecutive failures before opening the circuit.
        reset_timeout: Time in seconds before attempting to half-open the circuit.
        half_open_timeout: Time in seconds to wait in half-open state before closing.
        exceptions: Exception type(s) to catch and count as failures.
        on_open: Optional callback function to call when the circuit opens.
        on_close: Optional callback function to call when the circuit closes.
    
    Returns:
        The decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Circuit state
        failures = 0
        state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        last_failure_time = 0.0
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            nonlocal failures, state, last_failure_time
            
            current_time = time.time()
            
            # Check if the circuit is OPEN
            if state == "OPEN":
                if current_time - last_failure_time >= reset_timeout:
                    # Transition to HALF_OPEN
                    logger.info(f"Circuit for {func.__name__} transitioning from OPEN to HALF_OPEN")
                    state = "HALF_OPEN"
                else:
                    # Circuit is still OPEN
                    error_message = f"Circuit for {func.__name__} is OPEN. Retry after {reset_timeout - (current_time - last_failure_time):.2f} seconds"
                    logger.warning(error_message)
                    raise TradingAgentError(
                        message=error_message,
                        error_code=ErrorCode.SYSTEM_RESOURCE_ERROR,
                        error_category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.WARNING,
                        details={"circuit_state": state, "failures": failures}
                    )
            
            try:
                result = func(*args, **kwargs)
                
                # If we're in HALF_OPEN and the call succeeded, close the circuit
                if state == "HALF_OPEN":
                    logger.info(f"Circuit for {func.__name__} transitioning from HALF_OPEN to CLOSED")
                    state = "CLOSED"
                    failures = 0
                    
                    if on_close:
                        on_close()
                
                # If we're in CLOSED, reset the failure count on success
                if state == "CLOSED":
                    failures = 0
                
                return result
                
            except exceptions as e:
                # Increment failure count
                failures += 1
                last_failure_time = current_time
                
                # Check if we need to open the circuit
                if state == "CLOSED" and failures >= failure_threshold:
                    logger.warning(f"Circuit for {func.__name__} transitioning from CLOSED to OPEN after {failures} consecutive failures")
                    state = "OPEN"
                    
                    if on_open:
                        on_open()
                
                # If we're in HALF_OPEN and failed, go back to OPEN
                if state == "HALF_OPEN":
                    logger.warning(f"Circuit for {func.__name__} transitioning from HALF_OPEN back to OPEN after failure")
                    state = "OPEN"
                
                # Re-raise the exception
                if isinstance(e, TradingAgentError):
                    raise e
                else:
                    raise TradingAgentError.from_exception(
                        e,
                        error_code=ErrorCode.UNKNOWN_ERROR,
                        error_category=ErrorCategory.UNKNOWN,
                        severity=ErrorSeverity.ERROR,
                        details={"function": func.__name__, "circuit_state": state, "failures": failures}
                    )
        
        return wrapper
    
    return decorator


def error_handler(
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    error_category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    default_message: str = "An unexpected error occurred",
    troubleshooting: Optional[List[str]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for handling exceptions and converting them to TradingAgentError.
    
    Args:
        error_code: The error code to use for the TradingAgentError.
        error_category: The error category to use for the TradingAgentError.
        severity: The severity level to use for the TradingAgentError.
        default_message: Default error message to use if the exception has no message.
        troubleshooting: List of troubleshooting steps.
    
    Returns:
        The decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except TradingAgentError:
                # Re-raise TradingAgentError instances as-is
                raise
            except Exception as e:
                # Convert other exceptions to TradingAgentError
                raise TradingAgentError.from_exception(
                    e,
                    error_code=error_code,
                    error_category=error_category,
                    severity=severity,
                    details={"function": func.__name__},
                    troubleshooting=troubleshooting
                )
        
        return wrapper
    
    return decorator


# Error logging function
def log_error(error: Union[TradingAgentError, Exception], logger: logging.Logger = logger) -> None:
    """Log an error with appropriate severity level and details.
    
    Args:
        error: The error to log.
        logger: The logger to use.
    """
    if isinstance(error, TradingAgentError):
        error_dict = error.to_dict()
        message = f"[{error_dict['error_code']['name']}] {error_dict['message']}"
        
        # Add details if available
        if error_dict['details']:
            message += f"\nDetails: {error_dict['details']}"
        
        # Add troubleshooting steps if available
        if error_dict['troubleshooting']:
            message += "\nTroubleshooting steps:\n" + "\n".join(f"- {step}" for step in error_dict['troubleshooting'])
        
        # Log with appropriate severity
        if error.severity == ErrorSeverity.INFO:
            logger.info(message)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(message)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(message)
        elif error.severity == ErrorSeverity.CRITICAL:
            logger.critical(message)
    else:
        # For regular exceptions, log as error
        logger.error(f"Unexpected error: {str(error)}\n{traceback.format_exc()}")
