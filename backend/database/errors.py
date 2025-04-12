"""
Error handling module for database operations.
"""

from typing import Optional, Dict, Any, Type
import logging
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError, DataError

# Set up logger
logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base class for database errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.original_error = original_error
        self.details = details or {}
        super().__init__(self.message)


class RecordNotFoundError(DatabaseError):
    """Error raised when a record is not found."""
    pass


class DuplicateRecordError(DatabaseError):
    """Error raised when a duplicate record is attempted to be created."""
    pass


class ValidationError(DatabaseError):
    """Error raised when data validation fails."""
    pass


class ConnectionError(DatabaseError):
    """Error raised when a database connection error occurs."""
    pass


class TransactionError(DatabaseError):
    """Error raised when a transaction error occurs."""
    pass


def handle_database_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> DatabaseError:
    """
    Handle database errors and convert them to appropriate custom exceptions.
    
    Args:
        error: The original error
        context: Additional context information
        
    Returns:
        A custom DatabaseError
    """
    context = context or {}
    error_message = str(error)
    
    # Log the error with context
    log_message = f"Database error: {error_message}"
    if context:
        log_message += f" Context: {context}"
    logger.error(log_message, exc_info=True)
    
    # Map SQLAlchemy errors to custom exceptions
    if isinstance(error, IntegrityError):
        if "unique constraint" in error_message.lower() or "duplicate" in error_message.lower():
            return DuplicateRecordError("A record with these details already exists", error, context)
        return ValidationError("Data integrity error", error, context)
    
    elif isinstance(error, OperationalError):
        return ConnectionError("Database connection error", error, context)
    
    elif isinstance(error, DataError):
        return ValidationError("Invalid data format", error, context)
    
    elif isinstance(error, SQLAlchemyError):
        return TransactionError("Database transaction error", error, context)
    
    # Default case
    return DatabaseError("An unexpected database error occurred", error, context)


def with_error_handling(func=None, error_handler=None):
    """
    Decorator for handling database errors in repository methods.
    
    Args:
        func: The function to decorate
        error_handler: Optional custom error handler function
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RecordNotFoundError:
                # Re-raise custom exceptions
                raise
            except DatabaseError:
                # Re-raise custom exceptions
                raise
            except SQLAlchemyError as e:
                # Convert SQLAlchemy errors to custom exceptions
                context = {
                    "function": func.__name__,
                    "args": args[1:] if args else [],  # Skip self
                    "kwargs": kwargs
                }
                if error_handler:
                    raise error_handler(e, context)
                else:
                    raise handle_database_error(e, context)
            except Exception as e:
                # Log unexpected errors
                logger.exception(f"Unexpected error in {func.__name__}: {e}")
                if error_handler:
                    raise error_handler(e, {"function": func.__name__})
                else:
                    raise
        
        return wrapper
    
    # This allows the decorator to be used with or without arguments
    if func is None:
        return decorator
    return decorator(func)
