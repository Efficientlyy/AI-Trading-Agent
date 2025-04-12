"""
Test script for error handling components.
"""

import logging
import sys
import os

# Add the parent directory to the path so we can import the backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.errors import (
    DatabaseError, RecordNotFoundError, 
    ValidationError, handle_database_error,
    with_error_handling
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_custom_exceptions():
    """Test custom exception classes."""
    logger.info("Testing custom exceptions...")
    
    # Test RecordNotFoundError
    try:
        raise RecordNotFoundError("Test record not found")
    except RecordNotFoundError as e:
        assert str(e) == "Test record not found", "RecordNotFoundError message incorrect"
        logger.info("RecordNotFoundError test passed")
    
    # Test ValidationError
    try:
        raise ValidationError("Test validation error")
    except ValidationError as e:
        assert str(e) == "Test validation error", "ValidationError message incorrect"
        logger.info("ValidationError test passed")
    
    # Test DatabaseError
    original_error = ValueError("Original error")
    details = {"test": "context"}
    error = DatabaseError("Test database error", original_error, details)
    
    assert str(error) == "Test database error", "DatabaseError message incorrect"
    assert error.original_error == original_error, "Original error not preserved"
    assert error.details == details, "Error details not preserved"
    
    logger.info("DatabaseError test passed")
    return True

def test_error_handler():
    """Test error handler function."""
    logger.info("Testing error handler...")
    
    # Test with ValueError
    original_error = ValueError("Original error")
    handled_error = handle_database_error(original_error, {"test": "context"})
    
    assert isinstance(handled_error, DatabaseError), "Error handling failed"
    assert handled_error.original_error == original_error, "Original error not preserved"
    assert handled_error.details == {"test": "context"}, "Error context not preserved"
    
    logger.info("Error handler test passed")
    return True

def test_error_decorator():
    """Test error handling decorator."""
    logger.info("Testing error handling decorator...")
    
    # Define a custom error handler for testing
    def test_handler(error, context=None):
        return DatabaseError(f"Handled: {str(error)}", error, context or {})
    
    # Test with function that raises an error
    @with_error_handling(error_handler=test_handler)
    def function_with_error():
        raise ValueError("Test error")
    
    try:
        function_with_error()
        assert False, "Error handling decorator failed to raise exception"
    except Exception as e:
        assert isinstance(e, DatabaseError), f"Error handling decorator failed: {type(e)}"
        assert "Handled: Test error" in str(e), f"Error message incorrect: {str(e)}"
    
    # Test with function that returns normally
    @with_error_handling(error_handler=test_handler)
    def function_without_error():
        return "Success"
    
    result = function_without_error()
    assert result == "Success", f"Error handling decorator interfered with normal return: {result}"
    
    logger.info("Error handling decorator test passed")
    return True

def main():
    """Run all error handling tests."""
    logger.info("Starting error handling tests...")
    
    # Run tests
    tests = [
        ("Custom Exceptions", test_custom_exceptions),
        ("Error Handler", test_error_handler),
        ("Error Decorator", test_error_decorator)
    ]
    
    results = []
    
    for name, test_func in tests:
        logger.info(f"Running test: {name}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Test {name} failed with error: {e}")
            results.append((name, False))
    
    # Print results
    logger.info("\nTest Results:")
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{name}: {status}")
    
    # Check if all tests passed
    all_passed = all(success for _, success in results)
    
    if all_passed:
        logger.info("\nAll error handling tests passed successfully!")
    else:
        logger.error("\nSome error handling tests failed!")
    
    return all_passed

if __name__ == "__main__":
    main()
