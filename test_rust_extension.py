"""
Test script for the Rust extension using CFFI.
"""
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the CFFI loader
from ai_trading_agent.cffi_loader import add_numbers, multiply_numbers

# Test the Rust functions
def test_rust_functions():
    # Test add_numbers
    a, b = 5, 7
    result = add_numbers(a, b)
    logger.info(f"add_numbers({a}, {b}) = {result}")
    assert result == a + b, f"Expected {a + b}, got {result}"
    
    # Test multiply_numbers
    a, b = 3.5, 2.5
    result = multiply_numbers(a, b)
    logger.info(f"multiply_numbers({a}, {b}) = {result}")
    assert result == a * b, f"Expected {a * b}, got {result}"
    
    logger.info("All tests passed!")

if __name__ == "__main__":
    test_rust_functions()
