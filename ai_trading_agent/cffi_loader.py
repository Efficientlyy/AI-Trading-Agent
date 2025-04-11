"""
CFFI loader for Rust extensions.
This module uses CFFI to load Rust-compiled dynamic libraries directly.
"""
import os
import sys
import logging
from pathlib import Path
import platform
import json
import traceback

# Setup logging
logger = logging.getLogger(__name__)

# Determine the appropriate file extension for the dynamic library based on the platform
if platform.system() == "Windows":
    LIB_EXT = ".dll"
elif platform.system() == "Darwin":  # macOS
    LIB_EXT = ".dylib"
else:  # Linux and others
    LIB_EXT = ".so"

def load_rust_extension(extension_name, lib_path=None):
    """
    Load a Rust extension using CFFI.
    
    Args:
        extension_name: Name of the extension
        lib_path: Path to the dynamic library (optional)
        
    Returns:
        The loaded extension module or None if loading fails
    """
    try:
        from cffi import FFI
    except ImportError:
        logger.error("CFFI is not installed. Please install it with 'pip install cffi'.")
        return None
    
    # If lib_path is not provided, try to find the library in standard locations
    if lib_path is None:
        # Check in the current directory
        current_dir = Path(__file__).parent
        possible_paths = [
            current_dir / f"lib{extension_name}{LIB_EXT}",
            current_dir / f"{extension_name}{LIB_EXT}",
            current_dir.parent / "target" / "release" / f"lib{extension_name}{LIB_EXT}",
            current_dir.parent / "target" / "release" / f"{extension_name}{LIB_EXT}",
            current_dir.parent / f"{extension_name}" / "target" / "release" / f"lib{extension_name}{LIB_EXT}",
            current_dir.parent / f"{extension_name}" / "target" / "release" / f"{extension_name}{LIB_EXT}",
            # Add more specific paths for our project
            current_dir.parent / "rust_simple_extension" / "target" / "release" / f"lib{extension_name}{LIB_EXT}",
            current_dir.parent / "rust_simple_extension" / "target" / "release" / f"{extension_name}{LIB_EXT}",
        ]
        
        for path in possible_paths:
            if path.exists():
                lib_path = str(path)
                logger.info(f"Found library at {lib_path}")
                break
        
        if lib_path is None:
            logger.error(f"Could not find the {extension_name} library. Please build it first.")
            return None
    
    # Create FFI instance
    ffi = FFI()
    
    # Define the C interface
    ffi.cdef("""
        int64_t add_numbers(int64_t a, int64_t b);
        double multiply_numbers(double a, double b);
        char* run_backtest_c(const char* data_json, const char* config_json);
        void free_string(char* ptr);
    """)
    
    try:
        # Load the library
        lib = ffi.dlopen(lib_path)
        # Create a class wrapper to hold both the library and FFI instance
        class RustLib:
            def __init__(self, lib, ffi):
                self.lib = lib
                self.ffi = ffi
                
            def add_numbers(self, a, b):
                return self.lib.add_numbers(a, b)
                
            def multiply_numbers(self, a, b):
                return self.lib.multiply_numbers(a, b)
                
            def run_backtest_c(self, data_json, config_json):
                return self.lib.run_backtest_c(data_json, config_json)
                
            def free_string(self, ptr):
                return self.lib.free_string(ptr)
        
        rust_lib = RustLib(lib, ffi)
        logger.info(f"Successfully loaded {extension_name} from {lib_path}")
        return rust_lib
    except Exception as e:
        logger.error(f"Failed to load {extension_name}: {e}")
        logger.error(traceback.format_exc())
        return None

# Try to load the rust_simple_extension
rust_simple_extension = load_rust_extension("rust_simple_extension")

# Define Python wrappers for the Rust functions
if rust_simple_extension is not None:
    def add_numbers(a, b):
        """
        Add two numbers using the Rust implementation.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        return rust_simple_extension.add_numbers(a, b)
    
    def multiply_numbers(a, b):
        """
        Multiply two numbers using the Rust implementation.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b
        """
        return rust_simple_extension.multiply_numbers(a, b)
    
    def run_backtest(data, config):
        """
        Run a backtest using the Rust implementation.
        
        Args:
            data: List of OHLCV bars
            config: Backtest configuration
            
        Returns:
            Backtest results
        """
        try:
            # Convert data and config to JSON strings
            data_json = json.dumps(data)
            config_json = json.dumps(config)
            
            # Convert Python strings to C strings
            data_c_str = rust_simple_extension.ffi.new("char[]", data_json.encode('utf-8'))
            config_c_str = rust_simple_extension.ffi.new("char[]", config_json.encode('utf-8'))
            
            # Call the Rust function
            result_ptr = rust_simple_extension.run_backtest_c(data_c_str, config_c_str)
            
            if result_ptr == rust_simple_extension.ffi.NULL:
                logger.error("Rust function returned NULL pointer")
                return run_backtest_python(data, config)
            
            # Convert the result back to a Python string
            result_str = rust_simple_extension.ffi.string(result_ptr).decode('utf-8')
            
            # Free the memory allocated by Rust
            rust_simple_extension.free_string(result_ptr)
            
            # Parse the JSON result
            result = json.loads(result_str)
            
            # Check if there was an error
            if "error" in result:
                logger.error(f"Error in Rust backtest: {result['error']}")
                return run_backtest_python(data, config)
            
            return result
        except Exception as e:
            logger.error(f"Error running backtest in Rust: {e}")
            logger.error(traceback.format_exc())
            logger.warning("Falling back to Python implementation")
            return run_backtest_python(data, config)
else:
    # Fallback to Python implementations
    logger.warning("Using Python fallback implementations")
    
    def add_numbers(a, b):
        """Python fallback implementation of add_numbers"""
        return a + b
    
    def multiply_numbers(a, b):
        """Python fallback implementation of multiply_numbers"""
        return a * b
    
    def run_backtest(data, config):
        """Python fallback implementation of run_backtest"""
        return run_backtest_python(data, config)

def run_backtest_python(data, config):
    """
    Python implementation of the backtesting functionality.
    
    Args:
        data: List of OHLCV bars
        config: Backtest configuration
        
    Returns:
        Backtest results
    """
    logger.info("Running backtest using Python implementation")
    
    # Extract configuration parameters
    initial_capital = config.get('initial_capital', 10000.0)
    commission_rate = config.get('commission_rate', 0.001)
    slippage = config.get('slippage', 0.001)
    
    # Initialize variables
    capital = initial_capital
    position = 0.0
    returns = []
    equity_curve = []
    
    # Simple moving average parameters
    fast_period = 10
    slow_period = 30
    
    # Ensure we have enough data
    if len(data) <= slow_period:
        logger.warning("Not enough data for backtesting")
        return {
            'final_capital': initial_capital,
            'metrics': {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        }
    
    # Calculate moving averages and execute trading strategy
    for i in range(slow_period, len(data)):
        # Calculate fast and slow moving averages
        fast_ma = sum(bar['close'] for bar in data[i-fast_period:i]) / fast_period
        slow_ma = sum(bar['close'] for bar in data[i-slow_period:i]) / slow_period
        
        # Trading logic: Buy when fast MA crosses above slow MA, sell when it crosses below
        if fast_ma > slow_ma and position == 0.0:
            # Buy signal
            price = data[i]['close'] * (1.0 + slippage)
            shares = (capital * 0.95) / price  # Use 95% of capital
            cost = shares * price * (1.0 + commission_rate)
            
            if cost <= capital:
                position = shares
                capital -= cost
        elif fast_ma < slow_ma and position > 0.0:
            # Sell signal
            price = data[i]['close'] * (1.0 - slippage)
            proceeds = position * price * (1.0 - commission_rate)
            
            capital += proceeds
            position = 0.0
        
        # Calculate portfolio value and daily return
        portfolio_value = capital + (position * data[i]['close'])
        
        if i > slow_period and equity_curve:
            prev_value = equity_curve[-1]
            daily_return = (portfolio_value - prev_value) / prev_value
            returns.append(daily_return)
        
        equity_curve.append(portfolio_value)
    
    # Calculate final portfolio value
    final_capital = capital
    if position > 0.0 and data:
        final_capital += position * data[-1]['close']
    
    # Calculate performance metrics
    total_return = (final_capital - initial_capital) / initial_capital
    
    # Calculate Sharpe ratio (simplified)
    if returns:
        avg_return = sum(returns) / len(returns)
        std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe_ratio = (avg_return / std_dev) * (252 ** 0.5) if std_dev > 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    # Calculate max drawdown
    max_drawdown = 0.0
    peak = initial_capital
    
    for value in equity_curve:
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'final_capital': final_capital,
        'metrics': {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    }
