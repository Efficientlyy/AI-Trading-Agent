"""
Simplified version of the indicator engine that only handles lag features.
This module isolates the lag feature functionality to make it easier to test and debug.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimplifiedLagEngine")

# Import Rust accelerated functions if available
_RUST_AVAILABLE_FLAG = False
rs_features = None  # Initialize as None for the fallback case

# First try ai_trading_agent_rs (the new module name according to lib.rs)
try:
    import ai_trading_agent_rs
    rs_features = ai_trading_agent_rs
    _RUST_AVAILABLE_FLAG = True
    logger.info("Successfully imported 'ai_trading_agent_rs' for Rust acceleration.")
    
    # Explicitly check for the function we need
    if hasattr(rs_features, 'create_lag_features_rs'):
        logger.info("Found 'create_lag_features_rs' function in ai_trading_agent_rs module.")
    else:
        available_funcs = [f for f in dir(rs_features) if not f.startswith('_')]
        logger.warning(f"'create_lag_features_rs' not found in ai_trading_agent_rs module. Available functions: {available_funcs}")
        
# Then try rust_lag_features (the old module name)
except ImportError as e_main:
    logger.warning(f"Could not import 'ai_trading_agent_rs': {e_main}")
    try:
        import rust_lag_features
        rs_features = rust_lag_features
        _RUST_AVAILABLE_FLAG = True
        logger.info("Successfully imported 'rust_lag_features' for Rust acceleration.")
        
        # Explicitly check for the function we need
        if hasattr(rs_features, 'create_lag_features_rs'):
            logger.info("Found 'create_lag_features_rs' function in rust_lag_features module.")
        else:
            available_funcs = [f for f in dir(rs_features) if not f.startswith('_')]
            logger.warning(f"'create_lag_features_rs' not found in rust_lag_features module. Available functions: {available_funcs}")
            
    except ImportError as e_fallback:
        logger.warning(f"Could not import 'rust_lag_features': {e_fallback}")
        logger.warning("Neither 'ai_trading_agent_rs' nor 'rust_lag_features' available. Using Python fallbacks.")
        # Keep rs_features as None

class SimplifiedLagEngine:
    """
    A simplified version of the indicator engine that only handles lag features.
    This makes it easier to test and debug the Rust integration.
    """
    
    def __init__(self):
        """Initialize the simplified lag engine."""
        self.logger = logger
        self.rust_available = _RUST_AVAILABLE_FLAG
        self.rs_features = rs_features
        
        # Check if specific Rust functions exist
        if self.rust_available:
            self.has_rust_lag_features = hasattr(self.rs_features, 'create_lag_features_rs')
            if self.has_rust_lag_features:
                self.logger.info("Rust 'create_lag_features_rs' function is available.")
            else:
                self.logger.warning("Rust module imported but 'create_lag_features_rs' function not found.")
        else:
            self.has_rust_lag_features = False
        
    def calculate_lag_features_python(self, df: pd.DataFrame, lags: List[int], source_column: str = 'close') -> Dict[str, pd.Series]:
        """
        Calculate lag features using pure Python implementation.
        
        Args:
            df: DataFrame with market data
            lags: List of lag periods to calculate
            source_column: Column name to use as source for lag calculation
            
        Returns:
            Dictionary mapping feature names to Series of lag values
        """
        self.logger.debug(f"Calculating lag features using Python implementation for lags: {lags}")
        result = {}
        for lag in lags:
            key = f'lag_{lag}'
            result[key] = df[source_column].shift(lag)
        return result
    
    def calculate_lag_features_rust(self, df: pd.DataFrame, lags: List[int], source_column: str = 'close') -> Dict[str, pd.Series]:
        """
        Calculate lag features using Rust implementation.
        
        Args:
            df: DataFrame with market data
            lags: List of lag periods to calculate
            source_column: Column name to use as source for lag calculation
            
        Returns:
            Dictionary mapping feature names to Series of lag values
        """
        if not self.rust_available or not self.has_rust_lag_features:
            self.logger.warning("Rust lag features not available. Falling back to Python implementation.")
            return self.calculate_lag_features_python(df, lags, source_column)
        
        self.logger.debug(f"Calculating lag features using Rust implementation for lags: {lags}")
        try:
            # Extract the data we need
            timestamps = df.index.to_numpy()
            values = df[source_column].to_numpy()
            
            # Call the Rust function
            lag_results = self.rs_features.create_lag_features_rs(timestamps, values, lags)
            
            # Convert results to dictionary of Series
            result = {}
            for i, lag in enumerate(lags):
                key = f'lag_{lag}'
                result[key] = pd.Series(lag_results[i], index=df.index)
            
            return result
        except Exception as e:
            self.logger.error(f"Error in Rust lag features calculation: {e}")
            self.logger.info("Falling back to Python implementation after Rust error.")
            return self.calculate_lag_features_python(df, lags, source_column)
    
    def calculate_lag_features(self, df: pd.DataFrame, lags: List[int], source_column: str = 'close', force_python: bool = False) -> Dict[str, pd.Series]:
        """
        Calculate lag features using the best available implementation.
        
        Args:
            df: DataFrame with market data
            lags: List of lag periods to calculate
            source_column: Column name to use as source for lag calculation
            force_python: If True, use Python implementation regardless of Rust availability
            
        Returns:
            Dictionary mapping feature names to Series of lag values
        """
        if force_python or not self.rust_available or not self.has_rust_lag_features:
            return self.calculate_lag_features_python(df, lags, source_column)
        else:
            return self.calculate_lag_features_rust(df, lags, source_column)


# Test functionality if run directly
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'close': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    }, index=dates)
    
    # Create the engine
    engine = SimplifiedLagEngine()
    
    # Test Python implementation
    python_results = engine.calculate_lag_features_python(data, [1, 3, 5])
    print("\nPython Implementation Results:")
    for key, series in python_results.items():
        print(f"{key}:\n{series}")
    
    # Test Rust implementation (with fallback to Python if needed)
    if engine.rust_available and engine.has_rust_lag_features:
        rust_results = engine.calculate_lag_features_rust(data, [1, 3, 5])
        print("\nRust Implementation Results:")
        for key, series in rust_results.items():
            print(f"{key}:\n{series}")
        
        # Compare results
        print("\nComparing Python and Rust results:")
        for key in python_results.keys():
            match = python_results[key].equals(rust_results[key])
            print(f"{key}: {'Match' if match else 'Mismatch'}")
    else:
        print("\nRust implementation not available, skipping comparison.")
