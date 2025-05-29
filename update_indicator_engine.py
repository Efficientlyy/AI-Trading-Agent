"""
Script to update the IndicatorEngine class with our improved Rust integration.
This applies the fixes from our fixed_lag_engine.py to the main indicator_engine.py file.
"""

import os
import re

def update_indicator_engine():
    """Update the IndicatorEngine class with our improved Rust integration."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    # Make a backup of the original file if one doesn't already exist
    backup_path = file_path + '.bak'
    if not os.path.exists(backup_path):
        with open(file_path, 'r') as src:
            with open(backup_path, 'w') as dst:
                dst.write(src.read())
        print(f"Created backup of original file at {backup_path}")
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the _calculate_lag_features_rs method to match the correct Rust function signature
    rust_method_pattern = r'def _calculate_lag_features_rs\(self.*?\):'
    rust_method_match = re.search(rust_method_pattern, content, re.DOTALL)
    
    if rust_method_match:
        # Extract the method
        method_start = rust_method_match.start()
        
        # Find the end of the method by looking for the next def
        next_def = re.search(r'def\s+\w+', content[method_start + 10:], re.DOTALL)
        if next_def:
            method_end = method_start + 10 + next_def.start()
            old_method = content[method_start:method_end]
            
            # Create the new method implementation
            new_method = '''def _calculate_lag_features_rs(self, df: pd.DataFrame, lags: List[int], source_column: str = 'close') -> Dict[str, pd.Series]:
        """
        Calculate lag features using Rust implementation with fallback to Python.
        
        Args:
            df: DataFrame with market data
            lags: List of lag periods to calculate
            source_column: Column name to use as source for lag calculation
            
        Returns:
            Dictionary mapping feature names to Series of lag values
        """
        rust_available = hasattr(self, 'rs_features') and self.rs_features is not None
        
        # Add debug logging
        self.logger.debug(f"_calculate_lag_features_rs called with lags: {lags}, rust_available: {rust_available}")
        
        if rust_available and hasattr(self.rs_features, 'create_lag_features_rs'):
            try:
                # Get the values from the source column - this matches the Rust function signature
                # The Rust function only takes the series values, not timestamps
                values = df[source_column].to_numpy()
                
                # Convert lag values to integers
                lags_int = [int(lag) for lag in lags]
                
                self.logger.debug(f"Calling Rust function with values shape: {values.shape}, lags: {lags_int}")
                
                # Call the Rust function with the correct signature
                lag_results = self.rs_features.create_lag_features_rs(values.tolist(), lags_int)
                
                # Convert results to dictionary of Series
                result = {}
                for i, lag in enumerate(lags):
                    key = f'lag_{lag}'
                    # Convert None values to NaN
                    lag_values = [float('nan') if val is None else val for val in lag_results[i]]
                    result[key] = pd.Series(lag_values, index=df.index)
                
                self.logger.debug(f"Successfully calculated lag features using Rust for lags: {lags}")
                return result
            except Exception as e:
                self.logger.error(f"Error in Rust lag features calculation: {e}")
                self.logger.info("Falling back to Python implementation after Rust error.")
                return self._calculate_lag_features(df, lags, source_column)
        else:
            self.logger.info(f"Rust lag features not available. Falling back to Python implementation.")
            return self._calculate_lag_features(df, lags, source_column)
    
    def'''
            
            # Replace the old method with the new one
            updated_content = content.replace(old_method, new_method)
            
            # Write the updated content back to the file
            with open(file_path, 'w') as f:
                f.write(updated_content)
            
            print(f"Updated _calculate_lag_features_rs method in {file_path}")
            return True
        else:
            print("Could not find the end of the _calculate_lag_features_rs method")
            return False
    else:
        print("Could not find _calculate_lag_features_rs method in the file")
        return False

if __name__ == "__main__":
    update_indicator_engine()
