"""
Script to update all Rust indicator integrations in the IndicatorEngine class.
This applies the same pattern we used for lag features to all other Rust-accelerated indicators.
"""

import re

def update_ema_integration():
    """Update the EMA integration to match our successful pattern."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the _calculate_ema method
    ema_pattern = r'def _calculate_ema\(self, df: pd\.DataFrame, periods: List\[int\]\) -> Dict\[str, pd\.Series\]:(.*?)# Python implementation \(fallback\)'
    ema_match = re.search(ema_pattern, content, re.DOTALL)
    
    if ema_match:
        old_rust_section = ema_match.group(1)
        new_rust_section = '''
        """
        Calculate Exponential Moving Average (EMA) for the given periods.
        
        Args:
            df: DataFrame with market data
            periods: List of periods to calculate EMA for
            
        Returns:
            Dictionary mapping period strings to EMA Series
        """
        # Try Rust implementation first
        rust_available = hasattr(self, 'rs_features') and self.rs_features is not None
        
        if rust_available and hasattr(self.rs_features, 'create_ema_features_rs'):
            try:
                self.logger.debug(f"Using Rust implementation for EMA with periods: {periods}")
                
                # Extract data for Rust
                close_values = df['close'].to_numpy()
                
                # Convert to list of integers for Rust
                periods_int = [int(p) for p in periods]
                
                # Call Rust function
                ema_results = self.rs_features.create_ema_features_rs(close_values.tolist(), periods_int)
                
                # Process results
                results = {}
                for i, period in enumerate(periods):
                    # Convert None values to NaN
                    ema_values = [float('nan') if val is None else val for val in ema_results[i]]
                    results[str(period)] = pd.Series(ema_values, index=df.index)
                
                return results
            
            except Exception as e:
                self.logger.error(f"Error in Rust EMA calculation: {e}")
                self.logger.info("Falling back to Python implementation for EMA")
        
        # Python implementation (fallback)'''
        
        updated_content = content.replace(
            'def _calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> Dict[str, pd.Series]:' + old_rust_section + '# Python implementation (fallback)',
            'def _calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> Dict[str, pd.Series]:' + new_rust_section
        )
        
        # Write the updated content back to the file
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print("Updated EMA integration")
        return True
    else:
        print("Could not find EMA method in the file")
        return False

def update_rsi_integration():
    """Update the RSI integration to match our successful pattern."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the _calculate_rsi method
    rsi_pattern = r'def _calculate_rsi\(self, df: pd\.DataFrame, period: int\) -> Dict\[str, pd\.Series\]:(.*?)# Python implementation \(fallback\)'
    rsi_match = re.search(rsi_pattern, content, re.DOTALL)
    
    if rsi_match:
        old_rust_section = rsi_match.group(1)
        new_rust_section = '''
        """
        Calculate Relative Strength Index (RSI) for the given period.
        
        Args:
            df: DataFrame with market data
            period: Period to calculate RSI for
            
        Returns:
            Dictionary mapping period string to RSI Series
        """
        # Try Rust implementation first
        rust_available = hasattr(self, 'rs_features') and self.rs_features is not None
        
        if rust_available and hasattr(self.rs_features, 'create_rsi_features_rs'):
            try:
                self.logger.debug(f"Using Rust implementation for RSI with period: {period}")
                
                # Extract data for Rust
                close_values = df['close'].to_numpy()
                
                # Call Rust function with single period as a list
                rsi_result = self.rs_features.create_rsi_features_rs(close_values.tolist(), [period])
                
                # Convert None values to NaN
                rsi_values = [float('nan') if val is None else val for val in rsi_result[0]]
                
                return {str(period): pd.Series(rsi_values, index=df.index)}
            
            except Exception as e:
                self.logger.error(f"Error in Rust RSI calculation: {e}")
                self.logger.info("Falling back to Python implementation for RSI")
        
        # Python implementation (fallback)'''
        
        updated_content = content.replace(
            'def _calculate_rsi(self, df: pd.DataFrame, period: int) -> Dict[str, pd.Series]:' + old_rust_section + '# Python implementation (fallback)',
            'def _calculate_rsi(self, df: pd.DataFrame, period: int) -> Dict[str, pd.Series]:' + new_rust_section
        )
        
        # Write the updated content back to the file
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print("Updated RSI integration")
        return True
    else:
        print("Could not find RSI method in the file")
        return False

def update_bollinger_bands_integration():
    """Update the Bollinger Bands integration to match our successful pattern."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the _calculate_bollinger_bands method
    bb_pattern = r'def _calculate_bollinger_bands\(self, df: pd\.DataFrame, periods: List\[int\], deviations: float\) -> Dict\[str, Dict\[str, pd\.Series\]\]:(.*?)# Python implementation \(fallback\)'
    bb_match = re.search(bb_pattern, content, re.DOTALL)
    
    if bb_match:
        old_rust_section = bb_match.group(1)
        new_rust_section = '''
        """
        Calculate Bollinger Bands for the given periods and deviations.
        
        Args:
            df: DataFrame with market data
            periods: List of periods to calculate Bollinger Bands for
            deviations: Number of standard deviations for the bands
            
        Returns:
            Dictionary mapping period strings to dictionaries of upper, middle, and lower bands
        """
        # Try Rust implementation first
        rust_available = hasattr(self, 'rs_features') and self.rs_features is not None
        
        if rust_available and hasattr(self.rs_features, 'create_bollinger_bands_rs'):
            try:
                self.logger.debug(f"Using Rust implementation for Bollinger Bands with periods: {periods}, deviations: {deviations}")
                
                # Extract data for Rust
                close_values = df['close'].to_numpy()
                
                # Convert to list of integers for Rust
                periods_int = [int(p) for p in periods]
                
                # Call Rust function
                bb_results = self.rs_features.create_bollinger_bands_rs(close_values.tolist(), periods_int, float(deviations))
                
                # Process results
                results = {}
                for i, period in enumerate(periods):
                    period_key = str(period)
                    results[period_key] = {
                        'upper': pd.Series([float('nan') if val is None else val for val in bb_results[i][0]], index=df.index),
                        'middle': pd.Series([float('nan') if val is None else val for val in bb_results[i][1]], index=df.index),
                        'lower': pd.Series([float('nan') if val is None else val for val in bb_results[i][2]], index=df.index)
                    }
                
                return results
            
            except Exception as e:
                self.logger.error(f"Error in Rust Bollinger Bands calculation: {e}")
                self.logger.info("Falling back to Python implementation for Bollinger Bands")
        
        # Python implementation (fallback)'''
        
        updated_content = content.replace(
            'def _calculate_bollinger_bands(self, df: pd.DataFrame, periods: List[int], deviations: float) -> Dict[str, Dict[str, pd.Series]]:' + old_rust_section + '# Python implementation (fallback)',
            'def _calculate_bollinger_bands(self, df: pd.DataFrame, periods: List[int], deviations: float) -> Dict[str, Dict[str, pd.Series]]:' + new_rust_section
        )
        
        # Write the updated content back to the file
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print("Updated Bollinger Bands integration")
        return True
    else:
        print("Could not find Bollinger Bands method in the file")
        return False

def main():
    """Update all Rust indicator integrations."""
    update_ema_integration()
    update_rsi_integration()
    update_bollinger_bands_integration()
    print("All Rust integrations updated!")

if __name__ == "__main__":
    main()
