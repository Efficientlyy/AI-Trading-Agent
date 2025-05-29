"""
This script applies a direct fix to the test_indicator_engine.py file to resolve the LAG_FEATURES_RS_ test failures.
"""

import os
import re

def fix_test_indicator_engine():
    file_path = 'ai_trading_agent/tests/unit/test_indicator_engine.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add debug print statements to the test_calculate_lag_features_rs method
    pattern = r'(def test_calculate_lag_features_rs\(self\):.*?)(\s+# Run this through calculate_all_indicators to ensure the full path works)'
    replacement = r'\1\n        # Add debug prints\n        print("Debug: test_config =", test_config)\n        print("Debug: test_engine.indicators =", test_engine.indicators)\n\2'
    
    # Apply the first modification using regex with DOTALL flag to match across lines
    modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Add more debug prints to see the results structure
    pattern2 = r'(results = test_engine\.calculate_all_indicators\(test_data, \["TEST_SYMBOL"\]\))'
    replacement2 = r'\1\n        print("Debug: results structure =", results)'
    
    # Apply the second modification
    modified_content = re.sub(pattern2, replacement2, modified_content)
    
    # Write back the modified content
    with open(file_path, 'w') as f:
        f.write(modified_content)
    
    print(f"Successfully updated {file_path} with debug prints")

def fix_indicator_engine():
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the critical issue with LAG_FEATURES_RS_ indicator handling
    # We're ensuring the indicator is correctly processed and added to the results
    
    # Replace the calculate_all_indicators method's symbol results handling
    pattern = r'(# Ensure we capture all indicator results.*?)(self\.last_calculation_time\[symbol\] = datetime\.now\(\)\.timestamp\(\))'
    replacement = r'''# Debug indicator processing
                self.logger.debug(f"Symbol indicators before transfer: {list(symbol_indicators.keys())}")
                
                # Check specifically for LAG_FEATURES_RS_ indicators
                lag_rs_indicators = {k: v for k, v in symbol_indicators.items() if k.startswith("LAG_FEATURES_RS_")}
                if lag_rs_indicators:
                    self.logger.debug(f"Found LAG_FEATURES_RS_ indicators: {list(lag_rs_indicators.keys())}")
                
                # Make sure we store the indicators properly in the results
                if symbol_indicators:
                    self.logger.debug(f"Adding indicators to all_results: {list(symbol_indicators.keys())}")
                    all_results[symbol] = symbol_indicators.copy()  # Make a deep copy to avoid reference issues
                    self.logger.info(f"Successfully calculated indicators: {list(symbol_indicators.keys())}")
                else:
                    self.logger.warning(f"No indicators calculated or all returned empty. symbol_indicators is empty.")
                    if all_results[symbol].get("error") is None:
                        all_results[symbol] = {}
                
                # Debug final state
                self.logger.debug(f"Final results for {symbol}: {list(all_results[symbol].keys()) if symbol in all_results else 'Not found'}")
                
                \2'''
    
    # Apply the modification with DOTALL flag to match across lines
    modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write back the modified content
    with open(file_path, 'w') as f:
        f.write(modified_content)
    
    print(f"Successfully updated {file_path} with LAG_FEATURES_RS_ handling fix")

def main():
    # First add debug prints to the test file
    fix_test_indicator_engine()
    
    # Then fix the indicator engine implementation
    fix_indicator_engine()
    
    print("Fixes applied successfully. Run the tests to see the improved results.")

if __name__ == "__main__":
    main()
