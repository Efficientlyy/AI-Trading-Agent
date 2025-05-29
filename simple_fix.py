"""
A simple, focused fix for the LAG_FEATURES_RS_ indicators in the indicator_engine.py file.
"""

import os
import sys

def fix_indicator_engine():
    """Apply a targeted fix to the indicator_engine.py file."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the LAG_FEATURES_RS_ section in the _init_indicators method
    init_indicators_start = -1
    lag_features_rs_section = -1
    
    for i, line in enumerate(lines):
        if "def _init_indicators" in line:
            init_indicators_start = i
        if "LAG_FEATURES_RS_" in line and "startswith" in line and init_indicators_start > 0:
            lag_features_rs_section = i
            break
    
    if lag_features_rs_section > 0:
        # Check the implementation
        found_enabled = False
        for i in range(lag_features_rs_section, lag_features_rs_section + 20):
            if i < len(lines) and '"enabled": True' in lines[i]:
                found_enabled = True
                break
        
        # If we didn't find enabled: True, add it
        if not found_enabled:
            print("Adding 'enabled': True to LAG_FEATURES_RS_ section")
            for i in range(lag_features_rs_section, lag_features_rs_section + 20):
                if i < len(lines) and '"source_column"' in lines[i]:
                    lines.insert(i + 1, '                    "enabled": True,  # Force enabled to True for RS lag features\n')
                    break
    
    # Create a fixed version of the elif statement for LAG_FEATURES_RS_ in calculate_all_indicators
    # First, find the calculate_all_indicators method
    calculate_all_indicators_start = -1
    lag_features_rs_block_start = -1
    lag_features_rs_block_end = -1
    
    for i, line in enumerate(lines):
        if "def calculate_all_indicators" in line:
            calculate_all_indicators_start = i
        if calculate_all_indicators_start > 0 and "elif indicator_name.startswith(\"LAG_FEATURES_RS_\")" in line:
            lag_features_rs_block_start = i
            break
    
    if lag_features_rs_block_start > 0:
        # Find the end of the LAG_FEATURES_RS_ block
        for i in range(lag_features_rs_block_start, len(lines)):
            if "indicator_duration_ms =" in lines[i] or "elif " in lines[i]:
                lag_features_rs_block_end = i
                break
        
        if lag_features_rs_block_end > 0:
            # Replace the LAG_FEATURES_RS_ block with our fixed implementation
            fixed_block = [
                '                        elif indicator_name.startswith("LAG_FEATURES_RS_"):\n',
                '                            # Special handling for LAG_FEATURES_RS_ indicators\n',
                '                            source_column = settings.get("source_column", "close")\n',
                '                            lags_to_calculate = settings.get("lags_to_calculate", [])\n',
                '                            \n',
                '                            if not lags_to_calculate:\n',
                '                                self.logger.warning(f"No lags_to_calculate provided for {indicator_name}")\n',
                '                                continue\n',
                '                            \n',
                '                            # Calculate lag features directly\n',
                '                            df_copy = df.copy()\n',
                '                            lag_features_dict = self._calculate_lag_features(\n',
                '                                df=df_copy, \n',
                '                                lags=lags_to_calculate,\n',
                '                                source_column=source_column\n',
                '                            )\n',
                '                            \n',
                '                            if lag_features_dict:\n',
                '                                # Convert to DataFrame for the test expectations\n',
                '                                lag_features_df = pd.DataFrame(lag_features_dict)\n',
                '                                self.logger.debug(f"Created lag features DataFrame with shape {lag_features_df.shape}")\n',
                '                                \n',
                '                                # Store the results\n',
                '                                symbol_indicators[indicator_name] = lag_features_df\n',
                '                                calculated_indicator_names_for_symbol.append(indicator_name)\n',
                '                            else:\n',
                '                                self.logger.warning(f"No lag features calculated for {indicator_name}")\n',
                '                                symbol_indicators[indicator_name] = pd.DataFrame()\n',
                '                                calculated_indicator_names_for_symbol.append(indicator_name)\n',
                '                        \n'
            ]
            
            # Replace the block
            lines[lag_features_rs_block_start:lag_features_rs_block_end] = fixed_block
    
    # Write the modified file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Successfully updated {file_path} with LAG_FEATURES_RS_ fixes")

if __name__ == "__main__":
    fix_indicator_engine()
