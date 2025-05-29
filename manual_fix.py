"""
Script to manually fix the indicator_engine.py file focusing only on the LAG_FEATURES_RS_ indicator handling.
"""

import os

def fix_lag_features_rs():
    # First, let's check and make a backup of the original file
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    backup_path = 'ai_trading_agent/agent/indicator_engine.py.bak'
    
    # Create a backup if it doesn't exist
    if not os.path.exists(backup_path):
        with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        print(f"Created backup at {backup_path}")
    
    # Now let's fix just the LAG_FEATURES_RS_ indicator handling section
    # We'll read the entire file and make minimal targeted changes
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the LAG_FEATURES_RS_ section in the calculate_all_indicators method
    lag_features_rs_start = -1
    lag_features_rs_end = -1
    
    for i, line in enumerate(lines):
        if "indicator_name.startswith(\"LAG_FEATURES_RS_\")" in line:
            lag_features_rs_start = i
            break
    
    if lag_features_rs_start > 0:
        # Find the end of this section (next elif or the end of the method)
        for i in range(lag_features_rs_start + 1, len(lines)):
            if "elif " in lines[i] or "indicator_duration_ms =" in lines[i]:
                lag_features_rs_end = i
                break
        
        if lag_features_rs_end > 0:
            # Replace the LAG_FEATURES_RS_ section with a fixed implementation
            fixed_section = [
                "                        elif indicator_name.startswith(\"LAG_FEATURES_RS_\"):\n",
                "                            # Special handling for LAG_FEATURES_RS_ indicators\n",
                "                            self.logger.debug(f\"Processing LAG_FEATURES_RS_ indicator: {indicator_name}\")\n",
                "                            source_column = settings.get(\"source_column\", \"close\")\n",
                "                            lags_to_calculate = settings.get(\"lags_to_calculate\", [])\n",
                "                            \n",
                "                            if not lags_to_calculate:\n",
                "                                self.logger.warning(f\"No lags provided for {indicator_name}\")\n",
                "                                continue\n",
                "                            \n",
                "                            # Calculate lag features directly\n",
                "                            df_copy = df.copy()\n",
                "                            lag_features_dict = self._calculate_lag_features(\n",
                "                                df=df_copy, \n",
                "                                lags=lags_to_calculate,\n",
                "                                source_column=source_column\n",
                "                            )\n",
                "                            \n",
                "                            if lag_features_dict:\n",
                "                                # Convert to DataFrame for the tests\n",
                "                                lag_features_df = pd.DataFrame(lag_features_dict)\n",
                "                                self.logger.debug(f\"Created lag features DataFrame with shape {lag_features_df.shape}\")\n",
                "                                \n",
                "                                # Store in results directly in the symbols dictionary\n",
                "                                symbol_indicators[indicator_name] = lag_features_df\n",
                "                                calculated_indicator_names_for_symbol.append(indicator_name)\n",
                "                                self.logger.debug(f\"Added {indicator_name} to results with columns: {lag_features_df.columns.tolist()}\")\n",
                "                            else:\n",
                "                                self.logger.warning(f\"No lag features calculated for {indicator_name}\")\n",
                "                                # Add an empty DataFrame as a placeholder to avoid KeyError\n",
                "                                symbol_indicators[indicator_name] = pd.DataFrame()\n",
                "                                calculated_indicator_names_for_symbol.append(indicator_name)\n",
                "                        \n"
            ]
            
            # Replace the section
            new_lines = lines[:lag_features_rs_start] + fixed_section + lines[lag_features_rs_end:]
            
            # Write the changes back to the file
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            
            print(f"Successfully fixed LAG_FEATURES_RS_ indicator handling in {file_path}")
            return True
        else:
            print("Could not find the end of the LAG_FEATURES_RS_ section")
    else:
        print("Could not find the LAG_FEATURES_RS_ section")
    
    return False

if __name__ == "__main__":
    fix_lag_features_rs()
