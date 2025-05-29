"""
Script to fix the indentation in the import try-except blocks in indicator_engine.py.
"""

def fix_import_indentation():
    """Fix the indentation in the import try-except blocks in indicator_engine.py."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the indentation for the first try-except block
    fixed_content = content.replace(
        """try:
    import ai_trading_agent_rs as rs_features_imported
    _rs_features_module = rs_features_imported
    _RUST_AVAILABLE_FLAG = True
    logging.info("Successfully imported 'ai_trading_agent_rs' for Rust acceleration.")
            except ImportError as e:""",
        """try:
    import ai_trading_agent_rs as rs_features_imported
    _rs_features_module = rs_features_imported
    _RUST_AVAILABLE_FLAG = True
    logging.info("Successfully imported 'ai_trading_agent_rs' for Rust acceleration.")
except ImportError as e:""")
    
    # Fix the indentation for the nested try-except block
    fixed_content = fixed_content.replace(
        """    try:
        # Fallback to older module name if it exists
        import rust_lag_features as rs_features_imported
        _rs_features_module = rs_features_imported
        _RUST_AVAILABLE_FLAG = True
        logging.info("Successfully imported 'rust_lag_features' for Rust acceleration.")
            except ImportError:""",
        """    try:
        # Fallback to older module name if it exists
        import rust_lag_features as rs_features_imported
        _rs_features_module = rs_features_imported
        _RUST_AVAILABLE_FLAG = True
        logging.info("Successfully imported 'rust_lag_features' for Rust acceleration.")
    except ImportError:""")
    
    # Fix the indentation for the following line
    fixed_content = fixed_content.replace(
        """        _rs_features_module = None # Ensure rs_features is defined for type hinting or attribute access attempts
        _RUST_AVAILABLE_FLAG = False
        logging.warning(f"Neither 'ai_trading_agent_rs' nor 'rust_lag_features' available. Using Python fallbacks. Error: {e}")""",
        """        _rs_features_module = None # Ensure rs_features is defined for type hinting or attribute access attempts
        _RUST_AVAILABLE_FLAG = False
        logging.warning(f"Neither 'ai_trading_agent_rs' nor 'rust_lag_features' available. Using Python fallbacks. Error: {e}")""")
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed import indentation in {file_path}")

if __name__ == "__main__":
    fix_import_indentation()
