"""
Script to fix the try-except syntax error in indicator_engine.py.
"""

def fix_try_except_syntax():
    """Fix the try-except syntax error in indicator_engine.py."""
    file_path = 'ai_trading_agent/agent/indicator_engine.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the broken try-except block
    fixed_content = content.replace(
        """try:
            self.logger.critical("ENTERING _init_indicators - VERY EARLY TEST")
            except Exception as e:
        # Fallback print if logger itself is problematic here
            print(f"CRITICAL: FAILED TO LOG ENTRY TO _init_indicators: {e}")""",
        """try:
            self.logger.critical("ENTERING _init_indicators - VERY EARLY TEST")
        except Exception as e:
            # Fallback print if logger itself is problematic here
            print(f"CRITICAL: FAILED TO LOG ENTRY TO _init_indicators: {e}")""")
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed try-except syntax in {file_path}")

if __name__ == "__main__":
    fix_try_except_syntax()
