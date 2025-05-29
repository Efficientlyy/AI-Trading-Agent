"""
Script to restore the indicator_engine.py file from a clean version.
"""

def restore_indicator_engine():
    """Restore the indicator_engine.py file from a clean version."""
    # Check if we have a backup
    import os
    backup_path = 'ai_trading_agent/agent/indicator_engine.py.bak'
    
    if os.path.exists(backup_path):
        # Restore from backup
        with open(backup_path, 'r') as src:
            with open('ai_trading_agent/agent/indicator_engine.py', 'w') as dst:
                dst.write(src.read())
        print("Restored indicator_engine.py from backup")
        return True
    else:
        print("No backup found. Cannot restore indicator_engine.py")
        return False

if __name__ == "__main__":
    restore_indicator_engine()
