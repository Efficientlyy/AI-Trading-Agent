"""
Script to run the log dashboard with proper path configuration.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the Python path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

# Import and run the dashboard
try:
    # Import the run_dashboard function from log_dashboard
    from src.dashboard.log_dashboard import run_dashboard
    
    if __name__ == "__main__":
        print("Starting log dashboard at http://127.0.0.1:8050/")
        print("Press Ctrl+C to stop the dashboard")
        # Run the dashboard
        run_dashboard(debug=True)
except ImportError as e:
    print(f"Error importing dashboard: {e}")
    print("\nYou may need to install the required dependencies:")
    print("pip install dash dash-bootstrap-components pandas plotly structlog psutil")
