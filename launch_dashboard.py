#!/usr/bin/env python3
"""
AI Trading Agent Dashboard Launcher

This script properly launches the integrated dashboard with all necessary
compatibility fixes for Python 3.13, focusing specifically on the pandas/dateutil
compatibility issues.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Apply fix_pandas patch before importing anything else
print("=" * 60)
print("AI Trading Agent Dashboard Launcher")
print("=" * 60)

if sys.version_info >= (3, 13):
    print("Python 3.13 detected - applying compatibility patches...")
    
    # Execute our fixes before importing pandas
    try:
        # Import dateutil.tz fixes
        exec(open(os.path.join(project_root, "fix_pandas.py")).read())
        print("✓ Successfully applied pandas/dateutil fixes")
    except Exception as e:
        print(f"! Warning: Error applying pandas fix: {e}")
else:
    print(f"Running with Python {sys.version_info.major}.{sys.version_info.minor}")
    print("No compatibility fixes needed.")

# Now try to import required modules
try:
    import pandas as pd
    import numpy as np
    from fastapi import FastAPI
    import uvicorn
    
    print(f"✓ Successfully imported required packages:")
    print(f"  - pandas {pd.__version__}")
    print(f"  - numpy {np.__version__}")
    print("  - fastapi")
    print("  - uvicorn")
except ImportError as e:
    print(f"✗ Error importing required packages: {e}")
    print("\nPlease install missing dependencies:")
    print("pip install pandas numpy fastapi uvicorn jinja2 plotly")
    sys.exit(1)

# Import and run the dashboard
try:
    from integrated_dashboard import main as run_dashboard
    
    print("\n" + "=" * 60)
    print("Starting AI Trading Agent Dashboard")
    print("=" * 60)
    print("Features:")
    print("• Main Monitoring Dashboard with realistic market patterns")
    print("• Sentiment Analysis with Continuous Improvement visualization")
    print("• Risk Management metrics with correlated data")
    print("• Market Regime Analysis with scenario generation")
    print("• System Logs and Alerts")
    print("=" * 60)
    print("URL: http://127.0.0.1:8000/")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Run the dashboard
    run_dashboard()
    
except ImportError as e:
    print(f"✗ Error launching dashboard: {e}")
    sys.exit(1)
