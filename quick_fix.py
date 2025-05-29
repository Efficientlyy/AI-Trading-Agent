"""
Quick Fix for AI Trading Agent System

This script implements a simple, direct fix for the data feed connection issues.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Set up project path
project_dir = Path(__file__).resolve().parent
print(f"Project directory: {project_dir}")

# Create .env file with PYTHONPATH
env_file = project_dir / ".env"
with open(env_file, "w") as f:
    f.write(f"PYTHONPATH={project_dir}\n")
    f.write("USE_MOCK_DATA=false\n")
print(f"Created .env file with PYTHONPATH at {env_file}")

# Set environment variables for current process
os.environ["PYTHONPATH"] = str(project_dir)
os.environ["USE_MOCK_DATA"] = "false"
sys.path.insert(0, str(project_dir))

# Start the data feed manager directly
try:
    from ai_trading_agent.api.data_feed_manager import data_feed_manager
    print("Starting data feed manager...")
    data_feed_manager.start()
    time.sleep(2)  # Give it a moment to initialize
    status = data_feed_manager.get_status()
    print(f"Data feed status: {status}")
except ImportError:
    print("Error: Could not import data_feed_manager. Make sure your Python path is set correctly.")
    print(f"Current sys.path: {sys.path}")
except Exception as e:
    print(f"Error starting data feed manager: {e}")

print("\nKilling any existing API server processes...")
try:
    if os.name == 'nt':  # Windows
        os.system("taskkill /f /im uvicorn.exe 2>nul")
    else:  # Unix/Linux
        os.system("pkill -f uvicorn")
except Exception as e:
    print(f"Note: {e}")

print("\nStarting API server with fixed environment...")
try:
    # Set environment for subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_dir)
    env["USE_MOCK_DATA"] = "false"
    
    # Start the server in a subprocess
    api_path = project_dir / "ai_trading_agent" / "api" / "main.py"
    cmd = [sys.executable, str(api_path)]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.Popen(cmd, env=env)
    
    print("\n✅ API server started with fixed environment!")
    print("Please refresh your browser to see if the dashboard is now more stable.")
    print("If issues persist, try running the system with:")
    print(f"    set PYTHONPATH={project_dir}")
    print(f"    python {project_dir}/ai_trading_agent/api/main.py")
except Exception as e:
    print(f"Error starting API server: {e}")
