"""
Run Fixed API Server

This script starts the API server with the fixed configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

# Project path
project_dir = Path(__file__).resolve().parent
print(f"Project directory: {project_dir}")

# Set environment variables for proper operation
os.environ["PYTHONPATH"] = str(project_dir)
os.environ["USE_MOCK_DATA"] = "false"
sys.path.insert(0, str(project_dir))

# Create .env file with proper settings
env_file = project_dir / ".env"
with open(env_file, "w") as f:
    f.write(f"PYTHONPATH={project_dir}\n")
    f.write("USE_MOCK_DATA=false\n")
print(f"Created/updated .env file at {env_file}")

# The script path to run
api_path = project_dir / "ai_trading_agent" / "api" / "main.py"
print(f"Starting API server: {api_path}")

# Start the server
subprocess.Popen([sys.executable, str(api_path)], env=os.environ)

print("\n✅ API server started with fixed configuration!")
print("Please refresh your browser to see the updated dashboard.")
print("\nThe main issues fixed were:")
print("1. Phantom agents showing as running when not started (fixed)")
print("2. Stop functionality not working properly (fixed)")
print("3. Data feed connection issues (fixed with better error handling)")
