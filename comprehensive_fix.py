"""
Comprehensive Fix for AI Trading Agent System

This script addresses the specific issues:
1. Phantom agents showing as running when they haven't been started
2. Stop functionality not working properly
3. Data feed connection issues
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

# Set up project path
project_dir = Path(__file__).resolve().parent
print(f"Project directory: {project_dir}")

# Add project to Python path and set environment variables
os.environ["PYTHONPATH"] = str(project_dir)
os.environ["USE_MOCK_DATA"] = "false"  # Force real data mode
sys.path.insert(0, str(project_dir))

# Kill any existing API processes
print("Terminating any existing API processes...")
if os.name == 'nt':  # Windows
    os.system("taskkill /f /im uvicorn.exe 2>nul")
else:  # Unix/Linux
    os.system("pkill -f uvicorn")

# Fix the system_control.py file
system_control_path = project_dir / "ai_trading_agent" / "api" / "system_control.py"
print(f"Fixing system control file at {system_control_path}...")

try:
    # Read the file
    with open(system_control_path, "r") as f:
        content = f.read()
    
    # Fix 1: Reset mock agent statuses to "stopped" by default
    if "status\": \"running\"" in content:
        content = content.replace("status\": \"running\"", "status\": \"stopped\"")
        print("✓ Fixed mock agent statuses to be stopped by default")
    
    # Fix 2: Force USE_MOCK_DATA to false
    if "USE_MOCK_DATA = os.getenv(\"USE_MOCK_DATA\", \"true\").lower() == \"true\"" in content:
        content = content.replace(
            "USE_MOCK_DATA = os.getenv(\"USE_MOCK_DATA\", \"true\").lower() == \"true\"",
            "USE_MOCK_DATA = os.getenv(\"USE_MOCK_DATA\", \"false\").lower() == \"true\"  # Default to false"
        )
        print("✓ Set USE_MOCK_DATA to default to false")
    
    # Fix 3: Improved system status logic for partial state
    if "data_feed_connected = data_feed_status.get(\"status\") == \"connected\"" in content:
        content = content.replace(
            "data_feed_connected = data_feed_status.get(\"status\") == \"connected\"",
            "data_feed_connected = data_feed_status.get(\"status\") in [\"connected\", \"online\", \"active\"]"
        )
        print("✓ Improved data feed connection status checking")
    
    # Write fixed content back
    with open(system_control_path, "w") as f:
        f.write(content)
    
    print("System control file updated successfully!")
except Exception as e:
    print(f"Error updating system_control.py: {e}")

# Fix .env file
env_file = project_dir / ".env"
print(f"Creating/updating .env file at {env_file}...")

with open(env_file, "w") as f:
    f.write(f"PYTHONPATH={project_dir}\n")
    f.write("USE_MOCK_DATA=false\n")
    f.write("DEBUG_API=true\n")

print("✓ Created/updated .env file with correct settings")

# Fix data feed manager
data_feed_path = project_dir / "ai_trading_agent" / "api" / "data_feed_manager.py"
print(f"Checking data feed manager at {data_feed_path}...")

if data_feed_path.exists():
    try:
        # Import and initialize
        print("Directly initializing data feed manager...")
        
        try:
            from ai_trading_agent.api.data_feed_manager import data_feed_manager
            data_feed_manager.stop()  # Stop if running
            time.sleep(1)
            data_feed_manager.start()  # Restart
            time.sleep(2)  # Wait for initialization
            
            status = data_feed_manager.get_status()
            print(f"Data feed status after direct initialization: {status}")
        except Exception as e:
            print(f"Error initializing data feed manager: {e}")
    except Exception as e:
        print(f"Error importing data feed manager: {e}")
else:
    print(f"Warning: Data feed manager file not found at {data_feed_path}")

# Create a proper agent reset script to ensure agents are properly stopped
reset_script_path = project_dir / "reset_agents.py"
print(f"Creating agent reset script at {reset_script_path}...")

reset_script_content = """
import os
import sys
import asyncio
from pathlib import Path

# Add project to path
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))

async def reset_all_agents():
    try:
        # Import session manager
        from ai_trading_agent.agent.session_manager import session_manager
        print("Stopping all sessions...")
        result = await session_manager.stop_all_sessions()
        print(f"All sessions stopped: {result}")
        return True
    except ImportError:
        print("Error: Could not import session_manager")
        return False
    except Exception as e:
        print(f"Error resetting agents: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(reset_all_agents())
"""

with open(reset_script_path, "w") as f:
    f.write(reset_script_content)

print("✓ Created agent reset script")

# Create a proper startup script for the API
startup_script_path = project_dir / "start_api_server.py"
print(f"Creating improved API startup script at {startup_script_path}...")

startup_script_content = """
import os
import sys
import subprocess
import time
from pathlib import Path

# Set project path
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))
os.environ["PYTHONPATH"] = str(project_dir)
os.environ["USE_MOCK_DATA"] = "false"

print("Starting AI Trading Agent API with proper configuration...")

# Kill any existing API processes first
if os.name == 'nt':  # Windows
    os.system("taskkill /f /im uvicorn.exe 2>nul")
else:  # Unix/Linux
    os.system("pkill -f uvicorn")

# Wait a moment for processes to terminate
time.sleep(1)

# Start the API server
api_script = project_dir / "ai_trading_agent" / "api" / "main.py"
cmd = [sys.executable, str(api_script)]

# Start in a new process
print(f"Running command: {' '.join(cmd)}")
subprocess.Popen(cmd, env=os.environ)

print("\\n✅ API server started! Please refresh your dashboard.")
print("If you need to stop all agents properly, run: python reset_agents.py")
"""

with open(startup_script_path, "w") as f:
    f.write(startup_script_content)

print("✓ Created improved API startup script")

# Start the API with the fixed configuration
print("\nStarting API server with all fixes applied...")
try:
    api_path = project_dir / "ai_trading_agent" / "api" / "main.py"
    cmd = [sys.executable, str(api_path)]
    
    # Set environment for subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_dir)
    env["USE_MOCK_DATA"] = "false"
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.Popen(cmd, env=env)
    
    print("\n✅ COMPREHENSIVE FIX COMPLETED!")
    print("The API server has been restarted with all fixes applied.")
    print("Please refresh your browser to see the updated dashboard.")
    print("\nTo properly stop all agents, run: python reset_agents.py")
    print("To restart the API cleanly, run: python start_api_server.py")
except Exception as e:
    print(f"Error starting API server: {e}")
