"""
Simple debug fix for agent start/stop issues.
This script just modifies the minimum needed files to fix the agent control issues.
"""

import os
import sys
from pathlib import Path

# Get project paths
PROJECT_ROOT = Path(__file__).resolve().parent
SYSTEM_CONTROL_PATH = PROJECT_ROOT / "ai_trading_agent" / "api" / "system_control.py"

# Force mock mode
print("Setting USE_MOCK_DATA = True in system_control.py")
with open(SYSTEM_CONTROL_PATH, "r") as f:
    content = f.read()

if "USE_MOCK_DATA = False" in content:
    content = content.replace("USE_MOCK_DATA = False", "USE_MOCK_DATA = True")
    with open(SYSTEM_CONTROL_PATH, "w") as f:
        f.write(content)
    print("✓ Updated system_control.py to use mock data")

# Force mock agents to be stopped by default
print("Updating mock agents to be stopped by default")
with open(SYSTEM_CONTROL_PATH, "r") as f:
    content = f.read()

if '"status": "running"' in content:
    content = content.replace('"status": "running"', '"status": "stopped"')
    with open(SYSTEM_CONTROL_PATH, "w") as f:
        f.write(content)
    print("✓ Updated mock agents to be stopped by default")

# Add debug logging in the start agent endpoint
# We're going to add a few debug prints to help trace the issue
print("Adding debug logging to start_agent endpoint")
with open(SYSTEM_CONTROL_PATH, "r") as f:
    content = f.readlines()

# Find the start_agent function
updated_content = []
in_start_agent = False
for line in content:
    updated_content.append(line)
    
    if "@router.post(\"/agents/{agent_id}/start\", summary=\"Start a specific agent (session)\")" in line:
        in_start_agent = True
    
    if in_start_agent and "if USE_MOCK_DATA:" in line:
        # Add debug prints inside the function right after the if statement
        indent = line.split("if")[0]  # Get the indentation
        debug_line = f"{indent}    # DEBUG LOGGING\n"
        debug_line += f"{indent}    print(f\"DEBUG: Starting agent {{agent_id}}\")\n"
        debug_line += f"{indent}    sys.stdout.flush()  # Ensure output is displayed immediately\n\n"
        
        # Insert the debug lines after the if statement
        updated_content.append(debug_line)
        
        # We only want to add this once, so set flag to False
        in_start_agent = False

with open(SYSTEM_CONTROL_PATH, "w") as f:
    f.writelines(updated_content)
print("✓ Added debug logging to start_agent endpoint")

# Create a .env file to ensure environment is set correctly
env_path = PROJECT_ROOT / ".env"
with open(env_path, "w") as f:
    f.write("USE_MOCK_DATA=true\n")
    f.write("PYTHONPATH=.\n")
print("✓ Created .env file with USE_MOCK_DATA=true")

print("\n" + "="*60)
print("DEBUGGING FIX APPLIED")
print("="*60)
print("Please restart your API server with:")
print("   python -m ai_trading_agent.api.main")
print()
print("Then refresh your browser and try starting an agent")
print("Check the console output for DEBUG messages")
print("="*60)
