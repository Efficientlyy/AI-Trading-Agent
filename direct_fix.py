"""
Direct fix for the AI Trading Agent backend startup issue.
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

print("=== DIRECT FIX FOR AI TRADING AGENT ===")

# 1. Ensure the system_control.py uses mock data
print("1. Setting USE_MOCK_DATA = True in system_control.py")
system_control_path = PROJECT_ROOT / "ai_trading_agent" / "api" / "system_control.py"
with open(system_control_path, "r") as f:
    content = f.read()

if "USE_MOCK_DATA = False" in content:
    content = content.replace("USE_MOCK_DATA = False", "USE_MOCK_DATA = True")
    with open(system_control_path, "w") as f:
        f.write(content)
    print("   ✓ Updated system_control.py")

# 2. Create direct startup script for backend only
print("2. Creating direct API starter script")
api_starter_path = PROJECT_ROOT / "run_api.py"
api_starter_content = """
import os
import sys
import uvicorn

# Ensure the correct Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Force mock data
os.environ["USE_MOCK_DATA"] = "true"

if __name__ == "__main__":
    print("Starting API server on http://localhost:8000")
    print("Press Ctrl+C to stop")
    uvicorn.run(
        "ai_trading_agent.api.main:app",
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        reload=False,
        log_level="info"
    )
"""
with open(api_starter_path, "w") as f:
    f.write(api_starter_content.strip())
print("   ✓ Created run_api.py")

# 3. Run it directly to test
print("\n3. Starting API server directly (press Ctrl+C to stop)...")
print("="*50)

try:
    subprocess.run([sys.executable, "run_api.py"], cwd=PROJECT_ROOT)
except KeyboardInterrupt:
    print("\nAPI server stopped by user")
