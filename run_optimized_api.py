
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variables
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
os.environ["HOST"] = "0.0.0.0"  # Essential - binds to all network interfaces
os.environ["PORT"] = "8000"

print("===== STARTING OPTIMIZED API SERVER =====")
print("Starting API server with proper network binding...")

# Direct execution of the main module
cmd = [
    sys.executable,
    str(PROJECT_ROOT / "ai_trading_agent" / "api" / "main.py")
]

print(f"API will be available at: http://localhost:8000")
print(f"API documentation: http://localhost:8000/docs")
print("Press Ctrl+C to stop")
print("=" * 50)

# Run the command
subprocess.run(cmd, cwd=PROJECT_ROOT)
