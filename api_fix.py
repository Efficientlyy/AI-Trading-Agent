"""
Fix the AI Trading Agent API server connectivity issue
Uses only ASCII characters to avoid encoding problems
"""
import os
import sys
from pathlib import Path

# Get project paths
PROJECT_ROOT = Path(__file__).resolve().parent
API_DIR = PROJECT_ROOT / "ai_trading_agent" / "api"

print("==== FIXING API SERVER CONNECTIVITY ====")

# Fix 1: Make main.py directly executable
main_py_path = API_DIR / "main.py"
with open(main_py_path, "r") as f:
    main_content = f.read()

# Add direct execution block if not present
if "__name__ == \"__main__\"" not in main_content:
    print("[+] Adding direct execution capability to main.py")
    
    # Add proper main block at the end of the file
    main_execution_block = """

# Direct execution block for standalone API server
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        app, 
        host=host,
        port=port,
        log_level="info"
    )
"""
    
    with open(main_py_path, "a") as f:
        f.write(main_execution_block)
    print("[+] Added direct execution capability to main.py")

# Fix 2: Create the optimized API starter
run_api_path = PROJECT_ROOT / "run_optimized_api.py"
run_api_content = """
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
"""

with open(run_api_path, "w") as f:
    f.write(run_api_content)
print("[+] Created optimized API starter (run_optimized_api.py)")

print("\n==== INSTRUCTIONS ====")
print("To fix your API server connectivity:")
print("1. Run: python run_optimized_api.py")
print("2. Start your frontend separately")
print("\nThis runs the API directly with proper network binding")
print("and should resolve the connectivity issues.")
