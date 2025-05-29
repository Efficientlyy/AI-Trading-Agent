"""
Fix for AI Trading Agent API server connectivity issues
"""
import os
import sys
import subprocess
from pathlib import Path

# Get project paths
PROJECT_ROOT = Path(__file__).resolve().parent
API_DIR = PROJECT_ROOT / "ai_trading_agent" / "api"

print("==== FIXING API SERVER CONNECTIVITY ====")

# Check if the main.py file defines the FastAPI app correctly
main_py_path = API_DIR / "main.py"
with open(main_py_path, "r") as f:
    main_content = f.read()

# Fix 1: Make sure the app is properly initialized and exported
if "app = FastAPI" in main_content:
    print("✓ FastAPI app is properly initialized")
else:
    print("✗ FastAPI app initialization not found - this might be the issue")
    print("  Please check main.py manually")

# Fix 2: Create a proper if __name__ == "__main__" block in main.py 
# This ensures the app can be run directly
if "__name__ == \"__main__\"" not in main_content:
    print("- Adding direct execution capability to main.py")
    
    # Add the proper main block at the end of the file
    main_execution_block = """

# Enable direct execution of this file
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info")
    
    print(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "ai_trading_agent.api.main:app", 
        host=host,
        port=port,
        log_level=log_level
    )
"""
    
    with open(main_py_path, "a") as f:
        f.write(main_execution_block)
    print("✓ Added direct execution capability to main.py")

# Fix 3: Fix the start_servers.py script
start_servers_path = PROJECT_ROOT / "start_servers.py"
with open(start_servers_path, "r") as f:
    start_servers_content = f.read()

# Update the host binding to ensure proper network binding
if "\"--host\", BACKEND_HOST," in start_servers_content:
    updated_content = start_servers_content.replace(
        "\"--host\", BACKEND_HOST,",
        "\"--host\", \"0.0.0.0\",  # Bind to all interfaces"
    )
    
    with open(start_servers_path, "w") as f:
        f.write(updated_content)
    print("✓ Updated API server host binding to ensure proper network access")

# Fix 4: Create a specialized API testing script
test_api_path = PROJECT_ROOT / "test_api_connection.py"
test_api_content = """
import sys
import requests
import time

def test_api_connection():
    \"\"\"Test if the API server is accepting connections\"\"\"
    url = "http://127.0.0.1:8000/health"
    max_attempts = 10
    
    print(f"Testing API connection to {url}")
    print("Attempting to connect (this may take a few seconds)...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✓ Successfully connected to API server!")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"Attempt {attempt+1}/{max_attempts}: Received status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}/{max_attempts}: Connection failed - {e.__class__.__name__}")
        
        # Wait before retrying
        time.sleep(1)
    
    print("❌ Failed to connect to API server after multiple attempts")
    print("Possible issues:")
    print("1. API server is not running")
    print("2. API server is running but not binding to the network interface")
    print("3. Firewall is blocking the connection")
    return False

if __name__ == "__main__":
    test_api_connection()
"""
with open(test_api_path, "w") as f:
    f.write(test_api_content)
print("✓ Created API connection test script")

# Fix 5: Create a specialized startup script for the API server
fixed_api_starter_path = PROJECT_ROOT / "run_fixed_api.py"
fixed_api_starter_content = """
import os
import sys
import subprocess
from pathlib import Path

# Ensure the correct Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("==== RUNNING FIXED API SERVER ====")

# Set environment variables
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
os.environ["HOST"] = "0.0.0.0"  # Listen on all interfaces
os.environ["PORT"] = "8000"
os.environ["LOG_LEVEL"] = "info"

# Run the API server directly from the main module
cmd = [
    sys.executable,
    "-m", "ai_trading_agent.api.main"
]

# Print startup information
print(f"Starting API server on {os.environ['HOST']}:{os.environ['PORT']}")
print("API documentation will be available at http://localhost:8000/docs")
print("Press Ctrl+C to stop the server")
print("="*50)

# Execute the command
subprocess.run(cmd, cwd=PROJECT_ROOT)
"""
with open(fixed_api_starter_path, "w") as f:
    f.write(fixed_api_starter_content)
print("✓ Created fixed API starter script")

print("\n==== INSTRUCTIONS ====")
print("To run the fixed API server:")
print("1. Run: python run_fixed_api.py")
print("2. To test the API connection: python test_api_connection.py")
print("3. Once the API is working, restart your frontend")
print("\nIf you still experience issues, check the console output")
print("for detailed error messages about what's failing.")
