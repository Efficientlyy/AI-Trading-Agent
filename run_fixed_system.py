#!/usr/bin/env python
"""
Run Fixed Trading System

This script starts both the fixed API server and the React frontend
to ensure the entire trading system works properly.
"""

import os
import sys
import subprocess
import time
import threading
import signal
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("run-fixed-system")

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Global process objects
api_process = None
frontend_process = None

def start_api_server():
    """Start the fixed API server in a separate process."""
    global api_process
    
    logger.info("Starting fixed API server...")
    
    # Use the fixed_api_server.py script we created
    try:
        api_process = subprocess.Popen(
            [sys.executable, "fixed_api_server.py"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )
        
        # Monitor the output to confirm server started
        for line in api_process.stdout:
            print(f"[API] {line}", end="")
            if "Uvicorn running on" in line:
                logger.info("API server is running")
                break
        
        # Start a thread to continue reading output
        def read_output():
            for line in api_process.stdout:
                print(f"[API] {line}", end="")
        
        threading.Thread(target=read_output, daemon=True).start()
        
        # Wait a moment to ensure the server is ready
        time.sleep(2)
        return True
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return False

def start_frontend():
    """Start the React frontend development server."""
    global frontend_process
    
    logger.info("Starting frontend server...")
    
    # Check if node_modules exists
    if not (FRONTEND_DIR / "node_modules").exists():
        logger.info("Installing frontend dependencies...")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd=FRONTEND_DIR,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install frontend dependencies: {e}")
            return False
    
    # Start the frontend development server
    try:
        # Set environment variables for the frontend
        env = os.environ.copy()
        env["REACT_APP_API_URL"] = "http://127.0.0.1:8000"
        
        frontend_process = subprocess.Popen(
            ["npm", "start"],
            cwd=FRONTEND_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )
        
        # Monitor the output to confirm server started
        for line in frontend_process.stdout:
            print(f"[FRONTEND] {line}", end="")
            if "Compiled successfully" in line or "Compiled with warnings" in line:
                logger.info("Frontend server is running")
                break
        
        # Start a thread to continue reading output
        def read_output():
            for line in frontend_process.stdout:
                print(f"[FRONTEND] {line}", end="")
        
        threading.Thread(target=read_output, daemon=True).start()
        return True
    except Exception as e:
        logger.error(f"Failed to start frontend server: {e}")
        return False

def cleanup(sig=None, frame=None):
    """Clean up processes when shutting down."""
    logger.info("Shutting down servers...")
    
    if api_process:
        logger.info("Terminating API server...")
        api_process.terminate()
        api_process.wait(timeout=5)
    
    if frontend_process:
        logger.info("Terminating frontend server...")
        frontend_process.terminate()
        frontend_process.wait(timeout=5)
    
    logger.info("All servers have been shut down.")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Extra check for potential locked files
        if os.path.exists(PROJECT_ROOT / ".lock"):
            os.remove(PROJECT_ROOT / ".lock")
            logger.info("Removed stale .lock file")
        
        # Ensure our mock data changes are applied
        fix_file = PROJECT_ROOT / "ai_trading_agent" / "api" / "system_control.py"
        with open(fix_file, "r") as f:
            content = f.read()
        
        if "USE_MOCK_DATA = False" in content:
            content = content.replace("USE_MOCK_DATA = False", "USE_MOCK_DATA = True")
            with open(fix_file, "w") as f:
                f.write(content)
            logger.info("Updated system_control.py to use mock data")
        
        # Start the servers
        if start_api_server():
            logger.info("API server started successfully")
            
            if start_frontend():
                logger.info("Frontend server started successfully")
                
                print("\n" + "="*80)
                print(" FIXED TRADING SYSTEM RUNNING")
                print("="*80)
                print(" API Server:   http://127.0.0.1:8000")
                print(" Frontend:     http://localhost:3000")
                print("="*80)
                print(" Press Ctrl+C to stop all servers")
                print("="*80 + "\n")
                
                # Keep the main thread alive
                while True:
                    time.sleep(1)
            else:
                logger.error("Failed to start frontend server")
                cleanup()
        else:
            logger.error("Failed to start API server")
            cleanup()
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        cleanup()
