"""
Fix Startup Issues

This script addresses the startup issues with the AI Trading Agent system,
especially the import errors and data feed connection issues.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("startup-fixer")

# Project directory
project_dir = Path(__file__).resolve().parent
logger.info(f"Project directory: {project_dir}")

def check_python_path():
    """Check if the project directory is in Python path and add it if needed."""
    logger.info("Checking Python path...")
    
    # Check if project directory is in sys.path
    if str(project_dir) not in sys.path:
        logger.info(f"Adding {project_dir} to Python path")
        sys.path.insert(0, str(project_dir))
        
    # Create .env file with PYTHONPATH if it doesn't exist
    env_file = project_dir / ".env"
    if not env_file.exists():
        logger.info("Creating .env file with PYTHONPATH")
        with open(env_file, "w") as f:
            f.write(f"PYTHONPATH={project_dir}\n")
            f.write("USE_MOCK_DATA=false\n")
    else:
        logger.info(".env file already exists")
        
        # Check if PYTHONPATH is in .env file
        with open(env_file, "r") as f:
            content = f.read()
        if "PYTHONPATH" not in content:
            logger.info("Adding PYTHONPATH to .env file")
            with open(env_file, "a") as f:
                f.write(f"PYTHONPATH={project_dir}\n")
                
    # Create Python Path batch file for Windows
    bat_file = project_dir / "set_python_path.bat"
    logger.info(f"Creating batch file at {bat_file}")
    with open(bat_file, "w") as f:
        f.write(f"@echo off\n")
        f.write(f"set PYTHONPATH={project_dir}\n")
        f.write(f"echo Python Path set to %PYTHONPATH%\n")
        f.write(f"python -c \"import sys; print('Current Python path:', sys.path)\"\n")
        
    logger.info("Python path setup completed")
    
def fix_data_feed_manager():
    """Fix data feed manager initialization and connection."""
    logger.info("Fixing data feed manager...")
    
    # Path to data_feed_manager.py
    data_feed_path = project_dir / "ai_trading_agent" / "api" / "data_feed_manager.py"
    
    if not data_feed_path.exists():
        logger.error(f"Data feed manager not found at {data_feed_path}")
        return False
    
    # Check if the file has proper initializers
    with open(data_feed_path, "r") as f:
        content = f.read()
    
    # Make sure the data feed manager is properly initialized
    if "data_feed_manager = DataFeedManager()" not in content:
        logger.info("Adding data feed manager initialization")
        
        # Find the class definition
        if "class DataFeedManager:" in content:
            modified_content = content
            
            # Add initialization at the end if not present
            if "data_feed_manager = DataFeedManager()" not in content:
                modified_content += "\n\n# Create singleton instance\ndata_feed_manager = DataFeedManager()\n"
                
            # Write modified content
            with open(data_feed_path, "w") as f:
                f.write(modified_content)
                
            logger.info("Data feed manager initialization added")
        else:
            logger.error("DataFeedManager class not found in the file")
            return False
    
    logger.info("Data feed manager setup completed")
    return True

def fix_main_py():
    """Fix main.py startup logic."""
    logger.info("Fixing main.py startup logic...")
    
    # Path to main.py
    main_py_path = project_dir / "ai_trading_agent" / "api" / "main.py"
    
    if not main_py_path.exists():
        logger.error(f"main.py not found at {main_py_path}")
        return False
    
    with open(main_py_path, "r") as f:
        content = f.read()
    
    # Fix imports by adding parent directory to path
    if "import sys\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))" not in content:
        logger.info("Adding fix for Python imports in main.py")
        
        # Find import section
        import_section = content.split("\n\n")[0]
        
        # Add path fix after imports
        fix_code = "\n# Fix Python path for imports\nimport sys\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))\n"
        
        # Insert after imports
        modified_content = import_section + fix_code + "\n" + content[len(import_section):]
        
        # Write modified content
        with open(main_py_path, "w") as f:
            f.write(modified_content)
            
        logger.info("Python path fix added to main.py")
    
    # Fix startup event to initialize data feed manager
    if "@app.on_event(\"startup\")" in content:
        logger.info("Updating startup event handler in main.py")
        
        # Replace the existing startup event with a more robust one
        new_startup_handler = '''
@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Trading Agent API...")
    
    # Start data feed manager if available
    try:
        from ai_trading_agent.api.data_feed_manager import data_feed_manager
        logger.info("Initializing data feed manager...")
        data_feed_manager.start()
        logger.info("Data feed manager started successfully")
    except ImportError:
        logger.warning("Data Feed Manager not available, data feed connection may not work properly")
    except Exception as e:
        logger.error(f"Error starting data feed manager: {e}")
    
    # Log registered routes
    route_list = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            route_list.append(f"{route.path} - {route.methods}")
    logger.info("Registered routes:\\n" + "\\n".join(route_list))
'''
        
        # Find and replace the existing startup event handler
        import re
        pattern = r'@app\.on_event\("startup"\).*?\n    logger\.info\("Registered routes.*?\))'
        
        # Use regex with DOTALL to match across multiple lines
        modified_content = re.sub(pattern, new_startup_handler.strip(), content, flags=re.DOTALL)
        
        # Write modified content
        with open(main_py_path, "w") as f:
            f.write(modified_content)
            
        logger.info("Startup event handler updated in main.py")
    
    logger.info("main.py updates completed")
    return True

def restart_server():
    """Restart the API server with fixed configuration."""
    logger.info("Restarting API server...")
    
    # Kill any existing uvicorn processes
    try:
        if os.name == 'nt':  # Windows
            subprocess.run("taskkill /f /im uvicorn.exe", shell=True)
        else:  # Unix/Linux
            subprocess.run("pkill -f uvicorn", shell=True)
        logger.info("Killed existing uvicorn processes")
    except Exception as e:
        logger.warning(f"No existing uvicorn processes to kill or error: {e}")
    
    # Start the server with the new environment
    try:
        # Set PYTHONPATH in environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_dir)
        env["USE_MOCK_DATA"] = "false"
        
        # Start main.py
        cmd = [sys.executable, str(project_dir / "ai_trading_agent" / "api" / "main.py")]
        
        logger.info(f"Starting server with command: {' '.join(cmd)}")
        subprocess.Popen(cmd, env=env)
        
        logger.info("Server started successfully")
        return True
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False

def main():
    """Main entry point for startup fixes."""
    print("\n" + "="*60)
    print(" AI TRADING AGENT STARTUP FIX UTILITY ".center(60, "="))
    print("="*60 + "\n")
    
    success = True
    
    # Step 1: Fix Python path
    check_python_path()
    
    # Step 2: Fix data feed manager
    if not fix_data_feed_manager():
        success = False
    
    # Step 3: Fix main.py
    if not fix_main_py():
        success = False
        
    # Step 4: Restart server if all fixes were successful
    if success:
        restart_server()
        
        print("\n" + "="*60)
        print(" ✅ STARTUP FIXES APPLIED SUCCESSFULLY ".center(60, "="))
        print("\nThe API server has been restarted with the fixed configuration.")
        print("Please refresh your browser to see the updated dashboard.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(" ❌ SOME STARTUP FIXES FAILED ".center(60, "="))
        print("\nPlease check the logs above for details.")
        print("="*60)

if __name__ == "__main__":
    main()
