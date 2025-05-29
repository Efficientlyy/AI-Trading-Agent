"""
Script to rebuild and reinstall the Rust module for the AI Trading Agent.
This will ensure that the Rust functions are properly exposed to Python.
"""

import os
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RustRebuild")

def check_maturin_installed():
    """Check if maturin is installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "maturin"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking for maturin: {e}")
        return False

def install_maturin():
    """Install maturin if not already installed."""
    logger.info("Installing maturin...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "maturin"],
            check=True
        )
        logger.info("Maturin installed successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to install maturin: {e}")
        return False

def find_rust_extensions_directory():
    """Find the rust_extensions directory."""
    base_dir = os.getcwd()
    rust_dir = os.path.join(base_dir, "rust_extensions")
    
    if os.path.exists(rust_dir) and os.path.isdir(rust_dir):
        logger.info(f"Found rust_extensions directory: {rust_dir}")
        return rust_dir
    
    # Try to find it by walking the directory
    for root, dirs, files in os.walk(base_dir):
        if "rust_extensions" in dirs:
            rust_dir = os.path.join(root, "rust_extensions")
            logger.info(f"Found rust_extensions directory: {rust_dir}")
            return rust_dir
    
    logger.error("Could not find rust_extensions directory")
    return None

def rebuild_rust_module(rust_dir):
    """Rebuild the Rust module using maturin."""
    logger.info(f"Rebuilding Rust module in {rust_dir}...")
    try:
        # Change to the rust_extensions directory
        os.chdir(rust_dir)
        
        # Run maturin develop with release flag
        result = subprocess.run(
            [sys.executable, "-m", "maturin", "develop", "--release"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info("Successfully rebuilt and installed Rust module")
            print(result.stdout)
            return True
        else:
            logger.error(f"Failed to rebuild Rust module: {result.stderr}")
            print(result.stderr)
            return False
    except Exception as e:
        logger.error(f"Error rebuilding Rust module: {e}")
        return False
    finally:
        # Return to the original directory
        os.chdir(os.path.dirname(rust_dir))

def verify_installation():
    """Verify that the Rust module is installed and functions are available."""
    try:
        # Try importing the module
        logger.info("Verifying Rust module installation...")
        
        # Try the new module name first
        try:
            import ai_trading_agent_rs
            module = ai_trading_agent_rs
            module_name = "ai_trading_agent_rs"
        except ImportError:
            # Fall back to the old module name
            import rust_lag_features
            module = rust_lag_features
            module_name = "rust_lag_features"
        
        # Check if the create_lag_features_rs function exists
        if hasattr(module, 'create_lag_features_rs'):
            logger.info(f"Successfully verified that 'create_lag_features_rs' is available in {module_name}")
            return True
        else:
            logger.warning(f"Module {module_name} was imported but 'create_lag_features_rs' function was not found")
            # List available functions
            available_functions = [name for name in dir(module) if not name.startswith('_')]
            logger.info(f"Available functions in {module_name}: {available_functions}")
            return False
    except ImportError as e:
        logger.error(f"Failed to import Rust module: {e}")
        return False
    except Exception as e:
        logger.error(f"Error verifying installation: {e}")
        return False

def main():
    """Main function to rebuild and verify the Rust module."""
    logger.info("Starting Rust module rebuild process")
    
    # Check if maturin is installed
    if not check_maturin_installed():
        if not install_maturin():
            logger.error("Failed to install maturin. Aborting.")
            return False
    
    # Find the rust_extensions directory
    rust_dir = find_rust_extensions_directory()
    if not rust_dir:
        logger.error("Could not find rust_extensions directory. Aborting.")
        return False
    
    # Rebuild the Rust module
    if not rebuild_rust_module(rust_dir):
        logger.error("Failed to rebuild Rust module. Aborting.")
        return False
    
    # Verify the installation
    if verify_installation():
        logger.info("Rust module successfully rebuilt and verified!")
        return True
    else:
        logger.warning("Rust module was rebuilt but verification failed.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Rust module rebuilt and verified successfully!")
    else:
        print("\n❌ Failed to rebuild or verify Rust module. See logs for details.")
