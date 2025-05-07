"""
Build script for Rust extensions.

This script builds the Rust extensions and installs them in the Python package.
It uses maturin, which is the recommended way to build PyO3 extensions.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


def run_command(command, cwd=None):
    """Run a command and return its output."""
    print(f"Running command: {command}")
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    # Check if Rust is installed
    if not run_command("rustc --version"):
        print("Rust is not installed. Please install Rust from https://rustup.rs/")
        return False
    
    # Check if maturin is installed
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "maturin"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    version = line.split(":")[1].strip()
                    print(f"Found maturin version {version}")
                    break
        else:
            raise ImportError("maturin not found")
    except (ImportError, subprocess.SubprocessError):
        print("maturin is not installed. Installing...")
        if not run_command(f"{sys.executable} -m pip install maturin"):
            print("Failed to install maturin")
            return False
    
    return True


def build_extensions(release=True, debug=False):
    """Build the Rust extensions."""
    rust_extensions_dir = Path(__file__).parent / "rust_extensions"
    
    # Build command
    build_type = "--release" if release else ""
    if debug:
        build_type = ""  # Debug build
    
    # Use maturin to build the extension
    if platform.system() == "Windows":
        command = f"{sys.executable} -m maturin develop {build_type}"
    else:
        command = f"maturin develop {build_type}"
    
    if not run_command(command, cwd=str(rust_extensions_dir)):
        print("Failed to build Rust extensions")
        return False
    
    print("Successfully built Rust extensions")
    return True


def test_extensions():
    """Test if the Rust extensions are working."""
    try:
        # Try to import the extension
        from ai_trading_agent_rs import create_lag_features_rs
        print("Successfully imported Rust extensions")
        
        # Test with a simple example
        result = create_lag_features_rs([1.0, 2.0, 3.0, 4.0, 5.0], [1, 2])
        print(f"Test result: {result}")
        
        return True
    except ImportError as e:
        print(f"Failed to import Rust extensions: {e}")
        return False
    except Exception as e:
        print(f"Error testing Rust extensions: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build Rust extensions")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    parser.add_argument("--no-release", action="store_true", help="Don't build in release mode")
    parser.add_argument("--test", action="store_true", help="Test the extensions after building")
    args = parser.parse_args()
    
    if not check_dependencies():
        sys.exit(1)
    
    release = not args.no_release
    debug = args.debug
    
    if not build_extensions(release=release, debug=debug):
        sys.exit(1)
    
    if args.test:
        if not test_extensions():
            sys.exit(1)
    
    print("Build completed successfully")


if __name__ == "__main__":
    main()
