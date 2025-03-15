"""Script to run examples with the correct Python path."""

import os
import sys
import importlib.util
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import the example module
def run_example(example_path):
    """Run an example module."""
    example_path = Path(example_path)
    spec = importlib.util.spec_from_file_location(
        example_path.stem, example_path
    )
    example_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_module)
    
    # Run the main function if it exists
    if hasattr(example_module, "main"):
        example_module.main()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_example.py <example_path>")
        sys.exit(1)
        
    example_path = sys.argv[1]
    run_example(example_path)
