import sys
import os

# Compute absolute path to src directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"[conftest] Added to sys.path: {src_path}")
