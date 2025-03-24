"""Simple Flask check"""
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import flask
    print(f"Flask imported successfully from {flask.__file__}")
    print(f"Flask version: {flask.__version__}")
except Exception as e:
    print(f"Error importing Flask: {e}")
