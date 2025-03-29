"""
Python-dotenv compatibility layer for Python 3.13

This module provides a compatibility layer for the python-dotenv package,
ensuring that it works correctly with Python 3.13.
"""

import os
import sys
import importlib.util
from pathlib import Path

def find_dotenv(filename='.env', raise_error_if_not_found=False, usecwd=False):
    """
    Search for a .env file in parent directories.
    
    This is a simplified version of the original find_dotenv function.
    """
    # Start with the current working directory or calling script's directory
    if usecwd:
        path = Path(os.getcwd())
    else:
        frame = sys._getframe(1)
        caller_path = Path(frame.f_code.co_filename)
        path = caller_path.parent
    
    # Search for the file
    for path in path.parents:
        dotenv_path = path / filename
        if dotenv_path.exists():
            return str(dotenv_path)
    
    # Handle case where file isn't found
    if raise_error_if_not_found:
        raise ValueError(f"Could not find {filename} file")
    return None

def load_dotenv(dotenv_path=None, stream=None, verbose=False, override=False, interpolate=True, encoding='utf-8'):
    """
    Load environment variables from .env file.
    
    This is a compatibility wrapper around the actual python-dotenv function.
    """
    try:
        # Try to import python-dotenv properly
        try:
            from src.common.dotenv_compat import load_dotenv as original_load_dotenv
            return original_load_dotenv(
                dotenv_path=dotenv_path,
                stream=stream,
                verbose=verbose,
                override=override,
                interpolate=interpolate,
                encoding=encoding
            )
        except ImportError:
            # Try the alternative import path (Python 3.13 may have changed it)
            try:
                from python_dotenv import load_dotenv as original_load_dotenv
                return original_load_dotenv(
                    dotenv_path=dotenv_path,
                    stream=stream,
                    verbose=verbose,
                    override=override,
                    interpolate=interpolate,
                    encoding=encoding
                )
            except ImportError:
                # If all else fails, implement a basic version
                if dotenv_path is None:
                    dotenv_path = find_dotenv()
                
                if dotenv_path is None or not os.path.isfile(dotenv_path):
                    return False
                
                # Read the .env file
                with open(dotenv_path, 'r', encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Split on the first = sign
                        key_value = line.split('=', 1)
                        if len(key_value) == 2:
                            key, value = key_value
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            
                            # Set the environment variable
                            if override or key not in os.environ:
                                os.environ[key] = value
                                if verbose:
                                    print(f"Setting environment variable: {key}")
                
                return True
    except Exception as e:
        if verbose:
            print(f"Error loading .env file: {e}")
        return False

def dotenv_values(dotenv_path=None, stream=None, verbose=False, interpolate=True, encoding='utf-8'):
    """
    Parse a .env file and return a dict of the values.
    
    This is a compatibility wrapper around the actual python-dotenv function.
    """
    try:
        try:
            from dotenv import dotenv_values as original_dotenv_values
            return original_dotenv_values(
                dotenv_path=dotenv_path,
                stream=stream,
                verbose=verbose,
                interpolate=interpolate,
                encoding=encoding
            )
        except ImportError:
            try:
                from python_dotenv import dotenv_values as original_dotenv_values
                return original_dotenv_values(
                    dotenv_path=dotenv_path,
                    stream=stream,
                    verbose=verbose, 
                    interpolate=interpolate,
                    encoding=encoding
                )
            except ImportError:
                # Basic implementation
                result = {}
                
                if dotenv_path is None:
                    dotenv_path = find_dotenv()
                
                if dotenv_path is None or not os.path.isfile(dotenv_path):
                    return result
                
                # Read the .env file
                with open(dotenv_path, 'r', encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Split on the first = sign
                        key_value = line.split('=', 1)
                        if len(key_value) == 2:
                            key, value = key_value
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            result[key] = value
                
                return result
    except Exception as e:
        if verbose:
            print(f"Error parsing .env file: {e}")
        return {}
