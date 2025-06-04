#!/usr/bin/env python
"""
Deep trace script for debugging API key propagation in OptimizedMEXCClient
"""

import os
import sys
import logging
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deep_trace")

def trace_env_path_propagation(env_path):
    """Trace environment path propagation and API key loading"""
    logger.info(f"Starting deep trace with env_path: {env_path}")
    
    # Check if file exists
    if not os.path.exists(env_path):
        logger.error(f"Environment file not found: {env_path}")
        return False
    
    # Read file content
    try:
        with open(env_path, 'r') as f:
            content = f.read()
            logger.info(f"Environment file content: {content}")
    except Exception as e:
        logger.error(f"Error reading environment file: {str(e)}")
        return False
    
    # Check environment before loading
    api_key_before = os.environ.get('MEXC_API_KEY')
    api_secret_before = os.environ.get('MEXC_API_SECRET')
    logger.info(f"Before load_dotenv - MEXC_API_KEY: {api_key_before}")
    logger.info(f"Before load_dotenv - MEXC_API_SECRET: {api_secret_before}")
    
    # Load environment variables
    try:
        load_dotenv(env_path)
        logger.info("load_dotenv called successfully")
    except Exception as e:
        logger.error(f"Error in load_dotenv: {str(e)}")
        return False
    
    # Check environment after loading
    api_key_after = os.environ.get('MEXC_API_KEY')
    api_secret_after = os.environ.get('MEXC_API_SECRET')
    logger.info(f"After load_dotenv - MEXC_API_KEY: {api_key_after}")
    logger.info(f"After load_dotenv - MEXC_API_SECRET: {api_secret_after}")
    
    # Import OptimizedMEXCClient
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from optimized_mexc_client import OptimizedMEXCClient
        logger.info("Successfully imported OptimizedMEXCClient")
    except Exception as e:
        logger.error(f"Error importing OptimizedMEXCClient: {str(e)}")
        return False
    
    # Create client instance with direct credentials
    try:
        direct_client = OptimizedMEXCClient(
            api_key=api_key_after,
            secret_key=api_secret_after
        )
        logger.info("Created client with direct credentials")
        logger.info(f"Direct client API key type: {type(direct_client.api_key)}")
        logger.info(f"Direct client API key: {direct_client.api_key}")
    except Exception as e:
        logger.error(f"Error creating client with direct credentials: {str(e)}")
    
    # Create client instance with env_path
    try:
        env_client = OptimizedMEXCClient(env_path=env_path)
        logger.info("Created client with env_path")
        logger.info(f"Env client API key type: {type(env_client.api_key)}")
        logger.info(f"Env client API key: {env_client.api_key}")
    except Exception as e:
        logger.error(f"Error creating client with env_path: {str(e)}")
    
    # Patch the client class
    try:
        # Monkey patch the _init_session method to log API key
        original_init_session = OptimizedMEXCClient._init_session
        
        def patched_init_session(self):
            logger.info("Patched _init_session called")
            logger.info(f"API key before session init: {self.api_key}")
            logger.info(f"API key type: {type(self.api_key)}")
            result = original_init_session(self)
            logger.info("Original _init_session completed")
            return result
        
        OptimizedMEXCClient._init_session = patched_init_session
        logger.info("Monkey patched _init_session method")
    except Exception as e:
        logger.error(f"Error patching client: {str(e)}")
    
    # Create client instance after patching
    try:
        patched_client = OptimizedMEXCClient(env_path=env_path)
        logger.info("Created client after patching")
        
        # Initialize session
        patched_client._init_session()
        logger.info("Initialized session for patched client")
        
        # Check headers
        if hasattr(patched_client, 'session') and patched_client.session:
            headers = patched_client.session.headers
            logger.info(f"Session headers: {headers}")
            
            api_key_header = headers.get('X-MEXC-APIKEY')
            logger.info(f"X-MEXC-APIKEY header: {api_key_header}")
            logger.info(f"X-MEXC-APIKEY header type: {type(api_key_header)}")
    except Exception as e:
        logger.error(f"Error with patched client: {str(e)}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python deep_trace.py <env_file_path>")
        sys.exit(1)
    
    env_path = sys.argv[1]
    success = trace_env_path_propagation(env_path)
    
    if success:
        logger.info("Deep trace completed successfully")
        sys.exit(0)
    else:
        logger.error("Deep trace failed")
        sys.exit(1)
