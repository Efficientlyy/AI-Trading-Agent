#!/usr/bin/env python
"""
Debug script to fix API key validation in OptimizedMEXCClient
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_api_key")

def fix_optimized_mexc_client():
    """Fix API key validation in OptimizedMEXCClient"""
    
    # Path to the optimized_mexc_client.py file
    client_path = "optimized_mexc_client.py"
    
    # Check if file exists
    if not os.path.exists(client_path):
        logger.error(f"Client file not found: {client_path}")
        return False
    
    # Read the file
    try:
        with open(client_path, 'r') as f:
            content = f.read()
            logger.info(f"Successfully read client file. Content length: {len(content)}")
    except Exception as e:
        logger.error(f"Error reading client file: {str(e)}")
        return False
    
    # Fix the _init_session method
    if "_init_session" in content:
        # Find the _init_session method
        start_idx = content.find("def _init_session")
        if start_idx == -1:
            logger.error("Could not find _init_session method")
            return False
        
        # Find the end of the method
        end_idx = content.find("def ", start_idx + 1)
        if end_idx == -1:
            end_idx = len(content)
        
        # Extract the method
        method = content[start_idx:end_idx]
        logger.info(f"Found _init_session method. Length: {len(method)}")
        
        # Check if the method contains the issue
        if "API key is not a valid string" in method:
            logger.info("Found API key validation issue in _init_session")
            
            # Create fixed method
            fixed_method = """    def _init_session(self):
        \"\"\"Initialize HTTP session with connection pooling\"\"\"
        if self.session is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=0.1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST", "DELETE", "PUT"]
            )
            
            # Create session with connection pooling
            self.session = requests.Session()
            adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            # Set default headers
            # Ensure API key is a string
            api_key_str = str(self.api_key) if self.api_key is not None else ""
            
            self.session.headers.update({
                'User-Agent': 'FlashTradingBot/1.0',
                'Accept': 'application/json',
                'X-MEXC-APIKEY': api_key_str
            })
    """
            
            # Replace the method
            new_content = content[:start_idx] + fixed_method + content[end_idx:]
            
            # Write the fixed file
            try:
                with open(client_path, 'w') as f:
                    f.write(new_content)
                logger.info("Successfully fixed _init_session method")
            except Exception as e:
                logger.error(f"Error writing fixed client file: {str(e)}")
                return False
        else:
            logger.info("_init_session method does not contain the API key validation issue")
    
    # Fix the _init_async_session method
    if "_init_async_session" in content:
        # Find the _init_async_session method
        start_idx = content.find("def _init_async_session")
        if start_idx == -1:
            logger.error("Could not find _init_async_session method")
            return False
        
        # Find the end of the method
        end_idx = content.find("def ", start_idx + 1)
        if end_idx == -1:
            end_idx = len(content)
        
        # Extract the method
        method = content[start_idx:end_idx]
        logger.info(f"Found _init_async_session method. Length: {len(method)}")
        
        # Check if the method contains the issue
        if "API key is not a valid string" in method:
            logger.info("Found API key validation issue in _init_async_session")
            
            # Create fixed method
            fixed_method = """    async def _init_async_session(self):
        \"\"\"Initialize async HTTP session with connection pooling\"\"\"
        if self.async_session is None:
            timeout = aiohttp.ClientTimeout(total=5)
            
            # Ensure API key is a string
            api_key_str = str(self.api_key) if self.api_key is not None else ""
                
            self.async_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'FlashTradingBot/1.0',
                    'Accept': 'application/json',
                    'X-MEXC-APIKEY': api_key_str
                }
            )
    """
            
            # Replace the method
            new_content = content[:start_idx] + fixed_method + content[end_idx:]
            
            # Write the fixed file
            try:
                with open(client_path, 'w') as f:
                    f.write(new_content)
                logger.info("Successfully fixed _init_async_session method")
            except Exception as e:
                logger.error(f"Error writing fixed client file: {str(e)}")
                return False
        else:
            logger.info("_init_async_session method does not contain the API key validation issue")
    
    return True

if __name__ == "__main__":
    success = fix_optimized_mexc_client()
    
    if success:
        logger.info("Successfully fixed API key validation in OptimizedMEXCClient")
        sys.exit(0)
    else:
        logger.error("Failed to fix API key validation in OptimizedMEXCClient")
        sys.exit(1)
