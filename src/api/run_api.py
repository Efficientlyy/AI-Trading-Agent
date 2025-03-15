#!/usr/bin/env python
"""Script to run the Market Regime Detection API."""

import os
import sys
import logging
import uvicorn
import traceback

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from api import config
    
    # Configure logging
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    def main():
        """Run the API server."""
        logger.info(f"Starting {config.API_TITLE} v{config.API_VERSION}")
        logger.info(f"Server running at http://{config.API_HOST}:{config.API_PORT}")
        
        try:
            uvicorn.run(
                "api.main:app",
                host=config.API_HOST,
                port=config.API_PORT,
                reload=True
            )
        except Exception as e:
            logger.error(f"Error running API server: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing modules: {str(e)}")
    print("Make sure you're running this script from the project root directory.")
    print("Try: python -m src.api.run_api")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    print(traceback.format_exc())
    sys.exit(1) 