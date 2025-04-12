"""
Initialize the database by creating all tables.
"""

import os
import sys
import logging

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.config import init_db, engine, Base
from backend.database.models import user, strategy, backtest, market_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Initialize the database."""
    logger.info("Initializing database...")
    
    # Create all tables
    if init_db():
        logger.info("Database initialized successfully.")
    else:
        logger.error("Failed to initialize database.")
        return False
    
    logger.info("Database setup complete.")
    return True

if __name__ == "__main__":
    main()
