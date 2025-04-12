#!/usr/bin/env python
"""
Database setup script for the AI Trading Agent.

This script initializes the database and migrates data from in-memory storage.
"""

import os
import sys
import logging
import argparse
import subprocess

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up the database")
    
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize the database, don't migrate data"
    )
    
    parser.add_argument(
        "--migrate-only",
        action="store_true",
        help="Only migrate data, don't initialize the database"
    )
    
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Create sample data"
    )
    
    return parser.parse_args()


def run_script(script_path):
    """Run a Python script."""
    try:
        logger.info(f"Running {script_path}...")
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Script output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Script errors: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running script: {e}")
        logger.error(f"Script output: {e.stdout}")
        logger.error(f"Script errors: {e.stderr}")
        return False


def main():
    """Set up the database."""
    args = parse_args()
    
    # Initialize database
    if not args.migrate_only:
        logger.info("Initializing database...")
        init_script = os.path.join(project_root, "scripts", "init_database.py")
        if not run_script(init_script):
            logger.error("Database initialization failed")
            sys.exit(1)
        logger.info("Database initialization completed successfully")
    
    # Migrate data
    if not args.init_only:
        logger.info("Migrating data...")
        migrate_script = os.path.join(project_root, "scripts", "migrate_to_database.py")
        if not run_script(migrate_script):
            logger.error("Data migration failed")
            sys.exit(1)
        logger.info("Data migration completed successfully")
    
    # Create sample data
    if args.sample_data:
        logger.info("Creating sample data...")
        from backend.database import get_db, SessionLocal
        from scripts.init_database import create_sample_data
        
        # Create session
        db = SessionLocal()
        
        try:
            create_sample_data()
            logger.info("Sample data created successfully")
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            sys.exit(1)
        finally:
            db.close()
    
    logger.info("Database setup completed successfully")


if __name__ == "__main__":
    main()
