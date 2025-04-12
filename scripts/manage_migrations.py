"""
Manage database migrations using Alembic.
"""

import os
import sys
import argparse
import logging
from alembic.config import Config
from alembic import command

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get the absolute path to the alembic.ini file
alembic_ini = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alembic.ini')

def create_migration(message):
    """Create a new migration."""
    logger.info(f"Creating migration: {message}")
    
    # Create an Alembic configuration object
    alembic_cfg = Config(alembic_ini)
    
    # Create a new migration
    try:
        command.revision(alembic_cfg, message=message, autogenerate=True)
        logger.info("Migration created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating migration: {e}")
        return False

def upgrade_database(revision='head'):
    """Upgrade the database to the specified revision."""
    logger.info(f"Upgrading database to revision: {revision}")
    
    # Create an Alembic configuration object
    alembic_cfg = Config(alembic_ini)
    
    # Upgrade the database
    try:
        command.upgrade(alembic_cfg, revision)
        logger.info("Database upgraded successfully")
        return True
    except Exception as e:
        logger.error(f"Error upgrading database: {e}")
        return False

def downgrade_database(revision):
    """Downgrade the database to the specified revision."""
    logger.info(f"Downgrading database to revision: {revision}")
    
    # Create an Alembic configuration object
    alembic_cfg = Config(alembic_ini)
    
    # Downgrade the database
    try:
        command.downgrade(alembic_cfg, revision)
        logger.info("Database downgraded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downgrading database: {e}")
        return False

def show_history():
    """Show migration history."""
    logger.info("Showing migration history")
    
    # Create an Alembic configuration object
    alembic_cfg = Config(alembic_ini)
    
    # Show migration history
    try:
        command.history(alembic_cfg)
        return True
    except Exception as e:
        logger.error(f"Error showing migration history: {e}")
        return False

def stamp_database(revision='head'):
    """Stamp the database with the specified revision without running migrations."""
    logger.info(f"Stamping database with revision: {revision}")
    
    # Create an Alembic configuration object
    alembic_cfg = Config(alembic_ini)
    
    # Stamp the database
    try:
        command.stamp(alembic_cfg, revision)
        logger.info("Database stamped successfully")
        return True
    except Exception as e:
        logger.error(f"Error stamping database: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Manage database migrations')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create migration
    create_parser = subparsers.add_parser('create', help='Create a new migration')
    create_parser.add_argument('message', help='Migration message')
    
    # Upgrade database
    upgrade_parser = subparsers.add_parser('upgrade', help='Upgrade the database')
    upgrade_parser.add_argument('--revision', default='head', help='Revision to upgrade to (default: head)')
    
    # Downgrade database
    downgrade_parser = subparsers.add_parser('downgrade', help='Downgrade the database')
    downgrade_parser.add_argument('revision', help='Revision to downgrade to')
    
    # Show migration history
    subparsers.add_parser('history', help='Show migration history')
    
    # Stamp database
    stamp_parser = subparsers.add_parser('stamp', help='Stamp the database with a revision')
    stamp_parser.add_argument('--revision', default='head', help='Revision to stamp with (default: head)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if args.command == 'create':
        create_migration(args.message)
    elif args.command == 'upgrade':
        upgrade_database(args.revision)
    elif args.command == 'downgrade':
        downgrade_database(args.revision)
    elif args.command == 'history':
        show_history()
    elif args.command == 'stamp':
        stamp_database(args.revision)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
