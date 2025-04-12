"""
Database configuration for the AI Trading Agent.
"""

import os
import logging
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get database URL from environment variables or use SQLite as fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./ai_trading_agent.db"
)

# Database configuration
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # 30 minutes
DB_ECHO = os.getenv("SQL_ECHO", "false").lower() == "true"

# Create SQLAlchemy engine with connection pooling
engine_args = {
    "echo": DB_ECHO,
    "pool_pre_ping": True,
}

# Add connection pooling for PostgreSQL
if DATABASE_URL.startswith("postgresql"):
    engine_args.update({
        "pool_size": DB_POOL_SIZE,
        "max_overflow": DB_MAX_OVERFLOW,
        "pool_timeout": DB_POOL_TIMEOUT,
        "pool_recycle": DB_POOL_RECYCLE,
    })

# Create engine
engine = create_engine(DATABASE_URL, **engine_args)

# Add event listeners for connection debugging
if DB_ECHO:
    @event.listens_for(Engine, "connect")
    def connect(dbapi_connection, connection_record):
        logger.debug("Database connection established")

    @event.listens_for(Engine, "checkout")
    def checkout(dbapi_connection, connection_record, connection_proxy):
        logger.debug("Database connection checked out")

    @event.listens_for(Engine, "checkin")
    def checkin(dbapi_connection, connection_record):
        logger.debug("Database connection checked in")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db():
    """
    Get database session.
    
    Yields:
        Session: SQLAlchemy session
    """
    db = SessionLocal()
    try:
        logger.debug("Database session created")
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        logger.debug("Database session closed")
        db.close()

def init_db():
    """
    Initialize the database by creating all tables.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        return False
