"""
Database initialization module for the AI Trading Agent.
"""

import logging
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from .config import Base, engine, SessionLocal
from .models import User

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Logger
logger = logging.getLogger(__name__)


def init_db():
    """
    Initialize the database by creating all tables.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Create admin user if it doesn't exist
        create_admin_user()
        
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def create_admin_user():
    """
    Create admin user if it doesn't exist.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db = SessionLocal()
        
        # Check if admin user exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        
        if not admin_user:
            # Create admin user
            admin_user = User(
                username="admin",
                email="admin@example.com",
                hashed_password=pwd_context.hash("admin"),  # Change this in production
                is_active=True,
                is_superuser=True
            )
            
            db.add(admin_user)
            db.commit()
            logger.info("Admin user created successfully")
        
        db.close()
        return True
    except Exception as e:
        logger.error(f"Error creating admin user: {e}")
        return False


def get_user_by_username(username: str) -> User:
    """
    Get user by username.
    
    Args:
        username: Username to search for
        
    Returns:
        User: User object if found, None otherwise
    """
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()
    return user


def create_user(username: str, email: str, password: str, is_superuser: bool = False) -> User:
    """
    Create a new user.
    
    Args:
        username: Username
        email: Email address
        password: Plain text password
        is_superuser: Whether the user is a superuser
        
    Returns:
        User: Created user object
    """
    db = SessionLocal()
    
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first()
    
    if existing_user:
        db.close()
        raise ValueError("Username or email already exists")
    
    # Create new user
    user = User(
        username=username,
        email=email,
        hashed_password=pwd_context.hash(password),
        is_active=True,
        is_superuser=is_superuser
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    
    return user


if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database
    init_db()
