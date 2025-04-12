"""
User repository for database operations.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from jose import jwt
from passlib.context import CryptContext
import uuid
import os
from dotenv import load_dotenv

from ..models.user import User, UserSession, PasswordReset
from .base import BaseRepository
from ..errors import with_error_handling, RecordNotFoundError

# Load environment variables
load_dotenv()

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRepository(BaseRepository[User, Dict[str, Any], Dict[str, Any]]):
    """User repository for database operations."""
    
    def __init__(self):
        """Initialize the repository with the User model."""
        super().__init__(User)
    
    def get_by_username(self, db: Session, username: str) -> Optional[User]:
        """
        Get a user by username.
        
        Args:
            db: Database session
            username: Username to search for
            
        Returns:
            User if found, None otherwise
        """
        return db.query(User).filter(User.username == username).first()
    
    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        """
        Get a user by email.
        
        Args:
            db: Database session
            email: Email to search for
            
        Returns:
            User if found, None otherwise
        """
        return db.query(User).filter(User.email == email).first()
    
    def create_user(
        self,
        db: Session,
        username: str,
        email: str,
        password: str,
        is_superuser: bool = False
    ) -> User:
        """
        Create a new user.
        
        Args:
            db: Database session
            username: Username
            email: Email address
            password: Plain text password
            is_superuser: Whether the user is a superuser
            
        Returns:
            Created user
        """
        # Check if user already exists
        if self.get_by_username(db, username):
            raise ValueError(f"Username '{username}' already exists")
        
        if self.get_by_email(db, email):
            raise ValueError(f"Email '{email}' already exists")
        
        # Create user
        hashed_password = pwd_context.hash(password)
        user_data = {
            "username": username,
            "email": email,
            "hashed_password": hashed_password,
            "is_active": True,
            "is_superuser": is_superuser
        }
        
        return self.create(db, user_data)
    
    def authenticate(self, db: Session, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user.
        
        Args:
            db: Database session
            username: Username
            password: Plain text password
            
        Returns:
            User if authentication successful, None otherwise
        """
        user = self.get_by_username(db, username)
        
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not pwd_context.verify(password, user.hashed_password):
            return None
        
        return user
    
    @with_error_handling
    def authenticate_user(self, db: Session, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user with username and password.
        
        Args:
            db: Database session
            username: Username
            password: Password
            
        Returns:
            User if authenticated, None otherwise
        """
        # This is an alias for the authenticate method to match API usage
        return self.authenticate(db, username, password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: Data to encode in the token
            expires_delta: Token expiration time
            
        Returns:
            JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        return encoded_jwt
    
    def create_refresh_token(self, db: Session, user_id: int, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> str:
        """
        Create a refresh token and store it in the database.
        
        Args:
            db: Database session
            user_id: User ID
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Refresh token
        """
        # Create token
        refresh_token = os.urandom(32).hex()
        expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        # Create session
        session = UserSession(
            user_id=user_id,
            refresh_token=refresh_token,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        db.add(session)
        db.commit()
        
        return refresh_token
    
    def verify_refresh_token(self, db: Session, refresh_token: str) -> Optional[User]:
        """
        Verify a refresh token and return the associated user.
        
        Args:
            db: Database session
            refresh_token: Refresh token
            
        Returns:
            User if token is valid, None otherwise
        """
        session = db.query(UserSession).filter(
            UserSession.refresh_token == refresh_token,
            UserSession.expires_at > datetime.utcnow()
        ).first()
        
        if not session:
            return None
        
        user = db.query(User).filter(User.id == session.user_id).first()
        
        if not user or not user.is_active:
            return None
        
        return user
    
    def invalidate_refresh_token(self, db: Session, refresh_token: str) -> bool:
        """
        Invalidate a refresh token.
        
        Args:
            db: Database session
            refresh_token: Refresh token
            
        Returns:
            True if token was invalidated, False otherwise
        """
        session = db.query(UserSession).filter(UserSession.refresh_token == refresh_token).first()
        
        if not session:
            return False
        
        db.delete(session)
        db.commit()
        
        return True
    
    def invalidate_all_user_sessions(self, db: Session, user_id: int) -> int:
        """
        Invalidate all sessions for a user.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Number of sessions invalidated
        """
        sessions = db.query(UserSession).filter(UserSession.user_id == user_id).all()
        count = len(sessions)
        
        for session in sessions:
            db.delete(session)
        
        db.commit()
        
        return count
    
    def create_password_reset_token(self, db: Session, user_id: int) -> str:
        """
        Create a password reset token.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Password reset token
        """
        # Create token
        reset_token = os.urandom(32).hex()
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        # Create reset request
        reset_request = PasswordReset(
            user_id=user_id,
            reset_token=reset_token,
            expires_at=expires_at,
            is_used=False
        )
        
        db.add(reset_request)
        db.commit()
        
        return reset_token
    
    def verify_password_reset_token(self, db: Session, reset_token: str) -> Optional[User]:
        """
        Verify a password reset token and return the associated user.
        
        Args:
            db: Database session
            reset_token: Password reset token
            
        Returns:
            User if token is valid, None otherwise
        """
        reset_request = db.query(PasswordReset).filter(
            PasswordReset.reset_token == reset_token,
            PasswordReset.expires_at > datetime.utcnow(),
            PasswordReset.is_used == False
        ).first()
        
        if not reset_request:
            return None
        
        user = db.query(User).filter(User.id == reset_request.user_id).first()
        
        if not user or not user.is_active:
            return None
        
        return user
    
    def reset_password(self, db: Session, reset_token: str, new_password: str) -> bool:
        """
        Reset a user's password using a reset token.
        
        Args:
            db: Database session
            reset_token: Password reset token
            new_password: New password
            
        Returns:
            True if password was reset, False otherwise
        """
        reset_request = db.query(PasswordReset).filter(
            PasswordReset.reset_token == reset_token,
            PasswordReset.expires_at > datetime.utcnow(),
            PasswordReset.is_used == False
        ).first()
        
        if not reset_request:
            return False
        
        user = db.query(User).filter(User.id == reset_request.user_id).first()
        
        if not user or not user.is_active:
            return False
        
        # Update password
        user.hashed_password = pwd_context.hash(new_password)
        
        # Mark token as used
        reset_request.is_used = True
        
        db.add(user)
        db.add(reset_request)
        db.commit()
        
        return True
    
    def change_password(self, db: Session, user_id: int, current_password: str, new_password: str) -> bool:
        """
        Change a user's password.
        
        Args:
            db: Database session
            user_id: User ID
            current_password: Current password
            new_password: New password
            
        Returns:
            True if password was changed, False otherwise
        """
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            return False
        
        if not pwd_context.verify(current_password, user.hashed_password):
            return False
        
        user.hashed_password = pwd_context.hash(new_password)
        
        db.add(user)
        db.commit()
        
        return True
