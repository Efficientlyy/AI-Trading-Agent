"""
Authentication Module

This module provides authentication functionality for the dashboard.
Following modular design principles, authentication is separated from the
main dashboard logic to maintain single responsibility and keep files under size limits.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from functools import wraps

from flask import request, session, redirect, url_for, flash
from src.dashboard.utils.enums import UserRole

# Default session duration (12 hours)
DEFAULT_SESSION_DURATION = timedelta(hours=12)

class AuthManager:
    """
    Manages authentication and authorization for the dashboard
    
    Features:
    - Secure password hashing and verification
    - Role-based access control
    - Session management
    """
    
    def __init__(self, session_duration: timedelta = DEFAULT_SESSION_DURATION):
        """Initialize the auth manager with default user accounts"""
        self.session_duration = session_duration
        
        # Default user accounts (in-memory for demonstration)
        # In a production environment, this would use a secure database
        self.users = {
            'admin': {
                'password_hash': self._hash_password('admin123'),
                'role': UserRole.ADMIN,
                'name': 'Administrator',
                'last_login': None
            },
            'operator': {
                'password_hash': self._hash_password('operator123'),
                'role': UserRole.OPERATOR,
                'name': 'System Operator',
                'last_login': None
            },
            'viewer': {
                'password_hash': self._hash_password('viewer123'),
                'role': UserRole.VIEWER,
                'name': 'Dashboard Viewer',
                'last_login': None
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash a password for secure storage"""
        # Generate a random salt
        salt = secrets.token_hex(16)
        
        # Hash the password with the salt
        hash_obj = hashlib.sha256()
        hash_obj.update((password + salt).encode('utf-8'))
        password_hash = hash_obj.hexdigest()
        
        # Return the salt and hash, separated by a colon
        return f"{salt}:{password_hash}"
    
    def _verify_password(self, stored_password: str, provided_password: str) -> bool:
        """Verify a password against its hash"""
        # Split the stored password into salt and hash
        salt, stored_hash = stored_password.split(':')
        
        # Hash the provided password with the same salt
        hash_obj = hashlib.sha256()
        hash_obj.update((provided_password + salt).encode('utf-8'))
        computed_hash = hash_obj.hexdigest()
        
        # Compare the stored hash with the computed hash
        return secrets.compare_digest(stored_hash, computed_hash)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with username and password"""
        # Check if the user exists
        if username not in self.users:
            return None
        
        # Get the user record
        user = self.users[username]
        
        # Verify the password
        if not self._verify_password(user['password_hash'], password):
            return None
        
        # Update last login time
        user['last_login'] = datetime.now()
        
        # Return the user record (without the password hash)
        return {
            'username': username,
            'role': user['role'],
            'name': user['name'],
            'last_login': user['last_login']
        }
    
    def add_user(self, username: str, password: str, role: str, name: str) -> bool:
        """Add a new user"""
        # Check if the user already exists
        if username in self.users:
            return False
        
        # Validate the role
        if role not in [UserRole.ADMIN, UserRole.OPERATOR, UserRole.VIEWER]:
            return False
        
        # Add the user
        self.users[username] = {
            'password_hash': self._hash_password(password),
            'role': role,
            'name': name,
            'last_login': None
        }
        
        return True
    
    def update_user(self, username: str, password: Optional[str] = None,
                   role: Optional[str] = None, name: Optional[str] = None) -> bool:
        """Update an existing user"""
        # Check if the user exists
        if username not in self.users:
            return False
        
        # Get the user record
        user = self.users[username]
        
        # Update the password if provided
        if password:
            user['password_hash'] = self._hash_password(password)
        
        # Update the role if provided and valid
        if role and role in [UserRole.ADMIN, UserRole.OPERATOR, UserRole.VIEWER]:
            user['role'] = role
        
        # Update the name if provided
        if name:
            user['name'] = name
        
        return True
    
    def delete_user(self, username: str) -> bool:
        """Delete a user"""
        # Check if the user exists
        if username not in self.users:
            return False
        
        # Delete the user
        del self.users[username]
        
        return True
    
    def get_users(self) -> List[Dict[str, Any]]:
        """Get a list of all users (without password hashes)"""
        return [
            {
                'username': username,
                'role': user['role'],
                'name': user['name'],
                'last_login': user['last_login']
            }
            for username, user in self.users.items()
        ]
    
    def login_required(self, f):
        """Decorator to require login for routes"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                flash('Please log in to access this page.', 'warning')
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    
    def role_required(self, required_roles):
        """Decorator to require specific roles for routes"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # First check if the user is logged in
                if 'user' not in session:
                    flash('Please log in to access this page.', 'warning')
                    return redirect(url_for('login'))
                
                # Then check if the user has the required role
                if session['user']['role'] not in required_roles:
                    flash('You do not have permission to access this page.', 'danger')
                    return redirect(url_for('dashboard'))
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
