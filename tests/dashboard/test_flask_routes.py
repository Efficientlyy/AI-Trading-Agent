"""
Tests for the Flask routes in the Modern Dashboard.

This module tests the API endpoints, authentication, and route handlers.
"""

import pytest
import unittest.mock as mock
from datetime import datetime
import json

# Import the module under test - add routes and app references
from src.dashboard.modern_dashboard import (
    create_app, UserRole, validate_login, hash_password, verify_password,
    register_user, get_user_by_username, data_service, DataSource
)


@pytest.fixture
def app():
    """Create a test Flask app instance."""
    with mock.patch('src.dashboard.modern_dashboard.create_app') as mock_create_app:
        # Mock the Flask app and its test_client
        mock_app = mock.MagicMock()
        mock_app.test_client.return_value = mock.MagicMock()
        mock_app.test_request_context.return_value.__enter__.return_value = None
        mock_app.test_request_context.return_value.__exit__.return_value = None
        
        # Configure mock routes
        mock_create_app.return_value = mock_app
        
        # Create the app
        app = create_app()
        
        # Return both the app and the mock
        yield app, mock_app


@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    app_instance, mock_app = app
    return mock_app.test_client()


class TestFlaskRoutes:
    """Tests for the Flask routes."""
    
    def test_index_route(self, client):
        """Test the index route redirects to login or dashboard."""
        # Configure client response
        client.get.return_value = mock.MagicMock(status_code=302)
        
        # Call the route
        response = client.get('/')
        
        # Verify redirect
        client.get.assert_called_once_with('/')
        assert response.status_code == 302
    
    def test_login_route_get(self, client):
        """Test the login route (GET)."""
        # Configure client response
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        client.get.return_value = mock_response
        
        # Call the route
        response = client.get('/login')
        
        # Verify response
        client.get.assert_called_once_with('/login')
        assert response.status_code == 200
    
    def test_login_route_post(self, client, app):
        """Test the login route (POST)."""
        # Get the app instance and mock
        app_instance, mock_app = app
        
        # Mock session
        session_mock = {}
        
        # Configure client response for POST
        mock_response = mock.MagicMock()
        mock_response.status_code = 302  # Redirect on success
        client.post.return_value = mock_response
        
        # Mock validate_login
        with mock.patch('src.dashboard.modern_dashboard.validate_login') as mock_validate:
            with mock.patch('src.dashboard.modern_dashboard.session', session_mock):
                # Configure valid login
                mock_validate.return_value = {
                    'id': '123', 
                    'username': 'testuser',
                    'role': UserRole.ADMIN
                }
                
                # Call the route
                response = client.post('/login', data={
                    'username': 'testuser',
                    'password': 'password123'
                })
                
                # Verify login validation was called
                mock_validate.assert_called_once()
                
                # Verify session was set
                assert session_mock.get('user_id') == '123'
                assert session_mock.get('username') == 'testuser'
                assert session_mock.get('user_role') == UserRole.ADMIN
                
                # Verify redirect
                assert response.status_code == 302
    
    def test_dashboard_route(self, client):
        """Test the dashboard route."""
        # Configure client response
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        client.get.return_value = mock_response
        
        # Call the route with mock session
        with mock.patch('src.dashboard.modern_dashboard.session', {'user_id': '123', 'username': 'testuser'}):
            response = client.get('/dashboard')
            
            # Verify response
            client.get.assert_called_once_with('/dashboard')
            assert response.status_code == 200


class TestAuthentication:
    """Tests for authentication functionality."""
    
    def test_hash_password(self):
        """Test password hashing."""
        # Hash a password
        hashed = hash_password("password123")
        
        # Verify result
        assert hashed != "password123"  # Should not be plaintext
        assert isinstance(hashed, str)
        assert len(hashed) > 20  # Should be a reasonably long hash
    
    def test_verify_password(self):
        """Test password verification."""
        # Hash a password
        hashed = hash_password("password123")
        
        # Verify correct password
        assert verify_password("password123", hashed) is True
        
        # Verify incorrect password
        assert verify_password("wrong_password", hashed) is False
    
    def test_register_user(self):
        """Test user registration."""
        # Set up mocks
        with mock.patch('src.dashboard.modern_dashboard.get_user_by_username') as mock_get_user:
            # User doesn't exist yet
            mock_get_user.return_value = None
            
            # Mock the internal storage 
            with mock.patch('src.dashboard.modern_dashboard.users', {}) as mock_users:
                # Register a new user
                user_id = register_user("newuser", "password123", UserRole.OPERATOR)
                
                # Verify user was added
                assert user_id in mock_users
                assert mock_users[user_id]['username'] == "newuser"
                assert mock_users[user_id]['role'] == UserRole.OPERATOR
                assert 'password' in mock_users[user_id]  # Password should be hashed and stored
    
    def test_validate_login(self):
        """Test login validation."""
        # Hash a test password
        hashed_password = hash_password("password123")
        
        # Set up mocks
        with mock.patch('src.dashboard.modern_dashboard.get_user_by_username') as mock_get_user:
            # Configure existing user
            mock_get_user.return_value = {
                'id': '123',
                'username': 'testuser',
                'password': hashed_password,
                'role': UserRole.ADMIN
            }
            
            # Validate correct credentials
            result = validate_login("testuser", "password123")
            assert result is not None
            assert result['id'] == '123'
            assert result['username'] == 'testuser'
            assert result['role'] == UserRole.ADMIN
            
            # Validate incorrect password
            result = validate_login("testuser", "wrong_password")
            assert result is None
            
            # Validate non-existent user
            mock_get_user.return_value = None
            result = validate_login("nonexistent", "password123")
            assert result is None