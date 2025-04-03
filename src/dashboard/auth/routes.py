"""
Authentication Module

This module provides authentication functionality for the dashboard.
It includes user management, login/logout handling, and role-based access control.
"""

import os
from functools import wraps
from typing import Dict, Any, Optional, List

from flask import Blueprint, render_template, session, redirect, url_for, request, flash
from src.dashboard.utils.auth import AuthManager

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Initialize auth manager
auth_manager = AuthManager()

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    # Check if user is already logged in
    if 'user' in session:
        return redirect(url_for('dashboard.index'))
    
    # Handle login form submission
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Authenticate user
        user = auth_manager.authenticate_user(username, password)
        
        if user:
            # Store user in session
            session['user'] = user
            flash(f'Welcome back, {user["name"]}!', 'success')
            return redirect(url_for('dashboard.index'))
        else:
            # Authentication failed
            flash('Invalid username or password', 'danger')
    
    # Display login form
    return render_template('auth/login.html')

@auth_bp.route('/logout')
def logout():
    """Handle user logout."""
    # Clear session
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/profile')
@auth_manager.login_required
def profile():
    """Display user profile."""
    return render_template('auth/profile.html', user=session['user'])

@auth_bp.route('/users')
@auth_manager.role_required(['admin'])
def users():
    """Display user management page."""
    # Get all users
    users_list = auth_manager.get_users()
    return render_template('auth/users.html', users=users_list)

@auth_bp.route('/users/add', methods=['GET', 'POST'])
@auth_manager.role_required(['admin'])
def add_user():
    """Add a new user."""
    # Handle form submission
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')
        name = request.form.get('name')
        
        # Add user
        success = auth_manager.add_user(username, password, role, name)
        
        if success:
            flash(f'User {username} added successfully', 'success')
            return redirect(url_for('auth.users'))
        else:
            flash(f'Failed to add user {username}', 'danger')
    
    # Display form
    return render_template('auth/add_user.html')

@auth_bp.route('/users/edit/<username>', methods=['GET', 'POST'])
@auth_manager.role_required(['admin'])
def edit_user(username):
    """Edit an existing user."""
    # Get all users
    users_list = auth_manager.get_users()
    
    # Find user
    user = next((u for u in users_list if u['username'] == username), None)
    
    if not user:
        flash(f'User {username} not found', 'danger')
        return redirect(url_for('auth.users'))
    
    # Handle form submission
    if request.method == 'POST':
        password = request.form.get('password')
        role = request.form.get('role')
        name = request.form.get('name')
        
        # Update user
        success = auth_manager.update_user(username, password, role, name)
        
        if success:
            flash(f'User {username} updated successfully', 'success')
            return redirect(url_for('auth.users'))
        else:
            flash(f'Failed to update user {username}', 'danger')
    
    # Display form
    return render_template('auth/edit_user.html', user=user)

@auth_bp.route('/users/delete/<username>', methods=['POST'])
@auth_manager.role_required(['admin'])
def delete_user(username):
    """Delete a user."""
    # Delete user
    success = auth_manager.delete_user(username)
    
    if success:
        flash(f'User {username} deleted successfully', 'success')
    else:
        flash(f'Failed to delete user {username}', 'danger')
    
    return redirect(url_for('auth.users'))

def init_app(app):
    """Initialize authentication module with the Flask app."""
    # Register blueprint
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    # Add authentication manager to app context
    app.auth_manager = auth_manager
    
    # Add login_required decorator to app context
    app.login_required = auth_manager.login_required
    
    # Add role_required decorator to app context
    app.role_required = auth_manager.role_required
    
    # Add before_request handler to check for active session
    @app.before_request
    def check_session():
        # Skip for static files and login/logout routes
        if request.path.startswith('/static') or \
           request.path.startswith('/auth/login') or \
           request.path == '/auth/logout':
            return
        
        # Check if user is logged in
        if 'user' not in session:
            return redirect(url_for('auth.login'))
