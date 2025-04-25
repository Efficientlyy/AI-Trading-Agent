"""
Permissions Module

This module implements role-based access control (RBAC) for the AI Trading Agent.
It defines user roles, permissions, and provides utilities for permission checking
throughout the application.
"""

import enum
from typing import Dict, List, Set, Optional, Union, Callable, Any
from functools import wraps

from fastapi import HTTPException, status, Depends, Security
from fastapi.security import SecurityScopes

# Import the authentication module
from backend.security.auth import get_current_user


class Permission(str, enum.Enum):
    """
    Enumeration of all available permissions in the system.
    Permissions are fine-grained capabilities that can be assigned to roles.
    """
    # User management
    USER_VIEW = "user:view"
    USER_CREATE = "user:create"
    USER_EDIT = "user:edit"
    USER_DELETE = "user:delete"
    
    # Trading permissions
    TRADE_READ = "trade:read"
    TRADE_CREATE = "trade:create"
    TRADE_CANCEL = "trade:cancel"
    TRADE_ALL = "trade:all"
    
    # Strategy permissions
    STRATEGY_VIEW = "strategy:view"
    STRATEGY_CREATE = "strategy:create" 
    STRATEGY_EDIT = "strategy:edit"
    STRATEGY_DELETE = "strategy:delete"
    
    # Market data
    MARKET_DATA_READ = "market:read"
    MARKET_DATA_WRITE = "market:write"
    
    # API access
    API_READ = "api:read"
    API_WRITE = "api:write"
    
    # Admin permissions
    ADMIN_ACCESS = "admin:access"
    SYSTEM_CONFIG = "system:config"
    
    # Exchange management
    EXCHANGE_VIEW = "exchange:view"
    EXCHANGE_CONNECT = "exchange:connect"
    EXCHANGE_MODIFY = "exchange:modify"
    
    # Analytics and reporting
    ANALYTICS_VIEW = "analytics:view"
    REPORT_GENERATE = "report:generate"
    EXPORT_DATA = "export:data"


class Role(str, enum.Enum):
    """
    Enumeration of user roles in the system.
    Each role has a set of associated permissions.
    """
    GUEST = "guest"
    USER = "user"
    TRADER = "trader"
    ANALYST = "analyst"
    ADMIN = "admin"
    SYSTEM = "system"


# Role-permission mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.GUEST: {
        Permission.MARKET_DATA_READ,
    },
    
    Role.USER: {
        Permission.MARKET_DATA_READ,
        Permission.STRATEGY_VIEW,
        Permission.TRADE_READ,
        Permission.ANALYTICS_VIEW,
    },
    
    Role.TRADER: {
        Permission.MARKET_DATA_READ,
        Permission.STRATEGY_VIEW,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_EDIT,
        Permission.TRADE_READ,
        Permission.TRADE_CREATE,
        Permission.TRADE_CANCEL,
        Permission.ANALYTICS_VIEW,
        Permission.EXCHANGE_VIEW,
    },
    
    Role.ANALYST: {
        Permission.MARKET_DATA_READ,
        Permission.MARKET_DATA_WRITE,
        Permission.STRATEGY_VIEW,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_EDIT,
        Permission.TRADE_READ,
        Permission.ANALYTICS_VIEW,
        Permission.REPORT_GENERATE,
        Permission.EXPORT_DATA,
    },
    
    Role.ADMIN: {
        Permission.USER_VIEW,
        Permission.USER_CREATE,
        Permission.USER_EDIT,
        Permission.USER_DELETE,
        Permission.MARKET_DATA_READ,
        Permission.MARKET_DATA_WRITE,
        Permission.STRATEGY_VIEW,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_EDIT,
        Permission.STRATEGY_DELETE,
        Permission.TRADE_READ,
        Permission.TRADE_CREATE,
        Permission.TRADE_CANCEL,
        Permission.TRADE_ALL,
        Permission.ANALYTICS_VIEW,
        Permission.REPORT_GENERATE,
        Permission.EXPORT_DATA,
        Permission.ADMIN_ACCESS,
        Permission.SYSTEM_CONFIG,
        Permission.EXCHANGE_VIEW,
        Permission.EXCHANGE_CONNECT,
        Permission.EXCHANGE_MODIFY,
    },
    
    Role.SYSTEM: {
        # System has all permissions
        p for p in Permission
    },
}


class PermissionChecker:
    """
    Handles permission verification throughout the application.
    """
    
    @staticmethod
    def has_permission(
        user_permissions: Union[List[str], Set[str]], 
        required_permission: Permission
    ) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user_permissions: List or set of permission strings the user has
            required_permission: Permission to check for
            
        Returns:
            bool: True if user has the permission, False otherwise
        """
        return str(required_permission) in user_permissions
    
    @staticmethod
    def has_role(
        user_roles: Union[List[str], Set[str]],
        required_role: Role
    ) -> bool:
        """
        Check if a user has a specific role.
        
        Args:
            user_roles: List or set of roles the user has
            required_role: Role to check for
            
        Returns:
            bool: True if user has the role, False otherwise
        """
        return str(required_role) in user_roles
    
    @staticmethod
    def get_permissions_for_roles(roles: List[str]) -> Set[Permission]:
        """
        Get all permissions for a list of roles.
        
        Args:
            roles: List of role names
            
        Returns:
            Set[Permission]: Set of all permissions granted by the roles
        """
        permissions = set()
        for role_name in roles:
            try:
                role = Role(role_name)
                if role in ROLE_PERMISSIONS:
                    permissions.update(ROLE_PERMISSIONS[role])
            except ValueError:
                # Invalid role name, skip
                continue
        return permissions


# Dependency helpers for FastAPI routes
async def require_permission(
    permission: Permission,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    FastAPI dependency that requires a specific permission.
    
    Args:
        permission: The permission required to access the endpoint
        current_user: The current authenticated user
        
    Returns:
        Dict: Current user if permitted
        
    Raises:
        HTTPException: If user doesn't have the required permission
    """
    # Get user's permissions from roles
    user_roles = current_user.get("roles", [])
    user_permissions = PermissionChecker.get_permissions_for_roles(user_roles)
    
    # Additional explicit permissions (if any)
    explicit_permissions = current_user.get("permissions", [])
    for p in explicit_permissions:
        try:
            user_permissions.add(Permission(p))
        except ValueError:
            # Invalid permission, skip
            pass
    
    # Check if user has the required permission
    if permission not in user_permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorized. Required permission: {permission}"
        )
    
    return current_user


async def require_role(
    role: Role,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    FastAPI dependency that requires a specific role.
    
    Args:
        role: The role required to access the endpoint
        current_user: The current authenticated user
        
    Returns:
        Dict: Current user if permitted
        
    Raises:
        HTTPException: If user doesn't have the required role
    """
    user_roles = current_user.get("roles", [])
    if not PermissionChecker.has_role(user_roles, role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorized. Required role: {role}"
        )
    return current_user


# Convenience functions for common role checks
async def require_admin(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require admin role"""
    return await require_role(Role.ADMIN, current_user)


async def require_trader(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require trader role or higher"""
    user_roles = current_user.get("roles", [])
    if (not PermissionChecker.has_role(user_roles, Role.TRADER) and
        not PermissionChecker.has_role(user_roles, Role.ADMIN) and
        not PermissionChecker.has_role(user_roles, Role.SYSTEM)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Trader role or higher required"
        )
    return current_user


# Usage examples:
# @app.get("/api/trades", dependencies=[Depends(require_permission(Permission.TRADE_READ))])
# @app.post("/api/trades", dependencies=[Depends(require_permission(Permission.TRADE_CREATE))])
# @app.get("/api/admin", dependencies=[Depends(require_admin)])