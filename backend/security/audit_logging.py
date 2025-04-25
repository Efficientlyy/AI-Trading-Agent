"""
Security Audit Logging System

This module provides functionality to log security-relevant events
such as authentication attempts, rate limit violations, and other
security incidents.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime
from fastapi import Request, Response
import uuid
import os
from pathlib import Path

# Setup logging
logger = logging.getLogger("security.audit")

# Configure security audit logger
audit_handler = logging.FileHandler("logs/security_audit.log")
audit_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
logger.addHandler(audit_handler)
logger.setLevel(logging.INFO)

# Add JSON file handler for machine-readable logs
json_handler = logging.FileHandler("logs/security_audit.json")
json_handler.setFormatter(
    logging.Formatter("%(message)s")  # Raw JSON message
)
json_logger = logging.getLogger("security.audit.json")
json_logger.addHandler(json_handler)
json_logger.setLevel(logging.INFO)
json_logger.propagate = False  # Don't propagate to root logger


class SecurityEventType(str, Enum):
    """Types of security events to be logged"""
    
    # Authentication events
    AUTH_SUCCESS = "authentication_success"
    AUTH_FAILURE = "authentication_failure"
    AUTH_LOGOUT = "authentication_logout"
    AUTH_TOKEN_REFRESH = "authentication_token_refresh"
    AUTH_TOKEN_REVOKED = "authentication_token_revoked"
    AUTH_USER_CREATED = "authentication_user_created"
    AUTH_USER_DELETED = "authentication_user_deleted"
    AUTH_PASSWORD_CHANGED = "authentication_password_changed"
    AUTH_PASSWORD_RESET = "authentication_password_reset"
    AUTH_MFA_ENABLED = "authentication_mfa_enabled"
    AUTH_MFA_DISABLED = "authentication_mfa_disabled"
    AUTH_MFA_CHALLENGE = "authentication_mfa_challenge"
    AUTH_MFA_SUCCESS = "authentication_mfa_success"
    AUTH_MFA_FAILURE = "authentication_mfa_failure"
    
    # Access control events
    ACCESS_DENIED = "access_denied"
    PERMISSION_ERROR = "permission_error"
    PERMISSION_CHANGED = "permission_changed"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    
    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RATE_LIMIT_WARNING = "rate_limit_warning"
    RATE_LIMIT_RESTRICTION_APPLIED = "rate_limit_restriction_applied"
    
    # CSP violations
    CSP_VIOLATION = "csp_violation"
    
    # API key events
    API_KEY_CREATED = "api_key_created"
    API_KEY_DELETED = "api_key_deleted"
    API_KEY_MODIFIED = "api_key_modified"
    API_KEY_USED = "api_key_used"
    API_KEY_EXPIRED = "api_key_expired"
    API_KEY_ROTATION = "api_key_rotation"
    
    # Trading related security events
    TRADE_LIMIT_EXCEEDED = "trade_limit_exceeded"
    SUSPICIOUS_TRADE_PATTERN = "suspicious_trade_pattern"
    ORDER_LIMIT_EXCEEDED = "order_limit_exceeded"
    INVALID_ORDER_ATTEMPT = "invalid_order_attempt"
    
    # Data access events
    DATA_ACCESS = "sensitive_data_access"
    DATA_EXPORT = "data_export"
    DATA_MODIFICATION = "data_modification"
    
    # Other security events
    SECURITY_CONFIG_CHANGED = "security_configuration_changed"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    ACCOUNT_LOCKED = "account_locked"
    IP_BLOCKED = "ip_blocked"
    GEO_LOCATION_CHANGE = "geo_location_change"
    UNUSUAL_ACCESS_TIME = "unusual_access_time"


def _sanitize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize sensitive data before logging
    
    Args:
        data: Dictionary containing data to sanitize
        
    Returns:
        Sanitized data without sensitive information
    """
    sensitive_fields = [
        "password", "token", "secret", "api_key", "key", "credit_card",
        "ssn", "social_security", "auth", "authorization", "access_token",
        "refresh_token", "private_key", "apikey", "session", "credentials"
    ]
    
    sanitized = {}
    for k, v in data.items():
        if isinstance(k, str):
            # Check if the key contains any sensitive field
            is_sensitive = any(field in k.lower() for field in sensitive_fields)
            
            if is_sensitive:
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, dict):
                sanitized[k] = _sanitize_data(v)
            elif isinstance(v, list) and all(isinstance(item, dict) for item in v):
                sanitized[k] = [_sanitize_data(item) for item in v]
            else:
                sanitized[k] = v
        else:
            sanitized[k] = v
            
    return sanitized


def log_security_event(
    event_type: SecurityEventType,
    message: str,
    user_id: Optional[str] = None,
    request: Optional[Request] = None,
    details: Optional[Dict[str, Any]] = None,
    severity: str = "INFO"
) -> str:
    """
    Log a security-related event
    
    Args:
        event_type: Type of security event
        message: Description of the event
        user_id: ID of the user associated with the event (if applicable)
        request: FastAPI request object (if available)
        details: Additional details about the event
        severity: Severity level of the event (INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        The event_id of the logged event
    """
    # Create unique event ID
    event_id = str(uuid.uuid4())
    
    # Create event data structure
    event_data = {
        "event_id": event_id,
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "message": message,
        "severity": severity,
    }
    
    # Add user information if available
    if user_id:
        event_data["user_id"] = user_id
    
    # Add request information if available
    if request:
        event_data["request"] = {
            "method": request.method,
            "url": str(request.url),
            "client_host": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "referer": request.headers.get("referer", ""),
        }
        
        # Add X-Forwarded-For if available
        x_forwarded_for = request.headers.get("x-forwarded-for")
        if x_forwarded_for:
            event_data["request"]["x_forwarded_for"] = x_forwarded_for
    
    # Add additional details if provided (sanitize them first)
    if details:
        event_data["details"] = _sanitize_data(details)
    
    # Log the event with appropriate severity
    log_level = getattr(logging, severity)
    logger.log(log_level, json.dumps(event_data))
    
    # Also log to JSON logger for machine processing
    json_logger.log(log_level, json.dumps(event_data))
    
    return event_id


def log_auth_event(
    success: bool, 
    username: str, 
    request: Optional[Request] = None, 
    error_reason: Optional[str] = None
) -> None:
    """
    Log an authentication attempt
    
    Args:
        success: Whether authentication was successful
        username: Username that attempted to authenticate
        request: FastAPI request object
        error_reason: Reason for authentication failure
    """
    # Determine event type and severity based on success
    if success:
        event_type = SecurityEventType.AUTH_SUCCESS
        severity = "INFO"
        message = f"Successful authentication for user: {username}"
        details = {"username": username}
    else:
        event_type = SecurityEventType.AUTH_FAILURE
        severity = "WARNING"
        message = f"Failed authentication attempt for user: {username}"
        details = {
            "username": username,
            "error_reason": error_reason or "Unknown error"
        }
    
    # Log the event
    log_security_event(
        event_type=event_type,
        message=message,
        user_id=username if success else None,  # Only include user_id if successful
        request=request,
        details=details,
        severity=severity
    )


def log_rate_limit_event(
    limit_type: str,
    client_ip: str,
    endpoint: str,
    current_usage: int,
    limit: int,
    user_id: Optional[str] = None,
    request: Optional[Request] = None
) -> None:
    """
    Log a rate limit event
    
    Args:
        limit_type: Type of rate limit (e.g., 'api', 'login')
        client_ip: IP address of the client
        endpoint: The endpoint being accessed
        current_usage: Current usage count
        limit: Rate limit threshold
        user_id: User ID if authenticated
        request: FastAPI request object
    """
    # Calculate usage percentage
    usage_pct = (current_usage / limit) * 100
    
    # Determine event type and severity based on usage
    if usage_pct >= 100:
        event_type = SecurityEventType.RATE_LIMIT_EXCEEDED
        severity = "WARNING"
        message = f"Rate limit exceeded for {limit_type}: {endpoint}"
    elif usage_pct >= 80:
        event_type = SecurityEventType.RATE_LIMIT_WARNING
        severity = "INFO"
        message = f"Rate limit warning for {limit_type}: {endpoint}"
    else:
        # Don't log normal usage
        return
    
    # Log the event
    log_security_event(
        event_type=event_type,
        message=message,
        user_id=user_id,
        request=request,
        details={
            "limit_type": limit_type,
            "endpoint": endpoint,
            "client_ip": client_ip,
            "current_usage": current_usage,
            "limit": limit,
            "usage_percentage": usage_pct
        },
        severity=severity
    )


def log_data_access_event(
    data_type: str,
    access_type: str,
    resource_id: str,
    user_id: str,
    request: Optional[Request] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log sensitive data access
    
    Args:
        data_type: Type of data being accessed (e.g., 'user_data', 'trade_history')
        access_type: Type of access (e.g., 'read', 'write', 'export', 'delete')
        resource_id: ID of the resource being accessed
        user_id: ID of the user accessing the data
        request: FastAPI request object
        details: Additional details about the access
    """
    # Create message
    message = f"{access_type.capitalize()} access to {data_type} (ID: {resource_id}) by user {user_id}"
    
    # Determine event type based on access type
    if access_type.lower() == 'export':
        event_type = SecurityEventType.DATA_EXPORT
    elif access_type.lower() in ['write', 'update', 'modify', 'delete']:
        event_type = SecurityEventType.DATA_MODIFICATION
    else:
        event_type = SecurityEventType.DATA_ACCESS
    
    # Log the event
    log_security_event(
        event_type=event_type,
        message=message,
        user_id=user_id,
        request=request,
        details={
            "data_type": data_type,
            "access_type": access_type,
            "resource_id": resource_id,
            **(details or {})
        }
    )


def log_suspicious_activity(
    activity_type: str,
    reason: str,
    user_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    request: Optional[Request] = None,
    details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Log suspicious activity
    
    Args:
        activity_type: Type of suspicious activity
        reason: Reason activity is considered suspicious
        user_id: User ID if authenticated
        client_ip: Client IP address
        request: FastAPI request object
        details: Additional details about the activity
        
    Returns:
        The event_id of the logged event
    """
    # Create message
    message = f"Suspicious activity detected: {activity_type}"
    
    # Create details dictionary
    activity_details = {
        "activity_type": activity_type,
        "reason": reason,
        **(details or {})
    }
    
    if client_ip and not request:
        activity_details["client_ip"] = client_ip
    
    # Log the event
    return log_security_event(
        event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
        message=message,
        user_id=user_id,
        request=request,
        details=activity_details,
        severity="WARNING"
    )


def log_security_config_change(
    config_type: str,
    change_type: str,
    old_value: Any,
    new_value: Any,
    user_id: str,
    request: Optional[Request] = None
) -> None:
    """
    Log security configuration changes
    
    Args:
        config_type: Type of configuration being changed
        change_type: Type of change (e.g., 'update', 'create', 'delete')
        old_value: Previous configuration value
        new_value: New configuration value
        user_id: ID of the user making the change
        request: FastAPI request object
    """
    # Create message
    message = f"Security configuration changed: {config_type} {change_type}"
    
    # Sanitize values before logging
    sanitized_old = _sanitize_data({"value": old_value})["value"]
    sanitized_new = _sanitize_data({"value": new_value})["value"]
    
    # Log the event
    log_security_event(
        event_type=SecurityEventType.SECURITY_CONFIG_CHANGED,
        message=message,
        user_id=user_id,
        request=request,
        details={
            "config_type": config_type,
            "change_type": change_type,
            "old_value": sanitized_old,
            "new_value": sanitized_new
        }
    )


def parse_security_logs(limit: int = 100, event_types: Optional[List[SecurityEventType]] = None) -> List[Dict[str, Any]]:
    """
    Parse security audit logs from JSON log file
    
    Args:
        limit: Maximum number of events to return
        event_types: Filter events by type
        
    Returns:
        List of parsed security events
    """
    events = []
    log_file = Path("logs/security_audit.json")
    
    if not log_file.exists():
        return events
    
    # Parse each line as a JSON object
    with open(log_file, "r") as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                
                # Filter by event type if specified
                if event_types and event.get("event_type") not in [et.value for et in event_types]:
                    continue
                    
                events.append(event)
                
                # Stop if we've reached the limit
                if len(events) >= limit:
                    break
                    
            except json.JSONDecodeError:
                continue
    
    # Return events in reverse chronological order
    return list(reversed(events))


class SecurityAuditMiddleware:
    """Middleware that logs security-relevant requests and responses"""
    
    async def __call__(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Extract useful information from the request
        request_data = {
            "id": request_id,
            "method": request.method,
            "path": request.url.path,
            "ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
        }
        
        response = None
        try:
            # Process the request
            response = await call_next(request)
            
            # Log potentially security-relevant responses
            status_code = response.status_code
            if status_code >= 400:  # Log all errors
                severity = "WARNING" if status_code < 500 else "ERROR"
                
                # Determine event type based on status code
                if status_code == 401:
                    event_type = SecurityEventType.AUTH_FAILURE
                elif status_code == 403:
                    event_type = SecurityEventType.ACCESS_DENIED
                elif status_code == 429:
                    event_type = SecurityEventType.RATE_LIMIT_EXCEEDED
                else:
                    event_type = SecurityEventType.SUSPICIOUS_ACTIVITY
                
                log_security_event(
                    event_type=event_type,
                    message=f"Request resulted in {status_code} response",
                    request=request,
                    details={
                        "request": request_data,
                        "response_time": time.time() - start_time,
                        "status_code": status_code,
                    },
                    severity=severity
                )
            
            # Monitor sensitive endpoints even for successful responses
            path = request.url.path.lower()
            sensitive_patterns = [
                "/api/auth", 
                "/api/users", 
                "/api/admin",
                "/api/settings",
                "/api/security",
                "/api/trading",
                "/api/portfolio"
            ]
            
            if response.status_code == 200 and any(pattern in path for pattern in sensitive_patterns):
                # Try to get user_id from request state if available
                user_id = getattr(request.state, "user_id", None)
                
                log_security_event(
                    event_type=SecurityEventType.DATA_ACCESS,
                    message=f"Access to sensitive endpoint: {path}",
                    user_id=user_id,
                    request=request,
                    details={
                        "endpoint": path,
                        "response_time": time.time() - start_time,
                    },
                    severity="INFO"
                )
            
            return response
            
        except Exception as e:
            # Log exceptions as security events
            log_security_event(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                message=f"Exception during request processing: {str(e)}",
                request=request,
                details={
                    "request": request_data,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                severity="ERROR"
            )
            
            # Re-raise the exception for the application to handle
            raise
            

async def handle_csp_report(request: Request) -> Dict[str, str]:
    """
    Handle Content Security Policy violation reports
    
    Args:
        request: FastAPI request containing CSP violation report
        
    Returns:
        Acknowledgment message
    """
    try:
        report_data = await request.json()
        
        log_security_event(
            event_type=SecurityEventType.CSP_VIOLATION,
            message="CSP violation reported",
            request=request,
            details={"csp_report": report_data},
            severity="WARNING"
        )
        
        return {"status": "received"}
    except Exception as e:
        logger.error(f"Error processing CSP report: {e}")
        return {"status": "error", "message": str(e)}


def ensure_log_directory():
    """Ensure the logs directory exists"""
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)