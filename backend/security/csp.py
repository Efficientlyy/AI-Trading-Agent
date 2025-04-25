"""
Content Security Policy Implementation

This module provides a robust Content Security Policy implementation
with violation reporting and monitoring capabilities.
"""

import logging
from typing import Dict, Optional, List, Union
from enum import Enum
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from .audit_logging import log_security_event, SecurityEventType
import json
import os
from datetime import datetime

# Setup logging
logger = logging.getLogger("security.csp")


class CSPDirectiveType(str, Enum):
    """CSP directive types"""
    DEFAULT_SRC = 'default-src'
    SCRIPT_SRC = 'script-src'
    STYLE_SRC = 'style-src'
    IMG_SRC = 'img-src'
    CONNECT_SRC = 'connect-src'
    FONT_SRC = 'font-src'
    OBJECT_SRC = 'object-src'
    MEDIA_SRC = 'media-src'
    FRAME_SRC = 'frame-src'
    WORKER_SRC = 'worker-src'
    MANIFEST_SRC = 'manifest-src'
    FRAME_ANCESTORS = 'frame-ancestors'
    FORM_ACTION = 'form-action'
    BASE_URI = 'base-uri'
    REPORT_TO = 'report-to'
    REPORT_URI = 'report-uri'


class CSPViolation(BaseModel):
    """Model for CSP violation reports"""
    document_uri: Optional[str] = None
    referrer: Optional[str] = None
    blocked_uri: Optional[str] = None
    violated_directive: Optional[str] = None
    original_policy: Optional[str] = None
    disposition: Optional[str] = None
    effective_directive: Optional[str] = None
    status_code: Optional[int] = None
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    
    @classmethod
    def from_report(cls, report_json: Dict) -> 'CSPViolation':
        """Create a CSPViolation from a CSP violation report"""
        if "csp-report" in report_json:
            data = report_json["csp-report"]
            return cls(
                document_uri=data.get("document-uri"),
                referrer=data.get("referrer"),
                blocked_uri=data.get("blocked-uri"),
                violated_directive=data.get("violated-directive"),
                original_policy=data.get("original-policy"),
                disposition=data.get("disposition"),
                effective_directive=data.get("effective-directive"),
                status_code=data.get("status-code"),
                source_file=data.get("source-file"),
                line_number=data.get("line-number"),
                column_number=data.get("column-number")
            )
        return cls()


class CSPReportData(BaseModel):
    """Model for CSP violation report data"""
    violation: CSPViolation
    user_agent: str
    client_ip: str
    timestamp: str


class CSPConfig:
    """Content Security Policy configuration"""
    def __init__(self, 
                report_only: bool = False,
                report_uri: str = "/api/security/csp-report"):
        self.directives: Dict[str, List[str]] = {}
        self.report_only = report_only
        self.report_uri = report_uri
        
        # Initialize with secure defaults
        self.add_directive(CSPDirectiveType.DEFAULT_SRC, ["'self'"])
        self.add_directive(CSPDirectiveType.OBJECT_SRC, ["'none'"])
        self.add_directive(CSPDirectiveType.BASE_URI, ["'self'"])
        self.add_directive(CSPDirectiveType.FRAME_ANCESTORS, ["'self'"])
        self.add_directive(CSPDirectiveType.FORM_ACTION, ["'self'"])
        self.add_directive(CSPDirectiveType.REPORT_URI, [report_uri])
        
    def add_directive(self, directive: Union[CSPDirectiveType, str], sources: List[str]) -> None:
        """
        Add or update a CSP directive
        
        Args:
            directive: CSP directive name
            sources: List of allowed sources for this directive
        """
        if isinstance(directive, CSPDirectiveType):
            directive = directive.value
            
        self.directives[directive] = sources
        
    def get_header_value(self) -> str:
        """
        Get the CSP header value string
        
        Returns:
            String representation of the CSP policy
        """
        directives = []
        for directive, sources in self.directives.items():
            if sources:
                directives.append(f"{directive} {' '.join(sources)}")
            else:
                directives.append(f"{directive}")
                
        return "; ".join(directives)
    
    def get_header_name(self) -> str:
        """
        Get the appropriate CSP header name based on mode
        
        Returns:
            Header name (Content-Security-Policy or Content-Security-Policy-Report-Only)
        """
        if self.report_only:
            return "Content-Security-Policy-Report-Only"
        return "Content-Security-Policy"


def configure_csp_environment(environment: str) -> CSPConfig:
    """
    Configure CSP based on environment
    
    Args:
        environment: Application environment (development, staging, production)
        
    Returns:
        CSPConfig object with environment-appropriate settings
    """
    # Start with base config
    if environment.lower() in ["prod", "production"]:
        # Strict CSP for production
        csp = CSPConfig(report_only=False)
        
        # Define trusted domains
        app_domain = "ai-trading-agent.com"
        cdn_domain = "cdn.ai-trading-agent.com"
        api_domain = "api.ai-trading-agent.com"
        
        # Configure directives for production
        csp.add_directive(CSPDirectiveType.SCRIPT_SRC, [
            "'self'",
            cdn_domain,
            "'strict-dynamic'",
            "'nonce-{NONCE}'",  # Will be replaced dynamically
        ])
        
        csp.add_directive(CSPDirectiveType.STYLE_SRC, ["'self'", cdn_domain])
        csp.add_directive(CSPDirectiveType.IMG_SRC, ["'self'", cdn_domain, "data:"])
        csp.add_directive(CSPDirectiveType.CONNECT_SRC, ["'self'", api_domain])
        csp.add_directive(CSPDirectiveType.FONT_SRC, ["'self'", cdn_domain, "data:"])
        
    elif environment.lower() in ["staging", "test"]:
        # Moderate CSP for staging
        csp = CSPConfig(report_only=False)
        
        csp.add_directive(CSPDirectiveType.SCRIPT_SRC, [
            "'self'", 
            "'unsafe-inline'",  # More permissive for staging
            "https://cdn.jsdelivr.net",
        ])
        
        csp.add_directive(CSPDirectiveType.STYLE_SRC, [
            "'self'", 
            "'unsafe-inline'",
            "https://cdn.jsdelivr.net",
        ])
        
        csp.add_directive(CSPDirectiveType.CONNECT_SRC, ["'self'", "https://*"])
        
    else:
        # Development mode - report only to avoid breaking functionality during development
        csp = CSPConfig(report_only=True)
        
        # Very permissive for local development
        csp.add_directive(CSPDirectiveType.SCRIPT_SRC, [
            "'self'", 
            "'unsafe-inline'", 
            "'unsafe-eval'",
            "https://*",
        ])
        
        csp.add_directive(CSPDirectiveType.STYLE_SRC, [
            "'self'", 
            "'unsafe-inline'",
            "https://*",
        ])
        
        csp.add_directive(CSPDirectiveType.CONNECT_SRC, ["'self'", "*"])
        
    return csp


class CSPMiddleware:
    """Middleware to add CSP headers to responses"""
    
    def __init__(self, app: FastAPI, csp_config: CSPConfig):
        self.app = app
        self.csp_config = csp_config
        
    async def __call__(self, request: Request, call_next):
        # Generate CSP nonce (if needed)
        # In a real implementation, this would be a cryptographically secure random value
        # and would be stored in the request state for templates to access
        from secrets import token_hex
        nonce = token_hex(16)
        request.state.csp_nonce = nonce
        
        # Process the request
        response = await call_next(request)
        
        # Add CSP header to response
        header_name = self.csp_config.get_header_name()
        header_value = self.csp_config.get_header_value().replace("{NONCE}", nonce)
        response.headers[header_name] = header_value
        
        return response


async def handle_csp_report(request: Request) -> JSONResponse:
    """
    Handler for CSP violation reports
    
    Args:
        request: FastAPI request with CSP report data
        
    Returns:
        JSON response acknowledging receipt
    """
    try:
        report_data = await request.json()
        
        # Create structured violation data
        violation = CSPViolation.from_report(report_data)
        
        # Create full report with context
        report = CSPReportData(
            violation=violation,
            user_agent=request.headers.get("user-agent", "unknown"),
            client_ip=request.client.host if request.client else "unknown",
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Log the violation
        log_security_event(
            event_type=SecurityEventType.CSP_VIOLATION,
            message="Content Security Policy violation detected",
            request=request,
            details={
                "violation": json.loads(report.json())
            },
            severity="WARNING"
        )
        
        # Save detailed report to CSP violations log
        csp_logger = logging.getLogger("security.csp.violations")
        csp_logger.warning(report.json())
        
        return JSONResponse({"status": "report received"})
        
    except Exception as e:
        logger.error(f"Error processing CSP report: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


def configure_csp(app: FastAPI) -> None:
    """
    Configure Content Security Policy for a FastAPI application
    
    Args:
        app: FastAPI application instance
    """
    # Get environment
    environment = os.environ.get("APP_ENV", "development")
    
    # Create CSP configuration
    csp_config = configure_csp_environment(environment)
    
    # Add CSP middleware
    app.add_middleware(CSPMiddleware, csp_config=csp_config)
    
    # Register CSP report endpoint
    app.post("/api/security/csp-report")(handle_csp_report)
    
    # Add response header with nonce template
    @app.middleware("http")
    async def add_csp_nonce_to_html_response(request: Request, call_next):
        response = await call_next(request)
        
        # Only process HTML responses
        if isinstance(response, HTMLResponse):
            nonce = getattr(request.state, "csp_nonce", "")
            
            # Add nonce to response context for templates
            if hasattr(response, "context") and isinstance(response.context, dict):
                response.context["csp_nonce"] = nonce
                
        return response
    
    logger.info(f"CSP configured for {environment} environment")


def generate_csp_report_template() -> str:
    """
    Generate HTML template for CSP violation reporting
    
    Returns:
        HTML with JavaScript for CSP reporting
    """
    return """
    <script nonce="{csp_nonce}">
    document.addEventListener('securitypolicyviolation', (e) => {
        const violationData = {
            'csp-report': {
                'document-uri': e.documentURI,
                'referrer': e.referrer,
                'violated-directive': e.violatedDirective,
                'effective-directive': e.effectiveDirective,
                'original-policy': e.originalPolicy,
                'disposition': e.disposition,
                'blocked-uri': e.blockedURI,
                'status-code': 0
            }
        };
        
        fetch('/api/security/csp-report', {
            method: 'POST',
            body: JSON.stringify(violationData),
            headers: {
                'Content-Type': 'application/csp-report'
            }
        }).catch(console.error);
    });
    </script>
    """