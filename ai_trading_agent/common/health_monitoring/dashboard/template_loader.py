"""
Template Loader for Health Monitoring Dashboard.

This module provides utilities for loading dashboard templates from files
and creating default templates if they don't exist.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List

# Set up logger
logger = logging.getLogger(__name__)


class TemplateLoader:
    """
    Utility for loading dashboard templates from files.
    
    Handles loading HTML, CSS, and JavaScript templates and
    creating default templates if they don't exist.
    """
    
    def __init__(self, dashboard_dir: Optional[str] = None):
        """
        Initialize the template loader.
        
        Args:
            dashboard_dir: Optional directory for dashboard templates and static files
        """
        # Set up dashboard directory
        if dashboard_dir:
            self.dashboard_dir = Path(dashboard_dir)
        else:
            # Default to package directory
            current_file = Path(__file__).resolve()
            self.dashboard_dir = current_file.parent
            
        # Set up templates and static directories
        self.templates_dir = self.dashboard_dir / "templates"
        self.static_dir = self.dashboard_dir / "static"
        self.js_dir = self.static_dir / "js"
        self.css_dir = self.static_dir / "css"
        self.img_dir = self.static_dir / "img"
        
        # Ensure directories exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.js_dir.mkdir(parents=True, exist_ok=True)
        self.css_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Template loader initialized with dashboard directory: {self.dashboard_dir}")
    
    def load_template(self, template_name: str, template_type: str = "html") -> str:
        """
        Load a template from file or return default if it doesn't exist.
        
        Args:
            template_name: Name of the template to load
            template_type: Type of template (html, js, css)
            
        Returns:
            Template content as string
        """
        if template_type == "html":
            template_path = self.templates_dir / f"{template_name}.html"
        elif template_type == "js":
            template_path = self.js_dir / f"{template_name}.js"
        elif template_type == "css":
            template_path = self.css_dir / f"{template_name}.css"
        else:
            raise ValueError(f"Unknown template type: {template_type}")
            
        if template_path.exists():
            logger.debug(f"Loading template: {template_path}")
            with open(template_path, "r") as f:
                return f.read()
        else:
            logger.warning(f"Template not found: {template_path}, creating default")
            content = self._get_default_template(template_name, template_type)
            self._write_template(template_path, content)
            return content
    
    def _write_template(self, template_path: Path, content: str) -> None:
        """
        Write a template to file.
        
        Args:
            template_path: Path to the template file
            content: Template content to write
        """
        try:
            with open(template_path, "w") as f:
                f.write(content)
            logger.info(f"Created template: {template_path}")
        except Exception as e:
            logger.error(f"Error creating template {template_path}: {str(e)}")
    
    def _get_default_template(self, template_name: str, template_type: str) -> str:
        """
        Get default template content.
        
        Args:
            template_name: Name of the template
            template_type: Type of template (html, js, css)
            
        Returns:
            Default template content
        """
        # Import default templates only when needed to avoid circular imports
        from .default_templates import (
            get_default_html_template,
            get_default_js_template,
            get_default_css_template
        )
        
        if template_type == "html":
            return get_default_html_template(template_name)
        elif template_type == "js":
            return get_default_js_template(template_name)
        elif template_type == "css":
            return get_default_css_template(template_name)
        else:
            raise ValueError(f"Unknown template type: {template_type}")
    
    def ensure_all_templates_exist(self) -> None:
        """
        Ensure all required templates exist, creating defaults if needed.
        """
        # HTML templates
        html_templates = ["index", "components", "metrics", "alerts"]
        for template_name in html_templates:
            self.load_template(template_name, "html")
            
        # JavaScript templates
        js_templates = ["dashboard", "components", "metrics", "alerts"]
        for template_name in js_templates:
            self.load_template(template_name, "js")
            
        # CSS templates
        css_templates = ["dashboard"]
        for template_name in css_templates:
            self.load_template(template_name, "css")
            
        logger.info("Ensured all required templates exist")
