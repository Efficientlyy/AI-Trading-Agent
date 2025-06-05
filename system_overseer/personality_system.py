#!/usr/bin/env python
"""
Personality System for System Overseer.

This module provides the PersonalitySystem class for managing personality traits and templates.
"""

import os
import sys
import json
import logging
import threading
from typing import Dict, Any, List, Optional

logger = logging.getLogger("system_overseer.personality_system")

class PersonalitySystem:
    """Personality System for System Overseer."""
    
    def __init__(self, data_dir: str = "./data/personality"):
        """Initialize Personality System.
        
        Args:
            data_dir: Data directory for personality data
        """
        self.data_dir = data_dir
        self.traits = {}
        self.templates = {}
        self.lock = threading.RLock()  # Add lock attribute
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Load traits and templates
        self._load_traits()
        self._load_templates()
        
        logger.info("PersonalitySystem initialized")
    
    def _load_traits(self):
        """Load personality traits from file."""
        traits_file = os.path.join(self.data_dir, "traits.json")
        
        try:
            with self.lock:
                if os.path.exists(traits_file):
                    with open(traits_file, "r") as f:
                        self.traits = json.load(f)
                else:
                    logger.warning(f"Traits file not found: {traits_file}")
                    # Set default traits
                    self.traits = {
                        "formality": 0.7,
                        "verbosity": 0.6,
                        "helpfulness": 0.9,
                        "proactivity": 0.8
                    }
                    # Save default traits
                    with open(traits_file, "w") as f:
                        json.dump(self.traits, f, indent=2)
        except Exception as e:
            logger.error(f"Error loading traits: {e}")
    
    def _load_templates(self):
        """Load personality templates from file."""
        templates_file = os.path.join(self.data_dir, "templates.json")
        
        try:
            with self.lock:
                if os.path.exists(templates_file):
                    with open(templates_file, "r") as f:
                        self.templates = json.load(f)
                else:
                    logger.warning(f"Templates file not found: {templates_file}")
                    # Set default templates
                    self.templates = {
                        "greeting": "Hello! I'm your Trading System Overseer. How can I assist you today?",
                        "status_report": "The trading system is currently {status}. {details}",
                        "alert": "ALERT: {message}",
                        "recommendation": "Based on current market conditions, I recommend {action}.",
                        "error": "I encountered an error: {error_message}. Please try again or contact support."
                    }
                    # Save default templates
                    with open(templates_file, "w") as f:
                        json.dump(self.templates, f, indent=2)
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    def get_trait(self, trait_name: str, default: float = 0.5):
        """Get personality trait value.
        
        Args:
            trait_name: Trait name
            default: Default value if trait not found
            
        Returns:
            float: Trait value
        """
        with self.lock:
            return self.traits.get(trait_name, default)
    
    def set_trait(self, trait_name: str, value: float):
        """Set personality trait value.
        
        Args:
            trait_name: Trait name
            value: Trait value
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.lock:
                self.traits[trait_name] = value
                
                # Save traits to file
                traits_file = os.path.join(self.data_dir, "traits.json")
                with open(traits_file, "w") as f:
                    json.dump(self.traits, f, indent=2)
                
                return True
        except Exception as e:
            logger.error(f"Error setting trait {trait_name}: {e}")
            return False
    
    def get_template(self, template_name: str, default: str = ""):
        """Get personality template.
        
        Args:
            template_name: Template name
            default: Default value if template not found
            
        Returns:
            str: Template
        """
        with self.lock:
            return self.templates.get(template_name, default)
    
    def set_template(self, template_name: str, template: str):
        """Set personality template.
        
        Args:
            template_name: Template name
            template: Template
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.lock:
                self.templates[template_name] = template
                
                # Save templates to file
                templates_file = os.path.join(self.data_dir, "templates.json")
                with open(templates_file, "w") as f:
                    json.dump(self.templates, f, indent=2)
                
                return True
        except Exception as e:
            logger.error(f"Error setting template {template_name}: {e}")
            return False
    
    def get_traits(self):
        """Get all personality traits.
        
        Returns:
            dict: Personality traits
        """
        with self.lock:
            return self.traits.copy()
    
    def get_templates(self):
        """Get all personality templates.
        
        Returns:
            dict: Personality templates
        """
        with self.lock:
            return self.templates.copy()
    
    def format_template(self, template_name: str, **kwargs):
        """Format personality template with variables.
        
        Args:
            template_name: Template name
            **kwargs: Template variables
            
        Returns:
            str: Formatted template
        """
        template = self.get_template(template_name)
        if not template:
            return ""
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return template
        except Exception as e:
            logger.error(f"Error formatting template {template_name}: {e}")
            return template
