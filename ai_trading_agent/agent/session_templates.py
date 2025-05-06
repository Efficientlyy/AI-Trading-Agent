"""
Session Templates module for the AI Trading Agent.

This module provides predefined templates for paper trading sessions.
"""

from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path

from ..common import logger


class SessionTemplate:
    """
    Represents a template for paper trading sessions.
    """
    
    def __init__(
        self,
        template_id: str,
        name: str,
        description: str,
        config: Dict[str, Any],
        tags: List[str] = None
    ):
        """
        Initialize a session template.
        
        Args:
            template_id: Unique identifier for the template
            name: Display name for the template
            description: Description of the template
            config: Configuration settings for the template
            tags: Tags for categorizing the template
        """
        self.template_id = template_id
        self.name = name
        self.description = description
        self.config = config
        self.tags = tags or []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert template to dictionary.
        
        Returns:
            Template as a dictionary
        """
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionTemplate':
        """
        Create a template from a dictionary.
        
        Args:
            data: Dictionary with template data
            
        Returns:
            SessionTemplate instance
        """
        return cls(
            template_id=data["template_id"],
            name=data["name"],
            description=data["description"],
            config=data["config"],
            tags=data.get("tags", [])
        )


class SessionTemplateManager:
    """
    Manages session templates for quick configuration.
    """
    
    def __init__(self, templates_dir: str = None):
        """
        Initialize the session template manager.
        
        Args:
            templates_dir: Directory for storing templates
        """
        if templates_dir is None:
            # Default to a templates directory in the project
            templates_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "templates",
                "sessions"
            )
        
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, SessionTemplate] = {}
        
        # Create templates directory if it doesn't exist
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Load templates
        self._load_templates()
        
        # Create default templates if none exist
        if not self.templates:
            self._create_default_templates()
    
    def _load_templates(self) -> None:
        """Load templates from the templates directory."""
        try:
            for file_path in self.templates_dir.glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        template = SessionTemplate.from_dict(data)
                        self.templates[template.template_id] = template
                        logger.info(f"Loaded template: {template.name}")
                except Exception as e:
                    logger.error(f"Error loading template from {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
    
    def _create_default_templates(self) -> None:
        """Create default templates."""
        # Basic template
        basic = SessionTemplate(
            template_id="basic",
            name="Basic Trading",
            description="Basic paper trading session with default settings",
            config={
                "duration_minutes": 60,
                "interval_minutes": 1,
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "initial_capital": 10000.0
            },
            tags=["basic", "default"]
        )
        self.save_template(basic)
        
        # Technical Analysis template
        technical = SessionTemplate(
            template_id="technical",
            name="Technical Analysis",
            description="Paper trading using technical indicators",
            config={
                "duration_minutes": 120,
                "interval_minutes": 1,
                "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
                "initial_capital": 10000.0,
                "strategies": ["ma_crossover", "rsi", "bollinger_bands"],
                "risk_management": {
                    "max_position_size": 0.2,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.05
                }
            },
            tags=["technical", "indicators"]
        )
        self.save_template(technical)
        
        # Sentiment Analysis template
        sentiment = SessionTemplate(
            template_id="sentiment",
            name="Sentiment Trading",
            description="Paper trading using sentiment analysis",
            config={
                "duration_minutes": 180,
                "interval_minutes": 5,
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "initial_capital": 10000.0,
                "strategies": ["sentiment"],
                "sentiment_sources": ["twitter", "reddit", "news"],
                "risk_management": {
                    "max_position_size": 0.15,
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.06
                }
            },
            tags=["sentiment", "social"]
        )
        self.save_template(sentiment)
        
        # Multi-Strategy template
        multi_strategy = SessionTemplate(
            template_id="multi_strategy",
            name="Multi-Strategy Trading",
            description="Paper trading using multiple strategies",
            config={
                "duration_minutes": 240,
                "interval_minutes": 2,
                "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOT/USDT", "LINK/USDT"],
                "initial_capital": 20000.0,
                "strategies": ["ma_crossover", "rsi", "sentiment", "volume_profile"],
                "strategy_weights": {
                    "ma_crossover": 0.3,
                    "rsi": 0.2,
                    "sentiment": 0.3,
                    "volume_profile": 0.2
                },
                "risk_management": {
                    "max_position_size": 0.1,
                    "stop_loss_pct": 0.025,
                    "take_profit_pct": 0.05,
                    "max_open_positions": 3
                }
            },
            tags=["multi-strategy", "advanced"]
        )
        self.save_template(multi_strategy)
        
        # High-Frequency template
        high_frequency = SessionTemplate(
            template_id="high_frequency",
            name="High-Frequency Trading",
            description="Paper trading with high-frequency updates",
            config={
                "duration_minutes": 60,
                "interval_minutes": 0.5,  # 30 seconds
                "symbols": ["BTC/USDT"],
                "initial_capital": 10000.0,
                "strategies": ["momentum", "order_book_imbalance"],
                "risk_management": {
                    "max_position_size": 0.05,
                    "stop_loss_pct": 0.01,
                    "take_profit_pct": 0.02,
                    "max_trades_per_minute": 10
                }
            },
            tags=["high-frequency", "scalping"]
        )
        self.save_template(high_frequency)
    
    def get_template(self, template_id: str) -> Optional[SessionTemplate]:
        """
        Get a template by ID.
        
        Args:
            template_id: ID of the template to get
            
        Returns:
            The template or None if not found
        """
        return self.templates.get(template_id)
    
    def get_all_templates(self) -> List[SessionTemplate]:
        """
        Get all templates.
        
        Returns:
            List of all templates
        """
        return list(self.templates.values())
    
    def get_templates_by_tag(self, tag: str) -> List[SessionTemplate]:
        """
        Get templates by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of templates with the specified tag
        """
        return [
            template for template in self.templates.values()
            if tag in template.tags
        ]
    
    def save_template(self, template: SessionTemplate) -> None:
        """
        Save a template.
        
        Args:
            template: The template to save
        """
        # Add to in-memory templates
        self.templates[template.template_id] = template
        
        # Save to file
        file_path = self.templates_dir / f"{template.template_id}.json"
        try:
            with open(file_path, "w") as f:
                json.dump(template.to_dict(), f, indent=2)
            logger.info(f"Saved template: {template.name}")
        except Exception as e:
            logger.error(f"Error saving template {template.name}: {str(e)}")
    
    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template.
        
        Args:
            template_id: ID of the template to delete
            
        Returns:
            True if the template was deleted, False otherwise
        """
        if template_id not in self.templates:
            return False
        
        # Remove from in-memory templates
        del self.templates[template_id]
        
        # Delete file
        file_path = self.templates_dir / f"{template_id}.json"
        try:
            if file_path.exists():
                os.remove(file_path)
            logger.info(f"Deleted template: {template_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting template {template_id}: {str(e)}")
            return False


# Create singleton instance
template_manager = SessionTemplateManager()
