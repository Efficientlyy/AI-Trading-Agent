#!/usr/bin/env python
"""
Conversational Interface and Dialogue Manager for System Overseer

This module implements a conversational interface and dialogue manager with
plugin support for natural language interaction with the System Overseer.
"""

import os
import re
import json
import time
import logging
import threading
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dialogue_manager")

class DialogueContext:
    """Context for a dialogue session"""
    
    def __init__(
        self,
        user_id: int,
        session_id: str = None,
        context_data: Dict[str, Any] = None,
        history: List[Dict[str, Any]] = None,
        created_at: float = None,
        last_activity: float = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize dialogue context
        
        Args:
            user_id: User ID
            session_id: Session ID (defaults to generated UUID)
            context_data: Context data
            history: Message history
            created_at: Creation timestamp
            last_activity: Last activity timestamp
            metadata: Additional metadata
        """
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        self.context_data = context_data or {}
        self.history = history or []
        self.created_at = created_at or time.time()
        self.last_activity = last_activity or time.time()
        self.metadata = metadata or {}
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def add_message(
        self,
        role: str,
        content: str,
        message_type: str = "text",
        metadata: Dict[str, Any] = None
    ):
        """Add message to history
        
        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
            message_type: Message type
            metadata: Additional metadata
        """
        self.history.append({
            "role": role,
            "content": content,
            "type": message_type,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        self.update_activity()
    
    def get_recent_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent message history
        
        Args:
            count: Maximum number of messages to return
            
        Returns:
            list: Recent message history
        """
        return self.history[-count:] if len(self.history) > count else self.history
    
    def get_formatted_history(self, count: int = 10) -> List[Dict[str, str]]:
        """Get formatted message history for LLM
        
        Args:
            count: Maximum number of messages to return
            
        Returns:
            list: Formatted message history
        """
        recent = self.get_recent_history(count)
        formatted = []
        
        for msg in recent:
            if msg["role"] in ["user", "assistant", "system"]:
                formatted.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return formatted
    
    def clear_history(self):
        """Clear message history"""
        self.history = []
    
    def set_context_value(self, key: str, value: Any):
        """Set context value
        
        Args:
            key: Context key
            value: Context value
        """
        self.context_data[key] = value
    
    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get context value
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Any: Context value
        """
        return self.context_data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dialogue context to dictionary
        
        Returns:
            dict: Dictionary representation of dialogue context
        """
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "context_data": self.context_data,
            "history": self.history,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueContext':
        """Create dialogue context from dictionary
        
        Args:
            data: Dictionary representation of dialogue context
            
        Returns:
            DialogueContext: Dialogue context object
        """
        return cls(
            user_id=data["user_id"],
            session_id=data.get("session_id"),
            context_data=data.get("context_data", {}),
            history=data.get("history", []),
            created_at=data.get("created_at"),
            last_activity=data.get("last_activity"),
            metadata=data.get("metadata", {})
        )


class UserProfile:
    """User profile with preferences and settings"""
    
    def __init__(
        self,
        user_id: int,
        preferences: Dict[str, Any] = None,
        settings: Dict[str, Any] = None,
        created_at: float = None,
        last_updated: float = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize user profile
        
        Args:
            user_id: User ID
            preferences: User preferences
            settings: User settings
            created_at: Creation timestamp
            last_updated: Last update timestamp
            metadata: Additional metadata
        """
        self.user_id = user_id
        self.preferences = preferences or {}
        self.settings = settings or {}
        self.created_at = created_at or time.time()
        self.last_updated = last_updated or time.time()
        self.metadata = metadata or {}
    
    def update_preference(self, key: str, value: Any):
        """Update user preference
        
        Args:
            key: Preference key
            value: Preference value
        """
        self.preferences[key] = value
        self.last_updated = time.time()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference
        
        Args:
            key: Preference key
            default: Default value if key not found
            
        Returns:
            Any: Preference value
        """
        return self.preferences.get(key, default)
    
    def update_setting(self, key: str, value: Any):
        """Update user setting
        
        Args:
            key: Setting key
            value: Setting value
        """
        self.settings[key] = value
        self.last_updated = time.time()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get user setting
        
        Args:
            key: Setting key
            default: Default value if key not found
            
        Returns:
            Any: Setting value
        """
        return self.settings.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user profile to dictionary
        
        Returns:
            dict: Dictionary representation of user profile
        """
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "settings": self.settings,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create user profile from dictionary
        
        Args:
            data: Dictionary representation of user profile
            
        Returns:
            UserProfile: User profile object
        """
        return cls(
            user_id=data["user_id"],
            preferences=data.get("preferences", {}),
            settings=data.get("settings", {}),
            created_at=data.get("created_at"),
            last_updated=data.get("last_updated"),
            metadata=data.get("metadata", {})
        )


class Intent:
    """Intent definition for natural language understanding"""
    
    def __init__(
        self,
        intent_id: str,
        patterns: List[str] = None,
        examples: List[str] = None,
        handler: Callable = None,
        description: str = None,
        plugin_id: str = None,
        enabled: bool = True,
        metadata: Dict[str, Any] = None
    ):
        """Initialize intent
        
        Args:
            intent_id: Intent ID
            patterns: Regex patterns for matching
            examples: Example phrases
            handler: Intent handler function
            description: Intent description
            plugin_id: Plugin ID
            enabled: Whether intent is enabled
            metadata: Additional metadata
        """
        self.intent_id = intent_id
        self.patterns = patterns or []
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in (patterns or [])]
        self.examples = examples or []
        self.handler = handler
        self.description = description
        self.plugin_id = plugin_id
        self.enabled = enabled
        self.metadata = metadata or {}
    
    def match(self, text: str) -> Optional[re.Match]:
        """Match text against patterns
        
        Args:
            text: Text to match
            
        Returns:
            re.Match: Match object if matched, None otherwise
        """
        if not self.enabled:
            return None
        
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                return match
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert intent to dictionary
        
        Returns:
            dict: Dictionary representation of intent
        """
        return {
            "intent_id": self.intent_id,
            "patterns": self.patterns,
            "examples": self.examples,
            "description": self.description,
            "plugin_id": self.plugin_id,
            "enabled": self.enabled,
            "metadata": self.metadata
        }


class ResponseTemplate:
    """Template for generating responses"""
    
    def __init__(
        self,
        template_id: str,
        template: str,
        variables: List[str] = None,
        description: str = None,
        plugin_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize response template
        
        Args:
            template_id: Template ID
            template: Template string
            variables: Template variables
            description: Template description
            plugin_id: Plugin ID
            metadata: Additional metadata
        """
        self.template_id = template_id
        self.template = template
        self.variables = variables or []
        self.description = description
        self.plugin_id = plugin_id
        self.metadata = metadata or {}
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render template with variables
        
        Args:
            variables: Template variables
            
        Returns:
            str: Rendered template
        """
        result = self.template
        
        # Simple variable substitution
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            result = result.replace(placeholder, str(var_value))
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response template to dictionary
        
        Returns:
            dict: Dictionary representation of response template
        """
        return {
            "template_id": self.template_id,
            "template": self.template,
            "variables": self.variables,
            "description": self.description,
            "plugin_id": self.plugin_id,
            "metadata": self.metadata
        }


class PersonalityProfile:
    """Personality profile for the assistant"""
    
    def __init__(
        self,
        profile_id: str,
        name: str,
        description: str,
        traits: Dict[str, float] = None,
        system_message: str = None,
        plugin_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize personality profile
        
        Args:
            profile_id: Profile ID
            name: Profile name
            description: Profile description
            traits: Personality traits (name -> value between 0.0 and 1.0)
            system_message: System message for LLM
            plugin_id: Plugin ID
            metadata: Additional metadata
        """
        self.profile_id = profile_id
        self.name = name
        self.description = description
        self.traits = traits or {}
        self.system_message = system_message
        self.plugin_id = plugin_id
        self.metadata = metadata or {}
    
    def get_trait(self, trait_name: str, default: float = 0.5) -> float:
        """Get personality trait value
        
        Args:
            trait_name: Trait name
            default: Default value if trait not found
            
        Returns:
            float: Trait value
        """
        return self.traits.get(trait_name, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert personality profile to dictionary
        
        Returns:
            dict: Dictionary representation of personality profile
        """
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "traits": self.traits,
            "system_message": self.system_message,
            "plugin_id": self.plugin_id,
            "metadata": self.metadata
        }


class DialoguePlugin:
    """Base class for dialogue plugins"""
    
    def __init__(
        self,
        plugin_id: str,
        name: str,
        version: str,
        description: str,
        dependencies: List[str] = None
    ):
        """Initialize dialogue plugin
        
        Args:
            plugin_id: Plugin ID
            name: Plugin name
            version: Plugin version
            description: Plugin description
            dependencies: Plugin dependencies
        """
        self.plugin_id = plugin_id
        self.name = name
        self.version = version
        self.description = description
        self.dependencies = dependencies or []
        
        # Core components will be injected during initialization
        self.dialogue_manager = None
        self.config_registry = None
        self.event_bus = None
        self.service_locator = None
        
        self.enabled = False
        logger.info(f"DialoguePlugin {self.plugin_id} instance created.")
    
    def initialize(
        self,
        dialogue_manager,
        config_registry=None,
        event_bus=None,
        service_locator=None
    ):
        """Initialize plugin
        
        Args:
            dialogue_manager: Dialogue manager
            config_registry: Configuration registry
            event_bus: Event bus
            service_locator: Service locator
        """
        self.dialogue_manager = dialogue_manager
        self.config_registry = config_registry
        self.event_bus = event_bus
        self.service_locator = service_locator
        
        logger.info(f"Initializing DialoguePlugin: {self.plugin_id}")
        
        # Register common enable/disable parameter
        if self.config_registry:
            self.config_registry.register_parameter(
                module_id=self.plugin_id,
                param_id="enabled",
                default_value=True,
                param_type=bool,
                description=f"Enable/disable the {self.name} plugin",
                group="dialogue_plugins"
            )
        
        # Register plugin-specific parameters
        self._register_parameters()
        
        # Register intents
        self._register_intents()
        
        # Register response templates
        self._register_response_templates()
        
        # Register personality profiles
        self._register_personality_profiles()
        
        # Subscribe to events
        self._subscribe_to_events()
        
        self.enabled = self._get_config("enabled", True)
        logger.info(f"DialoguePlugin {self.plugin_id} initialized. Enabled: {self.enabled}")
    
    def start(self):
        """Start plugin"""
        self.enabled = self._get_config("enabled", True)
        if self.enabled:
            logger.info(f"Starting DialoguePlugin: {self.plugin_id}")
            self._start_plugin()
        else:
            logger.info(f"DialoguePlugin {self.plugin_id} is disabled, not starting.")
    
    def stop(self):
        """Stop plugin"""
        logger.info(f"Stopping DialoguePlugin: {self.plugin_id}")
        self._stop_plugin()
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information
        
        Returns:
            dict: Plugin information
        """
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dependencies": self.dependencies,
            "enabled": self.is_enabled()
        }
    
    def is_enabled(self) -> bool:
        """Check if plugin is enabled
        
        Returns:
            bool: True if enabled, False otherwise
        """
        if self.config_registry:
            return self.config_registry.get_parameter(self.plugin_id, "enabled", True)
        return self.enabled
    
    def process_message(
        self,
        message: str,
        context: DialogueContext,
        user_profile: UserProfile
    ) -> Optional[str]:
        """Process user message
        
        Args:
            message: User message
            context: Dialogue context
            user_profile: User profile
            
        Returns:
            str: Response message or None
        """
        if not self.is_enabled():
            return None
        
        # Default implementation does nothing
        return None
    
    def _register_parameters(self):
        """Register plugin-specific parameters"""
        pass
    
    def _register_intents(self):
        """Register intents"""
        pass
    
    def _register_response_templates(self):
        """Register response templates"""
        pass
    
    def _register_personality_profiles(self):
        """Register personality profiles"""
        pass
    
    def _subscribe_to_events(self):
        """Subscribe to events"""
        pass
    
    def _start_plugin(self):
        """Start plugin-specific operations"""
        pass
    
    def _stop_plugin(self):
        """Stop plugin-specific operations"""
        pass
    
    def _get_config(self, param_id: str, default: Any = None) -> Any:
        """Get configuration parameter
        
        Args:
            param_id: Parameter ID
            default: Default value if parameter not found
            
        Returns:
            Any: Parameter value
        """
        if self.config_registry:
            return self.config_registry.get_parameter(self.plugin_id, param_id, default)
        return default
    
    def _publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish event
        
        Args:
            event_type: Event type
            data: Event data
        """
        if self.event_bus:
            self.event_bus.publish(
                event_type=event_type,
                data=data,
                publisher_id=self.plugin_id
            )
    
    def _get_llm_client(self):
        """Get LLM client
        
        Returns:
            LLMClient: LLM client or None if not available
        """
        if self.service_locator:
            try:
                return self.service_locator.get_service("llm_client")
            except Exception as e:
                logger.error(f"Error getting LLM client: {e}")
        return None


class DialogueManager:
    """Conversational interface and dialogue manager with plugin support"""
    
    def __init__(
        self,
        config_registry=None,
        event_bus=None,
        service_locator=None,
        session_timeout: int = 3600,
        persistence_dir: str = None
    ):
        """Initialize dialogue manager
        
        Args:
            config_registry: Configuration registry
            event_bus: Event bus
            service_locator: Service locator
            session_timeout: Session timeout in seconds
            persistence_dir: Directory for persistence files
        """
        self.config_registry = config_registry
        self.event_bus = event_bus
        self.service_locator = service_locator
        self.session_timeout = session_timeout
        self.persistence_dir = persistence_dir
        
        # Create persistence directory if specified
        if self.persistence_dir:
            os.makedirs(self.persistence_dir, exist_ok=True)
        
        # Dialogue contexts
        self.contexts = {}  # {session_id: DialogueContext}
        
        # User profiles
        self.user_profiles = {}  # {user_id: UserProfile}
        
        # Intents
        self.intents = {}  # {intent_id: Intent}
        
        # Response templates
        self.response_templates = {}  # {template_id: ResponseTemplate}
        
        # Personality profiles
        self.personality_profiles = {}  # {profile_id: PersonalityProfile}
        self.default_personality = None
        
        # Dialogue plugins
        self.plugins = {}  # {plugin_id: DialoguePlugin}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Initialize from config if provided
        if self.config_registry:
            self._init_from_config()
        
        # Load user profiles and contexts
        self._load_user_profiles()
        self._load_dialogue_contexts()
        
        # Start cleanup thread
        self.running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_thread,
            name="DialogueManager-Cleanup",
            daemon=True
        )
        self.cleanup_thread.start()
        
        logger.info("DialogueManager initialized")
    
    def _init_from_config(self):
        """Initialize from configuration registry"""
        if not self.config_registry:
            return
        
        try:
            # Get dialogue manager configuration
            self.session_timeout = self.config_registry.get_parameter(
                "dialogue_manager", "session_timeout", self.session_timeout)
            
            self.persistence_dir = self.config_registry.get_parameter(
                "dialogue_manager", "persistence_dir", self.persistence_dir)
            
            logger.info("DialogueManager initialized from configuration")
        except Exception as e:
            logger.error(f"Error initializing DialogueManager from configuration: {e}")
    
    def register_plugin(self, plugin: DialoguePlugin) -> bool:
        """Register dialogue plugin
        
        Args:
            plugin: Dialogue plugin
            
        Returns:
            bool: True if plugin was registered, False otherwise
        """
        with self.lock:
            # Check if plugin already exists
            if plugin.plugin_id in self.plugins:
                logger.warning(f"Plugin {plugin.plugin_id} already registered")
                return False
            
            # Initialize plugin
            plugin.initialize(
                dialogue_manager=self,
                config_registry=self.config_registry,
                event_bus=self.event_bus,
                service_locator=self.service_locator
            )
            
            # Store plugin
            self.plugins[plugin.plugin_id] = plugin
            
            # Start plugin
            plugin.start()
            
            logger.info(f"Plugin {plugin.plugin_id} registered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="dialogue_manager.plugin_registered",
                    data={
                        "plugin_id": plugin.plugin_id,
                        "name": plugin.name,
                        "version": plugin.version
                    }
                )
            
            return True
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister dialogue plugin
        
        Args:
            plugin_id: Plugin ID
            
        Returns:
            bool: True if plugin was unregistered, False otherwise
        """
        with self.lock:
            # Check if plugin exists
            if plugin_id not in self.plugins:
                logger.warning(f"Plugin {plugin_id} not registered")
                return False
            
            # Get plugin
            plugin = self.plugins[plugin_id]
            
            # Stop plugin
            plugin.stop()
            
            # Remove plugin
            del self.plugins[plugin_id]
            
            logger.info(f"Plugin {plugin_id} unregistered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="dialogue_manager.plugin_unregistered",
                    data={
                        "plugin_id": plugin_id
                    }
                )
            
            return True
    
    def register_intent(
        self,
        intent_id: str,
        patterns: List[str],
        handler: Callable,
        description: str = None,
        examples: List[str] = None,
        plugin_id: str = None,
        enabled: bool = True,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Register intent
        
        Args:
            intent_id: Intent ID
            patterns: Regex patterns for matching
            handler: Intent handler function
            description: Intent description
            examples: Example phrases
            plugin_id: Plugin ID
            enabled: Whether intent is enabled
            metadata: Additional metadata
            
        Returns:
            bool: True if intent was registered, False otherwise
        """
        with self.lock:
            # Check if intent already exists
            if intent_id in self.intents:
                logger.warning(f"Intent {intent_id} already registered")
                return False
            
            # Create intent
            intent = Intent(
                intent_id=intent_id,
                patterns=patterns,
                handler=handler,
                description=description,
                examples=examples,
                plugin_id=plugin_id,
                enabled=enabled,
                metadata=metadata
            )
            
            # Store intent
            self.intents[intent_id] = intent
            
            logger.info(f"Intent {intent_id} registered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="dialogue_manager.intent_registered",
                    data={
                        "intent_id": intent_id,
                        "description": description,
                        "plugin_id": plugin_id
                    }
                )
            
            return True
    
    def unregister_intent(self, intent_id: str) -> bool:
        """Unregister intent
        
        Args:
            intent_id: Intent ID
            
        Returns:
            bool: True if intent was unregistered, False otherwise
        """
        with self.lock:
            # Check if intent exists
            if intent_id not in self.intents:
                logger.warning(f"Intent {intent_id} not registered")
                return False
            
            # Get intent
            intent = self.intents[intent_id]
            
            # Remove intent
            del self.intents[intent_id]
            
            logger.info(f"Intent {intent_id} unregistered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="dialogue_manager.intent_unregistered",
                    data={
                        "intent_id": intent_id,
                        "plugin_id": intent.plugin_id
                    }
                )
            
            return True
    
    def register_response_template(
        self,
        template_id: str,
        template: str,
        variables: List[str] = None,
        description: str = None,
        plugin_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Register response template
        
        Args:
            template_id: Template ID
            template: Template string
            variables: Template variables
            description: Template description
            plugin_id: Plugin ID
            metadata: Additional metadata
            
        Returns:
            bool: True if template was registered, False otherwise
        """
        with self.lock:
            # Check if template already exists
            if template_id in self.response_templates:
                logger.warning(f"Response template {template_id} already registered")
                return False
            
            # Create response template
            template_obj = ResponseTemplate(
                template_id=template_id,
                template=template,
                variables=variables,
                description=description,
                plugin_id=plugin_id,
                metadata=metadata
            )
            
            # Store response template
            self.response_templates[template_id] = template_obj
            
            logger.info(f"Response template {template_id} registered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="dialogue_manager.response_template_registered",
                    data={
                        "template_id": template_id,
                        "description": description,
                        "plugin_id": plugin_id
                    }
                )
            
            return True
    
    def unregister_response_template(self, template_id: str) -> bool:
        """Unregister response template
        
        Args:
            template_id: Template ID
            
        Returns:
            bool: True if template was unregistered, False otherwise
        """
        with self.lock:
            # Check if template exists
            if template_id not in self.response_templates:
                logger.warning(f"Response template {template_id} not registered")
                return False
            
            # Get template
            template = self.response_templates[template_id]
            
            # Remove template
            del self.response_templates[template_id]
            
            logger.info(f"Response template {template_id} unregistered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="dialogue_manager.response_template_unregistered",
                    data={
                        "template_id": template_id,
                        "plugin_id": template.plugin_id
                    }
                )
            
            return True
    
    def register_personality_profile(
        self,
        profile_id: str,
        name: str,
        description: str,
        traits: Dict[str, float] = None,
        system_message: str = None,
        plugin_id: str = None,
        metadata: Dict[str, Any] = None,
        set_as_default: bool = False
    ) -> bool:
        """Register personality profile
        
        Args:
            profile_id: Profile ID
            name: Profile name
            description: Profile description
            traits: Personality traits
            system_message: System message for LLM
            plugin_id: Plugin ID
            metadata: Additional metadata
            set_as_default: Whether to set as default personality
            
        Returns:
            bool: True if profile was registered, False otherwise
        """
        with self.lock:
            # Check if profile already exists
            if profile_id in self.personality_profiles:
                logger.warning(f"Personality profile {profile_id} already registered")
                return False
            
            # Create personality profile
            profile = PersonalityProfile(
                profile_id=profile_id,
                name=name,
                description=description,
                traits=traits,
                system_message=system_message,
                plugin_id=plugin_id,
                metadata=metadata
            )
            
            # Store personality profile
            self.personality_profiles[profile_id] = profile
            
            # Set as default if requested or if first profile
            if set_as_default or not self.default_personality:
                self.default_personality = profile
            
            logger.info(f"Personality profile {profile_id} registered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="dialogue_manager.personality_profile_registered",
                    data={
                        "profile_id": profile_id,
                        "name": name,
                        "description": description,
                        "plugin_id": plugin_id,
                        "is_default": profile == self.default_personality
                    }
                )
            
            return True
    
    def unregister_personality_profile(self, profile_id: str) -> bool:
        """Unregister personality profile
        
        Args:
            profile_id: Profile ID
            
        Returns:
            bool: True if profile was unregistered, False otherwise
        """
        with self.lock:
            # Check if profile exists
            if profile_id not in self.personality_profiles:
                logger.warning(f"Personality profile {profile_id} not registered")
                return False
            
            # Get profile
            profile = self.personality_profiles[profile_id]
            
            # Check if default profile
            is_default = profile == self.default_personality
            
            # Remove profile
            del self.personality_profiles[profile_id]
            
            # Update default profile if needed
            if is_default and self.personality_profiles:
                self.default_personality = next(iter(self.personality_profiles.values()))
            elif is_default:
                self.default_personality = None
            
            logger.info(f"Personality profile {profile_id} unregistered")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="dialogue_manager.personality_profile_unregistered",
                    data={
                        "profile_id": profile_id,
                        "plugin_id": profile.plugin_id,
                        "was_default": is_default
                    }
                )
            
            return True
    
    def set_default_personality(self, profile_id: str) -> bool:
        """Set default personality profile
        
        Args:
            profile_id: Profile ID
            
        Returns:
            bool: True if default was set, False otherwise
        """
        with self.lock:
            # Check if profile exists
            if profile_id not in self.personality_profiles:
                logger.warning(f"Personality profile {profile_id} not registered")
                return False
            
            # Set default profile
            self.default_personality = self.personality_profiles[profile_id]
            
            logger.info(f"Default personality profile set to {profile_id}")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="dialogue_manager.default_personality_changed",
                    data={
                        "profile_id": profile_id
                    }
                )
            
            return True
    
    def get_or_create_context(
        self,
        user_id: int,
        session_id: str = None
    ) -> DialogueContext:
        """Get or create dialogue context
        
        Args:
            user_id: User ID
            session_id: Session ID (optional)
            
        Returns:
            DialogueContext: Dialogue context
        """
        with self.lock:
            # If session ID provided, try to get existing context
            if session_id and session_id in self.contexts:
                context = self.contexts[session_id]
                context.update_activity()
                return context
            
            # Create new context
            context = DialogueContext(
                user_id=user_id,
                session_id=session_id
            )
            
            # Store context
            self.contexts[context.session_id] = context
            
            logger.info(f"Created dialogue context for user {user_id}, session {context.session_id}")
            
            return context
    
    def get_or_create_user_profile(self, user_id: int) -> UserProfile:
        """Get or create user profile
        
        Args:
            user_id: User ID
            
        Returns:
            UserProfile: User profile
        """
        with self.lock:
            # Try to get existing profile
            if user_id in self.user_profiles:
                return self.user_profiles[user_id]
            
            # Create new profile
            profile = UserProfile(user_id=user_id)
            
            # Store profile
            self.user_profiles[user_id] = profile
            
            logger.info(f"Created user profile for user {user_id}")
            
            return profile
    
    def process_message(
        self,
        user_id: int,
        message: str,
        session_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Process user message
        
        Args:
            user_id: User ID
            message: User message
            session_id: Session ID (optional)
            metadata: Additional metadata
            
        Returns:
            str: Response message
        """
        # Get or create dialogue context and user profile
        context = self.get_or_create_context(user_id, session_id)
        profile = self.get_or_create_user_profile(user_id)
        
        # Add message to context
        context.add_message(
            role="user",
            content=message,
            metadata=metadata
        )
        
        # Publish event if event bus is available
        if self.event_bus:
            self.event_bus.publish(
                event_type="dialogue_manager.message_received",
                data={
                    "user_id": user_id,
                    "session_id": context.session_id,
                    "message": message,
                    "metadata": metadata
                }
            )
        
        # Try to match intents
        intent_match = self._match_intent(message)
        if intent_match:
            intent, match = intent_match
            logger.info(f"Message matched intent: {intent.intent_id}")
            
            # Publish event if event bus is available
            if self.event_bus:
                self.event_bus.publish(
                    event_type="dialogue_manager.intent_matched",
                    data={
                        "intent_id": intent.intent_id,
                        "user_id": user_id,
                        "session_id": context.session_id,
                        "message": message
                    }
                )
            
            try:
                # Call intent handler
                response = intent.handler(message, match, context, profile)
                
                if response:
                    # Add response to context
                    context.add_message(
                        role="assistant",
                        content=response,
                        metadata={"intent_id": intent.intent_id}
                    )
                    
                    # Save context and profile
                    self._save_dialogue_context(context)
                    self._save_user_profile(profile)
                    
                    return response
            
            except Exception as e:
                logger.error(f"Error handling intent {intent.intent_id}: {e}")
        
        # Try plugins
        for plugin_id, plugin in self.plugins.items():
            if not plugin.is_enabled():
                continue
            
            try:
                response = plugin.process_message(message, context, profile)
                
                if response:
                    # Add response to context
                    context.add_message(
                        role="assistant",
                        content=response,
                        metadata={"plugin_id": plugin_id}
                    )
                    
                    # Save context and profile
                    self._save_dialogue_context(context)
                    self._save_user_profile(profile)
                    
                    return response
            
            except Exception as e:
                logger.error(f"Error in plugin {plugin_id}: {e}")
        
        # Fall back to LLM
        response = self._generate_llm_response(message, context, profile)
        
        # Add response to context
        context.add_message(
            role="assistant",
            content=response,
            metadata={"source": "llm"}
        )
        
        # Save context and profile
        self._save_dialogue_context(context)
        self._save_user_profile(profile)
        
        return response
    
    def _match_intent(self, message: str) -> Optional[Tuple[Intent, re.Match]]:
        """Match message against intents
        
        Args:
            message: User message
            
        Returns:
            tuple: (Intent, Match) if matched, None otherwise
        """
        for intent_id, intent in self.intents.items():
            match = intent.match(message)
            if match:
                return (intent, match)
        
        return None
    
    def _generate_llm_response(
        self,
        message: str,
        context: DialogueContext,
        profile: UserProfile
    ) -> str:
        """Generate response using LLM
        
        Args:
            message: User message
            context: Dialogue context
            profile: User profile
            
        Returns:
            str: Generated response
        """
        llm_client = self._get_llm_client()
        if not llm_client:
            return "I'm sorry, I'm having trouble understanding. Could you try again?"
        
        try:
            # Get personality profile
            personality = self._get_personality_for_user(profile)
            
            # Get system message
            system_message = personality.system_message if personality else None
            if not system_message:
                system_message = self._get_default_system_message()
            
            # Format conversation history for LLM
            messages = [{"role": "system", "content": system_message}]
            messages.extend(context.get_formatted_history(10))
            
            # Generate response
            response = llm_client.generate(messages=messages)
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I'm sorry, I'm having trouble responding right now. Could you try again later?"
    
    def _get_personality_for_user(self, profile: UserProfile) -> PersonalityProfile:
        """Get personality profile for user
        
        Args:
            profile: User profile
            
        Returns:
            PersonalityProfile: Personality profile
        """
        # Check if user has preferred personality
        preferred_profile_id = profile.get_preference("personality_profile")
        if preferred_profile_id and preferred_profile_id in self.personality_profiles:
            return self.personality_profiles[preferred_profile_id]
        
        # Fall back to default personality
        return self.default_personality
    
    def _get_default_system_message(self) -> str:
        """Get default system message
        
        Returns:
            str: Default system message
        """
        return (
            "You are an AI assistant for a trading system. "
            "You can help users with commands, answer questions about the system, "
            "and provide information about trading. "
            "Be concise, helpful, and informative."
        )
    
    def _get_llm_client(self):
        """Get LLM client
        
        Returns:
            LLMClient: LLM client or None if not available
        """
        if self.service_locator:
            try:
                return self.service_locator.get_service("llm_client")
            except Exception as e:
                logger.error(f"Error getting LLM client: {e}")
        return None
    
    def _save_dialogue_context(self, context: DialogueContext):
        """Save dialogue context
        
        Args:
            context: Dialogue context
        """
        if not self.persistence_dir:
            return
        
        try:
            # Create contexts directory if needed
            contexts_dir = os.path.join(self.persistence_dir, "contexts")
            os.makedirs(contexts_dir, exist_ok=True)
            
            # Save context to file
            file_path = os.path.join(contexts_dir, f"{context.session_id}.json")
            with open(file_path, "w") as f:
                json.dump(context.to_dict(), f)
        
        except Exception as e:
            logger.error(f"Error saving dialogue context: {e}")
    
    def _save_user_profile(self, profile: UserProfile):
        """Save user profile
        
        Args:
            profile: User profile
        """
        if not self.persistence_dir:
            return
        
        try:
            # Create profiles directory if needed
            profiles_dir = os.path.join(self.persistence_dir, "profiles")
            os.makedirs(profiles_dir, exist_ok=True)
            
            # Save profile to file
            file_path = os.path.join(profiles_dir, f"{profile.user_id}.json")
            with open(file_path, "w") as f:
                json.dump(profile.to_dict(), f)
        
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
    
    def _load_dialogue_contexts(self):
        """Load dialogue contexts from persistence directory"""
        if not self.persistence_dir:
            return
        
        try:
            # Check if contexts directory exists
            contexts_dir = os.path.join(self.persistence_dir, "contexts")
            if not os.path.exists(contexts_dir):
                return
            
            # Load contexts from files
            for filename in os.listdir(contexts_dir):
                if not filename.endswith(".json"):
                    continue
                
                try:
                    file_path = os.path.join(contexts_dir, filename)
                    with open(file_path, "r") as f:
                        context_dict = json.load(f)
                    
                    # Create context object
                    context = DialogueContext.from_dict(context_dict)
                    
                    # Check if context is expired
                    if time.time() - context.last_activity > self.session_timeout:
                        # Delete expired context file
                        os.remove(file_path)
                        continue
                    
                    # Store context
                    self.contexts[context.session_id] = context
                
                except Exception as e:
                    logger.error(f"Error loading dialogue context from {filename}: {e}")
            
            logger.info(f"Loaded {len(self.contexts)} dialogue contexts")
        
        except Exception as e:
            logger.error(f"Error loading dialogue contexts: {e}")
    
    def _load_user_profiles(self):
        """Load user profiles from persistence directory"""
        if not self.persistence_dir:
            return
        
        try:
            # Check if profiles directory exists
            profiles_dir = os.path.join(self.persistence_dir, "profiles")
            if not os.path.exists(profiles_dir):
                return
            
            # Load profiles from files
            for filename in os.listdir(profiles_dir):
                if not filename.endswith(".json"):
                    continue
                
                try:
                    file_path = os.path.join(profiles_dir, filename)
                    with open(file_path, "r") as f:
                        profile_dict = json.load(f)
                    
                    # Create profile object
                    profile = UserProfile.from_dict(profile_dict)
                    
                    # Store profile
                    self.user_profiles[profile.user_id] = profile
                
                except Exception as e:
                    logger.error(f"Error loading user profile from {filename}: {e}")
            
            logger.info(f"Loaded {len(self.user_profiles)} user profiles")
        
        except Exception as e:
            logger.error(f"Error loading user profiles: {e}")
    
    def _cleanup_thread(self):
        """Cleanup thread for expired contexts"""
        while self.running:
            try:
                # Sleep for a while
                time.sleep(60)
                
                # Find expired contexts
                expired_sessions = []
                
                with self.lock:
                    current_time = time.time()
                    for session_id, context in self.contexts.items():
                        if current_time - context.last_activity > self.session_timeout:
                            expired_sessions.append(session_id)
                
                # Remove expired contexts
                for session_id in expired_sessions:
                    with self.lock:
                        if session_id in self.contexts:
                            del self.contexts[session_id]
                    
                    # Delete context file if persistence enabled
                    if self.persistence_dir:
                        try:
                            file_path = os.path.join(self.persistence_dir, "contexts", f"{session_id}.json")
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            logger.error(f"Error deleting expired context file: {e}")
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired dialogue contexts")
            
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
    
    def stop(self):
        """Stop dialogue manager"""
        logger.info("Stopping DialogueManager")
        
        # Stop plugins
        for plugin_id, plugin in list(self.plugins.items()):
            try:
                plugin.stop()
            except Exception as e:
                logger.error(f"Error stopping plugin {plugin_id}: {e}")
        
        # Stop cleanup thread
        self.running = False
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1.0)
        
        # Save all contexts and profiles
        for context in self.contexts.values():
            self._save_dialogue_context(context)
        
        for profile in self.user_profiles.values():
            self._save_user_profile(profile)
        
        logger.info("DialogueManager stopped")


# Example usage
if __name__ == "__main__":
    # Create dialogue manager
    dialogue_manager = DialogueManager(
        session_timeout=3600,
        persistence_dir="./dialogue_data"
    )
    
    # Define intent handler
    def greeting_handler(message, match, context, profile):
        return "Hello! How can I help you with your trading today?"
    
    # Register intent
    dialogue_manager.register_intent(
        intent_id="greeting",
        patterns=[r"^(hi|hello|hey|greetings).*$"],
        handler=greeting_handler,
        description="Greeting intent",
        examples=["hi", "hello", "hey there"]
    )
    
    # Register response template
    dialogue_manager.register_response_template(
        template_id="market_status",
        template="The market is currently {status}. Bitcoin is trading at {btc_price} and Ethereum at {eth_price}.",
        variables=["status", "btc_price", "eth_price"],
        description="Market status template"
    )
    
    # Register personality profile
    dialogue_manager.register_personality_profile(
        profile_id="professional",
        name="Professional",
        description="Professional and formal personality",
        traits={
            "formality": 0.9,
            "friendliness": 0.5,
            "verbosity": 0.3
        },
        system_message=(
            "You are a professional trading assistant. "
            "Provide concise, accurate information about markets and trading. "
            "Use formal language and avoid unnecessary small talk."
        ),
        set_as_default=True
    )
    
    # Example plugin
    class MarketInfoPlugin(DialoguePlugin):
        def __init__(self):
            super().__init__(
                plugin_id="market_info_plugin",
                name="Market Information Plugin",
                version="1.0.0",
                description="Provides market information"
            )
        
        def _register_intents(self):
            self.dialogue_manager.register_intent(
                intent_id="market_status",
                patterns=[r"(what('s|s| is) the market (status|doing|like)|how('s|s| is) the market)"],
                handler=self.handle_market_status,
                description="Market status intent",
                examples=["what's the market status", "how is the market doing"],
                plugin_id=self.plugin_id
            )
        
        def handle_market_status(self, message, match, context, profile):
            # In a real implementation, this would fetch actual market data
            template = self.dialogue_manager.response_templates.get("market_status")
            if template:
                return template.render({
                    "status": "bullish",
                    "btc_price": "$50,000",
                    "eth_price": "$3,000"
                })
            return "The market is currently bullish. Bitcoin is trading at $50,000 and Ethereum at $3,000."
    
    # Register plugin
    market_plugin = MarketInfoPlugin()
    dialogue_manager.register_plugin(market_plugin)
    
    # Process some messages
    user_id = 123
    print(dialogue_manager.process_message(user_id, "Hello there!"))
    print(dialogue_manager.process_message(user_id, "How is the market doing?"))
    
    # Stop dialogue manager
    dialogue_manager.stop()
