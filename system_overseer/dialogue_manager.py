#!/usr/bin/env python
"""
Dialogue Manager for System Overseer

This module implements the Dialogue Manager component that handles conversational
context, message flow, and integration with the LLM client for natural language
interaction with the System Overseer.
"""

import os
import sys
import json
import time
import logging
import threading
import queue
import uuid
import copy
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Type
from pathlib import Path
from datetime import datetime

# Import LLM client
from .llm_client import LLMClient, LLMMessage, LLMResponse, LLMFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_overseer.dialogue_manager")


class DialogueContext:
    """Context for a dialogue session."""
    
    def __init__(
        self,
        context_id: str = None,
        user_id: str = None,
        max_history: int = 50,
        max_tokens: int = 8000,
        system_prompt: str = None
    ):
        """Initialize dialogue context.
        
        Args:
            context_id: Unique context identifier
            user_id: User identifier
            max_history: Maximum number of messages to keep in history
            max_tokens: Maximum number of tokens to keep in history
            system_prompt: System prompt for the conversation
        """
        self.context_id = context_id or str(uuid.uuid4())
        self.user_id = user_id
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.messages = []
        self.metadata = {
            "created_at": time.time(),
            "last_updated": time.time(),
            "message_count": 0,
            "token_count": 0
        }
        
        # Add system prompt if provided
        if system_prompt:
            self.add_message(LLMMessage(
                role="system",
                content=system_prompt
            ))
    
    def add_message(self, message: LLMMessage) -> bool:
        """Add message to context.
        
        Args:
            message: Message to add
            
        Returns:
            bool: True if message was added
        """
        # Add message
        self.messages.append(message)
        
        # Update metadata
        self.metadata["last_updated"] = time.time()
        self.metadata["message_count"] += 1
        
        # Return success
        return True
    
    def get_messages(self, max_tokens: int = None) -> List[LLMMessage]:
        """Get messages from context.
        
        Args:
            max_tokens: Maximum number of tokens to include
            
        Returns:
            list: List of messages
        """
        # If no token limit, return all messages
        if max_tokens is None:
            return self.messages.copy()
        
        # Get system messages (always include)
        system_messages = [m for m in self.messages if m.role == "system"]
        
        # Get non-system messages
        other_messages = [m for m in self.messages if m.role != "system"]
        
        # Start with system messages
        result = system_messages.copy()
        token_count = sum(len(m.content) // 4 for m in result)  # Rough estimate
        
        # Add other messages from newest to oldest until token limit
        for message in reversed(other_messages):
            # Estimate tokens for this message
            message_tokens = len(message.content) // 4
            
            # Check if adding this message would exceed token limit
            if token_count + message_tokens > max_tokens:
                break
            
            # Add message to result
            result.insert(len(system_messages), message)
            token_count += message_tokens
        
        return result
    
    def clear_messages(self) -> None:
        """Clear all messages except system messages."""
        # Keep system messages
        system_messages = [m for m in self.messages if m.role == "system"]
        
        # Reset messages
        self.messages = system_messages
        
        # Update metadata
        self.metadata["last_updated"] = time.time()
        self.metadata["message_count"] = len(system_messages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary.
        
        Returns:
            dict: Context as dictionary
        """
        return {
            "context_id": self.context_id,
            "user_id": self.user_id,
            "max_history": self.max_history,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "name": m.name,
                    "message_id": m.message_id,
                    "timestamp": getattr(m, "timestamp", time.time())
                }
                for m in self.messages
            ],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueContext':
        """Create context from dictionary.
        
        Args:
            data: Context dictionary
            
        Returns:
            DialogueContext: Context instance
        """
        context = cls(
            context_id=data.get("context_id"),
            user_id=data.get("user_id"),
            max_history=data.get("max_history", 50),
            max_tokens=data.get("max_tokens", 8000)
        )
        
        # Clear default system message
        context.messages = []
        
        # Add messages
        for msg_data in data.get("messages", []):
            message = LLMMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                name=msg_data.get("name"),
                message_id=msg_data.get("message_id")
            )
            
            # Set timestamp if available
            if "timestamp" in msg_data:
                message.timestamp = msg_data["timestamp"]
            
            context.messages.append(message)
        
        # Set metadata
        if "metadata" in data:
            context.metadata = data["metadata"]
        
        return context


class DialogueIntent:
    """Intent detected in user message."""
    
    def __init__(
        self,
        intent_type: str,
        confidence: float,
        parameters: Dict[str, Any] = None,
        raw_data: Dict[str, Any] = None
    ):
        """Initialize dialogue intent.
        
        Args:
            intent_type: Intent type identifier
            confidence: Confidence score (0-1)
            parameters: Intent parameters
            raw_data: Raw intent data
        """
        self.intent_type = intent_type
        self.confidence = confidence
        self.parameters = parameters or {}
        self.raw_data = raw_data or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert intent to dictionary.
        
        Returns:
            dict: Intent as dictionary
        """
        return {
            "intent_type": self.intent_type,
            "confidence": self.confidence,
            "parameters": self.parameters,
            "raw_data": self.raw_data,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueIntent':
        """Create intent from dictionary.
        
        Args:
            data: Intent dictionary
            
        Returns:
            DialogueIntent: Intent instance
        """
        intent = cls(
            intent_type=data["intent_type"],
            confidence=data["confidence"],
            parameters=data.get("parameters", {}),
            raw_data=data.get("raw_data", {})
        )
        
        if "timestamp" in data:
            intent.timestamp = data["timestamp"]
        
        return intent


class DialogueMemory:
    """Memory for dialogue sessions."""
    
    def __init__(
        self,
        memory_file: str = None,
        auto_save: bool = True,
        max_contexts: int = 100
    ):
        """Initialize dialogue memory.
        
        Args:
            memory_file: File path for persistent storage
            auto_save: Whether to automatically save changes
            max_contexts: Maximum number of contexts to keep
        """
        self.memory_file = memory_file
        self.auto_save = auto_save
        self.max_contexts = max_contexts
        self.contexts = {}  # context_id -> DialogueContext
        self.user_contexts = {}  # user_id -> [context_id]
        
        # Load existing memory if file exists
        if memory_file and os.path.exists(memory_file):
            self.load()
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def create_context(
        self,
        user_id: str = None,
        system_prompt: str = None,
        context_id: str = None,
        max_history: int = 50,
        max_tokens: int = 8000
    ) -> DialogueContext:
        """Create new dialogue context.
        
        Args:
            user_id: User identifier
            system_prompt: System prompt for the conversation
            context_id: Optional context identifier
            max_history: Maximum number of messages to keep in history
            max_tokens: Maximum number of tokens to keep in history
            
        Returns:
            DialogueContext: New context
        """
        with self.lock:
            # Create context
            context = DialogueContext(
                context_id=context_id,
                user_id=user_id,
                max_history=max_history,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
            
            # Store context
            self.contexts[context.context_id] = context
            
            # Associate with user
            if user_id:
                if user_id not in self.user_contexts:
                    self.user_contexts[user_id] = []
                
                self.user_contexts[user_id].append(context.context_id)
            
            # Save if auto-save enabled
            if self.auto_save:
                self.save()
            
            return context
    
    def get_context(self, context_id: str) -> Optional[DialogueContext]:
        """Get dialogue context by ID.
        
        Args:
            context_id: Context identifier
            
        Returns:
            DialogueContext: Context instance or None if not found
        """
        return self.contexts.get(context_id)
    
    def get_user_contexts(self, user_id: str) -> List[DialogueContext]:
        """Get all contexts for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            list: List of contexts
        """
        context_ids = self.user_contexts.get(user_id, [])
        return [self.contexts[cid] for cid in context_ids if cid in self.contexts]
    
    def get_latest_user_context(self, user_id: str) -> Optional[DialogueContext]:
        """Get latest context for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            DialogueContext: Context instance or None if not found
        """
        contexts = self.get_user_contexts(user_id)
        if not contexts:
            return None
        
        # Sort by last updated time
        contexts.sort(key=lambda c: c.metadata["last_updated"], reverse=True)
        return contexts[0]
    
    def add_message(
        self,
        context_id: str,
        message: LLMMessage
    ) -> bool:
        """Add message to context.
        
        Args:
            context_id: Context identifier
            message: Message to add
            
        Returns:
            bool: True if message was added
        """
        with self.lock:
            # Get context
            context = self.get_context(context_id)
            if not context:
                logger.error(f"Context not found: {context_id}")
                return False
            
            # Add message
            result = context.add_message(message)
            
            # Save if auto-save enabled
            if result and self.auto_save:
                self.save()
            
            return result
    
    def delete_context(self, context_id: str) -> bool:
        """Delete dialogue context.
        
        Args:
            context_id: Context identifier
            
        Returns:
            bool: True if context was deleted
        """
        with self.lock:
            # Check if context exists
            if context_id not in self.contexts:
                logger.warning(f"Context not found: {context_id}")
                return False
            
            # Get user ID
            user_id = self.contexts[context_id].user_id
            
            # Remove from user contexts
            if user_id and user_id in self.user_contexts:
                if context_id in self.user_contexts[user_id]:
                    self.user_contexts[user_id].remove(context_id)
            
            # Remove context
            del self.contexts[context_id]
            
            # Save if auto-save enabled
            if self.auto_save:
                self.save()
            
            return True
    
    def clear_user_contexts(self, user_id: str) -> bool:
        """Clear all contexts for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if contexts were cleared
        """
        with self.lock:
            # Check if user has contexts
            if user_id not in self.user_contexts:
                return True
            
            # Get context IDs
            context_ids = self.user_contexts[user_id].copy()
            
            # Delete each context
            for context_id in context_ids:
                self.delete_context(context_id)
            
            # Clear user contexts
            self.user_contexts[user_id] = []
            
            # Save if auto-save enabled
            if self.auto_save:
                self.save()
            
            return True
    
    def save(self) -> bool:
        """Save memory to file.
        
        Returns:
            bool: True if save successful
        """
        if not self.memory_file:
            logger.warning("No memory file specified")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # Prepare data
            data = {
                "contexts": {
                    context_id: context.to_dict()
                    for context_id, context in self.contexts.items()
                },
                "user_contexts": self.user_contexts,
                "timestamp": time.time()
            }
            
            # Write to file
            with open(self.memory_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Memory saved to {self.memory_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False
    
    def load(self) -> bool:
        """Load memory from file.
        
        Returns:
            bool: True if load successful
        """
        if not self.memory_file:
            logger.warning("No memory file specified")
            return False
        
        if not os.path.exists(self.memory_file):
            logger.warning(f"Memory file not found: {self.memory_file}")
            return False
        
        try:
            # Read from file
            with open(self.memory_file, "r") as f:
                data = json.load(f)
            
            # Load contexts
            contexts = {}
            for context_id, context_data in data.get("contexts", {}).items():
                contexts[context_id] = DialogueContext.from_dict(context_data)
            
            # Load user contexts
            user_contexts = data.get("user_contexts", {})
            
            # Update memory
            with self.lock:
                self.contexts = contexts
                self.user_contexts = user_contexts
            
            logger.debug(f"Memory loaded from {self.memory_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return False
    
    def cleanup(self) -> int:
        """Clean up old contexts.
        
        Returns:
            int: Number of contexts removed
        """
        with self.lock:
            # Check if we need to clean up
            if len(self.contexts) <= self.max_contexts:
                return 0
            
            # Sort contexts by last updated time
            sorted_contexts = sorted(
                self.contexts.items(),
                key=lambda x: x[1].metadata["last_updated"]
            )
            
            # Calculate how many to remove
            to_remove = len(self.contexts) - self.max_contexts
            
            # Remove oldest contexts
            removed = 0
            for context_id, _ in sorted_contexts[:to_remove]:
                if self.delete_context(context_id):
                    removed += 1
            
            # Save if auto-save enabled
            if removed > 0 and self.auto_save:
                self.save()
            
            return removed


class PersonalityProfile:
    """Personality profile for dialogue system."""
    
    def __init__(
        self,
        profile_id: str,
        name: str,
        system_prompt: str,
        description: str = None,
        parameters: Dict[str, Any] = None
    ):
        """Initialize personality profile.
        
        Args:
            profile_id: Unique profile identifier
            name: Profile name
            system_prompt: System prompt for the personality
            description: Profile description
            parameters: Personality parameters
        """
        self.profile_id = profile_id
        self.name = name
        self.system_prompt = system_prompt
        self.description = description or ""
        self.parameters = parameters or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary.
        
        Returns:
            dict: Profile as dictionary
        """
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "description": self.description,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityProfile':
        """Create profile from dictionary.
        
        Args:
            data: Profile dictionary
            
        Returns:
            PersonalityProfile: Profile instance
        """
        return cls(
            profile_id=data["profile_id"],
            name=data["name"],
            system_prompt=data["system_prompt"],
            description=data.get("description", ""),
            parameters=data.get("parameters", {})
        )


class PersonalityManager:
    """Manager for personality profiles."""
    
    def __init__(
        self,
        profiles_file: str = None,
        auto_save: bool = True
    ):
        """Initialize personality manager.
        
        Args:
            profiles_file: File path for persistent storage
            auto_save: Whether to automatically save changes
        """
        self.profiles_file = profiles_file
        self.auto_save = auto_save
        self.profiles = {}  # profile_id -> PersonalityProfile
        self.default_profile_id = None
        
        # Load existing profiles if file exists
        if profiles_file and os.path.exists(profiles_file):
            self.load()
        
        # Create default profile if none exists
        if not self.profiles:
            self._create_default_profiles()
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def _create_default_profiles(self) -> None:
        """Create default personality profiles."""
        # Create default profile
        default_profile = PersonalityProfile(
            profile_id="default",
            name="Default",
            system_prompt=(
                "You are the System Overseer, an AI assistant that helps manage and "
                "monitor the Trading Agent system. You provide insights, answer "
                "questions, and help control the system through natural conversation. "
                "Be helpful, informative, and concise in your responses."
            ),
            description="Default personality profile",
            parameters={
                "verbosity": 0.7,  # 0-1, higher is more verbose
                "formality": 0.5,  # 0-1, higher is more formal
                "creativity": 0.5,  # 0-1, higher is more creative
                "empathy": 0.7,    # 0-1, higher is more empathetic
                "humor": 0.3       # 0-1, higher is more humorous
            }
        )
        
        # Create professional profile
        professional_profile = PersonalityProfile(
            profile_id="professional",
            name="Professional",
            system_prompt=(
                "You are the System Overseer, a professional AI assistant that helps manage "
                "and monitor the Trading Agent system. You provide detailed technical insights, "
                "precise answers, and help control the system through natural conversation. "
                "Be thorough, accurate, and formal in your responses, focusing on technical "
                "details and system performance metrics."
            ),
            description="Professional and technical personality profile",
            parameters={
                "verbosity": 0.8,  # Higher verbosity for detailed explanations
                "formality": 0.9,  # Very formal tone
                "creativity": 0.3,  # Less creative, more factual
                "empathy": 0.4,    # Less emotional
                "humor": 0.1       # Minimal humor
            }
        )
        
        # Create friendly profile
        friendly_profile = PersonalityProfile(
            profile_id="friendly",
            name="Friendly",
            system_prompt=(
                "You are the System Overseer, a friendly and approachable AI assistant "
                "that helps manage and monitor the Trading Agent system. You provide "
                "insights, answer questions, and help control the system through natural "
                "conversation. Be warm, conversational, and engaging in your responses, "
                "using simple language and occasional humor to make technical concepts "
                "more accessible."
            ),
            description="Friendly and approachable personality profile",
            parameters={
                "verbosity": 0.6,  # Moderate verbosity
                "formality": 0.3,  # Less formal, more conversational
                "creativity": 0.7,  # More creative in explanations
                "empathy": 0.9,    # Very empathetic
                "humor": 0.7       # More humor
            }
        )
        
        # Create concise profile
        concise_profile = PersonalityProfile(
            profile_id="concise",
            name="Concise",
            system_prompt=(
                "You are the System Overseer, an AI assistant that helps manage and "
                "monitor the Trading Agent system. Provide brief, direct responses "
                "focused on essential information. Prioritize brevity and clarity "
                "over detailed explanations unless specifically requested."
            ),
            description="Concise and direct personality profile",
            parameters={
                "verbosity": 0.2,  # Very low verbosity
                "formality": 0.5,  # Moderate formality
                "creativity": 0.3,  # Less creative
                "empathy": 0.4,    # Less emotional
                "humor": 0.2       # Minimal humor
            }
        )
        
        # Add profiles
        self.profiles = {
            "default": default_profile,
            "professional": professional_profile,
            "friendly": friendly_profile,
            "concise": concise_profile
        }
        
        # Set default profile
        self.default_profile_id = "default"
        
        # Save profiles
        if self.auto_save:
            self.save()
    
    def get_profile(self, profile_id: str = None) -> Optional[PersonalityProfile]:
        """Get personality profile by ID.
        
        Args:
            profile_id: Profile identifier or None for default
            
        Returns:
            PersonalityProfile: Profile instance or None if not found
        """
        profile_id = profile_id or self.default_profile_id
        return self.profiles.get(profile_id)
    
    def get_profiles(self) -> List[PersonalityProfile]:
        """Get all personality profiles.
        
        Returns:
            list: List of profiles
        """
        return list(self.profiles.values())
    
    def add_profile(self, profile: PersonalityProfile) -> bool:
        """Add personality profile.
        
        Args:
            profile: Profile to add
            
        Returns:
            bool: True if profile was added
        """
        with self.lock:
            # Check if profile already exists
            if profile.profile_id in self.profiles:
                logger.warning(f"Profile already exists: {profile.profile_id}")
                return False
            
            # Add profile
            self.profiles[profile.profile_id] = profile
            
            # Set as default if first profile
            if self.default_profile_id is None:
                self.default_profile_id = profile.profile_id
            
            # Save if auto-save enabled
            if self.auto_save:
                self.save()
            
            return True
    
    def update_profile(self, profile: PersonalityProfile) -> bool:
        """Update personality profile.
        
        Args:
            profile: Profile to update
            
        Returns:
            bool: True if profile was updated
        """
        with self.lock:
            # Check if profile exists
            if profile.profile_id not in self.profiles:
                logger.warning(f"Profile not found: {profile.profile_id}")
                return False
            
            # Update profile
            self.profiles[profile.profile_id] = profile
            
            # Save if auto-save enabled
            if self.auto_save:
                self.save()
            
            return True
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete personality profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            bool: True if profile was deleted
        """
        with self.lock:
            # Check if profile exists
            if profile_id not in self.profiles:
                logger.warning(f"Profile not found: {profile_id}")
                return False
            
            # Check if default profile
            if profile_id == self.default_profile_id:
                logger.warning("Cannot delete default profile")
                return False
            
            # Delete profile
            del self.profiles[profile_id]
            
            # Save if auto-save enabled
            if self.auto_save:
                self.save()
            
            return True
    
    def set_default_profile(self, profile_id: str) -> bool:
        """Set default personality profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            bool: True if default was set
        """
        with self.lock:
            # Check if profile exists
            if profile_id not in self.profiles:
                logger.warning(f"Profile not found: {profile_id}")
                return False
            
            # Set default
            self.default_profile_id = profile_id
            
            # Save if auto-save enabled
            if self.auto_save:
                self.save()
            
            return True
    
    def save(self) -> bool:
        """Save profiles to file.
        
        Returns:
            bool: True if save successful
        """
        if not self.profiles_file:
            logger.warning("No profiles file specified")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.profiles_file), exist_ok=True)
            
            # Prepare data
            data = {
                "profiles": {
                    profile_id: profile.to_dict()
                    for profile_id, profile in self.profiles.items()
                },
                "default_profile_id": self.default_profile_id,
                "timestamp": time.time()
            }
            
            # Write to file
            with open(self.profiles_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Profiles saved to {self.profiles_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")
            return False
    
    def load(self) -> bool:
        """Load profiles from file.
        
        Returns:
            bool: True if load successful
        """
        if not self.profiles_file:
            logger.warning("No profiles file specified")
            return False
        
        if not os.path.exists(self.profiles_file):
            logger.warning(f"Profiles file not found: {self.profiles_file}")
            return False
        
        try:
            # Read from file
            with open(self.profiles_file, "r") as f:
                data = json.load(f)
            
            # Load profiles
            profiles = {}
            for profile_id, profile_data in data.get("profiles", {}).items():
                profiles[profile_id] = PersonalityProfile.from_dict(profile_data)
            
            # Load default profile ID
            default_profile_id = data.get("default_profile_id")
            
            # Update profiles
            with self.lock:
                self.profiles = profiles
                self.default_profile_id = default_profile_id
            
            logger.debug(f"Profiles loaded from {self.profiles_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            return False


class UserPreferences:
    """User preferences for dialogue system."""
    
    def __init__(
        self,
        user_id: str,
        personality_profile_id: str = None,
        notification_level: str = "normal",
        language: str = "en",
        timezone: str = "UTC",
        custom_settings: Dict[str, Any] = None
    ):
        """Initialize user preferences.
        
        Args:
            user_id: User identifier
            personality_profile_id: Personality profile identifier
            notification_level: Notification level (minimal, normal, detailed)
            language: Preferred language code
            timezone: Preferred timezone
            custom_settings: Custom user settings
        """
        self.user_id = user_id
        self.personality_profile_id = personality_profile_id
        self.notification_level = notification_level
        self.language = language
        self.timezone = timezone
        self.custom_settings = custom_settings or {}
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preferences to dictionary.
        
        Returns:
            dict: Preferences as dictionary
        """
        return {
            "user_id": self.user_id,
            "personality_profile_id": self.personality_profile_id,
            "notification_level": self.notification_level,
            "language": self.language,
            "timezone": self.timezone,
            "custom_settings": self.custom_settings,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        """Create preferences from dictionary.
        
        Args:
            data: Preferences dictionary
            
        Returns:
            UserPreferences: Preferences instance
        """
        prefs = cls(
            user_id=data["user_id"],
            personality_profile_id=data.get("personality_profile_id"),
            notification_level=data.get("notification_level", "normal"),
            language=data.get("language", "en"),
            timezone=data.get("timezone", "UTC"),
            custom_settings=data.get("custom_settings", {})
        )
        
        if "last_updated" in data:
            prefs.last_updated = data["last_updated"]
        
        return prefs


class UserManager:
    """Manager for user preferences."""
    
    def __init__(
        self,
        users_file: str = None,
        auto_save: bool = True
    ):
        """Initialize user manager.
        
        Args:
            users_file: File path for persistent storage
            auto_save: Whether to automatically save changes
        """
        self.users_file = users_file
        self.auto_save = auto_save
        self.users = {}  # user_id -> UserPreferences
        
        # Load existing users if file exists
        if users_file and os.path.exists(users_file):
            self.load()
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def get_user(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserPreferences: User preferences or None if not found
        """
        return self.users.get(user_id)
    
    def get_or_create_user(self, user_id: str) -> UserPreferences:
        """Get user preferences or create if not exists.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserPreferences: User preferences
        """
        with self.lock:
            # Check if user exists
            if user_id in self.users:
                return self.users[user_id]
            
            # Create user
            user = UserPreferences(user_id=user_id)
            self.users[user_id] = user
            
            # Save if auto-save enabled
            if self.auto_save:
                self.save()
            
            return user
    
    def update_user(self, user: UserPreferences) -> bool:
        """Update user preferences.
        
        Args:
            user: User preferences
            
        Returns:
            bool: True if update successful
        """
        with self.lock:
            # Update user
            user.last_updated = time.time()
            self.users[user.user_id] = user
            
            # Save if auto-save enabled
            if self.auto_save:
                self.save()
            
            return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if deletion successful
        """
        with self.lock:
            # Check if user exists
            if user_id not in self.users:
                logger.warning(f"User not found: {user_id}")
                return False
            
            # Delete user
            del self.users[user_id]
            
            # Save if auto-save enabled
            if self.auto_save:
                self.save()
            
            return True
    
    def save(self) -> bool:
        """Save users to file.
        
        Returns:
            bool: True if save successful
        """
        if not self.users_file:
            logger.warning("No users file specified")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            
            # Prepare data
            data = {
                "users": {
                    user_id: user.to_dict()
                    for user_id, user in self.users.items()
                },
                "timestamp": time.time()
            }
            
            # Write to file
            with open(self.users_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Users saved to {self.users_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving users: {e}")
            return False
    
    def load(self) -> bool:
        """Load users from file.
        
        Returns:
            bool: True if load successful
        """
        if not self.users_file:
            logger.warning("No users file specified")
            return False
        
        if not os.path.exists(self.users_file):
            logger.warning(f"Users file not found: {self.users_file}")
            return False
        
        try:
            # Read from file
            with open(self.users_file, "r") as f:
                data = json.load(f)
            
            # Load users
            users = {}
            for user_id, user_data in data.get("users", {}).items():
                users[user_id] = UserPreferences.from_dict(user_data)
            
            # Update users
            with self.lock:
                self.users = users
            
            logger.debug(f"Users loaded from {self.users_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            return False


class DialogueManager:
    """Manager for dialogue interactions."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        data_dir: str = None,
        auto_save: bool = True
    ):
        """Initialize dialogue manager.
        
        Args:
            llm_client: LLM client instance
            data_dir: Directory for data files
            auto_save: Whether to automatically save data
        """
        # Set up data directory
        self.data_dir = data_dir or os.path.join(os.getcwd(), "data", "dialogue")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Set up components
        self.llm_client = llm_client
        self.memory = DialogueMemory(
            memory_file=os.path.join(self.data_dir, "memory.json"),
            auto_save=auto_save
        )
        self.personality_manager = PersonalityManager(
            profiles_file=os.path.join(self.data_dir, "personalities.json"),
            auto_save=auto_save
        )
        self.user_manager = UserManager(
            users_file=os.path.join(self.data_dir, "users.json"),
            auto_save=auto_save
        )
        
        # Function registry
        self.functions = {}  # function_name -> LLMFunction
        
        # Intent handlers
        self.intent_handlers = {}  # intent_type -> callback
        
        # Response handlers
        self.response_handlers = []  # [(filter_func, callback)]
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("DialogueManager initialized")
    
    def register_function(self, function: LLMFunction) -> bool:
        """Register function for LLM function calling.
        
        Args:
            function: Function definition
            
        Returns:
            bool: True if registration successful
        """
        with self.lock:
            # Check if function already registered
            if function.name in self.functions:
                logger.warning(f"Function already registered: {function.name}")
                return False
            
            # Register function
            self.functions[function.name] = function
            return True
    
    def unregister_function(self, function_name: str) -> bool:
        """Unregister function.
        
        Args:
            function_name: Function name
            
        Returns:
            bool: True if unregistration successful
        """
        with self.lock:
            # Check if function exists
            if function_name not in self.functions:
                logger.warning(f"Function not registered: {function_name}")
                return False
            
            # Unregister function
            del self.functions[function_name]
            return True
    
    def register_intent_handler(
        self,
        intent_type: str,
        callback: Callable[[DialogueIntent, DialogueContext], None]
    ) -> bool:
        """Register intent handler.
        
        Args:
            intent_type: Intent type to handle
            callback: Callback function
            
        Returns:
            bool: True if registration successful
        """
        with self.lock:
            # Check if handler already registered
            if intent_type in self.intent_handlers:
                logger.warning(f"Intent handler already registered: {intent_type}")
                return False
            
            # Register handler
            self.intent_handlers[intent_type] = callback
            return True
    
    def unregister_intent_handler(self, intent_type: str) -> bool:
        """Unregister intent handler.
        
        Args:
            intent_type: Intent type
            
        Returns:
            bool: True if unregistration successful
        """
        with self.lock:
            # Check if handler exists
            if intent_type not in self.intent_handlers:
                logger.warning(f"Intent handler not registered: {intent_type}")
                return False
            
            # Unregister handler
            del self.intent_handlers[intent_type]
            return True
    
    def register_response_handler(
        self,
        filter_func: Callable[[LLMResponse, DialogueContext], bool],
        callback: Callable[[LLMResponse, DialogueContext], None]
    ) -> int:
        """Register response handler.
        
        Args:
            filter_func: Function to filter responses
            callback: Callback function
            
        Returns:
            int: Handler ID
        """
        with self.lock:
            # Register handler
            handler_id = len(self.response_handlers)
            self.response_handlers.append((filter_func, callback))
            return handler_id
    
    def unregister_response_handler(self, handler_id: int) -> bool:
        """Unregister response handler.
        
        Args:
            handler_id: Handler ID
            
        Returns:
            bool: True if unregistration successful
        """
        with self.lock:
            # Check if handler exists
            if handler_id < 0 or handler_id >= len(self.response_handlers):
                logger.warning(f"Response handler not found: {handler_id}")
                return False
            
            # Unregister handler
            self.response_handlers[handler_id] = (None, None)
            return True
    
    def create_context(
        self,
        user_id: str,
        profile_id: str = None,
        context_id: str = None
    ) -> DialogueContext:
        """Create dialogue context.
        
        Args:
            user_id: User identifier
            profile_id: Personality profile identifier
            context_id: Optional context identifier
            
        Returns:
            DialogueContext: New context
        """
        # Get user preferences
        user = self.user_manager.get_or_create_user(user_id)
        
        # Get personality profile
        profile_id = profile_id or user.personality_profile_id
        profile = self.personality_manager.get_profile(profile_id)
        
        # Create context
        context = self.memory.create_context(
            user_id=user_id,
            system_prompt=profile.system_prompt if profile else None,
            context_id=context_id
        )
        
        return context
    
    def get_or_create_context(
        self,
        user_id: str,
        context_id: str = None
    ) -> DialogueContext:
        """Get dialogue context or create if not exists.
        
        Args:
            user_id: User identifier
            context_id: Context identifier or None for latest
            
        Returns:
            DialogueContext: Context instance
        """
        # If context ID provided, try to get it
        if context_id:
            context = self.memory.get_context(context_id)
            if context:
                return context
        
        # Try to get latest context for user
        context = self.memory.get_latest_user_context(user_id)
        if context:
            return context
        
        # Create new context
        return self.create_context(user_id)
    
    def process_user_message(
        self,
        user_id: str,
        message_text: str,
        context_id: str = None
    ) -> Tuple[LLMResponse, DialogueContext]:
        """Process user message.
        
        Args:
            user_id: User identifier
            message_text: Message text
            context_id: Context identifier or None for latest
            
        Returns:
            tuple: (LLM response, dialogue context)
        """
        # Get or create context
        context = self.get_or_create_context(user_id, context_id)
        
        # Create user message
        user_message = LLMMessage(
            role="user",
            content=message_text
        )
        
        # Add message to context
        self.memory.add_message(context.context_id, user_message)
        
        # Detect intent
        intent = self._detect_intent(user_message, context)
        
        # Handle intent if detected
        if intent and intent.intent_type in self.intent_handlers:
            self.intent_handlers[intent.intent_type](intent, context)
        
        # Get available functions
        functions = list(self.functions.values()) if self.functions else None
        
        # Get completion from LLM
        response = self.llm_client.get_completion(
            messages=context.get_messages(),
            functions=functions
        )
        
        # Add response to context
        self.memory.add_message(context.context_id, response.message)
        
        # Handle response
        self._handle_response(response, context)
        
        return response, context
    
    def _detect_intent(
        self,
        message: LLMMessage,
        context: DialogueContext
    ) -> Optional[DialogueIntent]:
        """Detect intent in user message.
        
        Args:
            message: User message
            context: Dialogue context
            
        Returns:
            DialogueIntent: Detected intent or None
        """
        # Simple keyword-based intent detection
        # In a real system, this would use a more sophisticated approach
        content = message.content.lower()
        
        # Check for help intent
        if content == "help" or content.startswith("help ") or content.endswith(" help"):
            return DialogueIntent(
                intent_type="help",
                confidence=0.9
            )
        
        # Check for status intent
        if content == "status" or content.startswith("status ") or "system status" in content:
            return DialogueIntent(
                intent_type="status",
                confidence=0.9
            )
        
        # Check for settings intent
        if (content == "settings" or content.startswith("settings ") or 
            "change settings" in content or "update settings" in content):
            return DialogueIntent(
                intent_type="settings",
                confidence=0.8
            )
        
        # Check for clear intent
        if content == "clear" or content == "clear context" or content == "reset":
            return DialogueIntent(
                intent_type="clear",
                confidence=0.9
            )
        
        return None
    
    def _handle_response(
        self,
        response: LLMResponse,
        context: DialogueContext
    ) -> None:
        """Handle LLM response.
        
        Args:
            response: LLM response
            context: Dialogue context
        """
        # Check for function call
        if response.function_call:
            self._handle_function_call(response, context)
        
        # Call response handlers
        for filter_func, callback in self.response_handlers:
            if filter_func and callback and filter_func(response, context):
                callback(response, context)
    
    def _handle_function_call(
        self,
        response: LLMResponse,
        context: DialogueContext
    ) -> None:
        """Handle function call in response.
        
        Args:
            response: LLM response
            context: Dialogue context
        """
        # Get function call
        function_call = response.function_call
        function_name = function_call.get("name")
        
        if not function_name:
            logger.warning("Function call missing name")
            return
        
        # Check if function exists
        if function_name not in self.functions:
            logger.warning(f"Function not found: {function_name}")
            return
        
        # Log function call
        logger.info(f"Function call: {function_name}")
        logger.debug(f"Function arguments: {function_call.get('arguments', '{}')}")
        
        # In a real system, this would execute the function
        # For now, just log it
        pass


# Example usage
if __name__ == "__main__":
    # Create LLM client
    from llm_client import LLMClient, OpenAIProvider
    
    client = LLMClient()
    
    # Register OpenAI provider
    openai_provider = OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key")
    )
    client.register_provider(openai_provider, is_default=True)
    
    # Create dialogue manager
    manager = DialogueManager(
        llm_client=client,
        data_dir="./data/dialogue"
    )
    
    # Register function
    get_status_function = LLMFunction(
        name="get_system_status",
        description="Get current system status",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
    manager.register_function(get_status_function)
    
    # Process user message
    response, context = manager.process_user_message(
        user_id="user123",
        message_text="Hello, what's the current system status?"
    )
    
    # Print response
    print(f"Response: {response.message.content}")
    
    # Process another message
    response, context = manager.process_user_message(
        user_id="user123",
        message_text="Can you explain that in more detail?"
    )
    
    # Print response
    print(f"Response: {response.message.content}")
