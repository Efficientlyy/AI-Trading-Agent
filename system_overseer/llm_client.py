#!/usr/bin/env python
"""
LLM Client for System Overseer

This module implements the LLM Client component that provides access to large language
models for natural language understanding, generation, and reasoning capabilities.
"""

import os
import sys
import json
import time
import logging
import threading
import queue
import requests
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Type
import uuid
import tiktoken
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_overseer.llm_client")


class LLMMessage:
    """Message for LLM conversation."""
    
    def __init__(
        self,
        role: str,
        content: str,
        name: str = None,
        message_id: str = None
    ):
        """Initialize LLM message.
        
        Args:
            role: Message role (system, user, assistant, function)
            content: Message content
            name: Optional name for function messages
            message_id: Unique message identifier
        """
        self.role = role
        self.content = content
        self.name = name
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary.
        
        Returns:
            dict: Message as dictionary
        """
        message = {
            "role": self.role,
            "content": self.content
        }
        
        if self.name:
            message["name"] = self.name
        
        return message
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], message_id: str = None) -> 'LLMMessage':
        """Create message from dictionary.
        
        Args:
            data: Message dictionary
            message_id: Optional message identifier
            
        Returns:
            LLMMessage: Message instance
        """
        return cls(
            role=data["role"],
            content=data["content"],
            name=data.get("name"),
            message_id=message_id
        )
    
    def __str__(self) -> str:
        """Get string representation of message.
        
        Returns:
            str: String representation
        """
        if self.name:
            return f"{self.role}({self.name}): {self.content[:50]}..."
        return f"{self.role}: {self.content[:50]}..."


class LLMFunction:
    """Function definition for LLM function calling."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function_id: str = None
    ):
        """Initialize LLM function.
        
        Args:
            name: Function name
            description: Function description
            parameters: Function parameters schema
            function_id: Unique function identifier
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function_id = function_id or str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert function to dictionary.
        
        Returns:
            dict: Function as dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], function_id: str = None) -> 'LLMFunction':
        """Create function from dictionary.
        
        Args:
            data: Function dictionary
            function_id: Optional function identifier
            
        Returns:
            LLMFunction: Function instance
        """
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data["parameters"],
            function_id=function_id
        )


class LLMResponse:
    """Response from LLM API."""
    
    def __init__(
        self,
        message: LLMMessage,
        function_call: Dict[str, Any] = None,
        usage: Dict[str, int] = None,
        response_id: str = None
    ):
        """Initialize LLM response.
        
        Args:
            message: Response message
            function_call: Function call information
            usage: Token usage information
            response_id: Unique response identifier
        """
        self.message = message
        self.function_call = function_call
        self.usage = usage or {}
        self.response_id = response_id or str(uuid.uuid4())
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary.
        
        Returns:
            dict: Response as dictionary
        """
        return {
            "response_id": self.response_id,
            "message": {
                "role": self.message.role,
                "content": self.message.content,
                "name": self.message.name,
                "message_id": self.message.message_id
            },
            "function_call": self.function_call,
            "usage": self.usage,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMResponse':
        """Create response from dictionary.
        
        Args:
            data: Response dictionary
            
        Returns:
            LLMResponse: Response instance
        """
        message = LLMMessage(
            role=data["message"]["role"],
            content=data["message"]["content"],
            name=data["message"].get("name"),
            message_id=data["message"].get("message_id")
        )
        
        return cls(
            message=message,
            function_call=data.get("function_call"),
            usage=data.get("usage"),
            response_id=data.get("response_id")
        )


class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(
        self,
        provider_id: str,
        name: str,
        models: List[str] = None
    ):
        """Initialize LLM provider.
        
        Args:
            provider_id: Provider identifier
            name: Provider name
            models: List of supported models
        """
        self.provider_id = provider_id
        self.name = name
        self.models = models or []
    
    def get_completion(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        functions: List[LLMFunction] = None,
        function_call: str = None
    ) -> LLMResponse:
        """Get completion from LLM.
        
        Args:
            messages: List of conversation messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            functions: List of available functions
            function_call: Function to call
            
        Returns:
            LLMResponse: LLM response
        """
        raise NotImplementedError("get_completion must be implemented by subclass")
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            model: Model to use for counting
            
        Returns:
            int: Token count
        """
        raise NotImplementedError("count_tokens must be implemented by subclass")
    
    def get_models(self) -> List[str]:
        """Get list of supported models.
        
        Returns:
            list: List of model identifiers
        """
        return self.models.copy()


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(
        self,
        api_key: str,
        organization: str = None,
        base_url: str = "https://api.openai.com/v1",
        models: List[str] = None
    ):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            organization: OpenAI organization ID
            base_url: API base URL
            models: List of supported models
        """
        super().__init__(
            provider_id="openai",
            name="OpenAI",
            models=models or [
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo"
            ]
        )
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
        if organization:
            self.session.headers.update({
                "OpenAI-Organization": organization
            })
        
        # Token counters
        self.encoding_cache = {}
    
    def get_completion(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        functions: List[LLMFunction] = None,
        function_call: str = None
    ) -> LLMResponse:
        """Get completion from OpenAI API.
        
        Args:
            messages: List of conversation messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            functions: List of available functions
            function_call: Function to call
            
        Returns:
            LLMResponse: LLM response
        """
        try:
            # Prepare request
            url = f"{self.base_url}/chat/completions"
            
            payload = {
                "model": model,
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add functions if provided
            if functions:
                payload["functions"] = [f.to_dict() for f in functions]
                
                if function_call:
                    payload["function_call"] = function_call
            
            # Make request
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            choice = data["choices"][0]
            message_data = choice["message"]
            
            # Create message
            message = LLMMessage(
                role=message_data["role"],
                content=message_data.get("content", ""),
                name=message_data.get("name")
            )
            
            # Extract function call
            function_call_data = message_data.get("function_call")
            
            # Create response
            llm_response = LLMResponse(
                message=message,
                function_call=function_call_data,
                usage=data.get("usage")
            )
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error getting completion from OpenAI: {e}")
            
            # Create error message
            error_message = LLMMessage(
                role="assistant",
                content=f"Error: {str(e)}"
            )
            
            return LLMResponse(message=error_message)
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            model: Model to use for counting
            
        Returns:
            int: Token count
        """
        try:
            # Map model names to encoding names
            encoding_name = "cl100k_base"  # Default for newer models
            
            if model.startswith("gpt-3.5-turbo"):
                encoding_name = "cl100k_base"
            elif model.startswith("gpt-4"):
                encoding_name = "cl100k_base"
            elif model.startswith("text-davinci"):
                encoding_name = "p50k_base"
            
            # Get or create encoding
            if encoding_name not in self.encoding_cache:
                self.encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
            
            encoding = self.encoding_cache[encoding_name]
            
            # Count tokens
            return len(encoding.encode(text))
            
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            
            # Fallback: estimate 1 token per 4 characters
            return len(text) // 4


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        models: List[str] = None
    ):
        """Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            base_url: API base URL
            models: List of supported models
        """
        super().__init__(
            provider_id="anthropic",
            name="Anthropic",
            models=models or [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
        )
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        })
    
    def get_completion(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        functions: List[LLMFunction] = None,
        function_call: str = None
    ) -> LLMResponse:
        """Get completion from Anthropic API.
        
        Args:
            messages: List of conversation messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            functions: List of available functions
            function_call: Function to call
            
        Returns:
            LLMResponse: LLM response
        """
        try:
            # Prepare request
            url = f"{self.base_url}/v1/messages"
            
            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                # Map roles
                role = msg.role
                if role == "assistant":
                    role = "assistant"
                elif role == "user":
                    role = "user"
                elif role == "system":
                    role = "system"
                else:
                    # Skip function messages
                    continue
                
                anthropic_messages.append({
                    "role": role,
                    "content": msg.content
                })
            
            payload = {
                "model": model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make request
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            content = data["content"][0]["text"]
            
            # Create message
            message = LLMMessage(
                role="assistant",
                content=content
            )
            
            # Create response
            llm_response = LLMResponse(
                message=message,
                usage={
                    "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                    "total_tokens": data.get("usage", {}).get("input_tokens", 0) + 
                                   data.get("usage", {}).get("output_tokens", 0)
                }
            )
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error getting completion from Anthropic: {e}")
            
            # Create error message
            error_message = LLMMessage(
                role="assistant",
                content=f"Error: {str(e)}"
            )
            
            return LLMResponse(message=error_message)
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            model: Model to use for counting
            
        Returns:
            int: Token count
        """
        # Anthropic doesn't provide a token counting library
        # Estimate 1 token per 4 characters
        return len(text) // 4


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api",
        models: List[str] = None
    ):
        """Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key
            base_url: API base URL
            models: List of supported models
        """
        super().__init__(
            provider_id="openrouter",
            name="OpenRouter",
            models=models or [
                "openai/gpt-4o",
                "anthropic/claude-3-opus",
                "anthropic/claude-3-sonnet",
                "google/gemini-pro",
                "meta-llama/llama-3-70b-instruct"
            ]
        )
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://trading-agent.system-overseer",
            "X-Title": "Trading Agent System Overseer"
        })
        
        # Token counters
        self.encoding_cache = {}
    
    def get_completion(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        functions: List[LLMFunction] = None,
        function_call: str = None
    ) -> LLMResponse:
        """Get completion from OpenRouter API.
        
        Args:
            messages: List of conversation messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            functions: List of available functions
            function_call: Function to call
            
        Returns:
            LLMResponse: LLM response
        """
        try:
            # Prepare request
            url = f"{self.base_url}/v1/chat/completions"
            
            payload = {
                "model": model,
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add functions if provided and model supports it
            if functions and model.startswith(("openai/", "anthropic/")):
                payload["functions"] = [f.to_dict() for f in functions]
                
                if function_call:
                    payload["function_call"] = function_call
            
            # Make request
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            choice = data["choices"][0]
            message_data = choice["message"]
            
            # Create message
            message = LLMMessage(
                role=message_data["role"],
                content=message_data.get("content", ""),
                name=message_data.get("name")
            )
            
            # Extract function call
            function_call_data = message_data.get("function_call")
            
            # Create response
            llm_response = LLMResponse(
                message=message,
                function_call=function_call_data,
                usage=data.get("usage")
            )
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error getting completion from OpenRouter: {e}")
            
            # Create error message
            error_message = LLMMessage(
                role="assistant",
                content=f"Error: {str(e)}"
            )
            
            return LLMResponse(message=error_message)
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            model: Model to use for counting
            
        Returns:
            int: Token count
        """
        try:
            # For OpenAI models, use tiktoken
            if model.startswith("openai/"):
                # Map model names to encoding names
                encoding_name = "cl100k_base"  # Default for newer models
                
                if "gpt-3.5-turbo" in model:
                    encoding_name = "cl100k_base"
                elif "gpt-4" in model:
                    encoding_name = "cl100k_base"
                elif "text-davinci" in model:
                    encoding_name = "p50k_base"
                
                # Get or create encoding
                if encoding_name not in self.encoding_cache:
                    self.encoding_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
                
                encoding = self.encoding_cache[encoding_name]
                
                # Count tokens
                return len(encoding.encode(text))
            
            # For other models, estimate
            return len(text) // 4
            
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            
            # Fallback: estimate 1 token per 4 characters
            return len(text) // 4


class LLMClient:
    """Client for interacting with LLM providers."""
    
    def __init__(self):
        """Initialize LLM client."""
        self.providers = {}  # provider_id -> LLMProvider
        self.default_provider = None
        self.default_model = None
        
        # Rate limiting
        self.request_counts = {}  # provider_id -> count
        self.last_request_time = {}  # provider_id -> timestamp
        self.rate_limits = {}  # provider_id -> {requests_per_minute, tokens_per_minute}
        
        # Token usage tracking
        self.token_usage = {
            "total": 0,
            "prompt": 0,
            "completion": 0,
            "providers": {}  # provider_id -> {total, prompt, completion}
        }
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("LLMClient initialized")
    
    def register_provider(
        self,
        provider: LLMProvider,
        is_default: bool = False,
        rate_limit_rpm: int = None,
        rate_limit_tpm: int = None
    ) -> bool:
        """Register LLM provider.
        
        Args:
            provider: Provider instance
            is_default: Whether this is the default provider
            rate_limit_rpm: Rate limit in requests per minute
            rate_limit_tpm: Rate limit in tokens per minute
            
        Returns:
            bool: True if registration successful
        """
        with self.lock:
            # Check if provider already registered
            if provider.provider_id in self.providers:
                logger.warning(f"Provider {provider.provider_id} already registered")
                return False
            
            # Store provider
            self.providers[provider.provider_id] = provider
            
            # Initialize tracking
            self.request_counts[provider.provider_id] = 0
            self.last_request_time[provider.provider_id] = 0
            self.token_usage["providers"][provider.provider_id] = {
                "total": 0,
                "prompt": 0,
                "completion": 0
            }
            
            # Set rate limits
            self.rate_limits[provider.provider_id] = {
                "requests_per_minute": rate_limit_rpm,
                "tokens_per_minute": rate_limit_tpm
            }
            
            # Set as default if requested or if first provider
            if is_default or self.default_provider is None:
                self.default_provider = provider.provider_id
                
                # Set default model to first model of default provider
                if provider.models and self.default_model is None:
                    self.default_model = provider.models[0]
            
            logger.info(f"Provider registered: {provider.name} ({provider.provider_id})")
            return True
    
    def unregister_provider(self, provider_id: str) -> bool:
        """Unregister LLM provider.
        
        Args:
            provider_id: Provider identifier
            
        Returns:
            bool: True if unregistration successful
        """
        with self.lock:
            # Check if provider exists
            if provider_id not in self.providers:
                logger.warning(f"Provider {provider_id} not registered")
                return False
            
            # Remove provider
            del self.providers[provider_id]
            
            # Clean up tracking
            if provider_id in self.request_counts:
                del self.request_counts[provider_id]
            
            if provider_id in self.last_request_time:
                del self.last_request_time[provider_id]
            
            if provider_id in self.rate_limits:
                del self.rate_limits[provider_id]
            
            if provider_id in self.token_usage["providers"]:
                del self.token_usage["providers"][provider_id]
            
            # Update default provider if needed
            if self.default_provider == provider_id:
                if self.providers:
                    self.default_provider = next(iter(self.providers.keys()))
                    
                    # Update default model
                    provider = self.providers[self.default_provider]
                    if provider.models:
                        self.default_model = provider.models[0]
                else:
                    self.default_provider = None
                    self.default_model = None
            
            logger.info(f"Provider unregistered: {provider_id}")
            return True
    
    def get_provider(self, provider_id: str = None) -> Optional[LLMProvider]:
        """Get provider by ID or default provider.
        
        Args:
            provider_id: Provider identifier or None for default
            
        Returns:
            LLMProvider: Provider instance or None if not found
        """
        provider_id = provider_id or self.default_provider
        return self.providers.get(provider_id)
    
    def get_completion(
        self,
        messages: List[LLMMessage],
        model: str = None,
        provider_id: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        functions: List[LLMFunction] = None,
        function_call: str = None
    ) -> LLMResponse:
        """Get completion from LLM.
        
        Args:
            messages: List of conversation messages
            model: Model to use or None for default
            provider_id: Provider identifier or None for default
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            functions: List of available functions
            function_call: Function to call
            
        Returns:
            LLMResponse: LLM response
        """
        with self.lock:
            # Get provider
            provider = self.get_provider(provider_id)
            if not provider:
                logger.error(f"Provider not found: {provider_id}")
                
                # Create error message
                error_message = LLMMessage(
                    role="assistant",
                    content=f"Error: Provider not found: {provider_id}"
                )
                
                return LLMResponse(message=error_message)
            
            # Get model
            model = model or self.default_model
            if not model:
                logger.error("No model specified and no default model set")
                
                # Create error message
                error_message = LLMMessage(
                    role="assistant",
                    content="Error: No model specified and no default model set"
                )
                
                return LLMResponse(message=error_message)
            
            # Check rate limits
            if not self._check_rate_limits(provider.provider_id, messages, model):
                logger.warning(f"Rate limit exceeded for provider: {provider.provider_id}")
                
                # Create error message
                error_message = LLMMessage(
                    role="assistant",
                    content=f"Error: Rate limit exceeded for provider: {provider.provider_id}"
                )
                
                return LLMResponse(message=error_message)
            
            # Update request count
            self.request_counts[provider.provider_id] += 1
            self.last_request_time[provider.provider_id] = time.time()
            
            # Get completion
            response = provider.get_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                functions=functions,
                function_call=function_call
            )
            
            # Update token usage
            if response.usage:
                prompt_tokens = response.usage.get("prompt_tokens", 0)
                completion_tokens = response.usage.get("completion_tokens", 0)
                total_tokens = response.usage.get("total_tokens", 0)
                
                # Update provider usage
                provider_usage = self.token_usage["providers"][provider.provider_id]
                provider_usage["prompt"] += prompt_tokens
                provider_usage["completion"] += completion_tokens
                provider_usage["total"] += total_tokens
                
                # Update total usage
                self.token_usage["prompt"] += prompt_tokens
                self.token_usage["completion"] += completion_tokens
                self.token_usage["total"] += total_tokens
            
            return response
    
    def count_tokens(self, text: str, model: str = None, provider_id: str = None) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for
            model: Model to use or None for default
            provider_id: Provider identifier or None for default
            
        Returns:
            int: Token count
        """
        # Get provider
        provider = self.get_provider(provider_id)
        if not provider:
            logger.error(f"Provider not found: {provider_id}")
            return len(text) // 4  # Fallback estimate
        
        # Get model
        model = model or self.default_model
        if not model:
            logger.error("No model specified and no default model set")
            return len(text) // 4  # Fallback estimate
        
        # Count tokens
        return provider.count_tokens(text, model)
    
    def count_messages_tokens(
        self,
        messages: List[LLMMessage],
        model: str = None,
        provider_id: str = None
    ) -> int:
        """Count tokens in messages.
        
        Args:
            messages: List of messages
            model: Model to use or None for default
            provider_id: Provider identifier or None for default
            
        Returns:
            int: Token count
        """
        # Get provider
        provider = self.get_provider(provider_id)
        if not provider:
            logger.error(f"Provider not found: {provider_id}")
            
            # Fallback estimate
            return sum(len(m.content) // 4 for m in messages)
        
        # Get model
        model = model or self.default_model
        if not model:
            logger.error("No model specified and no default model set")
            
            # Fallback estimate
            return sum(len(m.content) // 4 for m in messages)
        
        # Count tokens for each message
        total_tokens = 0
        for message in messages:
            # Count content tokens
            content_tokens = provider.count_tokens(message.content, model)
            
            # Add overhead for message format
            # This is an approximation and varies by provider
            overhead = 4  # Role, formatting
            
            if message.name:
                overhead += 1 + provider.count_tokens(message.name, model)
            
            total_tokens += content_tokens + overhead
        
        # Add conversation overhead
        total_tokens += 2  # Start and end of conversation markers
        
        return total_tokens
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage statistics.
        
        Returns:
            dict: Token usage statistics
        """
        with self.lock:
            return copy.deepcopy(self.token_usage)
    
    def reset_token_usage(self) -> None:
        """Reset token usage statistics."""
        with self.lock:
            self.token_usage = {
                "total": 0,
                "prompt": 0,
                "completion": 0,
                "providers": {
                    provider_id: {
                        "total": 0,
                        "prompt": 0,
                        "completion": 0
                    }
                    for provider_id in self.providers
                }
            }
    
    def get_available_models(self, provider_id: str = None) -> List[str]:
        """Get list of available models.
        
        Args:
            provider_id: Provider identifier or None for all providers
            
        Returns:
            list: List of model identifiers
        """
        if provider_id:
            provider = self.get_provider(provider_id)
            if provider:
                return provider.get_models()
            return []
        
        # Get models from all providers
        models = []
        for provider in self.providers.values():
            models.extend(provider.get_models())
        
        return models
    
    def set_default_provider(self, provider_id: str) -> bool:
        """Set default provider.
        
        Args:
            provider_id: Provider identifier
            
        Returns:
            bool: True if successful
        """
        with self.lock:
            if provider_id not in self.providers:
                logger.warning(f"Provider {provider_id} not registered")
                return False
            
            self.default_provider = provider_id
            
            # Update default model
            provider = self.providers[provider_id]
            if provider.models:
                self.default_model = provider.models[0]
            
            return True
    
    def set_default_model(self, model: str) -> bool:
        """Set default model.
        
        Args:
            model: Model identifier
            
        Returns:
            bool: True if successful
        """
        with self.lock:
            # Check if model is available
            available_models = self.get_available_models()
            if model not in available_models:
                logger.warning(f"Model {model} not available")
                return False
            
            self.default_model = model
            return True
    
    def set_rate_limit(
        self,
        provider_id: str,
        requests_per_minute: int = None,
        tokens_per_minute: int = None
    ) -> bool:
        """Set rate limit for provider.
        
        Args:
            provider_id: Provider identifier
            requests_per_minute: Rate limit in requests per minute
            tokens_per_minute: Rate limit in tokens per minute
            
        Returns:
            bool: True if successful
        """
        with self.lock:
            if provider_id not in self.providers:
                logger.warning(f"Provider {provider_id} not registered")
                return False
            
            if provider_id not in self.rate_limits:
                self.rate_limits[provider_id] = {}
            
            if requests_per_minute is not None:
                self.rate_limits[provider_id]["requests_per_minute"] = requests_per_minute
            
            if tokens_per_minute is not None:
                self.rate_limits[provider_id]["tokens_per_minute"] = tokens_per_minute
            
            return True
    
    def _check_rate_limits(
        self,
        provider_id: str,
        messages: List[LLMMessage],
        model: str
    ) -> bool:
        """Check if request would exceed rate limits.
        
        Args:
            provider_id: Provider identifier
            messages: List of messages
            model: Model to use
            
        Returns:
            bool: True if request is allowed
        """
        # Get rate limits
        if provider_id not in self.rate_limits:
            return True
        
        rate_limits = self.rate_limits[provider_id]
        
        # Check requests per minute
        rpm_limit = rate_limits.get("requests_per_minute")
        if rpm_limit is not None:
            # Count requests in the last minute
            now = time.time()
            minute_ago = now - 60
            
            if (self.last_request_time.get(provider_id, 0) > minute_ago and
                self.request_counts.get(provider_id, 0) >= rpm_limit):
                return False
        
        # Check tokens per minute
        tpm_limit = rate_limits.get("tokens_per_minute")
        if tpm_limit is not None:
            # Estimate tokens for this request
            provider = self.providers[provider_id]
            estimated_tokens = self.count_messages_tokens(messages, model, provider_id)
            
            # Get tokens used in the last minute
            # This is an approximation since we don't track tokens by time
            # A more accurate implementation would track token usage with timestamps
            tokens_used = self.token_usage["providers"][provider_id]["total"]
            
            if tokens_used + estimated_tokens > tpm_limit:
                return False
        
        return True


# Example usage
if __name__ == "__main__":
    # Create LLM client
    client = LLMClient()
    
    # Register OpenAI provider
    openai_provider = OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key")
    )
    client.register_provider(openai_provider, is_default=True)
    
    # Create messages
    system_message = LLMMessage(
        role="system",
        content="You are a helpful assistant."
    )
    
    user_message = LLMMessage(
        role="user",
        content="Hello, what can you do?"
    )
    
    # Get completion
    response = client.get_completion(
        messages=[system_message, user_message],
        model="gpt-3.5-turbo"
    )
    
    # Print response
    print(f"Response: {response.message.content}")
    
    # Print token usage
    print(f"Token usage: {client.get_token_usage()}")
