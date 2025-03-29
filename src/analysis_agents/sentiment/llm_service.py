"""LLM service for advanced sentiment analysis.

This module provides integration with large language models
for sophisticated sentiment analysis and market event detection.
"""

import asyncio
import json
import os
import time
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

import aiohttp

from src.common.config import config
from src.common.logging import get_logger
from src.common.caching import Cache
from src.common.events import event_bus
from src.analysis_agents.sentiment.provider_failover import provider_failover_manager
from src.analysis_agents.sentiment.prompt_tuning import prompt_tuning_system
from src.analysis_agents.sentiment.ab_testing import ab_testing_framework, ExperimentType


class LLMService:
    """Service for large language model-based text analysis.
    
    This service provides access to advanced language models for financial
    text analysis, event detection, and impact assessment.
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        self.logger = get_logger("analysis_agents", "llm_service")
        
        # Load configuration
        self.primary_model = config.get("llm.primary_model", "gpt-4o")
        self.financial_model = config.get("llm.financial_model", "gpt-4o")
        self.batch_size = config.get("llm.batch_size", 5)
        self.use_cached_responses = config.get("llm.use_cached_responses", True)
        
        # API configuration
        self.api_keys = {
            "openai": config.get("apis.openai.api_key", os.getenv("OPENAI_API_KEY", "")),
            "anthropic": config.get("apis.anthropic.api_key", os.getenv("ANTHROPIC_API_KEY", "")),
            "azure": config.get("apis.azure_openai.api_key", os.getenv("AZURE_OPENAI_API_KEY", "")),
        }
        
        # API endpoints
        self.endpoints = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "anthropic": "https://api.anthropic.com/v1/messages",
            "azure": config.get("apis.azure_openai.endpoint", "")
        }
        
        # Model mapping
        self.model_providers = {
            "gpt-4o": "openai",
            "gpt-4-turbo": "openai",
            "gpt-3.5-turbo": "openai",
            "claude-3-opus": "anthropic",
            "claude-3-sonnet": "anthropic",
            "claude-3-haiku": "anthropic"
        }
        
        # Set up caching
        cache_ttl = config.get("llm.cache_ttl", 3600)  # 1 hour default
        self.response_cache = Cache(ttl=cache_ttl)
        
        # Generate hash value for prompt templates to detect changes
        self._prompt_template_hash = hash(json.dumps(self._get_prompt_templates()))
        
        # Create session for API calls
        self.session = None
    
    async def initialize(self) -> None:
        """Initialize the LLM service and validate API access."""
        self.logger.info("Initializing LLM service")
        
        # Create session
        self.session = aiohttp.ClientSession()
        
        # Initialize the provider failover manager
        provider_failover_manager.initialize()
        
        # Initialize the prompt tuning system
        prompt_tuning_system.initialize()
        
        # Initialize the A/B testing framework
        ab_testing_framework.initialize()
        
        # Register default prompts with the tuning system
        prompt_tuning_system.register_default_prompts(self._get_prompt_templates())
        
        # Validate API keys
        valid_providers = self._validate_api_keys()
        
        if not valid_providers:
            self.logger.error("No valid API keys found for any LLM provider")
            return
        
        self.logger.info(f"LLM service initialized with providers: {', '.join(valid_providers)}")
    
    async def close(self) -> None:
        """Close the LLM service and release resources."""
        if self.session:
            await self.session.close()
            self.session = None
            
        # Save the failover manager's fallback cache
        provider_failover_manager.save_fallback_cache()
    
    async def _validate_api_keys(self) -> List[str]:
        """Validate API keys for each provider.
        
        Returns:
            List of valid provider names
        """
        valid_providers = []
        
        for provider, api_key in self.api_keys.items():
            if not api_key:
                self.logger.warning(f"No API key configured for {provider}")
                continue
                
            # We'll consider it valid for now and deal with actual API failures later
            valid_providers.append(provider)
            
        return valid_providers
    
    def _get_prompt_templates(self) -> Dict[str, str]:
        """Get prompt templates for various analysis tasks.
        
        Returns:
            Dictionary of prompt templates
        """
        # Default templates as fallback
        default_templates = {
            "sentiment_analysis": """
You are a financial sentiment analyzer specialized in cryptocurrency and blockchain markets.
Analyze the following text and determine the overall market sentiment.

Text:
{text}

Instructions:
1. Analyze the text for bullish, bearish, or neutral sentiment.
2. Consider financial jargon and crypto-specific terminology.
3. Evaluate the credibility and potential impact of the content.
4. Provide an explanation for your reasoning.

Your response must be in the following JSON format:
{
    "sentiment_value": <float between 0 and 1, where 0 is extremely bearish, 0.5 is neutral, and 1 is extremely bullish>,
    "direction": <"bullish", "bearish", or "neutral">,
    "confidence": <float between 0 and 1 indicating your confidence level>,
    "explanation": <brief explanation of your reasoning>,
    "key_points": <list of key points that influenced your assessment>
}
""",

            "event_detection": """
You are a financial event detector specialized in identifying market-moving events for cryptocurrency markets.
Analyze the following text and determine if it contains a potentially market-moving event.

Text:
{text}

Instructions:
1. Identify if this contains a potentially market-moving event.
2. Analyze the significance, credibility, and potential market impact.
3. Categorize the event type (regulatory, technical, adoption, etc.).
4. Estimate the propagation timeline (how quickly this might affect the market).

Your response must be in the following JSON format:
{
    "is_market_event": <true or false>,
    "event_type": <category of event or null if not an event>,
    "assets_affected": <list of assets likely to be affected or null>,
    "severity": <integer from 0-10 indicating potential market impact or 0 if not an event>,
    "credibility": <float between 0 and 1 indicating credibility of the source and information>,
    "propagation_speed": <"immediate", "hours", "days", or "weeks">,
    "explanation": <brief explanation of your reasoning>
}
""",

            "impact_assessment": """
You are a financial impact assessor specialized in predicting how events affect cryptocurrency markets.
Analyze the following event and assess its potential market impact.

Event:
{event}

Market Context:
{market_context}

Instructions:
1. Analyze the potential short and long-term market impact of this event.
2. Predict the likely direction and magnitude of price movements for affected assets.
3. Estimate the duration of the potential impact.
4. Consider different market regimes (bull market, bear market, sideways).

Your response must be in the following JSON format:
{
    "primary_impact_direction": <"positive", "negative", or "neutral">,
    "impact_magnitude": <float between 0 and 1 indicating the magnitude>,
    "confidence": <float between 0 and 1 indicating your confidence level>,
    "estimated_duration": <"hours", "days", "weeks", or "months">,
    "affected_assets": <dictionary of assets and their relative impact scores>,
    "reasoning": <brief explanation of your reasoning>,
    "risk_factors": <list of factors that could alter your assessment>
}
"""
        }
        
        templates = {}
        
        try:
            # First check for active A/B tests related to prompt templates
            for prompt_type in default_templates.keys():
                # Map to correct experiment type
                experiment_type = None
                if prompt_type == "sentiment_analysis":
                    experiment_type = ExperimentType.PROMPT_TEMPLATE
                elif prompt_type == "event_detection":
                    experiment_type = ExperimentType.PROMPT_TEMPLATE
                elif prompt_type == "impact_assessment":
                    experiment_type = ExperimentType.PROMPT_TEMPLATE
                
                if experiment_type:
                    # Try to get an A/B test variant
                    experiment_id, variant_config = ab_testing_framework.get_experiment_variant(
                        experiment_type,
                        {"prompt_type": prompt_type}
                    )
                    
                    if experiment_id and variant_config and "template" in variant_config:
                        templates[prompt_type] = variant_config["template"]
                        continue  # Skip the prompt tuning check for this type
            
            # If no A/B test variant was assigned, try prompt tuning
            experiment = config.get("sentiment_analysis.prompt_tuning.experiment_enabled", False)
            
            for prompt_type in default_templates.keys():
                if prompt_type not in templates:  # Skip if already set by A/B test
                    template, metadata = prompt_tuning_system.get_prompt_template(prompt_type, experiment)
                    if template:
                        templates[prompt_type] = template
                    else:
                        templates[prompt_type] = default_templates[prompt_type]
                        
        except Exception as e:
            self.logger.error(f"Error getting prompt templates: {str(e)}")
            return default_templates
            
        return templates
    
    async def analyze_sentiment(self, texts: Union[str, List[str]], model: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Analyze sentiment of text using LLM.
        
        Args:
            texts: Text or list of texts to analyze
            model: Optional model override
            
        Returns:
            Sentiment analysis result (dictionary or list of dictionaries)
        """
        single_input = isinstance(texts, str)
        texts_list = [texts] if single_input else texts
        
        if not texts_list:
            return [] if not single_input else None
        
        # Use specified model or default to financial model for sentiment
        model_to_use = model or self.financial_model
        
        results = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts_list), self.batch_size):
            batch = texts_list[i:i+self.batch_size]
            batch_results = await asyncio.gather(*[
                self._analyze_single_text(text, "sentiment_analysis", model_to_use)
                for text in batch
            ])
            results.extend(batch_results)
        
        return results[0] if single_input else results
    
    async def detect_market_event(self, texts: Union[str, List[str]], model: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Detect market-moving events in text.
        
        Args:
            texts: Text or list of texts to analyze
            model: Optional model override
            
        Returns:
            Event detection results
        """
        single_input = isinstance(texts, str)
        texts_list = [texts] if single_input else texts
        
        if not texts_list:
            return [] if not single_input else None
        
        # Use specified model or default to primary model for event detection
        model_to_use = model or self.primary_model
        
        results = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts_list), self.batch_size):
            batch = texts_list[i:i+self.batch_size]
            batch_results = await asyncio.gather(*[
                self._analyze_single_text(text, "event_detection", model_to_use)
                for text in batch
            ])
            results.extend(batch_results)
        
        return results[0] if single_input else results
    
    async def assess_market_impact(self, event: str, market_context: str = "", model: str = None) -> Dict[str, Any]:
        """Assess the market impact of an event.
        
        Args:
            event: Description of the event
            market_context: Current market context (optional)
            model: Optional model override
            
        Returns:
            Impact assessment result
        """
        # Use specified model or default to primary model
        model_to_use = model or self.primary_model
        
        # Create a combined input for the impact assessment
        prompt_data = {
            "event": event,
            "market_context": market_context or "Current market conditions are not specified."
        }
        
        return await self._analyze_single_text(prompt_data, "impact_assessment", model_to_use)
    
    async def _analyze_single_text(self, text_data: Union[str, Dict[str, str]], prompt_type: str, model: str) -> Dict[str, Any]:
        """Analyze a single text using the specified prompt type and model.
        
        Args:
            text_data: Text to analyze or dictionary of prompt variables
            prompt_type: Type of prompt template to use
            model: Model to use for analysis
            
        Returns:
            Analysis result
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Create context for experiment tracking
        context = {
            "prompt_type": prompt_type,
            "model": model,
            "request_id": request_id,
            "symbol": text_data.get("symbol", "") if isinstance(text_data, dict) else ""
        }
        
        # Get the appropriate prompt template - this now checks both A/B tests and prompt tuning
        prompt_templates = self._get_prompt_templates()
        prompt_template = prompt_templates.get(prompt_type)
        
        if not prompt_template:
            self.logger.error(f"Unknown prompt type: {prompt_type}")
            return self._get_fallback_result(prompt_type)
        
        # Check for active A/B test for this prompt type
        experiment_id = None
        variant_id = None
        
        # Map to correct experiment type
        experiment_type = None
        if prompt_type == "sentiment_analysis":
            experiment_type = ExperimentType.PROMPT_TEMPLATE
        elif prompt_type == "event_detection":
            experiment_type = ExperimentType.PROMPT_TEMPLATE
        elif prompt_type == "impact_assessment":
            experiment_type = ExperimentType.PROMPT_TEMPLATE
        
        # Check for model selection experiments
        model_experiment_type = ExperimentType.MODEL_SELECTION
        model_experiment_id, model_variant_config = ab_testing_framework.get_experiment_variant(
            model_experiment_type, context
        )
        
        if model_experiment_id and model_variant_config and "model" in model_variant_config:
            # Override the model based on the experiment
            selected_model = model_variant_config["model"]
            context["model"] = selected_model
            model = selected_model
            
            # Track the experiment assignment
            event_bus.publish(
                "experiment_assignment",
                {
                    "experiment_id": model_experiment_id,
                    "variant_id": model_variant_config.get("variant_id", "unknown"),
                    "context": context
                }
            )
        
        # Check for temperature experiments
        temp_experiment_type = ExperimentType.TEMPERATURE
        temp_experiment_id, temp_variant_config = ab_testing_framework.get_experiment_variant(
            temp_experiment_type, context
        )
        
        temperature_override = None
        if temp_experiment_id and temp_variant_config and "temperature" in temp_variant_config:
            # Override temperature based on the experiment
            temperature_override = temp_variant_config["temperature"]
            
            # Track the experiment assignment
            event_bus.publish(
                "experiment_assignment",
                {
                    "experiment_id": temp_experiment_id,
                    "variant_id": temp_variant_config.get("variant_id", "unknown"),
                    "context": context
                }
            )
        
        # Extract metadata from template if present (from prompt tuning)
        if "<!-- METADATA:" in prompt_template and temperature_override is None:
            metadata_marker = "<!-- METADATA:"
            metadata_end = "-->"
            metadata_start = prompt_template.find(metadata_marker) + len(metadata_marker)
            metadata_finish = prompt_template.find(metadata_end, metadata_start)
            
            if metadata_start > 0 and metadata_finish > metadata_start:
                metadata_str = prompt_template[metadata_start:metadata_finish].strip()
                try:
                    metadata = json.loads(metadata_str)
                    temperature_override = metadata.get("temperature")
                    # Remove metadata from the template
                    prompt_template = prompt_template[:prompt_template.find(metadata_marker)]
                except Exception as e:
                    self.logger.warning(f"Error parsing prompt metadata: {str(e)}")
        
        # Format the prompt with the input text
        if isinstance(text_data, str):
            formatted_prompt = prompt_template.replace("{text}", text_data)
            input_text_hash = hashlib.md5(text_data.encode('utf-8')).hexdigest()
        else:
            formatted_prompt = prompt_template
            for key, value in text_data.items():
                formatted_prompt = formatted_prompt.replace(f"{{{key}}}", value)
            # Create a hash from all values
            input_text_hash = hashlib.md5(json.dumps(text_data, sort_keys=True).encode('utf-8')).hexdigest()
        
        # Generate a cache key
        cache_key = f"{model}:{prompt_type}:{hash(formatted_prompt)}"
        
        # Track usage in relevant systems (prompt tuning, A/B testing)
        if experiment_type:
            experiment_id, variant_config = ab_testing_framework.get_experiment_variant(
                experiment_type, context
            )
            
            if experiment_id and variant_config:
                variant_id = variant_config.get("variant_id", "unknown")
                
                # Publish experiment event for tracking
                event_bus.publish(
                    "experiment_assignment",
                    {
                        "experiment_id": experiment_id,
                        "variant_id": variant_id,
                        "context": context
                    }
                )
        
        # Also track in prompt tuning system if applicable
        prompt_id = None
        template_metadata = {}
        
        if "prompt_tuning" in config.get("sentiment_analysis", {}) and config.get("sentiment_analysis.prompt_tuning.enabled", False):
            # Get prompt ID from tuning system
            _, template_metadata = prompt_tuning_system.get_prompt_template(prompt_type, False)
            prompt_id = template_metadata.get("prompt_id")
            
            if prompt_id:
                prompt_tuning_system.track_prompt_usage(
                    request_id, 
                    prompt_id, 
                    {
                        "prompt_type": prompt_type,
                        "model": model,
                        "text_hash": input_text_hash,
                        "experiment": template_metadata.get("is_experiment", False)
                    }
                )
        
        # Check cache if enabled
        if self.use_cached_responses:
            cached_result = self.response_cache.get(cache_key)
            if cached_result:
                # Log as success but note that it's from cache
                if prompt_id:
                    event_bus.publish(
                        "llm_api_response",
                        {
                            "request_id": request_id,
                            "prompt_id": prompt_id,
                            "success": True,
                            "from_cache": True,
                            "latency_ms": 0
                        }
                    )
                    
                # Log for A/B testing if applicable
                if experiment_id and variant_id:
                    event_bus.publish(
                        "experiment_result",
                        {
                            "experiment_id": experiment_id,
                            "variant_id": variant_id,
                            "request_id": request_id,
                            "success": True,
                            "from_cache": True,
                            "latency_ms": 0
                        }
                    )
                
                return cached_result
        
        # Check for a fallback response if available
        fallback_response = provider_failover_manager.get_fallback_response(prompt_type, input_text_hash)
        
        # Use the failover manager to select the appropriate provider and model
        selected_provider, selected_model = await provider_failover_manager.select_provider_for_model(model)
        
        # Get the start time for latency tracking
        start_time = time.time()
        
        try:
            # Check if we have an API key for this provider
            if not self.api_keys.get(selected_provider):
                self.logger.error(f"No API key available for provider: {selected_provider}")
                
                # Log the failure if using a tuned prompt
                if prompt_id:
                    event_bus.publish(
                        "llm_api_response",
                        {
                            "request_id": request_id,
                            "prompt_id": prompt_id,
                            "success": False,
                            "error": "No API key available"
                        }
                    )
                
                # Use fallback if available, otherwise return standard fallback
                return fallback_response if fallback_response else self._get_fallback_result(prompt_type)
            
            # Call the appropriate API
            if selected_provider == "openai":
                result = await self._call_openai_api(
                    formatted_prompt, 
                    selected_model,
                    temperature=temperature_override
                )
                input_tokens, output_tokens = self._estimate_openai_tokens(formatted_prompt, result)
            elif selected_provider == "anthropic":
                result = await self._call_anthropic_api(
                    formatted_prompt, 
                    selected_model,
                    temperature=temperature_override
                )
                input_tokens, output_tokens = self._estimate_anthropic_tokens(formatted_prompt, result)
            elif selected_provider == "azure":
                result = await self._call_azure_openai_api(
                    formatted_prompt, 
                    selected_model,
                    temperature=temperature_override
                )
                input_tokens, output_tokens = self._estimate_openai_tokens(formatted_prompt, result)
            else:
                self.logger.error(f"Unsupported provider: {selected_provider}")
                
                # Log the failure if using a tuned prompt
                if prompt_id:
                    event_bus.publish(
                        "llm_api_response",
                        {
                            "request_id": request_id,
                            "prompt_id": prompt_id,
                            "success": False,
                            "error": f"Unsupported provider: {selected_provider}"
                        }
                    )
                
                # Use fallback if available, otherwise return standard fallback
                return fallback_response if fallback_response else self._get_fallback_result(prompt_type)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log the API request as an event
            event_bus.publish(
                "llm_api_request",
                {
                    "request_id": request_id,
                    "provider": selected_provider,
                    "model": selected_model,
                    "operation": prompt_type,
                    "prompt_type": prompt_type,
                    "prompt_id": prompt_id,
                    "success": True,
                    "latency_ms": latency_ms,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            )
            
            # Parse and validate the result
            parsed_result = self._parse_llm_response(result, prompt_type)
            
            # Cache the result if caching is enabled
            if self.use_cached_responses:
                self.response_cache.set(cache_key, parsed_result)
            
            # Store as a potential fallback response for future failures
            await provider_failover_manager.store_fallback_response(prompt_type, input_text_hash, parsed_result)
            
            # Log successful response for the tuning system
            if prompt_id:
                event_bus.publish(
                    "llm_api_response",
                    {
                        "request_id": request_id,
                        "prompt_id": prompt_id,
                        "success": True,
                        "latency_ms": latency_ms
                    }
                )
            
            # Log result for A/B testing if applicable
            if experiment_id and variant_id:
                # Create result data for the experiment
                result_data = {
                    "success": True,
                    "latency_ms": latency_ms,
                    "experiment_id": experiment_id,
                    "variant_id": variant_id,
                    "request_id": request_id,
                    "sentiment_accuracy": None,  # Will be updated by performance tracker later
                    "direction_accuracy": None,
                    "confidence_score": parsed_result.get("confidence", 0.0) if isinstance(parsed_result, dict) else 0.0
                }
                
                # Publish experiment result
                event_bus.publish(
                    "experiment_result",
                    result_data
                )
            
            return parsed_result
            
        except Exception as e:
            # Calculate latency even for errors
            latency_ms = (time.time() - start_time) * 1000
            
            # Log the API error as an event
            event_bus.publish(
                "llm_api_error",
                {
                    "request_id": request_id,
                    "provider": selected_provider,
                    "model": selected_model,
                    "operation": prompt_type,
                    "prompt_type": prompt_type,
                    "prompt_id": prompt_id,
                    "error": str(e),
                    "latency_ms": latency_ms
                }
            )
            
            # Log failure for the tuning system
            if prompt_id:
                event_bus.publish(
                    "llm_api_response",
                    {
                        "request_id": request_id,
                        "prompt_id": prompt_id,
                        "success": False,
                        "error": str(e)
                    }
                )
                
            # Log failure for A/B testing if applicable
            if experiment_id and variant_id:
                event_bus.publish(
                    "experiment_result",
                    {
                        "experiment_id": experiment_id,
                        "variant_id": variant_id,
                        "request_id": request_id,
                        "success": False,
                        "error": str(e),
                        "latency_ms": latency_ms
                    }
                )
            
            self.logger.error(f"Error analyzing text with {selected_provider}/{selected_model}: {str(e)}")
            
            # Use fallback if available, otherwise return standard fallback
            return fallback_response if fallback_response else self._get_fallback_result(prompt_type)
    
    async def _call_openai_api(self, prompt: str, model: str, temperature: Optional[float] = None) -> str:
        """Call the OpenAI API.
        
        Args:
            prompt: The formatted prompt
            model: The model to use
            temperature: Optional temperature override
            
        Returns:
            Raw API response content
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_keys['openai']}"
        }
        
        # Use provided temperature or default
        temp = temperature if temperature is not None else 0.1
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a financial analysis assistant specialized in cryptocurrency markets."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temp,  # Use specified temperature or default
            "response_format": {"type": "json_object"}  # Request JSON response
        }
        
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
        
        async with self.session.post(
            self.endpoints["openai"], 
            headers=headers, 
            json=payload,
            timeout=timeout
        ) as response:
            if response.status != 200:
                error_text = response.text()
                self.logger.error(f"OpenAI API error: {response.status} - {error_text}")
                raise Exception(f"API error: {response.status}")
            
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
    
    async def _call_anthropic_api(self, prompt: str, model: str, temperature: Optional[float] = None) -> str:
        """Call the Anthropic API.
        
        Args:
            prompt: The formatted prompt
            model: The model to use
            temperature: Optional temperature override
            
        Returns:
            Raw API response content
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_keys["anthropic"],
            "anthropic-version": "2023-06-01"
        }
        
        # Use provided temperature or default
        temp = temperature if temperature is not None else 0.1
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": temp
        }
        
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
        
        async with self.session.post(
            self.endpoints["anthropic"], 
            headers=headers, 
            json=payload,
            timeout=timeout
        ) as response:
            if response.status != 200:
                error_text = response.text()
                self.logger.error(f"Anthropic API error: {response.status} - {error_text}")
                raise Exception(f"API error: {response.status}")
            
            response_data = response.json()
            return response_data["content"][0]["text"]
    
    async def _call_azure_openai_api(self, prompt: str, model: str, temperature: Optional[float] = None) -> str:
        """Call the Azure OpenAI API.
        
        Args:
            prompt: The formatted prompt
            model: The model to use
            temperature: Optional temperature override
            
        Returns:
            Raw API response content
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # For Azure, we need to use deployment names, not model names
        # This uses the model name as the deployment name, adjust as needed
        deployment_name = config.get(f"apis.azure_openai.deployments.{model}", model)
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_keys["azure"]
        }
        
        endpoint = f"{self.endpoints['azure']}/openai/deployments/{deployment_name}/chat/completions?api-version=2023-05-15"
        
        # Use provided temperature or default
        temp = temperature if temperature is not None else 0.1
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a financial analysis assistant specialized in cryptocurrency markets."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temp,
            "response_format": {"type": "json_object"}
        }
        
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
        
        async with self.session.post(
            endpoint, 
            headers=headers, 
            json=payload,
            timeout=timeout
        ) as response:
            if response.status != 200:
                error_text = response.text()
                self.logger.error(f"Azure OpenAI API error: {response.status} - {error_text}")
                raise Exception(f"API error: {response.status}")
            
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
    
    def _parse_llm_response(self, response_text: str, prompt_type: str) -> Dict[str, Any]:
        """Parse and validate the LLM response.
        
        Args:
            response_text: Raw response from the API
            prompt_type: Type of prompt used
            
        Returns:
            Parsed and validated response
        """
        try:
            # Try to parse as JSON
            result = json.loads(response_text)
            
            # Basic validation based on prompt type
            if prompt_type == "sentiment_analysis":
                # Ensure required fields are present
                if "sentiment_value" not in result or "direction" not in result or "confidence" not in result:
                    self.logger.warning(f"Missing required fields in sentiment analysis result")
                    result = self._blend_with_fallback(result, self._get_fallback_result(prompt_type))
                
                # Validate and constrain values
                result["sentiment_value"] = max(0.0, min(1.0, float(result.get("sentiment_value", 0.5))))
                result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.7))))
                
                # Ensure direction is valid
                if result.get("direction") not in ["bullish", "bearish", "neutral"]:
                    # Derive direction from sentiment value
                    if result["sentiment_value"] > 0.6:
                        result["direction"] = "bullish"
                    elif result["sentiment_value"] < 0.4:
                        result["direction"] = "bearish"
                    else:
                        result["direction"] = "neutral"
            
            elif prompt_type == "event_detection":
                # Ensure required fields are present
                if "is_market_event" not in result:
                    self.logger.warning(f"Missing required fields in event detection result")
                    result = self._blend_with_fallback(result, self._get_fallback_result(prompt_type))
                
                # Convert severity to int if needed
                if "severity" in result and not isinstance(result["severity"], int):
                    try:
                        result["severity"] = int(result["severity"])
                    except (ValueError, TypeError):
                        result["severity"] = 0
                        
                # Constrain severity
                result["severity"] = max(0, min(10, result.get("severity", 0)))
                
                # Ensure credibility is a float between 0 and 1
                if "credibility" in result:
                    result["credibility"] = max(0.0, min(1.0, float(result.get("credibility", 0.5))))
            
            elif prompt_type == "impact_assessment":
                # Ensure required fields are present
                if "primary_impact_direction" not in result or "impact_magnitude" not in result:
                    self.logger.warning(f"Missing required fields in impact assessment result")
                    result = self._blend_with_fallback(result, self._get_fallback_result(prompt_type))
                
                # Validate and constrain values
                result["impact_magnitude"] = max(0.0, min(1.0, float(result.get("impact_magnitude", 0.5))))
                result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.7))))
                
                # Ensure direction is valid
                if result.get("primary_impact_direction") not in ["positive", "negative", "neutral"]:
                    result["primary_impact_direction"] = "neutral"
            
            # Add metadata
            result["_meta"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "prompt_type": prompt_type
            }
            
            return result
            
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON response: {response_text[:100]}...")
            return self._get_fallback_result(prompt_type)
        except Exception as e:
            self.logger.error(f"Error validating LLM response: {str(e)}")
            return self._get_fallback_result(prompt_type)
    
    def _blend_with_fallback(self, partial_result: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        """Blend a partial result with fallback values.
        
        Args:
            partial_result: The partial result from the LLM
            fallback: The fallback result
            
        Returns:
            Blended result
        """
        result = fallback.copy()
        # Only copy over values that exist in the partial result
        for key, value in partial_result.items():
            if value is not None:
                result[key] = value
        return result
        
    def get_provider_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all LLM providers.
        
        Returns:
            Dictionary of provider health statuses
        """
        return provider_failover_manager.get_provider_health_status()
    
    def _estimate_openai_tokens(self, prompt: str, response: str) -> Tuple[int, int]:
        """Estimate token usage for OpenAI models.
        
        Args:
            prompt: The input prompt
            response: The API response
            
        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        # Very rough estimation based on 1 token ≈ 4 characters
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4
        
        return input_tokens, output_tokens
    
    def _estimate_anthropic_tokens(self, prompt: str, response: str) -> Tuple[int, int]:
        """Estimate token usage for Anthropic models.
        
        Args:
            prompt: The input prompt
            response: The API response
            
        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        # Similar estimation for Anthropic
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4
        
        return input_tokens, output_tokens
        
    def _get_fallback_result(self, prompt_type: str) -> Dict[str, Any]:
        """Get a fallback result when the LLM call fails.
        
        Args:
            prompt_type: Type of prompt
            
        Returns:
            Fallback result
        """
        if prompt_type == "sentiment_analysis":
            return {
                "sentiment_value": 0.5,
                "direction": "neutral",
                "confidence": 0.3,
                "explanation": "Fallback response due to API error.",
                "key_points": ["No analysis available due to API error."]
            }
        elif prompt_type == "event_detection":
            return {
                "is_market_event": False,
                "event_type": None,
                "assets_affected": None,
                "severity": 0,
                "credibility": 0.0,
                "propagation_speed": "unknown",
                "explanation": "Fallback response due to API error."
            }
        elif prompt_type == "impact_assessment":
            return {
                "primary_impact_direction": "neutral",
                "impact_magnitude": 0.0,
                "confidence": 0.3,
                "estimated_duration": "unknown",
                "affected_assets": {},
                "reasoning": "Fallback response due to API error.",
                "risk_factors": ["API error prevented proper analysis."]
            }
        else:
            return {
                "error": "Unknown prompt type or API error",
                "confidence": 0.0
            }