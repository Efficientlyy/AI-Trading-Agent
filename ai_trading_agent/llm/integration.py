"""
Integration module for fine-tuned LLMs with the AI Trading Agent.

This module connects fine-tuned language models with the trading agent system,
enabling the use of specialized financial models for various trading tasks.
It also integrates with the LLM oversight system to ensure proper monitoring
and evaluation of model outputs.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime
import asyncio
from pathlib import Path
import uuid

from .fine_tuning.config import FineTuningConfig, ModelProvider
from .fine_tuning.prompt_templates import FinancialPromptTemplate, get_prompt_template, PromptTaskType

# Import the oversight module
from ..oversight.llm_oversight import LLMOversight, OversightLevel, OversightAction


logger = logging.getLogger(__name__)


class LLMIntegration:
    """
    Integration between fine-tuned LLMs and the AI Trading Agent.
    
    This class manages the connection between fine-tuned language models
    and the rest of the trading agent system, handling prompt creation,
    model inference, and oversight integration.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        oversight_level: OversightLevel = OversightLevel.MODERATE,
        cache_results: bool = True,
        default_model: Optional[str] = None
    ):
        """
        Initialize the LLM integration module.
        
        Args:
            config_path: Path to the fine-tuning configuration
            oversight_level: Level of oversight to apply to model outputs
            cache_results: Whether to cache model outputs for repeated queries
            default_model: Default model to use if not specified
        """
        self.config = self._load_config(config_path)
        self.oversight = LLMOversight(level=oversight_level)
        self.cache_results = cache_results
        self.result_cache = {}
        self.default_model = default_model or self.config.base_model
        self.clients = {}
        self._initialize_clients()
    
    def _load_config(self, config_path: Optional[str]) -> FineTuningConfig:
        """
        Load the fine-tuning configuration.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Loaded configuration
        """
        if config_path and os.path.exists(config_path):
            return FineTuningConfig.load(config_path)
        else:
            logger.warning("No config path provided or file not found. Using default configuration.")
            return FineTuningConfig()
    
    def _initialize_clients(self) -> None:
        """Initialize API clients for different model providers."""
        # OpenAI client
        if self.config.model_provider == ModelProvider.OPENAI:
            try:
                from openai import OpenAI
                self.clients["openai"] = OpenAI(api_key=self.config.api_key)
                logger.info("Initialized OpenAI client")
            except ImportError:
                logger.warning("OpenAI package not installed. OpenAI models will not be available.")
        
        # Anthropic client
        elif self.config.model_provider == ModelProvider.ANTHROPIC:
            try:
                import anthropic
                self.clients["anthropic"] = anthropic.Anthropic(api_key=self.config.api_key)
                logger.info("Initialized Anthropic client")
            except ImportError:
                logger.warning("Anthropic package not installed. Anthropic models will not be available.")
        
        # Azure OpenAI client
        elif self.config.model_provider == ModelProvider.AZURE_OPENAI:
            try:
                from openai import AzureOpenAI
                self.clients["azure_openai"] = AzureOpenAI(
                    api_key=self.config.api_key,
                    api_version=self.config.api_version or "2023-05-15",
                    azure_endpoint=self.config.api_base
                )
                logger.info("Initialized Azure OpenAI client")
            except ImportError:
                logger.warning("OpenAI package not installed. Azure OpenAI models will not be available.")
        
        # Hugging Face models
        elif self.config.model_provider in [ModelProvider.HUGGING_FACE, ModelProvider.LOCAL]:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                model_path = os.path.join(self.config.output_dir, "final_model")
                if os.path.exists(model_path):
                    # Load model and tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.config.fp16 else None,
                        device_map="auto"
                    )
                    
                    self.clients["huggingface"] = {
                        "model": model,
                        "tokenizer": tokenizer
                    }
                    logger.info(f"Initialized Hugging Face model from {model_path}")
                else:
                    logger.warning(f"Model not found at {model_path}. Local models will not be available.")
            except ImportError:
                logger.warning("Required packages not installed. Local models will not be available.")
    
    def create_prompt(
        self,
        task_type: Union[str, PromptTaskType],
        context: str,
        **kwargs
    ) -> Dict[str, str]:
        """
        Create a formatted prompt for a specific task.
        
        Args:
            task_type: Type of financial task
            context: Main context for the prompt
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary with system_prompt and user_prompt
        """
        # Get the appropriate template
        if isinstance(task_type, str):
            template = get_prompt_template(task_type)
        else:
            template = FinancialPromptTemplate(task_type)
        
        # Format the prompt with provided parameters
        return template.format(context=context, **kwargs)
    
    async def generate_async(
        self,
        prompt: Union[str, Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        task_id: Optional[str] = None,
        oversight_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text from the model asynchronously.
        
        Args:
            prompt: Input prompt or dictionary with system_prompt and user_prompt
            model: Model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            task_id: ID for the generation task
            oversight_metadata: Additional metadata for oversight
            
        Returns:
            Dictionary with generated text and metadata
        """
        # Generate a task ID if not provided
        if not task_id:
            task_id = str(uuid.uuid4())
        
        # Use default model if not specified
        model = model or self.default_model
        
        # Check cache if enabled
        cache_key = None
        if self.cache_results and isinstance(prompt, str):
            cache_key = f"{model}:{prompt}:{temperature}:{max_tokens}"
            if cache_key in self.result_cache:
                logger.info(f"Cache hit for prompt: {prompt[:30]}...")
                return self.result_cache[cache_key]
        
        # Format the prompt if necessary
        system_prompt = None
        user_prompt = None
        if isinstance(prompt, dict) and "system_prompt" in prompt and "user_prompt" in prompt:
            system_prompt = prompt["system_prompt"]
            user_prompt = prompt["user_prompt"]
        else:
            user_prompt = prompt
        
        # Prepare standard metadata
        metadata = {
            "task_id": task_id,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timestamp": datetime.now().isoformat(),
            "provider": self.config.model_provider.value
        }
        
        # Add additional oversight metadata
        if oversight_metadata:
            metadata.update(oversight_metadata)
        
        # Submit to oversight for pre-check
        oversight_result = await self.oversight.check_prompt(
            prompt=user_prompt,
            system_prompt=system_prompt,
            metadata=metadata
        )
        
        # Check if the prompt was rejected by oversight
        if oversight_result.action == OversightAction.REJECT:
            return {
                "text": oversight_result.message or "Prompt rejected by oversight system.",
                "metadata": metadata,
                "oversight": oversight_result.to_dict(),
                "success": False
            }
        
        # Modify prompt if needed
        if oversight_result.action == OversightAction.MODIFY:
            if oversight_result.modified_prompt:
                user_prompt = oversight_result.modified_prompt
            if oversight_result.modified_system_prompt:
                system_prompt = oversight_result.modified_system_prompt
        
        try:
            # Generate text based on provider
            if self.config.model_provider == ModelProvider.OPENAI:
                response = await self._generate_openai(model, system_prompt, user_prompt, temperature, max_tokens)
            elif self.config.model_provider == ModelProvider.ANTHROPIC:
                response = await self._generate_anthropic(model, system_prompt, user_prompt, temperature, max_tokens)
            elif self.config.model_provider == ModelProvider.AZURE_OPENAI:
                response = await self._generate_azure_openai(model, system_prompt, user_prompt, temperature, max_tokens)
            elif self.config.model_provider in [ModelProvider.HUGGING_FACE, ModelProvider.LOCAL]:
                response = await self._generate_local(model, system_prompt, user_prompt, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported model provider: {self.config.model_provider.value}")
            
            # Add metadata to response
            response["metadata"] = metadata
            
            # Submit to oversight for post-check
            oversight_result = await self.oversight.check_response(
                prompt=user_prompt,
                response=response["text"],
                system_prompt=system_prompt,
                metadata=metadata
            )
            
            # Add oversight result to response
            response["oversight"] = oversight_result.to_dict()
            
            # Replace response if oversight modified it
            if oversight_result.action == OversightAction.MODIFY and oversight_result.modified_response:
                response["text"] = oversight_result.modified_response
                response["was_modified"] = True
            
            # Add success flag
            response["success"] = True
            
            # Cache the result if enabled
            if self.cache_results and cache_key:
                self.result_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            error_response = {
                "text": f"Error generating text: {str(e)}",
                "metadata": metadata,
                "error": str(e),
                "success": False
            }
            
            # Log the error to oversight
            await self.oversight.log_error(
                error=str(e),
                prompt=user_prompt,
                system_prompt=system_prompt,
                metadata=metadata
            )
            
            return error_response
    
    async def _generate_openai(
        self,
        model: str,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate text using OpenAI."""
        client = self.clients.get("openai")
        if not client:
            raise ValueError("OpenAI client not initialized")
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Make API call
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "text": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    async def _generate_anthropic(
        self,
        model: str,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate text using Anthropic."""
        client = self.clients.get("anthropic")
        if not client:
            raise ValueError("Anthropic client not initialized")
        
        # Format prompt for Anthropic
        prompt = ""
        if system_prompt:
            prompt = f"{system_prompt}\n\n"
        
        prompt += f"Human: {user_prompt}\n\nAssistant:"
        
        # Make API call
        response = await asyncio.to_thread(
            client.completions.create,
            prompt=prompt,
            model=model,
            max_tokens_to_sample=max_tokens,
            temperature=temperature
        )
        
        return {
            "text": response.completion,
            "finish_reason": "stop",  # Anthropic doesn't provide this directly
            "usage": {}  # Anthropic doesn't provide token counts in the same way
        }
    
    async def _generate_azure_openai(
        self,
        model: str,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate text using Azure OpenAI."""
        client = self.clients.get("azure_openai")
        if not client:
            raise ValueError("Azure OpenAI client not initialized")
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Make API call
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "text": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    async def _generate_local(
        self,
        model: str,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate text using local Hugging Face model."""
        client = self.clients.get("huggingface")
        if not client:
            raise ValueError("Hugging Face client not initialized")
        
        model_obj = client["model"]
        tokenizer = client["tokenizer"]
        
        # Format prompt
        prompt = ""
        if system_prompt:
            prompt = f"{system_prompt}\n\n"
        prompt += user_prompt
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
        
        # Generate
        import torch
        with torch.no_grad():
            outputs = model_obj.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=inputs.input_ids.shape[1] + max_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=1
            )
        
        # Decode and remove the prompt
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return {
            "text": generated_text,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": len(outputs[0]) - inputs.input_ids.shape[1],
                "total_tokens": len(outputs[0])
            }
        }
    
    def generate(
        self,
        prompt: Union[str, Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        task_id: Optional[str] = None,
        oversight_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for generate_async.
        
        Args:
            prompt: Input prompt or dictionary with system_prompt and user_prompt
            model: Model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            task_id: ID for the generation task
            oversight_metadata: Additional metadata for oversight
            
        Returns:
            Dictionary with generated text and metadata
        """
        loop = asyncio.get_event_loop() if asyncio.get_event_loop_policy().get_event_loop().is_running() else asyncio.new_event_loop()
        return loop.run_until_complete(
            self.generate_async(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                task_id=task_id,
                oversight_metadata=oversight_metadata
            )
        )
    
    def generate_market_analysis(
        self,
        market: str,
        timeframe: str,
        context: str,
        indicators: Optional[str] = None,
        events: Optional[str] = None,
        prior_analysis: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate market analysis for the specified market and timeframe.
        
        Args:
            market: Market to analyze
            timeframe: Timeframe for analysis
            context: Market context information
            indicators: Technical/fundamental indicators to consider
            events: Recent relevant events
            prior_analysis: Previous analysis for continuity
            model: Model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with generated analysis and metadata
        """
        # Create the prompt
        prompt = self.create_prompt(
            task_type="market_analysis",
            context=context,
            market=market,
            timeframe=timeframe,
            indicators=indicators or "N/A",
            events=events or "N/A",
            prior_analysis=prior_analysis or "N/A"
        )
        
        # Add task-specific metadata
        metadata = {
            "task_type": "market_analysis",
            "market": market,
            "timeframe": timeframe
        }
        
        # Generate the analysis
        return self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            oversight_metadata=metadata
        )
    
    def generate_trading_strategy(
        self,
        market: str,
        timeframe: str,
        risk_tolerance: str,
        context: str,
        capital: Optional[str] = None,
        constraints: Optional[str] = None,
        prior_trades: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate a trading strategy for the specified market and parameters.
        
        Args:
            market: Market to trade
            timeframe: Trading timeframe
            risk_tolerance: Risk tolerance level
            context: Market context information
            capital: Available capital
            constraints: Trading constraints
            prior_trades: Previous trades for continuity
            model: Model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with generated strategy and metadata
        """
        # Create the prompt
        prompt = self.create_prompt(
            task_type="trading_strategy",
            context=context,
            market=market,
            timeframe=timeframe,
            risk_tolerance=risk_tolerance,
            capital=capital or "N/A",
            constraints=constraints or "N/A",
            prior_trades=prior_trades or "N/A"
        )
        
        # Add task-specific metadata
        metadata = {
            "task_type": "trading_strategy",
            "market": market,
            "timeframe": timeframe,
            "risk_tolerance": risk_tolerance
        }
        
        # Generate the strategy
        return self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            oversight_metadata=metadata
        )
    
    def generate_risk_assessment(
        self,
        portfolio: str,
        risk_factors: str,
        context: str,
        timeframe: Optional[str] = None,
        risk_metrics: Optional[str] = None,
        constraints: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate a risk assessment for the specified portfolio.
        
        Args:
            portfolio: Portfolio to assess
            risk_factors: Risk factors to consider
            context: Additional context information
            timeframe: Assessment timeframe
            risk_metrics: Specific risk metrics to evaluate
            constraints: Assessment constraints
            model: Model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with generated assessment and metadata
        """
        # Create the prompt
        prompt = self.create_prompt(
            task_type="risk_assessment",
            context=context,
            portfolio=portfolio,
            risk_factors=risk_factors,
            timeframe=timeframe or "N/A",
            risk_metrics=risk_metrics or "N/A",
            constraints=constraints or "N/A"
        )
        
        # Add task-specific metadata
        metadata = {
            "task_type": "risk_assessment",
            "portfolio_type": portfolio.split()[0] if portfolio else "Unknown"  # Just get the first word as type
        }
        
        # Generate the assessment
        return self.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            oversight_metadata=metadata
        )
    
    async def get_oversight_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get oversight metrics for a specified time period and model.
        
        Args:
            start_date: Start date for metrics (ISO format)
            end_date: End date for metrics (ISO format)
            model: Model to filter metrics for
            
        Returns:
            Dictionary with oversight metrics
        """
        return await self.oversight.get_metrics(
            start_date=start_date,
            end_date=end_date,
            model=model
        )
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self.result_cache.clear()
        logger.info("Result cache cleared")
