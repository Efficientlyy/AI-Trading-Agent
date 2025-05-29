"""
LLM Oversight Module for AI Trading Agent.

This module implements a Large Language Model (LLM) based oversight layer
for the AI Trading Agent, providing enhanced market analysis, trading strategy
validation, and autonomous decision-making capabilities.
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import traceback
from datetime import datetime

from .oversight_actions import OversightAction, OversightResult

# Set up logger
logger = logging.getLogger(__name__)

class OversightLevel(Enum):
    """Levels of LLM oversight intervention."""
    MONITOR = "monitor"          # Passive monitoring only
    ADVISE = "advise"            # Provide insights but don't intervene
    APPROVE = "approve"          # Approve/reject decisions before execution
    OVERRIDE = "override"        # Can modify or override decisions
    AUTONOMOUS = "autonomous"    # Full autonomous control


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"
    CUSTOM = "custom"


class LLMOversight:
    """
    LLM-based oversight system for trading operations.
    
    This system acts as a meta-layer analyzing trading decisions, market conditions,
    and system performance to enhance autonomy and resilience.
    """
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model_name: str = "gpt-4",
        oversight_level: OversightLevel = OversightLevel.ADVISE,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        max_history_items: int = 10,
        callbacks: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the LLM Oversight system.
        
        Args:
            provider: The LLM provider to use
            model_name: Name of the model to use
            oversight_level: Level of oversight/intervention
            api_key: API key for the provider
            api_base: Base URL for API (for custom endpoints)
            max_tokens: Maximum tokens in LLM responses
            temperature: Temperature parameter for LLM (randomness)
            system_prompt: Custom system prompt to define LLM behavior
            max_history_items: Maximum conversation history items to retain
            callbacks: Callback functions for different oversight events
        """
        self.provider = provider
        self.model_name = model_name
        self.oversight_level = oversight_level
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_history_items = max_history_items
        self.callbacks = callbacks or {}
        
        # Initialize history storage
        self.conversation_history = []
        self.decision_history = []
        self.check_history = []
        
        # Initialize metrics
        self.metrics = {
            "allowed": 0,
            "modified": 0,
            "flagged": 0,
            "rejected": 0,
            "log_only": 0,
            "prompt_checks": 0,
            "response_checks": 0
        }
        
        # Initialize LLM client
        self.client = self._initialize_client()
        
        # Set up system prompt
        self.system_prompt = system_prompt or self._generate_default_system_prompt()
        
        logger.info(f"LLM Oversight initialized with provider={provider.value}, "
                   f"model={model_name}, level={oversight_level.value}")
    
    def _initialize_client(self) -> Any:
        """
        Initialize the LLM client based on the selected provider.
        
        Returns:
            The initialized client
        """
        try:
            if self.provider == LLMProvider.OPENAI:
                # Try to import openai
                try:
                    import openai
                    client = openai.OpenAI(api_key=self.api_key)
                    return client
                except ImportError:
                    logger.error("OpenAI package not installed. Please install with 'pip install openai'")
                    raise
                
            elif self.provider == LLMProvider.ANTHROPIC:
                # Try to import anthropic
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=self.api_key)
                    return client
                except ImportError:
                    logger.error("Anthropic package not installed. Please install with 'pip install anthropic'")
                    raise
                
            elif self.provider == LLMProvider.AZURE_OPENAI:
                # Try to import openai with Azure configuration
                try:
                    import openai
                    client = openai.AzureOpenAI(
                        api_key=self.api_key,
                        api_version="2023-05-15",
                        azure_endpoint=self.api_base or "https://your-resource-name.openai.azure.com"
                    )
                    return client
                except ImportError:
                    logger.error("OpenAI package not installed. Please install with 'pip install openai'")
                    raise
                
            elif self.provider == LLMProvider.LOCAL:
                # This would integrate with a locally hosted model
                logger.warning("Local LLM provider selected - functionality may be limited")
                # Example with a REST client for a local server
                import requests
                class LocalLLMClient:
                    def __init__(self, base_url):
                        self.base_url = base_url or "http://localhost:8000"
                    
                    def generate(self, prompt, **kwargs):
                        response = requests.post(
                            f"{self.base_url}/v1/completions",
                            json={"prompt": prompt, **kwargs}
                        )
                        return response.json()
                        
                return LocalLLMClient(self.api_base)
                
            elif self.provider == LLMProvider.CUSTOM:
                # Custom provider requires the client to be passed in separately
                logger.warning("Custom LLM provider selected - please set client manually with set_custom_client()")
                return None
                
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def set_custom_client(self, client: Any) -> None:
        """
        Set a custom LLM client for custom providers.
        
        Args:
            client: Custom client object
        """
        self.client = client
        logger.info("Custom LLM client set")
    
    def _generate_default_system_prompt(self) -> str:
        """
        Generate the default system prompt based on the oversight level.
        
        Returns:
            Default system prompt
        """
        base_prompt = (
            "You are an advanced AI assistant specialized in financial markets and algorithmic trading. "
            "Your role is to provide oversight for an autonomous trading system with the following responsibilities:\n\n"
            "1. Analyze market conditions and identify regime changes\n"
            "2. Validate trading decisions and risk parameters\n"
            "3. Detect anomalies in market data and system behavior\n"
            "4. Suggest strategy adjustments when necessary\n"
            "5. Provide explanations for market events and trading decisions\n\n"
        )
        
        if self.oversight_level == OversightLevel.MONITOR:
            role_prompt = (
                "Your role is MONITORING ONLY. Observe and analyze, but make no suggestions "
                "or interventions. Your observations will be logged for review."
            )
        elif self.oversight_level == OversightLevel.ADVISE:
            role_prompt = (
                "Your role is ADVISORY. Provide analysis and suggestions, but the trading "
                "system will make final decisions. Be informative but concise."
            )
        elif self.oversight_level == OversightLevel.APPROVE:
            role_prompt = (
                "Your role is APPROVAL. You will review trading decisions before execution "
                "and either approve or reject them with clear reasoning."
            )
        elif self.oversight_level == OversightLevel.OVERRIDE:
            role_prompt = (
                "Your role is OVERSIGHT WITH OVERRIDE capability. You can modify trading "
                "decisions when necessary, providing clear justification for any changes."
            )
        else:  # AUTONOMOUS
            role_prompt = (
                "Your role is AUTONOMOUS CONTROL. You have full authority to direct the trading "
                "strategy based on your analysis. Provide clear directives and explanations."
            )
        
        formatting_prompt = (
            "\n\nFormat your responses as JSON with keys for 'analysis', 'decision', and 'explanation'. "
            "Keep your analysis factual and concise. Base your decisions on quantitative evidence when possible. "
            "Always consider risk management as the highest priority."
        )
        
        return base_prompt + role_prompt + formatting_prompt
    
    def generate_response(self, prompt: str, include_history: bool = True) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            include_history: Whether to include conversation history
            
        Returns:
            Parsed response as a dictionary
        """
        try:
            messages = []
            
            # Add system prompt
            messages.append({"role": "system", "content": self.system_prompt})
            
            # Add conversation history if requested
            if include_history and self.conversation_history:
                # Only include the most recent items based on max_history_items
                for item in self.conversation_history[-self.max_history_items:]:
                    messages.append(item)
            
            # Add the current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Generate response based on provider
            if self.provider == LLMProvider.OPENAI or self.provider == LLMProvider.AZURE_OPENAI:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                response_text = response.choices[0].message.content
                
            elif self.provider == LLMProvider.ANTHROPIC:
                message = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                response_text = message.content[0].text
                
            elif self.provider == LLMProvider.LOCAL or self.provider == LLMProvider.CUSTOM:
                # This would depend on the specific local or custom implementation
                response = self.client.generate(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                response_text = response.get("content", "")
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Try to parse as JSON
            try:
                # Check if the response is already JSON or needs extraction
                if response_text.strip().startswith("{") and response_text.strip().endswith("}"):
                    parsed_response = json.loads(response_text)
                else:
                    # Try to extract JSON from text
                    import re
                    json_match = re.search(r'```json\s*({.+?})\s*```', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        parsed_response = json.loads(json_str)
                    else:
                        # If no JSON found, create a basic structure with the full text
                        parsed_response = {
                            "analysis": response_text,
                            "decision": None,
                            "explanation": "Response was not in JSON format"
                        }
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                parsed_response = {
                    "analysis": response_text,
                    "decision": None,
                    "explanation": "Failed to parse response as JSON"
                }
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "analysis": "Error generating response",
                "decision": None,
                "explanation": f"Error: {str(e)}"
            }
    
    def analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market conditions and identify regimes.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Analysis results including regime identification
        """
        prompt = self._create_market_analysis_prompt(market_data)
        return self.generate_response(prompt)
    
    def validate_trading_decision(
        self, 
        decision: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate a trading decision based on current context.
        
        Args:
            decision: The trading decision to validate
            context: Current market and portfolio context
            
        Returns:
            Validation results including approval/rejection
        """
        prompt = self._create_decision_validation_prompt(decision, context)
        response = self.generate_response(prompt)
        
        # Store decision in history
        self.decision_history.append({
            "timestamp": time.time(),
            "decision": decision,
            "validation": response,
            "context": context
        })
        
        # Trigger callback if available
        if "on_decision_validation" in self.callbacks:
            try:
                self.callbacks["on_decision_validation"](decision, response, context)
            except Exception as e:
                logger.error(f"Error in decision validation callback: {str(e)}")
        
        return response
    
    def detect_anomalies(self, data: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in market data or system behavior.
        
        Args:
            data: Data to analyze for anomalies
            thresholds: Threshold parameters for anomaly detection
            
        Returns:
            Anomaly detection results
        """
        prompt = self._create_anomaly_detection_prompt(data, thresholds)
        return self.generate_response(prompt)
    
    def explain_market_event(self, event: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide natural language explanation for market events.
        
        Args:
            event: The market event to explain
            context: Additional context information
            
        Returns:
            Explanation of the market event
        """
        prompt = self._create_event_explanation_prompt(event, context)
        return self.generate_response(prompt)
    
    def suggest_strategy_adjustments(
        self,
        current_strategy: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest adjustments to trading strategy based on performance and conditions.
        
        Args:
            current_strategy: Current trading strategy configuration
            performance_metrics: Recent performance metrics
            market_conditions: Current market conditions
            
        Returns:
            Suggested strategy adjustments
        """
        prompt = self._create_strategy_adjustment_prompt(
            current_strategy, performance_metrics, market_conditions
        )
        return self.generate_response(prompt)
    
    def _create_market_analysis_prompt(self, market_data: Dict[str, Any]) -> str:
        """Create prompt for market analysis."""
        prompt = (
            "Please analyze the following market data and identify the current market regime "
            "or conditions. Consider volatility, trend strength, correlation patterns, and any "
            "anomalies.\n\n"
        )
        
        # Add formatted market data
        prompt += "Market Data:\n"
        prompt += json.dumps(market_data, indent=2)
        
        prompt += "\n\nProvide your analysis of current market conditions, regime identification, "
        prompt += "and any significant observations that would impact trading decisions."
        
        return prompt
    
    def _create_decision_validation_prompt(
        self, decision: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Create prompt for decision validation."""
        prompt = (
            "Please review and validate the following trading decision based on the provided context. "
            "Consider whether the decision aligns with current market conditions, follows risk management "
            "rules, and is justifiable based on the trading strategy.\n\n"
        )
        
        # Add decision and context
        prompt += "Trading Decision:\n"
        prompt += json.dumps(decision, indent=2)
        
        prompt += "\n\nContext:\n"
        prompt += json.dumps(context, indent=2)
        
        prompt += "\n\nValidate this trading decision. Should it be approved, modified, or rejected? "
        prompt += "Provide specific reasoning for your assessment and any recommended modifications."
        
        return prompt
    
    def _create_anomaly_detection_prompt(
        self, data: Dict[str, Any], thresholds: Dict[str, Any]
    ) -> str:
        """Create prompt for anomaly detection."""
        prompt = (
            "Please analyze the following data for anomalies or outliers that may indicate "
            "unusual market conditions or system behavior. Compare against the provided thresholds "
            "and identify any values that warrant attention.\n\n"
        )
        
        # Add data and thresholds
        prompt += "Data to Analyze:\n"
        prompt += json.dumps(data, indent=2)
        
        prompt += "\n\nThresholds:\n"
        prompt += json.dumps(thresholds, indent=2)
        
        prompt += "\n\nIdentify any anomalies, outliers, or patterns that deviate significantly "
        prompt += "from expected values. For each anomaly, provide an assessment of its significance "
        prompt += "and potential impact on trading operations."
        
        return prompt
    
    def _create_event_explanation_prompt(
        self, event: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Create prompt for event explanation."""
        prompt = (
            "Please explain the following market event in natural language, making it understandable "
            "even to someone without deep financial knowledge. Connect the event to broader market "
            "context and potential implications for trading strategies.\n\n"
        )
        
        # Add event and context
        prompt += "Market Event:\n"
        prompt += json.dumps(event, indent=2)
        
        prompt += "\n\nContext:\n"
        prompt += json.dumps(context, indent=2)
        
        prompt += "\n\nProvide a clear explanation of what happened, why it might have occurred, "
        prompt += "and what the implications are for our trading system. Include any relevant historical "
        prompt += "comparisons or important factors to consider."
        
        return prompt
    
    def _create_strategy_adjustment_prompt(
        self,
        current_strategy: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> str:
        """Create prompt for strategy adjustment suggestions."""
        prompt = (
            "Based on recent performance metrics and current market conditions, please suggest "
            "potential adjustments to our trading strategy. Consider parameter tuning, risk management "
            "changes, or broader strategic shifts that might improve performance.\n\n"
        )
        
        # Add strategy, metrics, and market conditions
        prompt += "Current Strategy:\n"
        prompt += json.dumps(current_strategy, indent=2)
        
        prompt += "\n\nPerformance Metrics:\n"
        prompt += json.dumps(performance_metrics, indent=2)
        
        prompt += "\n\nMarket Conditions:\n"
        prompt += json.dumps(market_conditions, indent=2)
        
        prompt += "\n\nProvide specific, actionable strategy adjustment recommendations. For each suggestion, "
        prompt += "explain the rationale, expected impact, and implementation considerations. Prioritize "
        prompt += "suggestions that address the most significant performance issues or opportunities."
        
        return prompt
        
    async def check_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OversightResult:
        """
        Check a prompt for compliance with trading policies.
        
        This method analyzes a prompt to determine if it should be allowed,
        modified, or rejected based on the current oversight level and policies.
        
        Args:
            prompt: The prompt to check
            system_prompt: Optional system prompt to check
            metadata: Additional metadata about the request
            
        Returns:
            OversightResult with action and any modifications
        """
        # Initialize result
        result = OversightResult(action=OversightAction.ALLOW)
        
        # Add metadata
        if metadata:
            result.metadata.update(metadata)
        result.metadata["timestamp"] = datetime.now().isoformat()
        
        # Skip detailed checks for MONITOR level
        if self.oversight_level == OversightLevel.MONITOR:
            result.message = "Prompt logged only (monitor mode)"
            result.action = OversightAction.LOG_ONLY
            self._update_metrics(result.action)
            self._log_check("prompt", prompt, result)
            return result
        
        # Perform checks based on oversight level
        if self.oversight_level in [OversightLevel.APPROVE, OversightLevel.OVERRIDE]:
            # Check for potentially risky trading instructions
            risky_phrases = [
                "ignore risk management",
                "bypass safety",
                "maximum leverage",
                "all available capital",
                "circumvent compliance",
                "ignore regulations",
                "100% of portfolio",
                "guaranteed profit",
                "risk-free",
                "all in"
            ]
            
            found_risks = [phrase for phrase in risky_phrases if phrase.lower() in prompt.lower()]
            
            if found_risks:
                if self.oversight_level == OversightLevel.APPROVE:
                    # Reject in strict mode
                    result.action = OversightAction.REJECT
                    result.reason = f"Prompt contains risky trading instructions: {', '.join(found_risks)}"
                    result.message = "This prompt appears to request potentially risky trading actions that violate our risk management policies."
                else:
                    # Modify in override mode
                    result.action = OversightAction.MODIFY
                    result.reason = f"Removed risky instructions: {', '.join(found_risks)}"
                    
                    # Create a modified prompt
                    modified_prompt = prompt
                    for phrase in found_risks:
                        modified_prompt = modified_prompt.replace(phrase, "[REDACTED]")
                    
                    result.modified_prompt = modified_prompt
                    result.message = "The prompt was modified to remove potentially risky trading instructions."
        
        # Check for system prompt issues
        if system_prompt and self.oversight_level != OversightLevel.MONITOR:
            problematic_system_phrases = [
                "ignore previous instructions",
                "disregard safety",
                "ignore risk limits",
                "maximum risk",
                "bypass compliance"
            ]
            
            found_system_risks = [phrase for phrase in problematic_system_phrases if phrase.lower() in system_prompt.lower()]
            
            if found_system_risks:
                if self.oversight_level in [OversightLevel.APPROVE, OversightLevel.OVERRIDE]:
                    # Either reject or modify
                    if self.oversight_level == OversightLevel.APPROVE:
                        result.action = OversightAction.REJECT
                        result.reason = f"System prompt contains problematic instructions: {', '.join(found_system_risks)}"
                        result.message = "The system prompt appears to request overriding safety mechanisms."
                    else:
                        result.action = OversightAction.MODIFY
                        result.reason = f"Modified system prompt with problematic instructions: {', '.join(found_system_risks)}"
                        
                        # Create a modified system prompt
                        modified_system_prompt = system_prompt
                        for phrase in found_system_risks:
                            modified_system_prompt = modified_system_prompt.replace(phrase, "[REDACTED]")
                        
                        result.modified_system_prompt = modified_system_prompt
                        result.message = "The system prompt was modified to remove potentially unsafe instructions."
                else:
                    # Flag but allow in less strict modes
                    result.action = OversightAction.FLAG
                    result.reason = f"System prompt contains potentially problematic instructions: {', '.join(found_system_risks)}"
                    result.message = "The system prompt contains instructions that may override safety mechanisms. Proceeding but flagged for review."
        
        # Update metrics and log
        self._update_metrics(result.action)
        self._log_check("prompt", prompt, result, system_prompt=system_prompt)
        
        return result
    
    async def check_response(
        self,
        prompt: str,
        response: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OversightResult:
        """
        Check a model response for compliance with trading policies.
        
        This method analyzes a response to determine if it should be allowed,
        modified, or rejected based on the current oversight level and policies.
        
        Args:
            prompt: The original prompt
            response: The response to check
            system_prompt: Optional system prompt that was used
            metadata: Additional metadata about the request
            
        Returns:
            OversightResult with action and any modifications
        """
        # Initialize result
        result = OversightResult(action=OversightAction.ALLOW)
        
        # Add metadata
        if metadata:
            result.metadata.update(metadata)
        result.metadata["timestamp"] = datetime.now().isoformat()
        
        # Skip detailed checks for MONITOR level
        if self.oversight_level == OversightLevel.MONITOR:
            result.message = "Response logged only (monitor mode)"
            result.action = OversightAction.LOG_ONLY
            self._update_metrics(result.action)
            self._log_check("response", response, result, prompt=prompt)
            return result
        
        # Perform checks based on oversight level
        if self.oversight_level in [OversightLevel.APPROVE, OversightLevel.OVERRIDE, OversightLevel.ADVISE]:
            # Check for potentially unsafe trading recommendations
            unsafe_phrases = [
                "invest all your capital",
                "maximum leverage",
                "ignore risk limits",
                "guaranteed profit",
                "risk-free investment",
                "certain to increase",
                "no possibility of loss",
                "bypass compliance",
                "avoid regulatory"
            ]
            
            found_risks = [phrase for phrase in unsafe_phrases if phrase.lower() in response.lower()]
            
            if found_risks:
                if self.oversight_level == OversightLevel.APPROVE:
                    # Reject in strict mode
                    result.action = OversightAction.REJECT
                    result.reason = f"Response contains unsafe trading recommendations: {', '.join(found_risks)}"
                    result.message = "This response contains potentially unsafe trading recommendations that violate our risk management policies."
                elif self.oversight_level == OversightLevel.OVERRIDE:
                    # Modify in override mode
                    result.action = OversightAction.MODIFY
                    result.reason = f"Removed unsafe recommendations: {', '.join(found_risks)}"
                    
                    # Create a modified response with warnings
                    modified_response = response
                    for phrase in found_risks:
                        modified_response = modified_response.replace(
                            phrase, 
                            f"[REDACTED: '{phrase}' removed due to risk management policies]"
                        )
                    
                    # Add disclaimer
                    disclaimer = (
                        "\n\n[OVERSIGHT NOTICE: This response was modified to comply with risk management "
                        "policies. Always exercise caution and follow proper risk management practices "
                        "when making trading decisions.]"
                    )
                    modified_response += disclaimer
                    
                    result.modified_response = modified_response
                    result.message = "The response was modified to remove potentially unsafe trading recommendations."
                else:  # ADVISE level
                    # Flag for approval
                    result.action = OversightAction.FLAG
                    result.reason = f"Response contains potentially risky recommendations: {', '.join(found_risks)}"
                    result.message = "The response contains recommendations that may need human review before proceeding."
            
            # Check for missing risk disclaimers in financial advice
            financial_advice_indicators = [
                "recommend",
                "should consider",
                "advisable to",
                "suggest you",
                "good opportunity",
                "profitable to",
                "consider buying",
                "consider selling"
            ]
            
            disclaimer_phrases = [
                "not financial advice",
                "consult a financial advisor",
                "past performance is not indicative",
                "investment involves risk",
                "make your own investment decisions",
                "do your own research"
            ]
            
            has_financial_advice = any(indicator.lower() in response.lower() for indicator in financial_advice_indicators)
            has_disclaimer = any(phrase.lower() in response.lower() for phrase in disclaimer_phrases)
            
            if has_financial_advice and not has_disclaimer and self.oversight_level in [OversightLevel.OVERRIDE, OversightLevel.ADVISE]:
                # Add disclaimer if missing
                result.action = OversightAction.MODIFY
                result.reason = "Added missing risk disclaimer to financial advice"
                
                disclaimer = (
                    "\n\n[IMPORTANT DISCLAIMER: This information is for educational purposes only and not "
                    "financial advice. Past performance is not indicative of future results. All investments "
                    "involve risk, including the possible loss of principal. Always consult with a qualified "
                    "financial advisor before making any investment decisions.]"
                )
                
                result.modified_response = response + disclaimer
                result.message = "A risk disclaimer was added to the response containing financial advice."
        
        # Update metrics and log
        self._update_metrics(result.action)
        self._log_check("response", response, result, prompt=prompt)
        
        return result
    
    async def log_error(
        self,
        error: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an error that occurred during model generation.
        
        Args:
            error: Error message
            prompt: The prompt that caused the error
            system_prompt: Optional system prompt that was used
            metadata: Additional metadata about the request
        """
        logger.error(f"LLM generation error: {error}")
        
        # Log the error details
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "metadata": metadata or {}
        }
        
        self.check_history.append({
            "type": "error",
            "data": error_data
        })
        
        # Execute error callback if available
        if "on_error" in self.callbacks and callable(self.callbacks["on_error"]):
            try:
                self.callbacks["on_error"](error_data)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def _update_metrics(self, action: OversightAction) -> None:
        """Update oversight metrics based on action."""
        if action == OversightAction.ALLOW:
            self.metrics["allowed"] += 1
        elif action == OversightAction.MODIFY:
            self.metrics["modified"] += 1
        elif action == OversightAction.FLAG:
            self.metrics["flagged"] += 1
        elif action == OversightAction.REJECT:
            self.metrics["rejected"] += 1
        elif action == OversightAction.LOG_ONLY:
            self.metrics["log_only"] += 1
    
    def _log_check(
        self,
        check_type: str,
        content: str,
        result: OversightResult,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> None:
        """
        Log a prompt or response check.
        
        Args:
            check_type: Type of check ('prompt' or 'response')
            content: Content being checked
            result: Check result
            prompt: Original prompt (for response checks)
            system_prompt: System prompt used
        """
        # Update metrics
        if check_type == "prompt":
            self.metrics["prompt_checks"] += 1
        else:
            self.metrics["response_checks"] += 1
        
        # Create log entry
        log_entry = {
            "type": check_type,
            "content": content,
            "result": result.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        if prompt:
            log_entry["prompt"] = prompt
            
        if system_prompt:
            log_entry["system_prompt"] = system_prompt
        
        # Add to history
        self.check_history.append(log_entry)
        
        # Keep history at reasonable size
        if len(self.check_history) > self.max_history_items * 2:
            self.check_history = self.check_history[-self.max_history_items:]
        
        # Execute callback if available
        callback_name = f"on_{check_type}_check"
        if callback_name in self.callbacks and callable(self.callbacks[callback_name]):
            try:
                self.callbacks[callback_name](log_entry)
            except Exception as e:
                logger.error(f"Error in {callback_name} callback: {e}")
    
    async def get_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get oversight metrics for a time period.
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            model: Filter by model name
            
        Returns:
            Dictionary of metrics
        """
        # Filter history by date range if specified
        filtered_history = self.check_history
        
        if start_date or end_date or model:
            filtered_history = []
            for entry in self.check_history:
                entry_timestamp = entry.get("timestamp", "")
                entry_model = entry.get("result", {}).get("metadata", {}).get("model")
                
                include = True
                
                if start_date and entry_timestamp < start_date:
                    include = False
                    
                if end_date and entry_timestamp > end_date:
                    include = False
                    
                if model and entry_model != model:
                    include = False
                    
                if include:
                    filtered_history.append(entry)
        
        # Calculate metrics from filtered history
        metrics = {
            "total_checks": len(filtered_history),
            "prompt_checks": sum(1 for e in filtered_history if e.get("type") == "prompt"),
            "response_checks": sum(1 for e in filtered_history if e.get("type") == "response"),
            "errors": sum(1 for e in filtered_history if e.get("type") == "error"),
            "actions": {
                "allow": sum(1 for e in filtered_history if e.get("result", {}).get("action") == OversightAction.ALLOW.value),
                "modify": sum(1 for e in filtered_history if e.get("result", {}).get("action") == OversightAction.MODIFY.value),
                "flag": sum(1 for e in filtered_history if e.get("result", {}).get("action") == OversightAction.FLAG.value),
                "reject": sum(1 for e in filtered_history if e.get("result", {}).get("action") == OversightAction.REJECT.value),
                "log_only": sum(1 for e in filtered_history if e.get("result", {}).get("action") == OversightAction.LOG_ONLY.value)
            },
            "time_period": {
                "start": start_date,
                "end": end_date
            },
            "model": model
        }
        
        return metrics
