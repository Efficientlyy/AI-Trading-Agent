"""
Oversight Integration Module for AI Trading Agent.

This module integrates the LLM oversight capabilities with the existing
trading system, providing the necessary hooks and interfaces for
autonomous operation supervision.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
import threading
import json
import os
from pathlib import Path

from ai_trading_agent.oversight.llm_oversight import LLMOversight, OversightLevel, LLMProvider
from ai_trading_agent.common.enhanced_circuit_breaker import EnhancedCircuitBreaker, register_circuit_breaker
from ai_trading_agent.common.error_handling import TradingAgentError, ErrorCode, ErrorCategory, ErrorSeverity

# Set up logger
logger = logging.getLogger(__name__)


class OversightManager:
    """
    Manages the integration of LLM oversight with the trading system.
    
    This class serves as the central coordinator for autonomous oversight,
    connecting the LLM capabilities with existing trading components while
    ensuring proper error handling and resilience.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        oversight_level: OversightLevel = OversightLevel.ADVISE,
        llm_provider: LLMProvider = LLMProvider.OPENAI,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_autonomous_recovery: bool = True,
        max_decision_cache_size: int = 100,
        decision_log_path: Optional[str] = None
    ):
        """
        Initialize the Oversight Manager.
        
        Args:
            config_path: Path to oversight configuration file
            oversight_level: Default level of LLM oversight
            llm_provider: LLM provider to use
            model_name: Name of the model to use (if None, uses default from config)
            api_key: API key for the LLM provider (if None, attempts to read from env or config)
            enable_autonomous_recovery: Whether to enable autonomous recovery mechanisms
            max_decision_cache_size: Maximum number of decisions to cache
            decision_log_path: Path to log validation decisions (if None, uses default path)
        """
        self.config = self._load_config(config_path)
        
        # Extract configuration with defaults
        self.oversight_level = oversight_level
        self.llm_provider = llm_provider
        self.model_name = model_name or self.config.get("model_name", "gpt-4")
        self.api_key = api_key or self._get_api_key()
        self.enable_autonomous_recovery = enable_autonomous_recovery
        self.max_decision_cache_size = max_decision_cache_size
        
        # Set up decision logging
        self.decision_log_path = decision_log_path
        if self.decision_log_path is None and "decision_log_path" in self.config:
            self.decision_log_path = self.config["decision_log_path"]
            
        # Initialize the LLM oversight system
        self.llm_oversight = self._initialize_llm_oversight()
        
        # Create circuit breaker for the LLM service
        self.llm_circuit_breaker = EnhancedCircuitBreaker(
            name="llm_oversight",
            warning_threshold=3,
            failure_threshold=5,
            recovery_time_base=30.0,
            max_recovery_time=600.0,
            reset_timeout=300.0
        )
        register_circuit_breaker(self.llm_circuit_breaker)
        
        # Initialize decision cache
        self.decision_cache = []
        
        # Threading lock for thread safety
        self._lock = threading.RLock()
        
        # Statistics tracking
        self.stats = {
            "total_decisions_validated": 0,
            "decisions_approved": 0,
            "decisions_rejected": 0,
            "decisions_modified": 0,
            "llm_errors": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0
        }
        
        logger.info(f"Oversight Manager initialized with {llm_provider.value} "
                   f"and oversight level {oversight_level.value}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load oversight configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "model_name": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 1000,
            "system_prompt": None,
            "decision_log_path": "logs/oversight_decisions"
        }
        
        if config_path is None:
            logger.info("No config path provided, using default configuration")
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded oversight configuration from {config_path}")
                return {**default_config, **config}
        except Exception as e:
            logger.warning(f"Failed to load oversight config from {config_path}: {str(e)}")
            logger.warning("Using default configuration")
            return default_config
    
    def _get_api_key(self) -> Optional[str]:
        """
        Get API key from environment or config.
        
        Returns:
            API key string or None if not found
        """
        # Check config first
        if "api_key" in self.config:
            return self.config["api_key"]
        
        # Check environment variables based on provider
        if self.llm_provider == LLMProvider.OPENAI:
            return os.environ.get("OPENAI_API_KEY")
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            return os.environ.get("ANTHROPIC_API_KEY")
        elif self.llm_provider == LLMProvider.AZURE_OPENAI:
            return os.environ.get("AZURE_OPENAI_API_KEY")
            
        # If no key found, log warning
        provider_name = self.llm_provider.value if hasattr(self.llm_provider, 'value') else str(self.llm_provider)
        logger.warning(f"No API key found for provider {provider_name}")
        return None
    
    def _initialize_llm_oversight(self) -> LLMOversight:
        """
        Initialize the LLM oversight system.
        
        Returns:
            Initialized LLMOversight instance
        """
        # Set up callbacks
        callbacks = {
            "on_decision_validation": self._on_decision_validation
        }
        
        # Create the LLM oversight system
        system_prompt = self.config.get("system_prompt", None)
        temperature = self.config.get("temperature", 0.2)
        max_tokens = self.config.get("max_tokens", 1000)
        
        return LLMOversight(
            provider=self.llm_provider,
            model_name=self.model_name,
            oversight_level=self.oversight_level,
            api_key=self.api_key,
            api_base=self.config.get("api_base", None),
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            callbacks=callbacks
        )
    
    def _on_decision_validation(
        self, 
        decision: Dict[str, Any], 
        validation: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> None:
        """
        Callback for decision validation events.
        
        Args:
            decision: The original trading decision
            validation: The validation result from LLM
            context: The context provided for validation
        """
        # Update statistics
        with self._lock:
            self.stats["total_decisions_validated"] += 1
            
            # Determine validation result
            result = validation.get("decision", {})
            if isinstance(result, dict):
                action = result.get("action", "unknown").lower()
                if action == "approve":
                    self.stats["decisions_approved"] += 1
                elif action == "reject":
                    self.stats["decisions_rejected"] += 1
                elif action == "modify":
                    self.stats["decisions_modified"] += 1
            
            # Log the decision if enabled
            if self.decision_log_path:
                self._log_decision(decision, validation, context)
    
    def _log_decision(
        self, 
        decision: Dict[str, Any], 
        validation: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> None:
        """
        Log a validated decision to file.
        
        Args:
            decision: The original trading decision
            validation: The validation result from LLM
            context: The context provided for validation
        """
        try:
            # Create log directory if it doesn't exist
            log_dir = Path(self.decision_log_path)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log entry
            log_entry = {
                "timestamp": time.time(),
                "decision": decision,
                "validation": validation,
                "context": context
            }
            
            # Generate filename based on timestamp
            filename = f"decision_{int(time.time())}.json"
            filepath = log_dir / filename
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(log_entry, f, indent=2)
                
            logger.debug(f"Logged decision validation to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to log decision validation: {str(e)}")
    
    def validate_trading_decision(
        self, 
        decision: Dict[str, Any], 
        context: Dict[str, Any],
        bypass_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Validate a trading decision using LLM oversight.
        
        Args:
            decision: Trading decision to validate
            context: Current market and portfolio context
            bypass_on_error: Whether to bypass validation on LLM errors
            
        Returns:
            Validation result with approval status and reasoning
        """
        try:
            # Use circuit breaker to protect against LLM service failures
            if not self.llm_circuit_breaker.is_allowed():
                logger.warning("LLM service circuit breaker is open - bypassing validation")
                return {
                    "decision": {"action": "approve", "reason": "LLM circuit breaker open"},
                    "analysis": "Validation bypassed due to LLM service issues",
                    "explanation": "The LLM oversight service is currently unavailable. Decision approved by default."
                }
            
            # Attempt validation with LLM
            validation_result = self.llm_oversight.validate_trading_decision(decision, context)
            
            # Record success with circuit breaker
            self.llm_circuit_breaker.record_success()
            
            # Cache the decision and validation
            self._cache_decision(decision, validation_result, context)
            
            return validation_result
            
        except Exception as e:
            # Record failure with circuit breaker
            self.llm_circuit_breaker.record_failure()
            
            # Update statistics
            with self._lock:
                self.stats["llm_errors"] += 1
            
            # Log the error
            logger.error(f"Error validating trading decision: {str(e)}")
            
            # Return bypass response if enabled
            if bypass_on_error:
                return {
                    "decision": {"action": "approve", "reason": "LLM validation error"},
                    "analysis": "Validation error",
                    "explanation": f"Error during LLM validation: {str(e)}. Decision approved by default."
                }
            else:
                # If bypass not enabled, raise the exception
                if isinstance(e, TradingAgentError):
                    raise e
                else:
                    raise TradingAgentError(
                        message=f"LLM validation error: {str(e)}",
                        error_code=ErrorCode.SYSTEM_DEPENDENCY_ERROR,
                        error_category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.ERROR,
                        cause=e
                    )
    
    def analyze_market_conditions(
        self, 
        market_data: Dict[str, Any],
        bypass_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze market conditions using LLM.
        
        Args:
            market_data: Market data to analyze
            bypass_on_error: Whether to return empty result on errors
            
        Returns:
            Market analysis results
        """
        try:
            # Use circuit breaker to protect against LLM service failures
            if not self.llm_circuit_breaker.is_allowed():
                logger.warning("LLM service circuit breaker is open - bypassing market analysis")
                return {
                    "analysis": "Analysis bypassed due to LLM service issues",
                    "market_regime": "unknown",
                    "explanation": "The LLM oversight service is currently unavailable."
                }
            
            # Attempt analysis with LLM
            analysis_result = self.llm_oversight.analyze_market_conditions(market_data)
            
            # Record success with circuit breaker
            self.llm_circuit_breaker.record_success()
            
            return analysis_result
            
        except Exception as e:
            # Record failure with circuit breaker
            self.llm_circuit_breaker.record_failure()
            
            # Update statistics
            with self._lock:
                self.stats["llm_errors"] += 1
            
            # Log the error
            logger.error(f"Error analyzing market conditions: {str(e)}")
            
            # Return empty response if bypass enabled
            if bypass_on_error:
                return {
                    "analysis": f"Error during analysis: {str(e)}",
                    "market_regime": "unknown",
                    "explanation": "Market analysis failed due to LLM service error."
                }
            else:
                # If bypass not enabled, raise the exception
                if isinstance(e, TradingAgentError):
                    raise e
                else:
                    raise TradingAgentError(
                        message=f"LLM market analysis error: {str(e)}",
                        error_code=ErrorCode.SYSTEM_DEPENDENCY_ERROR,
                        error_category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.ERROR,
                        cause=e
                    )
    
    def suggest_strategy_adjustments(
        self,
        current_strategy: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        market_conditions: Dict[str, Any],
        bypass_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Suggest strategy adjustments using LLM analysis.
        
        Args:
            current_strategy: Current trading strategy configuration
            performance_metrics: Recent performance metrics
            market_conditions: Current market conditions
            bypass_on_error: Whether to return empty suggestions on errors
            
        Returns:
            Strategy adjustment suggestions
        """
        try:
            # Use circuit breaker to protect against LLM service failures
            if not self.llm_circuit_breaker.is_allowed():
                logger.warning("LLM service circuit breaker is open - bypassing strategy suggestions")
                return {
                    "analysis": "Strategy suggestions bypassed due to LLM service issues",
                    "suggestions": [],
                    "explanation": "The LLM oversight service is currently unavailable."
                }
            
            # Attempt strategy suggestions with LLM
            adjustment_result = self.llm_oversight.suggest_strategy_adjustments(
                current_strategy, performance_metrics, market_conditions
            )
            
            # Record success with circuit breaker
            self.llm_circuit_breaker.record_success()
            
            return adjustment_result
            
        except Exception as e:
            # Record failure with circuit breaker
            self.llm_circuit_breaker.record_failure()
            
            # Update statistics
            with self._lock:
                self.stats["llm_errors"] += 1
            
            # Log the error
            logger.error(f"Error generating strategy suggestions: {str(e)}")
            
            # Return empty response if bypass enabled
            if bypass_on_error:
                return {
                    "analysis": f"Error during suggestion generation: {str(e)}",
                    "suggestions": [],
                    "explanation": "Strategy adjustment suggestions failed due to LLM service error."
                }
            else:
                # If bypass not enabled, raise the exception
                if isinstance(e, TradingAgentError):
                    raise e
                else:
                    raise TradingAgentError(
                        message=f"LLM strategy suggestion error: {str(e)}",
                        error_code=ErrorCode.SYSTEM_DEPENDENCY_ERROR,
                        error_category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.ERROR,
                        cause=e
                    )
    
    def _cache_decision(
        self, 
        decision: Dict[str, Any], 
        validation: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> None:
        """
        Cache a decision for learning and analysis.
        
        Args:
            decision: The original trading decision
            validation: The validation result
            context: The validation context
        """
        with self._lock:
            # Create cache entry
            cache_entry = {
                "timestamp": time.time(),
                "decision": decision,
                "validation": validation,
                "context": context
            }
            
            # Add to cache and maintain max size
            self.decision_cache.append(cache_entry)
            if len(self.decision_cache) > self.max_decision_cache_size:
                self.decision_cache.pop(0)
    
    def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent decision history.
        
        Args:
            limit: Maximum number of decisions to return
            
        Returns:
            List of recent decisions with validation results
        """
        with self._lock:
            # Return the most recent decisions up to the limit
            return self.decision_cache[-limit:]
    
    def get_oversight_stats(self) -> Dict[str, Any]:
        """
        Get oversight statistics.
        
        Returns:
            Dictionary with oversight statistics
        """
        with self._lock:
            # Calculate approval rate
            total = self.stats["total_decisions_validated"]
            approval_rate = 0
            if total > 0:
                approval_rate = (self.stats["decisions_approved"] / total) * 100
                
            # Calculate error rate
            error_rate = 0
            if total > 0:
                error_rate = (self.stats["llm_errors"] / (total + self.stats["llm_errors"])) * 100
                
            # Calculate recovery success rate
            recovery_rate = 0
            if self.stats["recovery_attempts"] > 0:
                recovery_rate = (self.stats["successful_recoveries"] / self.stats["recovery_attempts"]) * 100
                
            # Return stats with calculated metrics
            return {
                **self.stats,
                "approval_rate": approval_rate,
                "error_rate": error_rate,
                "recovery_success_rate": recovery_rate,
                "circuit_breaker_status": self.llm_circuit_breaker.state.value,
                "oversight_level": self.oversight_level.value
            }
    
    def set_oversight_level(self, level: OversightLevel) -> None:
        """
        Set the oversight level.
        
        Args:
            level: New oversight level
        """
        with self._lock:
            if level != self.oversight_level:
                self.oversight_level = level
                logger.info(f"Oversight level changed to {level.value}")
                
                # Update the LLM system with new oversight level
                self.llm_oversight.oversight_level = level
                self.llm_oversight.system_prompt = self.llm_oversight._generate_default_system_prompt()
    
    def perform_recovery(self) -> bool:
        """
        Attempt to recover the LLM service if it's in a failed state.
        
        Returns:
            True if recovery was attempted, False otherwise
        """
        with self._lock:
            # Update statistics
            self.stats["recovery_attempts"] += 1
            
            # Check if circuit breaker is open
            if self.llm_circuit_breaker.state.value != "open":
                logger.info("LLM service is not in failed state, no recovery needed")
                return False
                
            # Attempt reset
            reset_attempted = self.llm_circuit_breaker.attempt_reset()
            if reset_attempted:
                logger.info("LLM service circuit breaker reset attempted")
                
                # Try a simple test request
                try:
                    test_result = self.llm_oversight.generate_response("Test request to verify service recovery.")
                    if test_result:
                        logger.info("LLM service recovery successful")
                        self.stats["successful_recoveries"] += 1
                        return True
                except Exception as e:
                    logger.error(f"LLM service recovery test failed: {str(e)}")
                    return False
                    
            return False


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create oversight manager
    manager = OversightManager(
        oversight_level=OversightLevel.ADVISE,
        llm_provider=LLMProvider.OPENAI
    )
    
    # Example trading decision and context
    decision = {
        "action": "BUY",
        "symbol": "AAPL",
        "quantity": 10,
        "price": 150.00,
        "order_type": "LIMIT",
        "time_in_force": "DAY",
        "strategy_name": "momentum_strategy"
    }
    
    context = {
        "market_conditions": {
            "market_regime": "bullish",
            "volatility": "moderate",
            "sector_performance": {
                "technology": 1.2,
                "healthcare": 0.8,
                "financials": -0.3
            }
        },
        "portfolio": {
            "cash": 10000.00,
            "equity": 50000.00,
            "positions": {
                "MSFT": {"quantity": 5, "avg_price": 280.50},
                "AMZN": {"quantity": 3, "avg_price": 3200.75}
            }
        },
        "risk_metrics": {
            "portfolio_beta": 1.2,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.15,
            "value_at_risk": -0.05
        }
    }
    
    # Validate a decision
    print("Validating trading decision...")
    result = manager.validate_trading_decision(decision, context)
    print(f"Validation result: {json.dumps(result, indent=2)}")
    
    # Get oversight stats
    print("\nOversight statistics:")
    stats = manager.get_oversight_stats()
    print(json.dumps(stats, indent=2))
