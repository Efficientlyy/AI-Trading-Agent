"""
Trading Orchestrator Adapter for LLM Oversight.

This module connects the LLM Oversight system with the trading orchestrator,
enabling autonomous oversight and decision validation within the
existing trading workflow.
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import threading
from enum import Enum

from ai_trading_agent.oversight.llm_oversight import OversightLevel
from ai_trading_agent.oversight.oversight_integration import OversightManager
from ai_trading_agent.common.enhanced_circuit_breaker import enhanced_circuit_breaker
from ai_trading_agent.common.error_handling import TradingAgentError, ErrorCode, ErrorCategory, ErrorSeverity

# Set up logger
logger = logging.getLogger(__name__)


class OrchestratorHookType(Enum):
    """Types of hooks into the trading orchestrator process."""
    PRE_PROCESS = "pre_process"           # Before main processing cycle
    POST_PROCESS = "post_process"         # After main processing cycle
    PRE_SIGNAL_GENERATION = "pre_signal"  # Before signal generation
    POST_SIGNAL_GENERATION = "post_signal"  # After signal generation
    PRE_ORDER_EXECUTION = "pre_execution"  # Before order execution
    POST_ORDER_EXECUTION = "post_execution"  # After order execution
    ERROR_HANDLER = "error_handler"       # When errors occur


class OversightHook:
    """
    Represents a hook point in the trading orchestrator workflow
    where the LLM oversight system can intervene or provide analysis.
    """
    
    def __init__(
        self,
        hook_type: OrchestratorHookType,
        callback: Callable,
        name: str,
        priority: int = 0,
        enabled: bool = True
    ):
        """
        Initialize an oversight hook.
        
        Args:
            hook_type: Type of hook
            callback: Callback function to execute at this hook point
            name: Unique name for this hook
            priority: Priority of this hook (higher values run first)
            enabled: Whether this hook is currently enabled
        """
        self.hook_type = hook_type
        self.callback = callback
        self.name = name
        self.priority = priority
        self.enabled = enabled
        
        # Statistics
        self.execution_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0
        self.last_execution_time = None
        self.last_error = None
    
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the hook callback.
        
        Returns:
            Result of the callback
        """
        if not self.enabled:
            logger.debug(f"Hook '{self.name}' is disabled, skipping execution")
            return None
            
        start_time = time.time()
        self.execution_count += 1
        
        try:
            result = self.callback(*args, **kwargs)
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.last_execution_time = execution_time
            return result
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Error executing hook '{self.name}': {str(e)}")
            return None


class TradingOversightAdapter:
    """
    Adapter that connects the LLM oversight system with the trading orchestrator.
    
    This adapter provides hooks into different stages of the trading process,
    allowing the LLM to analyze data, validate decisions, and influence the
    trading workflow in a controlled manner.
    """
    
    def __init__(
        self,
        oversight_manager: Optional[OversightManager] = None,
        config_path: Optional[str] = None,
        default_oversight_level: OversightLevel = OversightLevel.ADVISE,
        enable_pre_process_analysis: bool = True,
        enable_signal_validation: bool = True,
        enable_execution_validation: bool = True,
        enable_error_analysis: bool = True
    ):
        """
        Initialize the trading oversight adapter.
        
        Args:
            oversight_manager: Existing oversight manager instance (creates new one if None)
            config_path: Path to configuration file
            default_oversight_level: Default oversight level
            enable_pre_process_analysis: Whether to enable market analysis before processing
            enable_signal_validation: Whether to enable trading signal validation
            enable_execution_validation: Whether to enable order execution validation
            enable_error_analysis: Whether to enable error analysis
        """
        # Create or use provided oversight manager
        if oversight_manager is None:
            logger.info("Creating new OversightManager")
            self.oversight_manager = OversightManager(
                config_path=config_path,
                oversight_level=default_oversight_level
            )
        else:
            logger.info("Using provided OversightManager")
            self.oversight_manager = oversight_manager
        
        # Configuration
        self.enable_pre_process_analysis = enable_pre_process_analysis
        self.enable_signal_validation = enable_signal_validation
        self.enable_execution_validation = enable_execution_validation
        self.enable_error_analysis = enable_error_analysis
        
        # Initialize hooks registry
        self.hooks: Dict[OrchestratorHookType, List[OversightHook]] = {}
        for hook_type in OrchestratorHookType:
            self.hooks[hook_type] = []
            
        # Register default hooks based on configuration
        self._register_default_hooks()
        
        # Initialize orchestrator and hooks
        self._initialize_orchestrator_hooks()
        
        # Cache for the most recent analysis and recommendations
        self.latest_analysis = {}
        self.latest_recommendations = {}
        
        # Cache for performance metrics
        self.performance_metrics = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info("TradingOversightAdapter initialized")
    
    def _register_default_hooks(self) -> None:
        """Register default hooks based on configuration."""
        # Register pre-process analysis hook
        if self.enable_pre_process_analysis:
            self.register_hook(
                OversightHook(
                    hook_type=OrchestratorHookType.PRE_PROCESS,
                    callback=self._analyze_market_conditions,
                    name="market_analysis",
                    priority=10
                )
            )
        
        # Register signal validation hook
        if self.enable_signal_validation:
            self.register_hook(
                OversightHook(
                    hook_type=OrchestratorHookType.POST_SIGNAL_GENERATION,
                    callback=self._validate_trading_signals,
                    name="signal_validation",
                    priority=10
                )
            )
        
        # Register execution validation hook
        if self.enable_execution_validation:
            self.register_hook(
                OversightHook(
                    hook_type=OrchestratorHookType.PRE_ORDER_EXECUTION,
                    callback=self._validate_order_execution,
                    name="execution_validation",
                    priority=10
                )
            )
        
        # Register error analysis hook
        if self.enable_error_analysis:
            self.register_hook(
                OversightHook(
                    hook_type=OrchestratorHookType.ERROR_HANDLER,
                    callback=self._analyze_error,
                    name="error_analysis",
                    priority=10
                )
            )
    
    def _initialize_orchestrator_hooks(self) -> None:
        """Initialize the orchestrator hooks integration."""
        # This method would typically register the hooks with the orchestrator
        # The exact implementation depends on how the orchestrator accepts hooks
        logger.info("Orchestrator hooks initialized")
    
    def register_hook(self, hook: OversightHook) -> None:
        """
        Register a new oversight hook.
        
        Args:
            hook: The hook to register
        """
        with self._lock:
            self.hooks[hook.hook_type].append(hook)
            # Sort hooks by priority (highest first)
            self.hooks[hook.hook_type].sort(key=lambda h: h.priority, reverse=True)
            logger.info(f"Registered {hook.hook_type.value} hook: {hook.name}")
    
    def unregister_hook(self, hook_type: OrchestratorHookType, hook_name: str) -> bool:
        """
        Unregister a hook by name and type.
        
        Args:
            hook_type: Type of the hook
            hook_name: Name of the hook
            
        Returns:
            True if the hook was found and removed, False otherwise
        """
        with self._lock:
            for i, hook in enumerate(self.hooks[hook_type]):
                if hook.name == hook_name:
                    self.hooks[hook_type].pop(i)
                    logger.info(f"Unregistered {hook_type.value} hook: {hook_name}")
                    return True
            
            logger.warning(f"Hook not found: {hook_type.value}/{hook_name}")
            return False
    
    def execute_hooks(
        self, 
        hook_type: OrchestratorHookType, 
        *args, 
        **kwargs
    ) -> List[Any]:
        """
        Execute all hooks of a specific type.
        
        Args:
            hook_type: Type of hooks to execute
            *args: Arguments to pass to the hooks
            **kwargs: Keyword arguments to pass to the hooks
            
        Returns:
            List of results from all executed hooks
        """
        results = []
        
        with self._lock:
            for hook in self.hooks[hook_type]:
                try:
                    result = hook.execute(*args, **kwargs)
                    results.append((hook.name, result))
                except Exception as e:
                    logger.error(f"Error executing hook {hook.name}: {str(e)}")
                    results.append((hook.name, None))
        
        return results
    
    def get_hook_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all registered hooks.
        
        Returns:
            Dictionary of hook statistics by name
        """
        stats = {}
        
        with self._lock:
            for hook_type in self.hooks:
                for hook in self.hooks[hook_type]:
                    avg_time = 0.0
                    if hook.execution_count > 0:
                        avg_time = hook.total_execution_time / hook.execution_count
                        
                    stats[hook.name] = {
                        "type": hook_type.value,
                        "enabled": hook.enabled,
                        "priority": hook.priority,
                        "execution_count": hook.execution_count,
                        "error_count": hook.error_count,
                        "avg_execution_time": avg_time,
                        "last_execution_time": hook.last_execution_time,
                        "last_error": hook.last_error
                    }
        
        return stats
    
    @enhanced_circuit_breaker(
        name="market_analysis_cb",
        warning_threshold=3,
        failure_threshold=5,
        recovery_time_base=30.0
    )
    def _analyze_market_conditions(
        self, 
        market_data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze market conditions using LLM.
        
        Args:
            market_data: Current market data
            context: Additional context
            
        Returns:
            Market analysis results
        """
        logger.info("Analyzing market conditions with LLM oversight")
        
        # Add metadata to market data
        enhanced_data = {
            "market_data": market_data,
            "timestamp": time.time(),
            "analysis_type": "market_conditions"
        }
        
        # Call the oversight manager
        result = self.oversight_manager.analyze_market_conditions(enhanced_data)
        
        # Store the latest analysis
        with self._lock:
            self.latest_analysis["market_conditions"] = {
                "timestamp": time.time(),
                "data": market_data,
                "result": result
            }
        
        return result
    
    @enhanced_circuit_breaker(
        name="signal_validation_cb",
        warning_threshold=3,
        failure_threshold=5,
        recovery_time_base=30.0
    )
    def _validate_trading_signals(
        self, 
        signals: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate trading signals using LLM.
        
        Args:
            signals: Trading signals to validate
            context: Current market and portfolio context
            
        Returns:
            Validation results
        """
        logger.info("Validating trading signals with LLM oversight")
        
        # Prepare decision for validation
        decision = {
            "type": "trading_signals",
            "signals": signals,
            "timestamp": time.time()
        }
        
        # Add market analysis if available
        enhanced_context = dict(context)
        if "market_conditions" in self.latest_analysis:
            enhanced_context["market_analysis"] = self.latest_analysis["market_conditions"]["result"]
        
        # Call the oversight manager
        result = self.oversight_manager.validate_trading_decision(decision, enhanced_context)
        
        # Store the latest validation
        with self._lock:
            self.latest_analysis["signal_validation"] = {
                "timestamp": time.time(),
                "signals": signals,
                "result": result
            }
        
        return result
    
    @enhanced_circuit_breaker(
        name="order_validation_cb",
        warning_threshold=3,
        failure_threshold=5,
        recovery_time_base=30.0
    )
    def _validate_order_execution(
        self, 
        orders: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate order execution using LLM.
        
        Args:
            orders: Orders to validate before execution
            context: Current market and portfolio context
            
        Returns:
            Validation results
        """
        logger.info("Validating order execution with LLM oversight")
        
        # Prepare decision for validation
        decision = {
            "type": "order_execution",
            "orders": orders,
            "timestamp": time.time()
        }
        
        # Add market analysis and signal validation if available
        enhanced_context = dict(context)
        if "market_conditions" in self.latest_analysis:
            enhanced_context["market_analysis"] = self.latest_analysis["market_conditions"]["result"]
        if "signal_validation" in self.latest_analysis:
            enhanced_context["signal_validation"] = self.latest_analysis["signal_validation"]["result"]
        
        # Call the oversight manager
        result = self.oversight_manager.validate_trading_decision(decision, enhanced_context)
        
        # Store the latest validation
        with self._lock:
            self.latest_analysis["order_validation"] = {
                "timestamp": time.time(),
                "orders": orders,
                "result": result
            }
        
        return result
    
    def _analyze_error(
        self, 
        error: Union[TradingAgentError, Exception], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze an error using LLM.
        
        Args:
            error: The error to analyze
            context: Error context
            
        Returns:
            Error analysis results
        """
        logger.info("Analyzing error with LLM oversight")
        
        # Convert the error to a dictionary for analysis
        if isinstance(error, TradingAgentError):
            error_dict = error.to_dict()
        else:
            error_dict = {
                "message": str(error),
                "type": type(error).__name__,
                "traceback": context.get("traceback", "")
            }
        
        # Prepare prompt context
        analysis_context = {
            "error": error_dict,
            "system_state": context.get("system_state", {}),
            "timestamp": time.time()
        }
        
        # Create custom prompt for error analysis
        prompt = (
            f"Please analyze the following error that occurred in the trading system:\n\n"
            f"Error: {json.dumps(error_dict, indent=2)}\n\n"
            f"Context: {json.dumps(context, indent=2)}\n\n"
            f"Provide the following analysis:\n"
            f"1. What is the most likely cause of this error?\n"
            f"2. Is this a systemic issue or a one-time occurrence?\n"
            f"3. What are the potential impacts on the trading system?\n"
            f"4. What recommended actions should be taken to resolve this issue?\n"
            f"5. Are there any preventive measures to avoid similar errors in the future?"
        )
        
        try:
            # Use the LLM directly for error analysis
            result = self.oversight_manager.llm_oversight.generate_response(prompt)
            
            # Store the analysis
            with self._lock:
                self.latest_analysis["error_analysis"] = {
                    "timestamp": time.time(),
                    "error": error_dict,
                    "result": result
                }
                
            return result
        except Exception as e:
            logger.error(f"Error during LLM error analysis: {str(e)}")
            return {
                "analysis": f"Error analysis failed: {str(e)}",
                "recommendations": ["Manual investigation required due to error analysis failure."]
            }
    
    def suggest_strategy_adaptations(
        self,
        current_strategy: Dict[str, Any],
        recent_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest strategy adaptations based on recent performance.
        
        Args:
            current_strategy: Current trading strategy configuration
            recent_performance: Recent performance metrics
            
        Returns:
            Strategy adaptation suggestions
        """
        logger.info("Generating strategy adaptation suggestions with LLM oversight")
        
        # Get the latest market analysis if available
        market_conditions = {}
        if "market_conditions" in self.latest_analysis:
            market_conditions = self.latest_analysis["market_conditions"]["result"]
        
        # Call the oversight manager
        result = self.oversight_manager.suggest_strategy_adjustments(
            current_strategy, recent_performance, market_conditions
        )
        
        # Store the latest recommendations
        with self._lock:
            self.latest_recommendations["strategy_adaptations"] = {
                "timestamp": time.time(),
                "current_strategy": current_strategy,
                "performance": recent_performance,
                "result": result
            }
        
        return result
    
    def update_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update performance metrics for use in future analysis.
        
        Args:
            metrics: Performance metrics to update
        """
        with self._lock:
            self.performance_metrics = {
                **self.performance_metrics,
                **metrics,
                "last_updated": time.time()
            }
    
    def get_latest_analysis(self, analysis_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the latest analysis.
        
        Args:
            analysis_type: Optional type of analysis to get
            
        Returns:
            Latest analysis results
        """
        with self._lock:
            if analysis_type is not None:
                return self.latest_analysis.get(analysis_type, {})
            else:
                return self.latest_analysis
    
    def get_latest_recommendations(self, recommendation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the latest recommendations.
        
        Args:
            recommendation_type: Optional type of recommendations to get
            
        Returns:
            Latest recommendations
        """
        with self._lock:
            if recommendation_type is not None:
                return self.latest_recommendations.get(recommendation_type, {})
            else:
                return self.latest_recommendations
    
    def get_oversight_status(self) -> Dict[str, Any]:
        """
        Get the current status of the oversight system.
        
        Returns:
            Dictionary with oversight status information
        """
        with self._lock:
            # Get oversight stats
            oversight_stats = self.oversight_manager.get_oversight_stats()
            
            # Get hook stats
            hook_stats = self.get_hook_stats()
            
            # Count enabled hooks by type
            enabled_hooks = {}
            for hook_type in OrchestratorHookType:
                enabled_hooks[hook_type.value] = sum(
                    1 for hook in self.hooks[hook_type] if hook.enabled
                )
            
            # Prepare status
            status = {
                "oversight_level": self.oversight_manager.oversight_level.value,
                "enabled_hooks": enabled_hooks,
                "total_hooks": sum(len(hooks) for hooks in self.hooks.values()),
                "last_analysis_time": max(
                    [0] + [a.get("timestamp", 0) for a in self.latest_analysis.values()]
                ),
                "last_recommendation_time": max(
                    [0] + [r.get("timestamp", 0) for r in self.latest_recommendations.values()]
                ),
                "oversight_stats": oversight_stats,
                "hook_stats": hook_stats
            }
            
            return status
