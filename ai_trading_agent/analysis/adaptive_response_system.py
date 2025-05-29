"""
Adaptive Response System Module

This module implements the adaptive response system that automatically adjusts
trading parameters and strategies based on detected market regimes.

Key capabilities:
- Strategy parameter adjustment based on market conditions
- Risk parameter modulation based on volatility regimes
- Trading behavior adaptation rules for different market environments
- Feedback loops for continuous learning and optimization
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
from datetime import datetime, timedelta
import json
import os

from ai_trading_agent.agent.market_regime import MarketRegimeType
from ai_trading_agent.analysis.enhanced_regime_classifier import EnhancedMarketRegimeClassifier

# Set up logger
logger = logging.getLogger(__name__)


class AdaptationType(Enum):
    """Types of adaptations that can be applied."""
    STRATEGY_PARAMETER = "strategy_parameter"
    RISK_PARAMETER = "risk_parameter"
    AGENT_WEIGHT = "agent_weight"
    ASSET_ALLOCATION = "asset_allocation"
    EXECUTION_TIMING = "execution_timing"
    TRADE_SIZING = "trade_sizing"
    CUSTOM = "custom"


class AdaptationRule:
    """Defines a rule for adapting behavior based on market conditions."""
    
    def __init__(
        self,
        name: str,
        condition: Dict[str, Any],
        adaptation: Dict[str, Any],
        adaptation_type: AdaptationType,
        priority: int = 5,
        cooldown_minutes: int = 60
    ):
        """
        Initialize an adaptation rule.
        
        Args:
            name: Descriptive name for the rule
            condition: Dictionary defining when this rule should trigger
                - regime: MarketRegimeType to match (optional)
                - confidence_min: Minimum confidence required (optional)
                - volatility_threshold: Volatility threshold (optional)
                - custom_condition: Custom condition function (optional)
            adaptation: Dictionary defining what should be changed
                - target: What to adapt (agent_id, strategy_name, parameter, etc.)
                - action: How to adapt (set, multiply, add, etc.)
                - value: Value to use in adaptation
                - min_value: Minimum allowed value (optional)
                - max_value: Maximum allowed value (optional)
            adaptation_type: Type of adaptation to apply
            priority: Rule priority (1-10, higher means more important)
            cooldown_minutes: Minimum time between applications of this rule
        """
        self.name = name
        self.condition = condition
        self.adaptation = adaptation
        self.adaptation_type = adaptation_type
        self.priority = priority
        self.cooldown_minutes = cooldown_minutes
        
        # Track rule applications
        self.last_applied = None
        self.application_count = 0
        self.success_count = 0
        
    def check_condition(self, context: Dict[str, Any]) -> bool:
        """
        Check if the rule's condition is met in the current context.
        
        Args:
            context: Current market and system context
            
        Returns:
            Boolean indicating if condition is met
        """
        # Check cooldown period
        if self.last_applied:
            elapsed = datetime.now() - self.last_applied
            if elapsed.total_seconds() < self.cooldown_minutes * 60:
                return False
                
        # Check regime match if specified
        if 'regime' in self.condition:
            required_regime = self.condition['regime']
            current_regime = context.get('current_regime')
            
            # Handle string or enum comparison
            if isinstance(required_regime, str) and isinstance(current_regime, str):
                if required_regime != current_regime:
                    return False
            elif hasattr(required_regime, 'value') and hasattr(current_regime, 'value'):
                if required_regime.value != current_regime.value:
                    return False
            else:
                # Mixing types or missing regime
                return False
        
        # Check confidence threshold if specified
        if 'confidence_min' in self.condition:
            if context.get('confidence', 0) < self.condition['confidence_min']:
                return False
        
        # Check volatility threshold if specified
        if 'volatility_threshold' in self.condition:
            threshold = self.condition['volatility_threshold']
            current_volatility = context.get('volatility', 0)
            comparison = self.condition.get('volatility_comparison', '>')
            
            if comparison == '>' and current_volatility <= threshold:
                return False
            elif comparison == '<' and current_volatility >= threshold:
                return False
            elif comparison == '==' and current_volatility != threshold:
                return False
        
        # Check custom condition if specified
        if 'custom_condition' in self.condition and callable(self.condition['custom_condition']):
            if not self.condition['custom_condition'](context):
                return False
        
        return True
    
    def apply_adaptation(self, target_system: Any) -> Dict[str, Any]:
        """
        Apply the adaptation to the target system.
        
        Args:
            target_system: System to which the adaptation should be applied
            
        Returns:
            Dictionary with adaptation results
        """
        try:
            self.application_count += 1
            self.last_applied = datetime.now()
            
            adaptation = self.adaptation
            target = adaptation['target']
            action = adaptation['action']
            value = adaptation['value']
            
            result = {
                "rule_name": self.name,
                "adaptation_type": self.adaptation_type.value,
                "target": target,
                "action": action,
                "value": value,
                "timestamp": self.last_applied.isoformat(),
                "success": False
            }
            
            # Strategy parameter adaptation
            if self.adaptation_type == AdaptationType.STRATEGY_PARAMETER:
                if hasattr(target_system, 'update_strategy_parameter'):
                    target_system.update_strategy_parameter(
                        strategy_name=target.get('strategy_name'),
                        parameter_name=target.get('parameter_name'),
                        new_value=value,
                        action_type=action
                    )
                    result["success"] = True
                    
            # Risk parameter adaptation
            elif self.adaptation_type == AdaptationType.RISK_PARAMETER:
                if hasattr(target_system, 'update_risk_parameter'):
                    target_system.update_risk_parameter(
                        parameter_name=target.get('parameter_name'),
                        new_value=value,
                        action_type=action
                    )
                    result["success"] = True
            
            # Agent weight adaptation
            elif self.adaptation_type == AdaptationType.AGENT_WEIGHT:
                if hasattr(target_system, 'update_agent_weight'):
                    target_system.update_agent_weight(
                        agent_id=target.get('agent_id'),
                        new_weight=value,
                        action_type=action
                    )
                    result["success"] = True
            
            # Asset allocation adaptation
            elif self.adaptation_type == AdaptationType.ASSET_ALLOCATION:
                if hasattr(target_system, 'update_asset_allocation'):
                    target_system.update_asset_allocation(
                        asset_id=target.get('asset_id'),
                        new_allocation=value,
                        action_type=action
                    )
                    result["success"] = True
                    
            # Execution timing adaptation
            elif self.adaptation_type == AdaptationType.EXECUTION_TIMING:
                if hasattr(target_system, 'update_execution_timing'):
                    target_system.update_execution_timing(
                        timing_parameter=target.get('timing_parameter'),
                        new_value=value,
                        action_type=action
                    )
                    result["success"] = True
            
            # Trade sizing adaptation
            elif self.adaptation_type == AdaptationType.TRADE_SIZING:
                if hasattr(target_system, 'update_position_sizing'):
                    target_system.update_position_sizing(
                        sizing_parameter=target.get('sizing_parameter'),
                        new_value=value,
                        action_type=action
                    )
                    result["success"] = True
            
            # Custom adaptation
            elif self.adaptation_type == AdaptationType.CUSTOM:
                custom_func = adaptation.get('custom_function')
                if custom_func and callable(custom_func):
                    custom_result = custom_func(target_system, target, value, action)
                    result["success"] = custom_result
                    result["custom_result"] = custom_result
            
            if result["success"]:
                self.success_count += 1
                logger.info(f"Applied adaptation rule '{self.name}': {action} {target} to {value}")
            else:
                logger.warning(f"Failed to apply adaptation rule '{self.name}'")
                
            return result
            
        except Exception as e:
            logger.error(f"Error applying adaptation rule '{self.name}': {e}")
            return {
                "rule_name": self.name,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary for serialization."""
        return {
            "name": self.name,
            "condition": {k: v for k, v in self.condition.items() if not callable(v)},
            "adaptation": {k: v for k, v in self.adaptation.items() if not callable(v)},
            "adaptation_type": self.adaptation_type.value,
            "priority": self.priority,
            "cooldown_minutes": self.cooldown_minutes,
            "last_applied": self.last_applied.isoformat() if self.last_applied else None,
            "application_count": self.application_count,
            "success_count": self.success_count
        }


class AdaptiveResponseSystem:
    """
    System for automatically adapting trading behavior based on market regimes.
    
    This system:
    1. Applies predefined adaptation rules based on market conditions
    2. Provides an extensible framework for custom adaptations
    3. Maintains a history of adaptations for analysis
    4. Includes feedback mechanisms to learn from outcomes
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adaptive response system.
        
        Args:
            config: Configuration dictionary with parameters
                - rules_dir: Directory for storing rule definitions
                - history_max_size: Maximum number of historical adaptations to store
                - enable_feedback: Whether to enable adaptation feedback loops
                - feedback_window_days: Days to look back for feedback analysis
                - default_rules: Whether to load default adaptation rules
        """
        default_config = {
            "rules_dir": "./config/adaptation_rules",
            "history_max_size": 1000,
            "enable_feedback": True,
            "feedback_window_days": 14,
            "default_rules": True
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Rules storage
        self.rules: Dict[str, AdaptationRule] = {}
        
        # Adaptation history
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Initialize rules directory
        rules_dir = self.config["rules_dir"]
        os.makedirs(rules_dir, exist_ok=True)
        
        # Load default rules if enabled
        if self.config["default_rules"]:
            self._load_default_rules()
            
        logger.info(f"Adaptive Response System initialized with {len(self.rules)} rules")
    
    def _load_default_rules(self) -> None:
        """Load default adaptation rules."""
        # Trending market rules
        self.register_rule(AdaptationRule(
            name="trending_up_momentum_boost",
            condition={"regime": MarketRegimeType.TRENDING_UP.value, "confidence_min": 0.7},
            adaptation={
                "target": {"strategy_name": "momentum_strategy", "parameter_name": "lookback_periods"},
                "action": "set",
                "value": 10
            },
            adaptation_type=AdaptationType.STRATEGY_PARAMETER,
            priority=7
        ))
        
        self.register_rule(AdaptationRule(
            name="trending_up_trend_following_weight",
            condition={"regime": MarketRegimeType.TRENDING_UP.value, "confidence_min": 0.7},
            adaptation={
                "target": {"agent_id": "trend_following_agent"},
                "action": "multiply",
                "value": 1.5
            },
            adaptation_type=AdaptationType.AGENT_WEIGHT,
            priority=8
        ))
        
        self.register_rule(AdaptationRule(
            name="trending_down_risk_reduction",
            condition={"regime": MarketRegimeType.TRENDING_DOWN.value, "confidence_min": 0.6},
            adaptation={
                "target": {"parameter_name": "max_position_size"},
                "action": "multiply",
                "value": 0.7
            },
            adaptation_type=AdaptationType.RISK_PARAMETER,
            priority=9
        ))
        
        # Volatile market rules
        self.register_rule(AdaptationRule(
            name="volatile_stop_widening",
            condition={"regime": MarketRegimeType.VOLATILE.value, "confidence_min": 0.6},
            adaptation={
                "target": {"parameter_name": "stop_loss_multiplier"},
                "action": "multiply",
                "value": 1.5
            },
            adaptation_type=AdaptationType.RISK_PARAMETER,
            priority=8
        ))
        
        self.register_rule(AdaptationRule(
            name="volatile_reduce_size",
            condition={"regime": MarketRegimeType.VOLATILE.value, "confidence_min": 0.7},
            adaptation={
                "target": {"sizing_parameter": "position_size_multiplier"},
                "action": "multiply",
                "value": 0.7
            },
            adaptation_type=AdaptationType.TRADE_SIZING,
            priority=9
        ))
        
        # Ranging market rules
        self.register_rule(AdaptationRule(
            name="ranging_mean_reversion_boost",
            condition={"regime": MarketRegimeType.RANGING.value, "confidence_min": 0.6},
            adaptation={
                "target": {"agent_id": "mean_reversion_agent"},
                "action": "multiply",
                "value": 1.5
            },
            adaptation_type=AdaptationType.AGENT_WEIGHT,
            priority=7
        ))
        
        # Calm market rules
        self.register_rule(AdaptationRule(
            name="calm_increase_exposure",
            condition={"regime": MarketRegimeType.CALM.value, "confidence_min": 0.7},
            adaptation={
                "target": {"parameter_name": "target_exposure"},
                "action": "multiply",
                "value": 1.2
            },
            adaptation_type=AdaptationType.RISK_PARAMETER,
            priority=6
        ))
        
        # Breakout rules
        self.register_rule(AdaptationRule(
            name="breakout_fast_execution",
            condition={"regime": MarketRegimeType.BREAKOUT.value, "confidence_min": 0.6},
            adaptation={
                "target": {"timing_parameter": "execution_delay_seconds"},
                "action": "multiply",
                "value": 0.5
            },
            adaptation_type=AdaptationType.EXECUTION_TIMING,
            priority=8
        ))
        
        logger.info("Loaded default adaptation rules")
    
    def register_rule(self, rule: AdaptationRule) -> bool:
        """
        Register a new adaptation rule.
        
        Args:
            rule: AdaptationRule instance to register
            
        Returns:
            Boolean indicating success
        """
        try:
            self.rules[rule.name] = rule
            self._save_rule_to_disk(rule)
            logger.info(f"Registered adaptation rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Error registering rule {rule.name}: {e}")
            return False
    
    def _save_rule_to_disk(self, rule: AdaptationRule) -> None:
        """Save rule definition to disk."""
        try:
            rule_path = os.path.join(self.config["rules_dir"], f"{rule.name}.json")
            
            # Convert rule to JSON-serializable dict
            rule_dict = rule.to_dict()
            
            with open(rule_path, 'w') as f:
                json.dump(rule_dict, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving rule {rule.name} to disk: {e}")
    
    def load_rules_from_disk(self) -> int:
        """
        Load rule definitions from disk.
        
        Returns:
            Number of rules loaded
        """
        count = 0
        rules_dir = self.config["rules_dir"]
        
        if not os.path.exists(rules_dir):
            logger.warning(f"Rules directory {rules_dir} does not exist")
            return 0
            
        for filename in os.listdir(rules_dir):
            if filename.endswith('.json'):
                try:
                    rule_path = os.path.join(rules_dir, filename)
                    with open(rule_path, 'r') as f:
                        rule_dict = json.load(f)
                    
                    # Convert string adaptation_type back to enum
                    adaptation_type_str = rule_dict.get("adaptation_type")
                    adaptation_type = None
                    for at in AdaptationType:
                        if at.value == adaptation_type_str:
                            adaptation_type = at
                            break
                    
                    if not adaptation_type:
                        logger.warning(f"Invalid adaptation type in rule {filename}")
                        continue
                    
                    # Create rule
                    rule = AdaptationRule(
                        name=rule_dict.get("name"),
                        condition=rule_dict.get("condition", {}),
                        adaptation=rule_dict.get("adaptation", {}),
                        adaptation_type=adaptation_type,
                        priority=rule_dict.get("priority", 5),
                        cooldown_minutes=rule_dict.get("cooldown_minutes", 60)
                    )
                    
                    # Set history if available
                    if rule_dict.get("last_applied"):
                        rule.last_applied = datetime.fromisoformat(rule_dict["last_applied"])
                    rule.application_count = rule_dict.get("application_count", 0)
                    rule.success_count = rule_dict.get("success_count", 0)
                    
                    # Register rule without saving (to avoid circular writes)
                    self.rules[rule.name] = rule
                    count += 1
                    
                except Exception as e:
                    logger.error(f"Error loading rule from {filename}: {e}")
        
        logger.info(f"Loaded {count} adaptation rules from disk")
        return count
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """
        Get all registered adaptation rules.
        
        Returns:
            List of rule dictionaries
        """
        return [rule.to_dict() for rule in self.rules.values()]
    
    def delete_rule(self, rule_name: str) -> bool:
        """
        Delete an adaptation rule.
        
        Args:
            rule_name: Name of rule to delete
            
        Returns:
            Boolean indicating success
        """
        try:
            if rule_name in self.rules:
                del self.rules[rule_name]
                
                # Remove from disk
                rule_path = os.path.join(self.config["rules_dir"], f"{rule_name}.json")
                if os.path.exists(rule_path):
                    os.remove(rule_path)
                
                logger.info(f"Deleted rule {rule_name}")
                return True
            else:
                logger.warning(f"Rule {rule_name} not found")
                return False
        except Exception as e:
            logger.error(f"Error deleting rule {rule_name}: {e}")
            return False
    
    def evaluate_adaptations(self, context: Dict[str, Any], target_system: Any) -> List[Dict[str, Any]]:
        """
        Evaluate all adaptation rules against the current context and apply matching ones.
        
        Args:
            context: Current market and system context
            target_system: System to apply adaptations to
            
        Returns:
            List of adaptation results
        """
        if not target_system:
            logger.error("No target system provided for adaptations")
            return []
            
        results = []
        matching_rules = []
        
        # Find all matching rules
        for rule_name, rule in self.rules.items():
            if rule.check_condition(context):
                matching_rules.append(rule)
        
        # Sort by priority (highest first)
        matching_rules.sort(key=lambda r: r.priority, reverse=True)
        
        # Apply each matching rule in priority order
        for rule in matching_rules:
            result = rule.apply_adaptation(target_system)
            results.append(result)
            
            # Add to history
            self.adaptation_history.append({
                **result,
                "context": {
                    "regime": context.get("current_regime"),
                    "confidence": context.get("confidence"),
                    "volatility": context.get("volatility")
                }
            })
            
            # Trim history if needed
            max_size = self.config["history_max_size"]
            if len(self.adaptation_history) > max_size:
                self.adaptation_history = self.adaptation_history[-max_size:]
        
        logger.info(f"Applied {len(results)} adaptation rules based on current context")
        return results
    
    def analyze_adaptation_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze the effectiveness of previous adaptations.
        
        Returns:
            Dictionary with effectiveness metrics
        """
        if not self.config["enable_feedback"] or not self.adaptation_history:
            return {}
            
        # Group adaptations by rule
        rule_metrics = {}
        
        for adaptation in self.adaptation_history:
            rule_name = adaptation.get("rule_name")
            if not rule_name:
                continue
                
            if rule_name not in rule_metrics:
                rule_metrics[rule_name] = {
                    "count": 0,
                    "success_count": 0,
                    "success_rate": 0.0
                }
                
            rule_metrics[rule_name]["count"] += 1
            if adaptation.get("success", False):
                rule_metrics[rule_name]["success_count"] += 1
        
        # Calculate success rates
        for metrics in rule_metrics.values():
            if metrics["count"] > 0:
                metrics["success_rate"] = metrics["success_count"] / metrics["count"]
        
        # Overall metrics
        total_adaptations = len(self.adaptation_history)
        successful_adaptations = sum(1 for a in self.adaptation_history if a.get("success", False))
        
        return {
            "rule_metrics": rule_metrics,
            "total_adaptations": total_adaptations,
            "successful_adaptations": successful_adaptations,
            "overall_success_rate": successful_adaptations / total_adaptations if total_adaptations > 0 else 0.0,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def get_adaptation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent adaptation history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of adaptation records
        """
        # Return most recent adaptations first
        return self.adaptation_history[-limit:][::-1] if self.adaptation_history else []
    
    def create_custom_adaptation_function(self, func: Callable) -> Callable:
        """
        Create a custom adaptation function wrapper.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """
        def wrapped_adaptation(target_system, target, value, action):
            try:
                return func(target_system, target, value, action)
            except Exception as e:
                logger.error(f"Error in custom adaptation function: {e}")
                return False
        
        return wrapped_adaptation
