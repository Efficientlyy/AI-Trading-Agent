"""
Adaptive Health-Integrated Trading Orchestrator.

This module extends the HealthIntegratedOrchestrator with advanced market regime classification
and adaptive response capabilities, enabling the system to detect market conditions
and dynamically adjust trading parameters and strategies based on changing market environments.

The enhanced implementation includes:
- Advanced market regime detection with ML-based classification
- Regime transition detection with early warning signals
- Dynamic adaptation system for strategy parameters
- Continuous learning and feedback mechanisms
"""

import logging
import time
import pandas as pd
import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from ai_trading_agent.agent.health_integrated_orchestrator import HealthIntegratedOrchestrator
from ai_trading_agent.agent.agent_definitions import BaseAgent, AgentStatus
from ai_trading_agent.agent.adaptive_manager import AdaptiveStrategyManager
from ai_trading_agent.agent.meta_strategy import DynamicAggregationMetaStrategy
from ai_trading_agent.market_regime import (
    MarketRegimeClassifier,
    MarketRegimeConfig,
    MarketRegimeType,
    VolatilityRegimeType
)
from ai_trading_agent.market_regime.temporal_patterns import TemporalPatternRecognition
from ai_trading_agent.risk.risk_orchestrator import RiskOrchestrator
from ai_trading_agent.oversight.client import OversightClient, OversightAction
from ai_trading_agent.analysis.enhanced_regime_classifier import EnhancedMarketRegimeClassifier
from ai_trading_agent.analysis.adaptive_response_system import AdaptiveResponseSystem, AdaptationRule, AdaptationType
from ai_trading_agent.analysis.regime_transition_detector import RegimeTransitionDetector
from ai_trading_agent.analysis.regime_adaptation_integrator import RegimeAdaptationIntegrator

# Set up logger
logger = logging.getLogger(__name__)


class AdaptiveHealthOrchestrator(HealthIntegratedOrchestrator):
    """
    Adaptive Health-Integrated Trading Orchestrator.
    
    Extends the HealthIntegratedOrchestrator with market regime classification
    and adaptive response capabilities, enabling the system to detect market
    conditions and dynamically adjust trading parameters and strategies.
    """
    
    def __init__(
        self,
        health_monitor=None,
        log_dir=None,
        heartbeat_interval=5.0,
        monitor_components=True,
        regime_config=None,
        temporal_pattern_enabled=True,
        adaptation_interval_minutes=60,
        base_portfolio_risk=0.02,
        max_position_size=0.20,
        enable_advanced_risk_management=True,
        enable_volatility_clustering=True,
        enable_correlation_optimization=True,
        enable_risk_parity=True,
        enable_llm_oversight=False,
        llm_oversight_service_url=None,
        llm_oversight_level="advise"
    ):
        """
        Initialize the adaptive health orchestrator.
        
        Args:
            health_monitor: Optional existing health monitor instance
            log_dir: Directory for health monitoring logs
            heartbeat_interval: Interval for orchestrator heartbeats in seconds
            monitor_components: Whether to monitor individual agents as components
            regime_config: Optional market regime configuration
            temporal_pattern_enabled: Whether to enable temporal pattern recognition
            adaptation_interval_minutes: How often to run the adaptation cycle (minutes)
            base_portfolio_risk: Base daily portfolio risk target (VaR)
            max_position_size: Maximum allowed position size as % of portfolio
            enable_advanced_risk_management: Whether to enable advanced risk management
            enable_volatility_clustering: Whether to use GARCH models for volatility
            enable_correlation_optimization: Whether to optimize for correlation
            enable_risk_parity: Whether to use risk parity position sizing
            enable_llm_oversight: Whether to enable LLM oversight integration
            llm_oversight_service_url: URL for the LLM oversight service
            llm_oversight_level: Level of LLM oversight (monitor, advise, approve, override)
        """
        super().__init__(
            health_monitor=health_monitor,
            log_dir=log_dir,
            heartbeat_interval=heartbeat_interval,
            monitor_components=monitor_components
        )
        
        # Market regime detection components
        self.regime_config = regime_config or MarketRegimeConfig()
        self.regime_classifier = MarketRegimeClassifier(self.regime_config)
        self.temporal_pattern_enabled = temporal_pattern_enabled
        
        if temporal_pattern_enabled:
            self.temporal_pattern = TemporalPatternRecognition()
        else:
            self.temporal_pattern = None
            
        # Adaptive response components
        self.adaptive_manager = AdaptiveStrategyManager()
        
        # Risk management components
        self.enable_advanced_risk_management = enable_advanced_risk_management
        if enable_advanced_risk_management:
            self.risk_orchestrator = RiskOrchestrator(
                base_portfolio_risk=base_portfolio_risk,
                max_position_size=max_position_size,
                enable_volatility_clustering=enable_volatility_clustering,
                enable_correlation_optimization=enable_correlation_optimization,
                enable_risk_parity=enable_risk_parity,
                enable_stress_detection=True
            )
            logger.info("Advanced Risk Management Adaptivity enabled")
        else:
            self.risk_orchestrator = None
            logger.info("Advanced Risk Management Adaptivity disabled")
        
        # Store market data
        self.market_data = {}
        self.current_regime = {
            "global": {
                "regime_type": MarketRegimeType.UNKNOWN,
                "volatility_type": VolatilityRegimeType.UNKNOWN,
                "confidence": 0.0,
                "last_updated": None
            }
        }
        
        # Track portfolio state
        self.portfolio_value = 0.0
        self.portfolio_positions = {}
        self.portfolio_drawdown = 0.0
        
        # Adaptation settings
        self.adaptation_interval_minutes = adaptation_interval_minutes
        self.last_adaptation_time = datetime.now()
        
        # LLM Oversight integration
        self.enable_llm_oversight = enable_llm_oversight
        self.llm_oversight_level = llm_oversight_level
        self.oversight_client = None
        
        if enable_llm_oversight:
            # Use environment variable or default service URL if not provided
            service_url = llm_oversight_service_url or os.environ.get(
                "LLM_OVERSIGHT_SERVICE_URL", "http://llm-oversight-service"
            )
            
            try:
                self.oversight_client = OversightClient(base_url=service_url)
                if self.oversight_client.check_health():
                    oversight_config = self.oversight_client.get_config()
                    logger.info(f"LLM Oversight initialized with level: {oversight_config.get('oversight_level', 'unknown')}")
                    logger.info(f"LLM Provider: {oversight_config.get('llm_provider', 'unknown')}")
                else:
                    logger.warning("LLM Oversight service health check failed. Service may be unavailable.")
            except Exception as e:
                logger.error(f"Failed to initialize LLM oversight client: {e}")
                self.oversight_client = None
        
        # Register asset groups
        self._register_default_asset_groups()
        
        logger.info("Adaptive Health Orchestrator initialized with market regime classification and risk management")
    
    def _register_default_asset_groups(self):
        """Register default asset groups for correlation analysis."""
        groups = {
            "major_indices": ["SPY", "QQQ", "IWM", "DIA"],
            "sectors": ["XLK", "XLF", "XLE", "XLV", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"],
            "bond_market": ["TLT", "IEF", "SHY", "HYG", "LQD"],
            "commodities": ["GLD", "SLV", "USO", "UNG"],
            "currencies": ["UUP", "FXE", "FXY", "FXB"]
        }
        
        for group_name, symbols in groups.items():
            self.regime_classifier.correlation_analyzer.register_asset_group(group_name, symbols)
            logger.debug(f"Registered asset group {group_name} with {len(symbols)} symbols")
    
    def register_market_data_source(self, data_provider_id: str, data_provider):
        """
        Register a market data provider to be used for regime detection.
        
        Args:
            data_provider_id: Identifier for the data provider
            data_provider: The data provider object (must implement get_market_data method)
        """
        if hasattr(data_provider, 'get_market_data'):
            self.agents[data_provider_id] = data_provider
            logger.info(f"Registered market data provider: {data_provider_id}")
        else:
            logger.error(f"Data provider {data_provider_id} does not implement get_market_data method")
    
    def run_cycle(self, external_inputs: Optional[Dict[str, List[Dict]]] = None) -> Dict[str, Any]:
        """
        Run a single orchestration cycle with advanced market regime detection and adaptation.
        
        Extends the base run_cycle method to include enhanced market regime detection,
        regime transition prediction, adaptive response, risk management, and LLM oversight
        before executing the standard agent cycle.
        
        Args:
            external_inputs: Optional external inputs for agents
            
        Returns:
            Dictionary with cycle results and metrics
        """
        cycle_start_time = time.time()
        
        # Check if it's time for adaptation
        current_time = datetime.now()
        time_since_last_adaptation = (current_time - self.last_adaptation_time).total_seconds() / 60
        
        if time_since_last_adaptation >= self.adaptation_interval_minutes:
            # Update market data
            self._update_market_data()
            
            # Use enhanced regime detection if enabled, otherwise fall back to basic detection
            if self.enable_enhanced_regime_detection and self.regime_integrator:
                # Perform comprehensive market analysis using enhanced classification
                analysis_results = self.regime_integrator.analyze_market_conditions(self.market_data)
                
                # Extract key information
                enhanced_regime_results = analysis_results.get('regimes', {})
                global_regime = analysis_results.get('global_regime', {})
                transition_signals = analysis_results.get('transitions', {})
                
                # Store results in standard format for compatibility
                regime_results = {
                    'global': {
                        'regime_type': global_regime.get('regime', MarketRegimeType.UNKNOWN.value),
                        'confidence': global_regime.get('confidence', 0.0),
                        'distribution': global_regime.get('distribution', {})
                    },
                    'assets': enhanced_regime_results,
                    'transitions': transition_signals
                }
                
                # Store transition signals for monitoring
                self.transition_signals = transition_signals
                
                logger.info(f"Enhanced market regime detection: {regime_results['global']['regime_type']} "
                           f"(confidence: {regime_results['global']['confidence']:.2f})")
                
                if transition_signals:
                    # Log potential regime transitions
                    transitions_count = sum(len(signals) for signals in transition_signals.values())
                    logger.info(f"Detected {transitions_count} potential regime transitions")
                    
                    # Extract high probability transitions for logging
                    high_prob_transitions = []
                    for symbol, signals in transition_signals.items():
                        for signal in signals:
                            if signal.get('probability', 0) > 0.7:
                                high_prob_transitions.append({
                                    'symbol': symbol,
                                    'from': signal.get('from_regime'),
                                    'to': signal.get('to_regime'),
                                    'probability': signal.get('probability'),
                                    'timeframe': signal.get('estimated_timeframe')
                                })
                    
                    if high_prob_transitions:
                        logger.warning(f"High probability transitions detected: {len(high_prob_transitions)}")
                        for t in high_prob_transitions[:3]:  # Log max 3 transitions
                            logger.warning(f"  {t['symbol']}: {t['from']} → {t['to']} "
                                          f"(p={t['probability']:.2f}, timeframe: {t['timeframe']})")
            else:
                # Use traditional regime detection
                regime_results = self._detect_market_regimes()
                global_regime = regime_results.get('global', {})
            
            # Enhance market analysis with LLM if enabled
            if self.enable_llm_oversight and self.oversight_client:
                try:
                    llm_analysis = self.analyze_market_conditions_with_llm()
                    if llm_analysis:
                        # Add LLM analysis insights to regime_results
                        regime_results["llm_analysis"] = llm_analysis
                        logger.info(f"LLM market analysis: {llm_analysis.get('summary', 'No summary available')}")
                except Exception as e:
                    logger.error(f"Error during LLM market analysis: {e}")
            
            # Apply adaptive responses using enhanced system if available
            if self.enable_enhanced_regime_detection and self.regime_integrator:
                # Use the enhanced adaptation system
                adaptation_results = self.regime_integrator.determine_adaptations(analysis_results, self)
                
                # Format results for compatibility with existing code
                actions = []
                for adaptation_name, adaptation_details in adaptation_results.get('active_adaptations', {}).items():
                    actions.append({
                        'type': 'regime_adaptation',
                        'name': adaptation_name,
                        'details': adaptation_details
                    })
                
                adaptation_results = {
                    'actions': actions,
                    'timestamp': adaptation_results.get('timestamp')
                }
            else:
                # Use traditional adaptation method
                adaptation_results = self._apply_adaptive_responses(regime_results)
            
            # Apply risk management adaptations if enabled
            if self.enable_advanced_risk_management and self.risk_orchestrator:
                # Get the current regime information
                regime_type = global_regime.get('regime_type')
                volatility_type = global_regime.get('volatility_type')
                
                # Update the risk orchestrator with market data
                if self.market_data:
                    self.risk_orchestrator.update_market_data(self.market_data)
                
                # Adapt risk parameters to the current market regime
                risk_params = self.risk_orchestrator.adapt_to_market_regime(
                    market_regime=regime_type,
                    volatility_regime=volatility_type,
                    drawdown=self.portfolio_drawdown
                )
                
                # Log risk adaptation
                logger.info(f"Risk parameters adapted to {regime_type} regime - "
                          f"Portfolio risk: {risk_params['portfolio_risk']:.1%}, "
                          f"Max position: {risk_params['max_position_size']:.1%}")
                
                # Add risk actions to adaptation results
                if 'actions' in adaptation_results:
                    adaptation_results['actions'].append({
                        'type': 'risk_adaptation',
                        'details': risk_params
                    })
                    
                # If LLM oversight is enabled, get strategy adjustment suggestions
                if self.enable_llm_oversight and self.oversight_client and self.market_data:
                    try:
                        # Get current strategy parameters
                        current_strategy = {
                            "risk_params": risk_params,
                            "adaptations": adaptation_results.get("actions", [])
                        }
                        
                        # Get performance metrics
                        performance_metrics = {
                            "drawdown": self.portfolio_drawdown,
                            "portfolio_value": self.portfolio_value,
                            "position_count": len(self.portfolio_positions or {})
                        }
                        
                        # Get strategy adjustment suggestions
                        strategy_suggestions = self.oversight_client.suggest_strategy_adjustments(
                            current_strategy=current_strategy,
                            performance_metrics=performance_metrics,
                            market_conditions=self.market_data
                        )
                        
                        if strategy_suggestions:
                            logger.info(f"LLM strategy suggestions received: {len(strategy_suggestions.get('suggestions', []))} suggestions")
                            adaptation_results["llm_suggestions"] = strategy_suggestions
                    except Exception as e:
                        logger.error(f"Error getting LLM strategy suggestions: {e}")
            
            self.last_adaptation_time = current_time
            
            # Log adaptation results
            logger.info(f"Market adaptation completed. "
                       f"Global regime: {global_regime.get('regime_type', 'unknown')}, "
                       f"Applied {len(adaptation_results.get('actions', []))} adaptation actions")
        
        # Track if we're using LLM oversight for this cycle
        using_oversight = self.enable_llm_oversight and self.oversight_client
        
        # Create validation context for the entire cycle
        validation_context = {
            "market_regime": self.current_regime,
            "portfolio": {
                "value": self.portfolio_value,
                "drawdown": self.portfolio_drawdown
            }
        }
        
        if self.enable_advanced_risk_management and self.risk_orchestrator:
            validation_context["risk_metrics"] = self.risk_orchestrator.get_risk_metrics()
        
        # If LLM oversight is enabled, wrap the parent's run_cycle to validate decisions
        if using_oversight:
            # Store the original external inputs
            original_inputs = external_inputs
            
            # If there are no external inputs, create an empty dict
            if external_inputs is None:
                external_inputs = {}
            
            # Process and validate trading decisions in external inputs
            for agent_id, decisions in external_inputs.items():
                if decisions and isinstance(decisions, list):
                    validated_decisions = []
                    for decision in decisions:
                        if isinstance(decision, dict) and 'action' in decision:
                            # This looks like a trading decision, validate it
                            is_approved, result = self.validate_trading_decision(decision, validation_context)
                            
                            if is_approved:
                                # If approved, keep the original decision or use modified version
                                if result.get('modified_decision') and self.llm_oversight_level in ["override", "autonomous"]:
                                    validated_decisions.append(result.get('modified_decision'))
                                    logger.info(f"Using LLM-modified decision for {agent_id}")
                                else:
                                    validated_decisions.append(decision)
                            else:
                                # If rejected, log the rejection
                                logger.warning(f"LLM oversight rejected decision from {agent_id}: {result.get('reason', 'No reason provided')}")
                                # In this case, we don't add the decision to validated_decisions
                        else:
                            # Not a trading decision, pass through unchanged
                            validated_decisions.append(decision)
                    
                    # Replace with validated decisions
                    external_inputs[agent_id] = validated_decisions
        
        # Run the standard orchestration cycle
        cycle_results = super().run_cycle(external_inputs)
        
        # Add LLM oversight metrics to results if used
        if using_oversight:
            cycle_results["llm_oversight"] = {
                "enabled": True,
                "level": self.llm_oversight_level
            }
        
        # Update metrics
        cycle_duration = time.time() - cycle_start_time
        cycle_results["cycle_duration"] = cycle_duration
        cycle_results["current_regime"] = self.current_regime.get("global", {}).get("regime_type", "UNKNOWN")
        
        # Add risk metrics if available
        if self.enable_advanced_risk_management and self.risk_orchestrator:
            cycle_results["risk_metrics"] = self.risk_orchestrator.get_risk_metrics()
        
        return cycle_results
    
    def _update_market_data(self) -> None:
        """Update market data from registered data providers."""
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'get_market_data'):
                try:
                    market_data = agent.get_market_data()
                    if market_data:
                        self.market_data.update(market_data)
                        logger.debug(f"Updated market data from {agent_id}: {len(market_data)} assets")
                except Exception as e:
                    logger.error(f"Error updating market data from {agent_id}: {str(e)}")
                    
                    # Record error in health monitoring
                    if self.monitor_components and self.health_monitor is not None:
                        self.health_monitor.add_alert(
                            component_id=agent_id,
                            severity="ERROR",
                            message=f"Market data update failed: {str(e)}"
                        )
    
    def _detect_market_regimes(self) -> Dict[str, Any]:
        """
        Detect current market regimes across assets.
        
        Returns:
            Dictionary with regime detection results
        """
        results = {"global": {}, "assets": {}}
        
        try:
            # Check if we have SPY data (or main market index) for global regime
            primary_asset = "SPY"
            if primary_asset in self.market_data:
                asset_data = self.market_data[primary_asset]
                
                # Prepare related assets data
                related_assets = {
                    ticker: data for ticker, data in self.market_data.items() 
                    if ticker != primary_asset
                }
                
                # Classify regime
                regime_info = self.regime_classifier.classify_regime(
                    prices=asset_data.get('prices'),
                    volumes=asset_data.get('volume'),
                    high_prices=asset_data.get('high'),
                    low_prices=asset_data.get('low'),
                    asset_id=primary_asset,
                    related_assets=related_assets
                )
                
                # Capture global regime
                global_regime = {
                    "regime_type": regime_info.regime_type.value,
                    "volatility_type": regime_info.volatility_type.value,
                    "confidence": regime_info.confidence,
                    "metrics": regime_info.metrics,
                    "last_updated": datetime.now()
                }
                
                self.current_regime["global"] = global_regime
                results["global"] = global_regime
                
                logger.info(f"Detected global market regime: {regime_info.regime_type.value} "
                           f"with {regime_info.volatility_type.value} volatility "
                           f"(confidence: {regime_info.confidence:.2f})")
                
                # Analyze temporal patterns if enabled
                if self.temporal_pattern_enabled and self.temporal_pattern:
                    temporal_results = self._analyze_temporal_patterns(primary_asset, asset_data)
                    results["temporal"] = temporal_results
            
            # Classify individual assets (future enhancement)
            # TODO: Add individual asset regime classification
                
        except Exception as e:
            logger.error(f"Error detecting market regimes: {str(e)}")
            if self.health_monitor:
                self.health_monitor.add_alert(
                    component_id="market_regime_classifier",
                    severity="ERROR",
                    message=f"Regime detection failed: {str(e)}"
                )
        
        return results
    
    def _analyze_temporal_patterns(self, asset_id: str, asset_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze temporal patterns for an asset.
        
        Args:
            asset_id: Asset identifier
            asset_data: Dictionary with asset data
            
        Returns:
            Dictionary with temporal analysis results
        """
        try:
            if self.temporal_pattern and 'prices' in asset_data:
                temporal_results = self.temporal_pattern.analyze_temporal_patterns(
                    prices=asset_data['prices'],
                    volumes=asset_data.get('volume'),
                    asset_id=asset_id
                )
                
                # Log interesting findings
                if 'transition_probability' in temporal_results:
                    probs = temporal_results['transition_probability']
                    current = self.current_regime["global"]["regime_type"]
                    
                    if current in probs:
                        next_regimes = sorted(
                            probs[current].items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        if next_regimes:
                            logger.info(f"Most likely next regime: {next_regimes[0][0]} "
                                      f"(probability: {next_regimes[0][1]:.2f})")
                
                return temporal_results
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {str(e)}")
        
        return {}
    
    def _apply_adaptive_responses(self, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply adaptive responses based on detected regimes.
        
        Args:
            regime_results: Dictionary with regime detection results
            
        Returns:
            Dictionary with adaptation results
        """
        adaptation_results = {"actions": []}
        
        try:
            # Get the global market regime
            global_regime = regime_results.get("global", {})
            regime_type = global_regime.get("regime_type")
            volatility_type = global_regime.get("volatility_type")
            
            if not regime_type:
                logger.warning("Cannot apply adaptations: missing regime type")
                return adaptation_results
            
            # Create metrics for adaptation
            metrics = {
                "market_regime": regime_type,
                "volatility_regime": volatility_type,
                "regime_confidence": global_regime.get("confidence", 0.0)
            }
            
            # Add temporal pattern metrics if available
            if "temporal" in regime_results:
                temporal = regime_results["temporal"]
                
                if "seasonality" in temporal:
                    metrics["seasonality"] = temporal["seasonality"]
                    
                if "transition_probability" in temporal:
                    metrics["transition_probability"] = temporal["transition_probability"]
                    
                if "multi_timeframe" in temporal:
                    metrics["multi_timeframe"] = temporal["multi_timeframe"]
            
            # Apply adaptations to relevant agents
            for agent_id, agent in self.agents.items():
                # Apply to adaptive strategy managers
                if isinstance(agent, AdaptiveStrategyManager):
                    strategy_adaptation = agent.evaluate_and_adapt(metrics, regime_type)
                    if strategy_adaptation:
                        adaptation_results["actions"].append({
                            "agent_id": agent_id,
                            "action": "strategy_adaptation",
                            "details": strategy_adaptation
                        })
                        logger.info(f"Applied strategy adaptation to {agent_id}: {strategy_adaptation}")
                
                # Apply to meta-strategies
                if isinstance(agent, DynamicAggregationMetaStrategy):
                    # Convert regime metrics to format expected by meta-strategy
                    market_conditions = {
                        "regime": regime_type,
                        "volatility": volatility_type,
                        "metrics": global_regime.get("metrics", {})
                    }
                    
                    # Select best method based on market conditions
                    best_method = agent.select_best_method(market_conditions)
                    if best_method:
                        adaptation_results["actions"].append({
                            "agent_id": agent_id,
                            "action": "method_selection",
                            "details": best_method
                        })
                        logger.info(f"Selected aggregation method for {agent_id}: {best_method}")
                        
                # Future: Add adaptations for other agent types
            
        except Exception as e:
            logger.error(f"Error applying adaptive responses: {str(e)}")
            if self.health_monitor:
                self.health_monitor.add_alert(
                    component_id="adaptive_response_system",
                    severity="ERROR",
                    message=f"Adaptation failed: {str(e)}"
                )
        
        return adaptation_results
    
    def register_adaptation_rule(self, agent_id: str, condition: Dict[str, Any], action: Dict[str, Any]) -> bool:
        """
        Register a custom adaptation rule for an agent.
        
        Args:
            agent_id: ID of the agent to adapt
            condition: Dictionary with conditions that trigger adaptation
            action: Dictionary with adaptation action parameters
            
        Returns:
            True if rule was registered successfully
        """
        if agent_id not in self.agents:
            logger.error(f"Cannot register rule for {agent_id}: agent not found")
            return False
        
        # For now, just pass to the adaptive manager if the agent has one
        if hasattr(self.agents[agent_id], 'register_adaptation_rule'):
            return self.agents[agent_id].register_adaptation_rule(condition, action)
            
        return False
    
    def get_regime_history(self, asset_id="global", days=30) -> Dict[str, Any]:
        """
        Get historical regime classifications for an asset.
        
        Args:
            asset_id: Asset identifier (default "global" for market-wide regime)
            days: Number of days of history to retrieve
            
        Returns:
            Dictionary with regime history
        """
        if asset_id == "global":
            return self.regime_classifier.get_regime_history("SPY")
        else:
            return self.regime_classifier.get_regime_history(asset_id)
    
    def get_current_regime(self, asset_id="global") -> Dict[str, Any]:
        """
        Get the current regime classification for an asset.
        
        Args:
            asset_id: Asset identifier (default "global" for market-wide regime)
            
        Returns:
            Dictionary with current regime information
        """
        if asset_id == "global":
            return self.current_regime.get("global", {
                "regime_type": "UNKNOWN",
                "volatility_type": "UNKNOWN",
                "confidence": 0.0
            })
        else:
            return self.current_regime.get(asset_id, {
                "regime_type": "UNKNOWN",
                "volatility_type": "UNKNOWN",
                "confidence": 0.0
            })
            
    def get_regime_statistics(self, asset_id="global") -> Dict[str, Any]:
        """
        Get statistics about regime classifications for an asset.
        
        Args:
            asset_id: Asset identifier (default "global" for market-wide regime)
            
        Returns:
            Dictionary with regime statistics
        """
        if asset_id == "global":
            return self.regime_classifier.get_regime_statistics("SPY")
        else:
            return self.regime_classifier.get_regime_statistics(asset_id)
    
    # ----- Advanced Risk Management Methods -----
    
    def update_portfolio_state(self, portfolio_value: float, positions: Dict[str, Any], drawdown: float = 0.0) -> None:
        """
        Update current portfolio state for risk management.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions dictionary with symbol keys
            drawdown: Current drawdown as a decimal (e.g., 0.05 for 5%)
        """
        self.portfolio_value = portfolio_value
        self.portfolio_positions = positions
        self.portfolio_drawdown = drawdown
        
        # Log significant changes in portfolio state
        if drawdown > 0.10:
            logger.warning(f"Significant drawdown detected: {drawdown:.1%}")
            
        # Make this information available to health monitoring
        if self.health_monitor is not None:
            self.health_monitor.add_metric(
                component_id="trading_orchestrator",
                metric_name="portfolio_value",
                value=portfolio_value
            )
            self.health_monitor.add_metric(
                component_id="trading_orchestrator",
                metric_name="drawdown",
                value=drawdown
            )
    
    def calculate_position_sizes(self, target_allocations: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate optimized position sizes using advanced risk management.
        
        This method integrates with the risk orchestrator to provide volatility-adjusted
        and correlation-optimized position sizing based on the current market regime.
        
        Args:
            target_allocations: Target portfolio allocations as a dictionary
                mapping symbols to weights (should sum to 1.0)
                
        Returns:
            Dictionary with optimized position sizes and metadata
        """
        if not self.enable_advanced_risk_management or not self.risk_orchestrator:
            # Fallback for when risk management is disabled
            results = {}
            for symbol, weight in target_allocations.items():
                results[symbol] = {
                    "target_weight": weight,
                    "adjusted_weight": weight,
                    "dollar_amount": self.portfolio_value * weight,
                    "volatility": 0.0  # Unknown
                }
            return results
        
        # Use the risk orchestrator to calculate optimized position sizes
        return self.risk_orchestrator.calculate_position_sizes(
            portfolio_value=self.portfolio_value,
            target_allocations=target_allocations
        )
    
    def get_risk_limits(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current risk limits for trading.
        
        Args:
            symbol: Optional symbol to get specific limits for
            
        Returns:
            Dictionary with risk limits and parameters
        """
        # Default limits when risk management is disabled
        default_limits = {
            "portfolio_risk": 0.02,  # 2% daily VaR
            "max_position_size": 0.20,  # 20% max for any position
            "in_stress_mode": False,
            "market_regime": self.current_regime.get("global", {}).get("regime_type", "UNKNOWN")
        }
        
        if not self.enable_advanced_risk_management or not self.risk_orchestrator:
            return default_limits
        
        # Get current risk metrics
        risk_metrics = self.risk_orchestrator.get_risk_metrics()
        
        # If a specific symbol is requested, include position limits
        if symbol is not None and self.risk_orchestrator:
            position_limits = self.risk_orchestrator.get_position_limits()
            if symbol in position_limits:
                risk_metrics["symbol_limits"] = position_limits[symbol]
        
        return risk_metrics
    
    def get_risk_alerts(self, max_alerts: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent risk management alerts.
        
        Args:
            max_alerts: Maximum number of alerts to return
            
        Returns:
            List of risk alert dictionaries
        """
        if not self.enable_advanced_risk_management or not self.risk_orchestrator:
            return []
        
        return self.risk_orchestrator.get_alerts(max_alerts=max_alerts)
    
    def validate_trading_decision(self, decision: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a trading decision using LLM oversight.
        
        This method sends trading decisions to the LLM oversight service for validation
        before execution. The decision can be approved, rejected, or modified based on
        the configured oversight level.
        
        Args:
            decision: Trading decision to validate (e.g., symbol, action, quantity, price)
            context: Optional additional context for decision validation
                (if not provided, current market and portfolio context will be used)
                
        Returns:
            Tuple of (is_approved, validation_result)
        """
        if not self.enable_llm_oversight or not self.oversight_client:
            # If oversight is disabled, automatically approve all decisions
            logger.debug("LLM oversight disabled, automatically approving trading decision")
            return True, {"action": "approve", "reason": "LLM oversight disabled"}
        
        # Ensure we have context for validation
        validation_context = context or {}
        
        # Add current market regime information if not in context
        if "market_regime" not in validation_context and self.current_regime:
            validation_context["market_regime"] = self.current_regime
        
        # Add current risk metrics if available and not in context
        if "risk_metrics" not in validation_context and self.enable_advanced_risk_management:
            validation_context["risk_metrics"] = self.risk_orchestrator.get_risk_metrics()
        
        # Send decision to oversight service for validation
        try:
            logger.info(f"Validating trading decision via LLM oversight: {decision.get('symbol', 'unknown')} {decision.get('action', 'unknown')}")
            
            # Get the oversight action (APPROVE, REJECT, MODIFY, LOG)
            oversight_action = self.oversight_client.get_decision_action(decision, validation_context)
            validation_result = self.oversight_client.validate_trading_decision(decision, validation_context)
            
            # Handle different oversight actions based on level
            if self.llm_oversight_level in ["monitor", "advise"]:
                # In monitor or advise mode, log the result but approve the decision
                if oversight_action == OversightAction.REJECT:
                    logger.warning(f"LLM oversight suggested rejecting decision, but oversight level is {self.llm_oversight_level}: {validation_result.get('reason', 'No reason provided')}")
                elif oversight_action == OversightAction.MODIFY:
                    logger.info(f"LLM oversight suggested modifying decision, but oversight level is {self.llm_oversight_level}: {validation_result.get('reason', 'No reason provided')}")
                
                # In these modes, always approve regardless of the oversight recommendation
                return True, validation_result
                
            elif self.llm_oversight_level in ["approve", "override", "autonomous"]:
                # In stricter modes, follow the oversight recommendation
                if oversight_action == OversightAction.APPROVE:
                    logger.info(f"LLM oversight approved trading decision: {validation_result.get('reason', 'No reason provided')}")
                    return True, validation_result
                    
                elif oversight_action == OversightAction.REJECT:
                    logger.warning(f"LLM oversight rejected trading decision: {validation_result.get('reason', 'No reason provided')}")
                    return False, validation_result
                    
                elif oversight_action == OversightAction.MODIFY:
                    logger.info(f"LLM oversight suggested modifications to trading decision: {validation_result.get('reason', 'No reason provided')}")
                    # In autonomous mode, we'd apply the suggested modifications here
                    if self.llm_oversight_level == "autonomous" and "modified_decision" in validation_result:
                        logger.info("Autonomously applying LLM-suggested decision modifications")
                        # Return approval with the modified decision
                        return True, validation_result
                    elif self.llm_oversight_level == "override" and "modified_decision" in validation_result:
                        logger.info("Applying LLM-suggested decision modifications (override mode)")
                        return True, validation_result
                    else:
                        # In approve mode, reject if modifications are needed
                        logger.info("Rejecting decision that requires modification (not in autonomous mode)")
                        return False, validation_result
                
                else:  # OversightAction.LOG
                    # Default to approving if action is just to log
                    logger.info("LLM oversight logged decision without specific recommendation")
                    return True, validation_result
            
            else:
                # Unknown oversight level, default to approving
                logger.warning(f"Unknown oversight level: {self.llm_oversight_level}, defaulting to approve")
                return True, validation_result
                
        except Exception as e:
            # Log the error and approve the decision to prevent system blockage
            logger.error(f"LLM oversight validation failed: {e}")
            return True, {"action": "approve", "reason": f"Oversight validation error: {str(e)}"}
    
    def analyze_market_conditions_with_llm(self, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze market conditions using LLM oversight.
        
        This enhances the traditional market regime classification with LLM-powered
        analysis and insights.
        
        Args:
            market_data: Optional market data to analyze
                (if not provided, current market data will be used)
                
        Returns:
            Analysis results including regime identification
        """
        if not self.enable_llm_oversight or not self.oversight_client:
            logger.debug("LLM oversight disabled, skipping LLM market analysis")
            return {}
        
        analysis_data = market_data or self.market_data
        if not analysis_data:
            logger.warning("No market data available for LLM analysis")
            return {}
        
        try:
            logger.info("Performing LLM-based market condition analysis")
            analysis_result = self.oversight_client.analyze_market_conditions(analysis_data)
            logger.info(f"LLM market analysis completed: {analysis_result.get('regime', 'unknown')} regime identified")
            return analysis_result
        except Exception as e:
            logger.error(f"LLM market analysis failed: {e}")
            return {}
