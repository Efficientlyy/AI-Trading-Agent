"""
Market Regime Adaptation Integrator

This module integrates the advanced market regime detection and adaptation components,
providing a unified interface for the adaptive orchestrator to leverage these capabilities.

It combines:
1. Enhanced market regime classification
2. Adaptive response system
3. Regime transition detection
4. Feedback mechanisms for continuous improvement
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import os
import json

from ai_trading_agent.agent.market_regime import MarketRegimeType
from ai_trading_agent.analysis.enhanced_regime_classifier import EnhancedMarketRegimeClassifier
from ai_trading_agent.analysis.adaptive_response_system import AdaptiveResponseSystem, AdaptationRule
from ai_trading_agent.analysis.regime_transition_detector import RegimeTransitionDetector, TransitionSignal

# Set up logger
logger = logging.getLogger(__name__)


class RegimeAdaptationIntegrator:
    """
    Integrates advanced market regime detection and adaptation components.
    
    This class serves as the main interface between the adaptive orchestrator
    and the various regime detection and adaptation components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the regime adaptation integrator.
        
        Args:
            config: Configuration dictionary with parameters
                - data_dir: Directory for data storage
                - model_dir: Directory for model storage
                - classifier_config: Configuration for the regime classifier
                - adaptation_config: Configuration for the adaptive response system
                - transition_config: Configuration for the transition detector
                - calibration_frequency_days: How often to recalibrate models (days)
                - use_transition_detection: Whether to enable transition detection
        """
        # Default configuration
        default_config = {
            "data_dir": "./data/regime_adaptation",
            "model_dir": "./models/regime_adaptation",
            "classifier_config": {},
            "adaptation_config": {},
            "transition_config": {},
            "calibration_frequency_days": 30,
            "use_transition_detection": True
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize directories
        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["model_dir"], exist_ok=True)
        
        # Initialize components
        self.classifier = EnhancedMarketRegimeClassifier(self.config["classifier_config"])
        self.response_system = AdaptiveResponseSystem(self.config["adaptation_config"])
        
        if self.config["use_transition_detection"]:
            self.transition_detector = RegimeTransitionDetector(self.config["transition_config"])
        else:
            self.transition_detector = None
        
        # Track performance metrics
        self.performance_metrics = {
            "classification_accuracy": [],
            "adaptation_effectiveness": [],
            "transition_detection_accuracy": []
        }
        
        # Last calibration timestamp
        self.last_calibration = datetime.now()
        
        # Historical regime classifications
        self.regime_history = {}
        
        # Active adaptations
        self.active_adaptations = {}
        
        logger.info("Regime Adaptation Integrator initialized")
        
        # Attempt to load saved state
        self.load_state()
    
    def analyze_market_conditions(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform comprehensive market regime analysis on multiple assets.
        
        Args:
            data_dict: Dictionary mapping symbols to their OHLCV DataFrames
            
        Returns:
            Dictionary with comprehensive analysis results including:
            - regimes: Detected market regimes for each symbol
            - transitions: Potential regime transitions
            - adaptations: Recommended adaptations
            - metadata: Analysis metadata
        """
        if not data_dict:
            logger.warning("No data provided for market regime analysis")
            return {}
        
        # Step 1: Classify market regimes for all symbols
        regime_results = self.classifier.classify_multiple(data_dict)
        
        # Step 2: Detect potential regime transitions
        transition_signals = {}
        if self.transition_detector:
            for symbol, data in data_dict.items():
                # Get current regime for this symbol
                current_regime = MarketRegimeType.UNKNOWN
                if symbol in regime_results:
                    regime_value = regime_results[symbol]["regime"]
                    for rt in MarketRegimeType:
                        if rt.value == regime_value:
                            current_regime = rt
                            break
                
                # Detect potential transitions
                signals = self.transition_detector.detect_transition_signals(data, current_regime)
                if signals:
                    transition_signals[symbol] = [s.to_dict() for s in signals]
        
        # Step 3: Update historical regime tracking
        timestamp = datetime.now().isoformat()
        for symbol, result in regime_results.items():
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
            
            # Add to history with timestamp
            self.regime_history[symbol].append({
                "timestamp": timestamp,
                "regime": result["regime"],
                "confidence": result["confidence"]
            })
            
            # Limit history size (keep last 1000 entries)
            if len(self.regime_history[symbol]) > 1000:
                self.regime_history[symbol] = self.regime_history[symbol][-1000:]
        
        # Create comprehensive results dictionary
        analysis_results = {
            "regimes": regime_results,
            "transitions": transition_signals,
            "global_regime": self._determine_global_regime(regime_results),
            "timestamp": timestamp
        }
        
        return analysis_results
    
    def _determine_global_regime(self, regime_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine the global market regime based on individual asset regimes.
        
        Args:
            regime_results: Dictionary mapping symbols to their regime classifications
            
        Returns:
            Dictionary with global regime information
        """
        if not regime_results:
            return {
                "regime": MarketRegimeType.UNKNOWN.value,
                "confidence": 0.0
            }
        
        # Count occurrences of each regime type
        regime_counts = {}
        confidence_sums = {}
        
        for symbol, result in regime_results.items():
            regime = result["regime"]
            confidence = result["confidence"]
            
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            confidence_sums[regime] = confidence_sums.get(regime, 0) + confidence
        
        # Find the most common regime
        if not regime_counts:
            return {
                "regime": MarketRegimeType.UNKNOWN.value,
                "confidence": 0.0
            }
            
        most_common_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average confidence for this regime
        avg_confidence = confidence_sums[most_common_regime] / regime_counts[most_common_regime]
        
        # Calculate regime distribution
        total_symbols = len(regime_results)
        regime_distribution = {
            regime: count / total_symbols for regime, count in regime_counts.items()
        }
        
        return {
            "regime": most_common_regime,
            "confidence": avg_confidence,
            "distribution": regime_distribution,
            "count": regime_counts[most_common_regime],
            "total_symbols": total_symbols
        }
    
    def determine_adaptations(self, 
                              analysis_results: Dict[str, Any], 
                              target_system: Any) -> Dict[str, Any]:
        """
        Determine appropriate adaptations based on market analysis.
        
        Args:
            analysis_results: Results from analyze_market_conditions
            target_system: System to apply adaptations to (usually the orchestrator)
            
        Returns:
            Dictionary with adaptation results
        """
        if not analysis_results:
            logger.warning("No analysis results provided for adaptation determination")
            return {}
        
        # Create context for adaptation rule evaluation
        adaptation_context = {
            "current_regime": analysis_results.get("global_regime", {}).get("regime"),
            "confidence": analysis_results.get("global_regime", {}).get("confidence", 0),
            "regime_distribution": analysis_results.get("global_regime", {}).get("distribution", {}),
            "asset_regimes": analysis_results.get("regimes", {}),
            "transitions": analysis_results.get("transitions", {})
        }
        
        # Run adaptive response system to determine adaptations
        adaptation_results = self.response_system.evaluate_adaptations(
            adaptation_context, target_system)
        
        # Store active adaptations
        self.active_adaptations = {
            result["rule_name"]: {
                "target": result["target"],
                "action": result["action"],
                "value": result["value"],
                "timestamp": result["timestamp"],
                "success": result["success"]
            }
            for result in adaptation_results if result.get("success", False)
        }
        
        return {
            "adaptations": adaptation_results,
            "active_adaptations": self.active_adaptations,
            "timestamp": datetime.now().isoformat()
        }
    
    def register_adaptation_rule(self, rule: AdaptationRule) -> bool:
        """
        Register a new adaptation rule.
        
        Args:
            rule: AdaptationRule instance to register
            
        Returns:
            Boolean indicating success
        """
        return self.response_system.register_rule(rule)
    
    def get_regime_history(self, symbol: str = None, days: int = 30) -> Dict[str, Any]:
        """
        Get historical regime classifications.
        
        Args:
            symbol: Symbol to get history for (None for all symbols)
            days: Number of days of history to retrieve
            
        Returns:
            Dictionary with regime history
        """
        # Calculate cutoff timestamp
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        
        if symbol:
            # Return history for specific symbol
            if symbol not in self.regime_history:
                return {"symbol": symbol, "history": []}
                
            history = [
                entry for entry in self.regime_history[symbol]
                if entry["timestamp"] >= cutoff_str
            ]
            
            return {
                "symbol": symbol,
                "history": history,
                "count": len(history)
            }
        else:
            # Return summary for all symbols
            all_symbols = {}
            for sym, history in self.regime_history.items():
                recent_history = [
                    entry for entry in history
                    if entry["timestamp"] >= cutoff_str
                ]
                
                if recent_history:
                    all_symbols[sym] = {
                        "count": len(recent_history),
                        "latest": recent_history[-1]
                    }
            
            return {
                "symbols": all_symbols,
                "total_symbols": len(all_symbols)
            }
    
    def calibrate_models(self, training_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calibrate models with new training data.
        
        Args:
            training_data: Optional training data to use
            
        Returns:
            Dictionary with calibration results
        """
        results = {
            "classifier_calibrated": False,
            "transition_matrix_updated": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if calibration is due
        days_since_calibration = (datetime.now() - self.last_calibration).days
        if days_since_calibration < self.config["calibration_frequency_days"] and training_data is None:
            logger.info(f"Skipping calibration. Next scheduled in {self.config['calibration_frequency_days'] - days_since_calibration} days")
            results["message"] = f"Calibration not due for {self.config['calibration_frequency_days'] - days_since_calibration} more days"
            return results
        
        try:
            # Calibrate regime classifier if training data provided
            if training_data:
                logger.info(f"Calibrating regime classifier with {len(training_data)} datasets")
                accuracy = self.classifier.train(training_data)
                results["classifier_calibrated"] = True
                results["classifier_accuracy"] = accuracy
                
                # Store accuracy in performance metrics
                self.performance_metrics["classification_accuracy"].append({
                    "timestamp": datetime.now().isoformat(),
                    "accuracy": accuracy,
                    "training_data_size": len(training_data)
                })
            
            # Update transition matrix if detector exists and history available
            if self.transition_detector and self.regime_history:
                # Extract actual transitions from history
                actual_transitions = []
                
                for symbol, history in self.regime_history.items():
                    if len(history) < 2:
                        continue
                        
                    # Find transitions in the history
                    for i in range(1, len(history)):
                        prev_regime = history[i-1]["regime"]
                        curr_regime = history[i]["regime"]
                        
                        if prev_regime != curr_regime:
                            # Convert string representations to enum
                            from_regime = MarketRegimeType.UNKNOWN
                            to_regime = MarketRegimeType.UNKNOWN
                            
                            for rt in MarketRegimeType:
                                if rt.value == prev_regime:
                                    from_regime = rt
                                if rt.value == curr_regime:
                                    to_regime = rt
                            
                            if from_regime != MarketRegimeType.UNKNOWN and to_regime != MarketRegimeType.UNKNOWN:
                                actual_transitions.append((from_regime, to_regime))
                
                # Update transition matrix
                if actual_transitions:
                    logger.info(f"Updating transition matrix with {len(actual_transitions)} observed transitions")
                    self.transition_detector.update_transition_matrix(actual_transitions)
                    results["transition_matrix_updated"] = True
                    results["transitions_processed"] = len(actual_transitions)
            
            # Update calibration timestamp
            self.last_calibration = datetime.now()
            
            # Save state
            self.save_state()
            
            return results
            
        except Exception as e:
            logger.error(f"Error during model calibration: {e}")
            results["error"] = str(e)
            return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the regime adaptation system.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate summary statistics for metrics
        metrics_summary = {}
        
        for metric_name, metric_values in self.performance_metrics.items():
            if not metric_values:
                metrics_summary[metric_name] = {"count": 0}
                continue
                
            # Get most recent metrics
            recent_values = metric_values[-10:]
            
            # Extract numeric values
            numeric_values = []
            for item in recent_values:
                for k, v in item.items():
                    if isinstance(v, (int, float)) and k != "timestamp":
                        numeric_values.append(v)
            
            if numeric_values:
                metrics_summary[metric_name] = {
                    "count": len(metric_values),
                    "recent_count": len(recent_values),
                    "mean": np.mean(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "latest": recent_values[-1] if recent_values else None
                }
            else:
                metrics_summary[metric_name] = {"count": len(metric_values), "recent_count": len(recent_values)}
        
        # Add adaptation effectiveness
        adaptation_effectiveness = self.response_system.analyze_adaptation_effectiveness()
        metrics_summary["adaptation_effectiveness"] = adaptation_effectiveness
        
        # Add transition detection metrics if available
        if self.transition_detector:
            metrics_summary["transition_matrix"] = self.transition_detector.get_transition_matrix()
            metrics_summary["active_transition_signals"] = len(self.transition_detector.get_active_signals())
        
        return metrics_summary
    
    def save_state(self) -> bool:
        """
        Save the current state of the regime adaptation system.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Create state dict
            state = {
                "config": self.config,
                "last_calibration": self.last_calibration.isoformat(),
                "performance_metrics": self.performance_metrics,
                "active_adaptations": self.active_adaptations
            }
            
            # Save regime history as separate file (could be large)
            history_path = os.path.join(self.config["data_dir"], "regime_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.regime_history, f)
            
            # Save main state
            state_path = os.path.join(self.config["model_dir"], "integrator_state.json")
            with open(state_path, 'w') as f:
                json.dump(state, f)
            
            # Save component states
            self.classifier.save_models() if hasattr(self.classifier, 'save_models') else None
            
            if self.transition_detector:
                self.transition_detector.save_state()
            
            logger.info(f"Saved regime adaptation integrator state to {self.config['model_dir']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving regime adaptation integrator state: {e}")
            return False
    
    def load_state(self) -> bool:
        """
        Load a previously saved state.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Check if state file exists
            state_path = os.path.join(self.config["model_dir"], "integrator_state.json")
            if not os.path.exists(state_path):
                logger.info("No saved state found for regime adaptation integrator")
                return False
            
            # Load main state
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            # Update fields
            self.config.update(state.get("config", {}))
            self.last_calibration = datetime.fromisoformat(state.get("last_calibration", datetime.now().isoformat()))
            self.performance_metrics = state.get("performance_metrics", self.performance_metrics)
            self.active_adaptations = state.get("active_adaptations", {})
            
            # Load regime history if available
            history_path = os.path.join(self.config["data_dir"], "regime_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.regime_history = json.load(f)
            
            # Load component states
            self.classifier.load_models() if hasattr(self.classifier, 'load_models') else None
            
            if self.transition_detector:
                self.transition_detector.load_state()
            
            logger.info(f"Loaded regime adaptation integrator state from {self.config['model_dir']}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading regime adaptation integrator state: {e}")
            return False
