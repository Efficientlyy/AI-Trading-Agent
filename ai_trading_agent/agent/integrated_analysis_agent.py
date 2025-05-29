"""
Integrated Analysis Agent - Combines technical and sentiment analysis.

This agent coordinates between the TechnicalAnalysisAgent and SentimentAnalysisAgent
to provide comprehensive trading signals that incorporate both technical indicators
and sentiment data.
"""

import logging
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np

from ai_trading_agent.common.utils import get_logger
from .technical_analysis_agent import TechnicalAnalysisAgent, DataMode
from .sentiment_analysis_agent import SentimentAnalysisAgent, SentimentSource
from .agent_definitions import AgentStatus, AgentRole, BaseAgent


class SignalSource(Enum):
    """Sources of trading signals."""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    COMBINED = "combined"


class SignalWeight(Enum):
    """Weight presets for different signal sources."""
    TECHNICAL_ONLY = {"technical": 1.0, "sentiment": 0.0}
    SENTIMENT_ONLY = {"technical": 0.0, "sentiment": 1.0}
    BALANCED = {"technical": 0.5, "sentiment": 0.5}
    TECHNICAL_PRIMARY = {"technical": 0.7, "sentiment": 0.3}
    SENTIMENT_PRIMARY = {"technical": 0.3, "sentiment": 0.7}


class IntegratedAnalysisAgent(BaseAgent):
    """
    Integrates technical and sentiment analysis to provide comprehensive trading signals.
    
    This agent combines the outputs of the TechnicalAnalysisAgent and SentimentAnalysisAgent,
    applying configurable weights to each signal source to generate the final trading signals.
    """
    
    AGENT_ID_PREFIX = "integrated_analysis_"
    
    def __init__(self, agent_id_suffix: str, name: str, symbols: List[str], 
                 config_details: Optional[Dict] = None):
        
        agent_id = f"{name.replace(' ', '_')}_{agent_id_suffix}"

        # Initialize BaseAgent
        super().__init__(
            agent_id=agent_id,
            name=name,
            agent_role=AgentRole.DECISION_AGGREGATOR,  # Using a valid role from the enum
            agent_type="IntegratedAnalysis",
            symbols=symbols,
            config_details=config_details
        )

        # Load configuration
        self.config = self._load_config(self.config_details)

        # Initialize logger
        log_level = self.config.get('logging', {}).get('level', 'INFO').upper()
        self.logger = get_logger(self.agent_id, level=log_level)
        self.logger.info(f"IntegratedAnalysisAgent '{self.name}' initialized. Log level: {log_level}")
        self.logger.debug(f"Agent configuration: {self.config}")

        # Initialize child agents
        self._init_child_agents()
        
        # Set up signal weights
        self._init_signal_weights()
        
        # Initialize metrics tracking
        self.metrics = {
            "processing_errors": 0,
            "signals_generated": 0,
            "technical_signals_used": 0,
            "sentiment_signals_used": 0,
            "avg_processing_time_ms": 0.0,
            "last_processing_time_ms": 0.0
        }
        
        # Storage for current state and signals
        self.integrated_state = {}
        self.current_signals = {}
        
        # Set initial status
        self.status = AgentStatus.IDLE
        self.logger.info(f"IntegratedAnalysisAgent '{self.name}' setup complete.")

    def _load_config(self, config_details: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Load configuration for the integrated analysis agent.
        
        Args:
            config_details: Optional configuration details
            
        Returns:
            Merged configuration dictionary
        """
        # Start with default configuration
        default_config = {
            "logging": {
                "level": "INFO"
            },
            "signal_weights": {
                "preset": "BALANCED",
                "custom": {
                    "technical": 0.5,
                    "sentiment": 0.5
                }
            },
            "technical_agent": {
                "data_mode": "real"
            },
            "sentiment_agent": {
                "sentiment_sources": ["news", "social_media"]
            },
            "integration": {
                "correlation_threshold": 0.5,
                "confidence_threshold": 0.6,
                "signal_alignment": "strict"  # or "flexible"
            }
        }
        
        # Merge with provided configuration
        if config_details:
            # Deep merge would be better but this is simpler for now
            for key, value in config_details.items():
                if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        return default_config

    def _init_child_agents(self):
        """Initialize technical and sentiment analysis agents."""
        # Initialize Technical Analysis Agent
        tech_config = self.config.get("technical_agent", {})
        tech_agent_id = f"tech_{self.agent_id}"
        
        # Get data mode from config
        data_mode_str = tech_config.get("data_mode", "real").upper()
        data_mode = DataMode.REAL if data_mode_str == "REAL" else DataMode.MOCK
        
        self.technical_agent = TechnicalAnalysisAgent(
            agent_id_suffix=tech_agent_id,
            name=f"{self.name}_Technical",
            symbols=self.symbols,
            config_details=tech_config,
            data_mode=data_mode
        )
        
        # Initialize Sentiment Analysis Agent
        sentiment_config = self.config.get("sentiment_agent", {})
        sentiment_agent_id = f"sentiment_{self.agent_id}"
        
        # Convert string source names to SentimentSource enum values
        sentiment_sources = sentiment_config.get("sentiment_sources", ["news", "social_media"])
        sentiment_source_enums = []
        
        for source in sentiment_sources:
            try:
                sentiment_source_enums.append(SentimentSource[source.upper()])
            except (KeyError, AttributeError):
                self.logger.warning(f"Unknown sentiment source: {source}. Skipping.")
        
        self.sentiment_agent = SentimentAnalysisAgent(
            agent_id_suffix=sentiment_agent_id,
            name=f"{self.name}_Sentiment",
            symbols=self.symbols,
            config_details=sentiment_config,
            sentiment_sources=sentiment_source_enums
        )
        
        self.logger.info("Initialized child agents: Technical and Sentiment Analysis")

    def _init_signal_weights(self):
        """Initialize weights for combining signals from different sources."""
        # Get weight configuration
        weight_config = self.config.get("signal_weights", {})
        preset = weight_config.get("preset", "BALANCED")
        
        # Set weights based on preset or custom values
        if preset and hasattr(SignalWeight, preset):
            self.signal_weights = SignalWeight[preset].value
            self.logger.info(f"Using preset signal weights: {preset}")
        else:
            # Use custom weights if provided, otherwise default to balanced
            self.signal_weights = weight_config.get("custom", {
                "technical": 0.5,
                "sentiment": 0.5
            })
            self.logger.info(f"Using custom signal weights: {self.signal_weights}")
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(self.signal_weights.values())
        if total_weight == 0:
            # If all weights are zero, use equal weights
            for key in self.signal_weights:
                self.signal_weights[key] = 1.0 / len(self.signal_weights)
        elif total_weight != 1.0:
            # Normalize to sum to 1.0
            for key in self.signal_weights:
                self.signal_weights[key] /= total_weight

    def update_status(self, new_status: AgentStatus):
        """Update the agent's status and log the change."""
        if self.status != new_status:
            self.logger.info(f"Agent status changed from {self.status.value} to {new_status.value}")
            self.status = new_status

    def update_metrics(self, new_metrics: Dict[str, Any]):
        """
        Update the agent's performance metrics.
        
        Args:
            new_metrics: Dictionary with new metric values
        """
        for key, value in new_metrics.items():
            if key == "avg_processing_time_ms":
                # Keep a running average of processing time
                if "last_processing_time_ms" in self.metrics:
                    self.metrics["last_processing_time_ms"] = value
                    count = self.metrics.get("signals_generated", 0)
                    if count > 0:
                        # Calculate running average
                        current_avg = self.metrics["avg_processing_time_ms"]
                        self.metrics["avg_processing_time_ms"] = (current_avg * (count - 1) + value) / count
                else:
                    self.metrics["avg_processing_time_ms"] = value
                    self.metrics["last_processing_time_ms"] = value
            elif key in self.metrics:
                # For counters, increment them
                self.metrics[key] += value
            else:
                # For new metrics, just set them
                self.metrics[key] = value

    def process(self, data: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Process market and sentiment data to generate integrated trading signals.
        
        This is the main entry point for the agent's functionality. It:
        1. Delegates data processing to the technical and sentiment agents
        2. Combines the signals from both sources with configurable weights
        3. Returns the integrated signals
        
        Args:
            data: Optional input data dictionary with market and sentiment data
            
        Returns:
            List of integrated signal dictionaries or None if processing fails
        """
        self.update_status(AgentStatus.RUNNING)
        self.logger.info(f"Processing data for {len(self.symbols)} symbols")
        start_time = datetime.now()
        
        try:
            # Extract market and sentiment data
            market_data = None
            sentiment_data = None
            
            if data:
                market_data = data.get("market_data")
                sentiment_data = data.get("sentiment_data")
            
            # Process with technical analysis agent
            technical_signals = self.technical_agent.process({"market_data": market_data})
            
            # Process with sentiment analysis agent
            sentiment_signals = self.sentiment_agent.process({"sentiment_data": sentiment_data})
            
            # Combine signals
            if technical_signals or sentiment_signals:
                integrated_signals = self._combine_signals(technical_signals, sentiment_signals)
            else:
                self.logger.warning("No signals generated from either agent")
                integrated_signals = []
            
            # Store the integrated state
            self.integrated_state = {
                "technical_state": self.technical_agent.get_technical_state(),
                "sentiment_state": self.sentiment_agent.get_sentiment_state(),
                "integrated_signals": integrated_signals
            }
            
            # Store signals by symbol for easy lookup
            self.current_signals = {
                s["payload"]["symbol"]: s for s in integrated_signals 
                if "payload" in s and "symbol" in s["payload"]
            }
            
            # Track processing time and metrics
            process_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update agent metrics
            self.update_metrics({
                "avg_processing_time_ms": process_time,
                "signals_generated": len(integrated_signals),
                "technical_signals_used": len(technical_signals) if technical_signals else 0,
                "sentiment_signals_used": len(sentiment_signals) if sentiment_signals else 0
            })
            
            self.logger.info(
                f"Processed data in {process_time:.2f}ms, generated {len(integrated_signals)} "
                f"integrated signals"
            )
            self.update_status(AgentStatus.IDLE)
            return integrated_signals
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}", exc_info=True)
            self.update_metrics({"processing_errors": 1})
            self.update_status(AgentStatus.ERROR)
            return None

    def _combine_signals(self, technical_signals: List[Dict[str, Any]], 
                          sentiment_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine technical and sentiment signals with configured weights.
        
        Args:
            technical_signals: List of technical analysis signals
            sentiment_signals: List of sentiment analysis signals
            
        Returns:
            List of integrated signals
        """
        # Initialize result
        integrated_signals = []
        
        # Convert signals to dictionary by symbol for easier lookup
        tech_by_symbol = {}
        if technical_signals:
            for signal in technical_signals:
                if "payload" in signal and "symbol" in signal["payload"]:
                    symbol = signal["payload"]["symbol"]
                    tech_by_symbol[symbol] = signal
        
        # Similar for sentiment signals
        sentiment_by_symbol = {}
        sentiment_by_topic = {}
        if sentiment_signals:
            for signal in sentiment_signals:
                if "payload" in signal:
                    if "symbol" in signal["payload"]:
                        symbol = signal["payload"]["symbol"]
                        sentiment_by_symbol[symbol] = signal
                    elif "topic" in signal["payload"]:
                        topic = signal["payload"]["topic"]
                        sentiment_by_topic[topic] = signal
        
        # Combine signals for each symbol
        for symbol in self.symbols:
            tech_signal = tech_by_symbol.get(symbol)
            sent_signal = sentiment_by_symbol.get(symbol)
            
            # Calculate combined signal
            combined_signal = self._calculate_combined_signal(symbol, tech_signal, sent_signal)
            
            if combined_signal:
                integrated_signals.append(combined_signal)
        
        # Add topic-based signals if any (these don't have a direct technical counterpart)
        for topic, signal in sentiment_by_topic.items():
            # Create an integrated signal with only sentiment component
            integrated_signal = {
                "source": SignalSource.SENTIMENT.value,
                "timestamp": datetime.now().isoformat(),
                "payload": {
                    "topic": topic,
                    "is_topic": True,
                    "signal_strength": signal["payload"].get("signal_strength", 0) * 
                                      self.signal_weights["sentiment"],
                    "signal_type": signal["payload"].get("signal_type", "neutral"),
                    "confidence": signal["payload"].get("confidence", 0.5),
                    "sentiment_factor": 1.0,
                    "technical_factor": 0.0
                }
            }
            integrated_signals.append(integrated_signal)
        
        return integrated_signals

    def _calculate_combined_signal(self, symbol: str, 
                                   tech_signal: Optional[Dict[str, Any]], 
                                   sent_signal: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Calculate a combined signal for a specific symbol.
        
        Args:
            symbol: The trading symbol
            tech_signal: Technical analysis signal (or None)
            sent_signal: Sentiment analysis signal (or None)
            
        Returns:
            Combined signal dictionary or None if no valid signals
        """
        if not (tech_signal or sent_signal):
            return None
        
        # Initialize combined values
        combined_strength = 0.0
        combined_confidence = 0.0
        signal_type = "neutral"
        technical_factor = 0.0
        sentiment_factor = 0.0
        
        # Extract technical signal values if available
        if tech_signal and "payload" in tech_signal:
            tech_strength = tech_signal["payload"].get("signal_strength", 0.0)
            tech_confidence = tech_signal["payload"].get("confidence_score", 0.5)
            tech_type = tech_signal["payload"].get("signal_type", "neutral")
            
            # Apply weight to technical signal
            weighted_tech = tech_strength * self.signal_weights["technical"]
            combined_strength += weighted_tech
            combined_confidence += tech_confidence * self.signal_weights["technical"]
            technical_factor = self.signal_weights["technical"]
            
            # Use technical signal type if it's the only one or it's stronger
            if not sent_signal or abs(weighted_tech) > abs(combined_strength) / 2:
                signal_type = tech_type
        
        # Extract sentiment signal values if available
        if sent_signal and "payload" in sent_signal:
            sent_strength = sent_signal["payload"].get("signal_strength", 0.0)
            sent_confidence = sent_signal["payload"].get("confidence", 0.5)
            sent_type = sent_signal["payload"].get("signal_type", "neutral")
            
            # Apply weight to sentiment signal
            weighted_sent = sent_strength * self.signal_weights["sentiment"]
            combined_strength += weighted_sent
            combined_confidence += sent_confidence * self.signal_weights["sentiment"]
            sentiment_factor = self.signal_weights["sentiment"]
            
            # Use sentiment signal type if it's the only one or it's stronger
            if not tech_signal or abs(weighted_sent) > abs(combined_strength) / 2:
                signal_type = sent_type
        
        # Create the combined signal
        combined_signal = {
            "source": SignalSource.COMBINED.value,
            "timestamp": datetime.now().isoformat(),
            "payload": {
                "symbol": symbol,
                "signal_strength": combined_strength,
                "signal_type": signal_type,
                "confidence": combined_confidence,
                "technical_factor": technical_factor,
                "sentiment_factor": sentiment_factor,
                "technical_signal": tech_signal["payload"] if tech_signal else None,
                "sentiment_signal": sent_signal["payload"] if sent_signal else None
            }
        }
        
        return combined_signal

    def get_integrated_state(self) -> Dict[str, Any]:
        """Get the current integrated analysis state."""
        return self.integrated_state
    
    def get_component_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all components."""
        metrics = {
            "agent": self.metrics,
            "technical_agent": self.technical_agent.get_component_metrics() if self.technical_agent else {},
            "sentiment_agent": self.sentiment_agent.get_component_metrics() if self.sentiment_agent else {}
        }
        
        return metrics
    
    def update_signal_weights(self, new_weights: Dict[str, float]):
        """
        Update the weights used for combining signals.
        
        Args:
            new_weights: Dictionary with new weights
        """
        # Validate weights
        for key in ["technical", "sentiment"]:
            if key not in new_weights:
                self.logger.warning(f"Missing weight for {key}, using current value")
                new_weights[key] = self.signal_weights.get(key, 0.5)
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight == 0:
            self.logger.warning("All weights are zero, setting to equal weights")
            for key in new_weights:
                new_weights[key] = 1.0 / len(new_weights)
        elif total_weight != 1.0:
            for key in new_weights:
                new_weights[key] /= total_weight
        
        # Update weights
        self.signal_weights = new_weights
        self.logger.info(f"Updated signal weights: {self.signal_weights}")
        
        # Update configuration to reflect the change
        self.config["signal_weights"]["preset"] = "CUSTOM"
        self.config["signal_weights"]["custom"] = dict(self.signal_weights)
