"""
Sentiment Analysis Agent - Unified agent for all sentiment analysis operations.

This agent consolidates sentiment analysis from various data sources including
news, social media, and financial reports into trading signals.
"""

import logging
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from ai_trading_agent.common.utils import get_logger
from ai_trading_agent.sentiment.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.nlp_processing.sentiment_processor import SentimentProcessor
from .agent_definitions import AgentStatus, AgentRole, BaseAgent


class SentimentSource(Enum):
    """Defines the available sources for sentiment data."""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    FINANCIAL_REPORTS = "financial_reports"
    BLOGS = "blogs"
    MOCK = "mock"


class SentimentTopic(Enum):
    """Common sentiment topics for analysis."""
    CRYPTO = "cryptocurrency"
    BLOCKCHAIN = "blockchain"
    DEFI = "defi"
    NFT = "nft"
    REGULATION = "regulation"
    MARKET = "market"


class SentimentAnalysisAgent(BaseAgent):
    """
    Comprehensive sentiment analysis agent that processes text data from various sources,
    extracts sentiment, and generates trading signals based on sentiment trends.
    
    This agent can analyze news, social media, and other text sources to detect market sentiment
    and predict potential price movements based on public perception.
    """
    
    AGENT_ID_PREFIX = "sentiment_analysis_"
    
    def __init__(self, agent_id_suffix: str, name: str, symbols: List[str], 
                 config_details: Optional[Dict] = None, 
                 sentiment_sources: Optional[List[SentimentSource]] = None):
        
        agent_id = f"{name.replace(' ', '_')}_{agent_id_suffix}"

        # Initialize BaseAgent
        super().__init__(
            agent_id=agent_id,
            name=name,
            agent_role=AgentRole.SPECIALIZED_TECHNICAL,  # Using an existing role type
            agent_type="SentimentAnalysis",
            symbols=symbols,
            config_details=config_details
        )

        # Load configuration
        self.config = self._load_config(self.config_details)

        # Initialize logger
        log_level = self.config.get('logging', {}).get('level', 'INFO').upper()
        self.logger = get_logger(self.agent_id, level=log_level)
        self.logger.info(f"SentimentAnalysisAgent '{self.name}' initialized. Log level: {log_level}")
        self.logger.debug(f"Agent configuration: {self.config}")

        # Set up sentiment sources
        self.sentiment_sources = sentiment_sources or [
            SentimentSource.NEWS, 
            SentimentSource.SOCIAL_MEDIA
        ]
        
        # Initialize sentiment analyzer with configuration
        sentiment_analyzer_config = self.config.get("sentiment_analyzer", {})
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_analyzer_config)
        
        # Initialize sentiment processor for raw data
        self.sentiment_processor = SentimentProcessor()
        
        # Set default topics to track
        self.topics = self.config.get("topics", [
            topic.value for topic in [
                SentimentTopic.CRYPTO, 
                SentimentTopic.BLOCKCHAIN,
                SentimentTopic.MARKET
            ]
        ])
        
        # Define which assets to track sentiment for
        self.tracked_assets = self.config.get("tracked_assets", self.symbols)
        
        # Set up signal threshold configurations
        self.signal_thresholds = self.config.get("signal_thresholds", {
            "strong_buy": 0.6,     # Strong positive sentiment
            "buy": 0.3,            # Moderate positive sentiment
            "neutral": 0.1,        # Neutral sentiment
            "sell": -0.3,          # Moderate negative sentiment
            "strong_sell": -0.6    # Strong negative sentiment
        })
        
        # Time windows for analysis (in hours)
        self.time_windows = self.config.get("time_windows", {
            "short_term": 24,      # 1 day
            "medium_term": 72,     # 3 days
            "long_term": 168       # 7 days
        })
        
        # Initialize metrics tracking
        self.metrics = {
            "processing_errors": 0,
            "signals_generated": 0,
            "texts_analyzed": 0,
            "avg_processing_time_ms": 0.0,
            "last_processing_time_ms": 0.0
        }
        
        # Storage for current state and signals
        self.sentiment_data_cache = {}
        self.sentiment_state = {}
        self.current_signals = {}
        
        # Set initial status
        self.status = AgentStatus.IDLE
        self.logger.info(f"SentimentAnalysisAgent '{self.name}' setup complete.")

    def _load_config(self, config_details: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Load configuration for the sentiment analysis agent.
        
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
            "sentiment_analyzer": {
                "use_nltk": True,
                "use_textblob": True,
                "use_custom": True,
                "method_weights": {
                    "nltk": 0.4,
                    "textblob": 0.3,
                    "custom": 0.3
                },
                "cache_enabled": True
            },
            "signal_generation": {
                "confidence_threshold": 0.6,
                "min_texts_required": 5,
                "volatility_factor": 0.5
            },
            "data_sources": {
                "news": {
                    "enabled": True,
                    "weight": 0.4
                },
                "social_media": {
                    "enabled": True,
                    "weight": 0.3
                },
                "financial_reports": {
                    "enabled": False,
                    "weight": 0.3
                }
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

    def _generate_mock_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """
        Generate mock sentiment data for development and testing.
        
        Args:
            symbol: Symbol to generate data for
            
        Returns:
            Dictionary with mock sentiment data
        """
        # Create random sentiment scores for testing
        sentiment_values = []
        for _ in range(30):  # 30 days of mock data
            # Generate random sentiment between -1 and 1
            sentiment = np.random.normal(0, 0.3)  # Mean 0, std 0.3
            # Clip to valid range
            sentiment = max(-1.0, min(1.0, sentiment))
            sentiment_values.append(sentiment)
        
        # Add some trends and patterns
        if symbol == "BTC":
            # Add positive trend for Bitcoin
            for i in range(10):
                sentiment_values[i] += 0.1 * i
                sentiment_values[i] = max(-1.0, min(1.0, sentiment_values[i]))
        elif symbol == "ETH":
            # Add negative trend for Ethereum
            for i in range(10):
                sentiment_values[i] -= 0.05 * i
                sentiment_values[i] = max(-1.0, min(1.0, sentiment_values[i]))
        
        # Create mock text data
        texts = [
            f"This is a mock sentiment text about {symbol} with sentiment {sentiment:.2f}"
            for sentiment in sentiment_values
        ]
        
        # Create timestamps (most recent first)
        now = datetime.now()
        timestamps = [
            now - pd.Timedelta(days=i) for i in range(len(sentiment_values))
        ]
        
        # Combine into a dataframe
        data = pd.DataFrame({
            'timestamp': timestamps,
            'sentiment': sentiment_values,
            'text': texts,
            'source': ['mock'] * len(timestamps)
        })
        
        return {
            'symbol': symbol,
            'data': data,
            'mock': True
        }

    def _extract_sentiment_data(self, data: Optional[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Extract sentiment data from input or fetch from available sources.
        
        Args:
            data: Optional input data dictionary
            
        Returns:
            Dictionary mapping symbols and topics to DataFrames of sentiment data
        """
        sentiment_data = {}
        
        # Check if input data contains sentiment data
        if data and "sentiment_data" in data:
            self.logger.debug("Using provided sentiment data")
            sentiment_data = data["sentiment_data"]
        else:
            # If no data provided, either fetch from sources or generate mock data
            if SentimentSource.MOCK in self.sentiment_sources:
                self.logger.info("Generating mock sentiment data")
                # Generate mock data for all symbols and topics
                for symbol in self.symbols:
                    sentiment_data[symbol] = self._generate_mock_sentiment_data(symbol)
                
                # Also generate for topics
                for topic in self.topics:
                    sentiment_data[topic] = self._generate_mock_sentiment_data(topic)
            else:
                # TODO: Implement fetching from real sources
                self.logger.warning("Fetching from real sources not yet implemented")
                # Fallback to mock data
                for symbol in self.symbols:
                    sentiment_data[symbol] = self._generate_mock_sentiment_data(symbol)
        
        return sentiment_data

    def _analyze_sentiment_trends(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment trends from the provided data.
        
        Args:
            sentiment_data: Dictionary with sentiment data by symbol/topic
            
        Returns:
            Dictionary with trend analysis results
        """
        trend_results = {}
        
        for key, data in sentiment_data.items():
            if isinstance(data, dict) and 'data' in data and isinstance(data['data'], pd.DataFrame):
                df = data['data']
                
                # Ensure dataframe is sorted by timestamp (newest first)
                df = df.sort_values('timestamp', ascending=False)
                
                # Get trend analysis from sentiment analyzer
                if 'text' in df.columns:
                    trend_analysis = self.sentiment_analyzer.get_sentiment_trend(df['text'].tolist())
                    trend_results[key] = trend_analysis
                elif 'sentiment' in df.columns:
                    # If we only have sentiment scores (no text), calculate trend directly
                    compounds = df['sentiment'].tolist()
                    
                    # Calculate trend metrics using numpy
                    mean = np.mean(compounds)
                    
                    # Detect trend direction
                    if len(compounds) > 1:
                        # Simple linear regression for trend slope
                        x = np.arange(len(compounds))
                        A = np.vstack([x, np.ones(len(x))]).T
                        slope, _ = np.linalg.lstsq(A, compounds, rcond=None)[0]
                        
                        if slope > 0.05:
                            trend = 'improving'
                        elif slope < -0.05:
                            trend = 'deteriorating'
                        else:
                            trend = 'stable'
                    else:
                        trend = 'stable'
                        slope = 0
                    
                    # Categorize overall sentiment
                    if mean > 0.2:
                        sentiment = 'positive'
                    elif mean < -0.2:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    
                    trend_results[key] = {
                        'trend': trend,
                        'sentiment': sentiment,
                        'slope': slope,
                        'mean': mean,
                        'volatility': np.std(compounds),
                        'data': compounds
                    }
        
        return trend_results

    def _generate_sentiment_signals(self, trend_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on sentiment trend analysis.
        
        Args:
            trend_results: Dictionary with sentiment trend analysis by symbol/topic
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        for symbol_or_topic, trend_data in trend_results.items():
            # Skip if this isn't one of our trading symbols
            if symbol_or_topic not in self.symbols and symbol_or_topic not in self.topics:
                continue
                
            # Get mean sentiment score
            mean_sentiment = trend_data.get('mean', 0)
            
            # Get trend direction
            trend = trend_data.get('trend', 'stable')
            
            # Get volatility
            volatility = trend_data.get('volatility', 0)
            
            # Determine signal strength
            signal_strength = 0.0
            signal_type = "neutral"
            
            # Base signal on mean sentiment
            if mean_sentiment > self.signal_thresholds['strong_buy']:
                signal_strength = 1.0
                signal_type = "strong_buy"
            elif mean_sentiment > self.signal_thresholds['buy']:
                signal_strength = 0.5
                signal_type = "buy"
            elif mean_sentiment < self.signal_thresholds['strong_sell']:
                signal_strength = -1.0
                signal_type = "strong_sell"
            elif mean_sentiment < self.signal_thresholds['sell']:
                signal_strength = -0.5
                signal_type = "sell"
            
            # Adjust signal strength based on trend
            if trend == 'improving':
                signal_strength += 0.2
            elif trend == 'deteriorating':
                signal_strength -= 0.2
            
            # Clamp signal strength to [-1, 1]
            signal_strength = max(-1.0, min(1.0, signal_strength))
            
            # Calculate confidence based on volatility (lower volatility = higher confidence)
            confidence = max(0.1, 1.0 - volatility * 2)
            
            # Create signal
            is_topic = symbol_or_topic in self.topics
            signal = {
                "source": "sentiment_analysis",
                "timestamp": datetime.now().isoformat(),
                "payload": {
                    "symbol" if not is_topic else "topic": symbol_or_topic,
                    "is_topic": is_topic,
                    "signal_strength": signal_strength,
                    "signal_type": signal_type,
                    "confidence": confidence,
                    "mean_sentiment": mean_sentiment,
                    "trend": trend,
                    "volatility": volatility
                }
            }
            
            signals.append(signal)
        
        return signals

    def process(self, data: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Process sentiment data and generate trading signals.
        
        This is the main entry point for the agent's functionality. It processes
        text data through the complete sentiment analysis pipeline:
        1. Extract/fetch sentiment data from sources
        2. Process raw sentiment data into scores
        3. Analyze sentiment trends over time
        4. Generate trading signals based on sentiment analysis
        
        Args:
            data: Optional input data dictionary
            
        Returns:
            List of signal dictionaries or None if processing fails
        """
        self.update_status(AgentStatus.RUNNING)
        self.logger.info(f"Processing sentiment data for {len(self.symbols)} symbols and {len(self.topics)} topics")
        start_time = datetime.now()
        
        try:
            # Extract sentiment data from input or fetch from sources
            sentiment_data = self._extract_sentiment_data(data)
            if not sentiment_data:
                self.logger.warning("No valid sentiment data provided or extracted")
                return None
            
            # Store the raw sentiment data in cache
            self.sentiment_data_cache = sentiment_data
            
            # Step 1: Analyze sentiment trends
            trend_results = self._analyze_sentiment_trends(sentiment_data)
            
            # Store the analyzed trends in the sentiment state
            self.sentiment_state["trends"] = trend_results
            
            # Step 2: Generate trading signals based on sentiment analysis
            signals = self._generate_sentiment_signals(trend_results)
            
            # Store the generated signals
            self.current_signals = {
                s["payload"].get("symbol", s["payload"].get("topic")): s 
                for s in signals
            }
            
            # Track processing time and metrics
            process_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Count total texts analyzed
            texts_analyzed = sum(
                len(data['data']) if isinstance(data, dict) and 'data' in data else 0
                for data in sentiment_data.values()
            )
            
            # Update agent metrics
            self.update_metrics({
                "avg_processing_time_ms": process_time,
                "signals_generated": len(signals),
                "texts_analyzed": texts_analyzed
            })
            
            self.logger.info(
                f"Processed sentiment data in {process_time:.2f}ms, analyzed {texts_analyzed} texts, "
                f"generated {len(signals)} signals"
            )
            self.update_status(AgentStatus.IDLE)
            return signals
            
        except Exception as e:
            self.logger.error(f"Error processing sentiment data: {str(e)}", exc_info=True)
            self.update_metrics({"processing_errors": 1})
            self.update_status(AgentStatus.ERROR)
            return None
    
    def get_sentiment_state(self) -> Dict[str, Any]:
        """Get the current sentiment analysis state."""
        return self.sentiment_state
    
    def get_component_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from all components."""
        metrics = {
            "agent": self.metrics,
            "sentiment_analyzer": self.sentiment_analyzer.get_performance_stats()
            # Add more component metrics as needed
        }
        
        return metrics
