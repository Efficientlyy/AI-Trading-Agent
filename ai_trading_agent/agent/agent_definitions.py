from enum import Enum
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
import time

# Import the SentimentAnalyzer for proper integration
from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer

class AgentRole(Enum):
    SPECIALIZED_SENTIMENT = "specialized_sentiment"
    SPECIALIZED_TECHNICAL = "specialized_technical"
    SPECIALIZED_NEWS = "specialized_news"
    SPECIALIZED_FUNDAMENTAL = "specialized_fundamental"
    DECISION_AGGREGATOR = "decision_aggregator"
    EXECUTION_BROKER = "execution_broker"
    # Add other roles as needed

class AgentStatus(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    # Add other statuses as needed

class BaseAgent:
    """
    Base class for all agents in the AI Trading Agent system.
    Corresponds to the conceptual data structure outlined in the agent_flow_architecture.md.
    """
    def __init__(self,
                 agent_id: str,
                 name: str,
                 agent_role: AgentRole,
                 agent_type: str, # e.g., "AlphaVantageSentiment", "RSIMACDStrategy"
                 status: AgentStatus = AgentStatus.IDLE,
                 inputs_from: Optional[List[str]] = None,
                 outputs_to: Optional[List[str]] = None,
                 config_details: Optional[Dict] = None,
                 metrics: Optional[Dict] = None,
                 symbols: Optional[List[str]] = None):
        self.agent_id: str = agent_id
        self.name: str = name
        self.agent_role: AgentRole = agent_role
        self.agent_type: str = agent_type
        self.status: AgentStatus = status
        self.inputs_from: List[str] = inputs_from or []
        self.outputs_to: List[str] = outputs_to or []
        self.config_details: Dict = config_details or {}
        self.metrics: Dict = metrics or {}
        self.last_updated: datetime = datetime.now()
        self.symbols: List[str] = symbols or []

    def update_status(self, new_status: AgentStatus):
        self.status = new_status
        self.last_updated = datetime.now()
        print(f"Agent {self.agent_id} ({self.name}) status updated to: {new_status.value}")

    def update_metrics(self, new_metrics: Dict):
        self.metrics.update(new_metrics)
        self.last_updated = datetime.now()

    def get_info(self) -> Dict:
        """Returns a dictionary representation of the agent's current state."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_role": self.agent_role.value,
            "type": self.agent_type,
            "status": self.status.value,
            "inputs_from": self.inputs_from,
            "outputs_to": self.outputs_to,
            "config_details": self.config_details,
            "metrics": self.metrics,
            "last_updated": self.last_updated.isoformat(),
            "symbols": self.symbols
        }

    def process(self, data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Main processing logic for the agent.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Each agent must implement its own process method.")

    def start(self):
        """Logic to start the agent's operation."""
        self.update_status(AgentStatus.INITIALIZING)
        # Placeholder for actual start logic
        self.update_status(AgentStatus.RUNNING)
        print(f"Agent {self.agent_id} ({self.name}) started.")

    def stop(self):
        """Logic to stop the agent's operation."""
        self.update_status(AgentStatus.STOPPED)
        print(f"Agent {self.agent_id} ({self.name}) stopped.")

# Example Specialized Agent
class SentimentAnalysisAgent(BaseAgent):
    AGENT_ID_PREFIX = "spec_sentiment_"

    def __init__(self, agent_id_suffix: str, name: str, agent_type: str, symbols: List[str], config_details: Optional[Dict] = None):
        super().__init__(
            agent_id=f"{self.AGENT_ID_PREFIX}{agent_id_suffix}",
            name=name,
            agent_role=AgentRole.SPECIALIZED_SENTIMENT,
            agent_type=agent_type,
            symbols=symbols,
            config_details=config_details or {}
        )
        
        # Enhanced logging setup with more context
        self.logger = logging.getLogger(f"SentimentAgent_{agent_id_suffix}")
        
        # Initialize the API key from config with environment variable fallback
        self.api_key = self.config_details.get("api_key", os.environ.get("ALPHA_VANTAGE_API_KEY", None))
        
        # Initialize local cache for sentiment data to reduce API calls
        self.cache = {
            "sentiment_data": {},  # Format: {symbol: {"data": df, "timestamp": datetime}}
            "signals": {},        # Format: {symbol: {"signal": dict, "timestamp": datetime}}
        }
        self.cache_ttl = self.config_details.get("cache_ttl", 3600)  # 1 hour default cache TTL
        
        # Configure the SentimentAnalyzer with enhanced options
        sentiment_config = {
            "alpha_vantage_api_key": self.api_key,
            "alpha_vantage_client": {
                "api_key": self.api_key,
                "tier": self.config_details.get("api_tier", "free"),
                "use_cache": self.config_details.get("use_cache", True),
                "cache_ttl": self.cache_ttl,
                "max_retries": self.config_details.get("max_retries", 3),
                "retry_delay": self.config_details.get("retry_delay", 10)
            },
            "sentiment_threshold": self.config_details.get("sentiment_threshold", 0.2),
            "sentiment_window": self.config_details.get("sentiment_window", 5),
            "default_lags": self.config_details.get("default_lags", [1, 2, 3, 5, 10, 21]),
            "default_windows": self.config_details.get("default_windows", [5, 10, 21, 63]),
            "custom_topics": self.config_details.get("custom_topics", {}),  # Symbol to topic mapping
            "include_twitter": self.config_details.get("include_twitter", False),
            "include_reddit": self.config_details.get("include_reddit", False)
        }
        
        # Initialize the SentimentAnalyzer
        try:
            self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)
            self.logger.info(f"SentimentAnalyzer initialized for agent {self.agent_id}")
            
            # Set up initial metrics for tracking agent performance
            initial_metrics = {
                "api_calls": 0,
                "cache_hits": 0, 
                "cache_misses": 0,
                "data_fetch_errors": 0,
                "analysis_errors": 0,
                "processing_errors": 0,
                "avg_processing_time": 0,
                "total_signals_generated": 0,
                "avg_sentiment_score": 0,
                "bullish_signals": 0,
                "bearish_signals": 0,
                "neutral_signals": 0
            }
            self.update_metrics(initial_metrics)
            
        except Exception as e:
            self.update_status(AgentStatus.ERROR)
            self.logger.error(f"Failed to initialize SentimentAnalyzer: {str(e)}")
            self.sentiment_analyzer = None

    def _fetch_news_data(self, symbol: str) -> Union[pd.DataFrame, None]:
        """
        Fetch real sentiment data for a given symbol using the SentimentAnalyzer.
        Includes caching, error handling, and API rate limit management.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD', 'ETH/USD')
            
        Returns:
            DataFrame with sentiment data or None if data couldn't be fetched
        """
        if not self.sentiment_analyzer:
            self.logger.error(f"Cannot fetch news: SentimentAnalyzer not initialized for {self.agent_id}")
            return None
            
        # Check for API key and log appropriate messages
        if not self.api_key and self.agent_type.lower() in ["alphavantage", "alphavantagesenti", "alphavantagesentiment", "alphavantagenews"]:
            self.logger.warning(f"API key not configured for {self.agent_type}. Will try fallback methods.")
        
        # Extract the base symbol (e.g., 'BTC' from 'BTC/USD')
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
        
        # Check cache first for this symbol
        if base_symbol in self.cache["sentiment_data"]:
            cache_entry = self.cache["sentiment_data"][base_symbol]
            cache_age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
            
            # If cache is still valid, use it
            if cache_age < self.cache_ttl:
                self.logger.info(f"Using cached sentiment data for {base_symbol} (age: {int(cache_age)}s)")
                self.update_metrics({"cache_hits": self.metrics.get("cache_hits", 0) + 1})
                return cache_entry["data"]
            else:
                self.logger.info(f"Cached data for {base_symbol} expired ({int(cache_age)}s old). Fetching fresh data.")
        
        self.logger.info(f"Fetching news data for {symbol}...")
        self.update_metrics({"cache_misses": self.metrics.get("cache_misses", 0) + 1})
        self.update_metrics({"api_calls": self.metrics.get("api_calls", 0) + 1})
        
        # Implement exponential backoff for retries
        max_retries = self.config_details.get("max_retries", 3)
        base_delay = self.config_details.get("retry_delay", 5)
        
        for attempt in range(max_retries + 1):
            try:
                # Get custom topics mapping if available in config
                custom_topics = self.config_details.get("custom_topics", {})
                custom_topic = custom_topics.get(base_symbol.upper())
                
                # Default topic mapping for common crypto symbols
                topic_map = {
                    "BTC": "bitcoin",
                    "ETH": "ethereum",
                    "XRP": "ripple",
                    "SOL": "solana",
                    "ADA": "cardano",
                    "DOT": "polkadot",
                    "AVAX": "avalanche",
                    "DOGE": "dogecoin",
                    "SHIB": "shiba inu",
                    "MATIC": "polygon",
                    "LTC": "litecoin"
                }
                
                sentiment_data = None
                
                # Try fetching by crypto ticker first
                sentiment_data = self.sentiment_analyzer.fetch_sentiment_data(crypto_ticker=base_symbol)
                
                # If empty, try by custom topic if available
                if (sentiment_data is None or sentiment_data.empty) and custom_topic:
                    self.logger.info(f"No direct sentiment data for {base_symbol}, trying custom topic: {custom_topic}")
                    sentiment_data = self.sentiment_analyzer.fetch_sentiment_data(topic=custom_topic)
                
                # If still empty, try by mapped topic
                if (sentiment_data is None or sentiment_data.empty):
                    topic = topic_map.get(base_symbol.upper(), "cryptocurrency")
                    self.logger.info(f"No sentiment data found, trying mapped topic: {topic}")
                    sentiment_data = self.sentiment_analyzer.fetch_sentiment_data(topic=topic)
                
                # If still empty after all attempts
                if sentiment_data is None or sentiment_data.empty:
                    self.logger.warning(f"No sentiment data found for {symbol} after all attempts")
                    return None
                
                # Successfully got data, cache it
                self.cache["sentiment_data"][base_symbol] = {
                    "data": sentiment_data,
                    "timestamp": datetime.now()
                }
                
                self.logger.info(f"Successfully fetched sentiment data for {symbol} with {len(sentiment_data)} entries")
                return sentiment_data
                
            except Exception as e:
                # Handle specific error types with different strategies
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    # Rate limit hit - implement backoff
                    if attempt < max_retries:
                        wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(f"Rate limit hit for {symbol}. Retrying in {wait_time}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        self.logger.error(f"Rate limit exceeded for {symbol} after {max_retries} retries")
                elif "API key" in str(e).lower() or "invalid key" in str(e).lower():
                    self.logger.error(f"Invalid API key for {self.agent_type}. Please check your configuration.")
                    self.update_status(AgentStatus.ERROR)
                    break
                else:
                    # Generic error handling
                    self.logger.error(f"Error fetching news data for {symbol} (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries:
                        wait_time = base_delay * (2 ** attempt)
                        time.sleep(wait_time)
                        continue
        
        # If we reached here, all attempts failed
        self.update_metrics({"data_fetch_errors": self.metrics.get("data_fetch_errors", 0) + 1})
        
        # As a last resort, check if we have any cached data, even if expired
        if base_symbol in self.cache["sentiment_data"]:
            self.logger.warning(f"Using expired cache for {symbol} as all fetch attempts failed")
            return self.cache["sentiment_data"][base_symbol]["data"]
            
        return None

    def _analyze_sentiment_from_news(self, sentiment_data: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """
        Analyze sentiment using the full capabilities of SentimentAnalyzer.
        Includes caching, richer metrics, and enhanced signal generation.
        
        Args:
            sentiment_data: DataFrame with sentiment data from Alpha Vantage
            symbol: Trading symbol the data relates to
            
        Returns:
            Dictionary with sentiment analysis results formatted for agent ecosystem
        """
        if sentiment_data is None or sentiment_data.empty:
            self.logger.warning(f"No sentiment data to analyze for {symbol}")
            return None
            
        # Extract the base symbol for cache lookup
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
        
        # Check signal cache for this symbol if it exists
        if base_symbol in self.cache["signals"]:
            cache_entry = self.cache["signals"][base_symbol]
            cache_age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
            
            # If cache is still valid and we don't want to force refresh
            if cache_age < self.cache_ttl / 2:  # Use a shorter TTL for signals than raw data
                self.logger.info(f"Using cached sentiment signal for {base_symbol} (age: {int(cache_age)}s)")
                self.update_metrics({"cache_hits": self.metrics.get("cache_hits", 0) + 1})
                return cache_entry["signal"]
                
        analysis_start = datetime.now()
        try:
            self.logger.info(f"Analyzing sentiment for {symbol} with {len(sentiment_data)} data points")
            
            # First try the multi-stage processing approach
            try:
                # Generate time series features
                processed_data = self.sentiment_analyzer.generate_time_series_features(sentiment_data)
                
                # Calculate weighted sentiment score
                processed_data = self.sentiment_analyzer.calculate_weighted_sentiment_score(processed_data)
                
                # Generate trading signals
                processed_data = self.sentiment_analyzer.generate_trading_signals(processed_data)
            except Exception as processing_err:
                self.logger.warning(f"Error in advanced sentiment processing: {str(processing_err)}. Falling back to basic analysis.")
                # Fall back to simpler analysis if the advanced pipeline fails
                processed_data = sentiment_data.copy()
                if 'sentiment_score' in processed_data.columns:
                    processed_data['weighted_sentiment_score'] = processed_data['sentiment_score']
                else:
                    # Create a basic sentiment score if none exists
                    processed_data['weighted_sentiment_score'] = processed_data.get('relevance_score', 0.5) * \
                                                           processed_data.get('sentiment', 0)
            
            if processed_data.empty:
                self.logger.warning(f"No processed sentiment data for {symbol} after analysis")
                return None
                
            # Extract the most recent sentiment data point
            latest = processed_data.iloc[-1]
            
            # Determine trend based on signal value
            signal_value = latest.get('signal', 0)
            if isinstance(signal_value, pd.Series):
                signal_value = signal_value.iloc[0]  # Handle Series objects
            
            # Use configurable thresholds
            threshold = self.config_details.get("sentiment_threshold", 0.2)
            
            trend = "neutral"
            if signal_value > threshold:
                trend = "bullish"
                self.update_metrics({"bullish_signals": self.metrics.get("bullish_signals", 0) + 1})
            elif signal_value < -threshold:
                trend = "bearish"
                self.update_metrics({"bearish_signals": self.metrics.get("bearish_signals", 0) + 1})
            else:
                self.update_metrics({"neutral_signals": self.metrics.get("neutral_signals", 0) + 1})
                
            # Get the raw sentiment score
            sentiment_score = latest.get('weighted_sentiment_score', 0)
            if isinstance(sentiment_score, pd.Series):
                sentiment_score = sentiment_score.iloc[0]
            
            # Track average sentiment score in metrics
            current_avg = self.metrics.get("avg_sentiment_score", 0)
            signal_count = self.metrics.get("total_signals_generated", 0)
            new_avg = ((current_avg * signal_count) + float(sentiment_score)) / (signal_count + 1) if signal_count > 0 else float(sentiment_score)
            self.update_metrics({"avg_sentiment_score": round(new_avg, 3)})
                
            # Calculate sentiment volatility as a confidence metric
            volatility = processed_data.get('sentiment_volatility', pd.Series([0.5])).iloc[-1]
            if isinstance(volatility, pd.Series):
                volatility = volatility.iloc[0]
            confidence = max(0.1, min(0.9, 1.0 - (float(volatility) if not pd.isna(volatility) else 0.5)))
            
            # Calculate additional metrics helpful for decision making
            sentiment_momentum = latest.get('sentiment_momentum', 0)
            if isinstance(sentiment_momentum, pd.Series):
                sentiment_momentum = sentiment_momentum.iloc[0]
            
            # Get sentiment volume - how much discussion there is (indicates importance)
            volume = len(sentiment_data)
            volume_score = min(1.0, volume / 50)  # Normalize: 50+ articles is max score
            
            # Get sentiment recency - how fresh the data is
            try:
                time_diffs = []
                current_time = datetime.now()
                for _, row in sentiment_data.iterrows():
                    if 'time_published' in row:
                        pub_time = pd.to_datetime(row['time_published'])
                        time_diffs.append((current_time - pub_time).total_seconds() / 3600)  # hours
                avg_age = sum(time_diffs) / len(time_diffs) if time_diffs else 48
                recency_score = max(0, min(1.0, 1 - (avg_age / 48)))  # 48hr+ is 0, 0hr is 1.0
            except Exception as time_err:
                self.logger.warning(f"Error calculating recency: {str(time_err)}")
                recency_score = 0.5  # Default if calculation fails
            
            # Calculate processing time for performance monitoring
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            self.update_metrics({"avg_processing_time": analysis_time})
            
            # Create enriched signal result
            result = {
                "symbol": symbol,
                "sentiment_score": round(float(sentiment_score), 3),
                "signal_strength": round(abs(float(signal_value)), 3),
                "trend": trend,
                "source_articles": volume,
                "confidence": round(confidence, 2),
                "momentum": round(float(sentiment_momentum), 3) if sentiment_momentum is not None else 0,
                "recency_score": round(recency_score, 2),
                "volume_score": round(volume_score, 2),
                "analysis_time": datetime.now().isoformat(),
                # Add actionable trading strategy suggestion
                "suggested_action": "buy" if trend == "bullish" else "sell" if trend == "bearish" else "hold",
                "action_confidence": round(confidence * abs(float(signal_value)), 2)
            }
            
            # Cache the result
            self.cache["signals"][base_symbol] = {
                "signal": result,
                "timestamp": datetime.now()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            self.update_metrics({"analysis_errors": self.metrics.get("analysis_errors", 0) + 1})
            
            # Check if we have a cached signal, even if expired
            if base_symbol in self.cache["signals"]:
                self.logger.warning(f"Using expired signal cache for {symbol} as analysis failed")
                return self.cache["signals"][base_symbol]["signal"]
                
            return None

    def process(self, data: Optional[Dict] = None) -> Optional[List[Dict]]:
        """
        Processes incoming data or fetches data to generate sentiment signals.
        Implements caching, better error handling, and comprehensive integration 
        with other components of the agent ecosystem.
        
        Args:
            data: Optional pre-processed data from other agents or external sources
            
        Returns:
            List of sentiment signals in the format expected by the agent ecosystem,
            or None if no signals could be generated
        """
        all_signals = []
        processing_start = datetime.now()
        self.update_status(AgentStatus.RUNNING)
        
        try:
            # Handle incoming data from other agents
            if data:
                self.logger.info(f"Processing incoming data: {data.get('type', 'unknown')}")
                
                # Case 1: Feedback from decision agent or execution agent
                if data.get("type") == "decision_feedback" or data.get("type") == "execution_feedback":
                    # Process feedback to adjust strategy
                    self.logger.info(f"Received feedback from {data.get('source', 'unknown')}")
                    feedback = data.get("payload", {})
                    
                    # Update agent's metrics based on feedback
                    if "accuracy" in feedback:
                        self.update_metrics({"signal_accuracy": feedback["accuracy"]})
                    if "trade_result" in feedback:
                        result = feedback["trade_result"]
                        if result == "success":
                            self.update_metrics({"successful_trades": self.metrics.get("successful_trades", 0) + 1})
                        elif result == "failure":
                            self.update_metrics({"failed_trades": self.metrics.get("failed_trades", 0) + 1})
                    
                    # Adjust sentiment thresholds based on feedback
                    if "suggested_adjustment" in feedback:
                        adjustment = feedback["suggested_adjustment"]
                        if "sentiment_threshold" in adjustment:
                            new_threshold = self.config_details.get("sentiment_threshold", 0.2) + adjustment["sentiment_threshold"]
                            # Ensure threshold stays reasonable
                            new_threshold = max(0.05, min(0.5, new_threshold))
                            self.config_details["sentiment_threshold"] = new_threshold
                            self.logger.info(f"Adjusted sentiment threshold to {new_threshold} based on feedback")
                    
                    # No signals to return for feedback messages
                    return [{"type": "sentiment_adjustment", "payload": {"agent_id": self.agent_id, "status": "adjusted"}}]
                    
                # Case 2: Pre-analyzed sentiment data from external source
                elif data.get("type") == "pre_analyzed_sentiment" and data.get("payload"):
                    signal = data.get("payload")
                    all_signals.append(signal)
                    
                    # Store in cache
                    if "symbol" in signal:
                        base_symbol = signal["symbol"].split('/')[0] if '/' in signal["symbol"] else signal["symbol"]
                        self.cache["signals"][base_symbol] = {
                            "signal": signal,
                            "timestamp": datetime.now()
                        }
                    self.update_metrics({"external_signals_processed": self.metrics.get("external_signals_processed", 0) + 1})
                
                # Case 3: Raw news data that needs analysis
                elif data.get("type") == "raw_news" and data.get("items") and data.get("symbol"):
                    # Convert raw news to DataFrame format expected by SentimentAnalyzer
                    news_df = pd.DataFrame(data.get("items"))
                    if not news_df.empty:
                        sentiment_result = self._analyze_sentiment_from_news(news_df, data.get("symbol"))
                        if sentiment_result:
                            all_signals.append(sentiment_result)
                            self.update_metrics({"raw_news_signals_generated": self.metrics.get("raw_news_signals_generated", 0) + 1})
                
                # Case 4: Request for specific symbol analysis from another agent
                elif data.get("type") == "sentiment_request" and data.get("symbol"):
                    symbol = data.get("symbol")
                    force_refresh = data.get("force_refresh", False)
                    
                    # Process the requested symbol, with option to force refresh
                    if force_refresh:
                        # Clear caches for this symbol
                        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                        if base_symbol in self.cache["sentiment_data"]:
                            del self.cache["sentiment_data"][base_symbol]
                        if base_symbol in self.cache["signals"]:
                            del self.cache["signals"][base_symbol]
                            
                    # Process the symbol with two-step approach
                    self._process_symbol_two_step(symbol, all_signals)
            else:
                # No external data, fetch our own based on configured symbols
                self.logger.info(f"Fetching data for {len(self.symbols)} configured symbols")
                
                # For each configured symbol, get sentiment data and analyze it
                for symbol in self.symbols:
                    # Try optimized path first
                    base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                    
                    # Process with optimized method if available - handle errors internally
                    if hasattr(self.sentiment_analyzer, 'analyze_sentiment'):
                        try:
                            # Function for unified signal processing and formatting
                            sentiment_result = self._process_with_analyze_sentiment(symbol)
                            if sentiment_result:
                                all_signals.append(sentiment_result)  
                        except Exception as e:
                            self.logger.error(f"Error using analyze_sentiment for {symbol}: {str(e)}")
                            # Fall back to two-step process if direct method fails
                            self._process_symbol_two_step(symbol, all_signals)
                    else:
                        # If direct method doesn't exist, use two-step process
                        self._process_symbol_two_step(symbol, all_signals)
                        
            # If no signals were generated after all processing
            if not all_signals:
                # Try using cached signals if nothing fresh is available
                for symbol in self.symbols:
                    base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                    if base_symbol in self.cache["signals"]:
                        self.logger.warning(f"No fresh signals generated. Using cached signal for {symbol}")
                        all_signals.append(self.cache["signals"][base_symbol]["signal"])
                        break
                
                # If still no signals
                if not all_signals:
                    self.logger.warning("No sentiment signals generated during processing")
                    return None
                
            # Calculate processing time and update metrics
            processing_time = (datetime.now() - processing_start).total_seconds()
            self.update_metrics({"avg_processing_time": processing_time})
            
            # Prepare signals for other agents
            formatted_signals = []
            for signal in all_signals:
                signal_entry = {
                    "type": "sentiment_signal",
                    "source": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                    "payload": signal
                }
                
                # Add correlations with other data types if available
                if hasattr(self, '_correlate_with_price_data'):
                    signal_entry["correlations"] = self._correlate_with_price_data(signal["symbol"])
                    
                formatted_signals.append(signal_entry)
                
            return formatted_signals
            
        except Exception as e:
            self.logger.error(f"Error in sentiment agent processing: {str(e)}")
            self.update_status(AgentStatus.ERROR)
            
            # Update metrics for monitoring
            self.update_metrics({"processing_errors": self.metrics.get("processing_errors", 0) + 1})
            
            # Try to provide a fallback response using cached data
            fallback_signals = []
            for symbol in self.symbols:
                base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                if base_symbol in self.cache["signals"]:
                    signal = self.cache["signals"][base_symbol]["signal"]
                    fallback_signals.append({
                        "type": "sentiment_signal",
                        "source": self.agent_id,
                        "timestamp": datetime.now().isoformat(),
                        "payload": signal,
                        "is_fallback": True
                    })
                    
            if fallback_signals:
                self.logger.warning(f"Returning {len(fallback_signals)} fallback signals due to processing error")
                return fallback_signals
                
            return None
            
    def _process_with_analyze_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Process a symbol using the SentimentAnalyzer's direct analyze_sentiment method.
        This is more efficient than the two-step approach when available.
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Formatted sentiment signal or None if analysis failed
        """
        # Extract base symbol for processing and caching
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
        
        # Check the cache first
        if base_symbol in self.cache["signals"]:
            cache_entry = self.cache["signals"][base_symbol]
            cache_age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
            
            # If cache is still valid
            if cache_age < self.cache_ttl / 2:  # Use shorter TTL for signals than raw data
                self.logger.info(f"Using cached sentiment signal for {base_symbol} (age: {int(cache_age)}s)")
                self.update_metrics({"cache_hits": self.metrics.get("cache_hits", 0) + 1})
                return cache_entry["signal"]
        
        try:
            # Call the direct method which handles both fetching and analysis
            sentiment_df = self.sentiment_analyzer.analyze_sentiment(crypto_ticker=base_symbol)
            
            if sentiment_df is None or sentiment_df.empty:
                self.logger.warning(f"analyze_sentiment returned no data for {symbol}")
                return None
                
            # Extract the most recent sentiment data point
            latest = sentiment_df.iloc[-1]
            
            # Get key metrics
            sentiment_score = float(latest.get('weighted_sentiment_score', 0)) 
            if isinstance(sentiment_score, pd.Series):
                sentiment_score = sentiment_score.iloc[0]
                
            signal_value = float(latest.get('signal', 0))
            if isinstance(signal_value, pd.Series):
                signal_value = signal_value.iloc[0]
            
            # Use configurable thresholds
            threshold = self.config_details.get("sentiment_threshold", 0.2)
            
            # Determine trend and update metrics
            trend = "neutral"
            if signal_value > threshold:
                trend = "bullish"
                self.update_metrics({"bullish_signals": self.metrics.get("bullish_signals", 0) + 1})
            elif signal_value < -threshold:
                trend = "bearish"
                self.update_metrics({"bearish_signals": self.metrics.get("bearish_signals", 0) + 1})
            else:
                self.update_metrics({"neutral_signals": self.metrics.get("neutral_signals", 0) + 1})
            
            # Try to extract volatility and momentum data
            volatility = 0.5  # Default value
            if 'sentiment_volatility' in latest:
                vol_value = latest['sentiment_volatility']
                if isinstance(vol_value, pd.Series):
                    vol_value = vol_value.iloc[0]
                volatility = float(vol_value) if not pd.isna(vol_value) else 0.5
            
            momentum = 0  # Default value
            if 'sentiment_momentum' in latest:
                mom_value = latest['sentiment_momentum']
                if isinstance(mom_value, pd.Series):
                    mom_value = mom_value.iloc[0]
                momentum = float(mom_value) if not pd.isna(mom_value) else 0
            
            # Calculate confidence based on volatility
            confidence = max(0.1, min(0.9, 1.0 - volatility))
            
            # Create final signal
            result = {
                "symbol": symbol,
                "sentiment_score": round(sentiment_score, 3),
                "signal_strength": round(abs(signal_value), 3),
                "trend": trend,
                "source_articles": len(sentiment_df),
                "confidence": round(confidence, 2),
                "momentum": round(momentum, 3),
                "analysis_time": datetime.now().isoformat(),
                "suggested_action": "buy" if trend == "bullish" else "sell" if trend == "bearish" else "hold",
                "action_confidence": round(confidence * abs(signal_value), 2)
            }
            
            # Cache the result
            self.cache["signals"][base_symbol] = {
                "signal": result,
                "timestamp": datetime.now()
            }
            
            # Update metrics
            self.update_metrics({
                f"signals_generated_{symbol.replace('/', '_')}": self.metrics.get(f"signals_generated_{symbol.replace('/', '_')}", 0) + 1,
                "total_signals_generated": self.metrics.get("total_signals_generated", 0) + 1
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in direct sentiment analysis for {symbol}: {str(e)}")
            # Don't update error metrics here - will be done in process() if fallback fails too
            return None

    def _process_symbol_two_step(self, symbol: str, all_signals: List[Dict]):
        """Helper method to process a symbol using the two-step fetch+analyze approach"""
        try:
            # Step 1: Fetch data
            sentiment_data = self._fetch_news_data(symbol)
            
            # Step 2: Analyze if data available
            if sentiment_data is not None and not sentiment_data.empty:
                sentiment_result = self._analyze_sentiment_from_news(sentiment_data, symbol)
                if sentiment_result:
                    all_signals.append(sentiment_result)
                    self.update_metrics({
                        f"signals_generated_{symbol.replace('/', '_')}": self.metrics.get(f"signals_generated_{symbol.replace('/', '_')}", 0) + 1,
                        "total_signals_generated": self.metrics.get("total_signals_generated", 0) + 1
                    })
            else:
                self.logger.warning(f"No sentiment data fetched for {symbol}")
        except Exception as e:
            self.logger.error(f"Error in two-step processing for {symbol}: {str(e)}")
            self.update_metrics({"processing_errors": self.metrics.get("processing_errors", 0) + 1})

# Example Technical Analysis Agent
class TechnicalAnalysisAgent(BaseAgent):
    AGENT_ID_PREFIX = "spec_technical_"

    def __init__(self, agent_id_suffix: str, name: str, agent_type: str, symbols: List[str], config_details: Optional[Dict] = None):
        super().__init__(
            agent_id=f"{self.AGENT_ID_PREFIX}{agent_id_suffix}",
            name=name,
            agent_role=AgentRole.SPECIALIZED_TECHNICAL,
            agent_type=agent_type,
            symbols=symbols,
            config_details=config_details
        )
        # Example: Store specific config like RSI period
        self.rsi_period = self.config_details.get("rsi_period", 14)

    def _fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Placeholder method to simulate fetching market data for a given symbol.
        In a real implementation, this would call a market data API.
        """
        print(f"TechnicalAnalysisAgent ({self.agent_id}) attempting to fetch market data for {symbol}...")
        # Simulate API call - returning OHLCV data for example
        if symbol == "BTC/USD":
            return {"timestamp": datetime.now().isoformat(), "open": 60000, "high": 61000, "low": 59000, "close": 60500, "volume": 1000}
        elif symbol == "ETH/USD":
            return {"timestamp": datetime.now().isoformat(), "open": 3000, "high": 3050, "low": 2950, "close": 3020, "volume": 15000}
        return None

    def _calculate_indicators(self, market_data: Dict, symbol: str) -> Optional[Dict]:
        """
        Placeholder for actual technical indicator calculation.
        """
        if not market_data:
            return None
        
        print(f"TechnicalAnalysisAgent ({self.agent_id}) calculating indicators for {symbol} using data: {market_data}")
        # Example: Simulate RSI and a simple Moving Average Crossover signal
        # This is extremely simplified. Real calculations would use libraries like TA-Lib or pandas_ta.
        rsi_value = 55 # Simulated RSI
        ma_short = market_data.get("close", 0) * 0.98 # Simulated short MA
        ma_long = market_data.get("close", 0) * 0.95  # Simulated long MA
        
        signal = "neutral"
        if ma_short > ma_long and rsi_value > 50:
            signal = "buy"
        elif ma_short < ma_long and rsi_value < 50:
            signal = "sell"
            
        return {
            "symbol": symbol,
            "indicator_rsi": rsi_value,
            "indicator_ma_short": round(ma_short, 2),
            "indicator_ma_long": round(ma_long, 2),
            "signal": signal,
            "price_at_signal": market_data.get("close")
        }

    def process(self, data: Optional[Dict] = None) -> Optional[List[Dict]]: # Can return multiple signals
        """
        Processes incoming market data or fetches its own to generate technical signals.
        """
        all_signals = []

        if data:
            # Process externally provided market data
            print(f"TechnicalAnalysisAgent ({self.agent_id}) processing externally provided data: {data}")
            # Assuming data contains 'symbol' and other necessary market info
            symbol = data.get("symbol")
            if symbol:
                technical_result = self._calculate_indicators(data, symbol)
                if technical_result:
                    all_signals.append(technical_result)
            else:
                print(f"TechnicalAnalysisAgent ({self.agent_id}): External data missing 'symbol'. Data: {data}")
        else:
            # No external data, fetch its own based on configured symbols
            print(f"TechnicalAnalysisAgent ({self.agent_id}) fetching its own data for symbols: {self.symbols}")
            for symbol in self.symbols:
                market_data = self._fetch_market_data(symbol)
                if market_data:
                    technical_result = self._calculate_indicators(market_data, symbol)
                    if technical_result:
                        all_signals.append(technical_result)
                        self.update_metrics({
                            f"indicators_calculated_{symbol.replace('/', '_')}": self.metrics.get(f"indicators_calculated_{symbol.replace('/', '_')}", 0) + 1,
                            "total_indicators_calculated": self.metrics.get("total_indicators_calculated", 0) + 1
                        })
                else:
                    print(f"TechnicalAnalysisAgent ({self.agent_id}): No market data fetched for {symbol}.")
        
        if not all_signals:
            return None
            
        return [{"type": "technical_signal", "payload": signal} for signal in all_signals]

# Example News Event Agent
class NewsEventAgent(BaseAgent):
    AGENT_ID_PREFIX = "spec_news_"

    def __init__(self, agent_id_suffix: str, name: str, agent_type: str, symbols: Optional[List[str]] = None, event_keywords: Optional[List[str]] = None, config_details: Optional[Dict] = None):
        super().__init__(
            agent_id=f"{self.AGENT_ID_PREFIX}{agent_id_suffix}",
            name=name,
            agent_role=AgentRole.SPECIALIZED_NEWS, # Ensure this role is in AgentRole enum
            agent_type=agent_type, # e.g., "EarningsWatcher", "MacroEventMonitor"
            symbols=symbols or [],
            config_details=config_details or {}
        )
        self.event_keywords = event_keywords or ["earnings", "fda approval", "acquisition", "regulatory change", "report"]
        self.news_api_key = self.config_details.get("news_api_key") # Example: if API key needed

    def _fetch_news_events(self, symbol: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Placeholder to simulate fetching news events.
        Could filter by symbol or look for general market-moving news.
        """
        print(f"NewsEventAgent ({self.agent_id}) attempting to fetch news events (symbol: {symbol})...")
        # Simulate API call or DB query
        simulated_events = [
            {"title": "CompanyX Earnings Beat Expectations!", "summary": "Shares of CompanyX surged today...", "source": "BusinessWire", "timestamp": datetime.now().isoformat(), "symbols_affected": ["X"], "keywords_detected": ["earnings", "beat expectations"]},
            {"title": "Major FDA Drug Approval for PharmaCo", "summary": "PharmaCo receives green light for new drug...", "source": "PharmaNews", "timestamp": datetime.now().isoformat(), "symbols_affected": ["PHRM"], "keywords_detected": ["fda approval", "drug"]},
            {"title": "TechGiant Acquires Startup Innovate Inc.", "summary": "The acquisition is valued at $500M...", "source": "TechCrunch", "timestamp": datetime.now().isoformat(), "symbols_affected": ["TGNT", "INVT"], "keywords_detected": ["acquisition"]},
            {"title": "New Crypto Regulations Proposed by SEC", "summary": "SEC Chair outlines new framework...", "source": "CoinDesk", "timestamp": datetime.now().isoformat(), "symbols_affected": ["BTC/USD", "ETH/USD"], "keywords_detected": ["regulatory change", "sec"]},
            {"title": "Global Oil Output Report Released", "summary": "OPEC+ releases monthly output figures...", "source": "Reuters", "timestamp": datetime.now().isoformat(), "symbols_affected": ["OIL"], "keywords_detected": ["report", "oil output"]}
        ]
        
        relevant_events = []
        for event in simulated_events:
            # Filter by symbol if provided for the scan
            if symbol and symbol not in event["symbols_affected"]:
                continue
            
            # Check if any of the agent's configured keywords match the event's detected keywords or title
            event_text_lower = event.get("title", "").lower() + " " + event.get("summary", "").lower()
            if any(keyword in event_text_lower for keyword in self.event_keywords) or \
               any(keyword in event.get("keywords_detected", []) for keyword in self.event_keywords):
                relevant_events.append(event)
        
        return relevant_events if relevant_events else None

    def _analyze_event_impact(self, news_event: Dict) -> Optional[Dict]:
        """
        Placeholder to analyze the potential impact of a news event.
        """
        print(f"NewsEventAgent ({self.agent_id}) analyzing event: {news_event.get('title')}")
        # Simplified impact assessment
        impact_score = 0.0 # Default neutral
        title_lower = news_event.get("title", "").lower()
        
        if "beat expectations" in title_lower or "approval" in title_lower or "positive" in title_lower or "surged" in title_lower:
            impact_score = 0.75
        elif "missed expectations" in title_lower or "regulatory concerns" in title_lower or "dropped" in title_lower or "negative" in title_lower:
            impact_score = -0.65
        elif "acquisition" in title_lower:
            impact_score = 0.5 # Could be positive or negative depending on context, simplified here
        
        return {
            "event_title": news_event.get("title"),
            "symbols_affected": news_event.get("symbols_affected", []),
            "potential_impact_score": impact_score, # -1 (very negative) to 1 (very positive)
            "event_type": next((kw for kw in self.event_keywords if kw in title_lower or kw in news_event.get("keywords_detected",[])), "general_news"),
            "source": news_event.get("source"),
            "timestamp": news_event.get("timestamp")
        }

    def process(self, data: Optional[Dict] = None) -> Optional[List[Dict]]:
        """
        Fetches news events and generates signals based on their potential impact.
        If `data` is provided with a 'symbol', it focuses on that symbol.
        Otherwise, it scans for general events or events related to its configured `symbols`.
        """
        all_event_signals = []
        
        # Determine which symbols to scan for events
        symbols_to_scan_for = set(self.symbols) # Start with configured symbols
        if data and "symbol" in data: # If a specific symbol is passed for targeted scan
            symbols_to_scan_for.add(data["symbol"])
            print(f"NewsEventAgent ({self.agent_id}) processing for specific symbol from data: {data['symbol']}")
        
        if not symbols_to_scan_for: # If no symbols configured and no specific symbol passed, scan general events
             print(f"NewsEventAgent ({self.agent_id}) scanning for general news events (no specific symbols configured/passed).")
             fetched_events = self._fetch_news_events(None) # Pass None to fetch general events
             if fetched_events:
                for event_item in fetched_events:
                    analyzed_event = self._analyze_event_impact(event_item)
                    if analyzed_event:
                        all_event_signals.append(analyzed_event)
                self.update_metrics({"events_processed_general": self.metrics.get("events_processed_general", 0) + len(fetched_events)})
        else:
            for symbol_target in symbols_to_scan_for:
                fetched_events = self._fetch_news_events(symbol_target) # Fetch events potentially relevant to this symbol
                if fetched_events:
                    processed_for_symbol_count = 0
                    for event_item in fetched_events:
                        # Further ensure the event is relevant if _fetch_news_events is broad
                        if symbol_target in event_item.get("symbols_affected", []):
                            analyzed_event = self._analyze_event_impact(event_item)
                            if analyzed_event:
                                all_event_signals.append(analyzed_event)
                            processed_for_symbol_count +=1
                    if processed_for_symbol_count > 0:
                        self.update_metrics({
                            f"events_processed_{symbol_target.replace('/', '_')}": self.metrics.get(f"events_processed_{symbol_target.replace('/', '_')}", 0) + processed_for_symbol_count,
                            "total_events_processed": self.metrics.get("total_events_processed", 0) + processed_for_symbol_count
                        })
        
        if not all_event_signals:
            return None
            
        return [{"type": "news_event_signal", "payload": signal} for signal in all_event_signals]

# Example Fundamental Analysis Agent
class FundamentalAnalysisAgent(BaseAgent):
    AGENT_ID_PREFIX = "spec_fundamental_"

    def __init__(self, agent_id_suffix: str, name: str, agent_type: str, symbols: List[str], config_details: Optional[Dict] = None):
        super().__init__(
            agent_id=f"{self.AGENT_ID_PREFIX}{agent_id_suffix}",
            name=name,
            agent_role=AgentRole.SPECIALIZED_FUNDAMENTAL, # Ensure this role is in AgentRole enum
            agent_type=agent_type, # e.g., "CompanyValuator", "SectorAnalyzer"
            symbols=symbols,
            config_details=config_details or {}
        )
        self.financial_data_provider_key = self.config_details.get("financial_data_api_key")

    def _fetch_fundamental_data(self, symbol: str) -> Optional[Dict]:
        """
        Placeholder to simulate fetching fundamental data (e.g., financial statements, ratios).
        """
        print(f"FundamentalAnalysisAgent ({self.agent_id}) attempting to fetch fundamental data for {symbol}...")
        # Simulate API call to a financial data provider
        if symbol == "X": # Assuming CompanyX from NewsEventAgent example
            return {
                "symbol": "X",
                "pe_ratio": 15.5,
                "pb_ratio": 2.1,
                "debt_to_equity": 0.4,
                "revenue_growth_yoy": 0.12, # 12%
                "earnings_per_share_ttm": 3.50,
                "report_date": (datetime.now() - timedelta(days=30)).isoformat() # type: ignore
            }
        elif symbol == "PHRM":
             return {
                "symbol": "PHRM",
                "pe_ratio": 25.0,
                "pb_ratio": 4.5,
                "debt_to_equity": 0.2,
                "revenue_growth_yoy": 0.18,
                "earnings_per_share_ttm": 2.75,
                "report_date": (datetime.now() - timedelta(days=45)).isoformat() # type: ignore
            }
        return None

    def _analyze_fundamentals(self, fundamental_data: Dict, symbol: str) -> Optional[Dict]:
        """
        Placeholder to analyze fundamental data and generate a valuation signal.
        """
        if not fundamental_data:
            return None
        
        print(f"FundamentalAnalysisAgent ({self.agent_id}) analyzing fundamentals for {symbol}: {fundamental_data}")
        
        # Simplified valuation logic
        valuation_signal = "neutral"
        valuation_score = 0.0 # -1 (overvalued) to 1 (undervalued)

        pe = fundamental_data.get("pe_ratio", 100) # Default to high P/E if not found
        pb = fundamental_data.get("pb_ratio", 10)
        rev_growth = fundamental_data.get("revenue_growth_yoy", 0)

        if pe < 20 and pb < 3 and rev_growth > 0.10:
            valuation_signal = "undervalued"
            valuation_score = 0.7
        elif pe > 30 or pb > 5:
            valuation_signal = "overvalued"
            valuation_score = -0.6
        
        return {
            "symbol": symbol,
            "valuation_signal": valuation_signal,
            "valuation_score": valuation_score,
            "key_metrics": {
                "pe_ratio": pe,
                "pb_ratio": pb,
                "revenue_growth_yoy": rev_growth
            }
        }

    def process(self, data: Optional[Dict] = None) -> Optional[List[Dict]]:
        """
        Fetches and analyzes fundamental data for configured symbols.
        Ignores external `data` input for now, as it primarily fetches its own.
        """
        all_signals = []
        print(f"FundamentalAnalysisAgent ({self.agent_id}) starting processing for symbols: {self.symbols}")
        for symbol in self.symbols:
            fund_data = self._fetch_fundamental_data(symbol)
            if fund_data:
                analysis_result = self._analyze_fundamentals(fund_data, symbol)
                if analysis_result:
                    all_signals.append(analysis_result)
                    self.update_metrics({
                        f"analysis_completed_{symbol.replace('/', '_')}": self.metrics.get(f"analysis_completed_{symbol.replace('/', '_')}", 0) + 1,
                        "total_analyses_completed": self.metrics.get("total_analyses_completed", 0) + 1
                    })
            else:
                print(f"FundamentalAnalysisAgent ({self.agent_id}): No fundamental data fetched for {symbol}.")

        if not all_signals:
            return None
            
        return [{"type": "fundamental_signal", "payload": signal} for signal in all_signals]

# Example Decision Agent
class DecisionAgent(BaseAgent):
    AGENT_ID_PREFIX = "decision_"

    def __init__(self, agent_id_suffix: str, name: str, agent_type: str, config_details: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=f"{self.AGENT_ID_PREFIX}{agent_id_suffix}",
            name=name,
            agent_role=AgentRole.DECISION_AGGREGATOR,
            agent_type=agent_type,
            config_details=config_details or {} # Ensure config_details is a dict
        )
        self.received_signals_buffer: Dict[str, List[Dict[str, Any]]] = {} # Buffer signals per symbol
        
        # Default weights, can be overridden by config_details
        default_weights = {
            "sentiment_signal": 0.3,
            "technical_signal": 0.4,
            "news_event_signal": 0.2,
            "fundamental_signal": 0.1
        }
        self.signal_weights: Dict[str, float] = self.config_details.get("signal_weights", default_weights)
        self.min_signals_for_decision: int = self.config_details.get("min_signals_for_decision", 2) # Min unique signal types
        self.buy_threshold: float = self.config_details.get("buy_threshold", 0.5) # Score threshold to buy
        self.sell_threshold: float = self.config_details.get("sell_threshold", -0.5) # Score threshold to sell

        # Risk Management Config
        self.risk_config: Dict[str, Any] = self.config_details.get("risk_management", {})
        self.default_trade_quantity: float = self.risk_config.get("default_trade_quantity", 0.01)
        self.max_trade_value_usd: float = self.risk_config.get("max_trade_value_usd", 100.0) # e.g., don't place trades over $100
        self.per_symbol_max_quantity: Dict[str, float] = self.risk_config.get("per_symbol_max_quantity", {}) # e.g. {"BTC/USD": 0.1, "ETH/USD": 1}
        
        print(f"DecisionAgent ({self.agent_id}) initialized with weights: {self.signal_weights}, min_signals_types: {self.min_signals_for_decision}, buy_thresh: {self.buy_threshold}, sell_thresh: {self.sell_threshold}")
        print(f"DecisionAgent ({self.agent_id}) Risk Config: default_qty={self.default_trade_quantity}, max_val_usd={self.max_trade_value_usd}, symbol_max_qty={self.per_symbol_max_quantity}")

    def _get_score_from_payload(self, signal_type: str, payload: Dict[str, Any]) -> float:
        """
        Extracts a numerical score from different signal types.
        Scores should ideally be normalized between -1 (strong sell/negative) and 1 (strong buy/positive).
        """
        if signal_type == "sentiment_signal":
            # 'sentiment_score' is expected to be -1 to 1
            return payload.get("sentiment_score", 0.0)
        elif signal_type == "technical_signal":
            # 'signal' can be 'buy', 'sell', 'neutral'
            signal_direction = payload.get("signal", "neutral")
            if signal_direction == "buy": return 1.0
            if signal_direction == "sell": return -1.0
            return 0.0 # Neutral
        elif signal_type == "news_event_signal":
            # 'potential_impact_score' is expected to be -1 to 1
            return payload.get("potential_impact_score", 0.0)
        elif signal_type == "fundamental_signal":
            # 'valuation_score' is expected to be -1 (overvalued) to 1 (undervalued)
            return payload.get("valuation_score", 0.0)
        
        print(f"DecisionAgent ({self.agent_id}): Unknown signal type '{signal_type}' for score extraction from payload: {payload}")
        return 0.0

    def process(self, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Processes incoming signals, aggregates them using weights, and makes trading decisions.
        Input `data` is expected to be a single signal dictionary with 'type' and 'payload'.
        """
        if not data or not isinstance(data, dict) or "type" not in data or "payload" not in data:
            return None # This agent primarily reacts to incoming signals.

        signal_type = str(data.get("type", "unknown_type"))
        payload = data.get("payload")
        
        if not payload or not isinstance(payload, dict):
            print(f"DecisionAgent ({self.agent_id}) received signal of type '{signal_type}' with invalid or missing payload: {payload}")
            return None

        # Determine affected symbols. News events might affect multiple. Fundamental signals usually one.
        symbols_in_payload = payload.get("symbols_affected", []) # Used by NewsEventAgent
        if not symbols_in_payload and "symbol" in payload: # Fallback to 'symbol' if 'symbols_affected' is not present or empty
            symbols_in_payload = [payload.get("symbol")]
        
        if not any(s for s in symbols_in_payload if isinstance(s, str)): # Check if there's at least one valid string symbol
            print(f"DecisionAgent ({self.agent_id}) received signal (type: {signal_type}) without a clear string symbol or symbols_affected: {payload}")
            return None

        # Iterate over each symbol potentially affected by the signal
        for symbol_key in filter(lambda s: isinstance(s, str), symbols_in_payload): # Process only valid string symbols
            print(f"DecisionAgent ({self.agent_id}) processing signal for {symbol_key}: Type={signal_type}")

            if symbol_key not in self.received_signals_buffer:
                self.received_signals_buffer[symbol_key] = []
            
            # Store the full signal (including its type) for context
            # Avoid duplicate signals of the same type for the same symbol from the same source if possible (e.g. by checking timestamp or unique ID if available)
            # For now, just append. A more robust buffer would handle updates or time decay.
            self.received_signals_buffer[symbol_key].append({"type": signal_type, "payload": payload, "received_at": datetime.now()})
            
            self.update_metrics({
                "signals_received_total": self.metrics.get("signals_received_total", 0) + 1,
                f"signals_received_{symbol_key.replace('/', '_')}": self.metrics.get(f"signals_received_{symbol_key.replace('/', '_')}", 0) + 1
            })

            # Aggregate signals for this symbol and decide
            # Consider only the latest signal of each unique type for the current decision
            latest_signals_of_each_type: Dict[str, Dict[str, Any]] = {}
            for buffered_signal in sorted(self.received_signals_buffer[symbol_key], key=lambda s: s["received_at"], reverse=True):
                if buffered_signal["type"] not in latest_signals_of_each_type:
                    latest_signals_of_each_type[buffered_signal["type"]] = buffered_signal["payload"]
            
            if len(latest_signals_of_each_type) >= self.min_signals_for_decision:
                current_weighted_score = 0.0
                total_weight_applied = 0.0
                
                contributing_signals_log = {}

                for s_type, s_payload in latest_signals_of_each_type.items():
                    weight = self.signal_weights.get(s_type, 0.0)
                    if weight > 0: # Only consider signals with configured positive weight
                        raw_score = self._get_score_from_payload(s_type, s_payload)
                        current_weighted_score += raw_score * weight
                        total_weight_applied += weight
                        contributing_signals_log[s_type] = {"raw_score": raw_score, "weight": weight, "weighted_contribution": raw_score * weight}
                
                final_decision_score = 0.0
                if total_weight_applied > 0: # Avoid division by zero
                    final_decision_score = current_weighted_score / total_weight_applied # Normalize by sum of weights of *contributing* signals
                
                print(f"DecisionAgent ({self.agent_id}) for {symbol_key}: ContributingSignals={contributing_signals_log}, WeightedScoreSum={current_weighted_score:.3f}, TotalWeightApplied={total_weight_applied:.2f}, FinalDecisionScore={final_decision_score:.3f}")

                action = "hold"
                if final_decision_score >= self.buy_threshold:
                    action = "buy"
                elif final_decision_score <= self.sell_threshold:
                    action = "sell"

                if action != "hold":
                    proposed_quantity = self.default_trade_quantity # Start with default

                    # Apply risk management: Max quantity per symbol
                    if symbol_key in self.per_symbol_max_quantity:
                        if proposed_quantity > self.per_symbol_max_quantity[symbol_key]:
                            print(f"DecisionAgent ({self.agent_id}) for {symbol_key}: Risk Alert! Proposed quantity {proposed_quantity} > symbol max {self.per_symbol_max_quantity[symbol_key]}. Adjusting.")
                            proposed_quantity = self.per_symbol_max_quantity[symbol_key]

                    # Apply risk management: Max trade value (needs price)
                    current_price: Optional[float] = None
                    # Try to get current price from one of the signals, e.g., technical signal payload
                    if "technical_signal" in latest_signals_of_each_type and "price_at_signal" in latest_signals_of_each_type["technical_signal"]:
                        current_price = latest_signals_of_each_type["technical_signal"]["price_at_signal"]
                    
                    if current_price is not None and current_price > 0: # Ensure price is valid
                        proposed_trade_value = proposed_quantity * current_price
                        if proposed_trade_value > self.max_trade_value_usd:
                            adjusted_quantity = self.max_trade_value_usd / current_price
                            print(f"DecisionAgent ({self.agent_id}) for {symbol_key}: Risk Alert! Proposed value ${proposed_trade_value:.2f} (qty:{proposed_quantity} @ ${current_price}) > max_trade_value ${self.max_trade_value_usd:.2f}. Adjusting quantity to ~{adjusted_quantity:.8f}.")
                            proposed_quantity = adjusted_quantity
                    elif action != "hold": # only print warning if we were about to trade
                        print(f"DecisionAgent ({self.agent_id}) for {symbol_key}: Could not determine current price for max trade value check. Proceeding with quantity {proposed_quantity}.")
                    
                    # Ensure quantity is not zero or negative after adjustments
                    if proposed_quantity <= 1e-8: # Using a small epsilon for float comparison
                        print(f"DecisionAgent ({self.agent_id}) for {symbol_key}: Proposed quantity is {proposed_quantity:.8f} after risk adjustment. Reverting to HOLD.")
                        action = "hold"
                    
                    if action != "hold":
                        # Construct and return the trading directive
                        trading_directive = {
                            "action": action,
                            "symbol": symbol_key,
                            "quantity": round(proposed_quantity, 8), # Round to a reasonable precision
                            "order_type": "market",
                            "reasoning_score": round(final_decision_score, 3),
                            "contributing_signals": contributing_signals_log
                        }
                        self.update_metrics({
                            "decisions_made_total": self.metrics.get("decisions_made_total", 0) + 1,
                            f"decisions_{symbol_key.replace('/', '_')}": self.metrics.get(f"decisions_{symbol_key.replace('/', '_')}",0) + 1,
                            f"last_action_{symbol_key.replace('/', '_')}": action,
                            f"last_trade_qty_{symbol_key.replace('/', '_')}": round(proposed_quantity, 8)
                        })
                        self.received_signals_buffer[symbol_key] = [] # Clear buffer for this symbol after decision
                        print(f"DecisionAgent ({self.agent_id}) produced directive for {symbol_key}: {trading_directive}")
                        return {"type": "trading_directive", "payload": trading_directive}
                    else: # Action became "hold" due to risk management
                         print(f"DecisionAgent ({self.agent_id}) for {symbol_key}: Action reverted to HOLD due to risk management adjustments.")
                         self.received_signals_buffer[symbol_key] = [] # Clear buffer as a decision (to hold due to risk) was made
                else: # Initial decision was "hold"
                    print(f"DecisionAgent ({self.agent_id}): Score {final_decision_score:.3f} for {symbol_key} not meeting thresholds (Buy >={self.buy_threshold}, Sell <={self.sell_threshold}). Holding.")
                    # If a decision to hold is made after considering enough signal types, clear the buffer.
                    self.received_signals_buffer[symbol_key] = []
            else:
                print(f"DecisionAgent ({self.agent_id}) for {symbol_key}: Not enough unique signal types ({len(latest_signals_of_each_type)}/{self.min_signals_for_decision}) to make decision. Buffered signals: {len(self.received_signals_buffer[symbol_key])}")

        return None # No directive made in this cycle if conditions not met for any symbol from this input signal

# Example Execution Layer Agent
class ExecutionLayerAgent(BaseAgent):
    AGENT_ID_PREFIX = "exec_"

    def __init__(self, agent_id_suffix: str, name: str, agent_type: str, config_details: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=f"{self.AGENT_ID_PREFIX}{agent_id_suffix}",
            name=name,
            agent_role=AgentRole.EXECUTION_BROKER,
            agent_type=agent_type, # e.g., "AlpacaBroker", "BinanceBroker", "InternalPaperBroker"
            config_details=config_details or {}
        )
        self.broker_api_key = self.config_details.get("broker_api_key")
        self.broker_api_secret = self.config_details.get("broker_api_secret")
        self.is_paper_trading = self.config_details.get("paper_trading", True)
        self.open_orders: Dict[str, Dict[str, Any]] = {} # order_id -> order_details
        self.positions: Dict[str, float] = {} # symbol -> quantity

        print(f"ExecutionLayerAgent ({self.agent_id}) initialized. Paper Trading: {self.is_paper_trading}")

    def _place_order(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder to simulate placing an order with a broker."""
        symbol = directive.get("symbol")
        action = directive.get("action") # "buy" or "sell"
        quantity = directive.get("quantity")
        order_type = directive.get("order_type", "market")
        
        print(f"ExecutionLayerAgent ({self.agent_id}): Attempting to place {action} order for {quantity} of {symbol} ({order_type}).")
        
        # Simulate order placement
        order_id = f"sim_ord_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Simulate different outcomes
        import random
        rand_val = random.random()
        
        if rand_val < 0.05: # 5% chance of rejection
            status = "rejected"
            fill_price = None
            filled_quantity = 0
            reason = "Insufficient funds" if random.random() < 0.5 else "Market closed"
            print(f"ExecutionLayerAgent ({self.agent_id}): Order {order_id} for {symbol} REJECTED. Reason: {reason}")
        elif rand_val < 0.15: # 10% chance of partial fill (after rejection chance)
            status = "partially_filled"
            fill_price = directive.get("price_at_signal", 60000 if symbol == "BTC/USD" else 3000) * (1 + (random.random() - 0.5) * 0.001) # Simulate slight price variation
            filled_quantity = quantity * random.uniform(0.3, 0.7)
            print(f"ExecutionLayerAgent ({self.agent_id}): Order {order_id} for {symbol} PARTIALLY FILLED. Filled {filled_quantity:.4f}/{quantity:.4f} @ ${fill_price:.2f}")
        else: # 85% chance of full fill
            status = "filled"
            fill_price = directive.get("price_at_signal", 60000 if symbol == "BTC/USD" else 3000) * (1 + (random.random() - 0.5) * 0.001)
            filled_quantity = quantity
            print(f"ExecutionLayerAgent ({self.agent_id}): Order {order_id} for {symbol} FILLED. Filled {filled_quantity:.4f} @ ${fill_price:.2f}")

        order_details = {
            "order_id": order_id, "symbol": symbol, "action": action,
            "requested_quantity": quantity, "order_type": order_type,
            "status": status, "filled_quantity": filled_quantity,
            "fill_price": fill_price, "timestamp": datetime.now().isoformat(),
            "reason": reason if status == "rejected" else None
        }
        if status != "rejected":
            self.open_orders[order_id] = order_details # Track open/partially_filled orders
            # Update positions (simplified)
            if action == "buy":
                self.positions[symbol] = self.positions.get(symbol, 0) + filled_quantity
            elif action == "sell":
                self.positions[symbol] = self.positions.get(symbol, 0) - filled_quantity
        
        return order_details

    def _check_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Placeholder to simulate checking an existing order's status."""
        print(f"ExecutionLayerAgent ({self.agent_id}): Checking status for order {order_id}.")
        if order_id in self.open_orders:
            # Simulate potential status update (e.g., partial fill becomes full)
            # For simplicity, we'll just return current stored status
            return self.open_orders[order_id]
        return {"order_id": order_id, "status": "unknown", "reason": "Order ID not found in active orders."}

    def _get_position_details(self, symbol: str) -> Dict[str, Any]:
        """Placeholder to get current position for a symbol."""
        return {"symbol": symbol, "quantity": self.positions.get(symbol, 0.0)}

    def process(self, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Processes trading directives from DecisionAgent or other management commands.
        """
        if not data or not isinstance(data, dict) or "type" not in data:
            return None

        directive_type = data.get("type")
        payload = data.get("payload")

        if directive_type == "trading_directive":
            if not payload or not isinstance(payload, dict):
                print(f"ExecutionLayerAgent ({self.agent_id}): Invalid trading_directive payload: {payload}")
                return {"type": "execution_error", "payload": {"error": "Invalid directive payload", "original_directive": payload}}

            print(f"ExecutionLayerAgent ({self.agent_id}) received trading directive: {payload}")
            
            execution_result = self._place_order(payload)
            
            self.update_metrics({
                "orders_processed": self.metrics.get("orders_processed", 0) + 1,
                f"orders_{execution_result.get('status')}": self.metrics.get(f"orders_{execution_result.get('status')}", 0) + 1
            })
            return {"type": "execution_feedback", "payload": execution_result}
        
        elif directive_type == "query_order_status":
            order_id = payload.get("order_id")
            if not order_id:
                return {"type": "execution_error", "payload": {"error": "Missing order_id for query_order_status"}}
            status_result = self._check_order_status(order_id)
            return {"type": "order_status_feedback", "payload": status_result}

        elif directive_type == "query_position":
            symbol = payload.get("symbol")
            if not symbol:
                 return {"type": "execution_error", "payload": {"error": "Missing symbol for query_position"}}
            position_result = self._get_position_details(symbol)
            return {"type": "position_feedback", "payload": position_result}
            
        print(f"ExecutionLayerAgent ({self.agent_id}): Received unknown directive type '{directive_type}'.")
        return None

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    sentiment_agent = SentimentAnalysisAgent(
        agent_id_suffix="alphavantage_news_btc",
        name="AlphaVantage BTC News Sentiment",
        agent_type="AlphaVantageNews",
        symbols=["BTC/USD"],
        config_details={"api_key": "YOUR_AV_KEY", "news_sources": ["google-news"]}
    )

    technical_agent = TechnicalAnalysisAgent(
        agent_id_suffix="rsi_macd_eth",
        name="RSI/MACD ETH Technicals",
        agent_type="RSIMACDStrategy",
        symbols=["ETH/USD"],
        config_details={"rsi_period": 14, "macd_fast": 12, "macd_slow": 26}
    )

    decision_agent = DecisionAgent(
        agent_id_suffix="main_crypto_v1",
        name="Main Crypto Decision Logic V1",
        agent_type="WeightedSignalAggregator",
        config_details={"risk_tolerance": 0.05}
    )
    decision_agent.inputs_from = [sentiment_agent.agent_id, technical_agent.agent_id]


    execution_agent = ExecutionLayerAgent(
        agent_id_suffix="alpaca_paper",
        name="Alpaca Paper Trading Executor",
        agent_type="AlpacaBroker",
        config_details={"api_key": "ALPACA_KEY", "api_secret": "ALPACA_SECRET", "paper_trading": True}
    )
    decision_agent.outputs_to = [execution_agent.agent_id]
    execution_agent.inputs_from = [decision_agent.agent_id]


    print("--- Agent Infos ---")
    print(sentiment_agent.get_info())
    print(technical_agent.get_info())
    print(decision_agent.get_info())
    print(execution_agent.get_info())
    print("\\n--- Starting Agents ---")
    sentiment_agent.start()
    technical_agent.start()
    decision_agent.start()
    execution_agent.start()

    print("\\n--- Simulating Data Flow ---")
    # Simulate data from data sources
    market_data_for_tech = {"symbol": "ETH/USD", "price": 3000, "volume": 10000}
    news_data_for_sentiment = {"symbol": "BTC/USD", "article_headline": "Bitcoin surges on positive news", "content": "..."}

    # Specialized agents process data
    sentiment_signal_output = sentiment_agent.process(news_data_for_sentiment)
    technical_signal_output = technical_agent.process(market_data_for_tech)

    # Decision agent receives signals
    if sentiment_signal_output:
        trading_directive_output = decision_agent.process(sentiment_signal_output)
        if trading_directive_output: # Decision might not be made on every signal
            execution_feedback = execution_agent.process(trading_directive_output)
            if execution_feedback:
                print(f"Execution Feedback: {execution_feedback}")
                # Feedback could go back to decision agent or portfolio manager
                decision_agent.process(execution_feedback) # Example of feedback loop

    if technical_signal_output:
        trading_directive_output = decision_agent.process(technical_signal_output)
        if trading_directive_output: # Decision might not be made on every signal
            execution_feedback = execution_agent.process(trading_directive_output)
            if execution_feedback:
                print(f"Execution Feedback: {execution_feedback}")
                decision_agent.process(execution_feedback)


    print("\\n--- Final Agent Metrics ---")
    print(f"{sentiment_agent.name} Metrics: {sentiment_agent.metrics}")
    print(f"{technical_agent.name} Metrics: {technical_agent.metrics}")
    print(f"{decision_agent.name} Metrics: {decision_agent.metrics}")
    print(f"{execution_agent.name} Metrics: {execution_agent.metrics}")

    print("\\n--- Stopping Agents ---")
    sentiment_agent.stop()
    technical_agent.stop()
    decision_agent.stop()
    execution_agent.stop()