"""
Sentiment-based trading strategy.

This module provides a trading strategy that uses sentiment analysis to generate trading signals.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.common.logging_config import logger
# Import RichSignal and RichSignalsDict from the same module as BaseStrategy
from ai_trading_agent.agent.strategy import BaseStrategy, RichSignal, RichSignalsDict
# Import required classes for order generation
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)

class SentimentStrategy(BaseStrategy):
    """
    Trading strategy based on sentiment analysis.
    
    This strategy uses sentiment analysis to generate trading signals and execute trades.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sentiment strategy.
        
        Args:
            name: Unique name for the strategy instance.
            config: Configuration dictionary for the strategy
        """
        super().__init__(name, config)
        
        # Initialize sentiment analyzer
        sentiment_analyzer_config = self.config.get("sentiment_analyzer", {})
        if not sentiment_analyzer_config: 
             logger.warning(f"No configuration provided for SentimentAnalyzer in strategy '{self.name}'. Using defaults.")
        self.sentiment_analyzer = SentimentAnalyzer(config=sentiment_analyzer_config)
        
        # Strategy parameters
        self.sentiment_threshold = Decimal(str(self.config.get("sentiment_threshold", 0.2)))
        self.position_sizing_method = self.config.get("position_sizing_method", "risk_based")
        self.risk_per_trade = Decimal(str(self.config.get("risk_per_trade", 0.02)))
        self.max_position_size = Decimal(str(self.config.get("max_position_size", 0.1)))
        self.stop_loss_pct = Decimal(str(self.config.get("stop_loss_pct", 0.05)))
        self.take_profit_pct = Decimal(str(self.config.get("take_profit_pct", 0.1)))
        
        # Topics and assets to analyze
        self.topics = self.config.get("topics", ["blockchain", "cryptocurrency"])
        self.assets = self.config.get("assets", ["BTC", "ETH"])
        
        # Days to look back for sentiment data
        self.days_back = self.config.get("days_back", 7)
        
        # Cache for sentiment data
        self.sentiment_cache = {}
        self.sentiment_cache_expiry = {}
        self.cache_expiry_hours = self.config.get("cache_expiry_hours", 24)
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the strategy's configuration parameters dynamically.
        
        Args:
            config_updates: A dictionary containing parameters to update.
        """
        logger.info(f"Updating {self.name} config. Old: {self.config}, New partial: {config_updates}")
        self.config.update(config_updates)
        # TODO: Re-initialize components if necessary based on updated config
        # e.g., self.sentiment_threshold = Decimal(str(self.config.get(...)))
        logger.info(f"Updated {self.name} config: {self.config}")

    def get_sentiment_data(self, topic_or_asset: str, is_topic: bool = True) -> pd.DataFrame:
        """
        Get sentiment data for a topic or asset.
        
        Args:
            topic_or_asset: Topic or asset to get sentiment data for
            is_topic: Whether the input is a topic (True) or an asset (False)
            
        Returns:
            DataFrame containing sentiment data
        """
        # Check if we have cached data that's still valid
        cache_key = f"{'topic' if is_topic else 'asset'}:{topic_or_asset}"
        current_time = datetime.now()
        
        if (cache_key in self.sentiment_cache and 
            cache_key in self.sentiment_cache_expiry and 
            current_time < self.sentiment_cache_expiry[cache_key]):
            logger.info(f"Using cached sentiment data for {cache_key}")
            return self.sentiment_cache[cache_key]
        
        # Fetch new sentiment data
        logger.info(f"Fetching sentiment data for {cache_key}")
        if is_topic:
            df = self.sentiment_analyzer.analyze_sentiment(topic=topic_or_asset, days_back=self.days_back)
        else:
            df = self.sentiment_analyzer.analyze_sentiment(crypto_ticker=topic_or_asset, days_back=self.days_back)
        
        # Cache the data
        if not df.empty:
            self.sentiment_cache[cache_key] = df
            self.sentiment_cache_expiry[cache_key] = current_time + timedelta(hours=self.cache_expiry_hours)
        
        return df
    
    def get_latest_signal(self, topic_or_asset: str, is_topic: bool = True) -> Dict[str, Any]:
        """
        Get the latest trading signal for a topic or asset.
        
        Args:
            topic_or_asset: Topic or asset to get signal for
            is_topic: Whether the input is a topic (True) or an asset (False)
            
        Returns:
            Dictionary containing signal information
        """
        # Get sentiment data
        df = self.get_sentiment_data(topic_or_asset, is_topic)
        
        if df.empty:
            logger.warning(f"No sentiment data available for {topic_or_asset}")
            return {
                "topic_or_asset": topic_or_asset,
                "is_topic": is_topic,
                "signal": 0,
                "weighted_score": Decimal("0"),
                "timestamp": datetime.now()
            }
        
        # Get the latest sentiment signal
        latest_signal = df.iloc[-1]
        signal_value = latest_signal["signal"]
        weighted_score = latest_signal["weighted_sentiment_score"]
        
        # Get timestamp if available
        timestamp = latest_signal.get("time_published", datetime.now())
        
        logger.info(f"Latest signal for {topic_or_asset}: {signal_value}, Weighted Score: {weighted_score}")
        
        return {
            "topic_or_asset": topic_or_asset,
            "is_topic": is_topic,
            "signal": signal_value,
            "weighted_score": weighted_score,
            "timestamp": timestamp
        }
    
    def generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], # DataManager provided data (currently unused by this logic)
        current_portfolio: Optional[Dict[str, Any]] = None # Optional portfolio state (currently unused)
    ) -> RichSignalsDict:
        """
        Generate rich trading signals based on sentiment analysis for configured assets.
        
        Implements the abstract method from BaseStrategy.

        Args:
            data: Market data provided by the DataManager (ignored in this simple version).
            current_portfolio: Current portfolio state (ignored in this simple version).

        Returns:
            RichSignalsDict mapping each asset symbol to its sentiment-based signal.
        """
        rich_signals: RichSignalsDict = {}
        
        # Use self.assets defined during __init__ from config
        for asset in self.assets:
            try:
                # Get the latest sentiment signal for the asset
                # Note: This fetches data on-demand, ignoring the 'data' argument from DataManager
                latest_signal_info = self.get_latest_signal(asset, is_topic=False)
                
                # Convert the result from get_latest_signal to RichSignal format
                signal_strength = float(latest_signal_info.get('weighted_score', 0.0)) # Map weighted score
                # Confidence could be derived, e.g., based on magnitude or set to a fixed value/related to score variance?
                # For simplicity, let's use the absolute strength scaled (capped at 1.0)
                confidence_score = min(abs(signal_strength) * 2.0, 1.0) # Example confidence derivation

                rich_signals[asset] = {
                    'signal_strength': signal_strength,
                    'confidence_score': confidence_score,
                    'signal_type': 'sentiment',
                    'metadata': {
                        'timestamp': latest_signal_info.get('timestamp'),
                        'raw_signal': latest_signal_info.get('signal') # Keep the original 0/1/-1 if needed
                    }
                }
                logger.debug(f"Generated sentiment signal for {asset}: Strength={signal_strength:.4f}, Conf={confidence_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error generating sentiment signal for asset {asset}: {e}", exc_info=True)
                rich_signals[asset] = {
                    'signal_strength': 0.0,
                    'confidence_score': 0.0,
                    'signal_type': 'sentiment',
                    'metadata': {'error': str(e)}
                }
                
        return rich_signals
    
    def generate_orders(self, 
                       signals: List[Dict[str, Any]], 
                       portfolio_manager: PortfolioManager,
                       market_prices: Dict[str, Decimal]) -> List[Order]:
        """
        Generate orders based on sentiment signals.
        
        Args:
            signals: List of signal dictionaries
            portfolio_manager: Portfolio manager instance
            market_prices: Dictionary of current market prices
            
        Returns:
            List of orders to execute
        """
        orders = []
        
        for signal_data in signals:
            # Skip topic signals for order generation
            if signal_data.get("is_topic", True):
                continue
            
            asset = signal_data.get("topic_or_asset")
            signal = signal_data.get("signal", 0)
            
            # Skip if no clear signal
            if abs(signal) < 0.5:
                logger.info(f"No clear signal for {asset}, skipping")
                continue
            
            # Get current market price
            price = market_prices.get(asset)
            if not price:
                logger.warning(f"No price available for {asset}, skipping")
                continue
            
            # Determine order side
            side = OrderSide.BUY if signal > 0 else OrderSide.SELL
            
            # Calculate position size based on risk
            stop_loss = None
            if side == OrderSide.BUY:
                stop_loss = price * (Decimal("1") - self.stop_loss_pct)
            else:
                stop_loss = price * (Decimal("1") + self.stop_loss_pct)
            
            # Calculate position size
            position_size = portfolio_manager.calculate_position_size(
                symbol=asset,
                price=price,
                stop_loss=stop_loss
            )
            
            # Scale position size by signal strength (0.5 to 1.0)
            position_size = position_size * Decimal(str(abs(signal)))
            
            # Convert Decimal to float for Order parameters
            position_size_float = float(position_size)
            price_float = float(price)
            
            # Create order
            order = Order(
                symbol=asset,
                side=side,
                order_type=OrderType.MARKET,  # Use order_type instead of type
                quantity=position_size_float,
                price=price_float
                # created_at will be set automatically with default_factory
            )
            
            orders.append(order)
            logger.info(f"Generated order: {order}")
        
        return orders
    
    def run_strategy(self, 
                    portfolio_manager: PortfolioManager,
                    market_prices: Dict[str, Decimal]) -> List[Order]:
        """
        Run the sentiment strategy to generate orders.
        
        Args:
            portfolio_manager: Portfolio manager instance
            market_prices: Dictionary of current market prices
            
        Returns:
            List of orders to execute
        """
        # Generate signals
        signals = self.generate_signals(None, None)
        
        # Generate orders based on signals
        orders = self.generate_orders(signals, portfolio_manager, market_prices)
        
        return orders
