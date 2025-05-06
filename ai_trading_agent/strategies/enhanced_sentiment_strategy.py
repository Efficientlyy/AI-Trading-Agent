"""
Enhanced Sentiment-based Trading Strategy.

This module provides an advanced trading strategy that uses sentiment analysis with
timeframe awareness and market regime detection to generate trading signals.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.signal_processing.sentiment_processor import SentimentSignalProcessor, SentimentSignal
from ai_trading_agent.signal_processing.regime import MarketRegimeDetector, MarketRegime
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType, Position, Portfolio
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class EnhancedSentimentStrategy:
    """
    Enhanced trading strategy based on sentiment analysis with timeframe awareness and market regime detection.
    
    This strategy uses the SentimentSignalProcessor and MarketRegimeDetector to generate
    more sophisticated trading signals that adapt to different market conditions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced sentiment strategy.
        
        Args:
            config: Configuration dictionary for the strategy
        """
        self.config = config or {}
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer(config=self.config.get("sentiment_analyzer", {}))
        
        # Initialize sentiment signal processor
        self.signal_processor = SentimentSignalProcessor(
            threshold=float(self.config.get("sentiment_threshold", 0.2)),
            window_size=int(self.config.get("window_size", 3)),
            sentiment_weight=float(self.config.get("sentiment_weight", 0.4)),
            min_confidence=float(self.config.get("min_confidence", 0.6)),
            enable_regime_detection=bool(self.config.get("enable_regime_detection", True))
        )
        
        # Initialize market regime detector
        self.regime_detector = MarketRegimeDetector(
            volatility_window=int(self.config.get("volatility_window", 20)),
            trend_window=int(self.config.get("trend_window", 50)),
            volatility_threshold=float(self.config.get("volatility_threshold", 0.015)),
            trend_threshold=float(self.config.get("trend_threshold", 0.6)),
            range_threshold=float(self.config.get("range_threshold", 0.3))
        )
        
        # Strategy parameters
        self.position_sizing_method = self.config.get("position_sizing_method", "risk_based")
        self.risk_per_trade = Decimal(str(self.config.get("risk_per_trade", 0.02)))
        self.max_position_size = Decimal(str(self.config.get("max_position_size", 0.1)))
        self.stop_loss_pct = Decimal(str(self.config.get("stop_loss_pct", 0.05)))
        self.take_profit_pct = Decimal(str(self.config.get("take_profit_pct", 0.1)))
        
        # Topics and assets to analyze
        self.topics = self.config.get("topics", ["blockchain", "cryptocurrency"])
        self.assets = self.config.get("assets", ["BTC", "ETH"])
        
        # Timeframe for analysis
        self.timeframe = self.config.get("timeframe", "1d")
        
        # Days to look back for sentiment data
        self.days_back = self.config.get("days_back", 7)
        
        # Cache for sentiment data and price data
        self.sentiment_cache = {}
        self.price_cache = {}
        self.sentiment_cache_expiry = {}
        self.cache_expiry_hours = self.config.get("cache_expiry_hours", 24)
        
        # Signal history for tracking performance
        self.signal_history = []
    
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
    
    def get_price_data(self, asset: str) -> pd.DataFrame:
        """
        Get historical price data for an asset.
        
        Args:
            asset: Asset symbol to get price data for
            
        Returns:
            DataFrame containing price data
        """
        # This is a placeholder for actual price data fetching
        # In a real implementation, this would fetch data from an exchange or data provider
        
        # For now, generate mock price data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Generate random price data with a trend
        base_price = 100.0 if asset != "BTC" else 30000.0
        prices = [base_price]
        for i in range(1, 100):
            # Add some trend and randomness
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            prices.append(prices[-1] * (1 + change))
        
        # Create OHLCV DataFrame
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(1000, 10000) for _ in prices]
        }, index=dates)
        
        return df
    
    def detect_market_regime(self, asset: str) -> MarketRegime:
        """
        Detect the current market regime for an asset.
        
        Args:
            asset: Asset symbol to detect regime for
            
        Returns:
            MarketRegime enum value
        """
        # Get price data
        price_data = self.get_price_data(asset)
        
        # Detect regime
        regime = self.regime_detector.detect_regime(price_data)
        logger.info(f"Detected {regime.value} regime for {asset}")
        
        return regime
    
    def process_sentiment_signals(self, asset: str) -> List[SentimentSignal]:
        """
        Process sentiment data into trading signals with timeframe awareness.
        
        Args:
            asset: Asset symbol to process signals for
            
        Returns:
            List of SentimentSignal objects
        """
        # Get sentiment data
        sentiment_df = self.get_sentiment_data(asset, is_topic=False)
        
        if sentiment_df.empty:
            logger.warning(f"No sentiment data available for {asset}")
            return []
        
        # Convert sentiment data to Series for processing
        sentiment_series = pd.Series(
            sentiment_df['compound_score'].values,
            index=pd.DatetimeIndex(sentiment_df['timestamp'])
        )
        
        # Get price data for correlation analysis
        price_data = self.get_price_data(asset)
        
        # Process sentiment data into signals
        signals = self.signal_processor.process_sentiment_data(
            symbol=asset,
            historical_sentiment=sentiment_series,
            timeframe=self.timeframe,
            price_data=price_data
        )
        
        return signals
    
    def generate_orders_from_signals(
        self, 
        signals: List[SentimentSignal], 
        portfolio_manager: PortfolioManager,
        market_prices: Dict[str, Decimal]
    ) -> List[Order]:
        """
        Generate orders based on sentiment signals with market regime awareness.
        
        Args:
            signals: List of SentimentSignal objects
            portfolio_manager: Portfolio manager instance
            market_prices: Dictionary of current market prices
            
        Returns:
            List of orders to execute
        """
        orders = []
        
        for signal in signals:
            asset = signal.symbol
            
            # Skip if no signal type
            if not signal.signal_type:
                logger.info(f"No signal type for {asset}, skipping")
                continue
            
            # Get current market price
            price = market_prices.get(asset)
            if not price:
                logger.warning(f"No price available for {asset}, skipping")
                continue
            
            # Detect market regime
            regime = self.detect_market_regime(asset)
            
            # Get regime-specific parameters
            regime_params = self.regime_detector.get_regime_parameters(regime)
            
            # Determine order side
            side = OrderSide.BUY if signal.signal_type == 'buy' else OrderSide.SELL
            
            # Calculate stop loss and take profit based on regime
            stop_loss_pct = Decimal(str(regime_params['stop_loss_pct']))
            take_profit_pct = Decimal(str(regime_params['take_profit_pct']))
            position_size_multiplier = Decimal(str(regime_params['position_size_pct']))
            
            # Calculate stop loss price
            stop_loss = None
            if side == OrderSide.BUY:
                stop_loss = price * (Decimal("1") - stop_loss_pct)
            else:
                stop_loss = price * (Decimal("1") + stop_loss_pct)
            
            # Calculate position size
            position_size = portfolio_manager.calculate_position_size(
                symbol=asset,
                price=price,
                stop_loss=stop_loss
            )
            
            # Scale position size by signal strength and confidence
            strength_factor = Decimal(str(signal.strength))
            confidence_factor = Decimal(str(signal.confidence))
            regime_factor = position_size_multiplier
            
            # Combined scaling factor (strength * confidence * regime)
            scaling_factor = strength_factor * confidence_factor * regime_factor
            position_size = position_size * scaling_factor
            
            # Ensure position size doesn't exceed maximum
            position_size = min(position_size, self.max_position_size * portfolio_manager.get_portfolio_value())
            
            # Convert Decimal to float for Order parameters
            position_size_float = float(position_size)
            price_float = float(price)
            stop_loss_float = float(stop_loss)
            take_profit_float = float(price * (Decimal("1") + take_profit_pct)) if side == OrderSide.BUY else float(price * (Decimal("1") - take_profit_pct))
            
            # Skip creating orders with zero or negative quantities
            if position_size_float <= 0:
                logger.warning(f"Skipping order for {asset}: calculated position size {position_size_float} is not positive")
                continue
            
            # Create order
            order = Order(
                symbol=asset,
                side=side,
                order_type=OrderType.MARKET,
                quantity=position_size_float,
                price=price_float,
                stop_price=stop_loss_float  # Add stop loss
                # created_at will be set automatically with default_factory
            )
            
            # Add order to list
            orders.append(order)
            
            # Log order details
            logger.info(f"Generated order for {asset}: {side.name} {position_size_float} @ {price_float} with stop @ {stop_loss_float}")
            
            # Record signal in history for performance tracking
            self.signal_history.append({
                "timestamp": datetime.now(),
                "symbol": asset,
                "signal_type": signal.signal_type,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "regime": regime.value,
                "order": order
            })
        
        return orders
    
    def run_strategy(
        self, 
        portfolio_manager: PortfolioManager,
        market_prices: Dict[str, Decimal]
    ) -> List[Order]:
        """
        Run the enhanced sentiment strategy to generate orders.
        
        Args:
            portfolio_manager: Portfolio manager instance
            market_prices: Dictionary of current market prices
            
        Returns:
            List of orders to execute
        """
        all_signals = []
        
        # Process signals for each asset
        for asset in self.assets:
            signals = self.process_sentiment_signals(asset)
            all_signals.extend(signals)
        
        # Generate orders based on signals
        orders = self.generate_orders_from_signals(all_signals, portfolio_manager, market_prices)
        
        return orders
    
    def evaluate_signal_performance(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Evaluate the performance of past signals.
        
        Args:
            days_back: Number of days to look back for signal evaluation
            
        Returns:
            Dictionary containing performance metrics
        """
        # Filter signals by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_signals = [s for s in self.signal_history if s["timestamp"] >= cutoff_date]
        
        if not recent_signals:
            return {"error": "No signals in the specified time period"}
        
        # Calculate performance metrics
        total_signals = len(recent_signals)
        buy_signals = len([s for s in recent_signals if s["signal_type"] == "buy"])
        sell_signals = len([s for s in recent_signals if s["signal_type"] == "sell"])
        
        # Group by regime
        regime_counts = {}
        for signal in recent_signals:
            regime = signal["regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Calculate average strength and confidence
        avg_strength = sum(s["strength"] for s in recent_signals) / total_signals
        avg_confidence = sum(s["confidence"] for s in recent_signals) / total_signals
        
        # Return performance metrics
        return {
            "total_signals": total_signals,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "regime_distribution": regime_counts,
            "avg_strength": avg_strength,
            "avg_confidence": avg_confidence
        }
