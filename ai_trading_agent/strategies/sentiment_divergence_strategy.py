"""
Sentiment Divergence Strategy.

This module provides a trading strategy that focuses on identifying divergences
between price action and sentiment data.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats

from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.signal_processing.sentiment_processor import SentimentSignalProcessor, SentimentSignal
from ai_trading_agent.signal_processing.signal_aggregator import TradingSignal, SignalDirection
from ai_trading_agent.signal_processing.regime import MarketRegimeDetector, MarketRegime
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType, Position, Portfolio
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class SentimentDivergenceStrategy:
    """
    Trading strategy that focuses on identifying divergences between price action and sentiment.
    
    This strategy looks for situations where price is moving in one direction but sentiment
    is moving in the opposite direction, which can signal potential reversals.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sentiment divergence strategy.
        
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
        self.divergence_window = int(self.config.get("divergence_window", 14))
        self.min_price_trend_strength = float(self.config.get("min_price_trend_strength", 0.5))
        self.min_sentiment_trend_strength = float(self.config.get("min_sentiment_trend_strength", 0.5))
        self.min_divergence_angle = float(self.config.get("min_divergence_angle", 90.0))  # Minimum angle in degrees
        self.min_samples = int(self.config.get("min_samples", 5))
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
        self.days_back = self.config.get("days_back", 30)
        
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
    
    def detect_price_trend(self, price_data: pd.DataFrame) -> Tuple[bool, float, float]:
        """
        Detect if there is a significant trend in price data.
        
        Args:
            price_data: DataFrame containing price data
            
        Returns:
            Tuple of (trend_exists, trend_strength, trend_direction)
            trend_exists: Boolean indicating if a significant trend exists
            trend_strength: Strength of the trend (0.0 to 1.0)
            trend_direction: Direction of the trend (-1.0 to 1.0, negative for downtrend)
        """
        if price_data.empty or len(price_data) < self.min_samples:
            logger.warning(f"Not enough price data for trend analysis: {len(price_data)} < {self.min_samples}")
            return False, 0.0, 0.0
        
        # Get the close prices
        prices = price_data['close'].values
        
        # Limit to the divergence window
        if len(prices) > self.divergence_window:
            prices = prices[-self.divergence_window:]
        
        # Calculate linear regression
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        # Calculate trend strength (r-squared)
        trend_strength = r_value ** 2
        
        # Determine if trend is significant
        trend_exists = (trend_strength >= self.min_price_trend_strength) and (p_value < 0.05)
        
        # Normalize slope to get direction (-1.0 to 1.0)
        # We need to scale based on the price level
        avg_price = np.mean(prices)
        max_possible_slope = avg_price * 0.1 / len(prices)  # Assume 10% change is max
        trend_direction = np.clip(slope / max_possible_slope, -1.0, 1.0)
        
        logger.info(f"Price trend analysis: exists={trend_exists}, strength={trend_strength:.2f}, direction={trend_direction:.2f}")
        
        return trend_exists, trend_strength, trend_direction
    
    def detect_sentiment_trend(self, sentiment_data: pd.DataFrame) -> Tuple[bool, float, float]:
        """
        Detect if there is a significant trend in sentiment data.
        
        Args:
            sentiment_data: DataFrame containing sentiment data
            
        Returns:
            Tuple of (trend_exists, trend_strength, trend_direction)
            trend_exists: Boolean indicating if a significant trend exists
            trend_strength: Strength of the trend (0.0 to 1.0)
            trend_direction: Direction of the trend (-1.0 to 1.0, negative for downtrend)
        """
        if sentiment_data.empty or len(sentiment_data) < self.min_samples:
            logger.warning(f"Not enough sentiment data for trend analysis: {len(sentiment_data)} < {self.min_samples}")
            return False, 0.0, 0.0
        
        # Sort by timestamp
        sentiment_data = sentiment_data.sort_values('timestamp')
        
        # Get the compound scores
        scores = sentiment_data['compound_score'].values
        
        # Limit to the divergence window
        if len(scores) > self.divergence_window:
            scores = scores[-self.divergence_window:]
        
        # Calculate linear regression
        x = np.arange(len(scores))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
        
        # Calculate trend strength (r-squared)
        trend_strength = r_value ** 2
        
        # Determine if trend is significant
        trend_exists = (trend_strength >= self.min_sentiment_trend_strength) and (p_value < 0.05)
        
        # Normalize slope to get direction (-1.0 to 1.0)
        max_possible_slope = 2.0 / len(scores)  # Max change from -1 to 1 over the window
        trend_direction = np.clip(slope / max_possible_slope, -1.0, 1.0)
        
        logger.info(f"Sentiment trend analysis: exists={trend_exists}, strength={trend_strength:.2f}, direction={trend_direction:.2f}")
        
        return trend_exists, trend_strength, trend_direction
    
    def detect_divergence(
        self, 
        price_trend: Tuple[bool, float, float], 
        sentiment_trend: Tuple[bool, float, float]
    ) -> Tuple[bool, float, str]:
        """
        Detect if there is a divergence between price and sentiment trends.
        
        Args:
            price_trend: Tuple of (trend_exists, trend_strength, trend_direction) for price
            sentiment_trend: Tuple of (trend_exists, trend_strength, trend_direction) for sentiment
            
        Returns:
            Tuple of (divergence_exists, divergence_strength, divergence_type)
            divergence_exists: Boolean indicating if a divergence exists
            divergence_strength: Strength of the divergence (0.0 to 1.0)
            divergence_type: Type of divergence ('bullish' or 'bearish')
        """
        price_trend_exists, price_trend_strength, price_trend_direction = price_trend
        sentiment_trend_exists, sentiment_trend_strength, sentiment_trend_direction = sentiment_trend
        
        # Both trends must exist
        if not (price_trend_exists and sentiment_trend_exists):
            return False, 0.0, ""
        
        # Calculate angle between trends (in degrees)
        # Convert directions to angles (0 to 180 degrees)
        price_angle = np.arctan2(price_trend_direction, 1.0) * 180 / np.pi
        sentiment_angle = np.arctan2(sentiment_trend_direction, 1.0) * 180 / np.pi
        
        # Calculate absolute angle difference
        angle_diff = abs(price_angle - sentiment_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Check if angle is large enough for divergence
        divergence_exists = angle_diff >= self.min_divergence_angle
        
        # Calculate divergence strength based on angle and trend strengths
        if divergence_exists:
            # Normalize angle to 0-1 range (90-180 degrees -> 0-1)
            angle_factor = (angle_diff - 90) / 90 if angle_diff >= 90 else 0
            # Combine with trend strengths
            divergence_strength = angle_factor * (price_trend_strength + sentiment_trend_strength) / 2
        else:
            divergence_strength = 0.0
        
        # Determine divergence type
        if divergence_exists:
            # Bullish divergence: price down, sentiment up
            if price_trend_direction < 0 and sentiment_trend_direction > 0:
                divergence_type = "bullish"
            # Bearish divergence: price up, sentiment down
            elif price_trend_direction > 0 and sentiment_trend_direction < 0:
                divergence_type = "bearish"
            else:
                # This shouldn't happen with our angle check, but just in case
                divergence_exists = False
                divergence_strength = 0.0
                divergence_type = ""
        else:
            divergence_type = ""
        
        logger.info(f"Divergence analysis: exists={divergence_exists}, strength={divergence_strength:.2f}, type={divergence_type}")
        
        return divergence_exists, divergence_strength, divergence_type
    
    def generate_divergence_signal(
        self, 
        asset: str, 
        divergence_exists: bool, 
        divergence_strength: float, 
        divergence_type: str
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal based on sentiment-price divergence.
        
        Args:
            asset: Asset symbol
            divergence_exists: Whether a divergence exists
            divergence_strength: Strength of the divergence (0.0 to 1.0)
            divergence_type: Type of divergence ('bullish' or 'bearish')
            
        Returns:
            TradingSignal or None if no signal
        """
        if not divergence_exists:
            logger.info(f"No significant divergence for {asset}")
            return None
        
        # Determine signal direction
        if divergence_type == "bullish":
            direction = SignalDirection.BUY
        elif divergence_type == "bearish":
            direction = SignalDirection.SELL
        else:
            return None
        
        # Create signal
        signal = TradingSignal(
            symbol=asset,
            signal_type="sentiment",
            direction=direction,
            strength=divergence_strength,
            confidence=divergence_strength,  # Use divergence strength as confidence
            timeframe=self.timeframe,
            source="SentimentDivergence",
            timestamp=datetime.now(),
            metadata={
                'divergence_type': divergence_type,
                'divergence_strength': divergence_strength
            }
        )
        
        logger.info(f"Generated sentiment divergence signal for {asset}: {signal}")
        
        return signal
    
    def run_strategy(
        self, 
        portfolio_manager: PortfolioManager,
        market_prices: Dict[str, Decimal]
    ) -> List[Order]:
        """
        Run the sentiment divergence strategy to generate orders.
        
        Args:
            portfolio_manager: Portfolio manager instance
            market_prices: Dictionary of current market prices
            
        Returns:
            List of orders to execute
        """
        orders = []
        
        for asset in self.assets:
            # Get sentiment data
            sentiment_data = self.get_sentiment_data(asset, is_topic=False)
            
            if sentiment_data.empty:
                logger.warning(f"No sentiment data available for {asset}")
                continue
            
            # Get price data
            price_data = self.get_price_data(asset)
            
            # Detect price trend
            price_trend = self.detect_price_trend(price_data)
            
            # Detect sentiment trend
            sentiment_trend = self.detect_sentiment_trend(sentiment_data)
            
            # Detect divergence
            divergence_exists, divergence_strength, divergence_type = self.detect_divergence(
                price_trend, sentiment_trend
            )
            
            # Generate signal
            signal = self.generate_divergence_signal(
                asset, divergence_exists, divergence_strength, divergence_type
            )
            
            if not signal:
                continue
            
            # Get current market price
            price = market_prices.get(asset)
            if not price:
                logger.warning(f"No price available for {asset}, skipping")
                continue
            
            # Determine order side
            side = OrderSide.BUY if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else OrderSide.SELL
            
            # Calculate stop loss price
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
            
            # Scale position size by signal strength and confidence
            strength_factor = Decimal(str(signal.strength))
            confidence_factor = Decimal(str(signal.confidence))
            
            # Combined scaling factor
            scaling_factor = strength_factor * confidence_factor
            position_size = position_size * scaling_factor
            
            # Ensure position size doesn't exceed maximum
            position_size = min(position_size, self.max_position_size * portfolio_manager.get_portfolio_value())
            
            # Convert Decimal to float for Order parameters
            position_size_float = float(position_size)
            price_float = float(price)
            stop_loss_float = float(stop_loss)
            take_profit_float = float(price * (Decimal("1") + self.take_profit_pct)) if side == OrderSide.BUY else float(price * (Decimal("1") - self.take_profit_pct))
            
            # Create order
            order = Order(
                symbol=asset,
                side=side,
                order_type=OrderType.MARKET,
                quantity=position_size_float,
                price=price_float,
                stop_price=stop_loss_float
            )
            
            # Add order to list
            orders.append(order)
            
            # Log order details
            logger.info(f"Generated order for {asset}: {side.name} {position_size_float} @ {price_float} with stop @ {stop_loss_float}")
            
            # Record signal in history for performance tracking
            self.signal_history.append({
                "timestamp": datetime.now(),
                "symbol": asset,
                "signal": signal.to_dict(),
                "order": order
            })
        
        return orders
