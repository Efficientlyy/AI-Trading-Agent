"""
Sentiment Shock Strategy.

This module provides a trading strategy that focuses on identifying and trading
sudden changes in sentiment data.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.signal_processing.sentiment_processor import SentimentSignalProcessor, SentimentSignal
from ai_trading_agent.signal_processing.signal_aggregator import TradingSignal, SignalDirection
from ai_trading_agent.signal_processing.regime import MarketRegimeDetector, MarketRegime
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType, Position, Portfolio
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class SentimentShockStrategy:
    """
    Trading strategy that focuses on identifying and trading sudden changes in sentiment.
    
    This strategy looks for significant jumps or drops in sentiment that may precede
    price movements, especially in news-driven markets.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sentiment shock strategy.
        
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
        self.shock_window = int(self.config.get("shock_window", 5))
        self.shock_threshold = float(self.config.get("shock_threshold", 0.4))
        self.volume_threshold = float(self.config.get("volume_threshold", 1.5))  # Minimum volume multiplier
        self.min_samples = int(self.config.get("min_samples", 10))
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
    
    def detect_sentiment_shock(self, sentiment_data: pd.DataFrame) -> Tuple[bool, float, float, datetime]:
        """
        Detect if there is a sudden shock in sentiment data.
        
        Args:
            sentiment_data: DataFrame containing sentiment data
            
        Returns:
            Tuple of (shock_exists, shock_magnitude, shock_direction, shock_time)
            shock_exists: Boolean indicating if a shock exists
            shock_magnitude: Magnitude of the shock (0.0 to 1.0)
            shock_direction: Direction of the shock (-1.0 to 1.0, negative for negative shock)
            shock_time: Timestamp of the shock
        """
        if sentiment_data.empty or len(sentiment_data) < self.min_samples:
            logger.warning(f"Not enough sentiment data for shock analysis: {len(sentiment_data)} < {self.min_samples}")
            return False, 0.0, 0.0, datetime.now()
        
        # Sort by timestamp
        sentiment_data = sentiment_data.sort_values('timestamp')
        
        # Get the compound scores and timestamps
        scores = sentiment_data['compound_score'].values
        timestamps = pd.to_datetime(sentiment_data['timestamp']).values
        
        # Calculate rolling mean and standard deviation
        window = min(self.shock_window, len(scores) - 1)
        rolling_mean = np.convolve(scores, np.ones(window)/window, mode='valid')
        
        # Calculate changes from previous average
        changes = np.zeros_like(scores)
        for i in range(window, len(scores)):
            # Change from previous window average to current score
            changes[i] = scores[i] - rolling_mean[i-window]
        
        # Calculate rolling standard deviation for normalization
        rolling_std = np.zeros_like(rolling_mean)
        for i in range(len(rolling_mean)):
            start_idx = max(0, i - window + 1)
            rolling_std[i] = np.std(scores[start_idx:i+1])
        
        # Avoid division by zero
        rolling_std = np.maximum(rolling_std, 0.01)
        
        # Normalize changes by standard deviation to get z-scores
        z_scores = np.zeros_like(changes)
        for i in range(window, len(scores)):
            z_scores[i] = changes[i] / rolling_std[i-window]
        
        # Check for volume spikes if volume data is available
        volume_spike = False
        if 'volume' in sentiment_data.columns:
            volumes = sentiment_data['volume'].values
            rolling_vol_mean = np.convolve(volumes, np.ones(window)/window, mode='valid')
            
            vol_ratios = np.zeros_like(volumes)
            for i in range(window, len(volumes)):
                # Ratio of current volume to previous window average
                if rolling_vol_mean[i-window] > 0:
                    vol_ratios[i] = volumes[i] / rolling_vol_mean[i-window]
                else:
                    vol_ratios[i] = 1.0
            
            # Check if any recent volume spike
            volume_spike = np.any(vol_ratios[-window:] >= self.volume_threshold)
        
        # Find the largest shock in the most recent window
        recent_z_scores = z_scores[-window:]
        if len(recent_z_scores) == 0:
            return False, 0.0, 0.0, datetime.now()
        
        # Find index of max absolute z-score
        max_idx = np.argmax(np.abs(recent_z_scores))
        max_z_score = recent_z_scores[max_idx]
        
        # Get the actual index in the original array
        original_idx = len(z_scores) - window + max_idx
        
        # Check if shock is significant
        shock_exists = (abs(max_z_score) >= self.shock_threshold) and (volume_spike or 'volume' not in sentiment_data.columns)
        
        # Calculate shock magnitude (0-1 range)
        shock_magnitude = min(1.0, abs(max_z_score) / 3.0)  # 3 sigma is considered very significant
        
        # Determine shock direction
        shock_direction = np.sign(max_z_score)
        
        # Get shock timestamp
        shock_time = pd.to_datetime(timestamps[original_idx])
        
        logger.info(f"Sentiment shock analysis: exists={shock_exists}, magnitude={shock_magnitude:.2f}, direction={shock_direction:.2f}, time={shock_time}")
        
        return shock_exists, shock_magnitude, shock_direction, shock_time
    
    def generate_shock_signal(
        self, 
        asset: str, 
        shock_exists: bool, 
        shock_magnitude: float, 
        shock_direction: float,
        shock_time: datetime
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal based on sentiment shock.
        
        Args:
            asset: Asset symbol
            shock_exists: Whether a shock exists
            shock_magnitude: Magnitude of the shock (0.0 to 1.0)
            shock_direction: Direction of the shock (-1.0 to 1.0)
            shock_time: Timestamp of the shock
            
        Returns:
            TradingSignal or None if no signal
        """
        if not shock_exists:
            logger.info(f"No significant sentiment shock for {asset}")
            return None
        
        # Check if shock is recent enough (within 24 hours)
        if (datetime.now() - shock_time).total_seconds() > 86400:
            logger.info(f"Sentiment shock for {asset} is too old: {shock_time}")
            return None
        
        # Determine signal direction
        if shock_direction > 0:
            direction = SignalDirection.STRONG_BUY if shock_magnitude > 0.7 else SignalDirection.BUY
        else:
            direction = SignalDirection.STRONG_SELL if shock_magnitude > 0.7 else SignalDirection.SELL
        
        # Create signal
        signal = TradingSignal(
            symbol=asset,
            signal_type="sentiment",
            direction=direction,
            strength=shock_magnitude,
            confidence=shock_magnitude * 0.8,  # Slightly lower confidence for shock events
            timeframe=self.timeframe,
            source="SentimentShock",
            timestamp=datetime.now(),
            metadata={
                'shock_direction': shock_direction,
                'shock_magnitude': shock_magnitude,
                'shock_time': shock_time.isoformat()
            }
        )
        
        logger.info(f"Generated sentiment shock signal for {asset}: {signal}")
        
        return signal
    
    def run_strategy(
        self, 
        portfolio_manager: PortfolioManager,
        market_prices: Dict[str, Decimal]
    ) -> List[Order]:
        """
        Run the sentiment shock strategy to generate orders.
        
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
            
            # Detect sentiment shock
            shock_exists, shock_magnitude, shock_direction, shock_time = self.detect_sentiment_shock(sentiment_data)
            
            # Generate signal
            signal = self.generate_shock_signal(
                asset, shock_exists, shock_magnitude, shock_direction, shock_time
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
