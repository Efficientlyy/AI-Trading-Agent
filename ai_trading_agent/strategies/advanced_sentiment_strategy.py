"""
Advanced sentiment-based trading strategy.

This module provides a sophisticated trading strategy that uses sentiment analysis
combined with technical indicators and risk management to generate trading signals.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType, OrderStatus
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
from ai_trading_agent.features.lag_features import create_lag_features, create_diff_features, create_pct_change_features

logger = logging.getLogger(__name__)

class AdvancedSentimentStrategy:
    """
    Advanced trading strategy based on sentiment analysis.
    
    This strategy uses sentiment analysis combined with technical indicators
    and risk management to generate trading signals and execute trades.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced sentiment strategy.
        
        Args:
            config: Configuration dictionary for the strategy
        """
        self.config = config or {}
        
        # Initialize sentiment analyzer with advanced configuration
        sentiment_config = self.config.get("sentiment_analyzer", {})
        
        # Extract Alpha Vantage API tier if provided
        alpha_vantage_tier = self.config.get("alpha_vantage_tier", "free")
        
        # Update sentiment analyzer config with tier information
        if "alpha_vantage_client" not in sentiment_config:
            sentiment_config["alpha_vantage_client"] = {}
        
        sentiment_config["alpha_vantage_client"]["tier"] = alpha_vantage_tier
        
        self.sentiment_analyzer = SentimentAnalyzer(config=sentiment_config)
        
        # Strategy parameters
        self.sentiment_threshold = Decimal(str(self.config.get("sentiment_threshold", 0.2)))
        self.position_sizing_method = self.config.get("position_sizing_method", "risk_based")
        self.risk_per_trade = Decimal(str(self.config.get("risk_per_trade", 0.02)))
        self.max_position_size = Decimal(str(self.config.get("max_position_size", 0.1)))
        self.stop_loss_pct = Decimal(str(self.config.get("stop_loss_pct", 0.05)))
        self.take_profit_pct = Decimal(str(self.config.get("take_profit_pct", 0.1)))
        
        # Topics and assets to analyze
        self.topics = self.config.get("topics", ["blockchain", "cryptocurrency", "defi"])
        self.assets = self.config.get("assets", {
            "blockchain": ["BTC", "ETH"],
            "cryptocurrency": ["BTC", "ETH", "XRP", "ADA", "SOL"],
            "defi": ["ETH", "UNI", "AAVE", "COMP", "MKR"]
        })
        
        # Topic to asset mapping
        self.topic_asset_weights = self.config.get("topic_asset_weights", {
            "blockchain": {"BTC": 0.7, "ETH": 0.3},
            "cryptocurrency": {"BTC": 0.4, "ETH": 0.3, "XRP": 0.1, "ADA": 0.1, "SOL": 0.1},
            "defi": {"ETH": 0.3, "UNI": 0.2, "AAVE": 0.2, "COMP": 0.15, "MKR": 0.15}
        })
        
        # Signal combination parameters
        self.signal_weights = self.config.get("signal_weights", {
            "sentiment": 0.6,
            "trend": 0.3,
            "volatility": 0.1
        })
        
        # Trend detection parameters
        self.trend_lookback = self.config.get("trend_lookback", 5)
        self.trend_threshold = Decimal(str(self.config.get("trend_threshold", 0.1)))
        
        # Position management
        self.max_positions = self.config.get("max_positions", 5)
        self.correlation_threshold = Decimal(str(self.config.get("correlation_threshold", 0.7)))
        
        # Cached sentiment data
        self.sentiment_cache = {}
        self.sentiment_cache_expiry = {}
        self.cache_ttl = timedelta(hours=self.config.get("cache_ttl_hours", 4))
        
        # Performance tracking
        self.performance_history = []
    
    def analyze_sentiment(self, topic: str) -> Dict[str, Any]:
        """
        Analyze sentiment for a topic with caching.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            Dictionary containing sentiment information
        """
        # Check if we have cached sentiment data that's still valid
        current_time = datetime.now()
        if (topic in self.sentiment_cache and 
            topic in self.sentiment_cache_expiry and 
            current_time < self.sentiment_cache_expiry[topic]):
            logger.info(f"Using cached sentiment data for {topic}")
            return self.sentiment_cache[topic]
        
        # Get sentiment data for the topic
        sentiment_df = self.sentiment_analyzer.analyze_sentiment(topic=topic, days_back=14)
        
        if sentiment_df.empty:
            result = {
                "topic": topic,
                "sentiment_score": Decimal("0"),
                "sentiment_trend": Decimal("0"),
                "sentiment_volatility": Decimal("0"),
                "signal": 0,
                "raw_data": pd.DataFrame()
            }
        else:
            # Calculate additional metrics
            sentiment_trend = self._calculate_sentiment_trend(sentiment_df)
            sentiment_volatility = self._calculate_sentiment_volatility(sentiment_df)
            
            # Get the latest sentiment score
            latest_sentiment = sentiment_df.iloc[-1]
            sentiment_score = Decimal(str(latest_sentiment.get("weighted_sentiment_score", 0.0)))
            
            # Generate a signal (-1, 0, 1) based on the combined metrics
            signal = self._generate_signal(sentiment_score, sentiment_trend, sentiment_volatility)
            
            result = {
                "topic": topic,
                "sentiment_score": sentiment_score,
                "sentiment_trend": sentiment_trend,
                "sentiment_volatility": sentiment_volatility,
                "signal": signal,
                "raw_data": sentiment_df
            }
        
        # Cache the result
        self.sentiment_cache[topic] = result
        self.sentiment_cache_expiry[topic] = current_time + self.cache_ttl
        
        return result
    
    def _calculate_sentiment_trend(self, sentiment_df: pd.DataFrame) -> Decimal:
        """
        Calculate the trend in sentiment scores.
        
        Args:
            sentiment_df: DataFrame containing sentiment data
            
        Returns:
            Decimal representing the sentiment trend
        """
        if sentiment_df.empty or len(sentiment_df) < 2:
            return Decimal("0")
        
        # Get the weighted sentiment scores
        scores = sentiment_df["weighted_sentiment_score"].astype(float).values
        
        # Calculate the linear regression slope
        x = np.arange(len(scores))
        slope, _ = np.polyfit(x, scores, 1)
        
        # Normalize the slope
        normalized_slope = slope * len(scores)
        
        return Decimal(str(normalized_slope))
    
    def _calculate_sentiment_volatility(self, sentiment_df: pd.DataFrame) -> Decimal:
        """
        Calculate the volatility in sentiment scores.
        
        Args:
            sentiment_df: DataFrame containing sentiment data
            
        Returns:
            Decimal representing the sentiment volatility
        """
        if sentiment_df.empty or len(sentiment_df) < 2:
            return Decimal("0")
        
        # Get the weighted sentiment scores
        scores = sentiment_df["weighted_sentiment_score"].astype(float).values
        
        # Calculate the standard deviation
        volatility = np.std(scores)
        
        return Decimal(str(volatility))
    
    def _generate_signal(self, 
                        sentiment_score: Decimal, 
                        sentiment_trend: Decimal, 
                        sentiment_volatility: Decimal) -> int:
        """
        Generate a trading signal based on sentiment metrics.
        
        Args:
            sentiment_score: Current sentiment score
            sentiment_trend: Trend in sentiment scores
            sentiment_volatility: Volatility in sentiment scores
            
        Returns:
            Signal (-1, 0, 1)
        """
        # Combine metrics with weights
        combined_score = (
            Decimal(str(self.signal_weights["sentiment"])) * sentiment_score +
            Decimal(str(self.signal_weights["trend"])) * sentiment_trend -
            Decimal(str(self.signal_weights["volatility"])) * sentiment_volatility
        )
        
        # Generate signal based on combined score
        if combined_score > self.sentiment_threshold:
            return 1
        elif combined_score < -self.sentiment_threshold:
            return -1
        else:
            return 0
    
    def _calculate_position_size(self, 
                               portfolio_manager: PortfolioManager, 
                               asset: str, 
                               price: Decimal, 
                               signal: int) -> Decimal:
        """
        Calculate the position size based on the strategy's position sizing method.
        
        Args:
            portfolio_manager: Portfolio manager instance
            asset: Asset to trade
            price: Current price of the asset
            signal: Trading signal (-1, 0, 1)
            
        Returns:
            Decimal representing the position size
        """
        if signal == 0:
            return Decimal("0")
        
        # Get the current portfolio value
        portfolio_value = portfolio_manager.portfolio.total_value
        
        # Calculate position size based on the selected method
        if self.position_sizing_method == "risk_based":
            # Calculate stop loss price
            stop_loss_pct = self.stop_loss_pct
            stop_loss_price = price * (Decimal("1") - stop_loss_pct * Decimal(str(signal)))
            
            # Calculate position size based on risk
            position_size = portfolio_manager.calculate_position_size(
                symbol=asset,
                price=price,
                stop_loss=stop_loss_price,
                risk_pct=self.risk_per_trade
            )
        elif self.position_sizing_method == "fixed_pct":
            # Use a fixed percentage of the portfolio
            position_value = portfolio_value * self.max_position_size
            position_size = position_value / price
        else:
            # Default to a small fixed size
            position_size = Decimal("0.01")
        
        return position_size
    
    def _calculate_stop_loss_take_profit(self, 
                                       price: Decimal, 
                                       side: OrderSide) -> Tuple[Decimal, Decimal]:
        """
        Calculate stop loss and take profit prices.
        
        Args:
            price: Current price of the asset
            side: Order side (BUY or SELL)
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if side == OrderSide.BUY:
            stop_loss_price = price * (Decimal("1") - self.stop_loss_pct)
            take_profit_price = price * (Decimal("1") + self.take_profit_pct)
        else:  # SELL
            stop_loss_price = price * (Decimal("1") + self.stop_loss_pct)
            take_profit_price = price * (Decimal("1") - self.take_profit_pct)
        
        return stop_loss_price, take_profit_price
    
    def generate_orders(self, portfolio_manager: PortfolioManager) -> List[Order]:
        """
        Generate orders based on sentiment analysis.
        
        Args:
            portfolio_manager: Portfolio manager instance
            
        Returns:
            List of orders to execute
        """
        orders = []
        
        # Analyze sentiment for each topic
        for topic in self.topics:
            sentiment_data = self.analyze_sentiment(topic)
            signal = sentiment_data["signal"]
            
            if signal == 0:
                continue
            
            # Get the assets associated with this topic
            topic_assets = self.assets.get(topic, [])
            if not topic_assets:
                continue
            
            # Get the weights for each asset
            asset_weights = self.topic_asset_weights.get(topic, {})
            
            # Generate orders for each asset
            for asset in topic_assets:
                # Get the weight for this asset
                weight = Decimal(str(asset_weights.get(asset, 1.0 / len(topic_assets))))
                
                # Get the current price (placeholder)
                # In a real implementation, this would come from a price feed
                price = Decimal("50000.0") if asset == "BTC" else Decimal("3000.0")
                
                # Determine order side
                side = OrderSide.BUY if signal > 0 else OrderSide.SELL
                
                # Calculate position size
                position_size = self._calculate_position_size(
                    portfolio_manager=portfolio_manager,
                    asset=asset,
                    price=price,
                    signal=signal
                )
                
                # Apply the asset weight
                position_size = position_size * weight
                
                # Skip if position size is too small
                if position_size < Decimal("0.001"):
                    continue
                
                # Calculate stop loss and take profit prices
                stop_loss_price, take_profit_price = self._calculate_stop_loss_take_profit(
                    price=price,
                    side=side
                )
                
                # Create order
                order = Order(
                    symbol=asset,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=position_size,
                    price=price,
                    stop_price=stop_loss_price
                )
                
                orders.append(order)
                logger.info(f"Generated order: {order}")
        
        return orders
    
    def run_strategy(self, portfolio_manager: PortfolioManager) -> List[Order]:
        """
        Run the sentiment strategy to generate orders.
        
        Args:
            portfolio_manager: Portfolio manager instance
            
        Returns:
            List of orders to execute
        """
        # Generate orders based on sentiment analysis
        orders = self.generate_orders(portfolio_manager)
        
        # Apply portfolio constraints
        orders = self._apply_portfolio_constraints(portfolio_manager, orders)
        
        return orders
    
    def _apply_portfolio_constraints(self, 
                                   portfolio_manager: PortfolioManager, 
                                   orders: List[Order]) -> List[Order]:
        """
        Apply portfolio constraints to the generated orders.
        
        Args:
            portfolio_manager: Portfolio manager instance
            orders: List of orders to execute
            
        Returns:
            Filtered list of orders
        """
        if not orders:
            return []
        
        # Get current positions
        current_positions = portfolio_manager.portfolio.positions
        
        # Check if we're already at the maximum number of positions
        if len(current_positions) >= self.max_positions:
            logger.info(f"Already at maximum positions ({self.max_positions}), filtering orders")
            
            # Sort orders by expected return (using sentiment score as a proxy)
            # In a real implementation, this would use a more sophisticated expected return model
            orders.sort(key=lambda x: abs(float(x.quantity) * float(x.price)), reverse=True)
            
            # Keep only the top orders
            remaining_slots = max(0, self.max_positions - len(current_positions))
            orders = orders[:remaining_slots]
        
        return orders
    
    def update_performance_history(self, portfolio_manager: PortfolioManager) -> None:
        """
        Update the strategy's performance history.
        
        Args:
            portfolio_manager: Portfolio manager instance
        """
        # Get the current portfolio value
        current_value = portfolio_manager.portfolio.total_value
        
        # Get the current timestamp
        timestamp = datetime.now()
        
        # Add to performance history
        self.performance_history.append({
            "timestamp": timestamp,
            "portfolio_value": current_value
        })
        
        # Limit the history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics for the strategy.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.performance_history:
            return {
                "total_return": Decimal("0"),
                "annualized_return": Decimal("0"),
                "sharpe_ratio": Decimal("0"),
                "max_drawdown": Decimal("0")
            }
        
        # Convert performance history to DataFrame
        df = pd.DataFrame(self.performance_history)
        
        # Calculate returns
        df["return"] = df["portfolio_value"].pct_change()
        
        # Calculate metrics
        total_return = (df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0]) - Decimal("1")
        
        # Calculate annualized return
        days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
        if days > 0:
            annualized_return = ((Decimal("1") + total_return) ** (Decimal("365") / Decimal(str(days)))) - Decimal("1")
        else:
            annualized_return = Decimal("0")
        
        # Calculate Sharpe ratio
        if len(df) > 1:
            returns = df["return"].dropna().astype(float).values
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = Decimal(str(np.mean(returns) / np.std(returns) * np.sqrt(252)))
            else:
                sharpe_ratio = Decimal("0")
        else:
            sharpe_ratio = Decimal("0")
        
        # Calculate maximum drawdown
        if len(df) > 1:
            cumulative = (1 + df["return"].fillna(0)).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative / running_max) - 1
            max_drawdown = Decimal(str(drawdown.min()))
        else:
            max_drawdown = Decimal("0")
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
