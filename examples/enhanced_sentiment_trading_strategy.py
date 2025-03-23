#!/usr/bin/env python
"""
Enhanced Sentiment Trading Strategy

This example demonstrates how to implement a trading strategy that combines:
1. Sentiment analysis from multiple sources (social media, news)
2. Technical indicators for confirmation
3. Market regime detection for context-aware decisions
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis_agents.sentiment.nlp_service import NLPService
from src.analysis_agents.sentiment.social_media_sentiment import TwitterClient, RedditClient
from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.common.logging import setup_logging
from src.execution.exchange.binance import BinanceExchange


class MarketRegimeDetector:
    """Detect market regimes for better strategy selection."""
    
    def __init__(self):
        """Initialize the market regime detector."""
        self.logger = logging.getLogger(__name__)
    
    def detect_regime(self, candles, window=20):
        """Detect the current market regime.
        
        Args:
            candles: List of candle data dictionaries
            window: Window size for regime detection
            
        Returns:
            Dictionary with regime information
        """
        if len(candles) < window:
            return {"regime": "unknown", "confidence": 0.0}
        
        # Extract close prices
        closes = [candle["close"] for candle in candles]
        
        # Calculate volatility (standard deviation of returns)
        returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
        volatility = np.std(returns[-window:]) * 100  # Convert to percentage
        
        # Calculate trend strength using linear regression
        x = np.arange(window)
        y = np.array(closes[-window:])
        slope, _ = np.polyfit(x, y, 1)
        trend_strength = slope / closes[-window] * 100 * window  # Normalize and convert to percentage
        
        # Calculate volume trend
        volumes = [candle["volume"] for candle in candles]
        volume_trend = sum(volumes[-int(window/2):]) / sum(volumes[-window:-int(window/2)]) - 1
        
        # Determine regime
        if abs(trend_strength) < 1:  # Less than 1% trend over the window
            if volatility > 4:  # High volatility
                regime = "choppy"
                regime_confidence = min(1.0, volatility / 8)  # Higher volatility = more confident it's choppy
            else:
                regime = "ranging"
                regime_confidence = min(1.0, (4 - volatility) / 3)  # Lower volatility = more confident it's ranging
        else:
            # Trending market
            if trend_strength > 1:  # Uptrend
                if volatility > 5:  # High volatility uptrend
                    regime = "volatile_bullish"
                    regime_confidence = min(1.0, trend_strength / 10) * min(1.0, volatility / 10)
                else:
                    regime = "bullish"
                    regime_confidence = min(1.0, trend_strength / 10)
            else:  # Downtrend
                if volatility > 5:  # High volatility downtrend
                    regime = "volatile_bearish"
                    regime_confidence = min(1.0, abs(trend_strength) / 10) * min(1.0, volatility / 10)
                else:
                    regime = "bearish"
                    regime_confidence = min(1.0, abs(trend_strength) / 10)
                    
        # Verify with volume
        if volume_trend > 0.2:  # Volume increasing
            volume_confirmation = "increasing"
            if "bullish" in regime:
                regime_confidence *= 1.2  # Increase confidence for bullish with increasing volume
            elif "bearish" in regime:
                regime_confidence *= 0.8  # Decrease confidence for bearish with increasing volume
        elif volume_trend < -0.2:  # Volume decreasing
            volume_confirmation = "decreasing"
            if "bearish" in regime:
                regime_confidence *= 1.2  # Increase confidence for bearish with decreasing volume
            elif "bullish" in regime:
                regime_confidence *= 0.8  # Decrease confidence for bullish with decreasing volume
        else:
            volume_confirmation = "neutral"
        
        # Cap confidence at 1.0
        regime_confidence = min(1.0, regime_confidence)
        
        self.logger.info(f"Detected market regime: {regime} (confidence: {regime_confidence:.2f}, volatility: {volatility:.2f}%, trend: {trend_strength:.2f}%)")
        
        return {
            "regime": regime,
            "confidence": regime_confidence,
            "volatility": volatility,
            "trend_strength": trend_strength,
            "volume_trend": volume_trend,
            "volume_confirmation": volume_confirmation
        }


class EnhancedSentimentStrategy:
    """Trading strategy combining sentiment analysis with technical indicators."""
    
    def __init__(self):
        """Initialize the sentiment-based strategy."""
        setup_logging(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Default settings
        self.symbol = "BTC/USDT"
        self.timeframe = "1h"
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.sentiment_threshold_buy = 0.65
        self.sentiment_threshold_sell = 0.35
        self.minimum_confidence = 0.5
        
        # Position management
        self.position = {
            "active": False,
            "type": None,  # "long" or "short"
            "entry_price": 0,
            "entry_time": None,
            "size": 0,
            "sentiment_at_entry": 0,
            "sentiment_shift": 0,  # Track sentiment shift since entry
            "technical_confirmation": False
        }
        
        # Performance metrics
        self.trades = []
        self.total_pnl = 0
        self.win_count = 0
        self.loss_count = 0
        
        # Components
        self.nlp_service = NLPService()
        self.twitter_client = None
        self.reddit_client = None
        self.exchange = None
        self.regime_detector = MarketRegimeDetector()
        
        # Data storage
        self.price_data = []
        self.sentiment_data = []
        self.signals = []
    
    async def initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing Enhanced Sentiment Strategy")
        
        # Initialize NLP service
        await self.nlp_service.initialize()
        
        # Initialize social media clients
        self.twitter_client = TwitterClient(
            api_key=os.environ.get("TWITTER_API_KEY", ""),
            api_secret=os.environ.get("TWITTER_API_SECRET", ""),
            access_token=os.environ.get("TWITTER_ACCESS_TOKEN", ""),
            access_secret=os.environ.get("TWITTER_ACCESS_SECRET", "")
        )
        
        self.reddit_client = RedditClient(
            client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
            client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
            user_agent=os.environ.get("REDDIT_USER_AGENT", "AI-Trading-Agent/1.0")
        )
        
        # Initialize exchange
        self.exchange = BinanceExchange()
        await self.exchange.initialize()
    
    async def get_market_data(self, lookback_periods=50):
        """Get market data for analysis.
        
        Args:
            lookback_periods: Number of periods to retrieve
            
        Returns:
            List of candle data
        """
        self.logger.info(f"Getting market data for {self.symbol}, {lookback_periods} periods of {self.timeframe}")
        
        # Get candles from exchange
        candles = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=lookback_periods)
        
        # Convert to list of dictionaries
        candle_data = []
        for candle in candles:
            timestamp = datetime.fromtimestamp(candle[0] / 1000)
            candle_dict = {
                "timestamp": timestamp,
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5]
            }
            candle_data.append(candle_dict)
        
        self.price_data = candle_data
        self.logger.info(f"Retrieved {len(candle_data)} candles for {self.symbol}")
        return candle_data
    
    async def get_sentiment(self):
        """Get latest sentiment data.
        
        Returns:
            Dictionary with sentiment data
        """
        base_currency = self.symbol.split('/')[0]  # e.g., "BTC" from "BTC/USDT"
        
        # Get Twitter sentiment
        twitter_query = f"#{base_currency} OR ${base_currency}"
        twitter_sentiment, twitter_confidence, twitter_count = await self._get_twitter_sentiment(twitter_query)
        
        # Get Reddit sentiment
        subreddits = [f"r/{base_currency}", "r/CryptoCurrency", "r/CryptoMarkets"]
        reddit_sentiment, reddit_confidence, reddit_count = await self._get_reddit_sentiment(subreddits)
        
        # Combine sentiments with weighting
        twitter_weight = 0.5
        reddit_weight = 0.5
        
        # Adjust weights based on confidence
        total_confidence = twitter_confidence + reddit_confidence
        if total_confidence > 0:
            twitter_weight = twitter_confidence / total_confidence
            reddit_weight = reddit_confidence / total_confidence
        
        # Calculate weighted sentiment
        combined_sentiment = (
            twitter_sentiment * twitter_weight +
            reddit_sentiment * reddit_weight
        )
        
        # Combined confidence is average of individual confidences
        combined_confidence = (twitter_confidence + reddit_confidence) / 2
        
        # Determine direction
        if combined_sentiment > 0.6:
            direction = "bullish"
        elif combined_sentiment < 0.4:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Record sentiment data
        sentiment_data = {
            "timestamp": datetime.now(),
            "value": combined_sentiment,
            "confidence": combined_confidence,
            "direction": direction,
            "sources": {
                "twitter": {
                    "sentiment": twitter_sentiment,
                    "confidence": twitter_confidence,
                    "count": twitter_count
                },
                "reddit": {
                    "sentiment": reddit_sentiment,
                    "confidence": reddit_confidence,
                    "count": reddit_count
                }
            }
        }
        
        self.sentiment_data.append(sentiment_data)
        self.logger.info(f"Current sentiment: {combined_sentiment:.2f} ({direction}) with confidence {combined_confidence:.2f}")
        
        return sentiment_data
    
    async def _get_twitter_sentiment(self, query):
        """Get sentiment from Twitter for a query.
        
        Args:
            query: The search query for Twitter
            
        Returns:
            Tuple of (sentiment_value, confidence, post_count)
        """
        # Search for tweets
        tweets = await self.twitter_client.search_tweets(
            query=query,
            count=100,
            result_type="recent"
        )
        
        if not tweets:
            self.logger.warning(f"No tweets found for query: {query}")
            return 0.5, 0.0, 0
        
        # Analyze sentiment of tweets
        sentiment_scores = await self.nlp_service.analyze_sentiment(tweets)
        
        # Calculate overall sentiment
        sentiment_value = sum(sentiment_scores) / len(sentiment_scores)
        
        # Calculate confidence based on post volume and agreement
        post_count = len(tweets)
        volume_factor = min(1.0, post_count / 100)
        
        # Calculate standard deviation to measure agreement
        std_dev = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.5
        agreement_factor = 1.0 - min(1.0, std_dev * 2)  # Lower std_dev means higher agreement
            
        confidence = volume_factor * agreement_factor
        
        self.logger.info(f"Twitter sentiment: {sentiment_value:.2f} with confidence {confidence:.2f} from {post_count} tweets")
        return sentiment_value, confidence, post_count
    
    async def _get_reddit_sentiment(self, subreddits):
        """Get sentiment from Reddit for a list of subreddits.
        
        Args:
            subreddits: List of subreddit names
            
        Returns:
            Tuple of (sentiment_value, confidence, post_count)
        """
        all_posts = []
        
        # Get posts from each subreddit
        for subreddit in subreddits:
            posts = await self.reddit_client.get_hot_posts(
                subreddit=subreddit,
                limit=50,
                time_filter="day"
            )
            all_posts.extend(posts)
        
        if not all_posts:
            self.logger.warning(f"No posts found for subreddits: {subreddits}")
            return 0.5, 0.0, 0
        
        # Analyze sentiment of posts
        sentiment_scores = await self.nlp_service.analyze_sentiment(all_posts)
        
        # Calculate overall sentiment
        sentiment_value = sum(sentiment_scores) / len(sentiment_scores)
        
        # Calculate confidence based on post volume and agreement
        post_count = len(all_posts)
        volume_factor = min(1.0, post_count / 100)
        
        # Calculate standard deviation to measure agreement
        std_dev = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.5
        agreement_factor = 1.0 - min(1.0, std_dev * 2)  # Lower std_dev means higher agreement
            
        confidence = volume_factor * agreement_factor
        
        self.logger.info(f"Reddit sentiment: {sentiment_value:.2f} with confidence {confidence:.2f} from {post_count} posts")
        return sentiment_value, confidence, post_count
    
    def calculate_rsi(self, prices, period=None):
        """Calculate RSI technical indicator.
        
        Args:
            prices: List of closing prices
            period: RSI period (default: use class setting)
            
        Returns:
            RSI value (0-100)
        """
        if period is None:
            period = self.rsi_period
            
        if len(prices) < period + 1:
            return 50  # Default to neutral if not enough data
        
        # Calculate price changes
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Calculate average gain and loss
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100  # Prevent division by zero
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def check_for_contrarian_signals(self, sentiment, confidence):
        """Check for contrarian signals based on extreme sentiment.
        
        Args:
            sentiment: Current sentiment value (0-1)
            confidence: Confidence in the sentiment value (0-1)
            
        Returns:
            Tuple of (has_contrarian_signal, contrarian_sentiment, message)
        """
        # Extreme bullish sentiment may indicate a local top
        if sentiment > 0.9 and confidence > 0.7:
            self.logger.warning("Detected extreme bullish sentiment - potential contrarian sell signal")
            return True, 1 - sentiment, "Extreme bullish sentiment may indicate market euphoria and a potential top"
        
        # Extreme bearish sentiment may indicate a local bottom
        if sentiment < 0.1 and confidence > 0.7:
            self.logger.warning("Detected extreme bearish sentiment - potential contrarian buy signal")
            return True, 1 - sentiment, "Extreme bearish sentiment may indicate capitulation and a potential bottom"
        
        return False, sentiment, None
    
    async def generate_signal(self):
        """Generate trading signal based on sentiment and technical indicators.
        
        Returns:
            Dictionary with signal details
        """
        # Get price data
        candles = await self.get_market_data()
        closes = [candle["close"] for candle in candles]
        current_price = closes[-1]
        
        # Get sentiment data
        sentiment_data = await self.get_sentiment()
        sentiment = sentiment_data["value"]
        confidence = sentiment_data["confidence"]
        direction = sentiment_data["direction"]
        
        # Calculate technical indicators
        rsi = self.calculate_rsi(closes)
        
        # Check for contrarian signals
        has_contrarian, contrarian_sentiment, contrarian_message = await self.check_for_contrarian_signals(sentiment, confidence)
        
        # If contrarian signal detected, use the adjusted sentiment
        if has_contrarian:
            sentiment = contrarian_sentiment
            direction = "bearish" if sentiment < 0.5 else "bullish"
        
        # Detect market regime
        regime_data = self.regime_detector.detect_regime(candles)
        regime = regime_data["regime"]
        regime_confidence = regime_data["confidence"]
        
        # Determine technical confirmation status
        technical_confirmation = False
        technical_message = ""
        
        if direction == "bullish" and rsi < 70:  # Not overbought
            technical_confirmation = True
            technical_message = f"RSI ({rsi:.1f}) confirms bullish sentiment"
        elif direction == "bearish" and rsi > 30:  # Not oversold
            technical_confirmation = True
            technical_message = f"RSI ({rsi:.1f}) confirms bearish sentiment"
        else:
            if direction == "bullish" and rsi >= 70:
                technical_message = f"RSI ({rsi:.1f}) indicates overbought conditions, contradicting bullish sentiment"
            elif direction == "bearish" and rsi <= 30:
                technical_message = f"RSI ({rsi:.1f}) indicates oversold conditions, contradicting bearish sentiment"
            else:
                technical_message = f"RSI ({rsi:.1f}) is neutral"
        
        # Adjust sentiment based on market regime
        regime_adjusted_sentiment = sentiment
        
        if direction == "bullish":
            if regime in ["bullish", "volatile_bullish"]:
                # Amplify bullish sentiment in bullish regimes
                regime_adjusted_sentiment = 0.5 + (sentiment - 0.5) * 1.3
                regime_message = f"{regime} regime enhances bullish sentiment"
            elif regime in ["bearish", "volatile_bearish"]:
                # Reduce bullish sentiment in bearish regimes
                regime_adjusted_sentiment = 0.5 + (sentiment - 0.5) * 0.7
                regime_message = f"{regime} regime weakens bullish sentiment"
            else:
                regime_message = f"{regime} regime has minimal impact on sentiment"
        else:  # bearish or neutral
            if regime in ["bearish", "volatile_bearish"]:
                # Amplify bearish sentiment in bearish regimes
                regime_adjusted_sentiment = 0.5 - (0.5 - sentiment) * 1.3
                regime_message = f"{regime} regime enhances bearish sentiment"
            elif regime in ["bullish", "volatile_bullish"]:
                # Reduce bearish sentiment in bullish regimes
                regime_adjusted_sentiment = 0.5 - (0.5 - sentiment) * 0.7
                regime_message = f"{regime} regime weakens bearish sentiment"
            else:
                regime_message = f"{regime} regime has minimal impact on sentiment"
        
        # Clip to valid range
        regime_adjusted_sentiment = max(0.0, min(1.0, regime_adjusted_sentiment))
        
        # Generate signal
        signal_type = None
        signal_strength = 0
        
        # Strong buy signal
        if (regime_adjusted_sentiment > self.sentiment_threshold_buy and 
            confidence > self.minimum_confidence and
            technical_confirmation):
            signal_type = "BUY"
            signal_strength = (regime_adjusted_sentiment - self.sentiment_threshold_buy) * 10
        
        # Strong sell signal
        elif (regime_adjusted_sentiment < self.sentiment_threshold_sell and 
              confidence > self.minimum_confidence and
              technical_confirmation):
            signal_type = "SELL"
            signal_strength = (self.sentiment_threshold_sell - regime_adjusted_sentiment) * 10
        
        # Weak signals - when technical doesn't confirm
        elif regime_adjusted_sentiment > self.sentiment_threshold_buy and confidence > self.minimum_confidence:
            signal_type = "WEAK_BUY"
            signal_strength = (regime_adjusted_sentiment - self.sentiment_threshold_buy) * 5
        
        elif regime_adjusted_sentiment < self.sentiment_threshold_sell and confidence > self.minimum_confidence:
            signal_type = "WEAK_SELL"
            signal_strength = (self.sentiment_threshold_sell - regime_adjusted_sentiment) * 5
        
        else:
            signal_type = "NEUTRAL"
            signal_strength = 0
            
        # Compile signal data
        signal = {
            "timestamp": datetime.now(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "price": current_price,
            "type": signal_type,
            "strength": signal_strength,
            "sentiment": {
                "original": sentiment,
                "regime_adjusted": regime_adjusted_sentiment,
                "confidence": confidence,
                "direction": direction,
                "has_contrarian": has_contrarian,
                "contrarian_message": contrarian_message
            },
            "technical": {
                "rsi": rsi,
                "confirmation": technical_confirmation,
                "message": technical_message
            },
            "regime": {
                "type": regime,
                "confidence": regime_confidence,
                "message": regime_message,
                "volatility": regime_data["volatility"],
                "trend_strength": regime_data["trend_strength"]
            }
        }
        
        self.signals.append(signal)
        self.logger.info(f"Generated signal: {signal_type} (strength: {signal_strength:.2f}) at price {current_price}")
        
        return signal
    
    async def execute_signal(self, signal, simulated=True):
        """Execute a trading signal.
        
        Args:
            signal: The signal to execute
            simulated: Whether to simulate the trade or execute real orders
            
        Returns:
            Dictionary with execution details
        """
        if simulated:
            # Simulate execution
            execution_price = signal["price"]
            slippage = 0  # No slippage in simulation
            position_size = 1.0  # Standard position size
            execution_time = signal["timestamp"]
            fees = 0.001 * execution_price * position_size  # 0.1% fee
            
            # Check if we need to close an existing position
            if self.position["active"]:
                # Calculate P&L for the position we're closing
                if self.position["type"] == "long":
                    pnl = (execution_price - self.position["entry_price"]) * self.position["size"] - fees
                    pnl_percent = (execution_price / self.position["entry_price"] - 1) * 100
                else:  # short
                    pnl = (self.position["entry_price"] - execution_price) * self.position["size"] - fees
                    pnl_percent = (1 - execution_price / self.position["entry_price"]) * 100
                
                # Record the trade
                trade = {
                    "entry_time": self.position["entry_time"],
                    "exit_time": execution_time,
                    "entry_price": self.position["entry_price"],
                    "exit_price": execution_price,
                    "type": self.position["type"],
                    "size": self.position["size"],
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "fees": fees,
                    "entry_sentiment": self.position["sentiment_at_entry"],
                    "exit_sentiment": signal["sentiment"]["regime_adjusted"],
                    "sentiment_shift": signal["sentiment"]["regime_adjusted"] - self.position["sentiment_at_entry"]
                }
                
                self.trades.append(trade)
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                
                self.logger.info(f"Closed {self.position['type']} position at {execution_price:.2f} with P&L: {pnl:.2f} ({pnl_percent:.2f}%)")
                
                # Reset position
                self.position = {
                    "active": False,
                    "type": None,
                    "entry_price": 0,
                    "entry_time": None,
                    "size": 0,
                    "sentiment_at_entry": 0,
                    "sentiment_shift": 0,
                    "technical_confirmation": False
                }
            
            # Check if we need to open a new position
            if signal["type"] in ["BUY", "SELL"]:
                position_type = "long" if signal["type"] == "BUY" else "short"
                
                # Open new position
                self.position = {
                    "active": True,
                    "type": position_type,
                    "entry_price": execution_price,
                    "entry_time": execution_time,
                    "size": position_size,
                    "sentiment_at_entry": signal["sentiment"]["regime_adjusted"],
                    "sentiment_shift": 0,
                    "technical_confirmation": signal["technical"]["confirmation"]
                }
                
                self.logger.info(f"Opened {position_type} position at {execution_price:.2f} with size {position_size}")
            
            execution_result = {
                "simulated": True,
                "success": True,
                "price": execution_price,
                "size": position_size if signal["type"] in ["BUY", "SELL"] else 0,
                "type": signal["type"],
                "slippage": slippage,
                "fees": fees,
                "timestamp": execution_time
            }
            
            return execution_result
        else:
            # Real execution via exchange
            self.logger.warning("Real order execution not implemented")
            return {"simulated": False, "success": False, "message": "Real order execution not implemented"}
    
    def visualize_strategy(self):
        """Visualize the strategy performance."""
        if not self.price_data or not self.sentiment_data or not self.signals:
            self.logger.warning("Not enough data to visualize strategy performance")
            return
        
        # Create DataFrames
        price_df = pd.DataFrame(self.price_data)
        
        # Convert sentiment data to DataFrame
        sentiment_df = pd.DataFrame(self.sentiment_data)
        
        # Convert signals to DataFrame
        signals_df = pd.DataFrame(self.signals)
        signals_df["buy"] = signals_df["type"].apply(lambda x: 1 if x == "BUY" else 0)
        signals_df["sell"] = signals_df["type"].apply(lambda x: 1 if x == "SELL" else 0)
        
        # Create figure with multiple y-axes
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot price with buy/sell markers on the top subplot
        ax1.plot(price_df['timestamp'], price_df['close'], color='blue', label='Price')
        
        # Add buy signals
        buy_signals = signals_df[signals_df["buy"] == 1]
        if not buy_signals.empty:
            buy_prices = []
            buy_times = []
            for idx, row in buy_signals.iterrows():
                matching_price = price_df[price_df['timestamp'] <= row['timestamp']].iloc[-1]['close']
                buy_prices.append(matching_price)
                buy_times.append(row['timestamp'])
            ax1.scatter(buy_times, buy_prices, color='green', s=100, marker='^', label='Buy Signal')
        
        # Add sell signals
        sell_signals = signals_df[signals_df["sell"] == 1]
        if not sell_signals.empty:
            sell_prices = []
            sell_times = []
            for idx, row in sell_signals.iterrows():
                matching_price = price_df[price_df['timestamp'] <= row['timestamp']].iloc[-1]['close']
                sell_prices.append(matching_price)
                sell_times.append(row['timestamp'])
            ax1.scatter(sell_times, sell_prices, color='red', s=100, marker='v', label='Sell Signal')
        
        # Format price chart
        ax1.set_ylabel('Price')
        ax1.set_title(f'Enhanced Sentiment Strategy - {self.symbol}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot sentiment on the middle subplot
        ax2.plot(sentiment_df['timestamp'], sentiment_df['value'], color='purple', label='Sentiment')
        ax2.axhline(y=0.5, color='grey', linestyle='--', alpha=0.7)
        ax2.axhline(y=self.sentiment_threshold_buy, color='green', linestyle='--', alpha=0.7)
        ax2.axhline(y=self.sentiment_threshold_sell, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Sentiment (0-1)')
        ax2.set_ylim(0, 1)
        ax2.fill_between(sentiment_df['timestamp'], 0.5, sentiment_df['value'], 
                         where=(sentiment_df['value'] > 0.5), color='green', alpha=0.2)
        ax2.fill_between(sentiment_df['timestamp'], sentiment_df['value'], 0.5, 
                         where=(sentiment_df['value'] < 0.5), color='red', alpha=0.2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot RSI on the bottom subplot
        rsi_values = [self.calculate_rsi([candle["close"] for candle in self.price_data[:i+1]]) 
                     for i in range(len(self.price_data))]
        ax3.plot(price_df['timestamp'], rsi_values, color='orange', label='RSI')
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Set common x-axis label
        ax3.set_xlabel('Time')
        
        # Format the plot
        plt.tight_layout()
        
        # Save the figure
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/enhanced_sentiment_strategy.png')
        self.logger.info("Generated strategy visualization in output/enhanced_sentiment_strategy.png")
        plt.show()
    
    def print_performance_report(self):
        """Print a performance report for the strategy."""
        print("\n" + "=" * 60)
        print(f"ENHANCED SENTIMENT STRATEGY PERFORMANCE REPORT: {self.symbol}")
        print("=" * 60)
        
        # Trade statistics
        win_rate = self.win_count / (self.win_count + self.loss_count) * 100 if (self.win_count + self.loss_count) > 0 else 0
        
        print(f"Total trades: {len(self.trades)}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total P&L: {self.total_pnl:.2f}")
        
        if self.trades:
            # Calculate more detailed stats
            pnl_values = [trade["pnl"] for trade in self.trades]
            pnl_percentages = [trade["pnl_percent"] for trade in self.trades]
            max_win = max(pnl_values) if pnl_values else 0
            max_loss = min(pnl_values) if pnl_values else 0
            avg_win = sum([pnl for pnl in pnl_values if pnl > 0]) / self.win_count if self.win_count > 0 else 0
            avg_loss = sum([pnl for pnl in pnl_values if pnl <= 0]) / self.loss_count if self.loss_count > 0 else 0
            
            print(f"Average win: {avg_win:.2f}")
            print(f"Average loss: {avg_loss:.2f}")
            print(f"Largest win: {max_win:.2f}")
            print(f"Largest loss: {max_loss:.2f}")
            
            if self.loss_count > 0 and avg_loss < 0:
                profit_factor = abs(sum([pnl for pnl in pnl_values if pnl > 0]) / sum([pnl for pnl in pnl_values if pnl < 0]))
                print(f"Profit factor: {profit_factor:.2f}")
            
            # Check if sentiment shift predicts trade outcome
            sentiment_correct_count = sum(1 for trade in self.trades 
                                         if (trade["type"] == "long" and trade["sentiment_shift"] > 0 and trade["pnl"] > 0) or
                                            (trade["type"] == "short" and trade["sentiment_shift"] < 0 and trade["pnl"] > 0) or
                                            (trade["type"] == "long" and trade["sentiment_shift"] < 0 and trade["pnl"] < 0) or
                                            (trade["type"] == "short" and trade["sentiment_shift"] > 0 and trade["pnl"] < 0))
            
            sentiment_accuracy = sentiment_correct_count / len(self.trades) * 100 if len(self.trades) > 0 else 0
            print(f"\nSentiment prediction accuracy: {sentiment_accuracy:.1f}%")
            
            # List recent trades
            print("\nRECENT TRADES:")
            for trade in self.trades[-5:]:
                trade_type = "LONG" if trade["type"] == "long" else "SHORT"
                result = "WIN" if trade["pnl"] > 0 else "LOSS"
                print(f"{trade_type} {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} to {trade['exit_time'].strftime('%Y-%m-%d %H:%M')}: {trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%) - {result}")
        
        print("=" * 60)


async def main():
    """Run the enhanced sentiment trading strategy."""
    # Initialize the strategy
    strategy = EnhancedSentimentStrategy()
    await strategy.initialize()
    
    # Set the trading parameters
    strategy.symbol = "BTC/USDT"
    strategy.timeframe = "1h"
    
    # Run a simulation of market updates
    print("Running trading simulation...")
    
    # Simulate market updates over 24 periods
    for i in range(24):
        # Generate a new signal
        signal = await strategy.generate_signal()
        
        # Execute the signal in simulation mode
        execution = await strategy.execute_signal(signal, simulated=True)
        
        # Simulate time passing
        await asyncio.sleep(0.1)
    
    # Visualize the results
    strategy.visualize_strategy()
    
    # Print performance report
    strategy.print_performance_report()


if __name__ == "__main__":
    asyncio.run(main())