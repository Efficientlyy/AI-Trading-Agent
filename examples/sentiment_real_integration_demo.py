#!/usr/bin/env python
"""
Sentiment Analysis with Real API Integration Demo

This example demonstrates the full integration of sentiment analysis with
real market data and API connections to social media platforms.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis_agents.sentiment.nlp_service import NLPService
from src.analysis_agents.sentiment.social_media_sentiment import TwitterClient, RedditClient
from src.analysis_agents.sentiment.sentiment_base import BaseSentimentAgent
from src.common.logging import setup_logging
from src.execution.exchange.binance import BinanceExchange


class SentimentDemo:
    """Demo class for sentiment analysis with real API integration."""
    
    def __init__(self):
        """Initialize the sentiment demo."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.symbol = "BTC/USDT"  # Default symbol
        
        # Initialize NLP service
        self.nlp_service = NLPService()
        
        # Initialize social media clients
        self.twitter_client = None
        self.reddit_client = None
        
        # Initialize exchange
        self.exchange = None
        
        # Store results
        self.price_data = []
        self.sentiment_data = []
    
    def setup_logging(self):
        """Set up logging for the demo."""
        setup_logging(level=logging.INFO)
    
    async def initialize(self):
        """Initialize all components for the demo."""
        self.logger.info("Initializing sentiment demo")
        
        # Initialize NLP service
        await self.nlp_service.initialize()
        
        # Initialize Twitter client
        self.twitter_client = TwitterClient(
            api_key=os.environ.get("TWITTER_API_KEY", ""),
            api_secret=os.environ.get("TWITTER_API_SECRET", ""),
            access_token=os.environ.get("TWITTER_ACCESS_TOKEN", ""),
            access_secret=os.environ.get("TWITTER_ACCESS_SECRET", "")
        )
        
        # Initialize Reddit client
        self.reddit_client = RedditClient(
            client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
            client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
            user_agent=os.environ.get("REDDIT_USER_AGENT", "AI-Trading-Agent/1.0")
        )
        
        # Initialize Binance exchange for price data
        self.exchange = BinanceExchange()
        await self.exchange.initialize()
    
    async def get_twitter_sentiment(self, query):
        """Get sentiment from Twitter for a query.
        
        Args:
            query: The search query for Twitter
            
        Returns:
            Tuple of (sentiment_value, confidence, post_count)
        """
        self.logger.info(f"Getting Twitter sentiment for {query}")
        
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
        if len(sentiment_scores) > 1:
            import numpy as np
            std_dev = np.std(sentiment_scores)
            agreement_factor = 1.0 - min(1.0, std_dev * 2)  # Lower std_dev means higher agreement
        else:
            agreement_factor = 0.5  # Default if only one post
            
        confidence = volume_factor * agreement_factor
        
        self.logger.info(f"Twitter sentiment: {sentiment_value:.2f} with confidence {confidence:.2f} from {post_count} tweets")
        return sentiment_value, confidence, post_count
    
    async def get_reddit_sentiment(self, subreddits):
        """Get sentiment from Reddit for a list of subreddits.
        
        Args:
            subreddits: List of subreddit names
            
        Returns:
            Tuple of (sentiment_value, confidence, post_count)
        """
        self.logger.info(f"Getting Reddit sentiment for {subreddits}")
        
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
        if len(sentiment_scores) > 1:
            import numpy as np
            std_dev = np.std(sentiment_scores)
            agreement_factor = 1.0 - min(1.0, std_dev * 2)  # Lower std_dev means higher agreement
        else:
            agreement_factor = 0.5  # Default if only one post
            
        confidence = volume_factor * agreement_factor
        
        self.logger.info(f"Reddit sentiment: {sentiment_value:.2f} with confidence {confidence:.2f} from {post_count} posts")
        return sentiment_value, confidence, post_count
    
    async def get_combined_sentiment(self, symbol):
        """Get combined sentiment from all sources for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Tuple of (sentiment_value, confidence, details)
        """
        self.logger.info(f"Getting combined sentiment for {symbol}")
        
        # Extract base currency from symbol
        base_currency = symbol.split('/')[0]  # e.g., "BTC" from "BTC/USDT"
        
        # Get Twitter sentiment
        twitter_query = f"#{base_currency} OR ${base_currency}"
        twitter_sentiment, twitter_confidence, twitter_count = await self.get_twitter_sentiment(twitter_query)
        
        # Get Reddit sentiment
        subreddits = [f"r/{base_currency}", "r/CryptoCurrency", "r/CryptoMarkets"]
        reddit_sentiment, reddit_confidence, reddit_count = await self.get_reddit_sentiment(subreddits)
        
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
        
        # Compile details
        details = {
            "twitter": {
                "sentiment": twitter_sentiment,
                "confidence": twitter_confidence,
                "post_count": twitter_count
            },
            "reddit": {
                "sentiment": reddit_sentiment,
                "confidence": reddit_confidence,
                "post_count": reddit_count
            },
            "direction": direction,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Combined sentiment: {combined_sentiment:.2f} ({direction}) with confidence {combined_confidence:.2f}")
        return combined_sentiment, combined_confidence, details
    
    async def get_price_data(self, symbol, timeframe="1h", limit=24):
        """Get price data for a symbol.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The timeframe for candles (e.g., "1h")
            limit: Number of candles to retrieve
            
        Returns:
            List of candle data
        """
        self.logger.info(f"Getting price data for {symbol} on {timeframe} timeframe")
        
        # Get candles from exchange
        candles = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
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
        
        self.logger.info(f"Retrieved {len(candle_data)} candles for {symbol}")
        return candle_data
    
    async def analyze_price_sentiment_correlation(self, symbol, timeframe="1h", periods=24):
        """Analyze correlation between price and sentiment.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            timeframe: The timeframe for analysis (e.g., "1h")
            periods: Number of periods to analyze
        """
        self.logger.info(f"Analyzing price-sentiment correlation for {symbol} over {periods} {timeframe} periods")
        
        # To simulate collecting data over time, we'll get current price data 
        # and run a one-time sentiment analysis
        price_data = await self.get_price_data(symbol, timeframe, limit=periods)
        sentiment_value, confidence, details = await self.get_combined_sentiment(symbol)
        
        # Store results
        self.price_data = price_data
        
        # Create a fake historical sentiment data that's somewhat correlated with price
        # (In a real implementation, you would collect sentiment data over time)
        sentiment_history = []
        for i, candle in enumerate(price_data):
            # For demo purposes, create sentiment that's somewhat related to price movement
            # but with some random noise
            
            # Start with the current sentiment and adjust based on price patterns
            base_sentiment = sentiment_value
            
            # Calculate price change since previous period
            if i > 0:
                price_change = (candle["close"] - price_data[i-1]["close"]) / price_data[i-1]["close"]
                # Sentiment tends to follow price with some lag and noise
                price_factor = price_change * 5  # Amplify the change for sentiment effect
                
                import random
                noise = random.uniform(-0.1, 0.1)  # Add some random noise
                period_sentiment = base_sentiment + price_factor + noise
                
                # Ensure sentiment is in valid range
                period_sentiment = max(0.0, min(1.0, period_sentiment))
            else:
                # For the first period, just use the base sentiment with some noise
                import random
                noise = random.uniform(-0.1, 0.1)
                period_sentiment = base_sentiment + noise
                period_sentiment = max(0.0, min(1.0, period_sentiment))
            
            # Calculate confidence (higher for more recent periods)
            period_confidence = 0.3 + (i / len(price_data)) * 0.6
            
            sentiment_data = {
                "timestamp": candle["timestamp"],
                "sentiment": period_sentiment,
                "confidence": period_confidence
            }
            sentiment_history.append(sentiment_data)
        
        # Store the simulated sentiment history
        self.sentiment_data = sentiment_history
        
        # Calculate correlation between price and sentiment
        prices = [candle["close"] for candle in price_data]
        sentiments = [data["sentiment"] for data in sentiment_history]
        
        # Calculate rolling correlation
        window = min(12, len(prices))  # Use half the periods for rolling window
        rolling_correlations = []
        
        for i in range(len(prices) - window + 1):
            import numpy as np
            price_window = prices[i:i+window]
            sentiment_window = sentiments[i:i+window]
            correlation = np.corrcoef(price_window, sentiment_window)[0, 1]
            rolling_correlations.append(correlation)
        
        self.logger.info(f"Average correlation: {sum(rolling_correlations) / len(rolling_correlations):.2f}")
        
        # Generate visualization
        self.visualize_price_sentiment()
        
        return {
            "price_data": price_data,
            "sentiment_data": sentiment_history,
            "rolling_correlations": rolling_correlations,
            "current_sentiment": sentiment_value,
            "current_confidence": confidence,
            "details": details
        }
    
    def visualize_price_sentiment(self):
        """Visualize price and sentiment data."""
        if not self.price_data or not self.sentiment_data:
            self.logger.warning("No data to visualize")
            return
        
        # Create DataFrames
        price_df = pd.DataFrame(self.price_data)
        sentiment_df = pd.DataFrame(self.sentiment_data)
        
        # Create figure with multiple y-axes
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot price on the primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(price_df['timestamp'], price_df['close'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create a secondary y-axis for sentiment
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Sentiment (0-1)', color=color)
        ax2.plot(sentiment_df['timestamp'], sentiment_df['sentiment'], color=color, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1)  # Sentiment is on a 0-1 scale
        
        # Fill the sentiment line
        ax2.fill_between(sentiment_df['timestamp'], 0.5, sentiment_df['sentiment'], 
                        where=(sentiment_df['sentiment'] > 0.5),
                        color='green', alpha=0.3, interpolate=True)
        ax2.fill_between(sentiment_df['timestamp'], sentiment_df['sentiment'], 0.5, 
                        where=(sentiment_df['sentiment'] < 0.5),
                        color='red', alpha=0.3, interpolate=True)
        
        # Add a horizontal line at neutral sentiment (0.5)
        ax2.axhline(y=0.5, color='grey', linestyle='-', alpha=0.7)
        
        # Add title
        plt.title(f'Price and Sentiment Analysis for {self.symbol}')
        
        # Add annotations for extreme sentiment points
        for i, row in sentiment_df.iterrows():
            if row['sentiment'] > 0.7 or row['sentiment'] < 0.3:
                sentiment_type = "Bullish" if row['sentiment'] > 0.7 else "Bearish"
                ax2.annotate(f"{sentiment_type}\n{row['sentiment']:.2f}",
                            (row['timestamp'], row['sentiment']),
                            xytext=(0, 10 if row['sentiment'] > 0.7 else -10),
                            textcoords='offset points',
                            ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Format the plot
        fig.tight_layout()
        
        # Save the figure
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/sentiment_price_analysis.png')
        self.logger.info("Generated price-sentiment visualization in output/sentiment_price_analysis.png")
        
        # Show plot
        plt.show()
    
    async def generate_trading_signals(self, symbol):
        """Generate trading signals based on sentiment analysis.
        
        Args:
            symbol: The trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Dictionary with trading signals and analysis
        """
        self.logger.info(f"Generating trading signals for {symbol}")
        
        # Get sentiment data
        sentiment_value, confidence, details = await self.get_combined_sentiment(symbol)
        
        # Get price data for technical confirmation
        price_data = await self.get_price_data(symbol, timeframe="1h", limit=24)
        
        # Calculate a simple RSI for technical confirmation
        closes = [candle["close"] for candle in price_data]
        rsi = self.calculate_rsi(closes)
        
        # Define signal thresholds
        strong_buy_threshold = 0.75
        buy_threshold = 0.6
        strong_sell_threshold = 0.25
        sell_threshold = 0.4
        
        # Generate base signal from sentiment
        if sentiment_value >= strong_buy_threshold and confidence > 0.6:
            base_signal = "STRONG_BUY"
        elif sentiment_value >= buy_threshold:
            base_signal = "BUY"
        elif sentiment_value <= strong_sell_threshold and confidence > 0.6:
            base_signal = "STRONG_SELL"
        elif sentiment_value <= sell_threshold:
            base_signal = "SELL"
        else:
            base_signal = "NEUTRAL"
        
        # Check for contrarian signals (extremely high/low sentiment may be a reversal indicator)
        contrarian_signal = None
        if sentiment_value >= 0.9 and confidence > 0.7:
            contrarian_signal = "Potential reversal - sentiment extremely bullish (contrarian SELL signal)"
        elif sentiment_value <= 0.1 and confidence > 0.7:
            contrarian_signal = "Potential reversal - sentiment extremely bearish (contrarian BUY signal)"
        
        # Confirm with technical indicator (RSI)
        technical_signal = None
        if rsi > 70:
            technical_signal = "Overbought (RSI: {:.2f})".format(rsi)
        elif rsi < 30:
            technical_signal = "Oversold (RSI: {:.2f})".format(rsi)
        else:
            technical_signal = "Neutral (RSI: {:.2f})".format(rsi)
        
        # Combine signals for final recommendation
        final_signal = base_signal
        
        # If we have a contrarian warning, moderate the signal
        if contrarian_signal and "STRONG_" in base_signal:
            # Remove the "STRONG_" prefix if there's a contrarian warning
            final_signal = base_signal.replace("STRONG_", "")
        
        # If technical signal contradicts sentiment, moderate the signal
        if ("BUY" in base_signal and "Overbought" in technical_signal) or \
           ("SELL" in base_signal and "Oversold" in technical_signal):
            # Downgrade the signal
            if base_signal == "STRONG_BUY":
                final_signal = "BUY"
            elif base_signal == "STRONG_SELL":
                final_signal = "SELL"
            else:
                final_signal = "NEUTRAL"
        
        # Compile results
        signal_data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "sentiment": {
                "value": sentiment_value,
                "confidence": confidence,
                "direction": details["direction"]
            },
            "technical": {
                "rsi": rsi,
                "signal": technical_signal
            },
            "signals": {
                "base_signal": base_signal,
                "contrarian_warning": contrarian_signal,
                "technical_confirmation": technical_signal,
                "final_signal": final_signal
            },
            "details": details
        }
        
        self.logger.info(f"Generated signal: {final_signal} for {symbol} (Sentiment: {sentiment_value:.2f}, RSI: {rsi:.2f})")
        return signal_data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate the Relative Strength Index (RSI).
        
        Args:
            prices: List of closing prices
            period: RSI period (default: 14)
            
        Returns:
            RSI value
        """
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


async def main():
    """Run the sentiment analysis demo."""
    demo = SentimentDemo()
    await demo.initialize()
    
    # Set the symbol to analyze
    demo.symbol = "BTC/USDT"
    
    # Analyze correlation between price and sentiment
    correlation_data = await demo.analyze_price_sentiment_correlation(demo.symbol)
    
    # Generate trading signals
    signals = await demo.generate_trading_signals(demo.symbol)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"SENTIMENT ANALYSIS RESULTS FOR {demo.symbol}")
    print("=" * 50)
    print(f"Current Sentiment: {signals['sentiment']['value']:.2f} ({signals['sentiment']['direction']})")
    print(f"Confidence: {signals['sentiment']['confidence']:.2f}")
    print(f"\nRSI: {signals['technical']['rsi']:.2f}")
    print(f"Technical Signal: {signals['technical']['signal']}")
    print("\nSOURCE BREAKDOWN:")
    print(f"  Twitter: {signals['details']['twitter']['sentiment']:.2f} (confidence: {signals['details']['twitter']['confidence']:.2f}, posts: {signals['details']['twitter']['post_count']})")
    print(f"  Reddit: {signals['details']['reddit']['sentiment']:.2f} (confidence: {signals['details']['reddit']['confidence']:.2f}, posts: {signals['details']['reddit']['post_count']})")
    print("\nSIGNALS:")
    print(f"  Base Signal: {signals['signals']['base_signal']}")
    if signals['signals']['contrarian_warning']:
        print(f"  Contrarian Warning: {signals['signals']['contrarian_warning']}")
    print(f"  Technical Confirmation: {signals['signals']['technical_confirmation']}")
    print("\n" + "=" * 50)
    print(f"FINAL SIGNAL: {signals['signals']['final_signal']}")
    print("=" * 50)
    print("\nVisualization saved to output/sentiment_price_analysis.png")


if __name__ == "__main__":
    asyncio.run(main())