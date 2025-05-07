"""
Trading Signal Integration Service

This module provides a comprehensive service for integrating various trading signals
from different sources (technical indicators, sentiment analysis, etc.) and generating
combined trading signals based on configurable rules and weights.
"""

import logging
import json
import asyncio
import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from enum import Enum

# Local imports
from ai_trading_agent.data_collectors.alpha_vantage_client import AlphaVantageClient
from ai_trading_agent.sentiment_analysis.nlp_processor import SentimentSignalGenerator
from ai_trading_agent.feature_engineering.advanced_features import (
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_macd
)

# Configure logging
logger = logging.getLogger(__name__)

class SignalType(str, Enum):
    """Enum for different types of trading signals"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalSource(str, Enum):
    """Enum for different sources of trading signals"""
    TECHNICAL = "TECHNICAL"
    SENTIMENT = "SENTIMENT"
    COMBINED = "COMBINED"


class TradingSignal:
    """Class representing a trading signal"""
    
    def __init__(
        self,
        symbol: str,
        timestamp: datetime.datetime,
        signal_type: SignalType,
        source: SignalSource,
        strength: float,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new trading signal
        
        Args:
            symbol: Trading symbol (e.g., BTC/USDT)
            timestamp: Time when the signal was generated
            signal_type: Type of signal (buy, sell, etc.)
            source: Source of the signal (technical, sentiment, etc.)
            strength: Signal strength between 0.0 and 1.0
            description: Human-readable description of the signal
            metadata: Additional metadata about the signal
        """
        self.id = f"{symbol}-{timestamp.isoformat()}-{source}"
        self.symbol = symbol
        self.timestamp = timestamp
        self.type = signal_type
        self.source = source
        self.strength = min(max(strength, 0.0), 1.0)  # Clamp between 0 and 1
        self.description = description
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the signal to a dictionary for serialization"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "type": self.type,
            "source": self.source,
            "strength": self.strength,
            "description": self.description,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create a TradingSignal from a dictionary"""
        return cls(
            symbol=data["symbol"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            signal_type=data["type"],
            source=data["source"],
            strength=data["strength"],
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )


class SignalIntegrationService:
    """
    Service for integrating trading signals from various sources
    and generating combined signals based on configurable rules.
    """
    
    def __init__(
        self,
        sentiment_weight: float = 0.3,
        technical_weight: float = 0.7,
        sentiment_threshold: float = 0.2,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        bollinger_band_threshold: float = 0.05,
        macd_signal_threshold: float = 0.0,
        alpha_vantage_api_key: Optional[str] = None
    ):
        """
        Initialize the signal integration service
        
        Args:
            sentiment_weight: Weight for sentiment signals (0.0 to 1.0)
            technical_weight: Weight for technical signals (0.0 to 1.0)
            sentiment_threshold: Threshold for sentiment signals (-1.0 to 1.0)
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            bollinger_band_threshold: Threshold for Bollinger Band signals
            macd_signal_threshold: Threshold for MACD signals
            alpha_vantage_api_key: API key for Alpha Vantage
        """
        # Normalize weights to sum to 1.0
        total_weight = sentiment_weight + technical_weight
        self.sentiment_weight = sentiment_weight / total_weight
        self.technical_weight = technical_weight / total_weight
        
        # Set thresholds
        self.sentiment_threshold = sentiment_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bollinger_band_threshold = bollinger_band_threshold
        self.macd_signal_threshold = macd_signal_threshold
        
        # Initialize clients and signal generators
        self.alpha_vantage_client = AlphaVantageClient(api_key=alpha_vantage_api_key)
        self.sentiment_signal_generator = SentimentSignalGenerator()
        
        # Signal cache
        self._signal_cache: Dict[str, List[TradingSignal]] = {}
        self._last_update: Dict[str, datetime.datetime] = {}
        
        # Cache expiry in seconds
        self.cache_expiry = 300  # 5 minutes
    
    async def get_technical_signals(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        window_size: int = 14
    ) -> List[TradingSignal]:
        """
        Generate technical trading signals from price data
        
        Args:
            symbol: Trading symbol
            price_data: DataFrame with OHLCV data
            window_size: Window size for technical indicators
            
        Returns:
            List of technical trading signals
        """
        signals = []
        now = datetime.datetime.now()
        
        # Ensure we have enough data
        if len(price_data) < window_size * 2:
            logger.warning(f"Not enough data for {symbol} to generate technical signals")
            return signals
        
        # Calculate technical indicators
        try:
            # Bollinger Bands
            upper, middle, lower = calculate_bollinger_bands(
                price_data['close'].values, window_size, 2.0
            )
            
            # RSI
            rsi_values = calculate_rsi(price_data['close'].values, window_size)
            
            # MACD
            macd_line, signal_line, histogram = calculate_macd(
                price_data['close'].values, 12, 26, 9
            )
            
            # Get the latest values
            latest_close = price_data['close'].iloc[-1]
            latest_upper = upper[-1]
            latest_lower = lower[-1]
            latest_rsi = rsi_values[-1]
            latest_macd = macd_line[-1]
            latest_signal = signal_line[-1]
            latest_histogram = histogram[-1]
            
            # Generate signals based on Bollinger Bands
            if latest_close > latest_upper * (1 + self.bollinger_band_threshold):
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        timestamp=now,
                        signal_type=SignalType.SELL,
                        source=SignalSource.TECHNICAL,
                        strength=min(
                            (latest_close - latest_upper) / (latest_upper * self.bollinger_band_threshold),
                            1.0
                        ),
                        description=f"Price above upper Bollinger Band ({latest_close:.2f} > {latest_upper:.2f})",
                        metadata={
                            "indicator": "bollinger_bands",
                            "upper": float(latest_upper),
                            "middle": float(middle[-1]),
                            "lower": float(latest_lower),
                            "close": float(latest_close)
                        }
                    )
                )
            elif latest_close < latest_lower * (1 - self.bollinger_band_threshold):
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        timestamp=now,
                        signal_type=SignalType.BUY,
                        source=SignalSource.TECHNICAL,
                        strength=min(
                            (latest_lower - latest_close) / (latest_lower * self.bollinger_band_threshold),
                            1.0
                        ),
                        description=f"Price below lower Bollinger Band ({latest_close:.2f} < {latest_lower:.2f})",
                        metadata={
                            "indicator": "bollinger_bands",
                            "upper": float(latest_upper),
                            "middle": float(middle[-1]),
                            "lower": float(latest_lower),
                            "close": float(latest_close)
                        }
                    )
                )
            
            # Generate signals based on RSI
            if latest_rsi > self.rsi_overbought:
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        timestamp=now,
                        signal_type=SignalType.SELL,
                        source=SignalSource.TECHNICAL,
                        strength=min((latest_rsi - self.rsi_overbought) / 30.0, 1.0),
                        description=f"RSI in overbought territory ({latest_rsi:.2f} > {self.rsi_overbought})",
                        metadata={
                            "indicator": "rsi",
                            "value": float(latest_rsi),
                            "overbought": self.rsi_overbought,
                            "oversold": self.rsi_oversold
                        }
                    )
                )
            elif latest_rsi < self.rsi_oversold:
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        timestamp=now,
                        signal_type=SignalType.BUY,
                        source=SignalSource.TECHNICAL,
                        strength=min((self.rsi_oversold - latest_rsi) / 30.0, 1.0),
                        description=f"RSI in oversold territory ({latest_rsi:.2f} < {self.rsi_oversold})",
                        metadata={
                            "indicator": "rsi",
                            "value": float(latest_rsi),
                            "overbought": self.rsi_overbought,
                            "oversold": self.rsi_oversold
                        }
                    )
                )
            
            # Generate signals based on MACD
            if latest_histogram > self.macd_signal_threshold and latest_histogram > histogram[-2]:
                # Bullish MACD crossover
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        timestamp=now,
                        signal_type=SignalType.BUY,
                        source=SignalSource.TECHNICAL,
                        strength=min(abs(latest_histogram) / 2.0, 1.0),
                        description=f"Bullish MACD crossover (histogram: {latest_histogram:.4f})",
                        metadata={
                            "indicator": "macd",
                            "macd_line": float(latest_macd),
                            "signal_line": float(latest_signal),
                            "histogram": float(latest_histogram)
                        }
                    )
                )
            elif latest_histogram < -self.macd_signal_threshold and latest_histogram < histogram[-2]:
                # Bearish MACD crossover
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        timestamp=now,
                        signal_type=SignalType.SELL,
                        source=SignalSource.TECHNICAL,
                        strength=min(abs(latest_histogram) / 2.0, 1.0),
                        description=f"Bearish MACD crossover (histogram: {latest_histogram:.4f})",
                        metadata={
                            "indicator": "macd",
                            "macd_line": float(latest_macd),
                            "signal_line": float(latest_signal),
                            "histogram": float(latest_histogram)
                        }
                    )
                )
        
        except Exception as e:
            logger.error(f"Error generating technical signals for {symbol}: {str(e)}")
        
        return signals
    
    async def get_sentiment_signals(
        self,
        symbol: str,
        topics: Optional[List[str]] = None
    ) -> List[TradingSignal]:
        """
        Generate sentiment trading signals from news and social media
        
        Args:
            symbol: Trading symbol
            topics: Optional list of topics to search for (e.g., "bitcoin", "crypto")
            
        Returns:
            List of sentiment trading signals
        """
        signals = []
        now = datetime.datetime.now()
        
        # Default topics based on symbol
        if topics is None:
            # Extract the base currency from the symbol (e.g., BTC from BTC/USDT)
            base_currency = symbol.split('/')[0].lower()
            
            # Map common symbols to search topics
            topic_map = {
                'btc': ['bitcoin', 'btc', 'crypto'],
                'eth': ['ethereum', 'eth', 'crypto'],
                'xrp': ['ripple', 'xrp', 'crypto'],
                'ada': ['cardano', 'ada', 'crypto'],
                'sol': ['solana', 'sol', 'crypto'],
                'doge': ['dogecoin', 'doge', 'crypto'],
                'dot': ['polkadot', 'dot', 'crypto'],
                'link': ['chainlink', 'link', 'crypto'],
                'uni': ['uniswap', 'uni', 'defi'],
                'aave': ['aave', 'defi'],
            }
            
            topics = topic_map.get(base_currency.lower(), ['crypto', 'blockchain'])
        
        try:
            # Fetch sentiment data from Alpha Vantage
            sentiment_data = await self.alpha_vantage_client.get_news_sentiment_by_topics(topics)
            
            if not sentiment_data or 'feed' not in sentiment_data:
                logger.warning(f"No sentiment data found for {symbol} with topics {topics}")
                return signals
            
            # Process sentiment data
            sentiment_scores = []
            for article in sentiment_data.get('feed', []):
                # Skip articles without sentiment
                if 'overall_sentiment_score' not in article:
                    continue
                
                # Get sentiment score (-1.0 to 1.0)
                sentiment_score = float(article['overall_sentiment_score'])
                sentiment_scores.append(sentiment_score)
                
                # Check if this article is specifically about our symbol
                relevance = 0.5  # Default relevance
                for ticker in article.get('ticker_sentiment', []):
                    if ticker.get('ticker', '').lower() == base_currency.lower():
                        # Use ticker-specific sentiment if available
                        ticker_sentiment = float(ticker.get('ticker_sentiment_score', sentiment_score))
                        sentiment_scores.append(ticker_sentiment)
                        relevance = float(ticker.get('relevance_score', 0.5))
            
            if not sentiment_scores:
                logger.warning(f"No sentiment scores found for {symbol} with topics {topics}")
                return signals
            
            # Calculate average sentiment score
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Generate signal based on sentiment
            if avg_sentiment > self.sentiment_threshold:
                signal_type = SignalType.BUY if avg_sentiment > 2 * self.sentiment_threshold else SignalType.STRONG_BUY
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        timestamp=now,
                        signal_type=signal_type,
                        source=SignalSource.SENTIMENT,
                        strength=min(abs(avg_sentiment), 1.0),
                        description=f"Positive sentiment detected ({avg_sentiment:.2f})",
                        metadata={
                            "sentiment_score": float(avg_sentiment),
                            "num_articles": len(sentiment_data.get('feed', [])),
                            "topics": topics
                        }
                    )
                )
            elif avg_sentiment < -self.sentiment_threshold:
                signal_type = SignalType.SELL if avg_sentiment > -2 * self.sentiment_threshold else SignalType.STRONG_SELL
                signals.append(
                    TradingSignal(
                        symbol=symbol,
                        timestamp=now,
                        signal_type=signal_type,
                        source=SignalSource.SENTIMENT,
                        strength=min(abs(avg_sentiment), 1.0),
                        description=f"Negative sentiment detected ({avg_sentiment:.2f})",
                        metadata={
                            "sentiment_score": float(avg_sentiment),
                            "num_articles": len(sentiment_data.get('feed', [])),
                            "topics": topics
                        }
                    )
                )
        
        except Exception as e:
            logger.error(f"Error generating sentiment signals for {symbol}: {str(e)}")
        
        return signals
    
    def _combine_signals(
        self,
        technical_signals: List[TradingSignal],
        sentiment_signals: List[TradingSignal]
    ) -> List[TradingSignal]:
        """
        Combine technical and sentiment signals to generate combined signals
        
        Args:
            technical_signals: List of technical signals
            sentiment_signals: List of sentiment signals
            
        Returns:
            List of combined signals
        """
        combined_signals = []
        now = datetime.datetime.now()
        
        # Group signals by symbol
        signals_by_symbol = {}
        
        for signal in technical_signals + sentiment_signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = {
                    'technical': [],
                    'sentiment': []
                }
            
            if signal.source == SignalSource.TECHNICAL:
                signals_by_symbol[signal.symbol]['technical'].append(signal)
            elif signal.source == SignalSource.SENTIMENT:
                signals_by_symbol[signal.symbol]['sentiment'].append(signal)
        
        # Process signals for each symbol
        for symbol, signals in signals_by_symbol.items():
            tech_signals = signals['technical']
            sent_signals = signals['sentiment']
            
            # Skip if we don't have both types of signals
            if not tech_signals or not sent_signals:
                continue
            
            # Calculate technical signal score
            tech_buy_strength = sum(s.strength for s in tech_signals if s.type in [SignalType.BUY, SignalType.STRONG_BUY])
            tech_sell_strength = sum(s.strength for s in tech_signals if s.type in [SignalType.SELL, SignalType.STRONG_SELL])
            tech_score = tech_buy_strength - tech_sell_strength
            
            # Calculate sentiment signal score
            sent_buy_strength = sum(s.strength for s in sent_signals if s.type in [SignalType.BUY, SignalType.STRONG_BUY])
            sent_sell_strength = sum(s.strength for s in sent_signals if s.type in [SignalType.SELL, SignalType.STRONG_SELL])
            sent_score = sent_buy_strength - sent_sell_strength
            
            # Calculate combined score with weights
            combined_score = (tech_score * self.technical_weight) + (sent_score * self.sentiment_weight)
            
            # Generate combined signal
            if combined_score > 0.5:
                signal_type = SignalType.STRONG_BUY
                strength = min(combined_score, 1.0)
                description = "Strong buy signal from combined technical and sentiment analysis"
            elif combined_score > 0.2:
                signal_type = SignalType.BUY
                strength = min(combined_score, 1.0)
                description = "Buy signal from combined technical and sentiment analysis"
            elif combined_score < -0.5:
                signal_type = SignalType.STRONG_SELL
                strength = min(abs(combined_score), 1.0)
                description = "Strong sell signal from combined technical and sentiment analysis"
            elif combined_score < -0.2:
                signal_type = SignalType.SELL
                strength = min(abs(combined_score), 1.0)
                description = "Sell signal from combined technical and sentiment analysis"
            else:
                signal_type = SignalType.NEUTRAL
                strength = 0.5
                description = "Neutral signal from combined technical and sentiment analysis"
            
            # Create combined signal
            combined_signals.append(
                TradingSignal(
                    symbol=symbol,
                    timestamp=now,
                    signal_type=signal_type,
                    source=SignalSource.COMBINED,
                    strength=strength,
                    description=description,
                    metadata={
                        "technical_score": float(tech_score),
                        "sentiment_score": float(sent_score),
                        "combined_score": float(combined_score),
                        "technical_weight": self.technical_weight,
                        "sentiment_weight": self.sentiment_weight
                    }
                )
            )
        
        return combined_signals
    
    async def get_signals(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        force_refresh: bool = False,
        topics: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all trading signals for a symbol
        
        Args:
            symbol: Trading symbol
            price_data: DataFrame with OHLCV data
            force_refresh: Force refresh of cached signals
            topics: Optional list of topics for sentiment analysis
            
        Returns:
            Dictionary with technical, sentiment, and combined signals
        """
        cache_key = f"{symbol}-{datetime.datetime.now().strftime('%Y-%m-%d')}"
        
        # Check cache
        if not force_refresh and cache_key in self._signal_cache:
            last_update = self._last_update.get(cache_key)
            if last_update and (datetime.datetime.now() - last_update).total_seconds() < self.cache_expiry:
                # Return cached signals
                return {
                    "technical": [s.to_dict() for s in self._signal_cache[cache_key] if s.source == SignalSource.TECHNICAL],
                    "sentiment": [s.to_dict() for s in self._signal_cache[cache_key] if s.source == SignalSource.SENTIMENT],
                    "combined": [s.to_dict() for s in self._signal_cache[cache_key] if s.source == SignalSource.COMBINED]
                }
        
        # Generate new signals
        technical_signals = await self.get_technical_signals(symbol, price_data)
        sentiment_signals = await self.get_sentiment_signals(symbol, topics)
        combined_signals = self._combine_signals(technical_signals, sentiment_signals)
        
        # Update cache
        all_signals = technical_signals + sentiment_signals + combined_signals
        self._signal_cache[cache_key] = all_signals
        self._last_update[cache_key] = datetime.datetime.now()
        
        # Return signals
        return {
            "technical": [s.to_dict() for s in technical_signals],
            "sentiment": [s.to_dict() for s in sentiment_signals],
            "combined": [s.to_dict() for s in combined_signals]
        }
    
    async def get_signals_for_multiple_symbols(
        self,
        symbols: List[str],
        price_data_dict: Dict[str, pd.DataFrame],
        force_refresh: bool = False
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Get trading signals for multiple symbols
        
        Args:
            symbols: List of trading symbols
            price_data_dict: Dictionary mapping symbols to price DataFrames
            force_refresh: Force refresh of cached signals
            
        Returns:
            Dictionary mapping symbols to signal dictionaries
        """
        results = {}
        
        # Process each symbol
        for symbol in symbols:
            if symbol in price_data_dict:
                signals = await self.get_signals(
                    symbol,
                    price_data_dict[symbol],
                    force_refresh
                )
                results[symbol] = signals
            else:
                logger.warning(f"No price data found for {symbol}")
        
        return results


# API endpoint functions
async def get_signals_for_symbol(
    symbol: str,
    price_data: pd.DataFrame,
    service: Optional[SignalIntegrationService] = None,
    force_refresh: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    API function to get signals for a symbol
    
    Args:
        symbol: Trading symbol
        price_data: DataFrame with OHLCV data
        service: Optional SignalIntegrationService instance
        force_refresh: Force refresh of cached signals
        
    Returns:
        Dictionary with technical, sentiment, and combined signals
    """
    if service is None:
        service = SignalIntegrationService()
    
    return await service.get_signals(symbol, price_data, force_refresh)


async def get_signals_for_multiple_symbols(
    symbols: List[str],
    price_data_dict: Dict[str, pd.DataFrame],
    service: Optional[SignalIntegrationService] = None,
    force_refresh: bool = False
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    API function to get signals for multiple symbols
    
    Args:
        symbols: List of trading symbols
        price_data_dict: Dictionary mapping symbols to price DataFrames
        service: Optional SignalIntegrationService instance
        force_refresh: Force refresh of cached signals
        
    Returns:
        Dictionary mapping symbols to signal dictionaries
    """
    if service is None:
        service = SignalIntegrationService()
    
    return await service.get_signals_for_multiple_symbols(symbols, price_data_dict, force_refresh)
