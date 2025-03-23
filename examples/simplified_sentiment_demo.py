"""
Simplified Sentiment Analysis Demo

This script demonstrates how the sentiment analysis system works without external dependencies.
"""

import os
import datetime
import random
import time

# Mock core components
class EventBus:
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event_type, event):
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(event)

class SentimentEvent:
    def __init__(self, source, payload):
        self.source = source
        self.payload = payload

class Signal:
    def __init__(self, symbol, signal_type, direction, price, confidence, reason="", metadata=None):
        self.symbol = symbol
        self.signal_type = signal_type
        self.direction = direction
        self.price = price
        self.confidence = confidence
        self.reason = reason
        self.metadata = metadata or {}

# Global event bus instance
event_bus = EventBus()

# Mock sentiment analysis components
class SocialMediaSentimentAgent:
    def __init__(self):
        self.name = "social_media_sentiment"
    
    def analyze(self, symbol):
        sentiment_value = random.uniform(0.3, 0.7)
        confidence = random.uniform(0.6, 0.9)
        
        if sentiment_value > 0.55:
            direction = "bullish"
        elif sentiment_value < 0.45:
            direction = "bearish"
        else:
            direction = "neutral"
            
        print(f"[Social Media] {symbol}: {direction} sentiment ({sentiment_value:.2f}) with {confidence:.2f} confidence")
        
        return {
            "value": sentiment_value,
            "direction": direction,
            "confidence": confidence
        }

class NewsSentimentAgent:
    def __init__(self):
        self.name = "news_sentiment"
    
    def analyze(self, symbol):
        sentiment_value = random.uniform(0.2, 0.8)
        confidence = random.uniform(0.6, 0.9)
        
        if sentiment_value > 0.55:
            direction = "bullish"
        elif sentiment_value < 0.45:
            direction = "bearish"
        else:
            direction = "neutral"
            
        print(f"[News] {symbol}: {direction} sentiment ({sentiment_value:.2f}) with {confidence:.2f} confidence")
        
        return {
            "value": sentiment_value,
            "direction": direction,
            "confidence": confidence
        }

class MarketSentimentAgent:
    def __init__(self):
        self.name = "market_sentiment"
    
    def analyze(self, symbol):
        sentiment_value = random.uniform(0.3, 0.7)
        confidence = random.uniform(0.7, 0.95)
        
        if sentiment_value > 0.55:
            direction = "bullish"
        elif sentiment_value < 0.45:
            direction = "bearish"
        else:
            direction = "neutral"
            
        print(f"[Market] {symbol}: {direction} sentiment ({sentiment_value:.2f}) with {confidence:.2f} confidence")
        
        return {
            "value": sentiment_value,
            "direction": direction,
            "confidence": confidence
        }

class SentimentAggregator:
    def __init__(self):
        self.name = "sentiment_aggregator"
        self.agents = [
            SocialMediaSentimentAgent(),
            NewsSentimentAgent(),
            MarketSentimentAgent()
        ]
        self.source_weights = {
            "social_media_sentiment": 0.25,
            "news_sentiment": 0.25,
            "market_sentiment": 0.5
        }
    
    def aggregate(self, symbol):
        results = {}
        
        # Collect results from all agents
        for agent in self.agents:
            results[agent.name] = agent.analyze(symbol)
        
        # Calculate weighted sentiment
        weighted_sum = 0
        total_weight = 0
        total_confidence = 0
        
        for source, data in results.items():
            weight = self.source_weights.get(source, 0.25)
            weighted_sum += data["value"] * weight
            total_weight += weight
            total_confidence += data["confidence"]
        
        # Calculate aggregated values
        if total_weight > 0:
            agg_value = weighted_sum / total_weight
            agg_confidence = total_confidence / len(results)
            
            if agg_value > 0.55:
                agg_direction = "bullish"
            elif agg_value < 0.45:
                agg_direction = "bearish"
            else:
                agg_direction = "neutral"
                
            print(f"\n[Aggregator] {symbol}: {agg_direction} sentiment ({agg_value:.2f}) with {agg_confidence:.2f} confidence")
            
            return {
                "value": agg_value,
                "direction": agg_direction,
                "confidence": agg_confidence
            }
        
        return {
            "value": 0.5,
            "direction": "neutral",
            "confidence": 0.5
        }

# Mock enhanced sentiment strategy
class EnhancedSentimentStrategy:
    def __init__(self):
        self.name = "enhanced_sentiment_strategy"
        self.aggregator = SentimentAggregator()
        self.sentiment_threshold_bullish = 0.6
        self.sentiment_threshold_bearish = 0.4
        self.min_confidence = 0.7
        self.use_market_regime = True
        self.use_technical_confirmation = True
        self.price_data = {}
        
        # Subscribe to signals
        event_bus.subscribe("signal", self.handle_signal)
    
    def get_market_regime(self, symbol):
        # Randomly select market regime
        return random.choice(["bullish", "bearish", "neutral", "volatile"])
    
    def get_technical_indicators(self, symbol):
        # Generate random RSI value
        return {
            "rsi": random.uniform(30, 70)
        }
    
    def handle_signal(self, signal):
        print(f"\n[Signal] {signal.symbol}: {signal.signal_type} {signal.direction}")
        print(f"  Price: {signal.price:.2f}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Reason: {signal.reason}")
        
        if signal.metadata:
            print("  Metadata:")
            for key, value in signal.metadata.items():
                print(f"    {key}: {value}")
    
    def analyze_sentiment(self, symbol, price):
        print(f"\n--- Analyzing sentiment for {symbol} (Price: {price:.2f}) ---")
        
        # Store price data
        self.price_data[symbol] = price
        
        # Get sentiment from aggregator
        sentiment = self.aggregator.aggregate(symbol)
        
        # Get market regime
        if self.use_market_regime:
            regime = self.get_market_regime(symbol)
            print(f"[Market Regime] {symbol}: {regime}")
        else:
            regime = "neutral"
        
        # Get technical indicators
        if self.use_technical_confirmation:
            indicators = self.get_technical_indicators(symbol)
            print(f"[Technical] {symbol}: RSI = {indicators['rsi']:.2f}")
        else:
            indicators = {"rsi": 50}
        
        # Check sentiment thresholds
        if sentiment["value"] >= self.sentiment_threshold_bullish:
            direction = "bullish"
            signal_type = "ENTRY"
            signal_direction = "long"
        elif sentiment["value"] <= self.sentiment_threshold_bearish:
            direction = "bearish"
            signal_type = "ENTRY"
            signal_direction = "short"
        else:
            print(f"[Strategy] No signal: Sentiment value {sentiment['value']:.2f} within neutral range")
            return None
        
        # Check confidence
        if sentiment["confidence"] < self.min_confidence:
            print(f"[Strategy] No signal: Confidence {sentiment['confidence']:.2f} below threshold {self.min_confidence}")
            return None
        
        # Check market regime alignment
        if self.use_market_regime:
            regime_aligned = True
            if signal_direction == "long" and regime == "bearish":
                regime_aligned = False
            elif signal_direction == "short" and regime == "bullish":
                regime_aligned = False
                
            if not regime_aligned:
                print(f"[Strategy] No signal: {signal_direction} signal not aligned with {regime} regime")
                return None
        
        # Check technical alignment
        if self.use_technical_confirmation:
            technical_aligned = True
            rsi = indicators["rsi"]
            
            if signal_direction == "long" and rsi > 70:
                technical_aligned = False
            elif signal_direction == "short" and rsi < 30:
                technical_aligned = False
                
            if not technical_aligned:
                print(f"[Strategy] No signal: {signal_direction} signal not aligned with RSI {rsi:.2f}")
                return None
        
        # Calculate signal score
        signal_score = sentiment["value"] if signal_direction == "long" else (1 - sentiment["value"])
        signal_score *= sentiment["confidence"]
        
        print(f"[Strategy] Generated {direction} signal")
        print(f"  Signal Type: {signal_type}")
        print(f"  Direction: {signal_direction}")
        print(f"  Sentiment Value: {sentiment['value']:.2f}")
        print(f"  Confidence: {sentiment['confidence']:.2f}")
        print(f"  Score: {signal_score:.2f}")
        
        # Create signal
        reason_parts = [f"Sentiment signal: {direction}"]
        if self.use_market_regime:
            reason_parts.append(f"[Regime: {regime}]")
        if self.use_technical_confirmation:
            reason_parts.append(f"[RSI: {indicators['rsi']:.1f}]")
            
        reason = " ".join(reason_parts)
        
        signal = Signal(
            symbol=symbol,
            signal_type=signal_type,
            direction=signal_direction,
            price=price,
            confidence=sentiment["confidence"],
            reason=reason,
            metadata={
                "sentiment_value": sentiment["value"],
                "sentiment_direction": direction,
                "signal_score": signal_score,
                "regime": regime,
                "rsi": indicators["rsi"]
            }
        )
        
        # Publish signal
        event_bus.publish("signal", signal)
        
        return signal

# Run the demo
def run_demo():
    print("=== Enhanced Sentiment Strategy Demo ===\n")
    
    strategy = EnhancedSentimentStrategy()
    
    # Define symbols to analyze
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    # Run for a few iterations
    for i in range(3):
        print(f"\n=== Iteration {i+1} ===")
        
        for symbol in symbols:
            # Generate random price
            price = 0
            if "BTC" in symbol:
                price = random.uniform(25000, 35000)
            elif "ETH" in symbol:
                price = random.uniform(1500, 2500)
            elif "SOL" in symbol:
                price = random.uniform(50, 150)
            
            # Analyze sentiment and generate signals
            strategy.analyze_sentiment(symbol, price)
            
            # Small delay between symbols
            time.sleep(1)
        
        # Delay between iterations
        if i < 2:
            print("\nWaiting for next iteration...")
            time.sleep(3)
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_demo()