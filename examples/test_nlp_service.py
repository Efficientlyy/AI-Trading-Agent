#!/usr/bin/env python
"""
Test script for NLP Service

This script tests the lexicon-based sentiment analysis functionality
without requiring external dependencies.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleNLP:
    """Simple NLP service for testing."""
    
    def __init__(self):
        """Initialize the simple NLP service."""
        self.logger = logging.getLogger("simple_nlp")
        
        # Load lexicons
        self._load_sentiment_lexicons()
    
    def _load_sentiment_lexicons(self):
        """Load sentiment lexicons for text analysis."""
        self.logger.info("Loading sentiment lexicons")
        
        # Bullish words/phrases with weights
        self.bullish_words = {
            "bullish": 0.8, "buy": 0.7, "long": 0.6, "potential": 0.5, "upside": 0.7, 
            "green": 0.6, "higher": 0.6, "surge": 0.8, "rally": 0.8, "moon": 1.0, 
            "strong": 0.7, "growth": 0.6, "breakout": 0.9, "outperform": 0.8, 
            "upgrade": 0.7, "accumulate": 0.7, "support": 0.5, "bottom": 0.7, 
            "opportunity": 0.6, "bullrun": 0.9, "pump": 0.8, "peak": 0.7,
            "profit": 0.7, "gain": 0.6, "hodl": 0.7, "winner": 0.7,
            "confidence": 0.6, "success": 0.6, "momentum": 0.7, "hopium": 0.9,
            "fundamentals": 0.6, "great": 0.7
        }
        
        # Bearish words/phrases with weights
        self.bearish_words = {
            "bearish": 0.8, "sell": 0.7, "short": 0.6, "downside": 0.7, "red": 0.6, 
            "lower": 0.6, "drop": 0.7, "fall": 0.7, "dump": 0.9, "weak": 0.7, 
            "decline": 0.7, "breakdown": 0.8, "underperform": 0.8, "downgrade": 0.7, 
            "distribute": 0.6, "resistance": 0.6, "top": 0.5, "risk": 0.6, 
            "crash": 1.0, "correction": 0.7, "fud": 0.8, "trouble": 0.7,
            "loss": 0.7, "bear": 0.7, "failure": 0.8, "bubble": 0.8,
            "panic": 0.9, "fear": 0.8, "scam": 0.9, "capitulate": 0.9
        }
        
        # Modifiers that intensify or reduce sentiment
        self.modifiers = {
            "very": 1.5, "extremely": 2.0, "slightly": 0.5, "somewhat": 0.7,
            "huge": 1.7, "major": 1.5, "massive": 1.8, "significant": 1.4,
            "not": -1.0, "no": -1.0, "hardly": -0.7, "barely": -0.8,
            "absolute": 1.5, "complete": 1.4, "total": 1.5
        }
    
    def analyze_with_lexicon(self, texts):
        """Analyze sentiment using lexicon-based approach.
        
        Args:
            texts: List of text to analyze
            
        Returns:
            List of sentiment scores (0-1 scale)
        """
        sentiment_scores = []
        
        for text in texts:
            text_lower = text.lower()
            words = text_lower.split()
            
            # Calculate weighted sentiment
            bullish_score = 0.0
            bearish_score = 0.0
            
            # Check for bullish and bearish words with their weights
            for word, weight in self.bullish_words.items():
                if word in text_lower:
                    # Look for modifiers before the word
                    for i, w in enumerate(words):
                        if w == word and i > 0:
                            prev_word = words[i-1]
                            if prev_word in self.modifiers:
                                modifier = self.modifiers[prev_word]
                                # Handle negations
                                if modifier < 0:
                                    bearish_score += weight * abs(modifier)
                                else:
                                    bullish_score += weight * modifier
                            else:
                                bullish_score += weight
                        elif w == word:
                            bullish_score += weight
            
            for word, weight in self.bearish_words.items():
                if word in text_lower:
                    # Look for modifiers before the word
                    for i, w in enumerate(words):
                        if w == word and i > 0:
                            prev_word = words[i-1]
                            if prev_word in self.modifiers:
                                modifier = self.modifiers[prev_word]
                                # Handle negations
                                if modifier < 0:
                                    bullish_score += weight * abs(modifier)
                                else:
                                    bearish_score += weight * modifier
                            else:
                                bearish_score += weight
                        elif w == word:
                            bearish_score += weight
            
            # Calculate sentiment score
            if bullish_score + bearish_score > 0:
                sentiment = bullish_score / (bullish_score + bearish_score)
                
                # Adjust sentiment based on text length and sentiment intensity
                intensity = (bullish_score + bearish_score) / len(words) if words else 0
                
                # Apply intensity modulation - more intense sentiment is more meaningful
                if intensity > 0.3:  # High intensity threshold
                    # Enhance the signal - push away from neutral
                    if sentiment > 0.5:
                        sentiment = 0.5 + (sentiment - 0.5) * 1.3
                    else:
                        sentiment = 0.5 - (0.5 - sentiment) * 1.3
                
                # Clip to valid range
                sentiment = max(0.0, min(1.0, sentiment))
            else:
                sentiment = 0.5  # Neutral if no sentiment words
                
            sentiment_scores.append(sentiment)
            
        return sentiment_scores


def main():
    """Run the test."""
    print("Testing NLP sentiment analysis with lexicon approach")
    print("-" * 50)
    
    nlp = SimpleNLP()
    
    # Test sentences
    texts = [
        "I am very bullish on Bitcoin right now, the fundamentals look great!",
        "BTC is looking extremely bearish, I would not buy at these levels.",
        "The market is uncertain, could go either way from here.",
        "This pump is not sustainable, expect a correction soon.",
        "Not bearish at all, actually quite bullish on the long term prospects."
    ]
    
    # Analyze sentiment
    results = nlp.analyze_with_lexicon(texts)
    
    # Print results
    for i, text in enumerate(texts):
        score = results[i]
        sentiment = "bullish" if score > 0.6 else "bearish" if score < 0.4 else "neutral"
        print(f"\nText: {text}")
        print(f"Score: {score:.2f} - {sentiment} sentiment")
    
    print("\n" + "-" * 50)
    print("NLP test completed")


if __name__ == "__main__":
    main()