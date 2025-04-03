"""NLP service for sentiment analysis.

This module provides natural language processing functionality
for sentiment analysis of text data from various sources.
"""

import asyncio
from typing import Dict, List, Optional, Any
import numpy as np

from src.common.config import config
from src.common.logging import get_logger


class NLPService:
    """Service for natural language processing tasks.
    
    This service provides NLP functionality for sentiment analysis,
    including text classification and entity recognition.
    """
    
    def __init__(self):
        """Initialize the NLP service."""
        self.logger = get_logger("analysis_agents", "nlp_service")
        
        # Load configuration
        self.model_name = config.get("nlp.sentiment_model", "distilbert-base-uncased-finetuned-sst-2-english")
        self.batch_size = config.get("nlp.batch_size", 16)
        
        # Initialize sentiment pipeline
        self.sentiment_pipeline = None
        
        # Load sentiment lexicons as fallback
        self._load_sentiment_lexicons()
    
    async def initialize(self) -> None:
        """Initialize the NLP models."""
        self.logger.info("Initializing NLP service")
        
        # Load sentiment analysis model
        try:
            # Import here to avoid requiring transformers for the whole system
            # Only users who want NLP functionality need the dependency
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Create a more specialized pipeline for financial/crypto sentiment if available
            # Use a model specifically fine-tuned for financial sentiment if possible
            try:
                # Try to use a financial sentiment model first
                financial_model_name = "yiyanghkust/finbert-tone"
                self.logger.info(f"Attempting to load financial sentiment model: {financial_model_name}")
                
                # Load the tokenizer and model
                tokenizer = await loop.run_in_executor(
                    None,
                    lambda: AutoTokenizer.from_pretrained(financial_model_name)
                )
                
                model = await loop.run_in_executor(
                    None,
                    lambda: AutoModelForSequenceClassification.from_pretrained(financial_model_name)
                )
                
                # Create a pipeline with the financial model
                self.sentiment_pipeline = await loop.run_in_executor(
                    None,
                    lambda: pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                )
                
                self.logger.info("Loaded financial sentiment analysis model", model=financial_model_name)
                
            except Exception as financial_error:
                # Fall back to general sentiment model if financial model fails
                self.logger.warning(
                    "Failed to load financial sentiment model, falling back to general model",
                    error=str(financial_error)
                )
                
                self.sentiment_pipeline = await loop.run_in_executor(
                    None, 
                    lambda: pipeline("sentiment-analysis", model=self.model_name)
                )
                self.logger.info("Loaded general sentiment analysis model", model=self.model_name)
                
        except Exception as e:
            self.logger.error("Failed to load sentiment analysis model", error=str(e))
            self.logger.info("Using lexicon-based approach as fallback")
            # Ensure sentiment lexicons are loaded
            self._load_sentiment_lexicons()
    
    def _load_sentiment_lexicons(self) -> None:
        """Load sentiment lexicons for text analysis."""
        self.logger.info("Loading sentiment lexicons as fallback")
        
        # Bullish words/phrases with weights
        self.bullish_words = {
            "bullish": 0.8, "buy": 0.7, "long": 0.6, "potential": 0.5, "upside": 0.7, 
            "green": 0.6, "higher": 0.6, "surge": 0.8, "rally": 0.8, "moon": 1.0, 
            "strong": 0.7, "growth": 0.6, "breakout": 0.9, "outperform": 0.8, 
            "upgrade": 0.7, "accumulate": 0.7, "support": 0.5, "bottom": 0.7, 
            "opportunity": 0.6, "bullrun": 0.9, "pump": 0.8, "peak": 0.7,
            "profit": 0.7, "gain": 0.6, "hodl": 0.7, "winner": 0.7,
            "confidence": 0.6, "success": 0.6, "momentum": 0.7, "hopium": 0.9
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
    
    async def analyze_sentiment(self, texts: List[str]) -> List[float]:
        """Analyze sentiment of text.
        
        Args:
            texts: List of text to analyze
            
        Returns:
            List of sentiment scores (0-1 scale)
        """
        if not texts:
            return []
            
        if self.sentiment_pipeline:
            return await self._analyze_with_model(texts)
        else:
            return self._analyze_with_lexicon(texts)
    
    async def _analyze_with_model(self, texts: List[str]) -> List[float]:
        """Analyze sentiment using the transformer model.
        
        Args:
            texts: List of text to analyze
            
        Returns:
            List of sentiment scores (0-1 scale)
        """
        try:
            # Process in batches to avoid memory issues
            results = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                
                # Run in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                batch_results = await loop.run_in_executor(
                    None, 
                    lambda: self.sentiment_pipeline(batch)
                )
                
                # Convert to sentiment scores (0-1 scale)
                for result in batch_results:
                    label = result["label"]
                    score = result["score"]
                    
                    # Handle different model output formats
                    if label in ["POSITIVE", "LABEL_1"]:
                        sentiment = score
                    elif label in ["NEGATIVE", "LABEL_0"]:
                        sentiment = 1.0 - score
                    elif label == "NEUTRAL":
                        sentiment = 0.5
                    elif isinstance(label, int):
                        # Handle numeric labels (0=negative, 1=positive)
                        sentiment = score if label == 1 else 1.0 - score
                    else:
                        sentiment = 0.5
                        
                    results.append(sentiment)
            
            # Apply market-specific sentiment calibration
            # Crypto sentiment can be more extreme than general text sentiment
            calibrated_results = []
            for score in results:
                # Apply a slight sigmoid curve to emphasize extremes
                if score > 0.5:
                    calibrated = 0.5 + (score - 0.5) * 1.25  # Enhance positive scores
                elif score < 0.5:
                    calibrated = 0.5 - (0.5 - score) * 1.25  # Enhance negative scores
                else:
                    calibrated = 0.5
                
                # Clip to valid range
                calibrated = max(0.0, min(1.0, calibrated))
                calibrated_results.append(calibrated)
            
            return calibrated_results
            
        except Exception as e:
            self.logger.error("Error in model-based sentiment analysis", error=str(e))
            # Fall back to lexicon-based approach
            return self._analyze_with_lexicon(texts)
    
    def _analyze_with_lexicon(self, texts: List[str]) -> List[float]:
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
    
    async def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Text to analyze
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        # In a production system, this would use a more sophisticated approach
        # For now, implement a simple frequency-based keyword extraction
        
        # Remove common stop words (a very simplified list)
        stop_words = ["the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
                     "in", "on", "at", "to", "for", "with", "by", "about", "as", "of"]
        
        # Tokenize and filter
        words = text.lower().split()
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_n]]