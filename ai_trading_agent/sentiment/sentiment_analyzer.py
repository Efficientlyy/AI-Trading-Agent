"""
Sentiment Analyzer for the AI Trading Agent.

This module provides utilities to analyze sentiment from various text sources
including news, social media, and financial reports.
"""

import logging
import re
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import time
from collections import defaultdict

# Import NLP libraries
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import textblob
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

from ai_trading_agent.common import logger


class SentimentAnalyzer:
    """
    Analyzes sentiment from text data using multiple underlying methods.
    
    Features:
    - Financial domain-specific sentiment analysis
    - Multiple analysis methods (rule-based, ML-based)
    - Specialized dictionaries for financial terms
    - Confidence scores for sentiment predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration dictionary containing:
                - financial_dict_path: Path to financial sentiment dictionary
                - use_nltk: Whether to use NLTK for sentiment analysis
                - use_textblob: Whether to use TextBlob for sentiment analysis
                - use_custom: Whether to use custom financial dictionary
                - method_weights: Weights for each analysis method
                - cache_enabled: Whether to enable sentiment caching
                - cache_duration: Duration to cache results (seconds)
        """
        self.name = "SentimentAnalyzer"
        
        # Load configuration
        self.financial_dict_path = config.get('financial_dict_path', None)
        self.use_nltk = config.get('use_nltk', NLTK_AVAILABLE)
        self.use_textblob = config.get('use_textblob', TEXTBLOB_AVAILABLE)
        self.use_custom = config.get('use_custom', True)
        
        # Weights for different methods
        self.method_weights = config.get('method_weights', {
            'nltk': 0.4,
            'textblob': 0.3,
            'custom': 0.3
        })
        
        # Caching settings
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_duration = config.get('cache_duration', 3600)  # 1 hour
        self.sentiment_cache = {}
        
        # Initialize NLTK analyzer if available
        self.nltk_analyzer = SentimentIntensityAnalyzer() if self.use_nltk and NLTK_AVAILABLE else None
        
        # Load financial sentiment dictionary
        self.financial_dict = self._load_financial_dictionary()
        
        # Initialize performance tracking
        self.analyzed_count = 0
        self.execution_times = []
        
        logger.info(f"Initialized {self.name} with NLTK: {self.use_nltk}, TextBlob: {self.use_textblob}, "
                   f"Custom: {self.use_custom}")
    
    def _load_financial_dictionary(self) -> Dict[str, float]:
        """Load the financial sentiment dictionary from file."""
        if not self.financial_dict_path or not self.use_custom:
            # Use a minimal built-in dictionary if no path provided
            return {
                # Positive financial terms
                "beat": 0.6, "exceeded": 0.7, "growth": 0.6, "profit": 0.7, "profitability": 0.7,
                "rally": 0.6, "upgrade": 0.7, "outperform": 0.7, "bullish": 0.8, "upside": 0.6,
                "strong": 0.5, "positive": 0.6, "gain": 0.5, "gains": 0.5, "higher": 0.4,
                
                # Negative financial terms
                "miss": -0.6, "decline": -0.5, "loss": -0.7, "losses": -0.7, "downgrade": -0.7,
                "underperform": -0.7, "bearish": -0.8, "downside": -0.6, "weak": -0.5,
                "negative": -0.6, "lower": -0.4, "risk": -0.4, "sell": -0.5, "volatile": -0.4,
                
                # Earnings-related terms
                "earnings beat": 0.8, "earnings miss": -0.8, "raised guidance": 0.8,
                "lowered guidance": -0.8, "in-line": 0.2, "dividend cut": -0.7,
                "dividend increase": 0.7
            }
        
        try:
            with open(self.financial_dict_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load financial dictionary: {e}")
            return {}
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of sentiment scores:
                - positive: Score for positive sentiment (0-1)
                - negative: Score for negative sentiment (0-1)
                - neutral: Score for neutral sentiment (0-1)
                - compound: Overall compound score (-1 to 1)
                - confidence: Confidence in the analysis (0-1)
        """
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = hash(text)
            if cache_key in self.sentiment_cache:
                cached_result, timestamp = self.sentiment_cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return cached_result
        
        start_time = time.time()
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        # Apply different analysis methods
        results = {}
        
        if self.use_nltk and self.nltk_analyzer:
            results['nltk'] = self._analyze_with_nltk(cleaned_text)
        
        if self.use_textblob and TEXTBLOB_AVAILABLE:
            results['textblob'] = self._analyze_with_textblob(cleaned_text)
        
        if self.use_custom:
            results['custom'] = self._analyze_with_custom_dict(cleaned_text)
        
        # Combine results with weights
        final_result = self._combine_results(results)
        
        # Track performance
        self.analyzed_count += 1
        execution_time = time.time() - start_time
        self.execution_times.append(execution_time)
        
        # Cache result
        if self.cache_enabled:
            self.sentiment_cache[cache_key] = (final_result, time.time())
        
        return final_result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters while preserving sentence structure
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _analyze_with_nltk(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using NLTK's VADER."""
        scores = self.nltk_analyzer.polarity_scores(text)
        
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    
    def _analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert TextBlob's -1 to 1 scale to our format
        if polarity > 0:
            positive = polarity
            negative = 0
            neutral = 1.0 - positive
        elif polarity < 0:
            positive = 0
            negative = abs(polarity)
            neutral = 1.0 - negative
        else:
            positive = 0
            negative = 0
            neutral = 1.0
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'compound': polarity,
            'subjectivity': subjectivity
        }
    
    def _analyze_with_custom_dict(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using custom financial dictionary."""
        positive_score = 0.0
        negative_score = 0.0
        match_count = 0
        
        # Tokenize text into words
        words = text.split()
        
        # Check for multi-word terms first
        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i+1]
            if bigram in self.financial_dict:
                score = self.financial_dict[bigram]
                if score > 0:
                    positive_score += score
                else:
                    negative_score += abs(score)
                match_count += 1
        
        # Now check individual words
        for word in words:
            if word in self.financial_dict:
                score = self.financial_dict[word]
                if score > 0:
                    positive_score += score
                else:
                    negative_score += abs(score)
                match_count += 1
        
        # Normalize scores
        if match_count > 0:
            positive_score /= match_count
            negative_score /= match_count
        
        # Calculate neutral and compound
        total = positive_score + negative_score
        if total > 0:
            neutral_score = 1.0 - (total / 2)  # Scale down to keep total reasonable
            compound = (positive_score - negative_score) / total
        else:
            neutral_score = 1.0
            compound = 0.0
        
        return {
            'positive': min(positive_score, 1.0),
            'negative': min(negative_score, 1.0),
            'neutral': max(0.0, min(neutral_score, 1.0)),
            'compound': compound,
            'matches': match_count
        }
    
    def _combine_results(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Combine results from different methods using weights."""
        # Default values
        combined = {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'compound': 0.0,
            'confidence': 0.0
        }
        
        total_weight = 0.0
        
        # Combine weighted scores
        for method, scores in results.items():
            weight = self.method_weights.get(method, 0.0)
            total_weight += weight
            
            for key in ['positive', 'negative', 'neutral', 'compound']:
                if key in scores:
                    combined[key] += scores[key] * weight
        
        # Normalize by total weight
        if total_weight > 0:
            for key in ['positive', 'negative', 'neutral', 'compound']:
                combined[key] /= total_weight
        
        # Calculate confidence based on agreement between methods
        if len(results) > 1:
            # Get compound scores from each method
            compounds = [scores.get('compound', 0.0) for scores in results.values()]
            
            # Calculate standard deviation of compound scores
            std_dev = np.std(compounds) if len(compounds) > 0 else 1.0
            
            # Higher agreement (lower std dev) = higher confidence
            combined['confidence'] = max(0.0, min(1.0, 1.0 - std_dev))
        else:
            # If only one method, use a medium confidence
            combined['confidence'] = 0.7
        
        return combined
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        return [self.analyze_text(text) for text in texts]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the sentiment analyzer.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.execution_times:
            return {
                'analyzed_count': 0,
                'avg_execution_time': 0,
                'min_execution_time': 0,
                'max_execution_time': 0
            }
            
        return {
            'analyzed_count': self.analyzed_count,
            'avg_execution_time': np.mean(self.execution_times),
            'min_execution_time': np.min(self.execution_times),
            'max_execution_time': np.max(self.execution_times),
            'cache_size': len(self.sentiment_cache) if self.cache_enabled else 0
        }
    
    def clear_cache(self):
        """Clear the sentiment analysis cache."""
        self.sentiment_cache = {}
        logger.info(f"Cleared sentiment analysis cache")
    
    def update_financial_dictionary(self, new_terms: Dict[str, float]):
        """
        Update the financial sentiment dictionary with new terms.
        
        Args:
            new_terms: Dictionary mapping terms to sentiment scores
        """
        self.financial_dict.update(new_terms)
        logger.info(f"Updated financial dictionary with {len(new_terms)} new terms")
        
        # Save updated dictionary if path is provided
        if self.financial_dict_path:
            try:
                with open(self.financial_dict_path, 'w') as f:
                    json.dump(self.financial_dict, f)
            except Exception as e:
                logger.error(f"Failed to save updated financial dictionary: {e}")
    
    def get_sentiment_trend(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment trend across a series of texts.
        
        Args:
            texts: List of texts in chronological order
            
        Returns:
            Dictionary with sentiment trend analysis
        """
        if not texts:
            return {'trend': 'neutral', 'data': []}
            
        # Analyze all texts
        results = self.analyze_batch(texts)
        
        # Extract compound scores
        compounds = [r['compound'] for r in results]
        
        # Calculate trend metrics
        mean = np.mean(compounds)
        
        # Detect trend direction
        if len(compounds) > 1:
            # Simple linear regression for trend slope
            x = np.arange(len(compounds))
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, compounds, rcond=None)[0]
            
            if slope > 0.05:
                trend = 'improving'
            elif slope < -0.05:
                trend = 'deteriorating'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            slope = 0
        
        # Categorize overall sentiment
        if mean > 0.2:
            sentiment = 'positive'
        elif mean < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'trend': trend,
            'sentiment': sentiment,
            'slope': slope,
            'mean': mean,
            'volatility': np.std(compounds),
            'data': compounds
        }
