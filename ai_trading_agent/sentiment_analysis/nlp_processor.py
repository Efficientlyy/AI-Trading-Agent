"""
Advanced NLP Processing Pipeline for Sentiment Analysis

This module provides an advanced NLP processing pipeline for analyzing news sentiment
and generating trading signals based on sentiment analysis.
"""

import pandas as pd
import numpy as np
import logging
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import os
import sys
import spacy
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logger.info("Downloading spaCy model...")
    import subprocess
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


class SentimentProcessor:
    """
    Advanced NLP processor for sentiment analysis of news articles.
    
    This class provides methods for preprocessing text, extracting features,
    and analyzing sentiment from news articles.
    """
    
    def __init__(self, use_vader: bool = True, use_spacy: bool = True):
        """
        Initialize the sentiment processor.
        
        Args:
            use_vader: Whether to use VADER sentiment analysis
            use_spacy: Whether to use spaCy for entity recognition and additional processing
        """
        self.use_vader = use_vader
        self.use_spacy = use_spacy
        
        # Initialize VADER sentiment analyzer
        if use_vader:
            self.vader = SentimentIntensityAnalyzer()
        
        # Load crypto-specific terms for sentiment analysis
        self.crypto_terms = self._load_crypto_terms()
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        logger.info("Sentiment processor initialized")
    
    def _load_crypto_terms(self) -> Dict[str, float]:
        """
        Load crypto-specific terms and their sentiment scores.
        
        Returns:
            Dictionary mapping crypto terms to sentiment scores
        """
        # This is a simplified version - in a real implementation, this would load from a file
        # or database that's regularly updated
        return {
            # Positive terms
            "hodl": 0.6,
            "moon": 0.7,
            "bullish": 0.8,
            "rally": 0.6,
            "surge": 0.5,
            "adoption": 0.4,
            "institutional": 0.3,
            "mainstream": 0.3,
            "breakthrough": 0.5,
            "innovation": 0.4,
            
            # Negative terms
            "dump": -0.6,
            "crash": -0.8,
            "bearish": -0.8,
            "ban": -0.7,
            "regulation": -0.4,
            "hack": -0.7,
            "scam": -0.9,
            "ponzi": -0.9,
            "bubble": -0.6,
            "correction": -0.4,
            "sell-off": -0.5,
            "fraud": -0.8,
            "investigation": -0.5,
            
            # Neutral/context-dependent terms
            "volatile": -0.2,
            "volatility": -0.2,
            "halving": 0.2,
            "mining": 0.1,
            "miner": 0.1,
            "fork": 0.0,
            "ico": -0.1,
            "token": 0.0,
            "blockchain": 0.2,
            "defi": 0.3,
            "nft": 0.2,
            "stablecoin": 0.1,
            "exchange": 0.0,
            "wallet": 0.1,
            "custody": 0.0
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing special characters, URLs, etc.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features from text for sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        if not text:
            return {}
        
        features = {}
        
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        
        # Count tokens
        features['token_count'] = len(filtered_tokens)
        
        # Count crypto-specific terms
        crypto_term_count = sum(1 for token in filtered_tokens if token in self.crypto_terms)
        features['crypto_term_count'] = crypto_term_count
        features['crypto_term_ratio'] = crypto_term_count / len(filtered_tokens) if filtered_tokens else 0
        
        # Calculate crypto sentiment from domain-specific terms
        crypto_sentiment_score = 0
        crypto_terms_found = []
        
        for token in filtered_tokens:
            if token in self.crypto_terms:
                crypto_sentiment_score += self.crypto_terms[token]
                crypto_terms_found.append(token)
        
        features['crypto_sentiment_score'] = crypto_sentiment_score
        features['crypto_terms_found'] = crypto_terms_found
        
        # Use spaCy for entity recognition if enabled
        if self.use_spacy:
            doc = nlp(text)
            
            # Extract entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            features['entities'] = entities
            
            # Count entity types
            entity_types = Counter([ent[1] for ent in entities])
            features['entity_types'] = dict(entity_types)
            
            # Extract noun chunks (for topic modeling)
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            features['noun_chunks'] = noun_chunks
        
        return features
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using multiple methods.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of sentiment scores
        """
        if not text:
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0,
                'crypto_specific': 0.0,
                'combined': 0.0
            }
        
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # Extract features
        features = self.extract_features(preprocessed_text)
        
        # Initialize sentiment scores
        sentiment = {
            'compound': 0.0,
            'pos': 0.0,
            'neu': 0.0,
            'neg': 0.0,
            'crypto_specific': 0.0,
            'combined': 0.0
        }
        
        # Use VADER for general sentiment analysis
        if self.use_vader:
            vader_scores = self.vader.polarity_scores(preprocessed_text)
            sentiment['compound'] = vader_scores['compound']
            sentiment['pos'] = vader_scores['pos']
            sentiment['neu'] = vader_scores['neu']
            sentiment['neg'] = vader_scores['neg']
        
        # Use crypto-specific sentiment
        if 'crypto_sentiment_score' in features:
            sentiment['crypto_specific'] = features['crypto_sentiment_score']
        
        # Combine sentiment scores (weighted average)
        # We give more weight to crypto-specific sentiment if there are crypto terms
        if features.get('crypto_term_count', 0) > 0:
            crypto_weight = min(0.7, features.get('crypto_term_ratio', 0) * 2)
            vader_weight = 1 - crypto_weight
            
            sentiment['combined'] = (
                vader_weight * sentiment['compound'] + 
                crypto_weight * sentiment['crypto_specific']
            )
        else:
            # If no crypto terms, rely more on VADER
            sentiment['combined'] = sentiment['compound']
        
        return sentiment
    
    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a news article for sentiment analysis.
        
        Args:
            article: News article data
            
        Returns:
            Processed article with sentiment analysis
        """
        # Create a copy of the article to avoid modifying the original
        processed = article.copy()
        
        # Combine title and summary for analysis
        title = article.get('title', '')
        summary = article.get('summary', '')
        combined_text = f"{title} {summary}"
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(combined_text)
        processed['sentiment_analysis'] = sentiment
        
        # Extract features
        preprocessed_text = self.preprocess_text(combined_text)
        features = self.extract_features(preprocessed_text)
        processed['nlp_features'] = features
        
        # Determine sentiment label based on combined score
        combined_score = sentiment['combined']
        
        if combined_score >= 0.35:
            sentiment_label = "bullish"
        elif combined_score <= -0.35:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"
        
        processed['sentiment_label'] = sentiment_label
        processed['sentiment_score'] = combined_score
        
        return processed


class SentimentSignalGenerator:
    """
    Generate trading signals based on sentiment analysis.
    
    This class aggregates sentiment data and generates trading signals
    based on sentiment trends and thresholds.
    """
    
    def __init__(self, 
                sentiment_threshold: float = 0.4,
                min_articles: int = 5,
                lookback_days: int = 3,
                sentiment_change_threshold: float = 0.2):
        """
        Initialize the sentiment signal generator.
        
        Args:
            sentiment_threshold: Threshold for sentiment score to generate a signal
            min_articles: Minimum number of articles required to generate a signal
            lookback_days: Number of days to look back for sentiment trend
            sentiment_change_threshold: Threshold for sentiment change to generate a signal
        """
        self.sentiment_threshold = sentiment_threshold
        self.min_articles = min_articles
        self.lookback_days = lookback_days
        self.sentiment_change_threshold = sentiment_change_threshold
        
        logger.info("Sentiment signal generator initialized")
    
    def aggregate_sentiment(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment data by day.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            
        Returns:
            Aggregated sentiment data
        """
        if sentiment_data.empty:
            return pd.DataFrame()
        
        # Ensure we have a datetime index
        if 'time_published' in sentiment_data.columns:
            sentiment_data = sentiment_data.set_index('time_published')
        
        # Resample by day and aggregate
        agg_data = sentiment_data.resample('D').agg({
            'sentiment_score': 'mean',
            'title': 'count'  # Count articles per day
        })
        
        # Rename columns
        agg_data = agg_data.rename(columns={'title': 'article_count'})
        
        # Calculate rolling averages
        agg_data['sentiment_score_3d'] = agg_data['sentiment_score'].rolling(window=3, min_periods=1).mean()
        agg_data['sentiment_score_7d'] = agg_data['sentiment_score'].rolling(window=7, min_periods=1).mean()
        
        # Calculate sentiment change (day-to-day)
        agg_data['sentiment_change_1d'] = agg_data['sentiment_score'].diff()
        
        # Calculate sentiment change (3-day)
        agg_data['sentiment_change_3d'] = agg_data['sentiment_score_3d'].diff(3)
        
        # Reset index to get date as a column
        agg_data = agg_data.reset_index()
        
        return agg_data
    
    def generate_signals(self, agg_sentiment: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on aggregated sentiment.
        
        Args:
            agg_sentiment: Aggregated sentiment data
            
        Returns:
            DataFrame with trading signals
        """
        if agg_sentiment.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        signals = agg_sentiment.copy()
        
        # Initialize signal column
        signals['signal'] = 'NEUTRAL'
        signals['signal_strength'] = 0.0
        
        # Generate signals based on sentiment score and article count
        for i, row in signals.iterrows():
            # Skip if not enough articles
            if row['article_count'] < self.min_articles:
                continue
            
            sentiment_score = row['sentiment_score']
            sentiment_change = row['sentiment_change_3d']
            
            # Strong bullish signal
            if (sentiment_score > self.sentiment_threshold and 
                sentiment_change > self.sentiment_change_threshold):
                signals.loc[i, 'signal'] = 'STRONG_BUY'
                signals.loc[i, 'signal_strength'] = min(1.0, sentiment_score)
            
            # Bullish signal
            elif sentiment_score > self.sentiment_threshold:
                signals.loc[i, 'signal'] = 'BUY'
                signals.loc[i, 'signal_strength'] = min(0.8, sentiment_score)
            
            # Strong bearish signal
            elif (sentiment_score < -self.sentiment_threshold and 
                  sentiment_change < -self.sentiment_change_threshold):
                signals.loc[i, 'signal'] = 'STRONG_SELL'
                signals.loc[i, 'signal_strength'] = min(1.0, -sentiment_score)
            
            # Bearish signal
            elif sentiment_score < -self.sentiment_threshold:
                signals.loc[i, 'signal'] = 'SELL'
                signals.loc[i, 'signal_strength'] = min(0.8, -sentiment_score)
            
            # Neutral with positive bias
            elif sentiment_score > 0:
                signals.loc[i, 'signal'] = 'NEUTRAL_BULLISH'
                signals.loc[i, 'signal_strength'] = min(0.4, sentiment_score * 2)
            
            # Neutral with negative bias
            elif sentiment_score < 0:
                signals.loc[i, 'signal'] = 'NEUTRAL_BEARISH'
                signals.loc[i, 'signal_strength'] = min(0.4, -sentiment_score * 2)
        
        return signals
    
    def process_sentiment_data(self, sentiment_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process sentiment data and generate trading signals.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            
        Returns:
            Tuple of (aggregated sentiment data, trading signals)
        """
        # Aggregate sentiment
        agg_sentiment = self.aggregate_sentiment(sentiment_data)
        
        # Generate signals
        signals = self.generate_signals(agg_sentiment)
        
        return agg_sentiment, signals


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sentiment processor
    processor = SentimentProcessor()
    
    # Example article
    article = {
        'title': 'Bitcoin Surges to New All-Time High as Institutional Adoption Grows',
        'summary': 'Bitcoin reached a new all-time high today as more institutional investors are buying the cryptocurrency. The rally comes amid growing mainstream adoption and positive regulatory developments.',
        'time_published': datetime.now()
    }
    
    # Process article
    processed_article = processor.process_article(article)
    print("Sentiment Analysis:")
    print(f"Score: {processed_article['sentiment_score']:.2f}")
    print(f"Label: {processed_article['sentiment_label']}")
    print(f"Features: {processed_article['nlp_features']['crypto_terms_found']}")
    
    # Create sentiment signal generator
    signal_generator = SentimentSignalGenerator()
    
    # Create a sample DataFrame with sentiment data
    data = []
    for i in range(10):
        day = datetime.now() - timedelta(days=i)
        score = 0.5 - (i * 0.1)  # Decreasing sentiment over time
        data.append({
            'time_published': day,
            'sentiment_score': score,
            'title': f'Article {i}'
        })
    
    sentiment_df = pd.DataFrame(data)
    
    # Generate signals
    agg_sentiment, signals = signal_generator.process_sentiment_data(sentiment_df)
    
    print("\nAggregated Sentiment:")
    print(agg_sentiment[['time_published', 'sentiment_score', 'article_count', 'sentiment_change_1d']])
    
    print("\nTrading Signals:")
    print(signals[['time_published', 'sentiment_score', 'signal', 'signal_strength']])
