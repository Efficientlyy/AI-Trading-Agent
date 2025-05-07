"""
Advanced NLP Processing Pipeline for Sentiment Analysis

This module provides an enhanced NLP processing pipeline using transformer models
for more accurate sentiment analysis and entity recognition.
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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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


class AdvancedSentimentProcessor:
    """
    Advanced NLP processor for sentiment analysis of financial and crypto news.
    
    This class provides methods for preprocessing text, extracting features,
    and analyzing sentiment using transformer models and traditional NLP techniques.
    """
    
    def __init__(
        self,
        use_vader: bool = True,
        use_transformers: bool = True,
        use_spacy: bool = True,
        transformer_model: str = "ProsusAI/finbert",
        device: str = None
    ):
        """
        Initialize the advanced sentiment processor.
        
        Args:
            use_vader: Whether to use VADER sentiment analysis
            use_transformers: Whether to use transformer models for sentiment analysis
            use_spacy: Whether to use spaCy for entity recognition
            transformer_model: Transformer model to use for sentiment analysis
            device: Device to use for transformer models ('cpu', 'cuda', or None for auto-detection)
        """
        self.use_vader = use_vader
        self.use_transformers = use_transformers
        self.use_spacy = use_spacy
        
        # Initialize VADER sentiment analyzer
        if use_vader:
            self.vader = SentimentIntensityAnalyzer()
        
        # Initialize transformer model
        if use_transformers:
            try:
                # Auto-detect device if not specified
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.device = device
                logger.info(f"Using device: {device}")
                
                # Load transformer model
                logger.info(f"Loading transformer model: {transformer_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
                self.model = AutoModelForSequenceClassification.from_pretrained(transformer_model)
                self.model.to(device)
                
                # Create sentiment pipeline
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if device == "cuda" else -1
                )
                
                # Create NER pipeline
                self.ner_pipeline = pipeline(
                    "ner",
                    model="Jean-Baptiste/roberta-large-ner-english",
                    tokenizer="Jean-Baptiste/roberta-large-ner-english",
                    device=0 if device == "cuda" else -1
                )
                
                logger.info("Transformer models loaded successfully")
            except Exception as e:
                logger.error(f"Error loading transformer models: {e}")
                self.use_transformers = False
        
        # Load crypto-specific terms for sentiment analysis
        self.crypto_terms = self._load_crypto_terms()
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize TF-IDF vectorizer for topic modeling
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize LDA model for topic modeling
        self.lda_model = LatentDirichletAllocation(
            n_components=5,
            random_state=42
        )
        
        logger.info("Advanced sentiment processor initialized")
    
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
            "etf": 0.6,
            "approval": 0.5,
            "partnership": 0.6,
            "upgrade": 0.5,
            "scaling": 0.4,
            "layer2": 0.4,
            "staking": 0.3,
            "yield": 0.3,
            "consensus": 0.2,
            
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
            "exploit": -0.7,
            "vulnerability": -0.6,
            "liquidation": -0.6,
            "margin call": -0.7,
            "delisting": -0.7,
            "shutdown": -0.8,
            
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
            "custody": 0.0,
            "smart contract": 0.2,
            "oracle": 0.1,
            "governance": 0.1,
            "dao": 0.2,
            "metaverse": 0.2,
            "web3": 0.3
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing special characters, URLs, etc.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract advanced features from text for sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        if not text:
            return {
                'word_count': 0,
                'crypto_terms_found': [],
                'crypto_sentiment_score': 0.0,
                'entities': [],
                'topics': [],
                'keywords': []
            }
        
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Count words
        word_count = len(filtered_tokens)
        
        # Find crypto terms
        crypto_terms_found = []
        crypto_sentiment_score = 0.0
        
        for term, score in self.crypto_terms.items():
            if term in text.lower():
                crypto_terms_found.append(term)
                crypto_sentiment_score += score
        
        # Normalize crypto sentiment score
        if crypto_terms_found:
            crypto_sentiment_score /= len(crypto_terms_found)
        
        # Extract entities using spaCy
        entities = []
        if self.use_spacy:
            doc = nlp(text)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Extract entities using transformer model
        if self.use_transformers:
            try:
                transformer_entities = self.ner_pipeline(text)
                for ent in transformer_entities:
                    if ent['score'] > 0.8:  # Only include high-confidence entities
                        entities.append({
                            'text': ent['word'],
                            'label': ent['entity'],
                            'score': ent['score']
                        })
            except Exception as e:
                logger.error(f"Error extracting entities with transformer model: {e}")
        
        # Extract keywords (most frequent non-stopwords)
        word_freq = Counter(filtered_tokens)
        keywords = [word for word, freq in word_freq.most_common(10)]
        
        # Extract topics
        topics = []
        try:
            # We need a corpus of documents for topic modeling
            # For a single document, we'll just identify the main themes
            if len(text.split()) > 20:  # Only if text is long enough
                # Create document-term matrix
                dtm = self.tfidf_vectorizer.fit_transform([text])
                
                # Get feature names
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                
                # Get top terms for each topic
                for topic_idx, topic in enumerate(self.lda_model.components_):
                    top_terms = [feature_names[i] for i in topic.argsort()[:-6:-1]]
                    topics.append({
                        'id': topic_idx,
                        'terms': top_terms
                    })
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
        
        return {
            'word_count': word_count,
            'crypto_terms_found': crypto_terms_found,
            'crypto_sentiment_score': crypto_sentiment_score,
            'entities': entities,
            'topics': topics,
            'keywords': keywords
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using multiple methods.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of sentiment scores
        """
        if not text:
            return {
                'vader_compound': 0.0,
                'vader_pos': 0.0,
                'vader_neg': 0.0,
                'vader_neu': 0.0,
                'transformer_label': 'neutral',
                'transformer_score': 0.0,
                'combined_score': 0.0,
                'sentiment_label': 'neutral'
            }
        
        # Initialize sentiment scores
        vader_scores = {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0}
        transformer_result = {'label': 'neutral', 'score': 0.0}
        
        # VADER sentiment analysis
        if self.use_vader:
            try:
                vader_scores = self.vader.polarity_scores(text)
            except Exception as e:
                logger.error(f"Error analyzing sentiment with VADER: {e}")
        
        # Transformer-based sentiment analysis
        if self.use_transformers:
            try:
                # Truncate text if too long for transformer model
                max_length = self.tokenizer.model_max_length
                if len(text.split()) > max_length:
                    text = ' '.join(text.split()[:max_length])
                
                # Get sentiment prediction
                result = self.sentiment_pipeline(text)[0]
                transformer_result = {
                    'label': result['label'].lower(),
                    'score': result['score']
                }
            except Exception as e:
                logger.error(f"Error analyzing sentiment with transformer model: {e}")
        
        # Extract crypto-specific sentiment
        features = self.extract_features(text)
        crypto_sentiment_score = features['crypto_sentiment_score']
        
        # Combine sentiment scores
        # Weight: 30% VADER, 50% Transformer, 20% Crypto-specific
        combined_score = (
            0.3 * vader_scores['compound'] +
            0.5 * (transformer_result['score'] if transformer_result['label'] == 'positive' else -transformer_result['score'] if transformer_result['label'] == 'negative' else 0.0) +
            0.2 * crypto_sentiment_score
        )
        
        # Determine sentiment label
        if combined_score >= 0.05:
            sentiment_label = 'positive'
        elif combined_score <= -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_pos': vader_scores['pos'],
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'transformer_label': transformer_result['label'],
            'transformer_score': transformer_result['score'],
            'crypto_sentiment_score': crypto_sentiment_score,
            'combined_score': combined_score,
            'sentiment_label': sentiment_label
        }
    
    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a news article for advanced sentiment analysis.
        
        Args:
            article: News article data
            
        Returns:
            Processed article with sentiment analysis
        """
        # Extract text from article
        title = article.get('title', '')
        content = article.get('content', '')
        summary = article.get('summary', '')
        
        # Combine text for analysis
        # Title is weighted more heavily
        full_text = f"{title} {title} {summary} {content}"
        
        # Preprocess text
        preprocessed_text = self.preprocess_text(full_text)
        
        # Extract features
        nlp_features = self.extract_features(preprocessed_text)
        
        # Analyze sentiment
        sentiment_results = self.analyze_sentiment(preprocessed_text)
        
        # Create processed article
        processed_article = article.copy()
        processed_article.update({
            'preprocessed_text': preprocessed_text,
            'nlp_features': nlp_features,
            'sentiment_results': sentiment_results,
            'sentiment_score': sentiment_results['combined_score'],
            'sentiment_label': sentiment_results['sentiment_label']
        })
        
        return processed_article
    
    def process_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of news articles for sentiment analysis.
        
        Args:
            articles: List of news article data
            
        Returns:
            List of processed articles with sentiment analysis
        """
        processed_articles = []
        
        for article in articles:
            try:
                processed_article = self.process_article(article)
                processed_articles.append(processed_article)
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                # Add the original article with an error flag
                article_with_error = article.copy()
                article_with_error['error'] = str(e)
                processed_articles.append(article_with_error)
        
        return processed_articles


class AdvancedSentimentSignalGenerator:
    """
    Generate advanced trading signals based on sentiment analysis.
    
    This class aggregates sentiment data and generates trading signals
    based on sentiment trends, volume, and market context.
    """
    
    def __init__(
        self,
        sentiment_threshold: float = 0.3,
        min_articles: int = 3,
        lookback_days: int = 5,
        sentiment_change_threshold: float = 0.15,
        volume_weight: float = 0.3,
        trend_weight: float = 0.4,
        sentiment_weight: float = 0.3
    ):
        """
        Initialize the advanced sentiment signal generator.
        
        Args:
            sentiment_threshold: Threshold for sentiment score to generate a signal
            min_articles: Minimum number of articles required to generate a signal
            lookback_days: Number of days to look back for trend analysis
            sentiment_change_threshold: Threshold for sentiment change to generate a signal
            volume_weight: Weight for article volume in signal generation
            trend_weight: Weight for sentiment trend in signal generation
            sentiment_weight: Weight for absolute sentiment in signal generation
        """
        self.sentiment_threshold = sentiment_threshold
        self.min_articles = min_articles
        self.lookback_days = lookback_days
        self.sentiment_change_threshold = sentiment_change_threshold
        self.volume_weight = volume_weight
        self.trend_weight = trend_weight
        self.sentiment_weight = sentiment_weight
        
        logger.info("Advanced sentiment signal generator initialized")
    
    def aggregate_sentiment(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment data by day with advanced metrics.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            
        Returns:
            Aggregated sentiment data
        """
        if sentiment_data.empty:
            return pd.DataFrame()
        
        # Ensure timestamp column exists
        if 'time_published' not in sentiment_data.columns:
            if 'timestamp' in sentiment_data.columns:
                sentiment_data['time_published'] = sentiment_data['timestamp']
            else:
                logger.error("No timestamp column found in sentiment data")
                return pd.DataFrame()
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(sentiment_data['time_published']):
            sentiment_data['time_published'] = pd.to_datetime(sentiment_data['time_published'])
        
        # Extract date from timestamp
        sentiment_data['date'] = sentiment_data['time_published'].dt.date
        
        # Group by date and calculate aggregated metrics
        agg_sentiment = sentiment_data.groupby('date').agg({
            'sentiment_score': ['mean', 'median', 'std', 'count'],
            'time_published': 'min'  # Keep earliest timestamp for reference
        })
        
        # Flatten column names
        agg_sentiment.columns = ['_'.join(col).strip() for col in agg_sentiment.columns.values]
        
        # Rename columns
        agg_sentiment = agg_sentiment.rename(columns={
            'sentiment_score_mean': 'sentiment_score',
            'sentiment_score_median': 'sentiment_median',
            'sentiment_score_std': 'sentiment_std',
            'sentiment_score_count': 'article_count',
            'time_published_min': 'time_published'
        })
        
        # Reset index to make date a column
        agg_sentiment = agg_sentiment.reset_index()
        
        # Sort by date
        agg_sentiment = agg_sentiment.sort_values('date', ascending=False)
        
        # Calculate sentiment changes over different periods
        for days in range(1, min(self.lookback_days + 1, len(agg_sentiment))):
            agg_sentiment[f'sentiment_change_{days}d'] = agg_sentiment['sentiment_score'].diff(-days)
        
        # Calculate volume changes over different periods
        for days in range(1, min(self.lookback_days + 1, len(agg_sentiment))):
            agg_sentiment[f'volume_change_{days}d'] = agg_sentiment['article_count'].diff(-days)
        
        # Calculate rolling averages
        if len(agg_sentiment) >= 3:
            agg_sentiment['sentiment_ma3'] = agg_sentiment['sentiment_score'].rolling(window=3, min_periods=1).mean()
            agg_sentiment['volume_ma3'] = agg_sentiment['article_count'].rolling(window=3, min_periods=1).mean()
        
        if len(agg_sentiment) >= 7:
            agg_sentiment['sentiment_ma7'] = agg_sentiment['sentiment_score'].rolling(window=7, min_periods=1).mean()
            agg_sentiment['volume_ma7'] = agg_sentiment['article_count'].rolling(window=7, min_periods=1).mean()
        
        # Calculate sentiment momentum (rate of change)
        if len(agg_sentiment) >= 3:
            agg_sentiment['sentiment_momentum'] = (
                agg_sentiment['sentiment_score'] - agg_sentiment['sentiment_score'].shift(-2)
            ) / 2
        
        return agg_sentiment
    
    def generate_signals(self, agg_sentiment: pd.DataFrame, market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate advanced trading signals based on aggregated sentiment and market data.
        
        Args:
            agg_sentiment: Aggregated sentiment data
            market_data: Optional market data for context (price, volume, etc.)
            
        Returns:
            DataFrame with trading signals
        """
        if agg_sentiment.empty:
            return pd.DataFrame()
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=agg_sentiment.index)
        
        # Copy relevant columns
        signals['date'] = agg_sentiment['date']
        signals['time_published'] = agg_sentiment['time_published']
        signals['sentiment_score'] = agg_sentiment['sentiment_score']
        signals['article_count'] = agg_sentiment['article_count']
        
        # Initialize signal columns
        signals['signal'] = 'NEUTRAL'
        signals['signal_strength'] = 0.0
        signals['signal_confidence'] = 0.0
        
        # Add market context if available
        if market_data is not None:
            # Ensure market_data has a datetime index
            if not isinstance(market_data.index, pd.DatetimeIndex):
                if 'date' in market_data.columns:
                    market_data = market_data.set_index('date')
                elif 'timestamp' in market_data.columns:
                    market_data = market_data.set_index('timestamp')
            
            # Merge market data with signals
            for date_idx, row in signals.iterrows():
                date = row['date']
                if date in market_data.index:
                    # Add market data columns
                    for col in market_data.columns:
                        signals.loc[date_idx, f'market_{col}'] = market_data.loc[date, col]
        
        # Generate signals for each day
        for i, row in agg_sentiment.iterrows():
            # Skip if not enough articles
            if row['article_count'] < self.min_articles:
                continue
            
            # Get sentiment score and change
            sentiment_score = row['sentiment_score']
            
            # Get sentiment change over lookback period
            sentiment_change_col = f'sentiment_change_{min(self.lookback_days, len(agg_sentiment)-1)}d'
            sentiment_change = row.get(sentiment_change_col, 0)
            
            # Get sentiment momentum
            sentiment_momentum = row.get('sentiment_momentum', 0)
            
            # Calculate volume factor (more articles = stronger signal)
            volume_factor = min(row['article_count'] / self.min_articles, 2.0)
            
            # Calculate trend factor
            trend_factor = 0.0
            if sentiment_change > 0:
                trend_factor = min(sentiment_change / self.sentiment_change_threshold, 2.0)
            elif sentiment_change < 0:
                trend_factor = max(sentiment_change / self.sentiment_change_threshold, -2.0)
            
            # Calculate sentiment factor
            sentiment_factor = 0.0
            if sentiment_score > 0:
                sentiment_factor = min(sentiment_score / self.sentiment_threshold, 2.0)
            elif sentiment_score < 0:
                sentiment_factor = max(sentiment_score / self.sentiment_threshold, -2.0)
            
            # Calculate combined signal strength
            signal_strength = (
                self.volume_weight * volume_factor +
                self.trend_weight * trend_factor +
                self.sentiment_weight * sentiment_factor
            )
            
            # Calculate signal confidence
            # Higher confidence if all factors align
            signal_confidence = (
                abs(volume_factor) * abs(trend_factor) * abs(sentiment_factor)
            ) ** (1/3)  # Geometric mean
            
            # Determine signal
            if signal_strength >= 1.0:
                signals.loc[i, 'signal'] = 'STRONG_BUY'
            elif signal_strength >= 0.5:
                signals.loc[i, 'signal'] = 'BUY'
            elif signal_strength >= 0.2:
                signals.loc[i, 'signal'] = 'WEAK_BUY'
            elif signal_strength <= -1.0:
                signals.loc[i, 'signal'] = 'STRONG_SELL'
            elif signal_strength <= -0.5:
                signals.loc[i, 'signal'] = 'SELL'
            elif signal_strength <= -0.2:
                signals.loc[i, 'signal'] = 'WEAK_SELL'
            elif sentiment_score > 0:
                signals.loc[i, 'signal'] = 'NEUTRAL_BULLISH'
            elif sentiment_score < 0:
                signals.loc[i, 'signal'] = 'NEUTRAL_BEARISH'
            else:
                signals.loc[i, 'signal'] = 'NEUTRAL'
            
            # Set signal strength and confidence
            signals.loc[i, 'signal_strength'] = signal_strength
            signals.loc[i, 'signal_confidence'] = signal_confidence
        
        return signals
    
    def process_sentiment_data(
        self,
        sentiment_data: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process sentiment data and generate trading signals.
        
        Args:
            sentiment_data: DataFrame with sentiment data
            market_data: Optional market data for context
            
        Returns:
            Tuple of (aggregated sentiment data, trading signals)
        """
        # Aggregate sentiment
        agg_sentiment = self.aggregate_sentiment(sentiment_data)
        
        # Generate signals
        signals = self.generate_signals(agg_sentiment, market_data)
        
        return agg_sentiment, signals


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create advanced sentiment processor
    processor = AdvancedSentimentProcessor(use_transformers=True)
    
    # Example article
    article = {
        'title': 'Bitcoin Surges to New All-Time High as Institutional Adoption Grows',
        'content': 'Bitcoin reached a new all-time high today as more institutional investors are buying the cryptocurrency. The rally comes amid growing mainstream adoption and positive regulatory developments. Several major financial institutions have announced plans to offer crypto services to their clients.',
        'summary': 'Bitcoin hits new ATH with growing institutional interest.',
        'time_published': datetime.now()
    }
    
    # Process article
    processed_article = processor.process_article(article)
    print("Advanced Sentiment Analysis:")
    print(f"Score: {processed_article['sentiment_score']:.2f}")
    print(f"Label: {processed_article['sentiment_label']}")
    print(f"Features: {processed_article['nlp_features']['crypto_terms_found']}")
    print(f"Entities: {[e['text'] for e in processed_article['nlp_features']['entities']]}")
    
    # Create advanced sentiment signal generator
    signal_generator = AdvancedSentimentSignalGenerator()
    
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
    print(agg_sentiment[['date', 'sentiment_score', 'article_count', 'sentiment_change_1d']])
    
    print("\nAdvanced Trading Signals:")
    print(signals[['date', 'sentiment_score', 'signal', 'signal_strength', 'signal_confidence']])
