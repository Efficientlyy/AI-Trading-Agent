"""
NLP Processing Pipeline for Sentiment Analysis.

This module provides functionality for preprocessing text data, calculating sentiment scores,
and extracting entities from text content related to financial assets.
"""

# Patch NLTK to fix punkt_tab bug
import src.common.nltk_patch

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


class TextPreprocessor:
    """
    Text preprocessing for NLP tasks.
    
    This class provides methods for cleaning and preprocessing text data before
    sentiment analysis or entity extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing settings
        """
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add custom financial stop words if configured
        custom_stop_words = self.config.get('custom_stop_words', [])
        self.stop_words.update(custom_stop_words)
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.html_pattern = re.compile(r'<.*?>')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\d+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        
        logger.info("Text preprocessor initialized")
        
    def preprocess(self, text: str, remove_stop_words: bool = True, 
                  lemmatize: bool = True) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        Args:
            text: Input text to preprocess
            remove_stop_words: Whether to remove stop words
            lemmatize: Whether to lemmatize words
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove HTML tags
        text = self.html_pattern.sub('', text)
        
        # Remove mentions and hashtags (common in social media)
        text = self.mention_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        
        # Remove numbers (optional, depends on use case)
        if self.config.get('remove_numbers', True):
            text = self.number_pattern.sub('', text)
            
        # Remove punctuation
        text = self.punctuation_pattern.sub('', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        if remove_stop_words:
            tokens = [word for word in tokens if word not in self.stop_words]
            
        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
        # Join tokens back into text
        text = ' '.join(tokens)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def batch_preprocess(self, texts: List[str], remove_stop_words: bool = True,
                        lemmatize: bool = True) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts to preprocess
            remove_stop_words: Whether to remove stop words
            lemmatize: Whether to lemmatize words
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, remove_stop_words, lemmatize) for text in texts]


class SentimentAnalyzer:
    """
    Sentiment analysis for financial text data.
    
    This class provides methods for calculating sentiment scores for text data
    related to financial assets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration dictionary with sentiment analysis settings
        """
        self.config = config
        self.preprocessor = TextPreprocessor(config.get('preprocessing', {}))
        
        # Load lexicon-based sentiment dictionaries
        self._load_lexicons()
        
        logger.info("Sentiment analyzer initialized")
        
    def _load_lexicons(self):
        """Load sentiment lexicons for lexicon-based sentiment analysis."""
        # TODO: Load actual financial sentiment lexicons
        # For now, use a simple mock lexicon
        
        # Simple positive/negative word lists as a starting point
        self.positive_words = {
            'bullish', 'uptrend', 'growth', 'profit', 'gain', 'positive',
            'increase', 'rise', 'up', 'higher', 'strong', 'strength',
            'opportunity', 'promising', 'outperform', 'beat', 'exceed',
            'improvement', 'recovery', 'rally', 'support', 'buy', 'long',
            'upgrade', 'optimistic', 'confident', 'successful', 'innovative',
            'leadership', 'momentum', 'efficient', 'robust', 'breakthrough'
        }
        
        self.negative_words = {
            'bearish', 'downtrend', 'decline', 'loss', 'negative', 'decrease',
            'fall', 'down', 'lower', 'weak', 'weakness', 'risk', 'concerning',
            'underperform', 'miss', 'below', 'deterioration', 'downturn',
            'resistance', 'sell', 'short', 'downgrade', 'pessimistic', 'worried',
            'disappointing', 'struggling', 'challenging', 'inefficient',
            'vulnerable', 'slowdown', 'competitive_pressure', 'overvalued'
        }
        
        # Financial-specific modifiers that can intensify sentiment
        self.intensifiers = {
            'very', 'extremely', 'significantly', 'substantially', 'highly',
            'strongly', 'sharply', 'considerably', 'notably', 'markedly',
            'exceptionally', 'remarkably', 'dramatically', 'decidedly',
            'materially', 'massively', 'vastly', 'immensely', 'tremendously'
        }
        
        # Words that negate sentiment
        self.negators = {
            'not', 'no', 'never', 'none', 'neither', 'nor', 'nothing',
            'nowhere', 'hardly', 'barely', 'scarcely', 'doesn\'t', 'don\'t',
            'didn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t',
            'haven\'t', 'hadn\'t', 'won\'t', 'wouldn\'t', 'can\'t', 'cannot',
            'couldn\'t', 'shouldn\'t', 'without', 'despite', 'in spite of'
        }
        
        logger.info("Loaded sentiment lexicons")
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment analysis results, including:
            - score: Overall sentiment score (-1 to 1)
            - polarity: Sentiment polarity (positive, negative, neutral)
            - positive_words: List of positive words found
            - negative_words: List of negative words found
            - confidence: Confidence score for the sentiment analysis
        """
        if not text or not isinstance(text, str):
            return {
                'score': 0.0,
                'polarity': 'neutral',
                'positive_words': [],
                'negative_words': [],
                'confidence': 0.0
            }
            
        # Preprocess text
        preprocessed_text = self.preprocessor.preprocess(text)
        tokens = preprocessed_text.split()
        
        # Count positive and negative words
        positive_matches = []
        negative_matches = []
        
        # Track negation context
        negation_active = False
        negation_window = self.config.get('negation_window', 4)  # Words affected by negation
        negation_counter = 0
        
        for i, token in enumerate(tokens):
            # Check for negators
            if token in self.negators:
                negation_active = True
                negation_counter = 0
                continue
                
            # Update negation counter
            if negation_active:
                negation_counter += 1
                if negation_counter >= negation_window:
                    negation_active = False
                    
            # Check for positive words
            if token in self.positive_words:
                if negation_active:
                    negative_matches.append(token)
                else:
                    positive_matches.append(token)
                    
            # Check for negative words
            elif token in self.negative_words:
                if negation_active:
                    positive_matches.append(token)
                else:
                    negative_matches.append(token)
                    
        # Calculate sentiment score
        positive_count = len(positive_matches)
        negative_count = len(negative_matches)
        total_count = positive_count + negative_count
        
        if total_count == 0:
            score = 0.0
            polarity = 'neutral'
            confidence = 0.0
        else:
            score = (positive_count - negative_count) / total_count
            
            if score > 0.1:
                polarity = 'positive'
            elif score < -0.1:
                polarity = 'negative'
            else:
                polarity = 'neutral'
                
            # Simple confidence calculation based on total matches and text length
            confidence = min(1.0, total_count / (len(tokens) + 1))
            
        return {
            'score': score,
            'polarity': polarity,
            'positive_words': positive_matches,
            'negative_words': negative_matches,
            'confidence': confidence
        }
        
    def batch_analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        return [self.analyze_sentiment(text) for text in texts]
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, 
                         output_prefix: str = 'sentiment_') -> pd.DataFrame:
        """
        Process a DataFrame with text data and add sentiment analysis results.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text data
            output_prefix: Prefix for output columns
            
        Returns:
            DataFrame with added sentiment analysis columns
        """
        if text_column not in df.columns:
            logger.error(f"Text column '{text_column}' not found in DataFrame")
            return df
            
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Apply sentiment analysis to each row
        sentiment_results = []
        for text in df[text_column]:
            sentiment_results.append(self.analyze_sentiment(text))
            
        # Add sentiment analysis results as new columns
        result_df[f'{output_prefix}score'] = [r['score'] for r in sentiment_results]
        result_df[f'{output_prefix}polarity'] = [r['polarity'] for r in sentiment_results]
        result_df[f'{output_prefix}confidence'] = [r['confidence'] for r in sentiment_results]
        result_df[f'{output_prefix}positive_words'] = [r['positive_words'] for r in sentiment_results]
        result_df[f'{output_prefix}negative_words'] = [r['negative_words'] for r in sentiment_results]
        
        return result_df


class EntityExtractor:
    """
    Entity extraction for financial text data.
    
    This class provides methods for extracting relevant entities (e.g., company names,
    financial terms, asset symbols) from text content.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the entity extractor.
        
        Args:
            config: Configuration dictionary with entity extraction settings
        """
        self.config = config
        self.preprocessor = TextPreprocessor(config.get('preprocessing', {}))
        
        # Load entity dictionaries
        self._load_entity_dictionaries()
        
        # Compile regex patterns for entity extraction
        self._compile_patterns()
        
        logger.info("Entity extractor initialized")
        
    def _load_entity_dictionaries(self):
        """Load dictionaries for entity recognition."""
        # TODO: Load actual entity dictionaries
        # For now, use simple mock dictionaries
        
        # Map of asset symbols to common names
        self.asset_map = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'XRP': ['ripple', 'xrp'],
            'ADA': ['cardano', 'ada'],
            'SOL': ['solana', 'sol'],
            'AAPL': ['apple', 'aapl'],
            'MSFT': ['microsoft', 'msft'],
            'AMZN': ['amazon', 'amzn'],
            'GOOGL': ['google', 'alphabet', 'googl'],
            'META': ['facebook', 'meta', 'fb'],
        }
        
        # Create reverse mapping for lookup
        self.asset_reverse_map = {}
        for symbol, names in self.asset_map.items():
            for name in names:
                self.asset_reverse_map[name] = symbol
                
        # Financial terms to recognize
        self.financial_terms = {
            'market', 'stock', 'bond', 'crypto', 'cryptocurrency', 'token', 'coin',
            'exchange', 'trading', 'investment', 'investor', 'portfolio', 'asset',
            'price', 'value', 'volatility', 'volume', 'liquidity', 'market cap',
            'bull', 'bear', 'rally', 'correction', 'crash', 'bubble', 'dip', 'ath',
            'all-time high', 'all-time low', 'support', 'resistance', 'trend',
            'breakout', 'consolidation', 'accumulation', 'distribution', 'fomo',
            'fear of missing out', 'hodl', 'buy the dip', 'sell the news',
            'short squeeze', 'long', 'short', 'leverage', 'margin', 'futures',
            'options', 'call', 'put', 'strike price', 'expiry', 'dividend',
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'inflation', 'deflation', 'recession', 'depression', 'recovery',
            'interest rate', 'fed', 'central bank', 'regulation', 'sec',
            'securities and exchange commission', 'cftc', 'commodity futures trading commission'
        }
        
        logger.info("Loaded entity dictionaries")
        
    def _compile_patterns(self):
        """Compile regex patterns for entity extraction."""
        # Pattern for asset symbols (e.g., $BTC, $ETH)
        self.symbol_pattern = re.compile(r'\$([A-Z]{2,5})')
        
        # Pattern for cashtags (e.g., #BTC, #crypto)
        self.cashtag_pattern = re.compile(r'#([A-Za-z0-9_]+)')
        
        # Pattern for price mentions (e.g., $50K, $42,000)
        self.price_pattern = re.compile(r'\$([0-9,.]+)([KMBTkmbt]?)')
        
        logger.info("Compiled entity extraction patterns")
        
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with extracted entities, including:
            - asset_symbols: List of asset symbols mentioned
            - financial_terms: List of financial terms mentioned
            - prices: List of prices mentioned
            - cashtags: List of cashtags mentioned
        """
        if not text or not isinstance(text, str):
            return {
                'asset_symbols': [],
                'financial_terms': [],
                'prices': [],
                'cashtags': []
            }
            
        # Original text for regex patterns
        original_text = text
        
        # Preprocess text for term matching
        preprocessed_text = self.preprocessor.preprocess(text, remove_stop_words=False, lemmatize=False)
        tokens = preprocessed_text.split()
        
        # Extract asset symbols and names
        asset_symbols = set()
        
        # Direct symbol mentions (e.g., BTC, ETH)
        for token in tokens:
            if token.upper() in self.asset_map:
                asset_symbols.add(token.upper())
                
        # Asset names (e.g., bitcoin, ethereum)
        for token in tokens:
            if token.lower() in self.asset_reverse_map:
                asset_symbols.add(self.asset_reverse_map[token.lower()])
                
        # Cashtag/symbol pattern (e.g., $BTC, #BTC)
        symbol_matches = self.symbol_pattern.findall(original_text)
        for match in symbol_matches:
            if match in self.asset_map:
                asset_symbols.add(match)
                
        # Extract financial terms
        financial_terms = set()
        for term in self.financial_terms:
            if ' ' in term:  # Multi-word term
                if term.lower() in preprocessed_text.lower():
                    financial_terms.add(term)
            else:  # Single word term
                if term.lower() in tokens:
                    financial_terms.add(term)
                    
        # Extract cashtags
        cashtags = set(self.cashtag_pattern.findall(original_text))
        
        # Extract price mentions
        price_matches = self.price_pattern.findall(original_text)
        prices = []
        for amount, unit in price_matches:
            # Clean the amount string
            amount = amount.replace(',', '')
            try:
                value = float(amount)
                # Apply unit multiplier
                if unit.upper() == 'K':
                    value *= 1000
                elif unit.upper() == 'M':
                    value *= 1000000
                elif unit.upper() == 'B':
                    value *= 1000000000
                elif unit.upper() == 'T':
                    value *= 1000000000000
                prices.append(value)
            except ValueError:
                continue
                
        return {
            'asset_symbols': list(asset_symbols),
            'financial_terms': list(financial_terms),
            'prices': prices,
            'cashtags': list(cashtags)
        }
        
    def batch_extract_entities(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract entities from a batch of texts.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of dictionaries with extracted entities
        """
        return [self.extract_entities(text) for text in texts]
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, 
                         output_prefix: str = 'entity_') -> pd.DataFrame:
        """
        Process a DataFrame with text data and add entity extraction results.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text data
            output_prefix: Prefix for output columns
            
        Returns:
            DataFrame with added entity extraction columns
        """
        if text_column not in df.columns:
            logger.error(f"Text column '{text_column}' not found in DataFrame")
            return df
            
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Apply entity extraction to each row
        entity_results = []
        for text in df[text_column]:
            entity_results.append(self.extract_entities(text))
            
        # Add entity extraction results as new columns
        result_df[f'{output_prefix}asset_symbols'] = [r['asset_symbols'] for r in entity_results]
        result_df[f'{output_prefix}financial_terms'] = [r['financial_terms'] for r in entity_results]
        result_df[f'{output_prefix}prices'] = [r['prices'] for r in entity_results]
        result_df[f'{output_prefix}cashtags'] = [r['cashtags'] for r in entity_results]
        
        return result_df


class NLPPipeline:
    """
    Complete NLP processing pipeline for sentiment analysis.
    
    This class combines text preprocessing, sentiment analysis, and entity extraction
    into a single pipeline for processing financial text data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the NLP pipeline.
        
        Args:
            config: Configuration dictionary with settings for all components
        """
        self.config = config
        
        # Initialize components
        self.preprocessor = TextPreprocessor(config.get('preprocessing', {}))
        self.sentiment_analyzer = SentimentAnalyzer(config.get('sentiment_analysis', {}))
        self.entity_extractor = EntityExtractor(config.get('entity_extraction', {}))
        
        logger.info("NLP pipeline initialized")
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process a single text through the complete NLP pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with combined results from all pipeline components
        """
        # Preprocess text
        preprocessed_text = self.preprocessor.preprocess(text)
        
        # Analyze sentiment
        sentiment_results = self.sentiment_analyzer.analyze_sentiment(text)
        
        # Extract entities
        entity_results = self.entity_extractor.extract_entities(text)
        
        # Combine results
        return {
            'preprocessed_text': preprocessed_text,
            'sentiment': sentiment_results,
            'entities': entity_results
        }
        
    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of texts through the complete NLP pipeline.
        
        Args:
            texts: List of input texts to process
            
        Returns:
            List of dictionaries with combined results from all pipeline components
        """
        return [self.process_text(text) for text in texts]
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Process a DataFrame with text data through the complete NLP pipeline.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text data
            
        Returns:
            DataFrame with added columns for all pipeline components
        """
        if text_column not in df.columns:
            logger.error(f"Text column '{text_column}' not found in DataFrame")
            return df
            
        # Process with sentiment analyzer
        df = self.sentiment_analyzer.process_dataframe(df, text_column, 'sentiment_')
        
        # Process with entity extractor
        df = self.entity_extractor.process_dataframe(df, text_column, 'entity_')
        
        # Add preprocessed text column
        df['preprocessed_text'] = self.preprocessor.batch_preprocess(df[text_column].tolist())
        
        return df
    
    def process_sentiment_data(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process sentiment data collected from various sources.
        
        This method is specifically designed to work with the output of the
        SentimentCollectionService.collect_all method.
        
        Args:
            sentiment_data: DataFrame with sentiment data from various sources
            
        Returns:
            DataFrame with added NLP processing results
        """
        if 'content' not in sentiment_data.columns:
            logger.warning("No 'content' column found in sentiment data")
            return sentiment_data
            
        # Process the data with the NLP pipeline
        processed_data = self.process_dataframe(sentiment_data, 'content')
        
        # Aggregate sentiment scores by symbol and source
        if 'symbol' in processed_data.columns and 'source' in processed_data.columns:
            # Group by symbol, source, and timestamp (if available)
            group_cols = ['symbol', 'source']
            if 'timestamp' in processed_data.columns:
                group_cols.append(pd.Grouper(key='timestamp', freq='D'))
                
            # Aggregate sentiment scores
            agg_data = processed_data.groupby(group_cols).agg({
                'sentiment_score': ['mean', 'count', 'std'],
                'sentiment_confidence': ['mean'],
                'entity_asset_symbols': lambda x: list(set(sum(x, [])))
            }).reset_index()
            
            # Flatten column names
            agg_data.columns = ['_'.join(col).strip('_') for col in agg_data.columns.values]
            
            return agg_data
        else:
            return processed_data
