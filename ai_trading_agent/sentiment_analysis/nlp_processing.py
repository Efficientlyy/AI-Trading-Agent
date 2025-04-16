"""
NLP Processing Pipeline for Sentiment Analysis.

This module provides functionality for preprocessing text data, calculating sentiment scores,
and extracting entities from text content related to financial assets.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import unicodedata
import emoji
import contractions
from langdetect import detect, LangDetectException

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
        self.default_language = config.get("default_language", "english")
        self.stopwords_cache = {}
        self.lemmatizer = WordNetLemmatizer()
        self.lemmatizers = {"english": self.lemmatizer}
        # Add SnowballStemmer for supported non-English languages
        for lang in ["spanish", "french", "german", "italian", "dutch"]:
            try:
                self.lemmatizers[lang] = SnowballStemmer(lang)
            except ValueError:
                self.lemmatizers[lang] = None
                
        # Add custom financial stop words if configured
        custom_stop_words = self.config.get('custom_stop_words', [])
        self.stopwords_cache["english"] = set(stopwords.words('english')).union(custom_stop_words)
        
        # Preload stopwords for supported languages
        self.supported_languages = ["english", "spanish", "french", "german", "italian", "dutch"]
        for lang in self.supported_languages:
            if lang not in self.stopwords_cache:
                try:
                    self.stopwords_cache[lang] = set(stopwords.words(lang))
                except LookupError:
                    self.stopwords_cache[lang] = set()
        
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
        Preprocess text by cleaning and normalizing, with multilingual support.
        
        Args:
            text: Input text to preprocess
            remove_stop_words: Whether to remove stop words
            lemmatize: Whether to lemmatize words
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Detect language
        try:
            lang_code = detect(text)
        except LangDetectException:
            lang_code = "en"
        lang_map = {
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "nl": "dutch"
        }
        lang = lang_map.get(lang_code, self.default_language)
        stop_words = self.stopwords_cache.get(lang, set())
        
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions (language-specific)
        if lang == "english":
            text = contractions.fix(text)
        elif lang in self.supported_languages:
            # Placeholder for non-English contraction expansion
            # Example: text = expand_contractions_non_english(text, lang)
            pass
                
        # Remove emojis (or convert to text)
        text = emoji.replace_emoji(text, replace="")  # Remove emojis
        
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
        
        # Slang normalization (optional, placeholder for future extension)
        # Example: slang_dict = {"u": "you", "r": "are", ...}
        # for slang, replacement in slang_dict.items():
        #     text = re.sub(r"\b{}\b".format(re.escape(slang)), replacement, text)
        
        # Tokenize (NLTK's word_tokenize supports some languages, fallback to split)
        try:
            tokens = word_tokenize(text, language=lang)
        except LookupError:
            tokens = text.split()
        
        # Remove stop words
        if remove_stop_words:
            tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize/stem based on language
        if lemmatize:
            lemmatizer = self.lemmatizers.get(lang)
            if lang == "english" and lemmatizer:
                tokens = [lemmatizer.lemmatize(word) for word in tokens]
            elif lemmatizer:
                tokens = [lemmatizer.stem(word) for word in tokens]
        
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


from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer:
    """
    Sentiment analysis for financial text data.
    
    This class provides methods for calculating sentiment scores for text data
    related to financial assets. Supports rule-based, transformer-based, and ensemble models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config: Configuration dictionary with sentiment analysis settings
        """
        self.config = config
        self.preprocessor = TextPreprocessor(config.get('preprocessing', {}))
        self.model_type = config.get("model_type", "rule")  # "rule", "transformer", or "ensemble"
        self.transformer_model_name = config.get("transformer_model", "ProsusAI/finbert")  # Default to FinBERT
        self.transformer_pipeline = None
        if self.model_type in ("transformer", "ensemble"):
            try:
                self.transformer_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=self.transformer_model_name,
                    tokenizer=self.transformer_model_name
                )
                logger.info(f"Loaded transformer model: {self.transformer_model_name}")
            except Exception as e:
                logger.warning(f"Could not load transformer model: {e}")
                self.transformer_pipeline = None

        # Load lexicon-based sentiment dictionaries for multiple languages
        self._load_lexicons()
        self.supported_languages = ["english", "spanish", "french", "german", "italian", "dutch"]
        logger.info("Sentiment analyzer initialized")
        
    def _load_lexicons(self):
        """Load sentiment lexicons for lexicon-based sentiment analysis."""
        # TODO: Load actual financial sentiment lexicons for each language
        # For now, use a simple mock lexicon for all supported languages (copy English)
        self.lexicons = {}
        for lang in ["english", "spanish", "french", "german", "italian", "dutch"]:
            self.lexicons[lang] = {
                "positive_words": {
                    'bullish', 'uptrend', 'growth', 'profit', 'gain', 'positive',
                    'increase', 'rise', 'up', 'higher', 'strong', 'strength',
                    'opportunity', 'promising', 'outperform', 'beat', 'exceed',
                    'improvement', 'recovery', 'rally', 'support', 'buy', 'long',
                    'upgrade', 'optimistic', 'confident', 'successful', 'innovative',
                    'leadership', 'momentum', 'efficient', 'robust', 'breakthrough'
                },
                "negative_words": {
                    'bearish', 'downtrend', 'decline', 'loss', 'negative', 'decrease',
                    'fall', 'down', 'lower', 'weak', 'weakness', 'risk', 'concerning',
                    'underperform', 'miss', 'below', 'deterioration', 'downturn',
                    'resistance', 'sell', 'short', 'downgrade', 'pessimistic', 'worried',
                    'disappointing', 'struggling', 'challenging', 'inefficient',
                    'vulnerable', 'slowdown', 'competitive_pressure', 'overvalued'
                },
                "intensifiers": {
                    'very', 'extremely', 'significantly', 'substantially', 'highly',
                    'strongly', 'sharply', 'considerably', 'notably', 'markedly',
                    'exceptionally', 'remarkably', 'dramatically', 'decidedly',
                    'materially', 'massively', 'vastly', 'immensely', 'tremendously'
                },
                "negators": {
                    'not', 'no', 'never', 'none', 'neither', 'nor', 'nothing',
                    'nowhere', 'hardly', 'barely', 'scarcely', 'doesn\'t', 'don\'t',
                    'didn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t',
                    'haven\'t', 'hadn\'t', 'won\'t', 'wouldn\'t', 'can\'t', 'cannot',
                    'couldn\'t', 'shouldn\'t', 'without', 'despite', 'in spite of'
                }
            }
        logger.info("Loaded sentiment lexicons for all supported languages")
        
    def analyze_sentiment(self, text: str, language: Optional[str] = None, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text using the selected model type.
        Optionally, use context for context-aware sentiment analysis (future extension).
        """
        # Placeholder for context-aware sentiment analysis
        # In the future, this could aggregate sentiment over context windows, use entity linking, etc.
        # For now, context is ignored.
        return self._analyze_sentiment_core(text, language)
    
    def _analyze_sentiment_core(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        # (Paste the previous implementation of analyze_sentiment here, unchanged)
        if not text or not isinstance(text, str):
            return {
                'score': 0.0,
                'polarity': 'neutral',
                'positive_words': [],
                'negative_words': [],
                'confidence': 0.0
            }
        # Preprocess text and detect language if not provided
        if language is None:
            try:
                from langdetect import detect, LangDetectException
                lang_code = detect(text)
            except Exception:
                lang_code = "en"
            lang_map = {
                "en": "english",
                "es": "spanish",
                "fr": "french",
                "de": "german",
                "it": "italian",
                "nl": "dutch"
            }
            language = lang_map.get(lang_code, "english")
        if language not in self.supported_languages:
            language = "english"
        preprocessed_text = self.preprocessor.preprocess(text)
        tokens = preprocessed_text.split()
    
        if self.model_type == "rule":
            # Rule-based sentiment (existing logic)
            lexicon = self.lexicons.get(language, self.lexicons["english"])
            positive_words = lexicon["positive_words"]
            negative_words = lexicon["negative_words"]
            intensifiers = lexicon["intensifiers"]
            negators = lexicon["negators"]
            positive_matches = []
            negative_matches = []
            negation_active = False
            negation_window = self.config.get('negation_window', 4)
            negation_counter = 0
            for i, token in enumerate(tokens):
                if token in negators:
                    negation_active = True
                    negation_counter = 0
                    continue
                if negation_active:
                    negation_counter += 1
                    if negation_counter >= negation_window:
                        negation_active = False
                if token in positive_words:
                    if negation_active:
                        negative_matches.append(token)
                    else:
                        positive_matches.append(token)
                elif token in negative_words:
                    if negation_active:
                        positive_matches.append(token)
                    else:
                        negative_matches.append(token)
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
# (Removed misplaced analyze_context method here; will re-insert at correct indentation below)

                elif score < -0.1:
                    polarity = 'negative'
                else:
                    polarity = 'neutral'
                confidence = min(1.0, total_count / (len(tokens) + 1))
            return {
                'score': score,
                'polarity': polarity,
                'positive_words': positive_matches,
                'negative_words': negative_matches,
                'confidence': confidence
            }
        elif self.model_type == "transformer" and self.transformer_pipeline is not None:
            # Transformer-based sentiment (e.g., FinBERT)
            try:
                result = self.transformer_pipeline(preprocessed_text)
                label = result[0]['label'].lower()
                score = result[0]['score']
                if "positive" in label:
                    polarity = "positive"
                    sentiment_score = score
                elif "negative" in label:
                    polarity = "negative"
                    sentiment_score = -score
                else:
                    polarity = "neutral"
                    sentiment_score = 0.0
                return {
                    'score': sentiment_score,
                    'polarity': polarity,
                    'positive_words': [],
                    'negative_words': [],
                    'confidence': score
                }
            except Exception as e:
                logger.warning(f"Transformer sentiment analysis failed: {e}")
                # Fallback to rule-based
                return self._analyze_sentiment_core(text, language=language)
        elif self.model_type == "ensemble":
            # Ensemble: combine rule-based and transformer-based (if available)
            rule_result = self._analyze_sentiment_core(text, language=language)
            transformer_result = {'score': 0.0, 'confidence': 0.0}
            if self.transformer_pipeline is not None:
                try:
                    t_result = self.transformer_pipeline(preprocessed_text)
                    t_label = t_result[0]['label'].lower()
                    t_score = t_result[0]['score']
                    if "positive" in t_label:
                        t_sentiment = t_score
                    elif "negative" in t_label:
                        t_sentiment = -t_score
                    else:
                        t_sentiment = 0.0
                    transformer_result = {'score': t_sentiment, 'confidence': t_score}
                except Exception as e:
                    logger.warning(f"Transformer sentiment analysis failed in ensemble: {e}")
            # Simple average ensemble
            avg_score = (rule_result['score'] + transformer_result['score']) / 2
            avg_conf = (rule_result['confidence'] + transformer_result['confidence']) / 2
            if avg_score > 0.1:
                polarity = 'positive'
            elif avg_score < -0.1:
                polarity = 'negative'
            else:
                polarity = 'neutral'
            return {
                'score': avg_score,
                'polarity': polarity,
                'positive_words': rule_result.get('positive_words', []),
                'negative_words': rule_result.get('negative_words', []),
                'confidence': avg_conf
            }
        else:
            logger.warning("No valid sentiment model configured, falling back to rule-based.")
    def analyze_context(self, segments: list, language: Optional[str] = None, entity: Optional[str] = None) -> Dict[str, Any]:
        """
        Context-aware sentiment analysis: analyze a list of text segments (sentences/paragraphs),
        aggregate their sentiment, and optionally link to a specific entity.

        Args:
            segments: List of text segments (sentences, paragraphs, etc.)
            language: Optional language code
            entity: Optional entity to link sentiment to (e.g., asset symbol)

        Returns:
            {
                "segment_sentiments": [per-segment sentiment dicts],
                "aggregate": {
                    "score": float,
                    "polarity": str,
                    "confidence": float
                },
                "entity": entity (if provided)
            }
        """
        segment_sentiments = [self.analyze_sentiment(seg, language=language) for seg in segments]
        # Aggregate: simple average of scores and confidences
        if segment_sentiments:
            avg_score = sum(s["score"] for s in segment_sentiments) / len(segment_sentiments)
            avg_conf = sum(s["confidence"] for s in segment_sentiments) / len(segment_sentiments)
            if avg_score > 0.1:
                polarity = "positive"
            elif avg_score < -0.1:
                polarity = "negative"
            else:
                polarity = "neutral"
        else:
            avg_score = 0.0
            avg_conf = 0.0
            polarity = "neutral"
        return {
            "segment_sentiments": segment_sentiments,
            "aggregate": {
                "score": avg_score,
                "polarity": polarity,
                "confidence": avg_conf
            },
            "entity": entity
        }
# Removed stray return statement
        
    def batch_analyze_sentiment(self, texts: List[str], language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of input texts to analyze
            language: Optional language code (applies to all texts). If None, auto-detect per text.
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        return [self.analyze_sentiment(text, language=language) for text in texts]
    
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
    RELATIONSHIP_KEYWORDS = [
        # Acquisition
        ("acquired", "acquisition"),
        ("acquisition", "acquisition"),
        ("acquire", "acquisition"),
        ("buy", "acquisition"),
        ("bought", "acquisition"),
        ("purchased", "acquisition"),
        # Merger
        ("merged with", "merger"),
        ("merger", "merger"),
        ("merge", "merger"),
        ("combined with", "merger"),
        # Partnership
        ("partnered with", "partnership"),
        ("partnership", "partnership"),
        ("partner", "partnership"),
        ("collaborated with", "partnership"),
        ("collaboration", "partnership"),
        # Lawsuit
        ("sued", "lawsuit"),
        ("lawsuit", "lawsuit"),
        ("sue", "lawsuit"),
        ("legal action", "lawsuit"),
        # Announcement
        ("announced", "announcement"),
        ("announcement", "announcement"),
        # Investigation
        ("investigated", "investigation"),
        ("investigation", "investigation"),
        ("probe", "investigation"),
        # Appointment/Resignation
        ("appointed", "appointment"),
        ("resigned", "resignation"),
        # Product Launch
        ("launched", "product_launch"),
        ("launch", "product_launch"),
        ("introduced", "product_launch"),
        # Split/Dividend/Bankruptcy/Other
        ("split", "split"),
        ("dividend", "dividend"),
        ("bankruptcy", "bankruptcy"),
        ("settlement", "settlement"),
        ("fine", "fine"),
        ("penalty", "penalty"),
        ("approval", "approval"),
        ("rejection", "rejection"),
    ]

    def extract_relationships(self, text: str) -> list:
        """
        Extract relationships between entities (companies/tickers) in the text.
        Returns a list of dicts: {entity1, relationship, entity2, context}
        """
        relationships = []
        if not text or not isinstance(text, str):
            return relationships
        # Find all company names and tickers in the text
        entities = set()
        for m in self.company_pattern.finditer(text):
            entities.add(m.group(0))
        for m in self.ticker_pattern.finditer(text):
            if m.group(0) in self.asset_map:
                entities.add(m.group(0))
        # For each relationship keyword, look for patterns like "A acquired B" or "A and B merged"
        for keyword, rel_type in self.RELATIONSHIP_KEYWORDS:
            # Pattern: entity1 <keyword> entity2 OR entity1 and entity2 <keyword>
            pattern1 = re.compile(rf"(\b[A-Z][A-Za-z0-9&\-. ]{{2,}}\b)[^\.]{{0,40}}{keyword}[^\.]{{0,40}}(\b[A-Z][A-Za-z0-9&\-. ]{{2,}}\b)", re.IGNORECASE)
            pattern2 = re.compile(rf"(\b[A-Z][A-Za-z0-9&\-. ]{{2,}}\b)\s+and\s+(\b[A-Z][A-Za-z0-9&\-. ]{{2,}}\b)[^\.]{{0,40}}{keyword}", re.IGNORECASE)
            for match in pattern1.finditer(text):
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()
        # Fallback: co-occurrence + keyword in the same sentence (treat any capitalized word as entity)
        if not relationships:
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                s_norm = sentence.strip()
                ents_in_sentence = re.findall(r'\b[A-Z][A-Za-z0-9&\-.]*\b', s_norm)
                for keyword, rel_type in self.RELATIONSHIP_KEYWORDS:
                    if keyword in s_norm.lower() and len(ents_in_sentence) >= 2:
                        print(f"DEBUG: sentence='{sentence.strip()}', entities={ents_in_sentence}, keyword='{keyword}', rel_type='{rel_type}'")
                        for i in range(len(ents_in_sentence)):
                            for j in range(i+1, len(ents_in_sentence)):
                                relationships.append({
                                    "entity1": ents_in_sentence[i],
                                    "relationship": rel_type,
                                    "entity2": ents_in_sentence[j],
                                    "context": sentence.strip()
                                })
                if entity1 != entity2 and entity1 in entities and entity2 in entities:
                    relationships.append({
                        "entity1": entity1,
                        "relationship": rel_type,
                        "entity2": entity2,
                        "context": match.group(0)
                    })
            for match in pattern2.finditer(text):
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()
                if entity1 != entity2 and entity1 in entities and entity2 in entities:
                    relationships.append({
                        "entity1": entity1,
                        "relationship": rel_type,
                        "entity2": entity2,
                        "context": match.group(0)
                    })
        # Fallback: co-occurrence + keyword in the same sentence
        if not relationships:
            # Split text into sentences (simple split on period for now)
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                ents_in_sentence = [e for e in entities if e in sentence]
                for keyword, rel_type in self.RELATIONSHIP_KEYWORDS:
                    if keyword in sentence.lower() and len(ents_in_sentence) >= 2:
                        for i in range(len(ents_in_sentence)):
                            for j in range(i+1, len(ents_in_sentence)):
                                relationships.append({
                                    "entity1": ents_in_sentence[i],
                                    "relationship": rel_type,
                                    "entity2": ents_in_sentence[j],
                                    "context": sentence.strip()
                                })
        # Fallback: co-occurrence + keyword in the same sentence (more robust)
        if not relationships:
            sentences = re.split(r'[.!?]', text)
            norm_entities = [e.lower().strip() for e in entities]
            for sentence in sentences:
                s_norm = sentence.lower()
                ents_in_sentence = [e for e in entities if e.lower().strip() in s_norm]
                for keyword, rel_type in self.RELATIONSHIP_KEYWORDS:
                    if keyword in s_norm and len(ents_in_sentence) >= 2:
                        for i in range(len(ents_in_sentence)):
                            for j in range(i+1, len(ents_in_sentence)):
                                relationships.append({
                                    "entity1": ents_in_sentence[i],
                                    "relationship": rel_type,
                                    "entity2": ents_in_sentence[j],
                                    "context": sentence.strip()
                                })
        # Fallback: co-occurrence + keyword in the same sentence (treat any capitalized word as entity)
        if not relationships:
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                s_norm = sentence.strip()
                # Find all capitalized words (potential entities)
                ents_in_sentence = re.findall(r'\b[A-Z][A-Za-z0-9&\-.]*\b', s_norm)
                for keyword, rel_type in self.RELATIONSHIP_KEYWORDS:
                    if keyword in s_norm.lower() and len(ents_in_sentence) >= 2:
                        for i in range(len(ents_in_sentence)):
                            for j in range(i+1, len(ents_in_sentence)):
                                relationships.append({
                                    "entity1": ents_in_sentence[i],
                                    "relationship": rel_type,
                                    "entity2": ents_in_sentence[j],
                                    "context": sentence.strip()
                                })
        return relationships
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
            'TSLA': ['tesla', 'tsla'],
        }
        
        # Create reverse mapping for lookup
        self.asset_reverse_map = {}
        for symbol, names in self.asset_map.items():
            for name in names:
                self.asset_reverse_map[name] = symbol
                
        # Financial terms to recognize
        self.financial_terms = {
            # Metrics
            'market', 'stock', 'bond', 'crypto', 'cryptocurrency', 'token', 'coin',
            'exchange', 'trading', 'investment', 'investor', 'portfolio', 'asset',
            'price', 'value', 'volatility', 'volume', 'liquidity', 'market cap',
            'p/e ratio', 'pe ratio', 'eps', 'earnings per share', 'dividend yield',
            'dividend', 'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'inflation', 'deflation', 'recession', 'depression', 'recovery',
            'interest rate', 'fed', 'central bank', 'regulation', 'sec',
            'securities and exchange commission', 'cftc', 'commodity futures trading commission',
            'cash flow', 'operating margin', 'gross margin', 'net income', 'ebitda',
            'debt', 'equity', 'return on equity', 'return on assets', 'beta', 'alpha',
            'dividend payout', 'book value', 'market value', 'enterprise value',
            'split', 'reverse split', 'buyback', 'repurchase', 'guidance', 'forecast',
        }
        self.financial_events = {
            'earnings call', 'dividend announcement', 'merger', 'acquisition', 'ipo',
            'bankruptcy', 'lawsuit', 'settlement', 'regulatory action', 'investigation',
            'upgrade', 'downgrade', 'buyback', 'repurchase', 'split', 'reverse split',
            'guidance', 'forecast', 'product launch', 'partnership', 'layoff', 'scandal',
            'resignation', 'appointment', 'expansion', 'closure', 'delisting', 'listing',
            'approval', 'rejection', 'fine', 'penalty', 'settlement', 'restatement',
            'fraud', 'hack', 'breach', 'recall', 'strike', 'protest', 'boycott'
        }
        logger.info("Loaded entity dictionaries")
        
    def _compile_patterns(self):
        """Compile regex patterns for entity extraction."""
        # Pattern for asset symbols (e.g., $BTC, $ETH)
        self.symbol_pattern = re.compile(r'\$([A-Z]{2,5})')
        # Pattern for plain tickers (e.g., AAPL, TSLA) - must be surrounded by word boundaries
        self.ticker_pattern = re.compile(r'\b([A-Z]{2,5})\b')
        # Pattern for company names (e.g., Apple, Microsoft) - match from asset_reverse_map
        self.company_pattern = re.compile(r'\b(' + '|'.join(re.escape(name) for name in self.asset_reverse_map.keys()) + r')\b', re.IGNORECASE)
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
        
        # Helper to wrap entity with confidence
        def wrap_conf(entity, conf=1.0):
            return {"value": entity, "confidence": conf}
        
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
                
        # Extract financial metrics
        financial_metrics = set()
        for term in self.financial_terms:
            if term in preprocessed_text:
                financial_metrics.add(term)
        # Extract financial events
        financial_events = set()
        for event in self.financial_events:
            if event in preprocessed_text:
                financial_events.add(event)
                
        # Extract prices
        prices = self.price_pattern.findall(original_text)
        
        # Extract cashtags
        cashtags = self.cashtag_pattern.findall(original_text)
        
        # Extract plain tickers (e.g., AAPL, TSLA)
        ticker_matches = set(self.ticker_pattern.findall(original_text))
        for ticker in ticker_matches:
            if ticker in self.asset_map:
                asset_symbols.add(ticker)
        # Extract company names (e.g., Apple, Microsoft)
        company_matches = set(m.group(0) for m in self.company_pattern.finditer(original_text))
        company_names = set()
        for name in company_matches:
            norm_name = name.lower()
            if norm_name in self.asset_reverse_map:
                company_names.add(name)
                asset_symbols.add(self.asset_reverse_map[norm_name])
        return {
            'asset_symbols': [wrap_conf(sym, 1.0) for sym in asset_symbols],
            'company_names': [wrap_conf(name, 1.0) for name in company_names],
            'financial_metrics': [wrap_conf(m, 1.0) for m in financial_metrics],
            'financial_events': [wrap_conf(e, 1.0) for e in financial_events],
            'prices': [wrap_conf(p, 1.0) for p in prices],
            'cashtags': [wrap_conf(c, 1.0) for c in cashtags]
        }
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
        
    def process_text(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single text through the complete NLP pipeline.
        
        Args:
            text: Input text to process
            language: Optional language code (e.g., 'english', 'spanish'). If None, auto-detect.
            
        Returns:
            Dictionary with combined results from all pipeline components
        """
        # Preprocess text
        preprocessed_text = self.preprocessor.preprocess(text)
        
        # Analyze sentiment
        sentiment_results = self.sentiment_analyzer.analyze_sentiment(text, language=language)
        
        # Extract entities
        entity_results = self.entity_extractor.extract_entities(text)
        
        # Combine results
        return {
            'preprocessed_text': preprocessed_text,
            'sentiment': sentiment_results,
            'entities': entity_results
        }
        
    def batch_process(self, texts: List[str], language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of texts through the complete NLP pipeline.
        
        Args:
            texts: List of input texts to process
            language: Optional language code (applies to all texts). If None, auto-detect per text.
            
        Returns:
            List of dictionaries with combined results from all pipeline components
        """
        return [self.process_text(text, language=language) for text in texts]
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, language: Optional[str] = None) -> pd.DataFrame:
        """
        Process a DataFrame with text data through the complete NLP pipeline.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text data
            language: Optional language code (applies to all texts). If None, auto-detect per text.
            
        Returns:
            DataFrame with added columns for all pipeline components
        """
        if text_column not in df.columns:
            logger.error(f"Text column '{text_column}' not found in DataFrame")
            return df
            
        # Process with sentiment analyzer
        if language is not None:
            df['sentiment'] = self.sentiment_analyzer.batch_analyze_sentiment(df[text_column].tolist(), language=language)
        else:
            df['sentiment'] = self.sentiment_analyzer.batch_analyze_sentiment(df[text_column].tolist())
        
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
