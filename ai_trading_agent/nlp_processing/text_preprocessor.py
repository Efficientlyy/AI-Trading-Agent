"""
Text Preprocessing Module for AI Trading Agent NLP Pipeline.

This module provides text preprocessing capabilities including:
- Text normalization (lowercasing, punctuation removal)
- Tokenization for multiple languages
- Entity detection for financial terms
- Emoji handling
- Contraction expansion
- Language detection and language-specific preprocessing
- Slang normalization
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Optional, Union, Set
import logging
from dataclasses import dataclass, field
import emoji

# Try importing language-specific libraries with graceful fallbacks
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    # Download necessary NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    nltk_available = True
except ImportError:
    nltk_available = False
    logging.warning("NLTK not available. Some advanced preprocessing features will be disabled.")

try:
    import spacy
    # Don't load models yet, only when needed
    spacy_available = True
except ImportError:
    spacy_available = False
    logging.warning("Spacy not available. Some advanced preprocessing features will be disabled.")

try:
    from langdetect import detect as detect_language
    langdetect_available = True
except ImportError:
    langdetect_available = False
    logging.warning("langdetect not available. Language detection will be disabled.")

try:
    from transformers import AutoTokenizer
    transformers_available = True
except ImportError:
    transformers_available = False
    logging.warning("transformers not available. Advanced tokenization will be disabled.")

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class TextPreprocessingConfig:
    """Configuration for text preprocessing pipeline."""
    
    # Basic preprocessing options
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_whitespace: bool = True
    remove_urls: bool = True
    remove_html_tags: bool = True
    
    # Advanced preprocessing options
    expand_contractions: bool = True
    handle_emojis: bool = True  # convert to text, remove, or keep
    emoji_handling: str = "text"  # "text", "remove", "keep"
    normalize_unicode: bool = True
    unicode_form: str = "NFKD"  # Unicode normalization form
    
    # Tokenization options
    tokenize: bool = False  # Set to True to return tokens instead of string
    tokenizer: str = "nltk"  # "nltk", "spacy", "transformers", "split"
    
    # Language-specific options
    language: str = "auto"  # "auto" for automatic detection, or specific language code
    remove_stopwords: bool = False
    stopwords_languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Financial text specific options
    detect_tickers: bool = True
    detect_financial_terms: bool = True
    normalize_slang: bool = True
    
    # Model configuration
    spacy_model: str = "en_core_web_sm"  # Default English model
    transformers_model: str = "bert-base-uncased"  # Default BERT tokenizer
    
    # Additional options
    additional_patterns_remove: List[str] = field(default_factory=list)
    additional_patterns_keep: List[str] = field(default_factory=list)
    replacements: Dict[str, str] = field(default_factory=dict)  # Custom replacements


class TextPreprocessor:
    """
    Text preprocessor with multilingual support and financial text specialization.
    """
    
    def __init__(self, config: Optional[TextPreprocessingConfig] = None):
        """
        Initialize the text preprocessor with the provided configuration.
        
        Args:
            config: Configuration for text preprocessing
        """
        self.config = config or TextPreprocessingConfig()
        
        # Initialize language resources
        self._spacy_models = {}
        self._transformers_tokenizers = {}
        self._stopwords_sets = {}
        
        # Initialize financial dictionaries
        self._initialize_financial_dictionaries()
        
        # Initialize contraction mappings
        self._initialize_contractions()
        
        # Initialize slang dictionary
        self._initialize_slang_dictionary()
        
        logger.info(f"Initialized TextPreprocessor with config: {self.config}")
    
    def _initialize_financial_dictionaries(self):
        """Initialize financial dictionaries including ticker symbols and financial terms."""
        # Common stock tickers (top companies)
        self.tickers = {
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", "JPM",
            "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM", "COST"
        }
        
        # Common crypto tickers
        self.crypto_tickers = {
            "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "DOGE", "SHIB", "AVAX", "MATIC",
            "BNB", "LTC", "LINK", "UNI", "ALGO", "ATOM", "XLM", "FTT", "FTM", "NEAR"
        }
        
        # Common financial terms
        self.financial_terms = {
            "bullish", "bearish", "long", "short", "call", "put", "option", "stock", "share",
            "market", "price", "trade", "buy", "sell", "hold", "volume", "volatility",
            "dividend", "yield", "earnings", "revenue", "growth", "profit", "loss", "margin",
            "rally", "crash", "correction", "resistance", "support", "breakout", "dip", "ath",
            "atl", "moon", "dump", "pump", "fud", "hodl", "whale", "portfolio", "market cap"
        }
        
        # Currency symbols
        self.currency_symbols = set([
            "$", "€", "£", "¥", "₹", "₽", "₩", "₿", "฿", "₫", "₴", "₱"
        ])
        
        # Regex patterns for financial entities
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})')  # Matches $AAPL
        self.price_pattern = re.compile(r'(\$\d+(\.\d+)?)')  # Matches $123.45
    
    def _initialize_contractions(self):
        """Initialize English contractions dictionary."""
        self.contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot", "could've": "could have",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
            "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will",
            "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
            "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "ma'am": "madam", "mightn't": "might not", "might've": "might have",
            "mustn't": "must not", "must've": "must have", "needn't": "need not", "o'clock": "of the clock",
            "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
            "should've": "should have", "shouldn't": "should not", "that'd": "that would",
            "that's": "that is", "there'd": "there would", "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not",
            "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
            "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is",
            "what've": "what have", "when's": "when is", "where'd": "where did", "where's": "where is",
            "who'd": "who would", "who'll": "who will", "who're": "who are", "who's": "who is",
            "who've": "who have", "why'll": "why will", "why're": "why are", "why's": "why is",
            "won't": "will not", "would've": "would have", "wouldn't": "would not", "y'all": "you all"
        }
    
    def _initialize_slang_dictionary(self):
        """Initialize dictionary for common internet/finance slang."""
        self.slang = {
            "hodl": "hold", "fud": "fear uncertainty doubt", "rekt": "wrecked", 
            "tendies": "profits", "stonks": "stocks", "gainz": "gains", "noob": "newcomer",
            "btfd": "buy the dip", "fomo": "fear of missing out", "yolo": "risky investment",
            "mooning": "price increasing", "bagholder": "investor with losses", "diamond hands": "holding through volatility",
            "paper hands": "selling quickly", "ath": "all time high", "atl": "all time low",
            "dyor": "do your own research", "lambo": "lamborghini (success)", "whale": "large investor",
            "rugpull": "scam", "ngmi": "not going to make it", "wagmi": "we're all going to make it",
            "wen moon": "when will price increase", "gm": "good morning", "ser": "sir",
            "degen": "degenerate gambler", "lfg": "let's go", "nfa": "not financial advice"
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the provided text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        if not text or len(text.strip()) < 10:
            return "en"  # Default to English for short texts
            
        if langdetect_available:
            try:
                return detect_language(text)
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
                return "en"
        else:
            return "en"  # Default to English if langdetect not available
    
    def load_language_resources(self, language: str):
        """
        Load language-specific resources like stopwords.
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
        """
        # Load stopwords for requested language if needed and not already loaded
        if self.config.remove_stopwords and language not in self._stopwords_sets:
            if nltk_available:
                try:
                    self._stopwords_sets[language] = set(stopwords.words(language))
                    logger.debug(f"Loaded stopwords for {language}")
                except Exception as e:
                    logger.warning(f"Failed to load stopwords for {language}: {e}")
                    self._stopwords_sets[language] = set()
            else:
                self._stopwords_sets[language] = set()
        
        # Load spaCy model if needed and not already loaded
        if self.config.tokenizer == "spacy" and language not in self._spacy_models and spacy_available:
            # Map language code to spaCy model
            lang_to_model = {
                "en": "en_core_web_sm",
                "es": "es_core_news_sm",
                "fr": "fr_core_news_sm",
                "de": "de_core_news_sm",
                "pt": "pt_core_news_sm",
                "it": "it_core_news_sm",
                "nl": "nl_core_news_sm",
                "ja": "ja_core_news_sm",
                "zh": "zh_core_web_sm"
            }
            
            model_name = lang_to_model.get(language, "en_core_web_sm")
            
            try:
                import spacy
                self._spacy_models[language] = spacy.load(model_name)
                logger.debug(f"Loaded spaCy model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model for {language}: {e}")
                # Fall back to English model
                try:
                    self._spacy_models[language] = spacy.load("en_core_web_sm")
                except Exception:
                    logger.error("Failed to load fallback spaCy model")
        
        # Load transformers tokenizer if needed
        if self.config.tokenizer == "transformers" and transformers_available:
            if "default" not in self._transformers_tokenizers:
                try:
                    self._transformers_tokenizers["default"] = AutoTokenizer.from_pretrained(
                        self.config.transformers_model
                    )
                    logger.debug(f"Loaded transformers tokenizer: {self.config.transformers_model}")
                except Exception as e:
                    logger.error(f"Failed to load transformers tokenizer: {e}")
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        return unicodedata.normalize(self.config.unicode_form, text)
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation characters."""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags."""
        html_pattern = re.compile(r'<.*?>')
        return html_pattern.sub('', text)
    
    def _handle_emojis(self, text: str) -> str:
        """Handle emojis according to configuration."""
        if self.config.emoji_handling == "text":
            # Replace emojis with their text description
            return emoji.demojize(text)
        elif self.config.emoji_handling == "remove":
            # Remove emojis
            return emoji.replace_emoji(text, replace='')
        else:
            # Keep emojis as is
            return text
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions (e.g., don't -> do not)."""
        for contraction, expansion in self.contractions.items():
            pattern = re.compile(r'\b' + contraction + r'\b', re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text
    
    def _normalize_slang(self, text: str) -> str:
        """Normalize internet and financial slang."""
        words = text.split()
        normalized_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.slang:
                normalized_words.append(self.slang[word_lower])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def _detect_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """Detect financial entities in text."""
        results = {
            "tickers": [],
            "prices": [],
            "financial_terms": []
        }
        
        # Detect tickers with $ symbol
        ticker_matches = self.ticker_pattern.findall(text)
        if ticker_matches:
            results["tickers"] = ticker_matches
        
        # Detect prices
        price_matches = self.price_pattern.findall(text)
        if price_matches:
            results["prices"] = [price[0] for price in price_matches]
        
        # Detect financial terms
        words = set(text.lower().split())
        financial_term_matches = words.intersection(self.financial_terms)
        if financial_term_matches:
            results["financial_terms"] = list(financial_term_matches)
        
        return results
    
    def _remove_stopwords(self, tokens: List[str], language: str) -> List[str]:
        """Remove stopwords for given language."""
        if language not in self._stopwords_sets:
            self.load_language_resources(language)
        
        return [token for token in tokens if token.lower() not in self._stopwords_sets.get(language, set())]
    
    def _tokenize(self, text: str, language: str) -> List[str]:
        """Tokenize text with the configured tokenizer."""
        if self.config.tokenizer == "nltk" and nltk_available:
            return word_tokenize(text, language=language[:2])  # nltk uses 2-letter codes
        
        elif self.config.tokenizer == "spacy" and spacy_available:
            if language not in self._spacy_models:
                self.load_language_resources(language)
            
            if language in self._spacy_models:
                doc = self._spacy_models[language](text)
                return [token.text for token in doc]
            else:
                # Fall back to simple tokenization
                return text.split()
        
        elif self.config.tokenizer == "transformers" and transformers_available:
            if "default" not in self._transformers_tokenizers:
                self.load_language_resources(language)
            
            if "default" in self._transformers_tokenizers:
                return self._transformers_tokenizers["default"].tokenize(text)
            else:
                # Fall back to simple tokenization
                return text.split()
        
        else:
            # Simple split-based tokenization
            return text.split()
    
    def preprocess(self, text: Union[str, List[str]], language: Optional[str] = None) -> Union[str, List[str], List[List[str]]]:
        """
        Preprocess text according to configuration.
        
        Args:
            text: Text string or list of strings to preprocess
            language: Optional language code (if None, auto-detection is used)
            
        Returns:
            Preprocessed text as string, list of strings, or tokens depending on configuration
        """
        # Handle single text or list of texts
        if isinstance(text, list):
            return [self.preprocess(t, language) for t in text]
        
        if not text or not isinstance(text, str):
            return "" if not self.config.tokenize else []
        
        # Auto-detect language if not provided and configuration requires language detection
        if language is None and self.config.language == "auto":
            language = self.detect_language(text)
        elif language is None:
            language = self.config.language
        
        # Ensure language resources are loaded
        self.load_language_resources(language)
        
        # Apply preprocessing steps based on configuration
        
        # Step 1: Unicode normalization
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Step 2: Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Step 3: Handle emojis
        if self.config.handle_emojis:
            text = self._handle_emojis(text)
        
        # Step 4: Remove URLs and HTML tags
        if self.config.remove_urls:
            text = self._remove_urls(text)
            
        if self.config.remove_html_tags:
            text = self._remove_html_tags(text)
        
        # Step 5: Expand contractions
        if self.config.expand_contractions:
            text = self._expand_contractions(text)
            
        # Step 6: Normalize slang
        if self.config.normalize_slang:
            text = self._normalize_slang(text)
        
        # Step 7: Apply custom replacements if any
        for pattern, replacement in self.config.replacements.items():
            text = text.replace(pattern, replacement)
        
        # Step 8: Remove any additional patterns specified in config
        for pattern in self.config.additional_patterns_remove:
            text = re.sub(pattern, '', text)
        
        # Step 9: Remove punctuation (but keep currency symbols if detecting financial terms)
        if self.config.remove_punctuation:
            if self.config.detect_financial_terms:
                # Remove punctuation except currency symbols
                for symbol in string.punctuation:
                    if symbol not in self.currency_symbols:
                        text = text.replace(symbol, ' ')
            else:
                text = self._remove_punctuation(text)
        
        # Step 10: Remove numbers
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Step 11: Remove extra whitespace
        if self.config.remove_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 12: Tokenize if required
        if self.config.tokenize:
            tokens = self._tokenize(text, language)
            
            # Step 13: Remove stopwords if requested
            if self.config.remove_stopwords:
                tokens = self._remove_stopwords(tokens, language)
                
            return tokens
        
        return text
    
    def batch_preprocess(self, texts: List[str]) -> List[Union[str, List[str]]]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detected tickers, prices, and financial terms
        """
        # Minimal preprocessing to keep important symbols like $ for entity detection
        prep_config = TextPreprocessingConfig(
            lowercase=False, 
            remove_punctuation=False,
            remove_numbers=False,
            expand_contractions=True,
            handle_emojis=True,
            emoji_handling="remove",
            normalize_unicode=True
        )
        
        # Create temporary preprocessor with minimal config
        temp_preprocessor = TextPreprocessor(config=prep_config)
        processed_text = temp_preprocessor.preprocess(text)
        
        # Now detect financial entities
        return self._detect_financial_entities(processed_text)
    
    def add_custom_tickers(self, tickers: Set[str]):
        """
        Add custom ticker symbols to the detector.
        
        Args:
            tickers: Set of ticker symbols to add
        """
        self.tickers.update(tickers)
    
    def add_custom_financial_terms(self, terms: Set[str]):
        """
        Add custom financial terms to the detector.
        
        Args:
            terms: Set of financial terms to add
        """
        self.financial_terms.update(terms)
    
    def add_custom_slang(self, slang_dict: Dict[str, str]):
        """
        Add custom slang terms to the normalizer.
        
        Args:
            slang_dict: Dictionary mapping slang terms to normalized forms
        """
        self.slang.update(slang_dict)


# --- Usage Examples ---

def example_sentiment_preprocessing():
    """Example preprocessing for sentiment analysis."""
    # Configuration optimized for sentiment analysis
    sentiment_config = TextPreprocessingConfig(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=False,
        remove_urls=True,
        expand_contractions=True,
        handle_emojis=True,
        emoji_handling="text",
        normalize_unicode=True,
        tokenize=False,  # Keep as string for sentiment analysis
        language="auto",
        remove_stopwords=False,  # Keeping stopwords for sentiment
        normalize_slang=True
    )
    
    preprocessor = TextPreprocessor(config=sentiment_config)
    
    # Example financial text
    text = "$AAPL is 🚀🚀 to the moon! I'm bullish on their new product line. BUY THE DIP! #investing"
    
    processed = preprocessor.preprocess(text)
    print(f"Original: {text}")
    print(f"Processed: {processed}")
    
    entities = preprocessor.extract_financial_entities(text)
    print(f"Financial Entities: {entities}")
    
    return processed


def example_multilingual_preprocessing():
    """Example preprocessing for multilingual text."""
    # Spanish text example
    text_es = "¡Los precios de $BTC están subiendo! Muy bullish para el mercado de criptomonedas."
    
    # French text example
    text_fr = "Les actions $AAPL sont en hausse aujourd'hui. C'est une bonne opportunité d'investissement."
    
    multilingual_config = TextPreprocessingConfig(
        lowercase=True,
        remove_punctuation=True,
        language="auto",  # Auto-detect language
        tokenize=True,  # Return tokens
        remove_stopwords=True  # Remove language-specific stopwords
    )
    
    preprocessor = TextPreprocessor(config=multilingual_config)
    
    # Process both texts
    tokens_es = preprocessor.preprocess(text_es)
    tokens_fr = preprocessor.preprocess(text_fr)
    
    print(f"Spanish Original: {text_es}")
    print(f"Spanish Tokens: {tokens_es}")
    
    print(f"French Original: {text_fr}")
    print(f"French Tokens: {tokens_fr}")
    
    return tokens_es, tokens_fr


if __name__ == "__main__":
    # Run examples
    example_sentiment_preprocessing()
    example_multilingual_preprocessing()