"""News analysis system for cryptocurrency markets.

This module provides functionality for analyzing news articles and
extracting sentiment, topics, entities, and market impact assessments.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import re
import numpy as np
from collections import defaultdict, Counter
import logging
import json
import os

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common.logging import get_logger
from src.common.config import config
from src.common.events import event_bus
from src.analysis_agents.sentiment.nlp_service import NLPService
from src.analysis_agents.news.crypto_news_categorizer import CryptoNewsCategorizer, CryptoNewsTopicGraph


class NewsArticle:
    """Represents a news article with metadata and analysis results."""
    
    def __init__(self, 
                article_id: str,
                title: str,
                content: str,
                url: str,
                source: str,
                published_at: datetime,
                author: Optional[str] = None,
                categories: Optional[List[str]] = None,
                tags: Optional[List[str]] = None):
        """Initialize a news article.
        
        Args:
            article_id: Unique identifier for the article
            title: Article title
            content: Article content
            url: Article URL
            source: News source
            published_at: Publication date and time
            author: Article author
            categories: Article categories
            tags: Article tags
        """
        self.article_id = article_id
        self.title = title
        self.content = content
        self.url = url
        self.source = source
        self.published_at = published_at
        self.author = author
        self.categories = categories or []
        self.tags = tags or []
        
        # Analysis results (will be populated later)
        self.sentiment: Optional[float] = None
        self.entities: List[Dict[str, Any]] = []
        self.topics: List[str] = []
        self.summary: Optional[str] = None
        self.relevance_scores: Dict[str, float] = {}
        self.market_impact: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the article to a dictionary.
        
        Returns:
            Dictionary representation of the article
        """
        return {
            "article_id": self.article_id,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "author": self.author,
            "categories": self.categories,
            "tags": self.tags,
            "sentiment": self.sentiment,
            "entities": self.entities,
            "topics": self.topics,
            "summary": self.summary,
            "relevance_scores": self.relevance_scores,
            "market_impact": self.market_impact
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsArticle':
        """Create a NewsArticle from a dictionary.
        
        Args:
            data: Dictionary representation of the article
            
        Returns:
            NewsArticle instance
        """
        article = cls(
            article_id=data["article_id"],
            title=data["title"],
            content=data["content"],
            url=data["url"],
            source=data["source"],
            published_at=datetime.fromisoformat(data["published_at"]),
            author=data.get("author"),
            categories=data.get("categories", []),
            tags=data.get("tags", [])
        )
        
        article.sentiment = data.get("sentiment")
        article.entities = data.get("entities", [])
        article.topics = data.get("topics", [])
        article.summary = data.get("summary")
        article.relevance_scores = data.get("relevance_scores", {})
        article.market_impact = data.get("market_impact", {})
        
        return article


class NewsAnalyzer:
    """System for analyzing news articles related to cryptocurrency markets.
    
    This class provides functionality for collecting, analyzing, and extracting
    insights from news articles related to cryptocurrencies.
    """
    
    def __init__(self):
        """Initialize the news analyzer."""
        self.logger = get_logger("news_analyzer", "main")
        
        # Configuration
        self.enabled = config.get("news_analyzer.enabled", True)
        self.update_interval = config.get("news_analyzer.update_interval", 3600)  # 1 hour
        self.max_articles = config.get("news_analyzer.max_articles", 1000)
        self.relevance_threshold = config.get("news_analyzer.relevance_threshold", 0.5)
        self.assets = config.get("news_analyzer.assets", ["BTC", "ETH", "SOL", "XRP"])
        
        # NLP components
        self.nlp_service = None
        self.summarizer = None
        self.entity_extractor = None
        self.topic_classifier = None
        
        # Crypto-specific categorization
        self.crypto_categorizer = None
        self.topic_graph = None
        self.use_crypto_categorization = config.get("news_analyzer.use_crypto_categorization", True)
        
        # Data storage
        self.articles: Dict[str, NewsArticle] = {}
        self.article_index: Dict[str, Set[str]] = defaultdict(set)  # Keyword to article IDs
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # Entity to article IDs
        
        # News sources
        self.news_clients = {}
        
        # Graph representation of news and entities
        self.news_graph = nx.Graph()
        
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the news analyzer components.
        
        This method loads NLP models and initializes news API clients.
        """
        if not self.enabled:
            self.logger.info("News analyzer is disabled")
            return
            
        self.logger.info("Initializing news analyzer")
        
        # Initialize NLP service
        self.nlp_service = NLPService()
        await self.nlp_service.initialize()
        
        # Load NLP models
        await self._load_nlp_models()
        
        # Initialize news API clients
        await self._initialize_news_clients()
        
        # Initialize crypto categorization if enabled
        if self.use_crypto_categorization:
            self.logger.info("Initializing cryptocurrency news categorization")
            self.crypto_categorizer = CryptoNewsCategorizer()
            await self.crypto_categorizer.initialize()
            self.topic_graph = CryptoNewsTopicGraph()
            self.logger.info("Cryptocurrency news categorization initialized")
        
        self.is_initialized = True
        self.logger.info("News analyzer initialized")
    
    async def _load_nlp_models(self) -> None:
        """Load NLP models for news analysis."""
        try:
            # Load models in a background thread to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Load summarization model
            self.logger.info("Loading summarization model")
            self.summarizer = await loop.run_in_executor(
                None,
                lambda: pipeline("summarization", model="facebook/bart-large-cnn", max_length=100)
            )
            
            # Load entity extraction model
            self.logger.info("Loading entity extraction model")
            self.entity_extractor = await loop.run_in_executor(
                None,
                lambda: pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
            )
            
            # Load topic classification model
            self.logger.info("Loading topic classification model")
            model_name = "facebook/bart-large-mnli"
            tokenizer = await loop.run_in_executor(
                None,
                lambda: AutoTokenizer.from_pretrained(model_name)
            )
            model = await loop.run_in_executor(
                None,
                lambda: AutoModelForSequenceClassification.from_pretrained(model_name)
            )
            self.topic_classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
            
            self.logger.info("NLP models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading NLP models: {e}")
            
            # Set up fallback methods
            self.summarizer = None
            self.entity_extractor = None
            self.topic_classifier = None
    
    async def _initialize_news_clients(self) -> None:
        """Initialize news API clients."""
        # Initialize with mock news client for now
        self.news_clients["mock"] = MockNewsClient()
        
        # Import the NewsAPIClient here to avoid circular imports
        from src.analysis_agents.news.news_api_client import NewsAPIClient
        from src.analysis_agents.news.cryptocompare_news_client import CryptoCompareNewsClient, MockCryptoCompareNewsClient
        
        # Check if we have news API credentials
        news_api_key = os.getenv("NEWS_API_KEY") or config.get("apis.news_api.key", "")
        if news_api_key:
            # Initialize NewsAPI client
            self.logger.info("Initializing NewsAPI client")
            self.news_clients["newsapi"] = NewsAPIClient(api_key=news_api_key)
        else:
            self.logger.warning("No NewsAPI key found - using mock client instead")
        
        # Check if we have CryptoCompare API credentials
        cryptocompare_api_key = os.getenv("CRYPTOCOMPARE_API_KEY") or config.get("apis.cryptocompare.key", "")
        if cryptocompare_api_key:
            # Initialize CryptoCompare client
            self.logger.info("Initializing CryptoCompare News client")
            self.news_clients["cryptocompare"] = CryptoCompareNewsClient(api_key=cryptocompare_api_key)
        else:
            self.logger.warning("No CryptoCompare API key found - using mock client instead")
            self.news_clients["cryptocompare"] = MockCryptoCompareNewsClient()
    
    async def collect_articles(self, 
                              timeframe: str = "24h", 
                              assets: Optional[List[str]] = None) -> List[NewsArticle]:
        """Collect news articles from various sources.
        
        Args:
            timeframe: Time period to collect articles for (e.g., "24h", "7d")
            assets: List of assets to collect news for
            
        Returns:
            List of collected NewsArticle objects
        """
        if not self.is_initialized:
            self.logger.warning("News analyzer not initialized")
            return []
            
        assets = assets or self.assets
        self.logger.info(f"Collecting news articles for {assets} in the last {timeframe}")
        
        all_articles = []
        
        # Parse timeframe to days
        days = 1
        if timeframe.endswith("h"):
            days = int(timeframe[:-1]) // 24 + 1
        elif timeframe.endswith("d"):
            days = int(timeframe[:-1])
        
        # Collect from each news source
        for source_id, client in self.news_clients.items():
            try:
                if source_id == "newsapi":
                    # Use the specific NewsAPI client methods
                    for asset in assets:
                        # Search for asset-specific news
                        asset_articles = await client.search_crypto_news(
                            asset=asset,
                            days=days,
                            page_size=20
                        )
                        self.logger.info(f"Collected {len(asset_articles)} articles about {asset} from {source_id}")
                        all_articles.extend(asset_articles)
                        
                    # Also get general crypto news
                    general_articles = await client.get_everything(
                        q="cryptocurrency OR blockchain OR crypto",
                        from_date=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                        to_date=datetime.now().strftime("%Y-%m-%d"),
                        page_size=20
                    )
                    # Convert to NewsArticle objects
                    general_news_articles = await client.convert_to_news_articles(general_articles, prefix="newsapi_general")
                    self.logger.info(f"Collected {len(general_news_articles)} general crypto articles from {source_id}")
                    all_articles.extend(general_news_articles)
                else:
                    # Use the generic interface for other clients
                    articles = await client.get_articles(timeframe=timeframe, assets=assets)
                    self.logger.info(f"Collected {len(articles)} articles from {source_id}")
                    all_articles.extend(articles)
            except Exception as e:
                self.logger.error(f"Error collecting articles from {source_id}: {e}")
        
        # Store articles
        for article in all_articles:
            self.articles[article.article_id] = article
        
        return all_articles
    
    async def analyze_articles(self, articles: List[NewsArticle]) -> None:
        """Analyze a list of news articles.
        
        Args:
            articles: List of articles to analyze
        """
        if not articles:
            return
            
        self.logger.info(f"Analyzing {len(articles)} news articles")
        
        # Process articles in batches
        batch_size = 10
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            
            # Create processing tasks
            tasks = []
            for article in batch:
                tasks.append(self._analyze_article(article))
            
            # Process batch
            await asyncio.gather(*tasks)
        
        # Update indices
        self._update_indices(articles)
        
        # Build relationship graph
        self._build_news_graph(articles)
        
        self.logger.info(f"Completed analysis of {len(articles)} news articles")
    
    async def _analyze_article(self, article: NewsArticle) -> None:
        """Analyze a single news article.
        
        Args:
            article: The article to analyze
        """
        try:
            # Extract full text (title + content)
            text = f"{article.title}. {article.content}"
            
            # Analyze sentiment
            if self.nlp_service:
                sentiment_scores = await self.nlp_service.analyze_sentiment([text])
                article.sentiment = sentiment_scores[0] if sentiment_scores else 0.5
            
            # Extract entities
            await self._extract_entities(article)
            
            # Classify topics
            await self._classify_topics(article)
            
            # Generate summary
            await self._generate_summary(article)
            
            # Calculate asset relevance
            await self._calculate_relevance(article)
            
            # Assess market impact
            await self._assess_market_impact(article)
            
            # Perform crypto-specific categorization if enabled
            if self.use_crypto_categorization and self.crypto_categorizer:
                try:
                    # Categorize the article
                    categories = await self.crypto_categorizer.categorize_article(article)
                    
                    # Store categories as property of the article
                    article.crypto_categories = categories
                    
                    # Add to topic graph if we have categories
                    if categories and self.topic_graph:
                        self.topic_graph.add_article(article, categories)
                        
                except Exception as e:
                    self.logger.error(f"Error in crypto categorization for article {article.article_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing article {article.article_id}: {e}")
    
    async def _extract_entities(self, article: NewsArticle) -> None:
        """Extract named entities from an article.
        
        Args:
            article: The article to process
        """
        entities = []
        
        if self.entity_extractor:
            try:
                # Use NER model to extract entities
                text = f"{article.title}. {article.content}"
                
                # Limit text length to avoid token limits
                if len(text) > 5000:
                    text = text[:5000]
                
                # Extract entities
                ner_results = self.entity_extractor(text)
                
                # Group entities
                current_entity = None
                current_type = None
                current_score = 0.0
                
                for token in ner_results:
                    if token["entity"].startswith("B-"):
                        # New entity starts
                        if current_entity:
                            entities.append({
                                "text": current_entity,
                                "type": current_type,
                                "confidence": current_score
                            })
                        
                        current_entity = token["word"]
                        current_type = token["entity"][2:]  # Remove B- prefix
                        current_score = token["score"]
                    
                    elif token["entity"].startswith("I-") and current_entity:
                        # Entity continues
                        current_entity += token["word"].replace("##", "")
                        current_score = (current_score + token["score"]) / 2
                
                # Add the last entity
                if current_entity:
                    entities.append({
                        "text": current_entity,
                        "type": current_type,
                        "confidence": current_score
                    })
                
            except Exception as e:
                self.logger.error(f"Error extracting entities: {e}")
                # Fall back to rule-based extraction
                entities = self._extract_entities_rule_based(article)
        else:
            # Use rule-based entity extraction
            entities = self._extract_entities_rule_based(article)
        
        # Filter and clean entities
        filtered_entities = []
        seen_entities = set()
        
        for entity in entities:
            text = entity["text"].strip()
            
            # Skip short or common words
            if len(text) < 2 or text.lower() in {"the", "a", "an", "and", "or", "but", "for"}:
                continue
            
            # Skip duplicates
            if text.lower() in seen_entities:
                continue
                
            seen_entities.add(text.lower())
            
            # Clean entity text
            text = self._clean_entity_text(text)
            
            # Add to filtered list
            filtered_entities.append({
                "text": text,
                "type": entity["type"],
                "confidence": entity.get("confidence", 1.0)
            })
        
        article.entities = filtered_entities
    
    def _extract_entities_rule_based(self, article: NewsArticle) -> List[Dict[str, Any]]:
        """Extract entities using rule-based approach.
        
        Args:
            article: The article to process
            
        Returns:
            List of extracted entities
        """
        entities = []
        text = f"{article.title}. {article.content}"
        
        # Extract cryptocurrencies
        crypto_patterns = [
            r'\b(bitcoin|btc|ethereum|eth|ripple|xrp|cardano|ada|solana|sol|dogecoin|doge)\b',
            r'\$[A-Z]{2,5}\b'  # Ticker symbols like $BTC
        ]
        
        for pattern in crypto_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "type": "CRYPTOCURRENCY",
                    "confidence": 1.0
                })
        
        # Extract organizations
        org_patterns = [
            r'\b[A-Z][a-z]+ (Inc|LLC|Ltd|Corporation|Corp|Exchange)\b',
            r'\b(Binance|Coinbase|Kraken|FTX|Huobi|Gemini|Bitstamp)\b'
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "type": "ORG",
                    "confidence": 1.0
                })
        
        # Extract people (simplified)
        person_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        ]
        
        for pattern in person_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Skip likely false positives
                if match.group() in {"United States", "New York", "Hong Kong"}:
                    continue
                    
                entities.append({
                    "text": match.group(),
                    "type": "PER",
                    "confidence": 0.8
                })
        
        return entities
    
    def _clean_entity_text(self, text: str) -> str:
        """Clean entity text.
        
        Args:
            text: Entity text to clean
            
        Returns:
            Cleaned entity text
        """
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    async def _classify_topics(self, article: NewsArticle) -> None:
        """Classify the topics of an article.
        
        Args:
            article: The article to process
        """
        topics = []
        
        if self.topic_classifier:
            try:
                # Prepare text
                text = f"{article.title}. {article.content[:1000]}"
                
                # Define candidate topics
                candidate_topics = [
                    "regulation", "adoption", "technology", "investment",
                    "market analysis", "security", "innovation", "partnership",
                    "mining", "defi", "nft", "stablecoin", "central bank",
                    "institutional investment", "retail trading", "technical analysis"
                ]
                
                # Classify topics
                result = self.topic_classifier(
                    text, 
                    candidate_topics,
                    multi_label=True
                )
                
                # Filter topics with high confidence
                threshold = 0.3
                for topic, score in zip(result["labels"], result["scores"]):
                    if score >= threshold:
                        topics.append(topic)
                
            except Exception as e:
                self.logger.error(f"Error classifying topics: {e}")
                # Fall back to keyword-based approach
                topics = self._classify_topics_keyword_based(article)
        else:
            # Use keyword-based topic classification
            topics = self._classify_topics_keyword_based(article)
        
        article.topics = topics
    
    def _classify_topics_keyword_based(self, article: NewsArticle) -> List[str]:
        """Classify topics using keyword-based approach.
        
        Args:
            article: The article to process
            
        Returns:
            List of topics
        """
        text = f"{article.title.lower()} {article.content.lower()}"
        topics = []
        
        topic_keywords = {
            "regulation": ["regulation", "regulatory", "compliance", "legal", "law", "ban", "approve", "sec", "cftc"],
            "adoption": ["adoption", "accept", "mainstream", "institutional", "corporate", "payment"],
            "technology": ["technology", "protocol", "blockchain", "upgrade", "fork", "scalability"],
            "investment": ["investment", "fund", "portfolio", "asset", "allocation", "diversify"],
            "market analysis": ["analysis", "indicator", "trend", "pattern", "chart", "technical", "support", "resistance"],
            "security": ["security", "hack", "breach", "vulnerability", "protect", "encrypt"],
            "innovation": ["innovation", "develop", "launch", "release", "new", "feature"],
            "partnership": ["partnership", "collaborate", "alliance", "join", "together"],
            "mining": ["mining", "miner", "hash", "proof of work", "energy", "power"],
            "defi": ["defi", "decentralized finance", "yield", "lending", "borrowing", "liquidity"],
            "nft": ["nft", "non-fungible", "collectible", "art", "unique", "token"],
            "stablecoin": ["stablecoin", "pegged", "tether", "usdc", "dai", "stable"],
            "central bank": ["central bank", "cbdc", "federal reserve", "ecb", "monetary policy"],
            "institutional investment": ["institutional", "hedge fund", "grayscale", "etf", "trust"],
            "retail trading": ["retail", "trader", "exchange", "volume", "trade"]
        }
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    topics.append(topic)
                    break
        
        return topics
    
    async def _generate_summary(self, article: NewsArticle) -> None:
        """Generate a summary of an article.
        
        Args:
            article: The article to process
        """
        if self.summarizer:
            try:
                # Prepare text
                text = article.content
                
                # Limit text length to avoid token limits
                if len(text) > 1000:
                    text = text[:1000]
                
                # Generate summary
                summary = self.summarizer(text, max_length=100, min_length=30, do_sample=False)
                
                if summary and len(summary) > 0:
                    article.summary = summary[0]["summary_text"]
                
            except Exception as e:
                self.logger.error(f"Error generating summary: {e}")
                # Fall back to extraction-based summary
                article.summary = self._generate_extraction_summary(article)
        else:
            # Use extraction-based summary
            article.summary = self._generate_extraction_summary(article)
    
    def _generate_extraction_summary(self, article: NewsArticle) -> str:
        """Generate an extraction-based summary.
        
        Args:
            article: The article to process
            
        Returns:
            Extracted summary
        """
        # Very simple extraction-based summary
        if len(article.content) <= 150:
            return article.content
            
        # Use first 2-3 sentences as summary
        sentences = re.split(r'(?<=[.!?])\s+', article.content)
        
        if len(sentences) <= 3:
            return " ".join(sentences)
            
        return " ".join(sentences[:3])
    
    async def _calculate_relevance(self, article: NewsArticle) -> None:
        """Calculate asset relevance scores for an article.
        
        Args:
            article: The article to process
        """
        text = f"{article.title.lower()} {article.content.lower()}"
        
        relevance_scores = {}
        
        # Check relevance for each asset
        for asset in self.assets:
            score = 0.0
            
            # Basic keyword matching
            keywords = [asset.lower()]
            
            # Add asset-specific keywords
            if asset == "BTC":
                keywords.extend(["bitcoin", "btc", "satoshi", "nakamoto"])
            elif asset == "ETH":
                keywords.extend(["ethereum", "eth", "buterin", "vitalik"])
            elif asset == "SOL":
                keywords.extend(["solana", "sol"])
            elif asset == "XRP":
                keywords.extend(["ripple", "xrp"])
            
            # Count keyword mentions
            mention_count = sum(text.count(keyword) for keyword in keywords)
            
            # Check title (higher weight)
            title_mentions = sum(article.title.lower().count(keyword) for keyword in keywords)
            mention_count += title_mentions * 2
            
            # Calculate base score
            if mention_count > 0:
                score = min(0.3 + (mention_count * 0.1), 1.0)
            
            # Check entities
            entity_mentions = sum(1 for entity in article.entities if any(keyword in entity["text"].lower() for keyword in keywords))
            if entity_mentions > 0:
                score = min(score + 0.2, 1.0)
            
            # Apply recency bonus
            hours_old = (datetime.now() - article.published_at).total_seconds() / 3600
            recency_factor = max(0.8, 1.0 - (hours_old / 72))  # Reduce score for older articles
            score *= recency_factor
            
            # Store relevance score if above threshold
            if score >= self.relevance_threshold:
                relevance_scores[asset] = score
        
        article.relevance_scores = relevance_scores
    
    async def _assess_market_impact(self, article: NewsArticle) -> None:
        """Assess potential market impact of an article.
        
        Args:
            article: The article to process
        """
        # Initialize market impact assessment
        impact = {
            "direction": "neutral",
            "magnitude": 0.0,
            "confidence": 0.0,
            "timeframe": "short",
            "affected_assets": []
        }
        
        # Determine affected assets
        affected_assets = []
        for asset, score in article.relevance_scores.items():
            if score >= self.relevance_threshold:
                affected_assets.append(asset)
        
        if not affected_assets:
            # No relevant assets
            article.market_impact = impact
            return
            
        impact["affected_assets"] = affected_assets
        
        # Determine impact direction from sentiment
        if article.sentiment is not None:
            if article.sentiment >= 0.7:
                impact["direction"] = "positive"
                impact["magnitude"] = (article.sentiment - 0.5) * 2  # Scale to 0-1
            elif article.sentiment <= 0.3:
                impact["direction"] = "negative"
                impact["magnitude"] = (0.5 - article.sentiment) * 2  # Scale to 0-1
        
        # Adjust based on topics
        impactful_topics = {
            "regulation": 0.8,
            "adoption": 0.6,
            "security": 0.7,
            "partnership": 0.5,
            "institutional investment": 0.7
        }
        
        for topic in article.topics:
            if topic in impactful_topics:
                impact["magnitude"] = max(impact["magnitude"], impactful_topics[topic])
        
        # Adjust confidence based on source credibility
        # (simplified for now)
        credible_sources = {"bloomberg", "coindesk", "reuters", "cointelegraph"}
        if article.source.lower() in credible_sources:
            impact["confidence"] = 0.8
        else:
            impact["confidence"] = 0.6
        
        # Determine timeframe
        short_term_topics = {"market analysis", "technical analysis", "security"}
        long_term_topics = {"regulation", "adoption", "technology", "innovation"}
        
        if any(topic in short_term_topics for topic in article.topics):
            impact["timeframe"] = "short"
        elif any(topic in long_term_topics for topic in article.topics):
            impact["timeframe"] = "long"
        
        article.market_impact = impact
    
    def _update_indices(self, articles: List[NewsArticle]) -> None:
        """Update article and entity indices.
        
        Args:
            articles: List of articles to index
        """
        for article in articles:
            # Extract keywords from title and content
            text = f"{article.title.lower()} {article.content.lower()}"
            words = re.findall(r'\b\w+\b', text)
            
            # Filter common words
            stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "but"}
            keywords = [word for word in words if word not in stopwords and len(word) > 2]
            
            # Index by keywords
            for keyword in set(keywords):
                self.article_index[keyword].add(article.article_id)
            
            # Index by entities
            for entity in article.entities:
                entity_text = entity["text"].lower()
                self.entity_index[entity_text].add(article.article_id)
    
    def _build_news_graph(self, articles: List[NewsArticle]) -> None:
        """Build a graph representation of news articles and their relationships.
        
        Args:
            articles: List of articles to include in the graph
        """
        # Create a new graph
        G = nx.Graph()
        
        # Add article nodes
        for article in articles:
            G.add_node(
                article.article_id,
                type="article",
                title=article.title,
                source=article.source,
                published_at=article.published_at.isoformat(),
                sentiment=article.sentiment or 0.5
            )
            
            # Add entity nodes and edges
            for entity in article.entities:
                entity_id = f"entity:{entity['type']}:{entity['text'].lower()}"
                
                if not G.has_node(entity_id):
                    G.add_node(
                        entity_id,
                        type="entity",
                        entity_type=entity["type"],
                        text=entity["text"],
                        confidence=entity.get("confidence", 1.0)
                    )
                
                # Connect article to entity
                G.add_edge(
                    article.article_id,
                    entity_id,
                    weight=entity.get("confidence", 1.0)
                )
            
            # Add topic nodes and edges
            for topic in article.topics:
                topic_id = f"topic:{topic}"
                
                if not G.has_node(topic_id):
                    G.add_node(
                        topic_id,
                        type="topic",
                        text=topic
                    )
                
                # Connect article to topic
                G.add_edge(
                    article.article_id,
                    topic_id,
                    weight=1.0
                )
        
        # Find article-article relationships
        self._find_article_relationships(articles, G)
        
        # Store the graph
        self.news_graph = G
    
    def _find_article_relationships(self, articles: List[NewsArticle], G: nx.Graph) -> None:
        """Find relationships between articles.
        
        Args:
            articles: List of articles to analyze
            G: NetworkX graph to update
        """
        # Compute text similarity between articles
        if len(articles) < 2:
            return
            
        # Prepare texts
        texts = [f"{article.title} {article.content[:1000]}" for article in articles]
        article_ids = [article.article_id for article in articles]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words="english")
        
        try:
            # Compute TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Connect similar articles
            threshold = 0.3
            for i in range(len(articles)):
                for j in range(i+1, len(articles)):
                    similarity = similarity_matrix[i, j]
                    
                    if similarity >= threshold:
                        G.add_edge(
                            article_ids[i],
                            article_ids[j],
                            weight=similarity,
                            type="similar"
                        )
            
        except Exception as e:
            self.logger.error(f"Error computing article similarities: {e}")
    
    async def search_articles(self, 
                             query: str, 
                             assets: Optional[List[str]] = None,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             limit: int = 10) -> List[NewsArticle]:
        """Search for articles matching a query.
        
        Args:
            query: Search query
            assets: List of assets to filter by
            start_date: Start date for filtering
            end_date: End date for filtering
            limit: Maximum number of articles to return
            
        Returns:
            List of matching articles
        """
        if not self.is_initialized:
            self.logger.warning("News analyzer not initialized")
            return []
            
        # Extract query terms
        query_terms = re.findall(r'\b\w+\b', query.lower())
        
        # Find matching article IDs
        matching_ids = set()
        
        for term in query_terms:
            if term in self.article_index:
                if not matching_ids:
                    matching_ids = self.article_index[term].copy()
                else:
                    matching_ids &= self.article_index[term]
        
        # Filter by assets
        if assets:
            asset_matching_ids = set()
            for asset in assets:
                asset = asset.upper()
                for article_id, article in self.articles.items():
                    if asset in article.relevance_scores:
                        asset_matching_ids.add(article_id)
            
            matching_ids &= asset_matching_ids
        
        # Convert to articles
        matching_articles = [self.articles[article_id] for article_id in matching_ids if article_id in self.articles]
        
        # Filter by date
        if start_date:
            matching_articles = [a for a in matching_articles if a.published_at >= start_date]
        
        if end_date:
            matching_articles = [a for a in matching_articles if a.published_at <= end_date]
        
        # Sort by relevance (simple implementation)
        matching_articles.sort(key=lambda a: a.published_at, reverse=True)
        
        # Limit results
        return matching_articles[:limit]
    
    async def find_related_articles(self, 
                                   article_id: str, 
                                   max_results: int = 5) -> List[NewsArticle]:
        """Find articles related to a given article.
        
        Args:
            article_id: ID of the article to find related articles for
            max_results: Maximum number of related articles to return
            
        Returns:
            List of related articles
        """
        if not self.is_initialized or article_id not in self.articles:
            return []
            
        # Use graph to find related articles
        if article_id in self.news_graph:
            # Get neighbors
            neighbors = list(self.news_graph.neighbors(article_id))
            
            # Filter to keep only article nodes
            article_neighbors = [n for n in neighbors if isinstance(n, str) and n.startswith("article:")]
            
            # Get edge weights
            weighted_neighbors = [
                (n, self.news_graph.get_edge_data(article_id, n).get("weight", 0))
                for n in article_neighbors
            ]
            
            # Sort by weight descending
            weighted_neighbors.sort(key=lambda x: x[1], reverse=True)
            
            # Get top related articles
            related_ids = [n[0] for n in weighted_neighbors[:max_results]]
            
            # Convert to articles
            return [self.articles[related_id] for related_id in related_ids if related_id in self.articles]
        
        # Fallback: find articles with common entities
        article = self.articles[article_id]
        
        # Get entities
        entity_texts = [entity["text"].lower() for entity in article.entities]
        
        # Find articles with common entities
        related_articles = []
        
        for entity_text in entity_texts:
            if entity_text in self.entity_index:
                for related_id in self.entity_index[entity_text]:
                    if related_id != article_id and related_id in self.articles:
                        related_articles.append(self.articles[related_id])
        
        # Remove duplicates and sort by publication date
        unique_related = {}
        for related in related_articles:
            if related.article_id not in unique_related:
                unique_related[related.article_id] = related
        
        related_list = list(unique_related.values())
        related_list.sort(key=lambda a: a.published_at, reverse=True)
        
        return related_list[:max_results]
    
    async def get_trending_topics(self, 
                                 timeframe: str = "24h",
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending topics from recent articles.
        
        Args:
            timeframe: Time period to analyze
            limit: Maximum number of topics to return
            
        Returns:
            List of trending topics with counts and sentiment
        """
        if not self.is_initialized:
            return []
            
        # Parse timeframe
        hours = 24
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
        elif timeframe.endswith("d"):
            hours = int(timeframe[:-1]) * 24
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Find recent articles
        recent_articles = [a for a in self.articles.values() if a.published_at >= cutoff_time]
        
        if not recent_articles:
            return []
            
        # Count topics
        topic_counter = Counter()
        topic_sentiments = defaultdict(list)
        
        for article in recent_articles:
            for topic in article.topics:
                topic_counter[topic] += 1
                if article.sentiment is not None:
                    topic_sentiments[topic].append(article.sentiment)
        
        # Calculate average sentiment for each topic
        topic_avg_sentiment = {}
        for topic, sentiments in topic_sentiments.items():
            if sentiments:
                topic_avg_sentiment[topic] = sum(sentiments) / len(sentiments)
            else:
                topic_avg_sentiment[topic] = 0.5
        
        # Get top topics
        top_topics = topic_counter.most_common(limit)
        
        # Format results
        results = []
        for topic, count in top_topics:
            results.append({
                "topic": topic,
                "count": count,
                "sentiment": topic_avg_sentiment.get(topic, 0.5)
            })
        
        return results
    
    async def get_entity_sentiment(self, 
                                  entity_text: str,
                                  entity_type: Optional[str] = None,
                                  timeframe: str = "7d") -> Dict[str, Any]:
        """Get sentiment analysis for a specific entity.
        
        Args:
            entity_text: The entity text to analyze
            entity_type: Optional entity type filter
            timeframe: Time period to analyze
            
        Returns:
            Dictionary with entity sentiment analysis
        """
        if not self.is_initialized:
            return {
                "entity": entity_text,
                "type": entity_type,
                "sentiment": 0.5,
                "article_count": 0,
                "timeframe": timeframe
            }
            
        # Parse timeframe
        hours = 24 * 7  # Default to 7 days
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
        elif timeframe.endswith("d"):
            hours = int(timeframe[:-1]) * 24
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Find articles mentioning the entity
        entity_text_lower = entity_text.lower()
        sentiments = []
        article_count = 0
        articles_over_time = defaultdict(int)
        
        for article in self.articles.values():
            if article.published_at < cutoff_time:
                continue
                
            # Check if article mentions the entity
            mentions_entity = False
            
            for entity in article.entities:
                if entity_text_lower in entity["text"].lower():
                    if entity_type is None or entity["type"] == entity_type:
                        mentions_entity = True
                        break
            
            if mentions_entity:
                article_count += 1
                
                # Record sentiment
                if article.sentiment is not None:
                    sentiments.append(article.sentiment)
                
                # Record publication date for trend analysis
                date_key = article.published_at.strftime("%Y-%m-%d")
                articles_over_time[date_key] += 1
        
        # Calculate average sentiment
        avg_sentiment = 0.5
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Prepare trend data
        dates = sorted(articles_over_time.keys())
        trend_data = [{"date": date, "count": articles_over_time[date]} for date in dates]
        
        return {
            "entity": entity_text,
            "type": entity_type,
            "sentiment": avg_sentiment,
            "article_count": article_count,
            "timeframe": timeframe,
            "trend": trend_data
        }
    
    async def generate_market_brief(self, assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a market brief for specified assets.
        
        Args:
            assets: List of assets to include in the brief
            
        Returns:
            Dictionary with market brief data
        """
        if not self.is_initialized:
            return {"assets": assets or [], "timestamp": datetime.now().isoformat(), "data": []}
            
        assets = assets or self.assets
        
        # Find recent articles for each asset
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        asset_data = []
        
        for asset in assets:
            asset_articles = []
            
            for article in self.articles.values():
                if article.published_at < cutoff_time:
                    continue
                    
                if asset in article.relevance_scores and article.relevance_scores[asset] >= self.relevance_threshold:
                    asset_articles.append(article)
            
            # Sort by relevance and recency
            asset_articles.sort(key=lambda a: (a.relevance_scores[asset], a.published_at), reverse=True)
            
            # Calculate overall sentiment
            sentiments = [a.sentiment for a in asset_articles if a.sentiment is not None]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5
            
            # Get top topics
            topics = Counter()
            for article in asset_articles:
                for topic in article.topics:
                    topics[topic] += 1
            
            top_topics = [topic for topic, _ in topics.most_common(3)]
            
            # Get impact assessment
            impacts = [article.market_impact for article in asset_articles if article.market_impact]
            positive_impacts = [i for i in impacts if i["direction"] == "positive"]
            negative_impacts = [i for i in impacts if i["direction"] == "negative"]
            
            market_impact = "neutral"
            if len(positive_impacts) > len(negative_impacts) * 1.5:
                market_impact = "positive"
            elif len(negative_impacts) > len(positive_impacts) * 1.5:
                market_impact = "negative"
            
            # Get top articles
            top_articles = asset_articles[:5]
            top_articles_data = []
            
            for article in top_articles:
                top_articles_data.append({
                    "title": article.title,
                    "source": article.source,
                    "url": article.url,
                    "published_at": article.published_at.isoformat(),
                    "sentiment": article.sentiment,
                    "summary": article.summary
                })
            
            asset_data.append({
                "asset": asset,
                "article_count": len(asset_articles),
                "sentiment": avg_sentiment,
                "top_topics": top_topics,
                "market_impact": market_impact,
                "top_articles": top_articles_data
            })
        
        return {
            "assets": assets,
            "timestamp": datetime.now().isoformat(),
            "data": asset_data
        }
    
    def get_news_graph(self) -> nx.Graph:
        """Get the news graph.
        
        Returns:
            NetworkX graph of news articles and entities
        """
        return self.news_graph
        
    async def get_trending_crypto_categories(self, 
                                         timeframe: str = "24h", 
                                         limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending cryptocurrency news categories.
        
        Args:
            timeframe: Time period to analyze (e.g. "24h", "7d")
            limit: Maximum number of categories to return
            
        Returns:
            List of trending category dictionaries
        """
        if not self.is_initialized or not self.use_crypto_categorization or not self.topic_graph:
            return []
            
        # Parse timeframe to hours
        hours = 24
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
        elif timeframe.endswith("d"):
            hours = int(timeframe[:-1]) * 24
            
        # Get trending categories from topic graph
        trending_categories = self.topic_graph.get_trending_categories(
            timeframe_hours=hours,
            limit=limit
        )
        
        return trending_categories
        
    async def get_crypto_narratives(self, 
                                timeframe: str = "72h") -> List[Dict[str, Any]]:
        """Get cryptocurrency narrative clusters from recent news.
        
        Args:
            timeframe: Time period to analyze (e.g. "72h", "7d")
            
        Returns:
            List of narrative cluster dictionaries
        """
        if not self.is_initialized or not self.use_crypto_categorization or not self.topic_graph:
            return []
            
        # Parse timeframe to hours
        hours = 72
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
        elif timeframe.endswith("d"):
            hours = int(timeframe[:-1]) * 24
            
        # Get narrative clusters from topic graph
        narratives = self.topic_graph.find_narrative_clusters(
            timeframe_hours=hours
        )
        
        return narratives
        
    async def get_asset_category_associations(self, 
                                         asset: str) -> Dict[str, float]:
        """Get categories most associated with a specific asset.
        
        Args:
            asset: The asset symbol to analyze
            
        Returns:
            Dictionary mapping categories to association scores (0-1)
        """
        if not self.is_initialized or not self.use_crypto_categorization or not self.topic_graph:
            return {}
            
        # Get asset-category associations from topic graph
        associations = self.topic_graph.get_asset_category_associations(asset)
        
        return associations
    
    async def extract_events(self, 
                            timeframe: str = "7d", 
                            min_importance: float = 0.5) -> List[Dict[str, Any]]:
        """Extract significant events from news articles.
        
        Args:
            timeframe: Time period to analyze
            min_importance: Minimum importance threshold for events
            
        Returns:
            List of extracted events
        """
        if not self.is_initialized:
            return []
            
        # Parse timeframe
        hours = 24 * 7  # Default to 7 days
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
        elif timeframe.endswith("d"):
            hours = int(timeframe[:-1]) * 24
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Find recent articles
        recent_articles = [a for a in self.articles.values() if a.published_at >= cutoff_time]
        
        if not recent_articles:
            return []
        
        # Group articles by topic and entity
        topic_articles = defaultdict(list)
        entity_articles = defaultdict(list)
        
        for article in recent_articles:
            # Group by topic
            for topic in article.topics:
                topic_articles[topic].append(article)
            
            # Group by entity
            for entity in article.entities:
                entity_key = f"{entity['type']}:{entity['text'].lower()}"
                entity_articles[entity_key].append(article)
        
        # Extract events
        events = []
        
        # Topic-based events
        for topic, articles in topic_articles.items():
            if len(articles) < 3:
                continue
                
            # Calculate importance based on article count and recency
            importance = min(0.3 + (len(articles) * 0.05), 1.0)
            
            # Skip if below importance threshold
            if importance < min_importance:
                continue
                
            # Find most relevant article
            articles.sort(key=lambda a: a.published_at, reverse=True)
            main_article = articles[0]
            
            # Create event
            event = {
                "id": f"topic_event:{topic}:{main_article.article_id}",
                "type": "topic",
                "topic": topic,
                "title": f"{topic.capitalize()} in Cryptocurrency",
                "description": main_article.summary or main_article.title,
                "importance": importance,
                "timestamp": main_article.published_at.isoformat(),
                "source": main_article.source,
                "url": main_article.url,
                "related_articles": [
                    {
                        "id": a.article_id,
                        "title": a.title,
                        "source": a.source,
                        "published_at": a.published_at.isoformat()
                    }
                    for a in articles[:5]
                ],
                "entities": [],
                "sentiment": sum(a.sentiment or 0.5 for a in articles) / len(articles),
                "affected_assets": self._find_affected_assets(articles)
            }
            
            # Add event
            events.append(event)
        
        # Entity-based events
        for entity_key, articles in entity_articles.items():
            if len(articles) < 2:
                continue
                
            # Parse entity key
            entity_parts = entity_key.split(":", 1)
            entity_type = entity_parts[0]
            entity_text = entity_parts[1]
            
            # Skip common entities
            if entity_type == "ORG" and entity_text in {"twitter", "reuters", "ap", "bloomberg"}:
                continue
                
            # Calculate importance based on article count and recency
            importance = min(0.3 + (len(articles) * 0.05), 1.0)
            
            # Skip if below importance threshold
            if importance < min_importance:
                continue
                
            # Find most relevant article
            articles.sort(key=lambda a: a.published_at, reverse=True)
            main_article = articles[0]
            
            # Create event
            event = {
                "id": f"entity_event:{entity_key}:{main_article.article_id}",
                "type": "entity",
                "entity_type": entity_type,
                "entity_text": entity_text,
                "title": f"{entity_text.capitalize()} in the News",
                "description": main_article.summary or main_article.title,
                "importance": importance,
                "timestamp": main_article.published_at.isoformat(),
                "source": main_article.source,
                "url": main_article.url,
                "related_articles": [
                    {
                        "id": a.article_id,
                        "title": a.title,
                        "source": a.source,
                        "published_at": a.published_at.isoformat()
                    }
                    for a in articles[:5]
                ],
                "topics": list(set(topic for a in articles for topic in a.topics)),
                "sentiment": sum(a.sentiment or 0.5 for a in articles) / len(articles),
                "affected_assets": self._find_affected_assets(articles)
            }
            
            # Add event
            events.append(event)
        
        # Filter duplicate events
        unique_events = {}
        for event in events:
            # Create a key based on affected assets and description
            key_parts = [event["description"][:100]]
            key_parts.extend(sorted(event["affected_assets"]))
            key = ":".join(key_parts)
            
            # Keep the event with higher importance
            if key not in unique_events or event["importance"] > unique_events[key]["importance"]:
                unique_events[key] = event
        
        # Sort events by importance
        result = list(unique_events.values())
        result.sort(key=lambda e: e["importance"], reverse=True)
        
        return result
    
    def _find_affected_assets(self, articles: List[NewsArticle]) -> List[str]:
        """Find assets affected by a set of articles.
        
        Args:
            articles: List of articles to analyze
            
        Returns:
            List of affected asset symbols
        """
        assets_scores = defaultdict(float)
        
        for article in articles:
            for asset, score in article.relevance_scores.items():
                assets_scores[asset] += score
        
        # Normalize by article count
        for asset in assets_scores:
            assets_scores[asset] /= len(articles)
        
        # Filter by threshold
        affected_assets = [asset for asset, score in assets_scores.items() 
                          if score >= self.relevance_threshold]
        
        return affected_assets


class MockNewsClient:
    """Mock news API client for testing."""
    
    async def get_articles(self, 
                          timeframe: str = "24h", 
                          assets: Optional[List[str]] = None) -> List[NewsArticle]:
        """Get mock news articles.
        
        Args:
            timeframe: Time period to fetch articles for
            assets: List of assets to fetch articles about
            
        Returns:
            List of mock NewsArticle objects
        """
        # Parse timeframe
        hours = 24
        if timeframe.endswith("h"):
            hours = int(timeframe[:-1])
        elif timeframe.endswith("d"):
            hours = int(timeframe[:-1]) * 24
        
        # Generate mock articles
        articles = []
        assets = assets or ["BTC", "ETH", "SOL", "XRP"]
        
        # Get current time
        now = datetime.now()
        
        # Mock news sources
        sources = ["CoinDesk", "CryptoBriefing", "Decrypt", "CoinTelegraph", "The Block"]
        
        # Mock article templates for each asset
        templates = {
            "BTC": [
                {
                    "title": "Bitcoin Surges Past $X as Market Sentiment Improves",
                    "content": "Bitcoin has climbed above $X in a significant move that analysts attribute to improving market sentiment and growing institutional interest. Technical indicators suggest the leading cryptocurrency could be entering a new bullish phase after weeks of consolidation. Trading volume has also increased substantially, indicating stronger conviction among market participants. Some experts point to regulatory developments and macroeconomic factors as potential catalysts for this move.",
                    "sentiment": 0.8
                },
                {
                    "title": "Bitcoin Faces Resistance at $X After Recent Rally",
                    "content": "After a strong upward movement, Bitcoin is now encountering resistance around the $X level. Market analysts suggest this could lead to a period of consolidation as traders take profits. Despite the short-term resistance, the overall trend remains positive with on-chain metrics showing continued accumulation by long-term holders. The cryptocurrency's volatility has decreased in recent days, which some see as a sign of market maturity.",
                    "sentiment": 0.6
                },
                {
                    "title": "Bitcoin Mining Difficulty Reaches All-Time High",
                    "content": "Bitcoin's mining difficulty has adjusted upward to a new all-time high, reflecting the increased computing power dedicated to securing the network. This adjustment comes as more miners bring operations online following recent price improvements. The higher difficulty level ensures that block times remain consistent despite fluctuations in hash rate. Industry experts note that this is a positive sign for Bitcoin's long-term security and decentralization.",
                    "sentiment": 0.7
                },
                {
                    "title": "Bitcoin Drops X% as Market Reacts to Regulatory News",
                    "content": "Bitcoin prices have fallen approximately X% in the last 24 hours as markets react to new regulatory developments. The drop comes amid concerns about potential restrictions on cryptocurrency trading and custody services in major markets. Trading volume has increased significantly during the sell-off, suggesting many investors are repositioning their portfolios. Some analysts view this as a temporary correction within a larger bullish trend, while others caution that regulatory uncertainty could persist.",
                    "sentiment": 0.3
                },
                {
                    "title": "Major Investment Firm Adds Bitcoin to Corporate Treasury",
                    "content": "A prominent investment firm has announced the addition of Bitcoin to its corporate treasury, allocating approximately X% of its cash reserves to the cryptocurrency. This move follows similar treasury investments by forward-thinking companies seeking inflation hedges and alternative stores of value. The firm's CEO cited concerns about currency debasement and the need for monetary alternatives in the current macroeconomic environment. This institutional adoption continues a trend that has been growing since 2020.",
                    "sentiment": 0.9
                }
            ],
            "ETH": [
                {
                    "title": "Ethereum Upgrade Promises Lower Fees and Faster Transactions",
                    "content": "An upcoming Ethereum network upgrade aims to address persistent concerns about high gas fees and network congestion. Developers have finalized the implementation details, with the update scheduled for deployment next month. The upgrade includes several EIPs (Ethereum Improvement Proposals) focused on optimization and scalability. Community response has been largely positive, with many users hopeful that this will make the network more accessible for smaller transactions.",
                    "sentiment": 0.8
                },
                {
                    "title": "Ethereum DeFi Ecosystem Reaches $X Billion in Total Value Locked",
                    "content": "Ethereum's decentralized finance ecosystem has reached a new milestone with $X billion now locked in various protocols. This growth represents a X% increase since the beginning of the year, highlighting continued interest despite market volatility. Lending protocols and decentralized exchanges account for the majority of this value. Analysts note that the increasing TVL demonstrates confidence in Ethereum's role as the primary blockchain for financial applications.",
                    "sentiment": 0.8
                },
                {
                    "title": "Ethereum Faces Scaling Challenges as Network Usage Peaks",
                    "content": "Ethereum users are experiencing record-high gas fees as network utilization reaches maximum capacity. The congestion comes amid increased activity in NFT marketplaces and yield farming protocols. While Layer 2 solutions offer some relief, many users are finding transactions prohibitively expensive on the main chain. This situation highlights the ongoing scaling challenges facing Ethereum as it transitions to a proof-of-stake consensus mechanism.",
                    "sentiment": 0.4
                },
                {
                    "title": "Ethereum Foundation Announces Research Grants for Scalability Solutions",
                    "content": "The Ethereum Foundation has announced a new round of grants focused on scalability research and implementation. The funding aims to accelerate development of solutions that will help the network handle increasing demand. Several research teams and startups will receive support to work on promising approaches. This initiative reflects the foundation's commitment to addressing Ethereum's most pressing technical challenges.",
                    "sentiment": 0.7
                },
                {
                    "title": "Ethereum NFT Market Shows Signs of Recovery After Slump",
                    "content": "Trading volume in Ethereum's NFT marketplaces has shown signs of recovery following months of declining activity. Several high-profile collections have seen renewed interest from collectors and investors. The average sale price has also increased, suggesting stronger demand at current market levels. While still below the peaks of the NFT boom, this recovery indicates that the technology continues to find use cases and supporters.",
                    "sentiment": 0.6
                }
            ],
            "SOL": [
                {
                    "title": "Solana Outage Raises Questions About Network Reliability",
                    "content": "Solana experienced another network outage lasting approximately X hours, raising fresh concerns about the blockchain's reliability. Validators worked to restart the network following the disruption, which was attributed to a bug in the consensus mechanism. This marks the Xth significant outage for Solana in the past year. Despite these challenges, developers continue to build on the platform, citing its high throughput and low transaction costs when operational.",
                    "sentiment": 0.2
                },
                {
                    "title": "Solana DeFi Ecosystem Expands with Launch of New Lending Protocol",
                    "content": "Solana's DeFi ecosystem continues to grow with the launch of a new lending protocol that has already attracted over $X million in liquidity. The protocol offers innovative features designed to leverage Solana's high transaction throughput. Early users report significantly lower costs compared to similar Ethereum-based services. This launch represents another milestone in Solana's quest to establish itself as a viable alternative for decentralized finance applications.",
                    "sentiment": 0.8
                },
                {
                    "title": "Major Exchange Announces Solana Staking Services",
                    "content": "A leading cryptocurrency exchange has announced the addition of Solana staking services for its users. The service will offer competitive yields with flexible redemption options. This move is expected to increase participation in Solana's proof-of-stake consensus mechanism. The exchange cited growing customer demand and Solana's technical advantages as reasons for adding the service.",
                    "sentiment": 0.7
                },
                {
                    "title": "Solana Price Surges Following Protocol Upgrade",
                    "content": "Solana's price has increased by approximately X% following a successful protocol upgrade that promises improved network stability and performance. Trading volume has also risen substantially as investors respond positively to the technical developments. The upgrade addresses several issues that had previously contributed to network congestion and outages. Developer activity on the platform continues to increase, with several new projects announcing launches in the coming months.",
                    "sentiment": 0.8
                },
                {
                    "title": "Solana NFT Marketplace Sees Record Trading Volume",
                    "content": "A leading Solana-based NFT marketplace has reported record trading volume, with over $X million in transactions in the past week. The platform has attracted users with its low fees and fast confirmation times. Several exclusive NFT collections launched on Solana have seen strong demand from collectors. This success highlights Solana's growing position in the NFT ecosystem, which has traditionally been dominated by Ethereum.",
                    "sentiment": 0.7
                }
            ],
            "XRP": [
                {
                    "title": "XRP Lawsuit: SEC and Ripple Present Final Arguments",
                    "content": "The long-running legal battle between the SEC and Ripple has reached a crucial phase as both parties presented their final arguments before the court. Legal experts are divided on the potential outcome, which could have far-reaching implications for the cryptocurrency industry. Ripple executives remain confident that the court will rule in their favor, citing recent developments and judicial comments. A decision is expected within the next few months, potentially ending years of uncertainty for XRP holders.",
                    "sentiment": 0.5
                },
                {
                    "title": "XRP Payment Corridor Launches in New Market",
                    "content": "Ripple has announced the launch of a new XRP-based payment corridor connecting major financial institutions across international borders. The service promises to reduce settlement times and costs for cross-border transactions. Early participants report significant efficiency improvements compared to traditional banking rails. This expansion represents another step in Ripple's strategy to build a global network for international money transfers.",
                    "sentiment": 0.8
                },
                {
                    "title": "XRP Ledger Adds New Functionality with Protocol Update",
                    "content": "The XRP Ledger has been upgraded with new features following a protocol update that received broad support from validators. The update introduces enhanced smart contract capabilities and improved DEX functionality. Developers can now build more complex applications on the XRP Ledger, expanding its use cases beyond payments. The upgrade was implemented smoothly with no reported issues during the transition.",
                    "sentiment": 0.7
                },
                {
                    "title": "Major Bank Trials XRP for International Settlements",
                    "content": "A multinational banking corporation has begun testing XRP for international settlement operations between its subsidiaries. The pilot program aims to evaluate potential cost savings and efficiency improvements. Initial results suggest significant reductions in settlement time compared to current methods. This trial represents a potential use case for XRP independent of the ongoing legal proceedings in the United States.",
                    "sentiment": 0.8
                },
                {
                    "title": "XRP Market Liquidity Concerns Surface Amid Exchange Delistings",
                    "content": "Market participants have raised concerns about XRP liquidity following additional exchange delistings in certain jurisdictions. Trading volumes have declined on regulated platforms, though global liquidity remains relatively stable. Some analysts suggest this could create arbitrage opportunities across different markets. The situation highlights the ongoing regulatory challenges facing XRP despite its technical capabilities for payments.",
                    "sentiment": 0.3
                }
            ]
        }
        
        # Generate articles for each asset
        article_id_counter = 1
        
        for asset in assets:
            if asset not in templates:
                continue
                
            asset_templates = templates[asset]
            
            # Generate 2-5 articles per asset
            num_articles = min(len(asset_templates), np.random.randint(2, 6))
            
            for i in range(num_articles):
                template = asset_templates[i]
                
                # Randomize publication time within timeframe
                hours_ago = np.random.randint(1, hours)
                published_at = now - timedelta(hours=hours_ago)
                
                # Randomize source
                source = np.random.choice(sources)
                
                # Create article ID
                article_id = f"mock_{article_id_counter}"
                article_id_counter += 1
                
                # Replace placeholders in title and content
                title = template["title"].replace("$X", str(np.random.randint(30000, 70000)))
                content = template["content"].replace("$X", str(np.random.randint(30000, 70000)))
                
                # Create mock article
                article = NewsArticle(
                    article_id=article_id,
                    title=title,
                    content=content,
                    url=f"https://example.com/news/{article_id}",
                    source=source,
                    published_at=published_at,
                    author=f"Mock Author {np.random.randint(1, 10)}",
                    categories=["Cryptocurrency", "Markets", "Technology"],
                    tags=[asset, "Crypto", "Blockchain"]
                )
                
                # Set sentiment
                article.sentiment = template["sentiment"] + np.random.uniform(-0.1, 0.1)
                
                # Add to articles list
                articles.append(article)
        
        return articles


# Example usage
async def main():
    """Example usage of the NewsAnalyzer."""
    # Initialize news analyzer
    analyzer = NewsAnalyzer()
    await analyzer.initialize()
    
    # Collect articles
    articles = await analyzer.collect_articles(timeframe="24h", assets=["BTC", "ETH"])
    print(f"Collected {len(articles)} articles")
    
    # Analyze articles
    await analyzer.analyze_articles(articles)
    
    # Get trending topics
    trending_topics = await analyzer.get_trending_topics()
    print("\nTrending Topics:")
    for topic in trending_topics:
        print(f"- {topic['topic']}: {topic['count']} articles, sentiment: {topic['sentiment']:.2f}")
    
    # Generate market brief
    market_brief = await analyzer.generate_market_brief(["BTC", "ETH"])
    print("\nMarket Brief:")
    for asset_data in market_brief["data"]:
        print(f"- {asset_data['asset']}: {asset_data['article_count']} articles, sentiment: {asset_data['sentiment']:.2f}")
        print(f"  Top topics: {', '.join(asset_data['top_topics'])}")
        print(f"  Market impact: {asset_data['market_impact']}")
        print(f"  Top article: {asset_data['top_articles'][0]['title']}")
    
    # Extract events
    events = await analyzer.extract_events()
    print("\nExtracted Events:")
    for event in events:
        print(f"- {event['title']}")
        print(f"  Importance: {event['importance']:.2f}, Sentiment: {event['sentiment']:.2f}")
        print(f"  Affected assets: {', '.join(event['affected_assets'])}")
        print(f"  Description: {event['description'][:100]}...")
    
    # Get cryptocurrency-specific categories and narratives
    if analyzer.use_crypto_categorization:
        print("\nTrending Crypto Categories:")
        crypto_categories = await analyzer.get_trending_crypto_categories(timeframe="48h")
        for category in crypto_categories:
            print(f"- {category['category']}: {category['score']:.2f} ({category['article_count']} articles)")
        
        print("\nCrypto Narratives:")
        narratives = await analyzer.get_crypto_narratives()
        for narrative in narratives:
            print(f"- {narrative['title']} - {narrative['size']} articles")
            if narrative['categories']:
                categories_str = ", ".join([cat["category"] for cat in narrative['categories']])
                print(f"  Top categories: {categories_str}")
        
        print("\nBitcoin Category Associations:")
        btc_categories = await analyzer.get_asset_category_associations("BTC")
        for category, score in sorted(btc_categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"- {category}: {score:.2f}")


if __name__ == "__main__":
    asyncio.run(main())