"""Cryptocurrency news categorization system.

This module provides functionality for categorizing cryptocurrency news
articles into specialized categories relevant to cryptocurrency markets.
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import asyncio
from collections import defaultdict

from transformers import pipeline
import networkx as nx

from src.common.logging import get_logger
from src.common.config import config
from src.analysis_agents.news.news_analyzer import NewsArticle


class CryptoNewsCategory:
    """Enumeration of cryptocurrency news categories."""
    
    # Market and Trading
    PRICE_MOVEMENT = "price_movement"               # News about significant price changes
    MARKET_ANALYSIS = "market_analysis"             # Technical and fundamental analysis
    TRADING_STRATEGY = "trading_strategy"           # Trading approaches and methods
    MARKET_SENTIMENT = "market_sentiment"           # Market mood and investor sentiment
    
    # Regulatory and Legal
    REGULATION = "regulation"                       # Regulatory news and developments
    LEGAL = "legal"                                 # Legal cases and proceedings
    COMPLIANCE = "compliance"                       # Compliance and KYC/AML issues
    TAX = "tax"                                     # Tax implications and regulations
    
    # Technology and Development
    PROTOCOL_UPDATE = "protocol_update"             # Network upgrades and protocol changes
    DEVELOPMENT = "development"                     # Development progress and roadmaps
    SECURITY = "security"                           # Security issues, bugs, and fixes
    SCALING = "scaling"                             # Scaling solutions and performance
    
    # Industry and Business
    PARTNERSHIP = "partnership"                     # Business partnerships and collaborations
    ADOPTION = "adoption"                           # Adoption by users, businesses, countries
    INVESTMENT = "investment"                       # Venture capital and fundraising
    EXCHANGE = "exchange"                           # Exchange listings, delistings, issues
    
    # Ecosystem
    DEFI = "defi"                                   # Decentralized finance news
    NFT = "nft"                                     # Non-fungible tokens
    GOVERNANCE = "governance"                       # DAO and governance proposals
    STAKING = "staking"                             # Staking, yield farming, rewards
    
    # Macro and Fundamental
    MACROECONOMIC = "macroeconomic"                 # Broader economic factors
    INSTITUTIONAL = "institutional"                 # Institutional involvement
    MINING = "mining"                               # Mining operations and hash rate
    SUPPLY_METRICS = "supply_metrics"               # Token supply, burns, emissions
    
    # Categorizations by impact
    HIGH_IMPACT = "high_impact"                     # News with significant market impact
    TRENDING = "trending"                           # Currently trending topics
    SPECULATIVE = "speculative"                     # Rumors and speculative news
    CONTRARIAN = "contrarian"                       # News that goes against market consensus


class CryptoNewsCategorizer:
    """System for categorizing cryptocurrency news articles.
    
    This class provides functionality for categorizing news articles into
    specialized categories relevant to cryptocurrency markets, which can
    help in generating more targeted trading signals.
    """
    
    def __init__(self):
        """Initialize the crypto news categorizer."""
        self.logger = get_logger("news_analyzer", "categorizer")
        
        # Configuration
        self.enable_ml_categorization = config.get("news_analyzer.use_ml_categorization", True)
        self.min_category_confidence = config.get("news_analyzer.min_category_confidence", 0.6)
        
        # Initialize ML components
        self.zero_shot_classifier = None
        
        # Category definitions with keywords
        self.category_keywords = self._initialize_category_keywords()
        
        # Initialize category patterns
        self.category_patterns = self._compile_category_patterns()
        
        # Asset-specific terms
        self.asset_terms = self._initialize_asset_terms()
    
    async def initialize(self) -> None:
        """Initialize the categorizer components."""
        self.logger.info("Initializing crypto news categorizer")
        
        if self.enable_ml_categorization:
            try:
                # Initialize ML model in a non-blocking way
                loop = asyncio.get_event_loop()
                self.zero_shot_classifier = await loop.run_in_executor(
                    None,
                    lambda: pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                )
                self.logger.info("Successfully loaded zero-shot classification model")
            except Exception as e:
                self.logger.error(f"Failed to load zero-shot classification model: {e}")
                self.enable_ml_categorization = False
    
    def _initialize_category_keywords(self) -> Dict[str, List[str]]:
        """Initialize category keywords dictionary.
        
        Returns:
            Dictionary mapping categories to lists of related keywords
        """
        return {
            # Market and Trading
            CryptoNewsCategory.PRICE_MOVEMENT: [
                "price", "surge", "rally", "dump", "crash", "plummet", "soar", "skyrocket", 
                "correction", "all-time high", "ath", "record high", "bottom", "resistance", 
                "support", "breakout", "breakdown", "price action", "price prediction"
            ],
            
            CryptoNewsCategory.MARKET_ANALYSIS: [
                "analysis", "analyst", "forecast", "prediction", "outlook", "chart", "technical", 
                "fundamental", "indicator", "trend", "pattern", "fibonacci", "support", "resistance",
                "moving average", "ma", "rsi", "macd", "volume profile", "order book", "depth chart"
            ],
            
            CryptoNewsCategory.TRADING_STRATEGY: [
                "strategy", "trading", "trader", "position", "long", "short", "leverage", "margin", 
                "futures", "options", "derivatives", "stop loss", "take profit", "entry", "exit",
                "risk management", "trade setup", "swing trading", "day trading", "scalping"
            ],
            
            CryptoNewsCategory.MARKET_SENTIMENT: [
                "sentiment", "fear", "greed", "bullish", "bearish", "optimism", "pessimism", 
                "fomo", "fud", "panic", "euphoria", "capitulation", "confidence", "fear & greed",
                "investor sentiment", "market mood", "sentiment analysis", "sentiment index"
            ],
            
            # Regulatory and Legal
            CryptoNewsCategory.REGULATION: [
                "regulation", "regulatory", "regulator", "sec", "cftc", "finma", "fca", "pboc", 
                "framework", "guideline", "compliance", "supervisory", "oversight", "policy",
                "central bank digital currency", "cbdc", "stablecoin regulation"
            ],
            
            CryptoNewsCategory.LEGAL: [
                "lawsuit", "legal", "court", "judge", "ruling", "verdict", "settlement", "appeal", 
                "litigation", "class action", "securities", "fraud", "investigation", "prosecutor",
                "defendant", "plaintiff", "subpoena", "testimony", "evidence"
            ],
            
            CryptoNewsCategory.COMPLIANCE: [
                "compliance", "kyc", "aml", "know your customer", "anti-money laundering", "travel rule", 
                "fatf", "identification", "verification", "screening", "monitoring", "reporting",
                "suspicious activity", "regulatory reporting", "due diligence"
            ],
            
            CryptoNewsCategory.TAX: [
                "tax", "taxation", "irs", "capital gains", "tax reporting", "tax compliance", 
                "tax liability", "tax evasion", "tax reporting", "tax return", "tax guidance",
                "tax basis", "tax exemption", "tax deduction", "tax regulation"
            ],
            
            # Technology and Development
            CryptoNewsCategory.PROTOCOL_UPDATE: [
                "update", "upgrade", "fork", "hard fork", "soft fork", "proposal", "improvement", 
                "eip", "bip", "sip", "roadmap", "release", "version", "protocol",
                "consensus", "implementation", "mainnet", "testnet", "activation"
            ],
            
            CryptoNewsCategory.DEVELOPMENT: [
                "developer", "development", "codebase", "github", "commit", "pull request", "merge", 
                "code", "programming", "engineer", "technical", "library", "sdk", "api",
                "hackathon", "grant", "bounty", "open source", "implementation"
            ],
            
            CryptoNewsCategory.SECURITY: [
                "security", "hack", "exploit", "vulnerability", "bug", "attack", "breach", "theft", 
                "stolen", "51% attack", "audit", "secure", "encryption", "key", "seed phrase",
                "private key", "multisig", "double-spend", "smart contract vulnerability"
            ],
            
            CryptoNewsCategory.SCALING: [
                "scaling", "scalability", "tps", "transactions per second", "throughput", "capacity", 
                "layer 2", "l2", "sidechain", "rollup", "optimistic rollup", "zk-rollup", "state channel",
                "payment channel", "lightning network", "raiden", "plasma", "sharding"
            ],
            
            # Industry and Business
            CryptoNewsCategory.PARTNERSHIP: [
                "partnership", "collaborate", "alliance", "joint venture", "strategic", "partner", 
                "cooperation", "agreement", "deal", "integration", "ecosystem", "initiative",
                "industry partner", "corporate partner", "business relationship"
            ],
            
            CryptoNewsCategory.ADOPTION: [
                "adoption", "use case", "implementation", "mainstream", "acceptance", "integration", 
                "real-world", "adoption rate", "user base", "customer", "merchant", "retail",
                "institutional adoption", "corporate adoption", "enterprise adoption", "country adoption"
            ],
            
            CryptoNewsCategory.INVESTMENT: [
                "investment", "fund", "venture capital", "vc", "seed", "series a", "funding", 
                "investor", "raise", "capital", "allocation", "portfolio", "diversification",
                "angel investor", "incubator", "accelerator", "treasury", "institutional investment"
            ],
            
            CryptoNewsCategory.EXCHANGE: [
                "exchange", "listing", "delisting", "trading pair", "trading volume", "liquidity", 
                "order book", "withdrawal", "deposit", "cex", "dex", "centralized", "decentralized",
                "binance", "coinbase", "kraken", "ftx", "bitfinex", "huobi", "okex", "gemini"
            ],
            
            # Ecosystem
            CryptoNewsCategory.DEFI: [
                "defi", "decentralized finance", "protocol", "lending", "borrowing", "yield", "farm", 
                "liquidity", "pool", "impermanent loss", "amm", "dex", "swap", "aggregator",
                "aave", "compound", "uniswap", "curve", "balancer", "maker", "tvl", "total value locked"
            ],
            
            CryptoNewsCategory.NFT: [
                "nft", "non-fungible token", "collectible", "art", "auction", "marketplace", "mint", 
                "creator", "royalty", "opensea", "rarible", "foundation", "superrare", "metaverse",
                "gaming", "play-to-earn", "p2e", "avatar", "virtual land", "digital ownership"
            ],
            
            CryptoNewsCategory.GOVERNANCE: [
                "governance", "dao", "decentralized autonomous organization", "proposal", "vote", 
                "voting power", "token holder", "community", "decision", "quorum", "snapshot",
                "governance token", "governance forum", "protocol governance", "on-chain governance"
            ],
            
            CryptoNewsCategory.STAKING: [
                "staking", "stake", "validator", "delegation", "reward", "yield", "apy", "apr", 
                "passive income", "proof of stake", "pos", "lock", "unbonding", "slashing", 
                "staking pool", "liquid staking", "staking service", "staking provider"
            ],
            
            # Macro and Fundamental
            CryptoNewsCategory.MACROECONOMIC: [
                "inflation", "deflation", "monetary policy", "fiscal policy", "fed", "interest rate", 
                "economic", "recession", "gdp", "economy", "financial crisis", "debt", "stimulus",
                "federal reserve", "central bank", "treasury", "fiscal", "monetary", "fomc"
            ],
            
            CryptoNewsCategory.INSTITUTIONAL: [
                "institutional", "institution", "bank", "financial institution", "wall street", 
                "hedge fund", "pension", "endowment", "etf", "exchange-traded fund", "futures",
                "cme", "bakkt", "grayscale", "microstrategy", "tesla", "square", "fund manager"
            ],
            
            CryptoNewsCategory.MINING: [
                "mining", "miner", "hash rate", "hashrate", "asic", "rig", "pool", "proof of work", 
                "pow", "difficulty", "block reward", "halving", "energy", "electricity", "power",
                "bitmain", "antminer", "f2pool", "antpool", "slushpool", "poolin", "gpu mining"
            ],
            
            CryptoNewsCategory.SUPPLY_METRICS: [
                "supply", "circulating supply", "total supply", "max supply", "inflation", "deflation", 
                "issuance", "distribution", "token economics", "tokenomics", "burn", "mint", "emission",
                "vesting", "unlock", "release", "supply schedule", "token supply"
            ],
            
            # Impact-based categories
            CryptoNewsCategory.HIGH_IMPACT: [
                "major", "significant", "breaking", "critical", "crucial", "game-changer", "milestone", 
                "historic", "unprecedented", "revolutionary", "dramatic", "extraordinary", "radical",
                "transformative", "groundbreaking", "landmark", "watershed", "pivotal"
            ],
            
            CryptoNewsCategory.TRENDING: [
                "trending", "viral", "hot", "buzz", "popularity", "attention", "spotlight", "hype", 
                "mainstream", "media coverage", "social media", "twitter", "reddit", "trending topic",
                "popular", "sensation", "current", "today", "now", "present"
            ],
            
            CryptoNewsCategory.SPECULATIVE: [
                "rumor", "speculation", "unconfirmed", "alleged", "reportedly", "sources say", 
                "anonymous", "insider", "leak", "hint", "tease", "potential", "possible", "might",
                "could", "may", "rumored", "speculated", "claimed", "purported"
            ],
            
            CryptoNewsCategory.CONTRARIAN: [
                "contrarian", "contrary", "against", "opposite", "different view", "alternative", 
                "challenge", "question", "doubt", "skeptical", "criticism", "critical", "bearish",
                "overvalued", "bubble", "correction", "overpriced", "unsustainable"
            ]
        }
    
    def _compile_category_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for each category.
        
        Returns:
            Dictionary mapping categories to compiled regex patterns
        """
        patterns = {}
        
        for category, keywords in self.category_keywords.items():
            # Create pattern with word boundaries to match whole words
            pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            patterns[category] = re.compile(pattern, re.IGNORECASE)
            
        return patterns
    
    def _initialize_asset_terms(self) -> Dict[str, List[str]]:
        """Initialize asset-specific terminology.
        
        Returns:
            Dictionary mapping asset symbols to related terms
        """
        return {
            "BTC": ["bitcoin", "btc", "satoshi", "btc/usd", "bitcoin core", "segwit", "lightning network",
                   "halving", "bitcoiner", "digital gold", "btc dominance"],
            
            "ETH": ["ethereum", "eth", "ether", "vitalik", "buterin", "eth2", "ethereum 2.0", "evm",
                   "ethereum virtual machine", "gas", "gas fee", "eip", "solidity", "web3", "dapp"],
            
            "SOL": ["solana", "sol", "solend", "serum", "phantom", "wormhole", "solanart", "metaplex",
                   "solana ecosystem", "sealevel", "gulf stream", "proof of history", "poh"],
            
            "XRP": ["ripple", "xrp", "xrp ledger", "xrpl", "ripple labs", "garlinghouse", "flare",
                   "interledger", "xrapid", "xcurrent", "sec case", "ripple lawsuit"],
            
            "DOGE": ["dogecoin", "doge", "shiba inu", "elon", "musk", "doge army", "much wow",
                    "meme coin", "dogethereum", "doge/usd"],
            
            "ADA": ["cardano", "ada", "hoskinson", "ouroboros", "plutus", "marlowe", "haskell",
                   "basho", "voltaire", "hydra", "stake pool", "cardano ecosystem"],
            
            "DOT": ["polkadot", "dot", "gavin", "wood", "substrate", "parachain", "kusama",
                   "web3 foundation", "relay chain", "cross-chain", "interoperability"],
            
            "LINK": ["chainlink", "link", "oracle", "data feed", "node operator", "vrf",
                    "chainlink ecosystem", "external data", "defi oracle"],
            
            "USDT": ["tether", "usdt", "stablecoin", "usdt reserve", "tether controversy",
                    "usdt peg", "tether treasury", "usdt printing"],
            
            "USDC": ["usd coin", "usdc", "circle", "stablecoin", "usdc reserve", "centre",
                    "usdc attestation", "regulated stablecoin"],
            
            "UNI": ["uniswap", "uni", "amm", "liquidity provider", "lp", "swap", "uniswap v2",
                   "uniswap v3", "dex", "decentralized exchange", "liquidity mining"],
            
            "AAVE": ["aave", "lend", "lending protocol", "borrowing", "flash loan", "atoken",
                    "aave governance", "liquidity market", "money market"]
        }
        
    async def categorize_article(self, article: NewsArticle) -> Dict[str, float]:
        """Categorize a news article into crypto-specific categories.
        
        Args:
            article: The NewsArticle to categorize
            
        Returns:
            Dictionary mapping categories to confidence scores (0-1)
        """
        # Prepare text for analysis
        text = f"{article.title} {article.content}"
        
        # Run rule-based categorization
        rule_based_scores = self._rule_based_categorization(text, article)
        
        # Run ML-based categorization if enabled
        ml_scores = {}
        if self.enable_ml_categorization and self.zero_shot_classifier:
            ml_scores = await self._ml_based_categorization(text)
            
        # Combine scores from both methods
        combined_scores = self._combine_categorization_scores(rule_based_scores, ml_scores)
        
        # Filter out low-confidence categories
        filtered_scores = {
            category: score for category, score in combined_scores.items() 
            if score >= self.min_category_confidence
        }
        
        # If no categories meet threshold, include the highest scoring one
        if not filtered_scores and combined_scores:
            top_category = max(combined_scores.items(), key=lambda x: x[1])
            filtered_scores = {top_category[0]: top_category[1]}
            
        return filtered_scores
    
    def _rule_based_categorization(
        self, 
        text: str,
        article: NewsArticle
    ) -> Dict[str, float]:
        """Perform rule-based categorization using keyword matching.
        
        Args:
            text: The text to analyze
            article: The NewsArticle object with metadata
            
        Returns:
            Dictionary mapping categories to confidence scores (0-1)
        """
        text_lower = text.lower()
        scores = {}
        
        # Calculate base scores from keyword matches
        for category, pattern in self.category_patterns.items():
            # Find all matching keywords
            matches = pattern.findall(text_lower)
            
            if not matches:
                continue
                
            # Calculate score based on match count and text length
            unique_matches = set(matches)
            match_count = len(matches)
            unique_count = len(unique_matches)
            
            # Normalize by text length and category keyword count
            text_length_factor = min(1.0, len(text) / 5000)  # Cap for very long texts
            keyword_count = len(self.category_keywords[category])
            
            # Base score calculation
            base_score = min(0.95, (0.3 + (unique_count / keyword_count) * 0.3 + 
                              (match_count / (len(text_lower.split()) / 20)) * 0.4))
            
            # Boost score for matches in title (more important)
            title_matches = sum(1 for match in unique_matches if match in article.title.lower())
            title_boost = min(0.2, title_matches * 0.1)
            
            # Adjust score for recency (newer articles more likely to be relevant)
            age_in_days = (datetime.now() - article.published_at).days
            recency_factor = max(0.85, 1.0 - (age_in_days / 30) * 0.15)  # Older articles get slightly reduced score
            
            # Calculate final score
            score = min(0.95, (base_score + title_boost) * recency_factor)
            
            # Store if above low threshold
            if score >= 0.25:
                scores[category] = score
        
        # Apply special rules for asset-specific content
        self._apply_asset_specific_rules(text_lower, scores, article)
        
        # Apply rules for impact-based categories
        self._apply_impact_based_rules(text_lower, scores, article)
        
        return scores
    
    def _apply_asset_specific_rules(
        self, 
        text: str, 
        scores: Dict[str, float],
        article: NewsArticle
    ) -> None:
        """Apply asset-specific categorization rules.
        
        Args:
            text: The lowercase text to analyze
            scores: Dictionary of category scores to update
            article: The NewsArticle object with metadata
        """
        # Check for strong asset focus
        asset_mention_counts = {}
        
        for asset, terms in self.asset_terms.items():
            # Count mentions of each asset
            mention_count = sum(text.count(term) for term in terms)
            if mention_count > 0:
                asset_mention_counts[asset] = mention_count
        
        # If the article has relevance scores, use them as additional signal
        if hasattr(article, 'relevance_scores') and article.relevance_scores:
            for asset, relevance in article.relevance_scores.items():
                if asset in asset_mention_counts:
                    # Scale the mention count by relevance score
                    asset_mention_counts[asset] *= (0.5 + relevance * 0.5)
        
        # If we have strong asset focus, boost related categories
        if asset_mention_counts:
            # Find dominant asset (most mentioned)
            dominant_asset = max(asset_mention_counts.items(), key=lambda x: x[1], default=(None, 0))
            
            if dominant_asset[0]:
                asset, mention_count = dominant_asset
                
                # Apply asset-specific category boosts
                if asset == "BTC":
                    # Bitcoin often relates to market trends, mining
                    self._boost_category_score(scores, CryptoNewsCategory.MARKET_ANALYSIS, 0.1)
                    self._boost_category_score(scores, CryptoNewsCategory.MINING, 0.15)
                    self._boost_category_score(scores, CryptoNewsCategory.INSTITUTIONAL, 0.1)
                
                elif asset == "ETH":
                    # Ethereum often relates to development, DeFi, scaling
                    self._boost_category_score(scores, CryptoNewsCategory.DEVELOPMENT, 0.1)
                    self._boost_category_score(scores, CryptoNewsCategory.DEFI, 0.15)
                    self._boost_category_score(scores, CryptoNewsCategory.SCALING, 0.15)
                    self._boost_category_score(scores, CryptoNewsCategory.PROTOCOL_UPDATE, 0.1)
                
                elif asset == "SOL":
                    # Solana often focuses on performance, ecosystem growth
                    self._boost_category_score(scores, CryptoNewsCategory.SCALING, 0.15)
                    self._boost_category_score(scores, CryptoNewsCategory.DEVELOPMENT, 0.1)
                    self._boost_category_score(scores, CryptoNewsCategory.NFT, 0.1)
                
                elif asset == "XRP":
                    # XRP often relates to regulation, legal issues, partnerships
                    self._boost_category_score(scores, CryptoNewsCategory.REGULATION, 0.15)
                    self._boost_category_score(scores, CryptoNewsCategory.LEGAL, 0.15)
                    self._boost_category_score(scores, CryptoNewsCategory.PARTNERSHIP, 0.1)
        
        # Check for multi-asset comparisons (often market analysis)
        asset_count = len(asset_mention_counts)
        if asset_count > 2:
            self._boost_category_score(scores, CryptoNewsCategory.MARKET_ANALYSIS, min(0.15, asset_count * 0.05))
    
    def _apply_impact_based_rules(
        self, 
        text: str, 
        scores: Dict[str, float],
        article: NewsArticle
    ) -> None:
        """Apply rules for impact-based categories.
        
        Args:
            text: The lowercase text to analyze
            scores: Dictionary of category scores to update
            article: The NewsArticle object with metadata
        """
        # Check for high-impact terms in title
        title_lower = article.title.lower()
        
        # High-impact news often has certain phrases in the title
        high_impact_phrases = ["breaking", "just in", "urgent", "alert", "major", "key", "critical"]
        if any(phrase in title_lower for phrase in high_impact_phrases):
            self._boost_category_score(scores, CryptoNewsCategory.HIGH_IMPACT, 0.2)
            
        # Trending news often references current timeframes
        trending_phrases = ["today", "just now", "this week", "this morning", "latest", "newest", "trending"]
        if any(phrase in title_lower for phrase in trending_phrases):
            self._boost_category_score(scores, CryptoNewsCategory.TRENDING, 0.15)
            
        # Speculative news contains uncertain language
        speculative_phrases = ["could", "may", "might", "possibly", "reportedly", "rumor", "speculation", "sources say"]
        if any(phrase in title_lower for phrase in speculative_phrases):
            self._boost_category_score(scores, CryptoNewsCategory.SPECULATIVE, 0.2)
            
        # Contrarian views often challenge prevailing narratives
        contrarian_phrases = ["despite", "however", "against", "contrary", "nonetheless", "challenge", "opposing"]
        if any(phrase in title_lower for phrase in contrarian_phrases):
            self._boost_category_score(scores, CryptoNewsCategory.CONTRARIAN, 0.15)
        
        # Adjust impact categories based on source credibility
        high_credibility_sources = ["coindesk", "the block", "bloomberg", "forbes", "reuters", "wsj"]
        medium_credibility_sources = ["cointelegraph", "decrypt", "cryptoslate"]
        
        source_lower = article.source.lower()
        
        if any(source in source_lower for source in high_credibility_sources):
            # Higher credibility sources get boost for high impact, decrease for speculative
            self._boost_category_score(scores, CryptoNewsCategory.HIGH_IMPACT, 0.1)
            self._reduce_category_score(scores, CryptoNewsCategory.SPECULATIVE, 0.1)
            
        elif any(source in source_lower for source in medium_credibility_sources):
            # No adjustment for medium credibility sources
            pass
            
        else:
            # Lower credibility sources get reduced high impact, increased speculative
            self._reduce_category_score(scores, CryptoNewsCategory.HIGH_IMPACT, 0.1)
            self._boost_category_score(scores, CryptoNewsCategory.SPECULATIVE, 0.1)
    
    def _boost_category_score(
        self, 
        scores: Dict[str, float], 
        category: str, 
        boost: float
    ) -> None:
        """Boost a category score, creating it if it doesn't exist.
        
        Args:
            scores: Dictionary of scores to update
            category: Category to boost
            boost: Amount to increase score (0-1)
        """
        if category in scores:
            scores[category] = min(0.95, scores[category] + boost)
        else:
            scores[category] = min(0.95, 0.3 + boost)  # Start with base score if not present
    
    def _reduce_category_score(
        self, 
        scores: Dict[str, float], 
        category: str, 
        reduction: float
    ) -> None:
        """Reduce a category score if it exists.
        
        Args:
            scores: Dictionary of scores to update
            category: Category to reduce
            reduction: Amount to decrease score (0-1)
        """
        if category in scores:
            scores[category] = max(0.0, scores[category] - reduction)
            
            # Remove if score falls below threshold
            if scores[category] < 0.25:
                del scores[category]
    
    async def _ml_based_categorization(self, text: str) -> Dict[str, float]:
        """Perform ML-based categorization using zero-shot classification.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary mapping categories to confidence scores (0-1)
        """
        if not self.zero_shot_classifier:
            return {}
            
        try:
            # Prepare text (truncate if too long)
            max_length = 1024
            if len(text) > max_length:
                # Keep the beginning and end, which often contain the most relevant info
                beginning = text[:max_length//2]
                end = text[-max_length//2:]
                text = beginning + "... " + end
            
            # List of categories to classify against
            categories = list(self.category_keywords.keys())
            
            # Organize categories into manageable batches (the model has input limits)
            batch_size = 15
            scores = {}
            
            for i in range(0, len(categories), batch_size):
                batch_categories = categories[i:i+batch_size]
                
                # Run model in thread to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.zero_shot_classifier(text, batch_categories, multi_label=True)
                )
                
                # Process results
                for category, score in zip(result["labels"], result["scores"]):
                    scores[category] = score
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error performing ML-based categorization: {e}")
            return {}
    
    def _combine_categorization_scores(
        self, 
        rule_scores: Dict[str, float], 
        ml_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine scores from rule-based and ML-based approaches.
        
        Args:
            rule_scores: Scores from rule-based categorization
            ml_scores: Scores from ML-based categorization
            
        Returns:
            Combined category scores
        """
        # If no ML scores, just return rule scores
        if not ml_scores:
            return rule_scores
            
        # Combine scores with rule-based having more weight
        combined_scores = {}
        all_categories = set(rule_scores.keys()) | set(ml_scores.keys())
        
        for category in all_categories:
            rule_score = rule_scores.get(category, 0.0)
            ml_score = ml_scores.get(category, 0.0)
            
            # Weight rule scores more heavily
            combined_scores[category] = (rule_score * 0.7) + (ml_score * 0.3)
            
        return combined_scores


class CryptoNewsTopicGraph:
    """Graph representation of cryptocurrency news topics and their relationships.
    
    This class builds and analyzes a graph of news articles, categories, assets,
    and other entities to identify trends, connections, and insights.
    """
    
    def __init__(self):
        """Initialize the topic graph."""
        self.logger = get_logger("news_analyzer", "topic_graph")
        
        # Initialize graph
        self.graph = nx.Graph()
        
        # Track when the graph was last updated
        self.last_update = datetime.now()
        
        # Minimum weight for edges
        self.min_edge_weight = 0.4
    
    def add_article(self, article: NewsArticle, categories: Dict[str, float]) -> None:
        """Add an article and its categories to the topic graph.
        
        Args:
            article: The NewsArticle to add
            categories: Dictionary of category to confidence scores (0-1)
        """
        # Skip if no categories
        if not categories:
            return
            
        # Add article node
        self.graph.add_node(
            article.article_id,
            type="article",
            title=article.title,
            url=article.url,
            source=article.source,
            published_at=article.published_at.isoformat()
        )
        
        # Add category nodes and connections
        for category, score in categories.items():
            if score < self.min_edge_weight:
                continue
                
            # Add category node if not exists
            if not self.graph.has_node(category):
                self.graph.add_node(
                    category,
                    type="category",
                    name=category
                )
                
            # Connect article to category
            self.graph.add_edge(
                article.article_id,
                category,
                weight=score,
                type="has_category"
            )
        
        # Add connections to assets if relevance scores exist
        if hasattr(article, 'relevance_scores') and article.relevance_scores:
            for asset, relevance in article.relevance_scores.items():
                if relevance < self.min_edge_weight:
                    continue
                    
                # Add asset node if not exists
                asset_id = f"asset:{asset}"
                if not self.graph.has_node(asset_id):
                    self.graph.add_node(
                        asset_id,
                        type="asset",
                        symbol=asset
                    )
                    
                # Connect article to asset
                self.graph.add_edge(
                    article.article_id,
                    asset_id,
                    weight=relevance,
                    type="about_asset"
                )
        
        # Connect with existing articles based on shared categories and assets
        self._connect_related_articles(article)
        
        # Update timestamp
        self.last_update = datetime.now()
    
    def _connect_related_articles(self, article: NewsArticle) -> None:
        """Connect the article to related existing articles in the graph.
        
        Args:
            article: The NewsArticle to connect
        """
        # Get all article nodes except the current one
        article_nodes = [n for n in self.graph.nodes if 
                       self.graph.nodes[n].get("type") == "article" and 
                       n != article.article_id]
        
        # Get categories and assets of the new article
        article_categories = {node for node in self.graph.neighbors(article.article_id) 
                           if self.graph.nodes[node].get("type") == "category"}
        
        article_assets = {node for node in self.graph.neighbors(article.article_id) 
                        if self.graph.nodes[node].get("type") == "asset"}
        
        # Connect to similar articles
        for other_article in article_nodes:
            # Get categories and assets of the other article
            other_categories = {node for node in self.graph.neighbors(other_article) 
                              if self.graph.nodes[node].get("type") == "category"}
            
            other_assets = {node for node in self.graph.neighbors(other_article) 
                          if self.graph.nodes[node].get("type") == "asset"}
            
            # Calculate similarity based on shared categories and assets
            shared_categories = article_categories & other_categories
            shared_assets = article_assets & other_assets
            
            # Weigh assets more heavily than categories
            similarity = (len(shared_categories) * 0.3) + (len(shared_assets) * 0.7)
            
            # Normalize by total possible overlap
            total_categories = max(1, len(article_categories | other_categories))
            total_assets = max(1, len(article_assets | other_assets))
            normalized_similarity = similarity / (total_categories * 0.3 + total_assets * 0.7)
            
            # Only connect if similarity is above threshold
            if normalized_similarity >= 0.3:
                # Check if articles are from the same day for temporal relevance
                try:
                    other_published = datetime.fromisoformat(self.graph.nodes[other_article]["published_at"])
                    time_diff = abs((article.published_at - other_published).total_seconds() / 3600)
                    
                    # Boost similarity if articles published close to each other
                    if time_diff <= 24:
                        temporal_factor = 1.0 - (time_diff / 24) * 0.5  # 1.0 down to 0.5
                        normalized_similarity *= temporal_factor
                except (ValueError, KeyError):
                    pass
                
                self.graph.add_edge(
                    article.article_id,
                    other_article,
                    weight=normalized_similarity,
                    type="related_to"
                )
    
    def get_related_articles(self, article_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get articles related to a specific article.
        
        Args:
            article_id: ID of the article to find related articles for
            limit: Maximum number of articles to return
            
        Returns:
            List of related article dictionaries
        """
        if not self.graph.has_node(article_id):
            return []
            
        # Find neighbors that are articles
        neighbors = [(n, self.graph[article_id][n].get("weight", 0.0)) for n in self.graph.neighbors(article_id)
                    if self.graph.nodes[n].get("type") == "article"]
        
        # Sort by weight (similarity) descending
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Get top related articles
        related_articles = []
        
        for node_id, weight in neighbors[:limit]:
            article_data = self.graph.nodes[node_id]
            related_articles.append({
                "id": node_id,
                "title": article_data.get("title", ""),
                "url": article_data.get("url", ""),
                "source": article_data.get("source", ""),
                "published_at": article_data.get("published_at", ""),
                "similarity": weight
            })
            
        return related_articles
    
    def get_trending_categories(self, timeframe_hours: int = 24, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending categories within a specified timeframe.
        
        Args:
            timeframe_hours: Number of hours to consider
            limit: Maximum number of categories to return
            
        Returns:
            List of trending category dictionaries
        """
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        # Find recent articles
        recent_articles = [n for n in self.graph.nodes 
                         if self.graph.nodes[n].get("type") == "article" and
                         datetime.fromisoformat(self.graph.nodes[n].get("published_at", datetime.min.isoformat())) >= cutoff_time]
        
        # Count categories
        category_counts: Dict[str, int] = defaultdict(int)
        
        for article in recent_articles:
            for neighbor in self.graph.neighbors(article):
                if self.graph.nodes[neighbor].get("type") == "category":
                    # Add the edge weight to give more significance to stronger categorizations
                    category_counts[neighbor] += 1 * self.graph[article][neighbor].get("weight", 1.0)
        
        # Get top categories
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Format results
        trending_categories = []
        
        for category, score in top_categories:
            # Count articles with this category
            article_count = sum(1 for a in recent_articles if category in self.graph.neighbors(a))
            
            trending_categories.append({
                "category": category,
                "score": score,
                "article_count": article_count
            })
            
        return trending_categories
    
    def get_asset_category_associations(self, asset: str) -> Dict[str, float]:
        """Get categories most associated with a specific asset.
        
        Args:
            asset: The asset symbol to analyze
            
        Returns:
            Dictionary mapping categories to association scores (0-1)
        """
        asset_id = f"asset:{asset}"
        
        if not self.graph.has_node(asset_id):
            return {}
            
        # Find articles about this asset
        asset_articles = [n for n in self.graph.neighbors(asset_id)
                        if self.graph.nodes[n].get("type") == "article"]
        
        if not asset_articles:
            return {}
            
        # Count category frequencies
        category_weights: Dict[str, float] = defaultdict(float)
        category_counts: Dict[str, int] = defaultdict(int)
        
        for article in asset_articles:
            for neighbor in self.graph.neighbors(article):
                if self.graph.nodes[neighbor].get("type") == "category":
                    category = neighbor
                    weight = self.graph[article][neighbor].get("weight", 0.0)
                    
                    category_weights[category] += weight
                    category_counts[category] += 1
        
        # Calculate average weights
        avg_weights = {}
        for category, total_weight in category_weights.items():
            count = category_counts[category]
            avg_weights[category] = total_weight / count
            
        # Normalize to get the top associations
        if avg_weights:
            max_weight = max(avg_weights.values())
            return {
                category: weight / max_weight 
                for category, weight in avg_weights.items()
            }
            
        return {}
    
    def find_narrative_clusters(self, timeframe_hours: int = 72) -> List[Dict[str, Any]]:
        """Find clusters of related articles forming narratives.
        
        Args:
            timeframe_hours: Number of hours to consider
            
        Returns:
            List of narrative cluster dictionaries
        """
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        # Create a subgraph with recent articles
        recent_articles = [n for n in self.graph.nodes 
                         if self.graph.nodes[n].get("type") == "article" and
                         datetime.fromisoformat(self.graph.nodes[n].get("published_at", datetime.min.isoformat())) >= cutoff_time]
        
        # If not enough articles, return empty list
        if len(recent_articles) < 5:
            return []
            
        # Create a new graph with just the articles and their relationships
        article_graph = nx.Graph()
        
        for article in recent_articles:
            article_graph.add_node(article, **self.graph.nodes[article])
            
            for other_article in recent_articles:
                if article != other_article and self.graph.has_edge(article, other_article):
                    weight = self.graph[article][other_article].get("weight", 0.0)
                    if weight >= 0.3:  # Only include stronger relationships
                        article_graph.add_edge(article, other_article, weight=weight)
        
        # Find communities using Louvain algorithm
        try:
            from community import best_partition
            partition = best_partition(article_graph)
        except ImportError:
            # Fall back to connected components if community package is not available
            self.logger.warning("Community detection package not available, falling back to connected components")
            clusters = list(nx.connected_components(article_graph))
            partition = {}
            for i, cluster in enumerate(clusters):
                for node in cluster:
                    partition[node] = i
        
        # Organize articles by community
        communities: Dict[int, List[str]] = defaultdict(list)
        for article, community_id in partition.items():
            communities[community_id].append(article)
            
        # Process each community
        narratives = []
        
        for community_id, articles in communities.items():
            # Skip small communities
            if len(articles) < 2:
                continue
                
            # Extract common categories and assets
            category_counts: Dict[str, int] = defaultdict(int)
            asset_counts: Dict[str, int] = defaultdict(int)
            
            for article in articles:
                for neighbor in self.graph.neighbors(article):
                    node_type = self.graph.nodes[neighbor].get("type")
                    
                    if node_type == "category":
                        category_counts[neighbor] += 1
                    elif node_type == "asset":
                        asset_symbol = self.graph.nodes[neighbor].get("symbol", "")
                        if asset_symbol:
                            asset_counts[asset_symbol] += 1
            
            # Determine the main theme (top categories)
            top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_assets = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Get article data
            article_data = []
            for article in articles:
                article_data.append({
                    "id": article,
                    "title": self.graph.nodes[article].get("title", ""),
                    "source": self.graph.nodes[article].get("source", ""),
                    "published_at": self.graph.nodes[article].get("published_at", "")
                })
                
            # Sort by publication date
            article_data.sort(key=lambda x: x.get("published_at", ""), reverse=True)
            
            # Generate narrative title
            title = "Cryptocurrency News Cluster"
            if top_assets and top_categories:
                title = f"{top_assets[0][0]} {top_categories[0][0].replace('_', ' ').title()}"
                
            narratives.append({
                "id": f"narrative_{community_id}",
                "title": title,
                "size": len(articles),
                "categories": [{"category": c, "count": count} for c, count in top_categories],
                "assets": [{"asset": a, "count": count} for a, count in top_assets],
                "articles": article_data[:5]  # Limit to top 5 articles
            })
        
        # Sort by size descending
        narratives.sort(key=lambda x: x["size"], reverse=True)
        
        return narratives