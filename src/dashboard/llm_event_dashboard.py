"""
LLM and Event Detection Dashboard.

This module provides dashboard components for visualizing LLM-based sentiment analysis,
consensus signals, and real-time event detection.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import random
import json

from fastapi import Request, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.analysis_agents.sentiment.llm_service import LLMService
from src.analysis_agents.sentiment.consensus_system import MultiModelConsensusAgent
from src.analysis_agents.early_detection.realtime_detector import RealtimeEventDetector
from src.analysis_agents.early_detection.sentiment_integration import SentimentEventIntegration
from src.common.config import config
from src.common.events import event_bus

logger = logging.getLogger(__name__)

# Initialize the router
router = APIRouter(prefix="/llm-events", tags=["llm-events"])

# Set up templates
templates_dir = Path("dashboard_templates")
templates = Jinja2Templates(directory=str(templates_dir))


class LLMEventDashboard:
    """Dashboard controller for LLM sentiment and event detection."""
    
    def __init__(self):
        """Initialize the LLM and Event Detection Dashboard."""
        self.last_update = datetime.now() - timedelta(minutes=10)
        self.cache_ttl = 60  # 1 minute (shorter for real-time data)
        self.cache: Dict[str, Any] = {}
        
        # System components (will be initialized later)
        self.llm_service = None
        self.consensus_agent = None
        self.event_detector = None
        self.integration = None
        
        # Event storage (in-memory for dashboard)
        self.recent_events = []
        self.recent_signals = []
        self.consensus_history = {}
        
        # Event registration flag
        self.events_registered = False
    
    async def initialize(self):
        """Initialize the dashboard components."""
        logger.info("Initializing LLM Event Dashboard")
        
        try:
            # Initialize LLM service
            self.llm_service = LLMService()
            await self.llm_service.initialize()
            
            # Initialize consensus agent
            self.consensus_agent = MultiModelConsensusAgent("consensus")
            await self.consensus_agent.initialize()
            
            # Initialize event detector
            self.event_detector = RealtimeEventDetector()
            await self.event_detector.initialize()
            
            # Initialize integration
            self.integration = SentimentEventIntegration(self.event_detector, self.consensus_agent)
            await self.integration.initialize()
            
            # Register event handlers
            if not self.events_registered:
                self._register_event_handlers()
                self.events_registered = True
            
            # Start components
            await self.consensus_agent.start()
            await self.event_detector.start()
            
            logger.info("LLM Event Dashboard initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LLM Event Dashboard: {e}")
    
    async def _register_event_handlers(self):
        """Register handlers for various events."""
        # Handle early event detection events
        async def handle_detected_event(event):
            payload = event.get("payload", {})
            # Store only the most recent 20 events
            self.recent_events.insert(0, {
                "id": payload.get("event_id", ""),
                "title": payload.get("title", "Unknown Event"),
                "category": payload.get("category", "unknown"),
                "confidence": payload.get("confidence", 0),
                "detected_at": datetime.now().isoformat(),
                "impact": payload.get("impact_assessment", {}),
                "full_event": payload
            })
            self.recent_events = self.recent_events[:20]
        
        # Handle early event signals
        async def handle_event_signal(signal):
            payload = signal.get("payload", {})
            # Store only the most recent 20 signals
            self.recent_signals.insert(0, {
                "id": payload.get("signal_id", ""),
                "title": payload.get("title", "Unknown Signal"),
                "confidence": payload.get("confidence", 0),
                "created_at": datetime.now().isoformat(),
                "expires_at": payload.get("expires_at"),
                "assets": payload.get("assets", {}),
                "priority": payload.get("priority", 0),
                "full_signal": payload
            })
            self.recent_signals = self.recent_signals[:20]
        
        # Register handlers
        await event_bus.subscribe("RealtimeEventDetected", handle_detected_event)
        await event_bus.subscribe("EarlyEventSignal", handle_event_signal)
        
        logger.info("Event handlers registered for LLM Event Dashboard")
    
    async def get_dashboard_data(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        """
        Get all data required for the LLM and event detection dashboard.
        
        Args:
            symbol: The trading pair symbol (default: "BTC/USDT")
            
        Returns:
            Dictionary containing all dashboard data
        """
        # Check if we have cached data that's still fresh
        cache_key = f"llm_dashboard_{symbol}"
        now = datetime.now()
        if cache_key in self.cache and (now - self.cache[cache_key]["timestamp"]).total_seconds() < self.cache_ttl:
            logger.debug(f"Using cached LLM dashboard data for {symbol}")
            return self.cache[cache_key]["data"]
        
        logger.info(f"Collecting LLM dashboard data for {symbol}")
        
        # Collect data from all sources
        try:
            # Run all data collection functions concurrently
            consensus_data, model_performance, detected_events, event_signals, llm_analysis = await asyncio.gather(
                self._get_consensus_data(symbol),
                self._get_model_performance(),
                self._get_detected_events(symbol),
                self._get_event_signals(symbol),
                self._get_llm_analysis(symbol)
            )
            
            # Combine all data
            dashboard_data = {
                "consensus_data": consensus_data,
                "model_performance": model_performance,
                "detected_events": detected_events,
                "event_signals": event_signals,
                "llm_analysis": llm_analysis,
                "current_time": now.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Cache the data
            self.cache[cache_key] = {
                "timestamp": now,
                "data": dashboard_data
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error collecting LLM dashboard data: {e}")
            # Return mock data in case of error
            return self._get_mock_dashboard_data(symbol)
    
    async def _get_consensus_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get consensus data for a symbol.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Dictionary with consensus data
        """
        try:
            # If consensus agent isn't initialized, return mock data
            if not self.consensus_agent:
                raise ValueError("Consensus agent not initialized")
            
            # Get consensus data
            consensus = self.consensus_agent.get_consensus(symbol)
            
            if not consensus:
                # No consensus data yet, return empty values
                return {
                    "value": 0.5,
                    "direction": "neutral",
                    "confidence": 0,
                    "disagreement_level": 0,
                    "source_count": 0,
                    "source_types": [],
                    "models": [],
                    "direction_counts": {"bullish": 0, "bearish": 0, "neutral": 0}
                }
            
            # Track this consensus in history
            if symbol not in self.consensus_history:
                self.consensus_history[symbol] = []
                
            # Add to history if it's a new value (check timestamp)
            last_update = consensus.get("last_update")
            if not self.consensus_history[symbol] or self.consensus_history[symbol][-1]["timestamp"] != last_update:
                self.consensus_history[symbol].append({
                    "timestamp": last_update,
                    "value": consensus.get("value", 0.5),
                    "confidence": consensus.get("confidence", 0),
                    "disagreement": consensus.get("disagreement_level", 0)
                })
                
            # Keep only the last 48 entries
            self.consensus_history[symbol] = self.consensus_history[symbol][-48:]
            
            # Return the consensus data
            return {
                "value": consensus.get("value", 0.5),
                "direction": consensus.get("direction", "neutral"),
                "confidence": consensus.get("confidence", 0),
                "disagreement_level": consensus.get("disagreement_level", 0),
                "source_count": consensus.get("source_count", 0),
                "source_types": consensus.get("source_types", []),
                "models": consensus.get("models", []),
                "direction_counts": consensus.get("direction_counts", {"bullish": 0, "bearish": 0, "neutral": 0}),
                "history": self.consensus_history.get(symbol, [])
            }
            
        except Exception as e:
            logger.error(f"Error getting consensus data: {e}")
            # Return mock data
            return {
                "value": 0.5,
                "direction": "neutral",
                "confidence": 0.7,
                "disagreement_level": 0.2,
                "source_count": 3,
                "source_types": ["social_media", "news", "llm"],
                "models": ["gpt-4", "finbert"],
                "direction_counts": {"bullish": 2, "bearish": 1, "neutral": 0},
                "history": [
                    {"timestamp": (datetime.now() - timedelta(hours=i)).isoformat(), 
                     "value": 0.5 + random.uniform(-0.2, 0.2),
                     "confidence": random.uniform(0.6, 0.9),
                     "disagreement": random.uniform(0.1, 0.4)}
                    for i in range(24)
                ]
            }
    
    async def _get_model_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for models and sources.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # If consensus agent isn't initialized, return mock data
            if not self.consensus_agent:
                raise ValueError("Consensus agent not initialized")
            
            # Get performance metrics
            if hasattr(self.consensus_agent, "get_performance_metrics"):
                performance = self.consensus_agent.get_performance_metrics()
                
                if performance:
                    return {
                        "source_performance": performance.get("source_performance", {}),
                        "model_performance": performance.get("model_performance", {}),
                        "timestamp": performance.get("timestamp")
                    }
            
            # If we couldn't get real data, raise exception to generate mock data
            raise ValueError("No performance metrics available")
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            # Return mock data
            source_performance = {
                "llm": random.uniform(0.7, 0.9),
                "social_media": random.uniform(0.5, 0.8),
                "news": random.uniform(0.6, 0.85),
                "market_sentiment": random.uniform(0.65, 0.85),
                "onchain": random.uniform(0.6, 0.8)
            }
            
            model_performance = {
                "gpt-4o": random.uniform(0.75, 0.9),
                "claude-3-opus": random.uniform(0.7, 0.9),
                "finbert": random.uniform(0.6, 0.8),
                "sentiment_transformer": random.uniform(0.55, 0.75)
            }
            
            return {
                "source_performance": source_performance,
                "model_performance": model_performance,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_detected_events(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get recently detected events.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            List of detected events
        """
        try:
            # Return recent events, filtering for the specific symbol
            symbol_events = []
            for event in self.recent_events:
                # Check if this event affects the symbol
                impact = event.get("impact", {})
                assets = impact.get("assets", {})
                
                # Map symbol to asset (e.g., "BTC/USDT" -> "BTC")
                asset = symbol.split("/")[0]
                
                # If we have impact data for this asset or it's a general event, include it
                if not assets or asset in assets or len(symbol_events) < 3:
                    symbol_events.append(event)
                    
                # Limit to 10 events
                if len(symbol_events) >= 10:
                    break
            
            return symbol_events
            
        except Exception as e:
            logger.error(f"Error getting detected events: {e}")
            # Return mock data
            return [
                {
                    "id": f"event_{i}",
                    "title": f"Mock Event {i} for {symbol}",
                    "category": random.choice(["monetary_policy", "market", "regulation", "technology"]),
                    "confidence": random.uniform(0.7, 0.95),
                    "detected_at": (datetime.now() - timedelta(hours=random.randint(0, 12))).isoformat(),
                    "impact": {
                        "magnitude": random.randint(3, 5),
                        "direction": random.choice(["positive", "negative", "mixed"])
                    }
                }
                for i in range(5)
            ]
    
    async def _get_event_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get recent event signals.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            List of event signals
        """
        try:
            # Return recent signals, filtering for the specific symbol
            symbol_signals = []
            for signal in self.recent_signals:
                # Check if this signal affects the symbol
                assets = signal.get("assets", {})
                
                # Map symbol to asset (e.g., "BTC/USDT" -> "BTC")
                asset = symbol.split("/")[0]
                
                # If we have data for this asset or it's a general signal, include it
                if not assets or asset in assets or len(symbol_signals) < 3:
                    symbol_signals.append(signal)
                    
                # Limit to 10 signals
                if len(symbol_signals) >= 10:
                    break
            
            return symbol_signals
            
        except Exception as e:
            logger.error(f"Error getting event signals: {e}")
            # Return mock data
            return [
                {
                    "id": f"signal_{i}",
                    "title": f"Mock Signal {i} for {symbol}",
                    "confidence": random.uniform(0.7, 0.95),
                    "created_at": (datetime.now() - timedelta(hours=random.randint(0, 8))).isoformat(),
                    "expires_at": (datetime.now() + timedelta(hours=random.randint(4, 24))).isoformat(),
                    "priority": random.randint(2, 5)
                }
                for i in range(3)
            ]
    
    async def _get_llm_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get LLM analysis for a symbol.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Dictionary with LLM analysis
        """
        try:
            # If LLM service isn't initialized, return mock data
            if not self.llm_service:
                raise ValueError("LLM service not initialized")
            
            # For real-time dashboard, we'll generate a fresh analysis
            # In a production system, this would likely use cached analysis or existing LLM outputs
            
            # Simple market context
            asset = symbol.split("/")[0]  # Get the base asset (e.g., "BTC" from "BTC/USDT")
            
            market_context = f"""Current market conditions for {asset}:
- Price has been relatively stable in the past 24 hours
- Trading volume is moderate
- Overall market sentiment has been mixed
- Most major cryptocurrencies are showing similar patterns
"""
            
            # Generate a basic analysis prompt
            event_prompt = f"""Analyze the current market conditions for {asset} and provide your assessment on:
1. Potential near-term price direction
2. Key factors to watch
3. Possible market-moving events on the horizon

Provide a concise, balanced analysis based on the available information."""
            
            # Use the LLM service to generate analysis
            analysis = await self.llm_service.assess_market_impact(
                event=event_prompt,
                market_context=market_context
            )
            
            # If we couldn't get real analysis, raise exception to generate mock data
            if not analysis:
                raise ValueError("Failed to generate LLM analysis")
            
            # Format the result
            return {
                "primary_direction": analysis.get("primary_impact_direction", "neutral"),
                "confidence": analysis.get("confidence", 0.7),
                "magnitude": analysis.get("impact_magnitude", 0.5),
                "duration": analysis.get("estimated_duration", "medium_term"),
                "reasoning": analysis.get("reasoning", "No reasoning provided"),
                "risk_factors": analysis.get("risk_factors", []),
                "timestamp": datetime.now().isoformat(),
                "model": analysis.get("_meta", {}).get("model", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            # Return mock data
            risk_factors = [
                f"Unexpected regulatory announcements related to {asset}",
                "Broader market volatility from macroeconomic conditions",
                "Technical resistance levels may limit upside potential",
                "Reduced liquidity during weekend hours"
            ]
            
            return {
                "primary_direction": random.choice(["positive", "negative", "neutral"]),
                "confidence": random.uniform(0.6, 0.9),
                "magnitude": random.uniform(0.3, 0.7),
                "duration": random.choice(["short_term", "medium_term"]),
                "reasoning": f"Based on current market conditions, {asset} appears to be in a consolidation phase with potential for a breakout in the coming days. Technical indicators suggest moderate directional bias with key support levels holding.",
                "risk_factors": random.sample(risk_factors, k=random.randint(2, 4)),
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4"
            }
    
    def _get_mock_dashboard_data(self, symbol: str) -> Dict[str, Any]:
        """
        Generate mock data for the dashboard in case of errors.
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Dictionary with mock dashboard data
        """
        # Consensus data
        consensus_data = {
            "value": 0.6,
            "direction": "bullish",
            "confidence": 0.8,
            "disagreement_level": 0.15,
            "source_count": 4,
            "source_types": ["llm", "social_media", "news", "market_sentiment"],
            "models": ["gpt-4", "finbert", "twitter-sentiment"],
            "direction_counts": {"bullish": 3, "bearish": 0, "neutral": 1},
            "history": [
                {"timestamp": (datetime.now() - timedelta(hours=i)).isoformat(), 
                "value": 0.5 + 0.1 * (i % 3 - 1),
                "confidence": 0.7 + 0.1 * (i % 2),
                "disagreement": 0.1 + 0.05 * (i % 3)}
                for i in range(24)
            ]
        }
        
        # Model performance
        model_performance = {
            "source_performance": {
                "llm": 0.85,
                "social_media": 0.72,
                "news": 0.78,
                "market_sentiment": 0.81,
                "onchain": 0.75
            },
            "model_performance": {
                "gpt-4": 0.87,
                "claude-3": 0.85,
                "finbert": 0.76,
                "sentiment_transformer": 0.72
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Detected events
        detected_events = [
            {
                "id": "event_1",
                "title": f"SEC Guidance on {symbol.split('/')[0]} Classification Expected Soon",
                "category": "regulation",
                "confidence": 0.82,
                "detected_at": (datetime.now() - timedelta(hours=3)).isoformat(),
                "impact": {
                    "magnitude": 4,
                    "direction": "mixed"
                }
            },
            {
                "id": "event_2",
                "title": f"Major Exchange Announces {symbol.split('/')[0]} Staking Program",
                "category": "market",
                "confidence": 0.91,
                "detected_at": (datetime.now() - timedelta(hours=5)).isoformat(),
                "impact": {
                    "magnitude": 3,
                    "direction": "positive"
                }
            }
        ]
        
        # Event signals
        event_signals = [
            {
                "id": "signal_1",
                "title": f"Potential Short-Term {symbol.split('/')[0]} Rally",
                "confidence": 0.85,
                "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=12)).isoformat(),
                "priority": 4
            }
        ]
        
        # LLM analysis
        llm_analysis = {
            "primary_direction": "positive",
            "confidence": 0.75,
            "magnitude": 0.6,
            "duration": "medium_term",
            "reasoning": f"{symbol.split('/')[0]} is showing signs of accumulation with decreasing selling pressure. Key technical indicators suggest a potential upward move in the next 1-2 weeks, though resistance levels may provide temporary obstacles.",
            "risk_factors": [
                "Unexpected regulatory news",
                "Broader market volatility",
                "Technical resistance at key levels"
            ],
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4"
        }
        
        return {
            "consensus_data": consensus_data,
            "model_performance": model_performance,
            "detected_events": detected_events,
            "event_signals": event_signals,
            "llm_analysis": llm_analysis,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# Initialize the dashboard controller
llm_event_dashboard = LLMEventDashboard()

# Initialize flag to track initialization
dashboard_initialized = False

@router.get("/", response_class=HTMLResponse)
async def get_llm_event_dashboard(request: Request, symbol: str = "BTC/USDT"):
    """
    Render the LLM and event detection dashboard.
    
    Args:
        request: The FastAPI request
        symbol: The trading pair symbol (default: "BTC/USDT")
        
    Returns:
        HTML response with the rendered dashboard
    """
    global dashboard_initialized
    
    try:
        # Initialize the dashboard if needed
        if not dashboard_initialized:
            llm_event_dashboard.initialize()
            dashboard_initialized = True
        
        # Get dashboard data
        dashboard_data = await llm_event_dashboard.get_dashboard_data(symbol)
        
        # Add request to template context
        dashboard_data["request"] = request
        dashboard_data["symbol"] = symbol
        
        # Render the template
        return templates.TemplateResponse(
            "llm_event_dashboard.html", 
            dashboard_data
        )
        
    except Exception as e:
        logger.error(f"Error rendering LLM event dashboard: {e}")
        return HTMLResponse(f"<h1>Error</h1><p>Failed to load LLM event dashboard: {str(e)}</p>")

@router.get("/api/data")
async def get_llm_event_data(symbol: str = "BTC/USDT"):
    """
    Get LLM and event dashboard data as JSON for API clients.
    
    Args:
        symbol: The trading pair symbol (default: "BTC/USDT")
        
    Returns:
        JSON response with dashboard data
    """
    global dashboard_initialized
    
    try:
        # Initialize the dashboard if needed
        if not dashboard_initialized:
            llm_event_dashboard.initialize()
            dashboard_initialized = True
        
        # Get dashboard data
        dashboard_data = await llm_event_dashboard.get_dashboard_data(symbol)
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting LLM event data: {e}")
        return {"error": str(e)}