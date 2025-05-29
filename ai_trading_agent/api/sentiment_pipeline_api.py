"""
API for detailed sentiment pipeline visualization and metrics.
Exposes the inner workings of the SentimentAnalysisAgent for visualization.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ai_trading_agent.agent.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

class PipelineStage(BaseModel):
    name: str
    status: str
    metrics: Dict
    last_updated: str
    description: str

class PipelineData(BaseModel):
    agent_id: str
    pipeline_status: str
    stages: List[PipelineStage]
    global_metrics: Dict
    pipeline_updated: str
    pipeline_latency: float = 0  # in milliseconds

@router.get("/pipeline/{agent_id}", response_model=PipelineData)
async def get_sentiment_pipeline(agent_id: str):
    """
    Get detailed sentiment pipeline data for a specific sentiment agent.
    """
    try:
        # Get the agent from the registry
        agent = AgentRegistry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Verify this is a sentiment analysis agent
        if not hasattr(agent, 'sentiment_analyzer') or agent.agent_role != "specialized_sentiment":
            raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not a sentiment analysis agent")
        
        # Construct the pipeline data
        now = datetime.now()
        
        # Extract the actual pipeline components and their metrics
        alpha_vantage_client = {}
        sentiment_processor = {}
        signal_generator = {}
        cache_manager = {}
        
        # If available, extract from the agent's internal components
        if hasattr(agent, 'sentiment_analyzer'):
            analyzer = agent.sentiment_analyzer
            
            # Extract Alpha Vantage client metrics if available
            if hasattr(analyzer, 'alpha_vantage_client'):
                av_client = analyzer.alpha_vantage_client
                alpha_vantage_client = {
                    "status": "online" if av_client else "offline",
                    "api_calls": agent.metrics.get("api_calls", 0),
                    "cache_hits": agent.metrics.get("cache_hits", 0),
                    "errors": agent.metrics.get("data_fetch_errors", 0),
                    "last_fetch": getattr(av_client, 'last_call_time', now).isoformat() if av_client else None,
                    "rate_limit_remains": getattr(av_client, 'rate_limit_remains', None)
                }
            
            # Extract sentiment processor metrics
            sentiment_processor = {
                "status": "online" if hasattr(analyzer, 'process_text') else "offline",
                "processedItems": (agent.metrics.get("bullish_signals", 0) + 
                                  agent.metrics.get("bearish_signals", 0) + 
                                  agent.metrics.get("neutral_signals", 0)),
                "avgProcessingTime": agent.metrics.get("avg_processing_time", 0),
                "errors": agent.metrics.get("analysis_errors", 0)
            }
            
            # Extract signal generator metrics
            signal_generator = {
                "status": "online",
                "signalsGenerated": agent.metrics.get("total_signals_generated", 0),
                "bullishSignals": agent.metrics.get("bullish_signals", 0),
                "bearishSignals": agent.metrics.get("bearish_signals", 0),
                "neutralSignals": agent.metrics.get("neutral_signals", 0),
                "avgConfidence": agent.metrics.get("avg_confidence", 0.5)
            }
            
            # Extract cache manager metrics
            if hasattr(agent, 'cache'):
                cache_size = 0
                cache_hits = agent.metrics.get("cache_hits", 0)
                cache_misses = agent.metrics.get("cache_misses", 0)
                
                if agent.cache.get("sentiment_data"):
                    cache_size += len(agent.cache["sentiment_data"])
                if agent.cache.get("signals"):
                    cache_size += len(agent.cache["signals"])
                
                total_cache_requests = cache_hits + cache_misses
                hit_ratio = cache_hits / total_cache_requests if total_cache_requests > 0 else 0
                
                cache_manager = {
                    "status": "online",
                    "cacheSize": cache_size,
                    "hitRatio": hit_ratio,
                    "ttl": agent.cache_ttl if hasattr(agent, 'cache_ttl') else 3600
                }
        
        # Package the pipeline stages
        stages = [
            PipelineStage(
                name="Alpha Vantage Client",
                status=alpha_vantage_client.get("status", "unknown"),
                metrics=alpha_vantage_client,
                last_updated=alpha_vantage_client.get("last_fetch", now.isoformat()),
                description="Fetches news sentiment data from Alpha Vantage API"
            ),
            PipelineStage(
                name="Sentiment Processor",
                status=sentiment_processor.get("status", "unknown"),
                metrics=sentiment_processor,
                last_updated=now.isoformat(),
                description="Processes raw sentiment data into analyzable format"
            ),
            PipelineStage(
                name="Signal Generator",
                status=signal_generator.get("status", "unknown"),
                metrics=signal_generator,
                last_updated=now.isoformat(),
                description="Generates trading signals from processed sentiment"
            ),
            PipelineStage(
                name="Cache Manager",
                status=cache_manager.get("status", "unknown"),
                metrics=cache_manager,
                last_updated=now.isoformat(),
                description="Manages caching of sentiment data and signals"
            )
        ]
        
        # Calculate overall pipeline status
        pipeline_status = "running" if agent.status == "running" else agent.status
        
        # Calculate pipeline latency (average processing time in milliseconds)
        pipeline_latency = agent.metrics.get("avg_processing_time", 0) * 1000
        
        # Compile the global metrics
        global_metrics = {
            "total_signals": agent.metrics.get("total_signals_generated", 0),
            "avg_sentiment_score": agent.metrics.get("avg_sentiment_score", 0),
            "success_rate": 0,  # Calculate if available
            "error_rate": 0,    # Calculate if available
        }
        
        # Calculate success rate if metrics available
        successful_trades = agent.metrics.get("successful_trades", 0)
        failed_trades = agent.metrics.get("failed_trades", 0)
        total_trades = successful_trades + failed_trades
        if total_trades > 0:
            global_metrics["success_rate"] = successful_trades / total_trades
            global_metrics["error_rate"] = failed_trades / total_trades
        
        return PipelineData(
            agent_id=agent_id,
            pipeline_status=pipeline_status,
            stages=stages,
            global_metrics=global_metrics,
            pipeline_updated=now.isoformat(),
            pipeline_latency=pipeline_latency
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving sentiment pipeline data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving pipeline data: {str(e)}")

@router.get("/symbols/{agent_id}")
async def get_sentiment_symbols(agent_id: str):
    """
    Get all symbols that the sentiment agent is monitoring.
    """
    try:
        # Get the agent from the registry
        agent = AgentRegistry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Verify this is a sentiment analysis agent
        if not hasattr(agent, 'sentiment_analyzer'):
            raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not a sentiment analysis agent")
            
        return {
            "agent_id": agent_id,
            "symbols": agent.symbols,
            "topic_mappings": getattr(agent, 'config_details', {}).get("custom_topics", {})
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving sentiment symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving symbols: {str(e)}")

@router.get("/signals/{agent_id}")
async def get_latest_signals(agent_id: str, limit: int = Query(5, ge=1, le=50)):
    """
    Get the latest sentiment signals generated by the agent.
    """
    try:
        # Get the agent from the registry
        agent = AgentRegistry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Verify this is a sentiment analysis agent
        if agent.agent_role != "specialized_sentiment":
            raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not a sentiment analysis agent")
        
        # Extract the latest signals from cache
        latest_signals = []
        if hasattr(agent, 'cache') and 'signals' in agent.cache:
            for symbol, signal_data in agent.cache['signals'].items():
                if signal_data and 'signal' in signal_data:
                    signal = signal_data['signal']
                    signal['cached_at'] = signal_data.get('timestamp', datetime.now()).isoformat()
                    latest_signals.append(signal)
        
        # Sort by timestamp (newest first) and limit
        latest_signals = sorted(latest_signals, key=lambda x: x.get('cached_at', ''), reverse=True)[:limit]
        
        return {
            "agent_id": agent_id,
            "signals_count": len(latest_signals),
            "signals": latest_signals
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving sentiment signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving signals: {str(e)}")
