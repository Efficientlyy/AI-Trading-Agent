"""
Technical Analysis API Module

This module provides API endpoints for technical analysis features,
respecting the mock/real data toggle configuration.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
import logging

# Import performance optimization tools
from ..utils.performance_optimization import (
    cache_result, 
    clear_cache, 
    optimize_dataframe_memory,
    memory_usage_report
)

from ..config.data_source_config import get_data_source_config
from ..data.data_source_factory import get_data_source_factory
from ..agent.advanced_technical_analysis import AdvancedTechnicalAnalysisAgent
from ..patterns.three_candle_patterns import ThreeCandlePatternDetector
from ..patterns.advanced_patterns import AdvancedPatternDetector
from ..models.response_models import IndicatorResponse, PatternResponse, TechnicalAnalysisResponse, PerformanceMetricsResponse, HealthStatusResponse
from ..orchestration.ta_agent_integration import TechnicalAgentOrchestrator
from ..monitoring.ta_agent_monitor import TAAgentMonitor, setup_production_monitoring

# Create router
router = APIRouter(
    prefix="/api/technical-analysis",
    tags=["technical-analysis"],
)

# Define response models
class IndicatorResponse(BaseModel):
    """Response model for technical indicators."""
    indicator_name: str
    values: Dict[str, Any]
    metadata: Dict[str, Any]

class PatternResponse(BaseModel):
    """Response model for chart patterns."""
    pattern: str
    position: int
    direction: str
    confidence: float
    candles: List[int]
    metadata: Dict[str, Any]

class TechnicalAnalysisResponse(BaseModel):
    """Response model for technical analysis."""
    symbol: str
    timeframe: str
    data_source: str
    timestamp: datetime
    indicators: List[IndicatorResponse]
    patterns: List[PatternResponse]

class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    api_calls: int
    cache_hits: int
    cache_misses: int
    avg_processing_time_ms: float
    last_cache_clear: datetime
    memory_usage_mb: float

class HealthStatusResponse(BaseModel):
    """Response model for health status."""
    status: str
    message: str
    timestamp: datetime

# Create global logger
logger = logging.getLogger("TechnicalAnalysisAPI")

# Performance metrics tracking
_perf_metrics = {
    "api_calls": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "processing_times": [],
    "last_cache_clear": datetime.now()
}

# Store reference to the TA agent singleton
_ta_agent = None
_pattern_detector = None
_advanced_pattern_detector = None
_orchestrator = None
_monitor = None

# Dependency for getting the TA agent instance
def get_ta_agent() -> AdvancedTechnicalAnalysisAgent:
    """
    Get or create the Technical Analysis Agent singleton.
    
    Returns:
        AdvancedTechnicalAnalysisAgent instance
    """
    global _ta_agent
    if _ta_agent is None:
        # Create with default configuration
        _ta_agent = AdvancedTechnicalAnalysisAgent()
        logger.info("Created Technical Analysis Agent for API access")
    return _ta_agent

# Dependency for getting the pattern detector instance
def get_pattern_detector() -> ThreeCandlePatternDetector:
    """
    Get or create the Pattern Detector singleton.
    
    Returns:
        ThreeCandlePatternDetector instance
    """
    global _pattern_detector
    if _pattern_detector is None:
        # Create with default configuration
        _pattern_detector = ThreeCandlePatternDetector({
            "confidence_threshold": 0.6
        })
        logger.info("Created Pattern Detector for API access")
    return _pattern_detector

def get_advanced_pattern_detector() -> AdvancedPatternDetector:
    """
    Get or create the Advanced Pattern Detector singleton.
    
    Returns:
        AdvancedPatternDetector instance
    """
    global _advanced_pattern_detector
    if _advanced_pattern_detector is None:
        # Create with default configuration
        _advanced_pattern_detector = AdvancedPatternDetector({
            "confidence_threshold": 0.65
        })
        logger.info("Created Advanced Pattern Detector for API access")
    return _advanced_pattern_detector

def get_orchestrator() -> TechnicalAgentOrchestrator:
    """
    Get or create the Technical Agent Orchestrator singleton.
    
    Returns:
        TechnicalAgentOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        # Create with default configuration
        _orchestrator = TechnicalAgentOrchestrator()
        logger.info("Created Technical Agent Orchestrator for API access")
    return _orchestrator

def get_monitor() -> TAAgentMonitor:
    """
    Get or create the Technical Analysis Agent Monitor singleton.
    
    Returns:
        TAAgentMonitor instance
    """
    global _monitor
    if _monitor is None:
        # Create with default configuration
        _monitor = setup_production_monitoring(get_ta_agent())
        logger.info("Created Technical Analysis Agent Monitor for API access")
    return _monitor

class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    api_calls: int
    cache_hits: int
    cache_misses: int
    avg_processing_time_ms: float
    last_cache_clear: datetime
    memory_usage_mb: float

@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics() -> PerformanceMetricsResponse:
    """Get performance metrics for the Technical Analysis API."""
    import sys
    
    # Calculate average processing time
    avg_time = 0
    if _perf_metrics["processing_times"]:
        avg_time = sum(_perf_metrics["processing_times"]) / len(_perf_metrics["processing_times"])
    
    # Get memory usage
    memory_usage = sys.getsizeof(_perf_metrics) / (1024 * 1024)  # Convert to MB
    
    return PerformanceMetricsResponse(
        api_calls=_perf_metrics["api_calls"],
        cache_hits=_perf_metrics["cache_hits"],
        cache_misses=_perf_metrics["cache_misses"],
        avg_processing_time_ms=avg_time,
        last_cache_clear=_perf_metrics["last_cache_clear"],
        memory_usage_mb=memory_usage
    )

@router.post("/clear-cache")
async def clear_api_cache(cache_type: Optional[str] = None) -> Dict[str, Any]:
    """Clear the API cache."""
    clear_cache(cache_type)
    _perf_metrics["last_cache_clear"] = datetime.now()
    
    return {
        "status": "success",
        "message": f"Cache cleared: {'all' if cache_type is None else cache_type}",
        "timestamp": datetime.now()
    }

@router.get("/indicators", response_model=List[IndicatorResponse])
@cache_result(cache_type="indicators", key_fields=["symbol", "timeframe"])
async def get_indicators(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe (e.g., 1h, 4h, 1d)"),
    indicators: List[str] = Query(None, description="List of indicators to calculate"),
    background_tasks: BackgroundTasks = None,
    agent: AdvancedTechnicalAnalysisAgent = Depends(get_ta_agent)
) -> List[IndicatorResponse]:
    """
    Get technical indicators for a symbol and timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe identifier
        indicators: List of indicators to calculate
        
    Returns:
        List of IndicatorResponse with indicator values
    """
    try:
        # Get data provider based on current configuration
        data_source_factory = get_data_source_factory()
        data_provider = data_source_factory.get_data_provider()
        
        # Get data source configuration
        config = get_data_source_config()
        is_mock = config.use_mock_data
        
        # Log request
        logger.info(f"Indicators request: {symbol}, {timeframe}, {indicators}, using {'mock' if is_mock else 'real'} data")
        
        # Get market data
        try:
            if is_mock:
                # Generate mock data
                market_data = data_provider.generate_data(
                    symbols=[symbol],
                    timeframes=[timeframe],
                    periods=100
                )
            else:
                # Get real market data
                market_data = data_provider.get_historical_data(
                    symbols=[symbol],
                    timeframes=[timeframe],
                    periods=100
                )
        except Exception as e:
            logger.error(f"Failed to get market data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch market data: {str(e)}"
            )
        
        # Use the agent to calculate indicators
        # This is a simplified version - in a real implementation,
        # you would use the agent's indicator engine directly
        results = []
        
        # Default indicators if none specified
        if not indicators:
            indicators = ["rsi", "macd", "bollinger_bands"]
        
        # Mock indicator calculations for demo
        for indicator in indicators:
            if indicator == "rsi":
                results.append(IndicatorResponse(
                    indicator_name="rsi",
                    values={
                        "rsi": [50 + (i % 30) for i in range(10)]  # Simple mock values
                    },
                    metadata={
                        "period": 14,
                        "overbought": 70,
                        "oversold": 30
                    }
                ))
            elif indicator == "macd":
                results.append(IndicatorResponse(
                    indicator_name="macd",
                    values={
                        "macd": [0.5 + (i % 10) * 0.1 for i in range(10)],
                        "signal": [0.7 + (i % 8) * 0.1 for i in range(10)],
                        "histogram": [0.2 + (i % 6) * 0.05 for i in range(10)]
                    },
                    metadata={
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9
                    }
                ))
            elif indicator == "bollinger_bands":
                results.append(IndicatorResponse(
                    indicator_name="bollinger_bands",
                    values={
                        "upper": [105 + i for i in range(10)],
                        "middle": [100 + i for i in range(10)],
                        "lower": [95 + i for i in range(10)]
                    },
                    metadata={
                        "period": 20,
                        "deviations": 2.0
                    }
                ))
        
        return results
            
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate indicators: {str(e)}"
        )

@router.get("/patterns", response_model=List[PatternResponse])
@cache_result(cache_type="patterns", key_fields=["symbol", "timeframe"])
async def get_patterns(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe (e.g., 1h, 4h, 1d)"),
    background_tasks: BackgroundTasks = None,
    pattern_detector: ThreeCandlePatternDetector = Depends(get_pattern_detector)
) -> List[PatternResponse]:
    """
    Get detected chart patterns for a symbol and timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe identifier
        
    Returns:
        List of PatternResponse with detected patterns
    """
    try:
        # Get data provider based on current configuration
        data_source_factory = get_data_source_factory()
        data_provider = data_source_factory.get_data_provider()
        
        # Get data source configuration
        config = get_data_source_config()
        is_mock = config.use_mock_data
        
        # Log request
        logger.info(f"Patterns request: {symbol}, {timeframe}, using {'mock' if is_mock else 'real'} data")
        
        # Get market data
        try:
            if is_mock:
                # Generate mock data
                market_data = data_provider.generate_data(
                    symbols=[symbol],
                    timeframes=[timeframe],
                    periods=100
                )
            else:
                # Get real market data
                market_data = data_provider.get_historical_data(
                    symbols=[symbol],
                    timeframes=[timeframe],
                    periods=100
                )
        except Exception as e:
            logger.error(f"Failed to get market data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch market data: {str(e)}"
            )
        
        # For demo purposes, return mock patterns
        # In a real implementation, you would use the pattern detector to find patterns
        # in the market data
        
        # Randomly generate some patterns based on symbol and timeframe
        import random
        random.seed(hash(f"{symbol}{timeframe}"))
        
        patterns = []
        pattern_types = [
            "morning_star",
            "evening_star",
            "three_white_soldiers",
            "three_black_crows",
            "three_inside_up",
            "three_inside_down",
            "three_outside_up",
            "three_outside_down",
            "abandoned_baby_bullish",
            "abandoned_baby_bearish"
        ]
        
        # Generate 0-3 patterns
        num_patterns = random.randint(0, 3)
        used_patterns = set()
        
        for _ in range(num_patterns):
            # Select a pattern type not used yet
            available_patterns = [p for p in pattern_types if p not in used_patterns]
            if not available_patterns:
                break
                
            pattern_type = random.choice(available_patterns)
            used_patterns.add(pattern_type)
            
            # Determine if bullish or bearish
            is_bullish = (
                pattern_type == "morning_star" or
                pattern_type == "three_white_soldiers" or
                pattern_type == "three_inside_up" or
                pattern_type == "three_outside_up" or
                pattern_type == "abandoned_baby_bullish"
            )
            
            # Create pattern response
            pattern = PatternResponse(
                pattern=pattern_type,
                position=random.randint(80, 95),
                direction="bullish" if is_bullish else "bearish",
                confidence=round(random.uniform(0.65, 0.95), 2),
                candles=[i for i in range(3)],
                metadata={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "detected_at": datetime.now()
                }
            )
            
            patterns.append(pattern)
        
        return patterns
            
    except Exception as e:
        logger.error(f"Error detecting patterns: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to detect patterns: {str(e)}"
        )

@router.get("/analysis", response_model=TechnicalAnalysisResponse)
@cache_result(cache_type="analysis", key_fields=["symbol", "timeframe"])
async def get_technical_analysis(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe (e.g., 1h, 4h, 1d)"),
    indicators: List[str] = Query(None, description="List of indicators to calculate"),
    include_advanced_patterns: bool = Query(False, description="Include advanced pattern detection"),
    background_tasks: BackgroundTasks = None,
    agent: AdvancedTechnicalAnalysisAgent = Depends(get_ta_agent),
    pattern_detector: ThreeCandlePatternDetector = Depends(get_pattern_detector),
    advanced_pattern_detector: AdvancedPatternDetector = Depends(get_advanced_pattern_detector)
) -> TechnicalAnalysisResponse:
    """
    Get comprehensive technical analysis for a symbol and timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe identifier
        indicators: List of indicators to calculate
        
    Returns:
        TechnicalAnalysisResponse with indicators and patterns
    """
    try:
        # Get indicator results
        indicator_results = await get_indicators(symbol, timeframe, indicators, agent)
        
        # Get pattern results
        pattern_results = await get_patterns(symbol, timeframe, pattern_detector)
        
        # Add advanced patterns if requested
        if include_advanced_patterns:
            advanced_patterns = await get_advanced_patterns(symbol, timeframe, advanced_pattern_detector)
            pattern_results.extend(advanced_patterns)
        
        # Get data source type
        data_source = "mock" if get_data_source_config().use_mock_data else "real"
        
        # Create combined response
        response = TechnicalAnalysisResponse(
            symbol=symbol,
            timeframe=timeframe,
            data_source=data_source,
            timestamp=datetime.now(),
            indicators=indicator_results,
            patterns=pattern_results
        )
        
        return response
            
    except Exception as e:
        logger.error(f"Error performing technical analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform technical analysis: {str(e)}"
        )

@router.get("/advanced-patterns", response_model=List[PatternResponse])
@cache_result(cache_type="advanced_patterns", key_fields=["symbol", "timeframe"])
async def get_advanced_patterns(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query(..., description="Timeframe (e.g., 1h, 4h, 1d)"),
    pattern_detector: AdvancedPatternDetector = Depends(get_advanced_pattern_detector)
) -> List[PatternResponse]:
    """
    Get advanced chart patterns for a symbol and timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe identifier
        
    Returns:
        List of PatternResponse with detected advanced patterns
    """
    try:
        # Track API call start time
        start_time = time.time()
        _perf_metrics["api_calls"] += 1
        
        # Get data based on current data source setting
        data_source_factory = get_data_source_factory()
        data_provider = data_source_factory.get_data_provider()
        
        # Get market data from appropriate source
        if get_data_source_config().use_mock_data:
            market_data = data_provider.generate_data(symbols=[symbol], timeframes=[timeframe], periods=100)
        else:
            market_data = data_provider.get_historical_data(symbols=[symbol], timeframes=[timeframe], periods=100)
        
        # Extract dataframe for the specific symbol and timeframe
        if symbol in market_data and timeframe in market_data[symbol]:
            df = market_data[symbol][timeframe]
            
            # Optimize memory usage of dataframe
            df = optimize_dataframe_memory(df)
            
            # Detect advanced patterns
            detected_patterns = pattern_detector.detect_patterns(df)
            
            # Convert to response format
            patterns = []
            for p in detected_patterns:
                pattern = PatternResponse(
                    pattern=p["pattern"],
                    position=p["position"],
                    direction=p["direction"],
                    confidence=p["confidence"],
                    candles=p["candles"],
                    metadata={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "detected_at": datetime.now(),
                        **p.get("metadata", {})
                    }
                )
                patterns.append(pattern)
            
            # Track processing time
            elapsed_ms = (time.time() - start_time) * 1000
            _perf_metrics["processing_times"].append(elapsed_ms)
            
            return patterns
        else:
            logger.warning(f"No data found for {symbol} on {timeframe}")
            return []
            
    except Exception as e:
        logger.error(f"Error detecting advanced patterns: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to detect advanced patterns: {str(e)}"
        )

@router.get("/monitoring/health", response_model=HealthStatusResponse)
async def get_health_status(
    monitor: TAAgentMonitor = Depends(get_monitor)
) -> HealthStatusResponse:
    """
    Get the health status of the Technical Analysis Agent.
    
    Returns:
        Health status information
    """
    try:
        # Get health summary from monitor
        health_summary = monitor.get_health_summary()
        
        # Create response
        response = HealthStatusResponse(
            status=health_summary["overall_status"],
            message=f"System status: {health_summary['overall_status']} with {health_summary['alert_count']} active alerts",
            timestamp=datetime.now()
        )
        
        return response
    except Exception as e:
        logger.error(f"Error getting health status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get health status: {str(e)}"
        )

@router.get("/monitoring/metrics", response_model=Dict[str, Any])
async def get_monitoring_metrics(
    monitor: TAAgentMonitor = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    Get detailed monitoring metrics for the Technical Analysis Agent.
    
    Returns:
        Detailed monitoring metrics
    """
    try:
        # Get detailed metrics from monitor
        monitoring_status = monitor.get_monitoring_status()
        
        return monitoring_status
    except Exception as e:
        logger.error(f"Error getting monitoring metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get monitoring metrics: {str(e)}"
        )

@router.post("/orchestrator/start")
async def start_orchestration(
    symbols: List[str] = Query(..., description="Symbols to monitor"),
    timeframes: List[str] = Query(..., description="Timeframes to monitor"),
    orchestrator: TechnicalAgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Start the Technical Analysis orchestration.
    
    Args:
        symbols: List of symbols to monitor
        timeframes: List of timeframes to monitor
        
    Returns:
        Orchestration status
    """
    try:
        # Start orchestration
        orchestrator.start(symbols, timeframes)
        
        # Return status
        return {
            "status": "started",
            "message": f"Started monitoring {len(symbols)} symbols on {len(timeframes)} timeframes",
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "timeframes": timeframes
        }
    except Exception as e:
        logger.error(f"Error starting orchestration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start orchestration: {str(e)}"
        )

@router.post("/orchestrator/stop")
async def stop_orchestration(
    orchestrator: TechnicalAgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Stop the Technical Analysis orchestration.
    
    Returns:
        Orchestration status
    """
    try:
        # Stop orchestration
        orchestrator.stop()
        
        # Return status
        return {
            "status": "stopped",
            "message": "Stopped Technical Analysis orchestration",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping orchestration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop orchestration: {str(e)}"
        )

@router.get("/orchestrator/status")
async def get_orchestration_status(
    orchestrator: TechnicalAgentOrchestrator = Depends(get_orchestrator)
) -> Dict[str, Any]:
    """
    Get the status of the Technical Analysis orchestration.
    
    Returns:
        Orchestration status
    """
    try:
        # Get status
        status = orchestrator.get_status()
        
        # Return status
        return {
            "status": "running" if status["running"] else "stopped",
            "agent_id": status["agent_id"],
            "last_update": status["last_update"].isoformat() if status["last_update"] else None,
            "symbols_monitored": status["symbols_monitored"],
            "timeframes_monitored": status["timeframes_monitored"],
            "data_source": status["data_source"],
            "signal_count": status["signal_count"],
            "error_count": status["error_count"],
            "queue_sizes": status["queue_sizes"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting orchestration status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get orchestration status: {str(e)}"
        )
