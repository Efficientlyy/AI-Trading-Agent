"""
LLM Oversight Service.

This module implements the REST API service for the LLM Oversight system,
allowing integration with the AI Trading Agent production system.
"""

import os
import logging
import json
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, multiprocess

from ai_trading_agent.oversight.llm_oversight import (
    LLMOversight, LLMProvider, OversightLevel
)
from ai_trading_agent.oversight.config import (
    OversightServiceConfig, validate_config
)
from ai_trading_agent.oversight.metrics import metrics
from ai_trading_agent.oversight.api_routes import router as dashboard_router

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/oversight_service.log")
    ]
)
logger = logging.getLogger("oversight_service")

# Initialize FastAPI app
app = FastAPI(
    title="LLM Oversight Service",
    description="API service for LLM-powered trading oversight",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Set up multiprocess metrics collection for Prometheus
registry = CollectorRegistry()
if 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
    multiprocess.MultiProcessCollector(registry)

# Include dashboard API routes
app.include_router(dashboard_router)

# Load oversight configuration
config = validate_config()

# Models for API requests and responses
class MarketData(BaseModel):
    """Market data input model."""
    data: Dict[str, Any] = Field(..., description="Market data to analyze")
    
class TradingDecision(BaseModel):
    """Trading decision input model."""
    decision: Dict[str, Any] = Field(..., description="Trading decision to validate") 
    context: Dict[str, Any] = Field(..., description="Context for decision validation")

class AnomalyDetectionInput(BaseModel):
    """Anomaly detection input model."""
    data: Dict[str, Any] = Field(..., description="Data to analyze for anomalies")
    thresholds: Dict[str, Any] = Field(..., description="Threshold parameters")
    
class MarketEventInput(BaseModel):
    """Market event explanation input model."""
    event: Dict[str, Any] = Field(..., description="Market event to explain")
    context: Dict[str, Any] = Field(..., description="Additional context")
    
class StrategyAdjustmentInput(BaseModel):
    """Strategy adjustment input model."""
    current_strategy: Dict[str, Any] = Field(..., description="Current trading strategy")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    market_conditions: Dict[str, Any] = Field(..., description="Market conditions")
    
class OversightResponse(BaseModel):
    """Standard oversight response model."""
    success: bool = Field(..., description="Whether the operation was successful")
    result: Dict[str, Any] = Field(..., description="Result of the oversight operation")
    processing_time: float = Field(..., description="Processing time in seconds")
    oversight_level: str = Field(..., description="Level of oversight applied")
    timestamp: str = Field(..., description="Timestamp of the response")

# Dependency to get LLM Oversight instance
def get_oversight() -> LLMOversight:
    """Get a configured LLM Oversight instance."""
    provider = config.get_llm_provider()
    oversight_level = config.get_oversight_level()
    llm_config = config.get_llm_config()
    
    try:
        return LLMOversight(
            provider=provider,
            model_name=llm_config.get("model", "gpt-4"),
            oversight_level=oversight_level,
            api_key=llm_config.get("api_key", os.environ.get("TRADING_LLM_API_KEY")),
            api_base=llm_config.get("api_base"),
            max_tokens=llm_config.get("max_tokens", 1000),
            temperature=llm_config.get("temperature", 0.2),
            max_history_items=llm_config.get("max_history_items", 10),
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM Oversight: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM Oversight: {str(e)}")

# Background task to log oversight decisions
def log_oversight_decision(decision_type: str, input_data: Dict[str, Any], result: Dict[str, Any]):
    """Log oversight decisions to file."""
    try:
        log_dir = os.environ.get("DECISION_LOG_PATH", "logs/oversight_decisions")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{log_dir}/{decision_type}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "type": decision_type,
                "timestamp": timestamp,
                "input": input_data,
                "result": result
            }, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to log oversight decision: {e}")

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint."""
    metrics.record_request("health")
    metrics.update_health_status(True)
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
def get_metrics():
    """Expose Prometheus metrics."""
    return metrics.generate_latest()

# API routes
@app.post("/analyze/market", response_model=OversightResponse)
async def analyze_market(
    data: MarketData,
    background_tasks: BackgroundTasks,
    oversight: LLMOversight = Depends(get_oversight)
):
    """Analyze market conditions using LLM oversight."""
    metrics.record_request("analyze/market")
    start_time = time.time()
    try:
        result = oversight.analyze_market_conditions(data.data)
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "result": result,
            "processing_time": processing_time,
            "oversight_level": oversight.oversight_level.value,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        background_tasks.add_task(log_oversight_decision, "market_analysis", data.dict(), result)
        metrics.record_market_analysis(regime=result.get("regime", "unknown"), latency=processing_time)
        metrics.record_llm_tokens(provider=oversight.provider.value, model=oversight.model_name, request_type="market_analysis", token_count=oversight.last_token_count)
        return response
    
    except Exception as e:
        metrics.record_failed_request("analyze/market", error_type=type(e).__name__)
        logger.error(f"Market analysis failed: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Market analysis failed: {str(e)}")

@app.post("/validate/decision", response_model=OversightResponse)
async def validate_decision(
    input_data: TradingDecision,
    background_tasks: BackgroundTasks,
    oversight: LLMOversight = Depends(get_oversight)
):
    """Validate trading decision using LLM oversight."""
    metrics.record_request("validate/decision")
    start_time = time.time()
    try:
        result = oversight.validate_trading_decision(input_data.decision, input_data.context)
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "result": result,
            "processing_time": processing_time,
            "oversight_level": oversight.oversight_level.value,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        background_tasks.add_task(log_oversight_decision, "decision_validation", input_data.dict(), result)
        metrics.record_validation_decision(action=result.get("action", "unknown"), latency=processing_time, confidence=result.get("confidence", 0.0))
        metrics.record_llm_tokens(provider=oversight.provider.value, model=oversight.model_name, request_type="validation", token_count=oversight.last_token_count)
        return response
    
    except Exception as e:
        metrics.record_failed_request("validate/decision", error_type=type(e).__name__)
        logger.error(f"Decision validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Decision validation failed: {str(e)}")

@app.post("/detect/anomalies", response_model=OversightResponse)
async def detect_anomalies(
    input_data: AnomalyDetectionInput,
    background_tasks: BackgroundTasks,
    oversight: LLMOversight = Depends(get_oversight)
):
    """Detect anomalies in data using LLM oversight."""
    metrics.record_request("detect/anomalies")
    start_time = time.time()
    try:
        result = oversight.detect_anomalies(input_data.data, input_data.thresholds)
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "result": result,
            "processing_time": processing_time,
            "oversight_level": oversight.oversight_level.value,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        background_tasks.add_task(log_oversight_decision, "anomaly_detection", input_data.dict(), result)
        metrics.record_llm_tokens(provider=oversight.provider.value, model=oversight.model_name, request_type="anomaly_detection", token_count=oversight.last_token_count)
        return response
    
    except Exception as e:
        metrics.record_failed_request("detect/anomalies", error_type=type(e).__name__)
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@app.post("/explain/event", response_model=OversightResponse)
async def explain_event(
    input_data: MarketEventInput,
    background_tasks: BackgroundTasks,
    oversight: LLMOversight = Depends(get_oversight)
):
    """Explain market event using LLM oversight."""
    metrics.record_request("explain/event")
    start_time = time.time()
    try:
        result = oversight.explain_market_event(input_data.event, input_data.context)
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "result": result,
            "processing_time": processing_time,
            "oversight_level": oversight.oversight_level.value,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        background_tasks.add_task(log_oversight_decision, "event_explanation", input_data.dict(), result)
        metrics.record_llm_tokens(provider=oversight.provider.value, model=oversight.model_name, request_type="event_explanation", token_count=oversight.last_token_count)
        return response
    
    except Exception as e:
        metrics.record_failed_request("explain/event", error_type=type(e).__name__)
        logger.error(f"Event explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Event explanation failed: {str(e)}")

@app.post("/suggest/strategy-adjustments", response_model=OversightResponse)
async def suggest_strategy_adjustments(
    input_data: StrategyAdjustmentInput,
    background_tasks: BackgroundTasks,
    oversight: LLMOversight = Depends(get_oversight)
):
    """Suggest strategy adjustments using LLM oversight."""
    metrics.record_request("suggest/strategy-adjustments")
    start_time = time.time()
    try:
        result = oversight.suggest_strategy_adjustments(
            input_data.current_strategy,
            input_data.performance_metrics,
            input_data.market_conditions
        )
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "result": result,
            "processing_time": processing_time,
            "oversight_level": oversight.oversight_level.value,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        background_tasks.add_task(log_oversight_decision, "strategy_adjustments", input_data.dict(), result)
        metrics.record_llm_tokens(provider=oversight.provider.value, model=oversight.model_name, request_type="strategy_adjustment", token_count=oversight.last_token_count)
        return response
    
    except Exception as e:
        metrics.record_failed_request("suggest/strategy-adjustments", error_type=type(e).__name__)
        logger.error(f"Strategy adjustment suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Strategy adjustment suggestion failed: {str(e)}")

@app.get("/config")
async def get_oversight_config():
    """Get the current oversight configuration."""
    return {
        "llm_config": config.get_llm_config(),
        "oversight_config": config.get_oversight_config(),
        "oversight_level": config.get_oversight_level().value,
        "llm_provider": config.get_llm_provider().value
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting LLM Oversight Service on {host}:{port}")
    uvicorn.run("ai_trading_agent.oversight.service:app", host=host, port=port, log_level="info")
