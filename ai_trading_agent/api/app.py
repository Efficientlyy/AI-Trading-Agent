# Re-export the main FastAPI application
from .main import app

# This file ensures that both 'ai_trading_agent.api.app:app' and 
# 'ai_trading_agent.api.main:app' both work as valid import paths

from .data_feed_manager import router as data_feed_router
from .sentiment_api import router as sentiment_router
from .paper_trading_api import router as paper_trading_router
from .sentiment_pipeline_api import router as sentiment_pipeline_router
from .technical_analysis_api import router as technical_analysis_router
from .agent_visualization_api import router as agent_visualization_router
from .websocket_api import router as websocket_router
from .mexc_websocket import router as mexc_websocket_router