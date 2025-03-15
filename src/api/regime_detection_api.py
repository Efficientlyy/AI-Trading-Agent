"""API service for market regime detection."""

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import logging
import uuid
import traceback
from pathlib import Path

# Import project components
import sys
sys.path.append("..")
try:
    from src.ml.analysis.market_regime_analyzer import MarketRegimeAnalyzer, AnalysisConfig
    from src.ml.validation.backtest_engine import BacktestEngine, BacktestConfig
    from src.ml.visualization.regime_visualizer import RegimeVisualizer
except ImportError:
    # Alternative import path for when running from within the src directory
    try:
        from ml.analysis.market_regime_analyzer import MarketRegimeAnalyzer, AnalysisConfig
        from ml.validation.backtest_engine import BacktestEngine, BacktestConfig
        from ml.visualization.regime_visualizer import RegimeVisualizer
    except ImportError:
        logging.error("Could not import required modules. Check your Python path.")
        raise

# Initialize FastAPI app
app = FastAPI(
    title="Market Regime Detection API",
    description="API for detecting and analyzing market regimes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_logs.log")
    ]
)

logger = logging.getLogger(__name__)

# Create data models
class DataPoint(BaseModel):
    date: datetime
    price: float
    volume: Optional[float] = None
    return_value: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    spread: Optional[float] = None

class MarketData(BaseModel):
    symbol: str
    data: List[DataPoint]
    
class RegimeRequest(BaseModel):
    market_data: Union[MarketData, List[MarketData]]
    methods: Optional[List[str]] = Field(default_factory=lambda: [
        "volatility", "momentum", "hmm"
    ])
    lookback_window: Optional[int] = 63
    include_statistics: Optional[bool] = True
    include_visualization: Optional[bool] = False

class BacktestRequest(BaseModel):
    market_data: Union[MarketData, List[MarketData]]
    strategy_type: str = "momentum"
    regime_methods: List[str] = Field(default_factory=lambda: [
        "volatility", "momentum"
    ])
    train_test_split: Optional[float] = 0.7
    walk_forward: Optional[bool] = False
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class AnalysisResponse(BaseModel):
    request_id: str
    symbol: str
    regimes: Dict[str, List[int]]
    statistics: Optional[Dict[str, Any]] = None
    visualization_urls: Optional[Dict[str, str]] = None
    execution_time: float

class BacktestResponse(BaseModel):
    request_id: str
    symbol: str
    strategy: str
    performance: Dict[str, float]
    regime_metrics: Dict[str, float]
    equity_curve_url: Optional[str] = None
    execution_time: float

# Global components
analyzer = MarketRegimeAnalyzer()
visualizer = RegimeVisualizer()

# Storage paths
VISUALIZATION_DIR = Path("./visualizations")
VISUALIZATION_DIR.mkdir(exist_ok=True)

# Create static directory if it doesn't exist
STATIC_DIR = Path("./static")
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper functions
def prepare_market_data(market_data: Union[MarketData, List[MarketData]]) -> Dict:
    """Convert API market data to internal format."""
    if isinstance(market_data, list):
        # Multi-asset data
        symbols = [data.symbol for data in market_data]
        dates = []
        returns = {}
        prices = {}
        volumes = {}
        
        for data in market_data:
            data_points = sorted(data.data, key=lambda x: x.date)
            dates = [dp.date for dp in data_points]
            
            returns[data.symbol] = np.array([
                dp.return_value if dp.return_value is not None 
                else None for dp in data_points
            ])
            
            prices[data.symbol] = np.array([dp.price for dp in data_points])
            
            volumes[data.symbol] = np.array([
                dp.volume if dp.volume is not None 
                else 0 for dp in data_points
            ])
            
        # Calculate returns if not provided
        for symbol in symbols:
            if None in returns[symbol]:
                price_array = prices[symbol]
                returns[symbol] = np.diff(np.log(price_array))
                returns[symbol] = np.insert(returns[symbol], 0, 0)
        
        # Create returns matrix for correlation analysis
        returns_matrix = np.column_stack([returns[s] for s in symbols])
        
        prepared_data = {
            'dates': np.array(dates),
            'prices': prices[symbols[0]],  # Primary asset prices
            'returns': returns[symbols[0]],  # Primary asset returns
            'volumes': volumes[symbols[0]],  # Primary asset volumes
            'returns_matrix': returns_matrix
        }
    else:
        # Single-asset data
        data_points = sorted(market_data.data, key=lambda x: x.date)
        
        dates = np.array([dp.date for dp in data_points])
        prices = np.array([dp.price for dp in data_points])
        
        # Extract returns or calculate if not provided
        returns = np.array([
            dp.return_value if dp.return_value is not None 
            else None for dp in data_points
        ])
        
        if None in returns:
            returns = np.diff(np.log(prices))
            returns = np.insert(returns, 0, 0)
        
        volumes = np.array([
            dp.volume if dp.volume is not None 
            else 0 for dp in data_points
        ])
        
        prepared_data = {
            'dates': dates,
            'prices': prices,
            'returns': returns,
            'volumes': volumes
        }
    
    return prepared_data

def save_visualization(fig, request_id: str, method_name: str) -> str:
    """Save visualization to file and return URL."""
    filename = f"{request_id}_{method_name}.png"
    filepath = VISUALIZATION_DIR / filename
    fig.savefig(filepath)
    return f"/visualizations/{filename}"

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/methods")
async def get_available_methods():
    """Get available regime detection methods."""
    return {
        "methods": [
            "volatility",
            "momentum",
            "hmm",
            "trend",
            "mean_reversion",
            "seasonality",
            "market_stress",
            "microstructure",
            "tail_risk",
            "dispersion"
        ],
        "strategies": [
            "momentum",
            "mean_reversion",
            "volatility",
            "regime_based"
        ]
    }

@app.post("/detect-regimes", response_model=AnalysisResponse)
async def detect_regimes(request: RegimeRequest):
    """Detect market regimes based on provided data."""
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Prepare data
        market_data = prepare_market_data(request.market_data)
        symbol = request.market_data.symbol if isinstance(request.market_data, MarketData) else request.market_data[0].symbol
        
        # Configure analysis
        config = AnalysisConfig(
            methods=request.methods,
            lookback_window=request.lookback_window,
            visualize=request.include_visualization
        )
        
        # Run analysis
        results = analyzer.analyze_market(market_data, config=config)
        
        # Prepare response
        response = {
            "request_id": request_id,
            "symbol": symbol,
            "regimes": {method: labels.tolist() for method, labels in results["labels"].items()},
            "execution_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Add statistics if requested
        if request.include_statistics and "statistics" in results:
            response["statistics"] = results["statistics"]
        
        # Add visualization URLs if requested
        if request.include_visualization and "figures" in results:
            visualization_urls = {}
            for method, fig in results["figures"].items():
                url = save_visualization(fig, request_id, method)
                visualization_urls[method] = url
            response["visualization_urls"] = visualization_urls
        
        return response
    
    except Exception as e:
        logger.error(f"Error in regime detection: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in regime detection: {str(e)}"
        )

@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run backtest using regime detection."""
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Prepare data
        market_data = prepare_market_data(request.market_data)
        symbol = request.market_data.symbol if isinstance(request.market_data, MarketData) else request.market_data[0].symbol
        
        # Configure backtest
        config = BacktestConfig(
            train_test_split=request.train_test_split,
            walk_forward=request.walk_forward,
            strategy_type=request.strategy_type,
            regime_methods=request.regime_methods
        )
        
        # Initialize backtest engine
        engine = BacktestEngine(config)
        
        # Run backtest
        result = engine.run_backtest(market_data, f"{symbol}_{request.strategy_type}")
        
        # Save equity curve visualization
        equity_curve_url = None
        if hasattr(result, 'plot_equity_curve'):
            fig = result.plot_equity_curve()
            equity_curve_url = save_visualization(fig, request_id, "equity_curve")
        
        # Prepare response
        response = {
            "request_id": request_id,
            "symbol": symbol,
            "strategy": request.strategy_type,
            "performance": {
                "total_return": result.returns["total_return"],
                "cagr": result.returns["cagr"],
                "sharpe_ratio": result.risk_metrics["sharpe_ratio"],
                "max_drawdown": result.risk_metrics["max_drawdown"],
                "win_rate": result.trade_metrics["win_rate"] if "win_rate" in result.trade_metrics else None
            },
            "regime_metrics": result.regime_metrics,
            "equity_curve_url": equity_curve_url,
            "execution_time": (datetime.now() - start_time).total_seconds()
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in backtest: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 