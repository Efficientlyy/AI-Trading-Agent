"""API routes for the Market Regime Detection API."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from datetime import datetime
import traceback
import logging
from typing import Union, List, Dict, Any
import matplotlib.pyplot as plt

from api.models import RegimeRequest, BacktestRequest, AnalysisResponse, BacktestResponse, MarketData
from api import utils, config

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint with web UI."""
    return FileResponse(str(config.STATIC_DIR / "index.html"))

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/methods")
async def get_available_methods():
    """Get available regime detection methods."""
    return {
        "methods": config.AVAILABLE_METHODS,
        "strategies": config.AVAILABLE_STRATEGIES
    }

@router.post("/detect-regimes", response_model=AnalysisResponse)
async def detect_regimes(request: RegimeRequest):
    """Detect market regimes based on provided data."""
    request_id = utils.generate_request_id()
    start_time = datetime.now()
    
    try:
        # Log request
        utils.log_request("detect-regimes", request)
        
        # Validate market data
        if not utils.validate_market_data(request.market_data):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid market data"
            )
        
        # Prepare data
        market_data = utils.prepare_market_data(request.market_data)
        
        # Get symbol from market data
        symbol = ""
        if isinstance(request.market_data, MarketData):
            symbol = request.market_data.symbol
        else:  # It's a List[MarketData]
            symbol = request.market_data[0].symbol
        
        # Configure analysis
        # In a real implementation, this would call the MarketRegimeAnalyzer
        # For now, we'll simulate the response
        
        # Simulate regime detection (replace with actual implementation)
        import numpy as np
        data_length = len(market_data['returns']) if 'returns' in market_data else len(market_data['prices'])
        simulated_regimes = {}
        
        # Ensure methods is not None and is a list
        methods = request.methods if request.methods is not None else config.DEFAULT_METHODS
        
        for method in methods:
            if method in config.AVAILABLE_METHODS:
                # Generate random labels (0, 1, 2) for demonstration
                np.random.seed(hash(method) % 10000)  # Consistent seed per method
                labels = np.random.randint(0, 3, size=data_length)
                simulated_regimes[method] = labels.tolist()
        
        # Prepare response
        response = {
            "request_id": request_id,
            "symbol": symbol,
            "regimes": simulated_regimes,
            "execution_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Add statistics if requested
        if request.include_statistics:
            statistics = {}
            for method, labels in simulated_regimes.items():
                labels_array = np.array(labels)
                statistics[method] = {
                    "regime_0_count": int(np.sum(labels_array == 0)),
                    "regime_1_count": int(np.sum(labels_array == 1)),
                    "regime_2_count": int(np.sum(labels_array == 2)),
                    "regime_0_percentage": float(np.sum(labels_array == 0) / len(labels_array)),
                    "regime_1_percentage": float(np.sum(labels_array == 1) / len(labels_array)),
                    "regime_2_percentage": float(np.sum(labels_array == 2) / len(labels_array)),
                    "transitions": int(np.sum(np.diff(labels_array) != 0))
                }
            response["statistics"] = statistics
        
        # Log response
        utils.log_response("detect-regimes", response, response["execution_time"])
        
        return response
    
    except Exception as e:
        logger.error(f"Error in regime detection: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in regime detection: {str(e)}"
        )

@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run backtest using regime detection."""
    request_id = utils.generate_request_id()
    start_time = datetime.now()
    
    try:
        # Log request
        utils.log_request("backtest", request)
        
        # Validate market data
        if not utils.validate_market_data(request.market_data):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid market data"
            )
        
        # Prepare data
        market_data = utils.prepare_market_data(request.market_data)
        
        # Get symbol from market data
        symbol = ""
        if isinstance(request.market_data, MarketData):
            symbol = request.market_data.symbol
        else:  # It's a List[MarketData]
            symbol = request.market_data[0].symbol
        
        # Import backtesting components
        from ml.backtesting import RegimeStrategy
        from ml.detection import RegimeDetectorFactory
        
        # Convert to format expected by RegimeStrategy
        data = {
            'symbol': symbol,
            'dates': market_data['dates'],
            'prices': market_data['prices'],
            'returns': market_data['returns'],
            'volumes': market_data.get('volumes', []),
            'highs': market_data.get('highs', []),
            'lows': market_data.get('lows', [])
        }
        
        # Configure strategy parameters
        strategy_params = {
            'detector_method': request.regime_methods[0] if request.regime_methods else 'trend',
            'detector_params': {
                'n_regimes': request.n_regimes,
            },
            'initial_capital': request.initial_capital,
            'position_sizing': request.position_sizing,
            'max_position_size': request.max_position_size,
            'stop_loss_pct': request.stop_loss_pct,
            'take_profit_pct': request.take_profit_pct
        }
        
        # Include additional methods if using ensemble
        if request.regime_methods and 'ensemble' in request.regime_methods:
            strategy_params['detector_method'] = 'ensemble'
            strategy_params['detector_params']['methods'] = [m for m in request.regime_methods if m != 'ensemble']
        
        # Create and run strategy
        strategy = RegimeStrategy(**strategy_params)
        results = strategy.backtest(data)
        
        # Generate visualizations
        equity_curve_file = f"{request_id}_equity.png"
        equity_curve_path = str(config.VISUALIZATION_DIR / equity_curve_file)
        
        # Get the equity curve figure and save it
        equity_fig = strategy.plot_equity_curve(figsize=(12, 6))
        equity_fig.savefig(equity_curve_path)
        plt.close(equity_fig)
        
        regime_chart_file = f"{request_id}_regimes.png"
        regime_chart_path = str(config.VISUALIZATION_DIR / regime_chart_file)
        
        # Get the regime performance figure and save it
        regime_fig = strategy.plot_regime_performance(figsize=(12, 6))
        regime_fig.savefig(regime_chart_path)
        plt.close(regime_fig)
        
        # Extract performance metrics
        metrics = results['performance_metrics']
        regime_metrics = results.get('regime_metrics', {})
        trades = results.get('trades', [])
        
        # Prepare response
        response = {
            "request_id": request_id,
            "symbol": symbol,
            "strategy": request.strategy_type,
            "performance_metrics": {
                "total_return": float(metrics['total_return']),
                "annual_return": float(metrics['annual_return']),
                "annual_volatility": float(metrics['annual_volatility']),
                "sharpe_ratio": float(metrics['sharpe_ratio']),
                "sortino_ratio": float(metrics['sortino_ratio']),
                "calmar_ratio": float(metrics['calmar_ratio']),
                "max_drawdown": float(metrics['max_drawdown']),
                "win_rate": float(metrics['win_rate']),
                "profit_factor": float(metrics['profit_factor']),
                "num_trades": int(metrics['num_trades'])
            },
            "regime_metrics": regime_metrics,
            "equity_curve_url": f"/static/visualizations/{equity_curve_file}",
            "regime_chart_url": f"/static/visualizations/{regime_chart_file}",
            "trades": trades,
            "execution_time": (datetime.now() - start_time).total_seconds()
        }
        
        # Log response
        utils.log_response("backtest", response, response["execution_time"])
        
        return response
        
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 