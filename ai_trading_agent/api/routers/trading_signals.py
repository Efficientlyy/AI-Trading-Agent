"""
Trading Signals API Router

This module provides FastAPI endpoints for accessing trading signals generated from
various sources including sentiment analysis, technical indicators, and market regime detection.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Depends, Body
from pydantic import BaseModel
from datetime import datetime
from decimal import Decimal

# Use absolute imports with fallback for development environment
try:
    from ai_trading_agent.signal_processing.signal_aggregator import (
        SignalAggregator, TradingSignal, SignalType, 
        SignalDirection, SignalTimeframe, ConflictResolutionStrategy
    )
    from ai_trading_agent.strategies.enhanced_sentiment_strategy import EnhancedSentimentStrategy
    from ai_trading_agent.strategies.sentiment_trend_strategy import SentimentTrendStrategy
    from ai_trading_agent.strategies.sentiment_divergence_strategy import SentimentDivergenceStrategy
    from ai_trading_agent.strategies.sentiment_shock_strategy import SentimentShockStrategy
    from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
    from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType
except ImportError:
    # Fallback for development environment - create mock classes
    logger = logging.getLogger(__name__)
    logger.warning("Using mock signal processing classes due to import error")
    
    # Create mock classes
    class SignalType:
        SENTIMENT = "sentiment"
        TECHNICAL = "technical"
        REGIME = "regime"
        AGGREGATE = "aggregate"
    
    class SignalDirection:
        BUY = "buy"
        STRONG_BUY = "strong_buy"
        SELL = "sell"
        STRONG_SELL = "strong_sell"
        NEUTRAL = "neutral"
    
    class SignalTimeframe:
        SHORT = "short"
        MEDIUM = "medium"
        LONG = "long"
    
    class ConflictResolutionStrategy:
        WEIGHTED_AVERAGE = "weighted_average"
        MAJORITY_VOTE = "majority_vote"
        HIGHEST_CONFIDENCE = "highest_confidence"
    
    class TradingSignal:
        def __init__(self, symbol, signal_type, direction, strength, confidence, timeframe, source, timestamp, metadata=None):
            self.symbol = symbol
            self.signal_type = signal_type
            self.direction = direction
            self.strength = strength
            self.confidence = confidence
            self.timeframe = timeframe
            self.source = source
            self.timestamp = timestamp
            self.metadata = metadata or {}
    
    class SignalAggregator:
        def __init__(self, config=None):
            self.config = config or {}
        
        def aggregate_signals(self, signals, config=None):
            return signals
    
    class EnhancedSentimentStrategy:
        def __init__(self, config=None):
            self.config = config or {}
        
        def generate_signals(self, sentiment_data, market_data=None):
            return []
        
        def evaluate_signal_performance(self, days_back=30):
            return {"win_rate": 0.5, "profit_factor": 1.2}
    
    class SentimentTrendStrategy(EnhancedSentimentStrategy):
        pass
    
    class SentimentDivergenceStrategy(EnhancedSentimentStrategy):
        pass
    
    class SentimentShockStrategy(EnhancedSentimentStrategy):
        pass
    
    class OrderSide:
        BUY = "buy"
        SELL = "sell"
    
    class OrderType:
        MARKET = "market"
        LIMIT = "limit"
        STOP = "stop"
    
    class Order:
        def __init__(self, symbol, side, order_type, quantity, price, stop_price=None):
            self.symbol = symbol
            self.side = side
            self.order_type = order_type
            self.quantity = quantity
            self.price = price
            self.stop_price = stop_price
    
    class PortfolioManager:
        def __init__(self, initial_capital=10000):
            self.initial_capital = initial_capital
        
        def get_position_size(self, symbol, risk_per_trade, price):
            return 1.0
from ..sentiment_api import SentimentAPI

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/trading-signals", tags=["trading-signals"])

# Define response models
class SignalModel(BaseModel):
    symbol: str
    signal_type: str
    direction: str
    strength: float
    confidence: float
    timeframe: str
    source: str
    timestamp: str
    metadata: Dict[str, Any] = {}

class OrderModel(BaseModel):
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    created_at: str

class SignalResponseModel(BaseModel):
    signals: List[SignalModel]
    orders: List[OrderModel]
    timestamp: str
    metadata: Dict[str, Any] = {}

class StrategyConfigModel(BaseModel):
    sentiment_threshold: float = 0.2
    window_size: int = 3
    sentiment_weight: float = 0.4
    min_confidence: float = 0.6
    enable_regime_detection: bool = True
    volatility_window: int = 20
    trend_window: int = 50
    volatility_threshold: float = 0.015
    trend_threshold: float = 0.6
    range_threshold: float = 0.3
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.1
    timeframe: str = "1d"
    assets: List[str] = ["BTC", "ETH"]
    topics: List[str] = ["blockchain", "cryptocurrency"]
    days_back: int = 7
    
class AggregatorConfigModel(BaseModel):
    conflict_strategy: str = "weighted_average"
    min_confidence: float = 0.5
    min_strength: float = 0.3
    min_signals: int = 1
    max_signal_age_hours: float = 24
    enable_regime_detection: bool = True
    signal_weights: Dict[str, float] = {}
    timeframe_weights: Dict[str, float] = {}
    source_weights: Dict[str, float] = {}

# Dependency for getting strategy instances
def get_sentiment_strategies(config: Optional[StrategyConfigModel] = None):
    """Dependency to get sentiment strategy instances"""
    config_dict = config.dict() if config else {}
    
    # Create strategies
    enhanced = EnhancedSentimentStrategy(config=config_dict)
    trend = SentimentTrendStrategy(config=config_dict)
    divergence = SentimentDivergenceStrategy(config=config_dict)
    shock = SentimentShockStrategy(config=config_dict)
    
    return {
        "enhanced": enhanced,
        "trend": trend,
        "divergence": divergence,
        "shock": shock
    }

def get_signal_aggregator(config: Optional[AggregatorConfigModel] = None):
    """Dependency to get signal aggregator instance"""
    config_dict = config.dict() if config else {}
    return SignalAggregator(config=config_dict)

def get_portfolio_manager():
    """Dependency to get portfolio manager instance"""
    # In a real implementation, this would load the actual portfolio
    # For now, create a simple portfolio with some initial capital
    return PortfolioManager(initial_capital=Decimal("10000"))

# Define endpoints
@router.post("/sentiment", response_model=SignalResponseModel)
async def get_sentiment_signals(
    strategy_type: str = Query("enhanced", description="Strategy type: enhanced, trend, divergence, shock"),
    strategy_config: Optional[StrategyConfigModel] = Body(None),
    sentiment_api: SentimentAPI = Depends(lambda: SentimentAPI(use_mock=False)),
    strategies: Dict[str, Any] = Depends(get_sentiment_strategies),
    portfolio_manager: PortfolioManager = Depends(get_portfolio_manager)
):
    """
    Get trading signals based on sentiment analysis.
    
    This endpoint generates trading signals using the specified sentiment strategy.
    It integrates real-time sentiment data with market regime detection to produce
    actionable trading signals.
    """
    try:
        # Validate strategy type
        if strategy_type not in strategies:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy type. Must be one of: {', '.join(strategies.keys())}"
            )
        
        # Get the requested strategy
        strategy = strategies[strategy_type]
        
        # Create mock market prices for now
        # In a real implementation, this would fetch current market prices
        market_prices = {
            "BTC": Decimal("30000"),
            "ETH": Decimal("2000"),
            "XRP": Decimal("0.5"),
            "ADA": Decimal("0.4"),
            "SOL": Decimal("100"),
            "DOGE": Decimal("0.1")
        }
        
        # Run the strategy to generate orders
        orders = strategy.run_strategy(portfolio_manager, market_prices)
        
        # Extract signals from strategy
        # Different strategies store signals differently, so we need to handle each case
        signals = []
        if strategy_type == "enhanced" and hasattr(strategy, "signal_history"):
            for entry in strategy.signal_history:
                if "signal" in entry:
                    signals.append(TradingSignal(
                        symbol=entry["symbol"],
                        signal_type=SignalType.SENTIMENT,
                        direction=SignalDirection.BUY if entry["signal"]["direction"] == "buy" else SignalDirection.SELL,
                        strength=entry["signal"]["strength"],
                        confidence=entry["signal"]["confidence"],
                        timeframe=entry["signal"]["timeframe"],
                        source=f"SentimentStrategy:{strategy_type}",
                        timestamp=datetime.now(),
                        metadata=entry["signal"]["metadata"] if "metadata" in entry["signal"] else {}
                    ))
        
        # Convert orders to response model
        order_models = []
        for order in orders:
            order_models.append(OrderModel(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                take_profit_price=None,  # Not implemented yet
                created_at=datetime.now().isoformat()
            ))
        
        # Convert signals to response model
        signal_models = []
        for signal in signals:
            signal_models.append(SignalModel(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                direction=signal.direction.value,
                strength=signal.strength,
                confidence=signal.confidence,
                timeframe=signal.timeframe.value,
                source=signal.source,
                timestamp=signal.timestamp.isoformat(),
                metadata=signal.metadata
            ))
        
        # Return result
        return SignalResponseModel(
            signals=signal_models,
            orders=order_models,
            timestamp=datetime.now().isoformat(),
            metadata={
                "strategy": strategy_type,
                "assets": strategy.assets
            }
        )
    except Exception as e:
        logger.error(f"Error generating sentiment signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/aggregate", response_model=SignalResponseModel)
async def aggregate_signals(
    symbols: List[str] = Query(["BTC", "ETH"], description="Symbols to aggregate signals for"),
    strategy_config: Optional[StrategyConfigModel] = Body(None),
    aggregator_config: Optional[AggregatorConfigModel] = Body(None),
    strategies: Dict[str, Any] = Depends(get_sentiment_strategies),
    aggregator: SignalAggregator = Depends(get_signal_aggregator),
    portfolio_manager: PortfolioManager = Depends(get_portfolio_manager)
):
    """
    Aggregate trading signals from multiple sources.
    
    This endpoint combines signals from different strategies and sources,
    using configurable weighting and conflict resolution to produce a unified signal.
    """
    try:
        # Create mock market prices for now
        market_prices = {symbol: Decimal("1000") for symbol in symbols}
        market_prices.update({
            "BTC": Decimal("30000"),
            "ETH": Decimal("2000"),
            "XRP": Decimal("0.5"),
            "ADA": Decimal("0.4"),
            "SOL": Decimal("100"),
            "DOGE": Decimal("0.1")
        })
        
        # Collect signals from all strategies
        all_signals = []
        for strategy_name, strategy in strategies.items():
            # Run each strategy to generate signals
            if hasattr(strategy, "run_strategy"):
                # Get orders from strategy
                orders = strategy.run_strategy(portfolio_manager, market_prices)
                
                # Extract signals from strategy
                if hasattr(strategy, "signal_history"):
                    for entry in strategy.signal_history:
                        if "signal" in entry:
                            signal_data = entry["signal"]
                            if isinstance(signal_data, dict):
                                all_signals.append(TradingSignal(
                                    symbol=entry["symbol"],
                                    signal_type=SignalType.SENTIMENT,
                                    direction=SignalDirection.BUY if signal_data.get("direction", "") == "buy" else SignalDirection.SELL,
                                    strength=signal_data.get("strength", 0.5),
                                    confidence=signal_data.get("confidence", 0.5),
                                    timeframe=signal_data.get("timeframe", "1d"),
                                    source=f"SentimentStrategy:{strategy_name}",
                                    timestamp=datetime.now(),
                                    metadata=signal_data.get("metadata", {})
                                ))
        
        # Aggregate signals for each symbol
        aggregated_signals = []
        orders = []
        
        for symbol in symbols:
            # Filter signals for this symbol
            symbol_signals = [s for s in all_signals if s.symbol == symbol]
            
            if symbol_signals:
                # Aggregate signals
                aggregated = aggregator.aggregate_signals(symbol_signals)
                
                if aggregated:
                    aggregated_signals.append(aggregated)
                    
                    # Generate order from aggregated signal
                    side = OrderSide.BUY if aggregated.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else OrderSide.SELL
                    
                    # Calculate position size based on signal strength and confidence
                    position_size = 0.1 * aggregated.strength * aggregated.confidence
                    
                    # Create order
                    price = float(market_prices.get(symbol, Decimal("1000")))
                    stop_price = price * 0.95 if side == OrderSide.BUY else price * 1.05
                    
                    order = Order(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=position_size,
                        price=price,
                        stop_price=stop_price
                    )
                    
                    orders.append(order)
        
        # Convert signals to response model
        signal_models = []
        for signal in aggregated_signals:
            signal_models.append(SignalModel(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                direction=signal.direction.value,
                strength=signal.strength,
                confidence=signal.confidence,
                timeframe=signal.timeframe.value,
                source=signal.source,
                timestamp=signal.timestamp.isoformat(),
                metadata=signal.metadata
            ))
        
        # Convert orders to response model
        order_models = []
        for order in orders:
            order_models.append(OrderModel(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                take_profit_price=None,  # Not implemented yet
                created_at=datetime.now().isoformat()
            ))
        
        # Return result
        return SignalResponseModel(
            signals=signal_models,
            orders=order_models,
            timestamp=datetime.now().isoformat(),
            metadata={
                "aggregator": aggregator_config.dict() if aggregator_config else {},
                "symbols": symbols
            }
        )
    except Exception as e:
        logger.error(f"Error aggregating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance", response_model=Dict[str, Any])
async def get_signal_performance(
    strategy_type: str = Query("enhanced", description="Strategy type: enhanced, trend, divergence, shock"),
    days_back: int = Query(30, description="Number of days to look back for performance evaluation"),
    strategies: Dict[str, Any] = Depends(get_sentiment_strategies)
):
    """
    Get performance metrics for a specific strategy.
    
    This endpoint evaluates the historical performance of signals generated by the specified strategy.
    """
    try:
        # Validate strategy type
        if strategy_type not in strategies:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy type. Must be one of: {', '.join(strategies.keys())}"
            )
        
        # Get the requested strategy
        strategy = strategies[strategy_type]
        
        # Check if the strategy has a performance evaluation method
        if hasattr(strategy, "evaluate_signal_performance"):
            performance = strategy.evaluate_signal_performance(days_back=days_back)
            return performance
        else:
            return {
                "error": f"Strategy {strategy_type} does not support performance evaluation"
            }
    except Exception as e:
        logger.error(f"Error evaluating signal performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
