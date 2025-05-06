"""
Example script demonstrating how to use the integrated sentiment trading system.

This script shows how to:
1. Fetch sentiment data from Alpha Vantage
2. Process it to generate trading signals using multiple sentiment strategies
3. Aggregate signals using the SignalAggregator
4. Execute trades via the LiveTradingBridge with paper trading
5. Monitor trading status and performance
"""

import os
import logging
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Import sentiment analysis components
from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.sentiment_analysis.alpha_vantage_connector import AlphaVantageSentimentConnector

# Import signal processing components
from ai_trading_agent.signal_processing.signal_aggregator import (
    SignalAggregator, TradingSignal, SignalType, 
    SignalDirection, SignalTimeframe
)

# Import strategy components
from ai_trading_agent.strategies.enhanced_sentiment_strategy import EnhancedSentimentStrategy
from ai_trading_agent.strategies.sentiment_trend_strategy import SentimentTrendStrategy
from ai_trading_agent.strategies.sentiment_divergence_strategy import SentimentDivergenceStrategy
from ai_trading_agent.strategies.sentiment_shock_strategy import SentimentShockStrategy

# Import trading engine components
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
from ai_trading_agent.trading_engine.live_trading_bridge import LiveTradingBridge, PaperTradingExchange
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType, Position, Portfolio

# Import data pipeline components
from ai_trading_agent.data_pipeline.data_integration import DataIntegrationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run the integrated sentiment trading example."""
    # Load environment variables
    load_dotenv()
    
    # Get API keys from environment
    alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    twelve_data_api_key = os.getenv("TWELVE_DATA_API_KEY")
    
    if not alpha_vantage_api_key:
        logger.error("Alpha Vantage API key not available. Please set ALPHA_VANTAGE_API_KEY in .env file.")
        return
    
    # Initialize the data integration manager
    logger.info("Initializing data integration manager...")
    data_manager = DataIntegrationManager({
        "alpha_vantage": {
            "api_key": alpha_vantage_api_key
        },
        "twelve_data": {
            "api_key": twelve_data_api_key
        },
        "sentiment_symbols": ["BTC", "ETH"],
        "market_symbols": ["BTC/USD", "ETH/USD"],
        "sentiment_collection_interval": 900,  # 15 minutes
        "market_collection_interval": 3600,    # 1 hour
        "cache_dir": "data_cache"
    })
    
    # Start the data manager
    await data_manager.start()
    
    # Initialize sentiment strategies
    logger.info("Initializing sentiment strategies...")
    strategy_config = {
        "sentiment_threshold": 0.2,
        "window_size": 3,
        "sentiment_weight": 0.4,
        "min_confidence": 0.6,
        "enable_regime_detection": True,
        "volatility_window": 20,
        "trend_window": 50,
        "volatility_threshold": 0.015,
        "trend_threshold": 0.6,
        "range_threshold": 0.3,
        "risk_per_trade": 0.02,
        "max_position_size": 0.1,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.1,
        "assets": ["BTC", "ETH"],
        "topics": ["blockchain", "cryptocurrency"],
        "days_back": 7
    }
    
    # Create the strategies
    enhanced_strategy = EnhancedSentimentStrategy(strategy_config)
    trend_strategy = SentimentTrendStrategy(strategy_config)
    divergence_strategy = SentimentDivergenceStrategy(strategy_config)
    shock_strategy = SentimentShockStrategy(strategy_config)
    
    # Initialize signal aggregator
    logger.info("Initializing signal aggregator...")
    aggregator = SignalAggregator({
        "conflict_strategy": "weighted_average",
        "min_confidence": 0.5,
        "min_strength": 0.3,
        "min_signals": 1,
        "max_signal_age_hours": 24,
        "enable_regime_detection": True,
        "signal_weights": {
            "SENTIMENT": 1.0,
            "TECHNICAL": 0.8,
            "FUNDAMENTAL": 0.7,
            "MARKET_REGIME": 1.2
        },
        "timeframe_weights": {
            "1h": 0.7,
            "4h": 0.8,
            "1d": 1.0,
            "1w": 1.2
        },
        "source_weights": {
            "EnhancedSentimentStrategy": 1.0,
            "SentimentTrendStrategy": 0.9,
            "SentimentDivergenceStrategy": 0.8,
            "SentimentShockStrategy": 0.7
        }
    })
    
    # Initialize portfolio manager
    logger.info("Initializing portfolio manager...")
    portfolio_manager = PortfolioManager(
        initial_capital=Decimal("10000"),  # $10,000 initial capital
        risk_per_trade=Decimal("0.02"),    # 2% risk per trade
        max_position_size=Decimal("0.1")   # Max 10% of portfolio in any position
    )
    
    # Initialize live trading bridge with paper trading
    logger.info("Initializing live trading bridge...")
    trading_bridge = LiveTradingBridge({
        "mode": "paper",
        "exchange_type": "paper",
        "exchange": {
            "initial_balance": 10000
        },
        "max_position_size": 0.1,
        "max_loss_pct": 0.02,
        "max_daily_trades": 10,
        "min_order_value": 10
    })
    
    # Initialize the trading bridge
    await trading_bridge.initialize()
    
    # Get sentiment data for BTC and ETH
    logger.info("Fetching sentiment data...")
    btc_sentiment = await data_manager.get_sentiment_data("BTC")
    eth_sentiment = await data_manager.get_sentiment_data("ETH")
    
    # Get market data for BTC/USD and ETH/USD
    logger.info("Fetching market data...")
    btc_market = await data_manager.get_historical_market_data("BTC/USD", "1day", 30)
    eth_market = await data_manager.get_historical_market_data("ETH/USD", "1day", 30)
    
    # Get real-time prices
    logger.info("Getting real-time prices...")
    btc_price = await data_manager.get_real_time_price("BTC/USD")
    eth_price = await data_manager.get_real_time_price("ETH/USD")
    
    market_prices = {
        "BTC": btc_price or Decimal("30000"),
        "ETH": eth_price or Decimal("2000")
    }
    
    logger.info(f"Current market prices: BTC=${market_prices['BTC']}, ETH=${market_prices['ETH']}")
    
    # Generate signals from each strategy
    logger.info("Generating trading signals from strategies...")
    all_signals = []
    
    # Run each strategy
    for name, strategy in {
        "enhanced": enhanced_strategy,
        "trend": trend_strategy,
        "divergence": divergence_strategy,
        "shock": shock_strategy
    }.items():
        logger.info(f"Running {name} strategy...")
        
        # Set sentiment data
        strategy.set_sentiment_data({
            "BTC": btc_sentiment,
            "ETH": eth_sentiment
        })
        
        # Set market data
        strategy.set_market_data({
            "BTC": btc_market,
            "ETH": eth_market
        })
        
        # Run strategy
        orders = strategy.run_strategy(portfolio_manager, market_prices)
        logger.info(f"  Generated {len(orders)} orders")
        
        # Extract signals from strategy
        if hasattr(strategy, "signal_history") and strategy.signal_history:
            for entry in strategy.signal_history:
                if "signal" in entry and isinstance(entry["signal"], dict):
                    signal_data = entry["signal"]
                    all_signals.append(TradingSignal(
                        symbol=entry["symbol"],
                        signal_type=SignalType.SENTIMENT,
                        direction=SignalDirection.BUY if signal_data.get("direction", "") == "buy" else SignalDirection.SELL,
                        strength=signal_data.get("strength", 0.5),
                        confidence=signal_data.get("confidence", 0.5),
                        timeframe=SignalTimeframe.D1,
                        source=f"{name.capitalize()}SentimentStrategy",
                        timestamp=datetime.now(),
                        metadata=signal_data.get("metadata", {})
                    ))
    
    # Aggregate signals
    logger.info(f"Aggregating {len(all_signals)} signals...")
    
    # Group signals by symbol
    btc_signals = [s for s in all_signals if s.symbol == "BTC"]
    eth_signals = [s for s in all_signals if s.symbol == "ETH"]
    
    aggregated_signals = []
    
    if btc_signals:
        aggregated_btc = aggregator.aggregate_signals(btc_signals)
        if aggregated_btc:
            aggregated_signals.append(aggregated_btc)
            logger.info(f"BTC aggregated signal: {aggregated_btc.direction.name} (strength={aggregated_btc.strength:.2f}, confidence={aggregated_btc.confidence:.2f})")
    
    if eth_signals:
        aggregated_eth = aggregator.aggregate_signals(eth_signals)
        if aggregated_eth:
            aggregated_signals.append(aggregated_eth)
            logger.info(f"ETH aggregated signal: {aggregated_eth.direction.name} (strength={aggregated_eth.strength:.2f}, confidence={aggregated_eth.confidence:.2f})")
    
    # Execute signals through trading bridge
    if aggregated_signals:
        logger.info(f"Executing {len(aggregated_signals)} aggregated signals...")
        results = await trading_bridge.execute_signals(aggregated_signals)
        
        for result in results:
            signal = result["signal"]
            execution = result["result"]
            success = execution.get("success", False)
            
            logger.info(f"Signal execution for {signal['symbol']}: {'SUCCESS' if success else 'FAILED'}")
            if not success:
                logger.error(f"  Error: {execution.get('error', 'Unknown error')}")
    else:
        logger.info("No aggregated signals to execute")
    
    # Get trading status
    status = await trading_bridge.get_trading_status()
    
    # Print trading status
    logger.info("\nTrading Status:")
    logger.info(f"Mode: {status['mode']}")
    logger.info(f"Portfolio Value: ${status['portfolio_value']:.2f}")
    logger.info(f"Cash: ${status['cash']:.2f}")
    logger.info(f"Daily Trades: {status['daily_trades']}/{status['max_daily_trades']}")
    logger.info(f"Active Orders: {status['active_orders']}")
    
    # Print positions
    logger.info("\nPositions:")
    for symbol, position in status['positions'].items():
        logger.info(f"  {symbol}: {position['quantity']} @ ${position['entry_price']} (Current: ${position['current_price']})")
        logger.info(f"    P&L: ${position['unrealized_pnl']:.2f}")
    
    # Shutdown components
    logger.info("\nShutting down...")
    await trading_bridge.shutdown()
    await data_manager.stop()

if __name__ == "__main__":
    asyncio.run(main())
