"""
Example script demonstrating how to use the sentiment analysis system for trading.

This script shows how to:
1. Fetch sentiment data from Alpha Vantage
2. Process it to generate trading signals
3. Integrate with the trading engine to execute trades based on sentiment
"""

import os
import logging
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

from ai_trading_agent.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from ai_trading_agent.strategies.simple_sentiment_strategy import SimpleSentimentStrategy
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
from ai_trading_agent.trading_engine.models import Portfolio, Position, Order, OrderSide, OrderType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the sentiment trading example."""
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.error("Alpha Vantage API key not available. Please set ALPHA_VANTAGE_API_KEY in .env file.")
        return
    
    # Initialize the sentiment strategy
    logger.info("Initializing sentiment strategy...")
    strategy = SimpleSentimentStrategy({
        "sentiment_analyzer": {
            "alpha_vantage_api_key": api_key,
            "sentiment_threshold": 0.25,  # Higher threshold for stronger signals
            "feature_weights": {
                "sentiment_score": 1.0,
                "sentiment_trend": 1.2,    # Increased weight for trend
                "sentiment_momentum": 0.9,
                "sentiment_volatility": -0.5  # Stronger negative weight for volatility
            }
        },
        "topics": ["blockchain", "cryptocurrency"],
        "assets": ["BTC", "ETH"],
        "sentiment_threshold": 0.25,
        "risk_per_trade": 0.02,
        "max_position_size": 0.1,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.1
    })
    
    # Current market prices (simulated)
    market_prices = {
        "BTC": Decimal("50000"),
        "ETH": Decimal("3000")
    }
    
    # Initialize portfolio manager with initial capital
    portfolio_manager = PortfolioManager(
        initial_capital=100000.0,  # $100,000 initial capital
        risk_per_trade=0.02,       # 2% risk per trade
        max_position_size=0.1      # Max 10% of portfolio in any position
    )
    
    # Run the sentiment strategy to generate orders
    logger.info("Running sentiment strategy to generate orders...")
    orders = strategy.run_strategy(portfolio_manager)
    
    # Log the generated orders
    for order in orders:
        logger.info(f"Generated order: {order}")
    
    # Print portfolio summary
    logger.info("\nPortfolio Summary:")
    portfolio_state = portfolio_manager.get_portfolio_state()
    logger.info(f"Cash Balance: ${portfolio_state['cash']}")
    logger.info(f"Total Portfolio Value: ${portfolio_state['total_value']}")
    logger.info(f"Positions: {len(portfolio_state['positions'])}")
    for symbol, position_data in portfolio_state['positions'].items():
        logger.info(f"  {symbol}: {position_data['quantity']} @ ${position_data['entry_price']}")
    
    # Print orders summary
    logger.info("\nOrders Summary:")
    logger.info(f"Total Orders: {len(orders)}")
    for order in orders:
        logger.info(f"  {order.side.name} {order.quantity} {order.symbol} @ ${order.price}")
    
    # Calculate total order value
    total_order_value = sum(order.quantity * order.price for order in orders)
    logger.info(f"Total Order Value: ${total_order_value}")
    
    # Calculate percentage of portfolio if we have a portfolio value
    if portfolio_state['total_value'] > 0:
        percentage = (float(total_order_value) / float(portfolio_state['total_value'])) * 100
        logger.info(f"Percentage of Portfolio: {percentage:.2f}%")

if __name__ == "__main__":
    main()
