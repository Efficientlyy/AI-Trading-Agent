"""
Advanced sentiment trading example.

This example demonstrates how to use the advanced sentiment strategy for trading.
"""

import os
import logging
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta
from dotenv import load_dotenv

from ai_trading_agent.strategies.advanced_sentiment_strategy import AdvancedSentimentStrategy
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
from ai_trading_agent.trading_engine.models import Portfolio, Position, Order, OrderSide, OrderType, OrderStatus, Trade

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def run_example():
    """Run the advanced sentiment trading example."""
    # Check if Alpha Vantage API key is available
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.error("Alpha Vantage API key not found. Please set the ALPHA_VANTAGE_API_KEY environment variable.")
        return
    
    logger.info("Starting advanced sentiment trading example")
    
    # Create a portfolio manager
    initial_capital = Decimal("100000")
    portfolio_manager = PortfolioManager(
        initial_capital=initial_capital,
        risk_per_trade=Decimal("0.02"),
        max_position_size=Decimal("0.1"),
        max_correlation=Decimal("0.7"),
        rebalance_frequency=7
    )
    
    # Create a strategy with advanced Alpha Vantage configuration
    strategy_config = {
        "sentiment_threshold": 0.2,
        "position_sizing_method": "risk_based",
        "risk_per_trade": 0.02,
        "max_position_size": 0.1,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.1,
        "topics": ["blockchain", "cryptocurrency", "defi"],
        "assets": {
            "blockchain": ["BTC", "ETH"],
            "cryptocurrency": ["BTC", "ETH", "XRP", "ADA", "SOL"],
            "defi": ["ETH", "UNI", "AAVE", "COMP", "MKR"]
        },
        "topic_asset_weights": {
            "blockchain": {"BTC": 0.7, "ETH": 0.3},
            "cryptocurrency": {"BTC": 0.4, "ETH": 0.3, "XRP": 0.1, "ADA": 0.1, "SOL": 0.1},
            "defi": {"ETH": 0.3, "UNI": 0.2, "AAVE": 0.2, "COMP": 0.15, "MKR": 0.15}
        },
        # Alpha Vantage API tier configuration
        "alpha_vantage_tier": "free",  # Can be 'free', 'premium', or 'enterprise'
        
        # Sentiment analyzer configuration
        "sentiment_analyzer": {
            # Alpha Vantage client configuration
            "alpha_vantage_client": {
                "api_key": api_key,
                "tier": "free",
                "cache_ttl": 3600,  # Cache TTL in seconds (1 hour)
                "use_cache": True
            },
            # Fallback topics when specific queries fail
            "fallback_topics": ["blockchain", "cryptocurrency", "finance", "technology"],
            # Feature weights for sentiment analysis
            "feature_weights": {
                "sentiment_score": 1.0,
                "sentiment_trend": 0.8,
                "sentiment_momentum": 0.7,
                "sentiment_anomaly": 0.5,
                "sentiment_volatility": -0.3,
                "sentiment_roc": 0.6,
                "sentiment_acceleration": 0.4
            }
        }
    }
    strategy = AdvancedSentimentStrategy(config=strategy_config)
    
    # Run the strategy for multiple days
    num_days = 7
    start_date = datetime.now() - timedelta(days=num_days)
    
    # Create a dictionary to store daily portfolio values
    portfolio_values = {}
    
    # Simulate trading for each day
    for day in range(num_days + 1):
        current_date = start_date + timedelta(days=day)
        logger.info(f"Running strategy for {current_date.strftime('%Y-%m-%d')}")
        
        # Run the strategy to generate orders
        orders = strategy.run_strategy(portfolio_manager)
        
        # Print the generated orders
        if orders:
            logger.info(f"Generated {len(orders)} orders:")
            for order in orders:
                logger.info(f"  {order}")
        else:
            logger.info("No orders generated")
        
        # Simulate order execution (in a real system, this would be handled by an execution engine)
        for order in orders:
            # Create a trade for the order
            trade = Trade(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                timestamp=current_date
            )
            
            # Update the portfolio with the trade
            # In a real system, we would use actual market prices
            current_prices = {
                "BTC": Decimal("50000"),
                "ETH": Decimal("3000"),
                "XRP": Decimal("0.5"),
                "ADA": Decimal("1.2"),
                "SOL": Decimal("100"),
                "UNI": Decimal("20"),
                "AAVE": Decimal("150"),
                "COMP": Decimal("200"),
                "MKR": Decimal("2000")
            }
            
            portfolio_manager.portfolio.update_from_trade(trade, current_prices)
            
            # Mark the order as filled
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.remaining_quantity = Decimal("0")
        
        # Update strategy performance history
        strategy.update_performance_history(portfolio_manager)
        
        # Store portfolio value for this day
        portfolio_values[current_date.strftime('%Y-%m-%d')] = float(portfolio_manager.portfolio.total_value)
        
        # Print portfolio summary
        print_portfolio_summary(portfolio_manager, current_date)
    
    # Print performance metrics
    performance_metrics = strategy.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(f"Total Return: {performance_metrics['total_return'] * 100:.2f}%")
    print(f"Annualized Return: {performance_metrics['annualized_return'] * 100:.2f}%")
    print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {performance_metrics['max_drawdown'] * 100:.2f}%")
    
    # Plot portfolio value over time
    try:
        import matplotlib.pyplot as plt
        
        # Create a DataFrame from the portfolio values
        df = pd.DataFrame(list(portfolio_values.items()), columns=['Date', 'Value'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Plot the portfolio value
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Value'])
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('portfolio_value.png')
        plt.close()
        
        logger.info("Portfolio value chart saved as 'portfolio_value.png'")
    except ImportError:
        logger.warning("Matplotlib not installed. Skipping portfolio value chart.")

def print_portfolio_summary(portfolio_manager, current_date):
    """Print a summary of the portfolio."""
    portfolio = portfolio_manager.portfolio
    
    print("\n" + "=" * 50)
    print(f"Portfolio Summary - {current_date.strftime('%Y-%m-%d')}")
    print("=" * 50)
    
    print(f"Total Value: ${float(portfolio.total_value):,.2f}")
    print(f"Cash Balance: ${float(portfolio.current_balance):,.2f}")
    
    if portfolio.positions:
        print("\nPositions:")
        for symbol, position in portfolio.positions.items():
            print(f"  {symbol}: {float(position.quantity):.6f} units @ ${float(position.entry_price):,.2f}")
            print(f"    Current Value: ${float(position.current_value):,.2f}")
            print(f"    Unrealized P&L: ${float(position.unrealized_pnl):,.2f} ({float(position.unrealized_pnl_pct) * 100:.2f}%)")
    else:
        print("\nNo open positions")
    
    print("=" * 50)

if __name__ == "__main__":
    run_example()
