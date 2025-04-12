"""
Integration test for sentiment-driven trading pipeline.
"""

import pytest
from ai_trading_agent.sentiment_analysis.manager import SentimentManager
from ai_trading_agent.trading_engine.models import Order, Portfolio
from ai_trading_agent.trading_engine.order_manager import OrderManager
from ai_trading_agent.trading_engine.execution_handler import ExecutionHandler
from ai_trading_agent.trading_engine.enums import OrderSide, OrderType

@pytest.mark.asyncio
async def test_sentiment_to_trade_pipeline():
    sentiment_manager = SentimentManager()
    signal = sentiment_manager.get_sentiment_signal()

    portfolio = Portfolio(initial_capital=100000)
    order_manager = OrderManager(portfolio)
    execution_handler = ExecutionHandler()

    if signal == "buy":
        order = order_manager.create_order(
            symbol="BTC/USD",
            side="buy",
            order_type="market",
            quantity=0.1
        )
    elif signal == "sell":
        order = order_manager.create_order(
            symbol="BTC/USD",
            side="sell",
            order_type="market",
            quantity=0.1
        )
    else:
        order = None

    if order:
        # Simulate market data
        import pandas as pd
        import numpy as np
        timestamp = pd.Timestamp.now()
        prices = np.linspace(50000, 51000, 10)
        market_data = pd.DataFrame({"close": prices}, index=pd.date_range(timestamp, periods=10))

        trades = execution_handler.execute_order(order, market_data, timestamp)
        assert isinstance(trades, list)

        # Update portfolio with trades
        for trade in trades:
            portfolio.update_from_trade(trade, {"BTC/USD": trade.price})

        # Check portfolio state
        pos = portfolio.positions.get("BTC/USD")
        assert pos is None or abs(pos.quantity) <= 0.1
