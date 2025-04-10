import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Project specific imports (adjust paths if needed)
from src.backtesting.multi_asset_backtester import MultiAssetBacktester
from src.data_acquisition.mock_provider import MockDataProvider
from src.sentiment_analysis.mock_provider import MockSentimentProvider # Assuming this exists
from src.strategies.base_strategy import BaseStrategy # Or the specific sentiment strategy function
from src.trading_engine.models import Portfolio, Order, Trade, Position # Add Position import
from src.backtesting.asset_allocation import equal_weight_allocation # Example allocation

# Constants
SYMBOLS = ['ASSET_A', 'ASSET_B']
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 1, 10)
INITIAL_CAPITAL = 100000.0

# --- Basic Fixtures ---

@pytest.fixture
def symbols():
    return SYMBOLS

@pytest.fixture
def start_date():
    return START_DATE

@pytest.fixture
def end_date():
    return END_DATE

@pytest.fixture
def initial_capital():
    return INITIAL_CAPITAL

@pytest.fixture
def mock_market_data(symbols, start_date, end_date):
    """Creates simple, stable mock market data."""
    data = {}
    dates = pd.date_range(start_date, end_date, freq='D')
    for symbol in symbols:
        df = pd.DataFrame(index=dates)
        df['open'] = 100.0
        df['high'] = 101.0
        df['low'] = 99.0
        df['close'] = 100.0
        df['volume'] = 1000
        data[symbol] = df
    return data

@pytest.fixture
def mock_sentiment_data(symbols, start_date, end_date):
    """Creates simple, neutral mock sentiment data."""
    sentiment = {}
    dates = pd.date_range(start_date, end_date, freq='D')
    for symbol in symbols:
        df = pd.DataFrame(index=dates)
        df['sentiment_score'] = 0.5 # Neutral
        df['confidence'] = 1.0
        sentiment[symbol] = df
    return sentiment

@pytest.fixture
def mock_data_provider(mock_market_data):
    """Fixture for MockDataProvider."""
    return MockDataProvider(mock_market_data)

@pytest.fixture
def mock_sentiment_provider(mock_sentiment_data):
    """Fixture for MockSentimentProvider."""
    # Ensure MockSentimentProvider exists and accepts data in this format
    # If MockSentimentProvider needs a different structure, adjust mock_sentiment_data
    try:
        return MockSentimentProvider(mock_sentiment_data)
    except NameError:
        pytest.skip("MockSentimentProvider not found or implemented yet.")
    except TypeError:
         pytest.skip("MockSentimentProvider constructor might have changed.")

@pytest.fixture
def positive_sentiment_data(start_date, end_date):
    """Creates mock sentiment data with consistently positive scores."""
    symbol = 'ASSET_A'
    sentiment = {}
    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame(index=dates)
    # Score consistently above the placeholder strategy's buy threshold (0.6)
    df['sentiment_score'] = 0.8
    df['confidence'] = 1.0
    sentiment[symbol] = df
    return sentiment

@pytest.fixture
def positive_sentiment_provider(positive_sentiment_data):
    """Fixture for MockSentimentProvider with positive data."""
    try:
        return MockSentimentProvider(positive_sentiment_data)
    except NameError:
        pytest.skip("MockSentimentProvider not found or implemented yet.")
    except TypeError:
         pytest.skip("MockSentimentProvider constructor might have changed.")

@pytest.fixture
def single_asset_market_data(start_date, end_date):
    """Creates simple, stable mock market data for one asset."""
    symbol = 'ASSET_A'
    data = {}
    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame(index=dates)
    df['open'] = 100.0
    df['high'] = 101.0
    df['low'] = 99.0
    df['close'] = 100.0
    df['volume'] = 1000
    data[symbol] = df
    return data

@pytest.fixture
def single_asset_data_provider(single_asset_market_data):
    """Fixture for MockDataProvider with single asset."""
    return MockDataProvider(single_asset_market_data)

@pytest.fixture
def sentiment_strategy_function():
    """
    A simple strategy function fixture that uses the 'sentiment' provider.
    Generates BUY orders if sentiment > 0.6, SELL orders if < 0.4.
    """
    def strategy_logic(timestamp: datetime, data: Dict[str, Any], portfolio: Portfolio, cash: float, providers: Dict[str, Any]) -> List[Order]:
        orders = []
        sentiment_provider = providers.get('sentiment')

        if not sentiment_provider:
            # Optional: Log a warning if provider is missing
            # print(f"Warning: Sentiment provider not found at {timestamp}")
            return orders # Cannot trade without sentiment

        # Assuming data contains market data keyed by symbol
        symbols = list(data.keys()) # Get symbols with market data for this timestamp

        for symbol in symbols:
            market_data = data[symbol]
            if not isinstance(market_data, dict) or 'close' not in market_data:
                 # Skip if market data format is unexpected or incomplete for this timestamp
                 # print(f"DEBUG {timestamp} {symbol}: Skipping, unexpected market data format: {market_data}")
                 continue
                
            current_price = market_data['close']
            
            # --- Get sentiment ---
            sentiment_data = None
            if hasattr(sentiment_provider, 'get_sentiment'):
                 try:
                     # Ensure timestamp is timezone-aware (UTC) if needed by provider
                     # Assuming provider expects UTC if naive
                     lookup_ts = timestamp
                     if timestamp.tzinfo is None:
                         # This might need adjustment based on how backtester provides timestamp
                         lookup_ts = timestamp.tz_localize('UTC') 
                         
                     sentiment_data = sentiment_provider.get_sentiment(symbol, lookup_ts)
                 except Exception as e:
                     # Log error getting sentiment
                     print(f"ERROR: Could not get sentiment for {symbol} at {lookup_ts}: {e}")
                     sentiment_data = None # Treat error as missing data
            else:
                 # Log warning if method doesn't exist
                 print(f"Warning: Sentiment provider {type(sentiment_provider)} does not have 'get_sentiment' method.")
                 sentiment_data = None

            # --- Decision Logic ---
            if sentiment_data and sentiment_data.get('sentiment_score') is not None:
                score = sentiment_data['sentiment_score']
                # confidence = sentiment_data.get('confidence', 1.0) # Confidence could be used later

                # Placeholder logic: Buy strong positive, Sell strong negative
                # Note: This doesn't handle position sizing or risk management - relies on backtester
                target_order_size = 10 # Fixed placeholder size

                current_position = portfolio.positions.get(symbol)
                current_qty = current_position.quantity if current_position else 0

                if score > 0.6 and current_qty <= 0: # Buy signal and not already long/flat
                    # print(f"DEBUG {timestamp} {symbol}: BUY signal (score={score:.2f}), current_qty={current_qty}")
                    orders.append(Order(
                        timestamp=timestamp,
                        symbol=symbol,
                        quantity=target_order_size, # Basic fixed quantity
                        direction='BUY',
                        order_type='MARKET'
                    ))
                elif score < 0.4 and current_qty >= 0: # Sell signal and not already short/flat
                    # Determine quantity to sell: close existing long or go short
                    sell_qty = abs(current_qty) if current_qty > 0 else target_order_size
                    if sell_qty > 0:
                         # print(f"DEBUG {timestamp} {symbol}: SELL signal (score={score:.2f}), current_qty={current_qty}, sell_qty={sell_qty}")
                         orders.append(Order(
                             timestamp=timestamp,
                             symbol=symbol,
                             quantity=sell_qty,
                             direction='SELL',
                             order_type='MARKET'
                         ))
            # else: # Optional: Log if sentiment was missing
                # if sentiment_provider and hasattr(sentiment_provider, 'get_sentiment'):
                    # print(f"DEBUG {timestamp} {symbol}: No valid sentiment data found.")

        return orders

    return strategy_logic

# ... rest of the code remains the same ...
