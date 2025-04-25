# AI Trading Agent - Main Orchestration Class

import asyncio
import logging
from decimal import Decimal # Use Decimal for precision
import pandas as pd

from ai_trading_agent.data_providers.twelve_data.client import TwelveDataClient
from ai_trading_agent.data_providers.alpha_vantage.client import AlphaVantageClient
from ai_trading_agent.config import TWELVEDATA_API_KEY, ALPHA_VANTAGE_API_KEY
# --- Import Trading Engine Components ---
from ai_trading_agent.trading_engine.portfolio_manager import PortfolioManager
from ai_trading_agent.trading_engine.execution_handler import ExecutionHandler
from ai_trading_agent.trading_engine.models import Order, OrderSide, OrderType, OrderStatus, Trade
from decimal import Decimal, InvalidOperation
from ai_trading_agent.common.logging_config import setup_logging # Import setup_logging
# --------------------------------------

logger = logging.getLogger(__name__)

class TradingAgent:
    """Main class to coordinate data gathering, analysis, and trading decisions."""

    def __init__(self, symbols: list[str], td_plan_level: str = "free", initial_capital: Decimal = Decimal("10000.0")):
        logger.info("Initializing Trading Agent...")
        self.symbols = symbols
        self.td_plan_level = td_plan_level
        self.initial_capital = initial_capital

        if not TWELVEDATA_API_KEY or not ALPHA_VANTAGE_API_KEY:
            raise ValueError("API keys for Twelve Data and Alpha Vantage must be set in config.")

        self.twelve_data_client = TwelveDataClient(
            api_key=TWELVEDATA_API_KEY,
            symbols=self.symbols,
            plan_level=self.td_plan_level
        )
        self.alpha_vantage_client = AlphaVantageClient(api_key=ALPHA_VANTAGE_API_KEY)

        # --- Initialize Trading Engine --- 
        # TODO: Load exchange fees, slippage models etc from config
        self.execution_handler = ExecutionHandler()
        self.portfolio_manager = PortfolioManager(initial_capital=self.initial_capital)
        logger.info(f"PortfolioManager and ExecutionHandler initialized with capital: {initial_capital}")
        # ---------------------------------

        # Placeholder for storing latest data
        self.latest_price_data = {symbol: None for symbol in self.symbols}
        self.latest_sentiment_data = None # Store raw sentiment feed if needed
        self.latest_sentiment_score: float | None = None # Store processed score
        self.orders_to_process: list[Order] = [] # Initialize orders_to_process

    async def _handle_price_update(self, data):
        """Callback to receive price updates from TwelveDataClient."""
        symbol = data.get('symbol')
        if symbol in self.latest_price_data:
            logger.debug(f"Received price update for {symbol}: {data.get('price')}")
            self.latest_price_data[symbol] = data
            # Trigger analysis based on new price
            await self._run_analysis()

    async def _fetch_sentiment(self):
        """Periodically fetches sentiment data from AlphaVantageClient."""
        # Define relevant topics for crypto sentiment
        # crypto_topics = ["cryptocurrency", "blockchain", "Bitcoin", "Ethereum"]
        update_interval = 120 # Seconds for normal update frequency (e.g., 2 minutes)
        # rate_limit_delay = 70 # Seconds to wait after a suspected rate limit error (e.g., > 1 minute)

        while True:
            logger.info("Fetching sentiment data...")
            try:
                # --- MOCK SENTIMENT --- 
                # Temporarily disable live fetch due to rate limits
                # Set a fixed positive score to allow testing of downstream logic
                mock_score = 0.6
                logger.info(f"Sentiment fetching disabled. Using mock score: {mock_score}")
                self.latest_sentiment_score = mock_score
                # ----------------------

            except Exception as e:
                # Catch any other unexpected error during fetch/processing
                logger.error(f"Error fetching/processing sentiment: {e}", exc_info=True)
                self.latest_sentiment_score = None # Ensure score is None on error
                # Still wait before retrying even after unexpected error
                await asyncio.sleep(update_interval)
                continue # Skip the normal sleep at the end

            # Wait interval before the next fetch attempt (if successful or feed missing)
            await asyncio.sleep(update_interval)

    async def _run_analysis(self):
        """Analyzes the latest price and sentiment data to make decisions."""
        logger.info("Running analysis...")
        
        if self.latest_sentiment_score is None:
            logger.info("Sentiment score not available. Holding.")
            return # Cannot make decision without sentiment

        # Simple Strategy: Use sentiment + latest price (placeholder for real TA)
        signals = {}
        self.orders_to_process = [] # Reset orders_to_process

        for symbol, price_data in self.latest_price_data.items():
            if price_data is None:
                logger.debug(f"No price data for {symbol}. Cannot generate signal.")
                signals[symbol] = "HOLD"
                continue

            latest_price = price_data.get('price')
            if latest_price is None:
                signals[symbol] = "HOLD"
                continue

            # --- Basic Rule --- 
            signal = "HOLD"
            if self.latest_sentiment_score > 0.15: # Threshold for bullish sentiment
                # Add TA check here later (e.g., price > moving_average)
                logger.info(f"[{symbol}] Positive sentiment ({self.latest_sentiment_score:.2f}). Considering BUY.")
                signal = "BUY"
            elif self.latest_sentiment_score < -0.15: # Threshold for bearish sentiment
                # Add TA check here later (e.g., price < moving_average)
                logger.info(f"[{symbol}] Negative sentiment ({self.latest_sentiment_score:.2f}). Considering SELL.")
                signal = "SELL"
            else:
                logger.info(f"[{symbol}] Neutral sentiment ({self.latest_sentiment_score:.2f}). Holding.")
                signal = "HOLD"
            
            signals[symbol] = signal
            
            # --- Create Order Object from Signal --- 
            if signal == "BUY" or signal == "SELL":
                # Use portfolio_manager.calculate_position_size to determine order quantity
                latest_data = self.latest_price_data.get(symbol)
                if not latest_data:
                    logger.warning(f"No latest data for {symbol}, skipping signal {signal}")
                    continue
                
                calculated_quantity = None # Initialize
                try:
                    current_price = latest_data['price']
                    logger.debug(f"[{symbol}] Attempting to calculate position size for signal {signal} at price {current_price}")
                    calculated_quantity = self.portfolio_manager.calculate_position_size(
                        symbol=symbol,
                        price=current_price
                    )
                    logger.debug(f"[{symbol}] Received quantity from PM: {calculated_quantity} (type: {type(calculated_quantity)})")
                except Exception as pm_error:
                    logger.error(f"[{symbol}] Error during PortfolioManager.calculate_position_size: {pm_error}", exc_info=True)
                    continue # Skip to next symbol if sizing fails

                try:
                    # Compare using Decimal for precision
                    decimal_threshold = Decimal('1e-8')
                    comparison_result = None
                    if calculated_quantity is not None:
                        comparison_result = calculated_quantity > decimal_threshold
                        logger.debug(
                            f"[{symbol}] Comparing: calculated_quantity ({calculated_quantity}, type: {type(calculated_quantity)}) > "
                            f"decimal_threshold ({decimal_threshold}, type: {type(decimal_threshold)}) = {comparison_result}"
                        )
                    # --------------------------------------
                    
                    # Use the pre-calculated comparison result
                    if calculated_quantity is not None and comparison_result:
                        logger.info(f"Calculated position size for {symbol}: {calculated_quantity}")
                        order = Order(
                            symbol=symbol,
                            side=OrderSide.BUY if signal == "BUY" else OrderSide.SELL,
                            type=OrderType.MARKET,
                            quantity=float(calculated_quantity) # Order expects float
                        )
                        logger.info(f"Generated Order: {order}")
                        self.orders_to_process.append(order) # Append to the instance list
                    else:
                        # Log if quantity is None or too small
                        if calculated_quantity is not None: # Only log value if not None
                            logger.info(f"Calculated quantity ({calculated_quantity}) is too small (< 1e-8) for {symbol}. No order created.")
                        else:
                            logger.info(f"Position size calculation resulted in None for {symbol}. No order created.")
                
                except Exception as sizing_error:
                    logger.error(f"[{symbol}] Error during position sizing or check: {sizing_error}", exc_info=True)
            # --------------------------------------

        # --- Process Orders through Portfolio Manager --- 
        logger.info(f"Generated signals: {signals}")
        if self.orders_to_process:
            logger.info(f"Submitting {len(self.orders_to_process)} orders for evaluation...")
            # In a real scenario, PM evaluates risk, position size etc.
            # For now, we assume PM approves and passes to execution
            for order in self.orders_to_process:
                # Simulate PM approval -> Execution 
                # TODO: Add proper PortfolioManager interaction 
                logger.info(f"Portfolio Manager approved order {order.order_id}. Sending to Execution Handler.")
                
                # --- Prepare Data for Execution Handler ---
                tick_data = self.latest_price_data.get(order.symbol)
                if not tick_data or 'price' not in tick_data or 'timestamp' not in tick_data:
                    logger.warning(f"Insufficient data for {order.symbol} to simulate execution. Skipping order {order.order_id}.")
                    continue

                price = tick_data['price']
                # Convert UNIX timestamp (seconds) to pandas Timestamp
                exec_timestamp = pd.Timestamp(tick_data['timestamp'], unit='s', tz='UTC') 
 
                # Create a dummy OHLCV bar (single-row DataFrame) using the tick price
                dummy_bar_df = pd.DataFrame({
                    'open': [price],
                    'high': [price],
                    'low': [price],
                    'close': [price],
                    'volume': [0] # Volume info not available from /quote endpoint
                }, index=[exec_timestamp])
                # -------------------------------------------

                try:
                    # Execute using the dummy bar
                    trades: list[Trade] = self.execution_handler.execute_order(
                        order=order,
                        market_data=dummy_bar_df, # Pass the DataFrame
                        timestamp=exec_timestamp
                    )
                except Exception as e:
                     logger.error(f"Error during execution simulation for order {order.order_id}: {e}", exc_info=True)
                     trades = [] # Ensure trades is defined

                if trades:
                     trade_event = trades[0] # Assuming one trade per order for now
                     logger.info(f"Received Fill Event: {trade_event}")
                     try:
                         self.portfolio_manager.update_from_trade(trade_event)
                         logger.info(f"PortfolioManager updated with trade: {trade_event.trade_id}")
                     except Exception as e:
                         logger.error(f"Error updating PortfolioManager for trade {trade_event.trade_id}: {e}", exc_info=True)
                else:
                      logger.warning(f"Order {order.order_id} did not result in a fill event (simulated). Status: {order.status}")
         # --------------------------------------------------

    async def run(self):
        """Starts the agent's main loop."""
        logger.info("Starting Trading Agent run loop...")
        
        # Start the Twelve Data client (will use REST polling on free plan)
        price_task = asyncio.create_task(
            self.twelve_data_client.connect_and_stream(self._handle_price_update)
        )

        # Start the sentiment fetching loop
        sentiment_task = asyncio.create_task(self._fetch_sentiment())

        # Keep agent running (tasks run in the background)
        try:
            await asyncio.gather(price_task, sentiment_task)
        except asyncio.CancelledError:
            logger.info("Agent run loop cancelled.")
        except Exception as e:
            logger.error(f"Error in agent run loop: {e}", exc_info=True)
        finally:
            logger.info("Stopping data clients...")
            await self.twelve_data_client.stop()
            # Sentiment task cancellation will stop its loop
            if sentiment_task and not sentiment_task.done():
                sentiment_task.cancel()
            logger.info("Trading Agent stopped.")

# Example of how to run the agent (can be moved to a main script later)
async def start_agent():
    # Use custom setup_logging to ensure DEBUG level is applied correctly
    setup_logging(log_level="DEBUG")
    
    # Define symbols to trade
    crypto_symbols = ["BTC/USD"]
    
    agent = TradingAgent(symbols=crypto_symbols, td_plan_level="free")
    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down agent...")
    except Exception as e:
        logger.error(f"Unhandled error running agent: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(start_agent())
