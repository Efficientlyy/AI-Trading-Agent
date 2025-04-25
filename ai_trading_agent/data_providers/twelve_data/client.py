# WebSocket client for Twelve Data API

import asyncio
import json
import logging
from twelvedata import TDClient  # Import the SDK client
import pandas as pd

from ai_trading_agent.config import TWELVEDATA_API_KEY

logger = logging.getLogger(__name__)

class TwelveDataClient:
    """Client for handling Twelve Data connections, supporting WebSocket (Pro+) or REST polling (Free).

    Args:
        api_key (str): Your Twelve Data API key.
        symbols (list[str]): List of symbols to fetch data for (e.g., ["BTC/USD", "ETH/USD"]).
        plan_level (str): Your Twelve Data plan level ('free' or 'pro'). Defaults to 'free'.
                           WebSocket streaming requires a 'pro' plan.
    """
    def __init__(self, api_key: str, symbols: list[str], plan_level: str = "free"):
        if not api_key:
            raise ValueError("Twelve Data API key is required.")
        if plan_level not in ["free", "pro"]:
            raise ValueError("plan_level must be 'free' or 'pro'")
        self.api_key = api_key
        self.symbols = symbols
        self.plan_level = plan_level.lower()
        self.td = TDClient(apikey=self.api_key)
        self.stream = None
        self._is_running = False
        self.user_callback = None

    def _on_event(self, event):
        """Internal callback to handle events from the SDK stream."""
        # The SDK passes the event dictionary directly
        # logger.debug(f"Received event: {event}") # Log raw event if needed

        event_type = event.get('event')

        if event_type == 'price':
            # Pass the price data to the user-defined callback
            if self.user_callback:
                asyncio.create_task(self.user_callback(event))
            else:
                logger.warning("Received price event but no user callback is set.")
        elif event_type == 'subscribe-status':
            status = event.get('status', 'unknown')
            if status == 'ok':
                subs = event.get('success', [])
                fails = event.get('fails', [])
                logger.info(f"Subscription status OK. Success: {len(subs)}, Fails: {len(fails)}")
                if fails:
                    logger.warning(f"Failed subscriptions: {fails}")
            else:
                logger.error(f"Subscription status error: {event}")
        elif event_type == 'heartbeat':
            logger.debug("Received heartbeat")
        elif event_type == 'error':
            logger.error(f"Received error from Twelve Data stream: {event.get('message', 'Unknown error')}")
        else:
            logger.warning(f"Received unhandled event type: {event_type}, Data: {event}")

    async def connect_and_stream(self, callback):
        """Connects to the appropriate data stream based on plan level.

        Args:
            callback: An async function to be called with each received price event.
        """
        if self._is_running:
            logger.warning("Client is already running.")
            return

        self.user_callback = callback
        self._is_running = True

        if self.plan_level == "pro":
            logger.info(f"Plan level '{self.plan_level}'. Attempting WebSocket connection for symbols: {self.symbols}")
            await self._connect_websocket()
        else: # Default to REST polling for 'free' plan
            logger.info(f"Plan level '{self.plan_level}'. Starting REST API polling for symbols: {self.symbols}")
            await self._poll_data_rest()

    async def _connect_websocket(self):
        """Handles the WebSocket connection and streaming logic (for Pro plan)."""
        try:
            # Attempt to create the stream object
            logger.info("Attempting to create WebSocket stream object...")
            self.stream = self.td.websocket(on_event=self._on_event)

            # Add a small delay to allow for potential internal SDK initialization
            await asyncio.sleep(0.1)

            if not self.stream:
                logger.error("Failed to create WebSocket stream object from SDK after delay. Check API key permissions?")
                self._is_running = False
                return # Exit if stream object creation failed

            # Attempt to connect
            logger.info("Attempting to connect WebSocket stream...")
            await self.stream.connect()
            logger.info("WebSocket stream connected successfully.")

            # Subscribe to symbols
            logger.info(f"Attempting to subscribe to symbols: {self.symbols}")
            await self.stream.subscribe(self.symbols)

            # Keep the stream running until stop() is called
            await self.stream.keep_alive()

        except asyncio.CancelledError:
            logger.info("Stream task cancelled.")
        except Exception as e:
            # Log the exception specifically where it occurs
            logger.error(f"Error during WebSocket setup or streaming: {e}", exc_info=True)
            self._is_running = False # Ensure state is reset on error
            # Optionally re-raise or handle specific exceptions (e.g., auth errors)
            # For now, we just log and stop, assuming WS is unavailable/error occurred
        finally:
            logger.info("WebSocket Stream stopped.")
            self._is_running = False
            # Attempt to clean up stream if it exists
            if self.stream and hasattr(self.stream, 'disconnect'):
                try:
                    await self.stream.disconnect()
                    logger.info("WebSocket stream disconnected.")
                except Exception as disc_e:
                    logger.error(f"Error disconnecting stream: {disc_e}")
            self.stream = None # Clear stream object

    async def _poll_data_rest(self):
        """Handles polling data using the REST API (for Free plan)."""
        polling_interval_seconds = 90 # Increased interval to reduce API calls
        logger.info(f"Starting REST polling loop with interval: {polling_interval_seconds} seconds.")
        
        while self._is_running:
            try:
                logger.debug(f"Polling REST API for symbols: {self.symbols}")
                results = {} # Store results for this poll cycle
                for symbol in self.symbols:
                    if not self._is_running: break # Exit loop if stopped
                    try:
                        # --- MOCK DATA IMPLEMENTATION --- 
                        logger.warning(f"!!! Using MOCK price data for {symbol} due to API limits !!!")
                        mock_price = 65000.0 # Example mock price
                        mock_timestamp = int(pd.Timestamp.utcnow().timestamp())
                        formatted_event = {
                            'event': 'price',
                            'symbol': symbol,
                            'price': mock_price, 
                            'timestamp': mock_timestamp
                        }
                        results[symbol] = formatted_event
                        logger.debug(f"MOCK Quote for {symbol}: {formatted_event}")
                        # ----------------------------------
                    except Exception as symbol_e: # Keep general exception handling for potential mock data issues
                        # Log the specific error for the symbol with traceback
                        logger.error(f"Error fetching REST quote for symbol {symbol}: {symbol_e}", exc_info=True)
                        results[symbol] = None # Mark as failed for this cycle

                # Process all collected results from this poll cycle
                for symbol, event_data in results.items():
                    if event_data and self.user_callback:
                        try:
                            # Create task for the callback
                            asyncio.create_task(self.user_callback(event_data))
                        except Exception as callback_e:
                            logger.error(f"Error creating/running callback task for {symbol}: {callback_e}", exc_info=True)
                
                # Wait for the next polling interval
                if self._is_running:
                    logger.debug(f"Finished polling cycle. Waiting {polling_interval_seconds} seconds...")
                    await asyncio.sleep(polling_interval_seconds)

            except asyncio.CancelledError:
                logger.info("REST polling task cancelled.")
                break # Exit loop
            except Exception as e:
                logger.error(f"Error during REST polling loop: {e}", exc_info=True)
                # Decide on error handling: continue, break, wait longer?
                await asyncio.sleep(polling_interval_seconds) # Wait before retrying
        
        logger.info("REST polling loop stopped.")
        self._is_running = False # Ensure state is false on exit

    async def stop(self):
        """Stops the WebSocket stream or REST polling loop."""
        if self._is_running:
            logger.info("Attempting to stop Twelve Data client...")
            self._is_running = False # Signal loops to stop
            # If WebSocket stream exists and has a disconnect method, use it
            if self.stream and hasattr(self.stream, 'disconnect'):
                 try:
                      await self.stream.disconnect()
                      logger.info("WebSocket stream explicitly disconnected during stop.")
                 except Exception as e:
                      logger.error(f"Error trying to disconnect stream during stop: {e}")
            self.stream = None # Clear stream object regardless
        else:
            logger.info("Client is not running or already stopped.")

async def handle_price_update(data):
    """Callback function to process incoming price data."""
    symbol = data.get('symbol', 'N/A')
    price = data.get('price', 'N/A') # Should work for both WS and formatted REST
    timestamp = data.get('timestamp', 'N/A') # Consider converting timestamp
    print(f"Callback received: {symbol} - {price} at {timestamp}")
    # Add logic here to process the data (e.g., update UI, feed to agent)

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    if not TWELVEDATA_API_KEY:
        logger.error("TWELVEDATA_API_KEY not found in environment variables. Exiting.")
        return

    symbols_to_subscribe = ["BTC/USD", "ETH/USD", "SOL/USD"]
    # Instantiate with explicit plan_level for clarity
    client = TwelveDataClient(
        api_key=TWELVEDATA_API_KEY, 
        symbols=symbols_to_subscribe,
        plan_level="free" # Explicitly set to use REST polling
    )
    stream_task = None

    try:
        # Run the streaming client in a separate task
        # This will eventually call either WebSocket or REST polling
        stream_task = asyncio.create_task(client.connect_and_stream(handle_price_update))
        await stream_task # Wait for the task to complete (or be cancelled)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping client...")
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}", exc_info=True)
    finally:
        if stream_task and not stream_task.done():
            stream_task.cancel()
            try:
                await stream_task # Wait for cancellation to complete
            except asyncio.CancelledError:
                logger.info("Client task successfully cancelled.")
        # Explicitly stop client resources
        await client.stop()
        logger.info("Exited main function.")

if __name__ == "__main__":
    asyncio.run(main())
