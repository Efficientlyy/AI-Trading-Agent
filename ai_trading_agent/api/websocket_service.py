"""
WebSocket Service for Real-time Trading Signals

This module provides a WebSocket service for streaming real-time trading signals
to connected clients. It uses FastAPI's WebSocket support and integrates with
the TwelveData WebSocket API for real-time market data.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Set, Any, Optional
import datetime
import uuid
from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Depends
import pandas as pd
import numpy as np

# Local imports
from ai_trading_agent.signal_generation.signal_integration_service import (
    SignalIntegrationService,
    TradingSignal,
    SignalType,
    SignalSource
)
from ai_trading_agent.data_collectors.twelve_data_client import TwelveDataClient
from ai_trading_agent.api.signal_endpoints import get_signal_service, get_data_client

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/ws",
    tags=["websocket"],
)


class ConnectionManager:
    """
    WebSocket connection manager for handling multiple client connections
    """
    
    def __init__(self):
        """Initialize the connection manager"""
        self.active_connections: List[WebSocket] = []
        self.connection_symbols: Dict[WebSocket, Set[str]] = {}
        self.symbol_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, symbols: List[str] = None):
        """
        Connect a new WebSocket client
        
        Args:
            websocket: The WebSocket connection
            symbols: List of symbols the client is interested in
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Initialize symbols for this connection
        if symbols:
            self.connection_symbols[websocket] = set(symbols)
            
            # Add connection to each symbol's set
            for symbol in symbols:
                if symbol not in self.symbol_connections:
                    self.symbol_connections[symbol] = set()
                self.symbol_connections[symbol].add(websocket)
        else:
            self.connection_symbols[websocket] = set()
    
    def disconnect(self, websocket: WebSocket):
        """
        Disconnect a WebSocket client
        
        Args:
            websocket: The WebSocket connection to disconnect
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
            # Remove from symbol mappings
            if websocket in self.connection_symbols:
                symbols = self.connection_symbols[websocket]
                del self.connection_symbols[websocket]
                
                # Remove from each symbol's set
                for symbol in symbols:
                    if symbol in self.symbol_connections and websocket in self.symbol_connections[symbol]:
                        self.symbol_connections[symbol].remove(websocket)
                        
                        # Clean up empty sets
                        if not self.symbol_connections[symbol]:
                            del self.symbol_connections[symbol]
    
    async def subscribe(self, websocket: WebSocket, symbols: List[str]):
        """
        Subscribe a client to symbols
        
        Args:
            websocket: The WebSocket connection
            symbols: List of symbols to subscribe to
        """
        if websocket not in self.connection_symbols:
            self.connection_symbols[websocket] = set()
        
        # Add new symbols
        for symbol in symbols:
            self.connection_symbols[websocket].add(symbol)
            
            if symbol not in self.symbol_connections:
                self.symbol_connections[symbol] = set()
            self.symbol_connections[symbol].add(websocket)
        
        # Send confirmation
        await websocket.send_json({
            "type": "subscription",
            "status": "success",
            "symbols": list(self.connection_symbols[websocket])
        })
    
    async def unsubscribe(self, websocket: WebSocket, symbols: List[str]):
        """
        Unsubscribe a client from symbols
        
        Args:
            websocket: The WebSocket connection
            symbols: List of symbols to unsubscribe from
        """
        if websocket in self.connection_symbols:
            # Remove symbols
            for symbol in symbols:
                if symbol in self.connection_symbols[websocket]:
                    self.connection_symbols[websocket].remove(symbol)
                
                if symbol in self.symbol_connections and websocket in self.symbol_connections[symbol]:
                    self.symbol_connections[symbol].remove(websocket)
                    
                    # Clean up empty sets
                    if not self.symbol_connections[symbol]:
                        del self.symbol_connections[symbol]
            
            # Send confirmation
            await websocket.send_json({
                "type": "unsubscription",
                "status": "success",
                "symbols": list(self.connection_symbols[websocket])
            })
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all connected clients
        
        Args:
            message: The message to broadcast
        """
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")
    
    async def broadcast_to_symbol(self, symbol: str, message: Dict[str, Any]):
        """
        Broadcast a message to clients subscribed to a symbol
        
        Args:
            symbol: The symbol to broadcast to
            message: The message to broadcast
        """
        if symbol in self.symbol_connections:
            for connection in self.symbol_connections[symbol]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting message to {symbol}: {str(e)}")
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Send a message to a specific client
        
        Args:
            websocket: The WebSocket connection
            message: The message to send
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")


# Create connection manager
manager = ConnectionManager()


class SignalGenerator:
    """
    Signal generator for creating real-time trading signals
    """
    
    def __init__(
        self,
        signal_service: SignalIntegrationService,
        data_client: TwelveDataClient,
        connection_manager: ConnectionManager
    ):
        """
        Initialize the signal generator
        
        Args:
            signal_service: The signal integration service
            data_client: The data client for fetching price data
            connection_manager: The WebSocket connection manager
        """
        self.signal_service = signal_service
        self.data_client = data_client
        self.connection_manager = connection_manager
        
        # Price data cache
        self.price_data_cache: Dict[str, pd.DataFrame] = {}
        
        # Running flag
        self.is_running = False
        
        # Task references
        self.tasks = []
    
    async def start(self, interval_seconds: int = 60):
        """
        Start the signal generator
        
        Args:
            interval_seconds: Interval between signal checks in seconds
        """
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start the main loop
        main_task = asyncio.create_task(self._main_loop(interval_seconds))
        self.tasks.append(main_task)
        
        logger.info(f"Signal generator started with interval {interval_seconds} seconds")
    
    async def stop(self):
        """Stop the signal generator"""
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        self.tasks = []
        
        logger.info("Signal generator stopped")
    
    async def _main_loop(self, interval_seconds: int):
        """
        Main loop for generating signals
        
        Args:
            interval_seconds: Interval between signal checks in seconds
        """
        try:
            while self.is_running:
                # Get all symbols with active subscriptions
                active_symbols = list(self.connection_manager.symbol_connections.keys())
                
                if active_symbols:
                    # Update price data for active symbols
                    await self._update_price_data(active_symbols)
                    
                    # Generate signals for active symbols
                    await self._generate_and_broadcast_signals(active_symbols)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
        
        except asyncio.CancelledError:
            logger.info("Signal generator main loop cancelled")
        
        except Exception as e:
            logger.error(f"Error in signal generator main loop: {str(e)}")
            
            # Restart the loop
            if self.is_running:
                logger.info("Restarting signal generator main loop")
                main_task = asyncio.create_task(self._main_loop(interval_seconds))
                self.tasks.append(main_task)
    
    async def _update_price_data(self, symbols: List[str]):
        """
        Update price data for symbols
        
        Args:
            symbols: List of symbols to update
        """
        try:
            # Calculate start and end dates
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=30)  # Last 30 days
            
            # Update price data for each symbol
            for symbol in symbols:
                try:
                    # Fetch historical price data
                    price_data = await self.data_client.get_historical_prices(
                        symbol=symbol,
                        interval="1d",  # Daily data for signals
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(price_data)
                    
                    # Update cache
                    self.price_data_cache[symbol] = df
                
                except Exception as e:
                    logger.error(f"Error updating price data for {symbol}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error updating price data: {str(e)}")
    
    async def _generate_and_broadcast_signals(self, symbols: List[str]):
        """
        Generate and broadcast signals for symbols
        
        Args:
            symbols: List of symbols to generate signals for
        """
        try:
            # Create price data dictionary
            price_data_dict = {
                symbol: df for symbol, df in self.price_data_cache.items()
                if symbol in symbols
            }
            
            # Skip if no price data
            if not price_data_dict:
                return
            
            # Get signals for all symbols
            signals_dict = await self.signal_service.get_signals_for_multiple_symbols(
                symbols=list(price_data_dict.keys()),
                price_data_dict=price_data_dict,
                force_refresh=True  # Always get fresh signals
            )
            
            # Broadcast signals for each symbol
            for symbol, signals in signals_dict.items():
                # Combine all signals
                all_signals = []
                all_signals.extend(signals["technical"])
                all_signals.extend(signals["sentiment"])
                all_signals.extend(signals["combined"])
                
                # Skip if no signals
                if not all_signals:
                    continue
                
                # Sort by timestamp (newest first)
                all_signals.sort(
                    key=lambda s: datetime.datetime.fromisoformat(s["timestamp"]),
                    reverse=True
                )
                
                # Broadcast each signal
                for signal in all_signals:
                    # Create message
                    message = {
                        "type": "signal",
                        "data": signal
                    }
                    
                    # Broadcast to subscribers
                    await self.connection_manager.broadcast_to_symbol(symbol, message)
        
        except Exception as e:
            logger.error(f"Error generating and broadcasting signals: {str(e)}")


# Create signal generator
_signal_generator = None


def get_signal_generator(
    signal_service: SignalIntegrationService = Depends(get_signal_service),
    data_client: TwelveDataClient = Depends(get_data_client)
) -> SignalGenerator:
    """
    Get or create a SignalGenerator instance
    
    Args:
        signal_service: The signal integration service
        data_client: The data client for fetching price data
        
    Returns:
        SignalGenerator: The signal generator instance
    """
    global _signal_generator
    if _signal_generator is None:
        _signal_generator = SignalGenerator(
            signal_service=signal_service,
            data_client=data_client,
            connection_manager=manager
        )
    
    return _signal_generator


@router.websocket("/signals")
async def websocket_signals(
    websocket: WebSocket,
    signal_generator: SignalGenerator = Depends(get_signal_generator)
):
    """
    WebSocket endpoint for trading signals
    
    Args:
        websocket: The WebSocket connection
        signal_generator: The signal generator
    """
    # Accept connection
    await manager.connect(websocket)
    
    # Start signal generator if not running
    if not signal_generator.is_running:
        await signal_generator.start()
    
    try:
        # Send welcome message
        await manager.send_personal_message(
            websocket,
            {
                "type": "welcome",
                "message": "Connected to trading signals WebSocket",
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
        
        # Process messages
        while True:
            # Wait for message
            data = await websocket.receive_text()
            
            # Parse message
            try:
                message = json.loads(data)
                
                # Process message
                if "action" in message:
                    action = message["action"]
                    
                    if action == "subscribe" and "symbols" in message:
                        # Subscribe to symbols
                        symbols = message["symbols"]
                        await manager.subscribe(websocket, symbols)
                    
                    elif action == "unsubscribe" and "symbols" in message:
                        # Unsubscribe from symbols
                        symbols = message["symbols"]
                        await manager.unsubscribe(websocket, symbols)
                    
                    elif action == "ping":
                        # Respond to ping
                        await manager.send_personal_message(
                            websocket,
                            {
                                "type": "pong",
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                        )
                    
                    else:
                        # Unknown action
                        await manager.send_personal_message(
                            websocket,
                            {
                                "type": "error",
                                "message": f"Unknown action: {action}",
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                        )
                
                else:
                    # Invalid message
                    await manager.send_personal_message(
                        websocket,
                        {
                            "type": "error",
                            "message": "Invalid message format",
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
            
            except json.JSONDecodeError:
                # Invalid JSON
                await manager.send_personal_message(
                    websocket,
                    {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
            
            except Exception as e:
                # Other error
                logger.error(f"Error processing WebSocket message: {str(e)}")
                await manager.send_personal_message(
                    websocket,
                    {
                        "type": "error",
                        "message": f"Error processing message: {str(e)}",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                )
    
    except WebSocketDisconnect:
        # Client disconnected
        manager.disconnect(websocket)
    
    except Exception as e:
        # Other error
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)


@router.websocket("/market")
async def websocket_market(
    websocket: WebSocket,
    data_client: TwelveDataClient = Depends(get_data_client)
):
    """
    WebSocket endpoint for real-time market data
    
    Args:
        websocket: The WebSocket connection
        data_client: The data client for WebSocket connection
    """
    # Accept connection
    await websocket.accept()
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to market data WebSocket",
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Process messages
        while True:
            # Wait for message
            data = await websocket.receive_text()
            
            # Parse message
            try:
                message = json.loads(data)
                
                # Process message
                if "action" in message:
                    action = message["action"]
                    
                    if action == "subscribe" and "symbols" in message:
                        # Subscribe to market data
                        symbols = message["symbols"]
                        
                        # Forward to Twelve Data WebSocket (in a real implementation)
                        # For now, just acknowledge
                        await websocket.send_json({
                            "type": "subscription",
                            "status": "success",
                            "symbols": symbols,
                            "message": "Subscribed to market data (mock)",
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        
                        # Start sending mock data
                        asyncio.create_task(send_mock_market_data(websocket, symbols))
                    
                    elif action == "ping":
                        # Respond to ping
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                    
                    else:
                        # Unknown action
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Unknown action: {action}",
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                
                else:
                    # Invalid message
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid message format",
                        "timestamp": datetime.datetime.now().isoformat()
                    })
            
            except json.JSONDecodeError:
                # Invalid JSON
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.datetime.now().isoformat()
                })
            
            except Exception as e:
                # Other error
                logger.error(f"Error processing WebSocket message: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}",
                    "timestamp": datetime.datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        # Client disconnected
        pass
    
    except Exception as e:
        # Other error
        logger.error(f"WebSocket error: {str(e)}")


async def send_mock_market_data(websocket: WebSocket, symbols: List[str]):
    """
    Send mock market data for testing
    
    Args:
        websocket: The WebSocket connection
        symbols: List of symbols to send data for
    """
    try:
        # Initial prices
        prices = {
            "BTC/USDT": 50000.0,
            "ETH/USDT": 3000.0,
            "XRP/USDT": 0.5,
            "ADA/USDT": 1.2,
            "SOL/USDT": 100.0,
            "DOGE/USDT": 0.1,
            "DOT/USDT": 20.0,
            "LINK/USDT": 15.0,
            "UNI/USDT": 10.0,
            "AAVE/USDT": 200.0
        }
        
        # Send data every second
        while True:
            for symbol in symbols:
                # Skip unknown symbols
                if symbol not in prices:
                    continue
                
                # Get base price
                base_price = prices[symbol]
                
                # Add random movement
                change_pct = np.random.normal(0, 0.002)  # 0.2% standard deviation
                new_price = base_price * (1 + change_pct)
                
                # Update base price
                prices[symbol] = new_price
                
                # Create message
                message = {
                    "type": "price",
                    "symbol": symbol,
                    "price": round(new_price, 8),
                    "change_pct": round(change_pct * 100, 4),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # Send message
                await websocket.send_json(message)
            
            # Wait before next update
            await asyncio.sleep(1)
    
    except WebSocketDisconnect:
        # Client disconnected
        pass
    
    except asyncio.CancelledError:
        # Task cancelled
        pass
    
    except Exception as e:
        # Other error
        logger.error(f"Error sending mock market data: {str(e)}")


# Register startup and shutdown events
async def startup_event():
    """Startup event handler"""
    # Start signal generator
    signal_generator = get_signal_generator()
    await signal_generator.start()


async def shutdown_event():
    """Shutdown event handler"""
    # Stop signal generator
    if _signal_generator is not None:
        await _signal_generator.stop()
