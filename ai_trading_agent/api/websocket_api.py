"""WebSocket API for real-time updates.

This module provides WebSocket endpoints for real-time updates from the AI Trading Agent,
including paper trading status, portfolio updates, and trade notifications.
"""

import json
import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock event emitter for development
class EventEmitter:
    def __init__(self):
        self.handlers = {}
    
    def subscribe_async(self, event, handler):
        if event not in self.handlers:
            self.handlers[event] = []
        self.handlers[event].append(handler)
    
    async def emit_async(self, event, data):
        if event in self.handlers:
            for handler in self.handlers[event]:
                await handler(data)

# Mock agent activity tracker for development
class AgentActivityTracker:
    def __init__(self):
        self.session_data = {}
    
    def get_session_activity_data(self, session_id):
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                "status": "active",
                "reasoning": "Market conditions are favorable for trading",
                "confidence": 0.85,
                "last_update": datetime.now().isoformat()
            }
        return self.session_data[session_id]

# Create instances
global_event_emitter = EventEmitter()
agent_activity_tracker = AgentActivityTracker()

# Store active connections
active_connections: Dict[str, List[WebSocket]] = {}

# Store active sessions
active_sessions: Set[str] = set()

# Mock data generation tasks
mock_data_tasks: Dict[str, asyncio.Task] = {}

# Create API router - updated for WebSocket endpoint
router = APIRouter()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time updates.
    
    Args:
        websocket: The WebSocket connection
        session_id: The session ID
    """
    logger.info(f"WebSocket connection attempt for session: {session_id}")
    
    # Accept the connection
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for session: {session_id}")
    
    # Register the connection with the manager
    if session_id not in active_connections:
        active_connections[session_id] = []
    active_connections[session_id].append(websocket)
    
    # Send connection established message
    await websocket.send_text(json.dumps({
        "type": "connection_established",
        "data": {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "topics": []
        }
    }))
    
    try:
        # Start mock data generation if it's not already running
        if session_id not in mock_data_tasks or mock_data_tasks[session_id].done():
            logger.info(f"Starting mock data generation for session: {session_id}")
            mock_data_tasks[session_id] = asyncio.create_task(generate_mock_performance_data(session_id))
        
        # Keep the connection alive until client disconnects
        while True:
            logger.info(f"Waiting for messages from client for session: {session_id}")
            data = await websocket.receive_text()
            logger.info(f"Received data from client: {data}")
            
            try:
                message = json.loads(data)
                # Handle subscription message
                if message.get('type') == 'subscribe' and message.get('topics'):
                    topics = message.get('topics', [])
                    logger.info(f"Client subscribed to topics: {topics}")
                    
                    # Send initial data for each topic
                    for topic in topics:
                        logger.info(f"Sending initial data for topic: {topic}")
                        if topic == 'performance':
                            # Send mock performance data
                            await websocket.send_text(json.dumps({
                                'type': 'update',
                                'topic': topic,
                                'data': {
                                    'session_id': session_id,
                                    'timestamp': datetime.now().isoformat(),
                                    'performance_metrics': {
                                        'total_return': 0.127,
                                        'annualized_return': 0.215,
                                        'sharpe_ratio': 1.85,
                                        'max_drawdown': -0.089,
                                        'win_rate': 0.68,
                                        'profit_factor': 2.3
                                    }
                                },
                                'timestamp': datetime.now().timestamp()
                            }))
                # Handle ping message
                elif message.get('type') == 'ping':
                    logger.info(f"Received ping, sending pong for session: {session_id}")
                    await websocket.send_text(json.dumps({
                        'type': 'pong',
                        'timestamp': datetime.now().timestamp()
                    }))
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {session_id}")
        if session_id in active_connections:
            active_connections[session_id].remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if session_id in active_connections and websocket in active_connections[session_id]:
            active_connections[session_id].remove(websocket)

# WebSocket topic enum
class WebSocketTopic:
    STATUS = 'status'
    PORTFOLIO = 'portfolio'
    TRADES = 'trades'
    ALERTS = 'alerts'
    PERFORMANCE = 'performance'
    AGENT_STATUS = 'agent_status'
    PAPER_TRADING = 'paper_trading'
    DRAWDOWN = 'drawdown'
    TRADE_STATS = 'trade_stats'

async def generate_mock_performance_data(session_id: str):
    """
    Generate mock performance data for a paper trading session.
    
    Args:
        session_id: The paper trading session ID
    """
    try:
        # Initial values
        equity = 10000.0
        timestamp = datetime.now() - timedelta(hours=24)  # Start 24 hours ago
        equity_curve = []
        drawdown_curve = []
        peak = equity
        
        # Generate 24 hours of data at 15-minute intervals
        for _ in range(96):  # 24 hours * 4 intervals per hour
            # Random price movement
            change_pct = random.uniform(-0.01, 0.012)  # Slight upward bias
            equity = equity * (1 + change_pct)
            
            # Update peak and calculate drawdown
            peak = max(peak, equity)
            drawdown = (equity - peak) / peak if peak > 0 else 0
            
            # Add to curves
            equity_curve.append({
                "timestamp": int(timestamp.timestamp() * 1000),
                "value": equity
            })
            drawdown_curve.append({
                "timestamp": int(timestamp.timestamp() * 1000),
                "value": drawdown
            })
            
            # Move to next interval
            timestamp += timedelta(minutes=15)
        
        # Calculate performance metrics
        initial_equity = equity_curve[0]["value"]
        final_equity = equity_curve[-1]["value"]
        total_return = (final_equity - initial_equity) / initial_equity
        max_drawdown = min([d["value"] for d in drawdown_curve])
        
        # Generate mock trades
        trades = []
        for i in range(10):
            entry_time = datetime.now() - timedelta(hours=random.randint(1, 24))
            exit_time = entry_time + timedelta(minutes=random.randint(15, 240))
            entry_price = random.uniform(30000, 32000) if i % 2 == 0 else random.uniform(1800, 2000)
            
            # Slightly more winning trades than losing
            is_win = random.random() > 0.4
            pnl_pct = random.uniform(0.01, 0.05) if is_win else -random.uniform(0.01, 0.03)
            exit_price = entry_price * (1 + pnl_pct)
            
            symbol = "BTC/USDT" if i % 2 == 0 else "ETH/USDT"
            quantity = 0.1 if symbol == "BTC/USDT" else 1.0
            
            trades.append({
                "id": f"trade-{i+1}",
                "symbol": symbol,
                "entry_time": int(entry_time.timestamp() * 1000),
                "exit_time": int(exit_time.timestamp() * 1000),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "pnl": (exit_price - entry_price) * quantity,
                "pnl_percent": pnl_pct,
                "duration": int((exit_time - entry_time).total_seconds()),
                "side": "buy",
                "status": "closed"
            })
        
        # Calculate trade statistics
        win_count = sum(1 for t in trades if t["pnl"] > 0)
        loss_count = sum(1 for t in trades if t["pnl"] < 0)
        win_rate = win_count / len(trades) if trades else 0
        
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] < 0]
        
        avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Create and send performance data
        while session_id in active_sessions:
            # Performance metrics
            performance_data = {
                "type": "update",
                "topic": "performance",
                "data": {
                    "metrics": {
                        "total_return": total_return,
                        "annualized_return": total_return * 365,  # Simple annualization
                        "sharpe_ratio": random.uniform(1.5, 2.5),
                        "max_drawdown": max_drawdown,
                        "win_rate": win_rate,
                        "profit_factor": abs(sum(t["pnl"] for t in winning_trades) / sum(t["pnl"] for t in losing_trades)) if losing_trades and sum(t["pnl"] for t in losing_trades) != 0 else 0,
                        "average_win": avg_win,
                        "average_loss": avg_loss,
                        "risk_reward_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                        "recovery_factor": abs(total_return / max_drawdown) if max_drawdown != 0 else 0,
                        "volatility": random.uniform(0.1, 0.2),
                        "sortino_ratio": random.uniform(2.0, 3.0),
                        "calmar_ratio": random.uniform(2.0, 3.0),
                        "max_consecutive_wins": random.randint(2, 5),
                        "max_consecutive_losses": random.randint(1, 3),
                        "profit_per_day": total_return / 1,  # 1 day of data
                        "current_drawdown": drawdown_curve[-1]["value"],
                        "drawdown_duration": random.randint(1, 5)
                    }
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            
            await send_paper_trading_update(session_id, performance_data)
            
            # Drawdown data
            drawdown_data = {
                "type": "update",
                "topic": "drawdown",
                "data": {
                    "drawdown_data": {
                        "timestamps": [d["timestamp"] for d in drawdown_curve],
                        "equity": [d["value"] for d in equity_curve],
                        "drawdown": [d["value"] for d in drawdown_curve]
                    }
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            await send_paper_trading_update(session_id, drawdown_data)
            
            # Trade statistics
            trade_stats = {
                "type": "update",
                "topic": "trade_stats",
                "data": {
                    "trade_statistics": {
                        "win_rate": win_rate,
                        "profit_factor": abs(sum(t["pnl"] for t in winning_trades) / sum(t["pnl"] for t in losing_trades)) if losing_trades and sum(t["pnl"] for t in losing_trades) != 0 else 0,
                        "average_win": avg_win,
                        "average_loss": avg_loss,
                        "risk_reward_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                        "max_consecutive_wins": random.randint(2, 5),
                        "max_consecutive_losses": random.randint(1, 3),
                        "average_duration": sum(t["duration"] for t in trades) / len(trades) if trades else 0
                    }
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            await send_paper_trading_update(session_id, trade_stats)
            
            # Trades
            trades_data = {
                "type": "update",
                "topic": "trades",
                "data": {
                    "trades": trades
                },
                "timestamp": int(datetime.now().timestamp() * 1000)
            }
            await send_paper_trading_update(session_id, trades_data)
            
            # Wait before sending next update
            await asyncio.sleep(5)  # Send updates every 5 seconds
    
    except asyncio.CancelledError:
        logger.info(f"Mock data generation task cancelled for session {session_id}")
    except Exception as e:
        logger.error(f"Error generating mock performance data: {e}", exc_info=True)
    finally:
        if session_id in active_sessions:
            active_sessions.remove(session_id)
        
async def send_paper_trading_update(session_id: str, data: Dict[str, Any]):
    """
    Send a paper trading update to all connected clients for a specific topic.
    
    Args:
        session_id: The session ID to send the update to
        data: The data to send
    """
    if session_id in active_connections:
        disconnected_ws = []
        for websocket in active_connections[session_id]:
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Error sending message to websocket: {e}")
                disconnected_ws.append(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected_ws:
            try:
                active_connections[session_id].remove(ws)
            except ValueError:
                pass


@router.websocket("/ws/{session_id}")
async def session_websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time updates.
    
    Args:
        websocket: The WebSocket connection
        session_id: The session ID
    """
    await websocket.accept()
    
    # Register connection
    if session_id not in active_connections:
        active_connections[session_id] = []
    
    active_connections[session_id].append(websocket)
    logger.info(f"WebSocket connection established for session {session_id}")
    
    # Add session to active sessions
    active_sessions.add(session_id)
    
    # Start mock data generation if not already running
    if session_id not in mock_data_tasks or mock_data_tasks[session_id].done():
        mock_data_tasks[session_id] = asyncio.create_task(generate_mock_performance_data(session_id))
    
    try:
        # Send initial data
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "WebSocket connection established"
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message from client: {data}")
            
            # Echo message back to client
            await websocket.send_json({
                "type": "echo",
                "message": f"Received: {data}"
            })
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Remove connection
        try:
            active_connections[session_id].remove(websocket)
        except (ValueError, KeyError):
            pass
        
        # Check if this was the last connection for this session
        if session_id in active_connections and not active_connections[session_id]:
            # Cancel mock data generation task
            if session_id in mock_data_tasks and not mock_data_tasks[session_id].done():
                mock_data_tasks[session_id].cancel()
            
            # Remove session from active sessions
            if session_id in active_sessions:
                active_sessions.remove(session_id)

    async def connect(self, websocket: WebSocket, client_id: str, topics: List[str]):
        await websocket.accept()
        
        # Store connection with topics
        for topic in topics:
            if topic not in self.active_connections:
                self.active_connections[topic] = []
            self.active_connections[topic].append(websocket)
        
        logger.info(f"Client {client_id} connected to WebSocket with topics: {topics}")
    
    def disconnect(self, websocket: WebSocket, topics: List[str]):
        # Remove connection from all topics
        for topic in topics:
            if topic in self.active_connections:
                if websocket in self.active_connections[topic]:
                    self.active_connections[topic].remove(websocket)
                
                # Clean up empty topics
                if not self.active_connections[topic]:
                    del self.active_connections[topic]
    
    async def broadcast(self, topic: str, message: Dict[str, Any]):
        """Broadcast a message to all clients subscribed to a topic."""
        if topic not in self.active_connections:
            return
            
        # Convert message to JSON
        json_message = json.dumps(message)
        
        # Send to all connections for this topic
        for connection in self.active_connections[topic]:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                # Don't disconnect here, will be handled by the connection handler

# Connection Manager class to handle WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, topics: List[str]):
        await websocket.accept()
        for topic in topics:
            if topic not in self.active_connections:
                self.active_connections[topic] = []
            self.active_connections[topic].append(websocket)
        logger.info(f"Client {client_id} connected to topics: {topics}")
    
    async def disconnect(self, websocket: WebSocket, topics: List[str]):
        for topic in topics:
            if topic in self.active_connections and websocket in self.active_connections[topic]:
                self.active_connections[topic].remove(websocket)
        logger.info(f"Client disconnected from topics: {topics}")
    
    async def broadcast(self, topic: str, message: Dict[str, Any]):
        if topic not in self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections[topic]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            if topic in self.active_connections and connection in self.active_connections[topic]:
                self.active_connections[topic].remove(connection)

# Create connection manager
manager = ConnectionManager()

# Set up event handlers for trading orchestrator events

# Handler for paper trading status updates
async def handle_paper_trading_status(data):
    await manager.broadcast('status', data)

# Handler for portfolio updates
async def handle_portfolio_update(data):
    await manager.broadcast('portfolio', data)

# Handler for trade updates
async def handle_order_execution(data):
    # Extract trade information from orders
    trades = []
    for order in data.get('orders', []):
        if isinstance(order, dict) and order.get('status') == 'FILLED':
            trades.append({
                'symbol': order.get('symbol'),
                'quantity': order.get('executed_quantity', order.get('quantity')),
                'price': order.get('executed_price', order.get('price')),
                'timestamp': data.get('timestamp'),
                'side': 'buy' if order.get('quantity', 0) > 0 else 'sell'
            })
    
    if trades:
        await manager.broadcast('trades', {'trades': trades})

# Handler for performance metrics updates
async def handle_performance_metrics(data):
    # Broadcast performance metrics
    await manager.broadcast('performance', data)
    
    # If drawdown data is available, broadcast it separately
    if 'drawdown_data' in data:
        drawdown_data = {
            'timestamp': data.get('timestamp'),
            'drawdown_data': data.get('drawdown_data')
        }
        await manager.broadcast('drawdown', drawdown_data)
    
    # If trade statistics are available, broadcast them separately
    if 'trade_statistics' in data:
        trade_stats = {
            'timestamp': data.get('timestamp'),
            'trade_statistics': data.get('trade_statistics')
        }
        await manager.broadcast('trade_stats', trade_stats)
        
    # If trade data is available, broadcast it separately
    if 'trades' in data:
        trades_data = {
            'timestamp': data.get('timestamp'),
            'trades': data.get('trades')
        }
        await manager.broadcast('trades', trades_data)

# Handler for market data updates
async def handle_market_data_update(data):
    # Only broadcast a summary of market data to avoid overwhelming clients
    summary = {
        'timestamp': data.get('timestamp'),
        'symbols': list(data.get('data', {}).keys()),
        'prices': {symbol: data.get('data', {}).get(symbol, {}).get('close', 0) 
                  for symbol in data.get('data', {}).keys()}
    }
    await manager.broadcast('market_data', summary)

# Handler for signal updates
async def handle_signal_update(data):
    # Only broadcast risk-adjusted signals
    signals = {
        'timestamp': data.get('timestamp'),
        'signals': data.get('risk_adjusted_signals', {})
    }
    await manager.broadcast('signals', signals)

# Handler for agent activity updates
async def handle_agent_activity_update(data):
    session_id = data.get('session_id')
    if not session_id:
        return
        
    # Get agent activity data for the session
    activity_data = agent_activity_tracker.get_session_activity_data(session_id)
    
    # Add session ID to the data
    activity_data['session_id'] = session_id
    
    # Broadcast to agent_status topic
    await manager.broadcast('agent_status', activity_data)

# Subscribe to events
global_event_emitter.subscribe_async('paper_trading_status', handle_paper_trading_status)
global_event_emitter.subscribe_async('portfolio_update', handle_portfolio_update)
global_event_emitter.subscribe_async('order_execution', handle_order_execution)
global_event_emitter.subscribe_async('performance_metrics', handle_performance_metrics)
global_event_emitter.subscribe_async('market_data_update', handle_market_data_update)
global_event_emitter.subscribe_async('signal_update', handle_signal_update)
global_event_emitter.subscribe_async('agent_activity_update', handle_agent_activity_update)

async def handle_websocket_connection(websocket: WebSocket, session_id: str):
    """
    Handle a WebSocket connection.
    
    Args:
        websocket: The WebSocket connection
        session_id: The paper trading session ID
    """
    await websocket.accept()
    
    # Register connection
    if session_id not in active_connections:
        active_connections[session_id] = []
    
    active_connections[session_id].append(websocket)
    logger.info(f"WebSocket connection established for session {session_id}")


async def send_paper_trading_update(topic: str, data: Dict[str, Any]):
    """
    Send a paper trading update to all subscribed clients.
    
    Args:
        topic: The topic to broadcast to
        data: The data to send
    """
    message = {
        "type": "update",
        "topic": topic,
        "data": data,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    await manager.broadcast(topic, message)
