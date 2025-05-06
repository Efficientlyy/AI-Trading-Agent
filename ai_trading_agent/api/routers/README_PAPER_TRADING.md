# Paper Trading Implementation

This document provides an overview of the paper trading implementation in the AI Trading Agent system.

## Overview

The paper trading system provides a realistic simulation environment for testing trading strategies with real-time market data without risking real capital. It uses the full implementation in `paper_trading.py` which integrates with the `LiveTradingBridge` for realistic execution simulation.

## Key Components

1. **Paper Trading Router** (`paper_trading.py`):
   - Provides comprehensive API endpoints for session management, order placement, and position tracking
   - Integrates with the `LiveTradingBridge` for realistic execution
   - Uses pickle serialization for session persistence

2. **Live Trading Bridge** (`trading_engine/live_trading_bridge.py`):
   - Serves as the interface between signal generation and execution
   - Supports both paper trading (simulated) and live trading modes
   - Implements realistic market conditions including slippage, partial fills, and execution delays

3. **Session Management**:
   - Sessions are stored in `paper_trading_sessions.pkl`
   - Each session has isolated state (portfolio, orders, trading history)
   - Sessions can be started, monitored, and stopped independently

## Session Clearing

To clear all paper trading sessions, use the `clear_sessions.py` script at the project root. This script:
- Clears the main sessions file (`paper_trading_sessions.pkl`)
- Creates a backup of any simplified session files before clearing them
- Ensures consistency across the system

## API Endpoints

The paper trading API is accessible under the `/api/paper-trading` prefix and includes:

- `POST /api/paper-trading/sessions`: Create a new paper trading session
- `GET /api/paper-trading/sessions`: List all paper trading sessions
- `GET /api/paper-trading/sessions/{session_id}`: Get details of a specific session
- `POST /api/paper-trading/sessions/{session_id}/stop`: Stop a paper trading session
- `GET /api/paper-trading/{session_id}/positions`: Get positions for a session
- `GET /api/paper-trading/{session_id}/orders`: Get orders for a session
- `POST /api/paper-trading/{session_id}/orders`: Place an order

## Important Notes

1. After clearing sessions, restart the backend server for changes to take effect.
2. The system uses the full paper trading implementation for advanced features and realistic simulation.
3. Session data is persisted across server restarts.
