# MEXC Dashboard Integration

## Overview
This directory contains components for the MEXC trading dashboard integrated into the AI Trading Agent application. The dashboard provides a real-time view of cryptocurrency trading data from the MEXC exchange.

## Components

### Main Components
- `SimpleMexcDashboard.tsx`: The main container component that integrates all dashboard elements
- `TradingViewChart.tsx`: Chart component for displaying price history using TradingView
- `OrderBook.tsx`: Displays the current order book with bids and asks
- `MarketTrades.tsx`: Shows recent trades for the selected trading pair
- `TradingPanel.tsx`: Interface for placing buy/sell orders
- `PriceTickerBar.tsx`: Horizontal bar displaying price tickers for multiple coins

### API and Data Integration
The dashboard connects to the MEXC exchange API to fetch real-time data:
- REST API for initial data loading
- WebSocket connections for real-time updates
- Error handling and fallback mechanisms

## Usage

### Adding to Routes
The dashboard is accessible at the `/mexc-dashboard` route and can be integrated into any React application:

```jsx
<Route path="/mexc-dashboard" element={<SimpleMexcDashboardPage />} />
```

### Configuration
Configuration settings can be modified in:
- `api/mexc/config.ts`: API endpoints, supported pairs, and timeframes

### Data Hooks
Several custom React hooks are available for accessing MEXC data:
- `useMexcData`: Provides combined data for a specific trading pair
- `useMexcTickers`: Provides ticker data for multiple trading pairs

## Features
- Real-time price updates
- Order book visualization
- Trade history
- Trading pair selection
- Timeframe selection
- TradingView chart integration

## Error Handling
The dashboard includes comprehensive error handling:
- Connection status indicators
- Fallback data when API fails
- Automatic reconnection for WebSockets
- User-friendly error messages