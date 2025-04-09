# AI Trading Agent Dashboard

A modern, responsive dashboard for the AI Trading Agent system, providing portfolio management, trading capabilities, backtesting, and sentiment analysis visualization.

## Features

- **Authentication System**: Secure JWT-based authentication with login and registration
- **Real-time Data**: WebSocket integration for live portfolio and market updates (with mock data support for development)
- **Portfolio Management**: Track positions, performance, and asset allocation
- **Trading Interface**: Execute trades with various order types
- **Backtesting**: Configure and visualize strategy backtests
- **Sentiment Analysis**: View sentiment signals and their impact on trading decisions
- **Responsive Design**: Works across desktop, tablet, and mobile devices
- **Dark/Light Mode**: Theme customization for different environments

## Technology Stack

- **React**: Frontend library for building the user interface
- **TypeScript**: Type-safe JavaScript for better development experience
- **Tailwind CSS**: Utility-first CSS framework for styling
- **React Router**: Routing and navigation
- **Axios**: HTTP client for API requests
- **Lightweight Charts**: Financial charting library
- **React Query**: Data fetching and caching
- **WebSocket**: Real-time data updates with reconnection strategy and fallback to mock data

## Project Structure

```
src/
├── api/                # API client services 
├── assets/             # Static assets
├── components/         # Reusable components
│   ├── charts/         # Chart components
│   ├── dashboard/      # Dashboard widgets
│   ├── layout/         # Layout components
│   └── common/         # Common UI elements
├── context/            # React contexts
├── hooks/              # Custom hooks
├── pages/              # Page components
├── types/              # TypeScript types
└── utils/              # Utility functions
```

## Getting Started

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

### Prerequisites

- Node.js (v14+)
- npm or yarn

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file with the following variables:
   ```
   REACT_APP_API_URL=http://localhost:8000
   REACT_APP_WS_URL=ws://localhost:8000/ws/updates
   ```

### Available Scripts

In the project directory, you can run:

#### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

#### `npm test`

Launches the test runner in the interactive watch mode.

#### `npm run build`

Builds the app for production to the `build` folder.

### Mock Credentials

For testing purposes, you can use the following mock credentials:
- Email: `testuser@example.com` / Password: `Password123!`
- Or: Email: `admin` / Password: `admin123`

## Development Roadmap

- **Stage 1**: Foundation (Authentication, Layout, API Integration) - ✅ COMPLETED
- **Stage 2**: Dashboard Core (Portfolio Widgets, Charts) - ✅ COMPLETED
  - Basic dashboard widgets implemented
  - WebSocket integration with mock data support
  - Dark/light theme toggle
  - Enhanced UI components with better visualization and user experience
- **Stage 3**: Trading Tools (Order Management, Technical Analysis) - 🔄 IN PROGRESS
  - Order entry form with position information and validation
  - Asset allocation visualization with detailed metrics
  - Sentiment analysis dashboard with signal distribution
- **Stage 4**: Advanced Features (Strategy Builder, Backtest Visualization) - ⏱️ PLANNED
- **Stage 5**: Polish and Optimization (Performance, Responsive Design) - ⏱️ PLANNED

## Recent Updates

### v0.3.0 (April 2025)
- Enhanced OrderEntryForm with position information display and MAX/CURRENT buttons
- Improved AssetAllocationChart with better visualization and summary statistics
- Redesigned SentimentSummary with overall market sentiment indicator and signal distribution
- Fixed notification system by properly implementing NotificationProvider
- Added better null handling and loading states across all components
- Improved type safety throughout the application

### v0.2.0 (April 2025)
- Fixed WebSocket connection issues with reconnection strategy
- Added mock data support for development mode
- Fixed dark/light theme toggle functionality
- Improved error handling and TypeScript type safety
- Enhanced authentication flow with proper error handling

### v0.1.0 (March 2025)
- Initial project setup with React and TypeScript
- Implemented authentication system with JWT
- Created responsive layout with Tailwind CSS
- Set up basic dashboard structure

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).
