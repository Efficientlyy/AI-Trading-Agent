# Environment Configuration Guide

This document provides detailed instructions for configuring the AI Trading Agent's environment for development, testing, and production deployments.

## Table of Contents

1. [Overview](#overview)
2. [Environment Variables](#environment-variables)
3. [Development Environment](#development-environment)
4. [Testing Environment](#testing-environment)
5. [Production Environment](#production-environment)
6. [Docker Configuration](#docker-configuration)
7. [CI/CD Environment Setup](#cicd-environment-setup)

## Overview

The AI Trading Agent uses environment variables for configuration to:
- Keep sensitive information out of the codebase
- Allow for different configurations across environments
- Enable easy deployment to various hosting platforms
- Support containerization and orchestration

## Environment Variables

### Core Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NODE_ENV` | Environment mode | `development`, `test`, `production` |
| `REACT_APP_API_URL` | Backend API URL | `http://localhost:8000` |
| `REACT_APP_WEBSOCKET_URL` | WebSocket server URL | `ws://localhost:8000/ws` |
| `REACT_APP_DEFAULT_EXCHANGE` | Default trading exchange | `binance` |

### API Keys and Secrets

| Variable | Description |
|----------|-------------|
| `REACT_APP_BINANCE_API_KEY` | Binance API key |
| `REACT_APP_BINANCE_API_SECRET` | Binance API secret |
| `REACT_APP_COINBASE_API_KEY` | Coinbase API key |
| `REACT_APP_COINBASE_API_SECRET` | Coinbase API secret |
| `REACT_APP_COINBASE_PASSPHRASE` | Coinbase API passphrase |
| `REACT_APP_ALPACA_API_KEY` | Alpaca API key |
| `REACT_APP_ALPACA_API_SECRET` | Alpaca API secret |

### Feature Flags

| Variable | Description | Default |
|----------|-------------|---------|
| `REACT_APP_ENABLE_PAPER_TRADING` | Enable paper trading mode | `true` |
| `REACT_APP_ENABLE_CIRCUIT_BREAKER` | Enable circuit breaker pattern | `true` |
| `REACT_APP_ENABLE_PERFORMANCE_MONITORING` | Enable performance monitoring | `true` |
| `REACT_APP_ENABLE_ADVANCED_CHARTS` | Enable advanced chart features | `false` |

### Performance Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `REACT_APP_MAX_CACHE_SIZE` | Maximum cache size for memoization | `100` |
| `REACT_APP_CACHE_TTL_MS` | Cache time-to-live in milliseconds | `60000` |
| `REACT_APP_BATCH_SIZE` | Batch size for API calls | `10` |
| `REACT_APP_BATCH_DELAY_MS` | Batch delay in milliseconds | `100` |

## Development Environment

### Local Setup

1. Create a `.env.development` file in the root of the frontend directory:

```
# Development Environment Configuration
NODE_ENV=development
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WEBSOCKET_URL=ws://localhost:8000/ws

# Feature Flags
REACT_APP_ENABLE_PAPER_TRADING=true
REACT_APP_ENABLE_CIRCUIT_BREAKER=true
REACT_APP_ENABLE_PERFORMANCE_MONITORING=true
REACT_APP_ENABLE_ADVANCED_CHARTS=true

# Mock API Keys (for development only)
REACT_APP_BINANCE_API_KEY=development_key
REACT_APP_BINANCE_API_SECRET=development_secret
REACT_APP_COINBASE_API_KEY=development_key
REACT_APP_COINBASE_API_SECRET=development_secret
REACT_APP_COINBASE_PASSPHRASE=development_passphrase
REACT_APP_ALPACA_API_KEY=development_key
REACT_APP_ALPACA_API_SECRET=development_secret

# Performance Configuration
REACT_APP_MAX_CACHE_SIZE=100
REACT_APP_CACHE_TTL_MS=60000
REACT_APP_BATCH_SIZE=10
REACT_APP_BATCH_DELAY_MS=100
```

2. Start the development server:

```bash
npm start
```

### Mock API Mode

For development without connecting to real exchanges:

1. Set the following in your `.env.development` file:

```
REACT_APP_USE_MOCK_API=true
REACT_APP_MOCK_DATA_DELAY_MS=500
```

2. This will use mock data providers instead of real API calls

## Testing Environment

### Test Configuration

1. Create a `.env.test` file in the root of the frontend directory:

```
# Test Environment Configuration
NODE_ENV=test
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WEBSOCKET_URL=ws://localhost:8000/ws

# Always use mock APIs for tests
REACT_APP_USE_MOCK_API=true
REACT_APP_MOCK_DATA_DELAY_MS=10

# Feature Flags
REACT_APP_ENABLE_PAPER_TRADING=true
REACT_APP_ENABLE_CIRCUIT_BREAKER=true
REACT_APP_ENABLE_PERFORMANCE_MONITORING=false

# Test API Keys
REACT_APP_BINANCE_API_KEY=test_key
REACT_APP_BINANCE_API_SECRET=test_secret
REACT_APP_COINBASE_API_KEY=test_key
REACT_APP_COINBASE_API_SECRET=test_secret
REACT_APP_COINBASE_PASSPHRASE=test_passphrase
REACT_APP_ALPACA_API_KEY=test_key
REACT_APP_ALPACA_API_SECRET=test_secret
```

2. Run tests with the test environment:

```bash
npm test
```

### End-to-End Testing Configuration

For Cypress E2E tests, create a `cypress.env.json` file:

```json
{
  "apiUrl": "http://localhost:8000",
  "websocketUrl": "ws://localhost:8000/ws",
  "mockApiEnabled": true,
  "testUser": {
    "email": "test@example.com",
    "password": "TestPassword123"
  }
}
```

## Production Environment

### Production Setup

1. Create a `.env.production` file in the root of the frontend directory:

```
# Production Environment Configuration
NODE_ENV=production
REACT_APP_API_URL=https://api.aitrading.com
REACT_APP_WEBSOCKET_URL=wss://api.aitrading.com/ws

# Feature Flags
REACT_APP_ENABLE_PAPER_TRADING=true
REACT_APP_ENABLE_CIRCUIT_BREAKER=true
REACT_APP_ENABLE_PERFORMANCE_MONITORING=true
REACT_APP_ENABLE_ADVANCED_CHARTS=true

# DO NOT include actual API keys in this file
# They should be injected at runtime or through a secure vault

# Performance Configuration
REACT_APP_MAX_CACHE_SIZE=500
REACT_APP_CACHE_TTL_MS=30000
REACT_APP_BATCH_SIZE=20
REACT_APP_BATCH_DELAY_MS=50
```

2. Build the production application:

```bash
npm run build
```

### Secure API Key Management in Production

For production, **never** include actual API keys in your `.env.production` file. Instead:

1. Use a secrets management service:
   - AWS Secrets Manager
   - HashiCorp Vault
   - Azure Key Vault

2. Inject secrets at runtime:
   - Use environment variables in your hosting platform
   - Implement a secure backend proxy for API calls

3. Use a backend service to handle all exchange API calls:
   - Frontend makes authenticated calls to your backend
   - Backend adds API keys before forwarding to exchanges
   - This keeps API keys off client devices entirely

## Docker Configuration

### Dockerfile

Create a `Dockerfile` in the frontend directory:

```dockerfile
# Build stage
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    environment:
      - NODE_ENV=production
      - REACT_APP_API_URL=http://api:8000
      - REACT_APP_WEBSOCKET_URL=ws://api:8000/ws
      - REACT_APP_ENABLE_PAPER_TRADING=true
      - REACT_APP_ENABLE_CIRCUIT_BREAKER=true
      - REACT_APP_ENABLE_PERFORMANCE_MONITORING=true
    depends_on:
      - api

  api:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgres://user:password@db:5432/trading
      - JWT_SECRET=${JWT_SECRET}
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
      - COINBASE_API_KEY=${COINBASE_API_KEY}
      - COINBASE_API_SECRET=${COINBASE_API_SECRET}
      - COINBASE_PASSPHRASE=${COINBASE_PASSPHRASE}
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_API_SECRET=${ALPACA_API_SECRET}
    depends_on:
      - db

  db:
    image: postgres:14-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=trading
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## CI/CD Environment Setup

### GitHub Actions

For GitHub Actions, add secrets in the repository settings:

1. Go to Settings > Secrets > Actions
2. Add the following secrets:
   - `NETLIFY_AUTH_TOKEN`
   - `NETLIFY_SITE_ID`
   - `CYPRESS_RECORD_KEY`

### Environment-Specific Deployments

Configure environment-specific deployments in your CI/CD pipeline:

```yaml
deploy-staging:
  environment:
    name: staging
    url: https://staging.aitrading.com
  env:
    REACT_APP_API_URL: https://api-staging.aitrading.com
    REACT_APP_WEBSOCKET_URL: wss://api-staging.aitrading.com/ws
    REACT_APP_ENABLE_PAPER_TRADING: true

deploy-production:
  environment:
    name: production
    url: https://aitrading.com
  env:
    REACT_APP_API_URL: https://api.aitrading.com
    REACT_APP_WEBSOCKET_URL: wss://api.aitrading.com/ws
    REACT_APP_ENABLE_PAPER_TRADING: true
```

### Environment Variables in Netlify

If using Netlify for hosting:

1. Go to Site settings > Build & deploy > Environment
2. Add environment variables for each deployment context:
   - Production
   - Deploy Previews
   - Branch Deploys

Example variables:
```
REACT_APP_API_URL=https://api.aitrading.com
REACT_APP_WEBSOCKET_URL=wss://api.aitrading.com/ws
REACT_APP_ENABLE_PERFORMANCE_MONITORING=true
```

## Runtime Configuration

For runtime configuration that can be changed without rebuilding:

1. Create a `public/config.json` file:

```json
{
  "apiUrl": "https://api.aitrading.com",
  "websocketUrl": "wss://api.aitrading.com/ws",
  "features": {
    "paperTrading": true,
    "circuitBreaker": true,
    "performanceMonitoring": true,
    "advancedCharts": true
  },
  "performance": {
    "maxCacheSize": 500,
    "cacheTtlMs": 30000,
    "batchSize": 20,
    "batchDelayMs": 50
  }
}
```

2. Load this configuration at runtime:

```typescript
// src/config/runtime-config.ts
import axios from 'axios';

export interface RuntimeConfig {
  apiUrl: string;
  websocketUrl: string;
  features: {
    paperTrading: boolean;
    circuitBreaker: boolean;
    performanceMonitoring: boolean;
    advancedCharts: boolean;
  };
  performance: {
    maxCacheSize: number;
    cacheTtlMs: number;
    batchSize: number;
    batchDelayMs: number;
  };
}

// Default configuration (fallback)
const defaultConfig: RuntimeConfig = {
  apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  websocketUrl: process.env.REACT_APP_WEBSOCKET_URL || 'ws://localhost:8000/ws',
  features: {
    paperTrading: process.env.REACT_APP_ENABLE_PAPER_TRADING === 'true',
    circuitBreaker: process.env.REACT_APP_ENABLE_CIRCUIT_BREAKER === 'true',
    performanceMonitoring: process.env.REACT_APP_ENABLE_PERFORMANCE_MONITORING === 'true',
    advancedCharts: process.env.REACT_APP_ENABLE_ADVANCED_CHARTS === 'true'
  },
  performance: {
    maxCacheSize: parseInt(process.env.REACT_APP_MAX_CACHE_SIZE || '100'),
    cacheTtlMs: parseInt(process.env.REACT_APP_CACHE_TTL_MS || '60000'),
    batchSize: parseInt(process.env.REACT_APP_BATCH_SIZE || '10'),
    batchDelayMs: parseInt(process.env.REACT_APP_BATCH_DELAY_MS || '100')
  }
};

// Load runtime configuration
export async function loadRuntimeConfig(): Promise<RuntimeConfig> {
  try {
    const response = await axios.get<RuntimeConfig>('/config.json');
    return { ...defaultConfig, ...response.data };
  } catch (error) {
    console.warn('Failed to load runtime configuration, using defaults', error);
    return defaultConfig;
  }
}
```

This comprehensive environment configuration guide provides all the information needed to set up and configure the AI Trading Agent for different environments and deployment scenarios.
