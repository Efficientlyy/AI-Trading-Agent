# Continuous Integration and Deployment Setup Guide

This document provides instructions for setting up and configuring the CI/CD pipeline for the AI Trading Agent project.

## Table of Contents

1. [Overview](#overview)
2. [GitHub Actions Setup](#github-actions-setup)
3. [Test Configuration](#test-configuration)
4. [Deployment Configuration](#deployment-configuration)
5. [Environment Variables and Secrets](#environment-variables-and-secrets)
6. [Monitoring and Notifications](#monitoring-and-notifications)

## Overview

The AI Trading Agent uses GitHub Actions for continuous integration and deployment, with the following workflow:

1. **Code Push/PR**: Triggers the CI pipeline when code is pushed to main/develop branches or when a PR is created
2. **Build and Test**: Runs linting, type checking, unit tests, and builds the application
3. **E2E Tests**: Runs Cypress end-to-end tests against the built application
4. **Deployment**: Deploys to staging (PR previews) or production (main branch merges)

## GitHub Actions Setup

### Directory Structure

Create the following directory structure in your repository:

```
.github/
  workflows/
    frontend-ci.yml
```

### Workflow Configuration

1. Copy the `github-actions-config.yml` file from `frontend/src/ci/` to `.github/workflows/frontend-ci.yml`
2. Commit and push this file to your repository

The workflow includes the following jobs:

- **build-and-test**: Builds the application and runs unit tests
- **e2e-tests**: Runs Cypress end-to-end tests
- **deploy-preview**: Deploys PR previews to Netlify
- **deploy-production**: Deploys production builds to Netlify

## Test Configuration

### Unit Tests

Unit tests are run using Jest. Configure Jest in your `package.json`:

```json
{
  "scripts": {
    "test": "react-scripts test",
    "test:ci": "react-scripts test --watchAll=false --coverage"
  },
  "jest": {
    "collectCoverageFrom": [
      "src/**/*.{js,jsx,ts,tsx}",
      "!src/**/*.d.ts",
      "!src/index.tsx",
      "!src/serviceWorker.ts"
    ],
    "coverageThreshold": {
      "global": {
        "statements": 70,
        "branches": 60,
        "functions": 70,
        "lines": 70
      }
    }
  }
}
```

### End-to-End Tests

E2E tests use Cypress. Configure Cypress in `cypress.json`:

```json
{
  "baseUrl": "http://localhost:3000",
  "video": true,
  "screenshotOnRunFailure": true,
  "projectId": "YOUR_CYPRESS_PROJECT_ID",
  "integrationFolder": "cypress/integration",
  "testFiles": "**/*.spec.{js,jsx,ts,tsx}"
}
```

Create the following test structure:

```
cypress/
  integration/
    trading/
      order_flow.spec.ts
    dashboard/
      portfolio_view.spec.ts
    authentication/
      login.spec.ts
```

## Deployment Configuration

### Netlify Setup

1. Create a Netlify account and site
2. Connect your GitHub repository to Netlify
3. Configure build settings:
   - Build command: `cd frontend && npm ci && npm run build`
   - Publish directory: `frontend/build`
4. Set up branch deploys for PR previews

### Environment-Specific Configuration

Create environment-specific configuration files:

```
frontend/
  .env.development
  .env.test
  .env.production
```

Configure environment variables in each file:

```
# .env.production
REACT_APP_API_URL=https://api.aitrading.com
REACT_APP_WEBSOCKET_URL=wss://api.aitrading.com/ws
```

## Environment Variables and Secrets

Add the following secrets to your GitHub repository:

1. Go to Settings > Secrets > Actions
2. Add the following secrets:
   - `NETLIFY_AUTH_TOKEN`: Your Netlify personal access token
   - `NETLIFY_SITE_ID`: Your Netlify site ID
   - `CYPRESS_RECORD_KEY`: Your Cypress dashboard record key

## Monitoring and Notifications

### GitHub Notifications

Configure notification settings in the workflow:

```yaml
- name: Notify on failure
  if: failure()
  uses: rtCamp/action-slack-notify@v2
  env:
    SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
    SLACK_CHANNEL: ci-failures
    SLACK_COLOR: danger
    SLACK_TITLE: CI Pipeline Failed
    SLACK_MESSAGE: 'Build failed in ${{ github.workflow }}'
```

### Deployment Status Checks

Add status checks in your GitHub repository:

1. Go to Settings > Branches
2. Add branch protection rule for `main`
3. Require status checks to pass before merging
4. Select the following status checks:
   - `build-and-test`
   - `e2e-tests`

## Integration Tests for Trading Functionality

Create integration tests for trading functionality in the `cypress/integration/trading/` directory:

### Order Flow Test

```typescript
// cypress/integration/trading/order_flow.spec.ts
describe('Order Flow', () => {
  beforeEach(() => {
    cy.login();
    cy.visit('/trade');
  });

  it('should place a market buy order', () => {
    cy.get('[data-testid="asset-selector"]').click();
    cy.get('[data-testid="asset-option-BTC"]').click();
    cy.get('[data-testid="order-type-selector"]').select('Market');
    cy.get('[data-testid="side-buy"]').click();
    cy.get('[data-testid="quantity-input"]').type('0.01');
    cy.get('[data-testid="place-order-button"]').click();
    cy.get('[data-testid="order-confirmation"]').should('be.visible');
    cy.get('[data-testid="order-confirmation"]').should('contain', 'Order placed successfully');
  });

  it('should place a limit sell order', () => {
    cy.get('[data-testid="asset-selector"]').click();
    cy.get('[data-testid="asset-option-ETH"]').click();
    cy.get('[data-testid="order-type-selector"]').select('Limit');
    cy.get('[data-testid="side-sell"]').click();
    cy.get('[data-testid="quantity-input"]').type('0.1');
    cy.get('[data-testid="price-input"]').type('2000');
    cy.get('[data-testid="place-order-button"]').click();
    cy.get('[data-testid="order-confirmation"]').should('be.visible');
    cy.get('[data-testid="order-confirmation"]').should('contain', 'Order placed successfully');
  });

  it('should cancel an open order', () => {
    cy.get('[data-testid="open-orders-tab"]').click();
    cy.get('[data-testid="order-row"]').first().find('[data-testid="cancel-order-button"]').click();
    cy.get('[data-testid="confirm-cancel-button"]').click();
    cy.get('[data-testid="order-cancelled-notification"]').should('be.visible');
  });
});
```

## End-to-End Testing for Order Flow

Create a complete end-to-end test for the order flow:

```typescript
// cypress/integration/e2e/complete_trade_flow.spec.ts
describe('Complete Trading Flow', () => {
  beforeEach(() => {
    cy.login();
    cy.visit('/dashboard');
  });

  it('should complete a full trade lifecycle', () => {
    // Navigate to trading page from dashboard
    cy.get('[data-testid="quick-link-trading"]').click();
    cy.url().should('include', '/trade');

    // Select asset from portfolio
    cy.get('[data-testid="asset-selector"]').click();
    cy.get('[data-testid="asset-option-BTC"]').click();

    // Check market data is loaded
    cy.get('[data-testid="price-chart"]').should('be.visible');
    cy.get('[data-testid="current-price"]').should('not.be.empty');

    // Place a market buy order
    cy.get('[data-testid="order-type-selector"]').select('Market');
    cy.get('[data-testid="side-buy"]').click();
    cy.get('[data-testid="quantity-input"]').type('0.01');
    cy.get('[data-testid="place-order-button"]').click();
    
    // Verify order confirmation
    cy.get('[data-testid="order-confirmation"]').should('be.visible');
    cy.get('[data-testid="order-confirmation"]').should('contain', 'Order placed successfully');
    
    // Check order appears in order history
    cy.get('[data-testid="order-history-tab"]').click();
    cy.get('[data-testid="order-history-table"]').should('contain', 'BTC');
    cy.get('[data-testid="order-history-table"]').should('contain', 'Market');
    cy.get('[data-testid="order-history-table"]').should('contain', 'Buy');
    
    // Check position is updated
    cy.get('[data-testid="positions-tab"]').click();
    cy.get('[data-testid="position-row-BTC"]').should('be.visible');
    cy.get('[data-testid="position-row-BTC"]').should('contain', '0.01');
    
    // Navigate back to dashboard
    cy.get('[data-testid="nav-dashboard"]').click();
    cy.url().should('include', '/dashboard');
    
    // Verify portfolio is updated
    cy.get('[data-testid="portfolio-summary"]').should('contain', 'BTC');
  });
});
```

This comprehensive guide provides all the information needed to set up and configure the CI/CD pipeline for the AI Trading Agent project.
