/**
 * End-to-end tests for the trading order flow
 * 
 * These tests verify the complete order flow from the user's perspective,
 * including asset selection, order entry, submission, and confirmation.
 */

/// <reference types="cypress" />
export {};


describe('Order Flow', () => {
  beforeEach(() => {
    // Mock authentication
    cy.intercept('POST', '/api/auth/login', {
      statusCode: 200,
      body: {
        token: 'mock-jwt-token',
        user: {
          id: 'user123',
          username: 'testuser',
          email: 'test@example.com',
          role: 'user'
        }
      }
    }).as('loginRequest');

    // Mock market data
    cy.intercept('GET', '/api/market/data/BTC-USD', {
      statusCode: 200,
      body: {
        symbol: 'BTC-USD',
        price: 50000,
        change24h: 1200,
        changePercent24h: 2.4,
        high24h: 51200,
        low24h: 49200,
        volume24h: 1500000000,
        marketCap: 950000000000,
        lastUpdated: new Date().toISOString()
      }
    }).as('getBTCData');

    cy.intercept('GET', '/api/market/data/ETH-USD', {
      statusCode: 200,
      body: {
        symbol: 'ETH-USD',
        price: 3500,
        change24h: 120,
        changePercent24h: 3.5,
        high24h: 3600,
        low24h: 3400,
        volume24h: 800000000,
        marketCap: 420000000000,
        lastUpdated: new Date().toISOString()
      }
    }).as('getETHData');

    // Mock order book
    cy.intercept('GET', '/api/market/orderbook/*', {
      statusCode: 200,
      body: {
        bids: [
          { price: 49800, size: 1.5 },
          { price: 49750, size: 2.3 },
          { price: 49700, size: 3.1 }
        ],
        asks: [
          { price: 50000, size: 1.2 },
          { price: 50050, size: 2.0 },
          { price: 50100, size: 1.5 }
        ]
      }
    }).as('getOrderBook');

    // Mock available assets
    cy.intercept('GET', '/api/market/assets', {
      statusCode: 200,
      body: [
        {
          symbol: 'BTC-USD',
          name: 'Bitcoin',
          minOrderSize: 0.0001,
          maxOrderSize: 100,
          pricePrecision: 2,
          quantityPrecision: 8,
          status: 'ACTIVE'
        },
        {
          symbol: 'ETH-USD',
          name: 'Ethereum',
          minOrderSize: 0.001,
          maxOrderSize: 1000,
          pricePrecision: 2,
          quantityPrecision: 6,
          status: 'ACTIVE'
        }
      ]
    }).as('getAssets');

    // Mock portfolio data
    cy.intercept('GET', '/api/portfolio', {
      statusCode: 200,
      body: {
        totalValue: 25000,
        availableCash: 10000,
        positions: [
          {
            symbol: 'BTC-USD',
            quantity: 0.2,
            averageEntryPrice: 48000,
            currentPrice: 50000,
            marketValue: 10000,
            unrealizedPnL: 400,
            unrealizedPnLPercent: 4,
            allocation: 40
          },
          {
            symbol: 'ETH-USD',
            quantity: 1.5,
            averageEntryPrice: 3300,
            currentPrice: 3500,
            marketValue: 5250,
            unrealizedPnL: 300,
            unrealizedPnLPercent: 6.06,
            allocation: 21
          }
        ]
      }
    }).as('getPortfolio');

    // Mock order placement
    cy.intercept('POST', '/api/orders', {
      statusCode: 200,
      body: {
        id: 'ord123456789',
        clientOrderId: 'client-ord-123',
        symbol: 'BTC-USD',
        side: 'buy',
        type: 'MARKET',
        status: 'FILLED',
        quantity: 0.1,
        price: 50000,
        filledQuantity: 0.1,
        filledPrice: 50000,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        fees: 25,
        totalValue: 5000
      }
    }).as('placeOrder');

    // Mock order cancellation
    cy.intercept('DELETE', '/api/orders/*', {
      statusCode: 200,
      body: {
        success: true,
        message: 'Order cancelled successfully'
      }
    }).as('cancelOrder');

    // Mock open orders
    cy.intercept('GET', '/api/orders/open', {
      statusCode: 200,
      body: [
        {
          id: 'open-ord1',
          clientOrderId: 'client-open-ord1',
          symbol: 'BTC-USD',
          side: 'sell',
          type: 'LIMIT',
          status: 'OPEN',
          quantity: 0.05,
          price: 52000,
          filledQuantity: 0,
          filledPrice: null,
          createdAt: new Date(Date.now() - 3600000).toISOString(),
          updatedAt: new Date(Date.now() - 3600000).toISOString()
        }
      ]
    }).as('getOpenOrders');

    // Login and navigate to trading page
    cy.visit('/login');
    cy.get('[data-testid="username-input"]').type('testuser');
    cy.get('[data-testid="password-input"]').type('password123');
    cy.get('[data-testid="login-button"]').click();
    cy.wait('@loginRequest');
    cy.visit('/trade');
  });

  it('should place a market buy order successfully', () => {
    // Wait for page to load
    cy.wait(['@getBTCData', '@getOrderBook', '@getAssets', '@getPortfolio']);
    
    // Verify trading page is loaded
    cy.get('[data-testid="asset-selector"]').should('be.visible');
    cy.get('[data-testid="order-entry-form"]').should('be.visible');
    
    // Select market order type
    cy.get('[data-testid="order-type-selector"]').select('MARKET');
    
    // Select buy side
    cy.get('[data-testid="side-buy-button"]').click();
    
    // Enter quantity
    cy.get('[data-testid="quantity-input"]').clear().type('0.1');
    
    // Verify order preview is updated
    cy.get('[data-testid="order-preview"]').should('contain', 'Market Buy');
    cy.get('[data-testid="order-preview"]').should('contain', '0.1 BTC');
    cy.get('[data-testid="order-preview"]').should('contain', 'Estimated Total: $5,000.00');
    
    // Submit order
    cy.get('[data-testid="submit-order-button"]').click();
    
    // Wait for order to be placed
    cy.wait('@placeOrder');
    
    // Verify order confirmation is shown
    cy.get('[data-testid="order-confirmation"]').should('be.visible');
    cy.get('[data-testid="order-confirmation"]').should('contain', 'Order placed successfully');
    cy.get('[data-testid="order-confirmation"]').should('contain', 'BTC-USD');
    cy.get('[data-testid="order-confirmation"]').should('contain', 'Market Buy');
    cy.get('[data-testid="order-confirmation"]').should('contain', '0.1 BTC');
    
    // Close confirmation
    cy.get('[data-testid="close-confirmation-button"]').click();
    cy.get('[data-testid="order-confirmation"]').should('not.exist');
    
    // Verify order form is reset
    cy.get('[data-testid="quantity-input"]').should('have.value', '');
  });

  it('should place a limit sell order successfully', () => {
    // Wait for page to load
    cy.wait(['@getBTCData', '@getOrderBook', '@getAssets', '@getPortfolio']);
    
    // Verify trading page is loaded
    cy.get('[data-testid="asset-selector"]').should('be.visible');
    cy.get('[data-testid="order-entry-form"]').should('be.visible');
    
    // Select limit order type
    cy.get('[data-testid="order-type-selector"]').select('LIMIT');
    
    // Select sell side
    cy.get('[data-testid="side-sell-button"]').click();
    
    // Enter quantity and price
    cy.get('[data-testid="quantity-input"]').clear().type('0.05');
    cy.get('[data-testid="price-input"]').clear().type('52000');
    
    // Verify order preview is updated
    cy.get('[data-testid="order-preview"]').should('contain', 'Limit Sell');
    cy.get('[data-testid="order-preview"]').should('contain', '0.05 BTC');
    cy.get('[data-testid="order-preview"]').should('contain', '@$52,000.00');
    cy.get('[data-testid="order-preview"]').should('contain', 'Total: $2,600.00');
    
    // Submit order
    cy.get('[data-testid="submit-order-button"]').click();
    
    // Wait for order to be placed
    cy.wait('@placeOrder');
    
    // Verify order confirmation is shown
    cy.get('[data-testid="order-confirmation"]').should('be.visible');
    cy.get('[data-testid="order-confirmation"]').should('contain', 'Order placed successfully');
    cy.get('[data-testid="order-confirmation"]').should('contain', 'BTC-USD');
    cy.get('[data-testid="order-confirmation"]').should('contain', 'Limit Sell');
    cy.get('[data-testid="order-confirmation"]').should('contain', '0.05 BTC');
    cy.get('[data-testid="order-confirmation"]').should('contain', '@$52,000.00');
    
    // Close confirmation
    cy.get('[data-testid="close-confirmation-button"]').click();
    cy.get('[data-testid="order-confirmation"]').should('not.exist');
  });

  it('should change asset and update market data', () => {
    // Wait for page to load with BTC data
    cy.wait(['@getBTCData', '@getOrderBook', '@getAssets', '@getPortfolio']);
    
    // Verify BTC is selected
    cy.get('[data-testid="asset-selector"]').should('contain', 'BTC-USD');
    cy.get('[data-testid="current-price"]').should('contain', '$50,000.00');
    
    // Select ETH
    cy.get('[data-testid="asset-selector"]').click();
    cy.get('[data-testid="asset-option-ETH-USD"]').click();
    
    // Wait for ETH data to load
    cy.wait('@getETHData');
    
    // Verify ETH data is displayed
    cy.get('[data-testid="asset-selector"]').should('contain', 'ETH-USD');
    cy.get('[data-testid="current-price"]').should('contain', '$3,500.00');
    
    // Verify order form is updated for ETH
    cy.get('[data-testid="order-entry-form"]').should('contain', 'ETH-USD');
  });

  it('should show validation errors for invalid inputs', () => {
    // Wait for page to load
    cy.wait(['@getBTCData', '@getOrderBook', '@getAssets', '@getPortfolio']);
    
    // Select market order type
    cy.get('[data-testid="order-type-selector"]').select('MARKET');
    
    // Select buy side
    cy.get('[data-testid="side-buy-button"]').click();
    
    // Try to submit without quantity
    cy.get('[data-testid="submit-order-button"]').click();
    cy.get('[data-testid="quantity-error"]').should('be.visible');
    cy.get('[data-testid="quantity-error"]').should('contain', 'Quantity is required');
    
    // Enter invalid quantity (negative)
    cy.get('[data-testid="quantity-input"]').clear().type('-0.1');
    cy.get('[data-testid="submit-order-button"]').click();
    cy.get('[data-testid="quantity-error"]').should('be.visible');
    cy.get('[data-testid="quantity-error"]').should('contain', 'Quantity must be greater than 0');
    
    // Enter valid quantity but switch to limit without price
    cy.get('[data-testid="quantity-input"]').clear().type('0.1');
    cy.get('[data-testid="order-type-selector"]').select('LIMIT');
    cy.get('[data-testid="submit-order-button"]').click();
    cy.get('[data-testid="price-error"]').should('be.visible');
    cy.get('[data-testid="price-error"]').should('contain', 'Price is required for limit orders');
    
    // Enter invalid price (negative)
    cy.get('[data-testid="price-input"]').clear().type('-50000');
    cy.get('[data-testid="submit-order-button"]').click();
    cy.get('[data-testid="price-error"]').should('be.visible');
    cy.get('[data-testid="price-error"]').should('contain', 'Price must be greater than 0');
  });

  it('should cancel an open order', () => {
    // Wait for page to load
    cy.wait(['@getBTCData', '@getOrderBook', '@getAssets', '@getPortfolio']);
    
    // Navigate to open orders tab
    cy.get('[data-testid="open-orders-tab"]').click();
    
    // Wait for open orders to load
    cy.wait('@getOpenOrders');
    
    // Verify open order is displayed
    cy.get('[data-testid="open-orders-table"]').should('be.visible');
    cy.get('[data-testid="order-row"]').should('have.length', 1);
    cy.get('[data-testid="order-row"]').should('contain', 'BTC-USD');
    cy.get('[data-testid="order-row"]').should('contain', 'Limit Sell');
    
    // Click cancel button
    cy.get('[data-testid="cancel-order-button"]').click();
    
    // Confirm cancellation
    cy.get('[data-testid="confirm-cancel-dialog"]').should('be.visible');
    cy.get('[data-testid="confirm-cancel-button"]').click();
    
    // Wait for cancellation request
    cy.wait('@cancelOrder');
    
    // Verify cancellation success message
    cy.get('[data-testid="cancel-success-message"]').should('be.visible');
    cy.get('[data-testid="cancel-success-message"]').should('contain', 'Order cancelled successfully');
  });

  it('should handle API errors gracefully', () => {
    // Mock API error for order placement
    cy.intercept('POST', '/api/orders', {
      statusCode: 400,
      body: {
        error: 'Insufficient funds',
        code: 'INSUFFICIENT_FUNDS',
        message: 'Not enough balance to place this order'
      }
    }).as('placeOrderError');
    
    // Wait for page to load
    cy.wait(['@getBTCData', '@getOrderBook', '@getAssets', '@getPortfolio']);
    
    // Select market order type
    cy.get('[data-testid="order-type-selector"]').select('MARKET');
    
    // Select buy side
    cy.get('[data-testid="side-buy-button"]').click();
    
    // Enter quantity
    cy.get('[data-testid="quantity-input"]').clear().type('100'); // Large amount to trigger error
    
    // Submit order
    cy.get('[data-testid="submit-order-button"]').click();
    
    // Wait for error response
    cy.wait('@placeOrderError');
    
    // Verify error message is shown
    cy.get('[data-testid="order-error"]').should('be.visible');
    cy.get('[data-testid="order-error"]').should('contain', 'Insufficient funds');
    cy.get('[data-testid="order-error"]').should('contain', 'Not enough balance to place this order');
  });

  it('should complete the full order lifecycle', () => {
    // This test simulates a complete order lifecycle:
    // 1. Navigate from dashboard to trading
    // 2. Select asset
    // 3. Place order
    // 4. View order in history
    // 5. Check updated portfolio
    
    // First, visit dashboard
    cy.visit('/dashboard');
    cy.wait('@getPortfolio');
    
    // Click on trading link
    cy.get('[data-testid="quick-link-trading"]').click();
    
    // Verify we're on trading page
    cy.url().should('include', '/trade');
    cy.wait(['@getBTCData', '@getOrderBook', '@getAssets', '@getPortfolio']);
    
    // Select market order type
    cy.get('[data-testid="order-type-selector"]').select('MARKET');
    
    // Select buy side
    cy.get('[data-testid="side-buy-button"]').click();
    
    // Enter quantity
    cy.get('[data-testid="quantity-input"]').clear().type('0.1');
    
    // Submit order
    cy.get('[data-testid="submit-order-button"]').click();
    
    // Wait for order to be placed
    cy.wait('@placeOrder');
    
    // Verify order confirmation
    cy.get('[data-testid="order-confirmation"]').should('be.visible');
    cy.get('[data-testid="close-confirmation-button"]').click();
    
    // Mock order history
    cy.intercept('GET', '/api/orders/history', {
      statusCode: 200,
      body: [
        {
          id: 'ord123456789',
          clientOrderId: 'client-ord-123',
          symbol: 'BTC-USD',
          side: 'buy',
          type: 'MARKET',
          status: 'FILLED',
          quantity: 0.1,
          price: null,
          filledQuantity: 0.1,
          filledPrice: 50000,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          fees: 25,
          totalValue: 5000
        }
      ]
    }).as('getOrderHistory');
    
    // Go to order history tab
    cy.get('[data-testid="order-history-tab"]').click();
    
    // Wait for order history to load
    cy.wait('@getOrderHistory');
    
    // Verify our order appears in history
    cy.get('[data-testid="order-history-table"]').should('be.visible');
    cy.get('[data-testid="order-history-row"]').should('contain', 'BTC-USD');
    cy.get('[data-testid="order-history-row"]').should('contain', 'Market Buy');
    cy.get('[data-testid="order-history-row"]').should('contain', '0.1 BTC');
    
    // Mock updated portfolio after order
    cy.intercept('GET', '/api/portfolio', {
      statusCode: 200,
      body: {
        totalValue: 25000,
        availableCash: 5000, // Reduced by order amount
        positions: [
          {
            symbol: 'BTC-USD',
            quantity: 0.3, // Increased by 0.1
            averageEntryPrice: 48667, // Weighted average
            currentPrice: 50000,
            marketValue: 15000,
            unrealizedPnL: 400,
            unrealizedPnLPercent: 2.74,
            allocation: 60
          },
          {
            symbol: 'ETH-USD',
            quantity: 1.5,
            averageEntryPrice: 3300,
            currentPrice: 3500,
            marketValue: 5250,
            unrealizedPnL: 300,
            unrealizedPnLPercent: 6.06,
            allocation: 21
          }
        ]
      }
    }).as('getUpdatedPortfolio');
    
    // Go back to dashboard to check updated portfolio
    cy.get('[data-testid="nav-dashboard"]').click();
    
    // Wait for updated portfolio to load
    cy.wait('@getUpdatedPortfolio');
    
    // Verify portfolio is updated
    cy.get('[data-testid="portfolio-summary"]').should('contain', '$25,000.00');
    cy.get('[data-testid="btc-position"]').should('contain', '0.3 BTC');
  });
});
