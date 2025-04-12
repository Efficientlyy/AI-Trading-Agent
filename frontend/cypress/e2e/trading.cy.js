/**
 * End-to-end tests for trading functionality
 */

describe('Trading', () => {
  beforeEach(() => {
    // Login before each test
    cy.visit('/login');
    cy.get('input[name="username"]').type('test_user');
    cy.get('input[name="password"]').type('test_password');
    cy.get('button[type="submit"]').click();
    
    // Navigate to trading page
    cy.contains('Trading').click();
    cy.url().should('include', '/trading');
  });

  it('should display trading interface components', () => {
    // Check for main trading components
    cy.get('[data-testid="order-form"]').should('be.visible');
    cy.get('[data-testid="market-data"]').should('be.visible');
    cy.get('[data-testid="order-book"]').should('be.visible');
    cy.get('[data-testid="trade-history"]').should('be.visible');
  });

  it('should allow selecting different assets', () => {
    // Select asset dropdown
    cy.get('[data-testid="asset-selector"]').click();
    
    // Select BTC
    cy.contains('BTC').click();
    
    // Check if market data updates
    cy.get('[data-testid="loading-indicator"]').should('exist');
    cy.get('[data-testid="loading-indicator"]').should('not.exist');
    
    // Price chart should update
    cy.get('[data-testid="price-chart"]').should('be.visible');
    
    // Order book should update
    cy.get('[data-testid="order-book"]').should('contain', 'BTC');
  });

  it('should validate order form inputs', () => {
    // Try to submit empty form
    cy.get('[data-testid="order-form"] button[type="submit"]').click();
    
    // Should show validation errors
    cy.get('[data-testid="order-form"]').contains('Quantity is required').should('be.visible');
    
    // Enter invalid quantity (negative)
    cy.get('[data-testid="quantity-input"]').type('-1');
    cy.get('[data-testid="order-form"] button[type="submit"]').click();
    
    // Should show validation error
    cy.get('[data-testid="order-form"]').contains('Quantity must be positive').should('be.visible');
    
    // Enter valid quantity
    cy.get('[data-testid="quantity-input"]').clear().type('0.1');
    
    // For limit orders, price is required
    cy.get('[data-testid="order-type-selector"]').select('limit');
    cy.get('[data-testid="order-form"] button[type="submit"]').click();
    
    // Should show validation error for price
    cy.get('[data-testid="order-form"]').contains('Price is required for limit orders').should('be.visible');
  });

  it('should place a market buy order', () => {
    // Select BTC
    cy.get('[data-testid="asset-selector"]').click();
    cy.contains('BTC').click();
    
    // Select buy side
    cy.get('[data-testid="buy-side"]').click();
    
    // Select market order
    cy.get('[data-testid="order-type-selector"]').select('market');
    
    // Enter quantity
    cy.get('[data-testid="quantity-input"]').type('0.01');
    
    // Submit order
    cy.get('[data-testid="order-form"] button[type="submit"]').click();
    
    // Confirmation dialog should appear
    cy.get('[data-testid="order-confirmation"]').should('be.visible');
    cy.get('[data-testid="order-confirmation"]').contains('Confirm Order').click();
    
    // Success notification should appear
    cy.get('[data-testid="notification"]').contains('Order placed successfully').should('be.visible');
    
    // Order should appear in trade history
    cy.get('[data-testid="trade-history"]').contains('BTC').should('be.visible');
    cy.get('[data-testid="trade-history"]').contains('Buy').should('be.visible');
  });

  it('should place a limit sell order', () => {
    // Select BTC
    cy.get('[data-testid="asset-selector"]').click();
    cy.contains('BTC').click();
    
    // Select sell side
    cy.get('[data-testid="sell-side"]').click();
    
    // Select limit order
    cy.get('[data-testid="order-type-selector"]').select('limit');
    
    // Enter quantity
    cy.get('[data-testid="quantity-input"]').type('0.01');
    
    // Enter price (10% above current price)
    cy.get('[data-testid="current-price"]').invoke('text').then((priceText) => {
      const currentPrice = parseFloat(priceText.replace(/[^0-9.]/g, ''));
      const limitPrice = (currentPrice * 1.1).toFixed(2);
      
      cy.get('[data-testid="price-input"]').type(limitPrice);
      
      // Submit order
      cy.get('[data-testid="order-form"] button[type="submit"]').click();
      
      // Confirmation dialog should appear
      cy.get('[data-testid="order-confirmation"]').should('be.visible');
      cy.get('[data-testid="order-confirmation"]').contains('Confirm Order').click();
      
      // Success notification should appear
      cy.get('[data-testid="notification"]').contains('Order placed successfully').should('be.visible');
      
      // Order should appear in open orders
      cy.get('[data-testid="open-orders"]').contains('BTC').should('be.visible');
      cy.get('[data-testid="open-orders"]').contains('Sell').should('be.visible');
      cy.get('[data-testid="open-orders"]').contains('Limit').should('be.visible');
      cy.get('[data-testid="open-orders"]').contains(limitPrice).should('be.visible');
    });
  });

  it('should cancel an open order', () => {
    // Go to open orders tab
    cy.get('[data-testid="open-orders-tab"]').click();
    
    // If there are open orders, cancel the first one
    cy.get('[data-testid="open-orders"]').then(($orders) => {
      if ($orders.find('tr').length > 1) { // More than header row
        cy.get('[data-testid="open-orders"] tr').eq(1).find('[data-testid="cancel-order"]').click();
        
        // Confirm cancellation
        cy.get('[data-testid="cancel-confirmation"]').contains('Confirm').click();
        
        // Success notification should appear
        cy.get('[data-testid="notification"]').contains('Order cancelled').should('be.visible');
      } else {
        // Skip test if no open orders
        cy.log('No open orders to cancel');
      }
    });
  });

  it('should display order history', () => {
    // Go to order history tab
    cy.get('[data-testid="order-history-tab"]').click();
    
    // Order history should be visible
    cy.get('[data-testid="order-history"]').should('be.visible');
    
    // Should have at least one row (header)
    cy.get('[data-testid="order-history"] tr').should('have.length.at.least', 1);
  });

  it('should update portfolio after trade', () => {
    // Get initial portfolio value
    cy.visit('/');
    cy.get('[data-testid="portfolio-value"]').invoke('text').then((initialValue) => {
      const initialPortfolioValue = parseFloat(initialValue.replace(/[^0-9.]/g, ''));
      
      // Place a trade
      cy.contains('Trading').click();
      cy.get('[data-testid="asset-selector"]').click();
      cy.contains('BTC').click();
      cy.get('[data-testid="buy-side"]').click();
      cy.get('[data-testid="order-type-selector"]').select('market');
      cy.get('[data-testid="quantity-input"]').type('0.01');
      cy.get('[data-testid="order-form"] button[type="submit"]').click();
      cy.get('[data-testid="order-confirmation"]').contains('Confirm Order').click();
      
      // Wait for order to process
      cy.get('[data-testid="notification"]').contains('Order placed successfully').should('be.visible');
      
      // Go back to dashboard
      cy.contains('Dashboard').click();
      
      // Portfolio value should be different
      cy.get('[data-testid="portfolio-value"]').invoke('text').then((newValue) => {
        const newPortfolioValue = parseFloat(newValue.replace(/[^0-9.]/g, ''));
        expect(newPortfolioValue).to.not.equal(initialPortfolioValue);
      });
    });
  });
});
