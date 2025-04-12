/**
 * End-to-end tests for dashboard functionality
 */

describe('Dashboard', () => {
  beforeEach(() => {
    // Login before each test
    cy.visit('/login');
    cy.get('input[name="username"]').type('test_user');
    cy.get('input[name="password"]').type('test_password');
    cy.get('button[type="submit"]').click();
    
    // Should be on dashboard
    cy.url().should('include', '/');
  });

  it('should display dashboard components', () => {
    // Check for main dashboard components
    cy.get('[data-testid="portfolio-summary"]').should('be.visible');
    cy.get('[data-testid="market-overview"]').should('be.visible');
    cy.get('[data-testid="recent-trades"]').should('be.visible');
    cy.get('[data-testid="performance-chart"]').should('be.visible');
  });

  it('should navigate between dashboard tabs', () => {
    // Check navigation to different tabs
    cy.contains('Portfolio').click();
    cy.url().should('include', '/portfolio');
    
    cy.contains('Trading').click();
    cy.url().should('include', '/trading');
    
    cy.contains('Backtest').click();
    cy.url().should('include', '/backtest');
    
    cy.contains('Strategies').click();
    cy.url().should('include', '/strategies');
    
    cy.contains('Sentiment').click();
    cy.url().should('include', '/sentiment');
    
    cy.contains('Settings').click();
    cy.url().should('include', '/settings');
    
    // Back to dashboard
    cy.contains('Dashboard').click();
    cy.url().should('include', '/');
  });

  it('should display portfolio data correctly', () => {
    cy.contains('Portfolio').click();
    
    // Check portfolio components
    cy.get('[data-testid="portfolio-value"]').should('be.visible');
    cy.get('[data-testid="asset-allocation"]').should('be.visible');
    cy.get('[data-testid="holdings-table"]').should('be.visible');
    
    // Check that portfolio value is a number
    cy.get('[data-testid="portfolio-value"]')
      .invoke('text')
      .should('match', /\$[\d,]+\.\d{2}/);
    
    // Check that holdings table has rows
    cy.get('[data-testid="holdings-table"] tbody tr').should('have.length.at.least', 1);
  });

  it('should toggle between light and dark themes', () => {
    // Open theme settings
    cy.get('[data-testid="theme-toggle"]').click();
    
    // Select dark theme
    cy.contains('Dark').click();
    
    // Check if dark theme is applied
    cy.get('html').should('have.class', 'dark');
    
    // Select light theme
    cy.contains('Light').click();
    
    // Check if light theme is applied
    cy.get('html').should('not.have.class', 'dark');
  });

  it('should display performance metrics', () => {
    // Check performance metrics on dashboard
    cy.get('[data-testid="performance-metrics"]').within(() => {
      cy.contains('Total Return').should('be.visible');
      cy.contains('Sharpe Ratio').should('be.visible');
      cy.contains('Max Drawdown').should('be.visible');
      cy.contains('Win Rate').should('be.visible');
    });
  });

  it('should filter market data by date range', () => {
    // Go to market overview section
    cy.get('[data-testid="market-overview"]').within(() => {
      // Open date range picker
      cy.get('[data-testid="date-range-picker"]').click();
      
      // Select last 7 days
      cy.contains('Last 7 Days').click();
      
      // Check if data is updated
      cy.get('[data-testid="loading-indicator"]').should('exist');
      cy.get('[data-testid="loading-indicator"]').should('not.exist');
      
      // Chart should be visible
      cy.get('[data-testid="price-chart"]').should('be.visible');
    });
  });

  it('should display notifications', () => {
    // Trigger a notification (e.g., by refreshing data)
    cy.get('[data-testid="refresh-data"]').click();
    
    // Check if notification appears
    cy.get('[data-testid="notification"]').should('be.visible');
    
    // Notification should disappear after a while
    cy.get('[data-testid="notification"]', { timeout: 10000 }).should('not.exist');
  });
});
