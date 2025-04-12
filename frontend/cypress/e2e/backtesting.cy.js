/**
 * End-to-end tests for backtesting and strategy optimization functionality
 */

describe('Backtesting', () => {
  beforeEach(() => {
    // Login before each test
    cy.visit('/login');
    cy.get('input[name="username"]').type('test_user');
    cy.get('input[name="password"]').type('test_password');
    cy.get('button[type="submit"]').click();
    
    // Navigate to backtesting page
    cy.contains('Backtest').click();
    cy.url().should('include', '/backtest');
  });

  it('should display backtesting interface components', () => {
    // Check for main backtesting components
    cy.get('[data-testid="backtest-form"]').should('be.visible');
    cy.get('[data-testid="strategy-selector"]').should('be.visible');
    cy.get('[data-testid="asset-selector"]').should('be.visible');
    cy.get('[data-testid="date-range-picker"]').should('be.visible');
    cy.get('[data-testid="parameter-inputs"]').should('be.visible');
  });

  it('should validate backtest form inputs', () => {
    // Try to submit empty form
    cy.get('[data-testid="backtest-form"] button[type="submit"]').click();
    
    // Should show validation errors
    cy.get('[data-testid="backtest-form"]').contains('Strategy is required').should('be.visible');
    cy.get('[data-testid="backtest-form"]').contains('Asset is required').should('be.visible');
    cy.get('[data-testid="backtest-form"]').contains('Date range is required').should('be.visible');
    
    // Select strategy
    cy.get('[data-testid="strategy-selector"]').click();
    cy.contains('Moving Average Crossover').click();
    
    // Select asset
    cy.get('[data-testid="asset-selector"]').click();
    cy.contains('BTC').click();
    
    // Select date range
    cy.get('[data-testid="date-range-picker"]').click();
    cy.get('[data-testid="date-range-picker"]').within(() => {
      cy.get('[data-testid="start-date"]').type('2023-01-01');
      cy.get('[data-testid="end-date"]').type('2023-12-31');
    });
    cy.get('body').click(); // Close date picker
    
    // Submit form - should now be valid
    cy.get('[data-testid="backtest-form"] button[type="submit"]').click();
    
    // Should start backtest
    cy.get('[data-testid="backtest-progress"]').should('be.visible');
  });

  it('should run a backtest and display results', () => {
    // Set up and run a backtest
    cy.get('[data-testid="strategy-selector"]').click();
    cy.contains('Moving Average Crossover').click();
    
    cy.get('[data-testid="asset-selector"]').click();
    cy.contains('BTC').click();
    
    cy.get('[data-testid="date-range-picker"]').click();
    cy.get('[data-testid="date-range-picker"]').within(() => {
      cy.get('[data-testid="start-date"]').type('2023-01-01');
      cy.get('[data-testid="end-date"]').type('2023-12-31');
    });
    cy.get('body').click(); // Close date picker
    
    // Set strategy parameters
    cy.get('[data-testid="parameter-inputs"]').within(() => {
      cy.get('[data-testid="fast-period"]').clear().type('10');
      cy.get('[data-testid="slow-period"]').clear().type('30');
    });
    
    // Submit form
    cy.get('[data-testid="backtest-form"] button[type="submit"]').click();
    
    // Wait for backtest to complete
    cy.get('[data-testid="backtest-progress"]', { timeout: 30000 }).should('not.exist');
    
    // Results should be displayed
    cy.get('[data-testid="backtest-results"]').should('be.visible');
    cy.get('[data-testid="equity-curve"]').should('be.visible');
    cy.get('[data-testid="trade-list"]').should('be.visible');
    cy.get('[data-testid="performance-metrics"]').should('be.visible');
    
    // Check metrics
    cy.get('[data-testid="performance-metrics"]').within(() => {
      cy.contains('Total Return').should('be.visible');
      cy.contains('Sharpe Ratio').should('be.visible');
      cy.contains('Max Drawdown').should('be.visible');
      cy.contains('Win Rate').should('be.visible');
    });
  });

  it('should compare multiple backtest results', () => {
    // Run first backtest
    cy.get('[data-testid="strategy-selector"]').click();
    cy.contains('Moving Average Crossover').click();
    
    cy.get('[data-testid="asset-selector"]').click();
    cy.contains('BTC').click();
    
    cy.get('[data-testid="date-range-picker"]').click();
    cy.get('[data-testid="date-range-picker"]').within(() => {
      cy.get('[data-testid="start-date"]').type('2023-01-01');
      cy.get('[data-testid="end-date"]').type('2023-12-31');
    });
    cy.get('body').click(); // Close date picker
    
    // Set strategy parameters
    cy.get('[data-testid="parameter-inputs"]').within(() => {
      cy.get('[data-testid="fast-period"]').clear().type('10');
      cy.get('[data-testid="slow-period"]').clear().type('30');
    });
    
    // Submit form
    cy.get('[data-testid="backtest-form"] button[type="submit"]').click();
    
    // Wait for backtest to complete
    cy.get('[data-testid="backtest-progress"]', { timeout: 30000 }).should('not.exist');
    
    // Save this backtest
    cy.get('[data-testid="save-backtest"]').click();
    cy.get('[data-testid="backtest-name-input"]').type('Backtest 1');
    cy.get('[data-testid="save-backtest-confirm"]').click();
    
    // Run second backtest with different parameters
    cy.get('[data-testid="new-backtest"]').click();
    
    cy.get('[data-testid="strategy-selector"]').click();
    cy.contains('Moving Average Crossover').click();
    
    cy.get('[data-testid="asset-selector"]').click();
    cy.contains('BTC').click();
    
    cy.get('[data-testid="date-range-picker"]').click();
    cy.get('[data-testid="date-range-picker"]').within(() => {
      cy.get('[data-testid="start-date"]').type('2023-01-01');
      cy.get('[data-testid="end-date"]').type('2023-12-31');
    });
    cy.get('body').click(); // Close date picker
    
    // Set different strategy parameters
    cy.get('[data-testid="parameter-inputs"]').within(() => {
      cy.get('[data-testid="fast-period"]').clear().type('5');
      cy.get('[data-testid="slow-period"]').clear().type('20');
    });
    
    // Submit form
    cy.get('[data-testid="backtest-form"] button[type="submit"]').click();
    
    // Wait for backtest to complete
    cy.get('[data-testid="backtest-progress"]', { timeout: 30000 }).should('not.exist');
    
    // Save this backtest
    cy.get('[data-testid="save-backtest"]').click();
    cy.get('[data-testid="backtest-name-input"]').type('Backtest 2');
    cy.get('[data-testid="save-backtest-confirm"]').click();
    
    // Go to comparison view
    cy.get('[data-testid="compare-backtests"]').click();
    
    // Select both backtests
    cy.get('[data-testid="backtest-selection"]').within(() => {
      cy.contains('Backtest 1').click();
      cy.contains('Backtest 2').click();
    });
    
    // Comparison should be displayed
    cy.get('[data-testid="comparison-chart"]').should('be.visible');
    cy.get('[data-testid="comparison-metrics"]').should('be.visible');
    
    // Check that both backtests are in the comparison
    cy.get('[data-testid="comparison-metrics"]').should('contain', 'Backtest 1');
    cy.get('[data-testid="comparison-metrics"]').should('contain', 'Backtest 2');
  });

  it('should run strategy optimization', () => {
    // Go to optimization tab
    cy.get('[data-testid="optimization-tab"]').click();
    
    // Set up optimization
    cy.get('[data-testid="strategy-selector"]').click();
    cy.contains('Moving Average Crossover').click();
    
    cy.get('[data-testid="asset-selector"]').click();
    cy.contains('BTC').click();
    
    cy.get('[data-testid="date-range-picker"]').click();
    cy.get('[data-testid="date-range-picker"]').within(() => {
      cy.get('[data-testid="start-date"]').type('2023-01-01');
      cy.get('[data-testid="end-date"]').type('2023-12-31');
    });
    cy.get('body').click(); // Close date picker
    
    // Set parameter ranges
    cy.get('[data-testid="parameter-ranges"]').within(() => {
      // Fast period range
      cy.get('[data-testid="fast-period-min"]').clear().type('5');
      cy.get('[data-testid="fast-period-max"]').clear().type('20');
      cy.get('[data-testid="fast-period-step"]').clear().type('5');
      
      // Slow period range
      cy.get('[data-testid="slow-period-min"]').clear().type('20');
      cy.get('[data-testid="slow-period-max"]').clear().type('50');
      cy.get('[data-testid="slow-period-step"]').clear().type('10');
    });
    
    // Set optimization target
    cy.get('[data-testid="optimization-target"]').select('Sharpe Ratio');
    
    // Start optimization
    cy.get('[data-testid="start-optimization"]').click();
    
    // Wait for optimization to complete
    cy.get('[data-testid="optimization-progress"]', { timeout: 60000 }).should('not.exist');
    
    // Results should be displayed
    cy.get('[data-testid="optimization-results"]').should('be.visible');
    cy.get('[data-testid="parameter-heatmap"]').should('be.visible');
    cy.get('[data-testid="top-parameters"]').should('be.visible');
    
    // Should show best parameters
    cy.get('[data-testid="best-parameters"]').should('be.visible');
    cy.get('[data-testid="best-parameters"]').within(() => {
      cy.contains('Fast Period').should('be.visible');
      cy.contains('Slow Period').should('be.visible');
    });
  });

  it('should save and load backtest results', () => {
    // Run a backtest
    cy.get('[data-testid="strategy-selector"]').click();
    cy.contains('Moving Average Crossover').click();
    
    cy.get('[data-testid="asset-selector"]').click();
    cy.contains('BTC').click();
    
    cy.get('[data-testid="date-range-picker"]').click();
    cy.get('[data-testid="date-range-picker"]').within(() => {
      cy.get('[data-testid="start-date"]').type('2023-01-01');
      cy.get('[data-testid="end-date"]').type('2023-12-31');
    });
    cy.get('body').click(); // Close date picker
    
    // Submit form
    cy.get('[data-testid="backtest-form"] button[type="submit"]').click();
    
    // Wait for backtest to complete
    cy.get('[data-testid="backtest-progress"]', { timeout: 30000 }).should('not.exist');
    
    // Save this backtest with a unique name
    const backtestName = `Test Backtest ${Date.now()}`;
    cy.get('[data-testid="save-backtest"]').click();
    cy.get('[data-testid="backtest-name-input"]').type(backtestName);
    cy.get('[data-testid="save-backtest-confirm"]').click();
    
    // Go to saved backtests
    cy.get('[data-testid="saved-backtests-tab"]').click();
    
    // Find and load the saved backtest
    cy.contains(backtestName).click();
    
    // Backtest results should be displayed
    cy.get('[data-testid="backtest-results"]').should('be.visible');
    cy.get('[data-testid="equity-curve"]').should('be.visible');
    cy.get('[data-testid="trade-list"]').should('be.visible');
    cy.get('[data-testid="performance-metrics"]').should('be.visible');
  });
});
