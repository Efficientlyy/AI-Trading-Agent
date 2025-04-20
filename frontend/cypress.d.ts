/// <reference types="cypress" />

declare namespace Cypress {
  interface Chainable {
    /**
     * Custom command to login with email and password
     * @example cy.login('user@example.com', 'password123')
     */
    login(email: string, password: string): Chainable<Element>;
    
    /**
     * Custom command to select asset by symbol
     * @example cy.selectAsset('BTC-USD')
     */
    selectAsset(symbol: string): Chainable<Element>;
    
    /**
     * Custom command to place a market order
     * @example cy.placeMarketOrder('buy', '0.1')
     */
    placeMarketOrder(side: 'buy' | 'sell', quantity: string): Chainable<Element>;
    
    /**
     * Custom command to place a limit order
     * @example cy.placeLimitOrder('buy', '0.1', '50000')
     */
    placeLimitOrder(side: 'buy' | 'sell', quantity: string, price: string): Chainable<Element>;
    
    /**
     * Custom command to verify order confirmation
     * @example cy.verifyOrderConfirmation('BTC-USD', 'buy', '0.1')
     */
    verifyOrderConfirmation(symbol: string, side: 'buy' | 'sell', quantity: string): Chainable<Element>;
    
    /**
     * Custom command to navigate to a specific page
     * @example cy.navigateTo('trade')
     */
    navigateTo(page: 'trade' | 'portfolio' | 'analytics' | 'settings'): Chainable<Element>;
  }
}
