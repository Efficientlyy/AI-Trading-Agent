/**
 * End-to-end tests for authentication functionality
 */

describe('Authentication', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    cy.clearLocalStorage();
    cy.visit('/login');
  });

  it('should display login form', () => {
    cy.get('form').should('be.visible');
    cy.get('input[name="username"]').should('be.visible');
    cy.get('input[name="password"]').should('be.visible');
    cy.get('button[type="submit"]').should('be.visible');
  });

  it('should show validation errors for empty fields', () => {
    cy.get('button[type="submit"]').click();
    cy.get('form').contains('Username is required').should('be.visible');
    cy.get('form').contains('Password is required').should('be.visible');
  });

  it('should show error for invalid credentials', () => {
    cy.get('input[name="username"]').type('invalid_user');
    cy.get('input[name="password"]').type('invalid_password');
    cy.get('button[type="submit"]').click();
    
    // Wait for error message
    cy.contains('Invalid username or password').should('be.visible');
  });

  it('should redirect to dashboard after successful login', () => {
    // Use test account credentials
    cy.get('input[name="username"]').type('test_user');
    cy.get('input[name="password"]').type('test_password');
    cy.get('button[type="submit"]').click();
    
    // Should redirect to dashboard
    cy.url().should('include', '/');
    cy.contains('Dashboard').should('be.visible');
  });

  it('should navigate to registration page', () => {
    cy.contains('Register').click();
    cy.url().should('include', '/register');
    
    // Registration form should be visible
    cy.get('input[name="username"]').should('be.visible');
    cy.get('input[name="email"]').should('be.visible');
    cy.get('input[name="password"]').should('be.visible');
    cy.get('input[name="confirmPassword"]').should('be.visible');
  });

  it('should register a new user', () => {
    cy.visit('/register');
    
    // Generate unique username
    const username = `test_user_${Date.now().toString().slice(-6)}`;
    
    cy.get('input[name="username"]').type(username);
    cy.get('input[name="email"]').type(`${username}@example.com`);
    cy.get('input[name="password"]').type('Test@123');
    cy.get('input[name="confirmPassword"]').type('Test@123');
    
    cy.get('button[type="submit"]').click();
    
    // Should redirect to login page with success message
    cy.url().should('include', '/login');
    cy.contains('Registration successful').should('be.visible');
  });

  it('should logout successfully', () => {
    // Login first
    cy.get('input[name="username"]').type('test_user');
    cy.get('input[name="password"]').type('test_password');
    cy.get('button[type="submit"]').click();
    
    // Should be on dashboard
    cy.url().should('include', '/');
    
    // Click logout
    cy.get('[data-testid="user-menu"]').click();
    cy.contains('Logout').click();
    
    // Should redirect to login page
    cy.url().should('include', '/login');
  });
});
