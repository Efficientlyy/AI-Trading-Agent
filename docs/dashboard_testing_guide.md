# Modern Dashboard Testing Guide

This guide explains the testing setup for the AI Trading Agent's modern dashboard implementation.

## Testing Architecture

The testing architecture is designed to verify the functionality of all key dashboard components through different types of tests:

1. **DataService and Theme System Tests**: Tests for data retrieval, caching, and theme management
2. **WebSocket Tests**: Tests for real-time data updates
3. **Flask Routes and Authentication Tests**: Tests for API endpoints and user authorization
4. **UI Component Tests**: Tests for UI elements like notifications, guided tour, and tooltips
5. **Performance Optimization Tests**: Tests for lazy loading, chunked rendering, and other performance features
6. **Data Export Tests**: Tests for exporting data in various formats

## Test Files

The following test files have been implemented:

- `test_modern_dashboard.py`: Tests for the DataService class and theme system
- `test_websocket.py`: Tests for WebSocket functionality and data emissions
- `test_flask_routes.py`: Tests for Flask routes and authentication
- `test_frontend_utils.py`: Tests for UI components and performance optimizations
- `test_notifications_export.py`: Tests for notification system and data export functionality

## Running Tests

To run the dashboard tests, use the `run_dashboard_tests.py` script:

```bash
# Run all tests
python run_dashboard_tests.py

# Run tests for specific components
python run_dashboard_tests.py --data-service
python run_dashboard_tests.py --websocket
python run_dashboard_tests.py --flask-routes
python run_dashboard_tests.py --ui-components
python run_dashboard_tests.py --performance
python run_dashboard_tests.py --export

# Run tests from a specific file
python run_dashboard_tests.py --file=test_modern_dashboard.py

# Generate code coverage report
python run_dashboard_tests.py --coverage

# Run tests with verbose output
python run_dashboard_tests.py --verbose

# Combine options
python run_dashboard_tests.py --data-service --websocket --coverage
```

## Code Coverage

The testing setup includes code coverage measurement using pytest-cov. Coverage reports are generated in two formats:

1. Terminal output (for quick reference)
2. HTML report at `reports/dashboard_coverage/index.html` (for detailed analysis)

To generate a code coverage report, use the `--coverage` flag:

```bash
python run_dashboard_tests.py --coverage
```

## Testing Strategy

Each component type has a specific testing approach:

### DataService and Theme System

- Test caching mechanisms with different expiry times
- Test data source switching between mock and real data
- Test fallback to mock data when real data sources fail
- Test data refresh and expiration
- Test theme persistence and switching between light/dark modes

### WebSocket Communication

- Test event emission for different data types
- Test connection handling and initialization
- Test reconnection logic during network issues
- Test data update broadcasting to clients
- Test client-initiated data requests

### Flask Routes and Authentication

- Test route protection with login requirements
- Test role-based access control
- Test login validation and error handling
- Test password hashing and verification
- Test user registration and management
- Test API endpoints for data retrieval

### UI Components

- Test notification creation and management
- Test settings persistence in localStorage
- Test guided tour functionality
- Test tooltip initialization and display
- Test interactive controls and state management

### Performance Optimizations

- Test lazy loading of tab content
- Test chunked rendering for large tables
- Test pagination for large datasets
- Test debouncing and throttling for user interactions
- Test caching mechanisms for API responses

### Data Export Functionality

- Test export to different formats (CSV, JSON, Excel, PDF)
- Test scheduled export functionality
- Test export configuration and persistence
- Test export file generation and downloading

## Mock Objects

The tests use various mock objects to isolate components and simulate behavior:

- `MockDataGenerator`: For generating controllable test data
- `MockDocument` and `MockElement`: For testing DOM manipulation
- `MockLocalStorage`: For testing browser storage
- `MockSocketIO`: For testing WebSocket communication
- `MockEvent`: For simulating DOM events
- `MockTour`: For testing guided tour functionality

## Extending Tests

When adding new features to the dashboard, follow these steps to maintain test coverage:

1. Identify the appropriate test file for your new feature
2. Create focused test methods for each aspect of the feature
3. Use the appropriate mock objects to simulate dependencies
4. Ensure tests verify both success cases and error handling
5. Run the tests with coverage to identify any gaps
6. Update the test runner if needed for new test categories

## Continuous Integration

The dashboard tests are integrated into the project's CI pipeline through `.github/workflows/dashboard-tests.yml`, which:

1. Runs all dashboard tests on each push
2. Generates code coverage reports
3. Fails the build if coverage drops below thresholds
4. Archives test artifacts for review

## Troubleshooting

If tests fail, check these common issues:

1. **Import errors**: Make sure all required modules are installed
2. **Mock conflicts**: Ensure mocks don't interfere with each other
3. **Test isolation**: Each test should clean up after itself
4. **Async timing**: For WebSocket tests, ensure proper handling of async operations
5. **DOM manipulation**: Verify DOM mock objects accurately simulate browser behavior
6. **Local storage**: Check localStorage mock implementation for state persistence issues