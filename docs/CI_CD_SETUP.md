# CI/CD Setup Guide for AI Trading Agent

This document provides a comprehensive guide to the Continuous Integration and Continuous Deployment (CI/CD) setup for the AI Trading Agent project.

## Table of Contents

1. [Overview](#overview)
2. [GitHub Actions Workflow](#github-actions-workflow)
3. [Testing Strategy](#testing-strategy)
4. [Code Coverage Requirements](#code-coverage-requirements)
5. [Deployment Process](#deployment-process)
6. [Environment Configuration](#environment-configuration)
7. [Troubleshooting](#troubleshooting)

## Overview

The AI Trading Agent project uses GitHub Actions for continuous integration and testing. The CI/CD pipeline ensures code quality, prevents regressions, and automates the deployment process.

Key components:
- Automated testing for all components (Python ML, Rust, Frontend)
- Code coverage enforcement
- End-to-end testing
- Automated deployment to staging and production environments

## GitHub Actions Workflow

The CI workflow is defined in `.github/workflows/ci.yml` and includes the following jobs:

### Python ML Components
- Tests Python ML code across multiple Python versions (3.8, 3.9, 3.10)
- Runs tests for different modules (detection, models, evaluation)
- Enforces code coverage thresholds specific to each module

### Rust Components
- Runs Clippy for static analysis
- Executes tests with coverage reporting
- Enforces a minimum of 80% code coverage

### Frontend Components
- Tests across multiple Node.js versions (16.x, 18.x, 20.x)
- Runs linting checks
- Executes unit and integration tests
- Enforces 80% code coverage for branches, functions, lines, and statements

### End-to-End Tests
- Uses Cypress for E2E testing
- Records videos of test runs for debugging
- Uploads screenshots and videos as artifacts

### Coverage Status Check
- Aggregates results from all test jobs
- Ensures all components meet their coverage thresholds

## Testing Strategy

### Unit Tests
- Test individual functions and components in isolation
- Mock external dependencies
- Focus on edge cases and error handling

### Integration Tests
- Test interactions between components
- Verify API contracts
- Test data flow between modules

### End-to-End Tests
- Test complete user workflows
- Verify UI functionality
- Test real-world scenarios

## Code Coverage Requirements

| Component | Minimum Coverage |
|-----------|------------------|
| ML Detection | 85% |
| ML Models | 80% |
| ML Evaluation | 75% |
| Rust | 80% |
| Frontend | 80% |

## Deployment Process

The deployment process is automated through GitHub Actions:

1. **Build Phase**
   - Compile and bundle the application
   - Run tests to ensure quality
   - Generate deployment artifacts

2. **Staging Deployment**
   - Automatically deploy to staging environment on successful builds from the `develop` branch
   - Run smoke tests to verify deployment

3. **Production Deployment**
   - Manual approval required
   - Deploy to production from the `main` branch
   - Run verification tests post-deployment

## Environment Configuration

The CI/CD pipeline uses environment variables for configuration:

### Required Environment Variables
- `API_KEY`: API key for external services
- `DATABASE_URL`: Connection string for the database
- `DEPLOYMENT_TOKEN`: Token for deployment service

### Environment-Specific Configuration
- Development: `.env.development`
- Testing: `.env.test`
- Staging: `.env.staging`
- Production: `.env.production`

## Troubleshooting

### Common Issues

1. **Failed Tests**
   - Check the test logs for specific errors
   - Verify that all dependencies are installed
   - Ensure environment variables are properly set

2. **Coverage Threshold Not Met**
   - Add tests for uncovered code paths
   - Check if tests are properly configured to report coverage

3. **Deployment Failures**
   - Verify deployment credentials
   - Check network connectivity to deployment services
   - Ensure build artifacts are correctly generated

### Getting Help

If you encounter issues with the CI/CD pipeline:
1. Check the GitHub Actions logs for detailed error messages
2. Review the documentation for the specific tools (Jest, Cypress, etc.)
3. Contact the DevOps team for assistance

---

*Last updated: 2023-11-15*
