# Exchange Setup Guide

This document provides detailed instructions for setting up and configuring the AI Trading Agent with different cryptocurrency exchanges and trading platforms.

## Table of Contents

1. [Environment Configuration](#environment-configuration)
2. [Binance Setup](#binance-setup)
3. [Coinbase Setup](#coinbase-setup)
4. [Alpaca Setup](#alpaca-setup)
5. [API Key Security Best Practices](#api-key-security-best-practices)
6. [Troubleshooting](#troubleshooting)

## Environment Configuration

The AI Trading Agent uses environment variables to manage API keys and configuration settings. Create a `.env` file in the root directory of the project with the following structure:

```
# General Configuration
NODE_ENV=development
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WEBSOCKET_URL=ws://localhost:8000/ws

# Exchange API Keys
REACT_APP_BINANCE_API_KEY=your_binance_api_key
REACT_APP_BINANCE_API_SECRET=your_binance_api_secret
REACT_APP_COINBASE_API_KEY=your_coinbase_api_key
REACT_APP_COINBASE_API_SECRET=your_coinbase_api_secret
REACT_APP_COINBASE_PASSPHRASE=your_coinbase_passphrase
REACT_APP_ALPACA_API_KEY=your_alpaca_api_key
REACT_APP_ALPACA_API_SECRET=your_alpaca_api_secret

# Feature Flags
REACT_APP_ENABLE_PAPER_TRADING=true
REACT_APP_ENABLE_CIRCUIT_BREAKER=true
REACT_APP_ENABLE_PERFORMANCE_MONITORING=true
```

For production environments, create a `.env.production` file with appropriate values.

## Binance Setup

### Step 1: Create a Binance Account

1. Go to [Binance](https://www.binance.com/en/register) and create an account
2. Complete the identity verification process (KYC)
3. Enable Two-Factor Authentication (2FA) for your account

### Step 2: Create API Keys

1. Log in to your Binance account
2. Navigate to **API Management** (found under your profile or account settings)
3. Click **Create API Key** and follow the security verification steps
4. Set the following permissions:
   - Enable Reading
   - Enable Spot & Margin Trading
   - Disable Withdrawals
5. Set IP restrictions to only allow your application server's IP address

### Step 3: Configure the AI Trading Agent

1. Add your API keys to the `.env` file:
   ```
   REACT_APP_BINANCE_API_KEY=your_binance_api_key
   REACT_APP_BINANCE_API_SECRET=your_binance_api_secret
   ```

2. Configure Binance-specific settings in the application:
   - Navigate to Settings > Exchange Configuration
   - Select Binance as your primary exchange
   - Set appropriate trading limits and risk parameters

### Step 4: Test the Connection

1. Use the API Health page to verify connectivity
2. Check that market data is being received
3. Place a small test order in paper trading mode

## Coinbase Setup

### Step 1: Create a Coinbase Pro Account

1. Go to [Coinbase Pro](https://pro.coinbase.com/) and create an account
2. Complete the identity verification process
3. Enable Two-Factor Authentication (2FA)

### Step 2: Create API Keys

1. Log in to your Coinbase Pro account
2. Navigate to **API** in the menu
3. Click **+ New API Key**
4. Set the following permissions:
   - View (required for balance and order information)
   - Trade (required for placing/canceling orders)
   - Do NOT enable Transfer (withdrawals)
5. Set IP whitelist to restrict access to your application server

### Step 3: Configure the AI Trading Agent

1. Add your API keys to the `.env` file:
   ```
   REACT_APP_COINBASE_API_KEY=your_coinbase_api_key
   REACT_APP_COINBASE_API_SECRET=your_coinbase_api_secret
   REACT_APP_COINBASE_PASSPHRASE=your_coinbase_passphrase
   ```

2. Configure Coinbase-specific settings in the application:
   - Navigate to Settings > Exchange Configuration
   - Select Coinbase as your primary exchange
   - Set appropriate trading limits and risk parameters

### Step 4: Test the Connection

1. Use the API Health page to verify connectivity
2. Check that market data is being received
3. Place a small test order in paper trading mode

## Alpaca Setup

### Step 1: Create an Alpaca Account

1. Go to [Alpaca](https://app.alpaca.markets/signup) and create an account
2. Choose between a paper trading account or a live trading account
3. Complete the identity verification process for live trading

### Step 2: Generate API Keys

1. Log in to your Alpaca dashboard
2. Navigate to **Paper API** or **Live API** depending on your needs
3. Click **Generate New Key**
4. Save both the API Key ID and Secret Key securely

### Step 3: Configure the AI Trading Agent

1. Add your API keys to the `.env` file:
   ```
   REACT_APP_ALPACA_API_KEY=your_alpaca_api_key
   REACT_APP_ALPACA_API_SECRET=your_alpaca_api_secret
   ```

2. Configure Alpaca-specific settings in the application:
   - Navigate to Settings > Exchange Configuration
   - Select Alpaca as your primary exchange
   - Set appropriate trading limits and risk parameters
   - Configure market data sources (Alpaca offers different data subscriptions)

### Step 4: Test the Connection

1. Use the API Health page to verify connectivity
2. Check that market data is being received
3. Place a small test order in paper trading mode

## API Key Security Best Practices

### Never Commit API Keys to Version Control

- Always use environment variables or secure vaults for API keys
- Add `.env` files to your `.gitignore`
- Consider using a secrets management service for production environments

### Implement Proper Access Controls

- Create dedicated API keys for each application or service
- Apply the principle of least privilege - only enable permissions that are necessary
- Use IP whitelisting to restrict access to known IP addresses

### Rotate API Keys Regularly

- Change your API keys every 30-90 days
- Immediately revoke and replace any keys that may have been compromised
- Implement a key rotation schedule and process

### Monitor API Usage

- Enable notifications for unusual API activity
- Regularly review API usage logs
- Set up alerts for failed authentication attempts

### Secure Your Application

- Implement HTTPS for all communications
- Use secure storage for API keys on your server
- Never expose API keys in client-side code
- Consider using a proxy service to make API calls

### Emergency Response Plan

1. If you suspect your API keys have been compromised:
   - Immediately disable the compromised keys
   - Generate new keys with fresh permissions
   - Review account activity for unauthorized actions
   - Change your account password and 2FA if necessary

2. Document the following information:
   - Exchange support contact information
   - Steps to disable API keys for each exchange
   - Incident response procedures

## Troubleshooting

### Common Issues and Solutions

#### Connection Errors

- **Issue**: Unable to connect to exchange API
- **Solution**: 
  - Verify internet connectivity
  - Check that API keys are correct
  - Ensure IP restrictions aren't blocking your requests
  - Verify the exchange's API status page for outages

#### Authentication Failures

- **Issue**: API returns authentication errors
- **Solution**:
  - Double-check API key and secret for typos
  - Verify that the API key hasn't expired or been revoked
  - Check that your system clock is synchronized (important for timestamp-based authentication)

#### Rate Limiting

- **Issue**: Receiving rate limit errors
- **Solution**:
  - Implement exponential backoff retry logic
  - Reduce the frequency of API calls
  - Use websocket connections for real-time data instead of REST API polling

#### Order Placement Failures

- **Issue**: Unable to place orders
- **Solution**:
  - Verify trading permissions are enabled for your API key
  - Check account balance and trading limits
  - Ensure order parameters meet exchange requirements (size, price, etc.)
  - Verify the trading pair is active and not in maintenance mode

### Exchange-Specific Troubleshooting

#### Binance

- Check Binance's [API documentation](https://binance-docs.github.io/apidocs/) for error codes
- Verify that your account has completed the necessary verification levels for trading
- For US users, ensure you're using Binance.US API endpoints

#### Coinbase

- Review Coinbase Pro's [API documentation](https://docs.pro.coinbase.com/) for specific error messages
- Ensure your API passphrase is correctly entered
- Check that your account is fully verified for the trading pairs you're attempting to use

#### Alpaca

- Consult Alpaca's [API documentation](https://alpaca.markets/docs/api-documentation/)
- Verify market hours for the assets you're trading (stocks have limited trading hours)
- Check that you're using the correct endpoint for paper vs. live trading

### Getting Help

If you continue to experience issues after trying the troubleshooting steps:

1. Check the project's GitHub Issues page for similar problems and solutions
2. Consult the exchange's developer forum or support channels
3. Reach out to the AI Trading Agent community for assistance

Remember to never share your API keys or secrets when asking for help.
