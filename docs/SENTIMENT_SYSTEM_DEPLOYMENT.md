# Sentiment Analysis System Deployment Guide

This document provides instructions for deploying, configuring, and monitoring the enhanced sentiment analysis system with LLM integration.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment Process](#deployment-process)
- [Configuration](#configuration)
- [Verification](#verification)
- [Monitoring](#monitoring)
- [Performance Tracking](#performance-tracking)
- [Troubleshooting](#troubleshooting)

## Overview

The sentiment analysis system includes several advanced components:

1. **LLM Service**: Connects to OpenAI, Anthropic, and Azure APIs for language model-based analysis
2. **LLM Sentiment Agent**: Processes financial text using LLMs for sophisticated sentiment analysis
3. **Consensus System**: Combines outputs from multiple sentiment sources with Bayesian aggregation
4. **Real-Time Event Detection**: Identifies market-moving events as they occur
5. **Bidirectional Integration**: Connects sentiment analysis and event detection systems
6. **Performance Tracking**: Records and uses historical model performance to improve accuracy
7. **Dashboard Visualization**: Displays consensus data, model performance, and detected events

## Prerequisites

Before deploying the system, ensure you have:

1. **API Keys**: Set up and configure all required API keys
   - `OPENAI_API_KEY`: Required for OpenAI LLM access
   - `TWITTER_API_KEY`: Required for social media sentiment analysis
   - `CRYPTOCOMPARE_API_KEY`: Required for crypto news data
   - `NEWS_API_KEY`: Required for general news data
   - `ANTHROPIC_API_KEY`: Optional, for Claude models
   - `AZURE_OPENAI_API_KEY`: Optional, for Azure OpenAI deployment

2. **Python Environment**: Python 3.8+ with all dependencies installed
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration Files**: Ensure all configuration files are available
   - `config/sentiment_analysis.yaml`
   - `config/early_detection.yaml`
   - `config/dashboard.yaml`

## Deployment Process

The deployment process is automated using the `deploy_sentiment_system.py` script. This script handles the entire deployment process, including configuration updates, testing, and verification.

### Basic Deployment

For a basic deployment to the development environment:

```bash
python deploy_sentiment_system.py --config deployments/deployment_config_dev.yaml --environment dev
```

### Production Deployment

For production deployment (requires passing all tests):

```bash
python deploy_sentiment_system.py --config deployments/deployment_config_prod.yaml --environment prod
```

### Deployment Options

The deployment script supports the following options:

- `--config`: Path to the deployment configuration file
- `--environment`: Deployment environment (dev, staging, prod)
- `--skip-tests`: Skip running system tests

### Deployment Process Steps

1. **API Key Validation**: Checks all required API keys are available
2. **System Testing**: Runs comprehensive system tests
3. **Configuration Updates**: Updates all configuration files
4. **Deployment Verification**: Ensures all components are properly deployed
5. **Report Generation**: Creates a detailed deployment report

## Configuration

The system uses YAML configuration files for all components. The main configuration files are:

### Sentiment Analysis Configuration (`config/sentiment_analysis.yaml`)

```yaml
# Core settings
enabled: true

# LLM settings
llm:
  primary_model: "gpt-4o"
  financial_model: "gpt-4o"
  batch_size: 10
  use_cached_responses: true
  cache_ttl: 1800  # 30 minutes

# Consensus system settings
consensus_system:
  enabled: true
  weights:
    llm: 0.35
    news: 0.25
    social_media: 0.20
    market: 0.10
    onchain: 0.10
  adaptive_weights: true
  performance_tracking: true
  tracking_window: 1440  # 24 hours
  confidence_threshold: 0.75
```

### Early Detection Configuration (`config/early_detection.yaml`)

```yaml
# Core settings
enabled: true

# Realtime detector settings
realtime_detector:
  enabled: true
  refresh_interval: 180  # 3 minutes
  source_weights:
    twitter: 0.25
    news: 0.45
    reddit: 0.20
    alerts: 0.10
  minimum_score_threshold: 0.75

# Sentiment integration settings
sentiment_integration:
  enabled: true
  bidirectional: true
  event_impact_threshold: 0.75
```

### Dashboard Configuration (`config/dashboard.yaml`)

```yaml
# LLM Event Dashboard settings
llm_event_dashboard:
  enabled: true
  port: 8051
  max_display_items: 100
  update_interval: 15
  theme: "dark"
  charts:
    sentiment_trend: true
    confidence_distribution: true
    event_timeline: true
    model_performance: true
```

## Verification

After deployment, you should verify that the system is working correctly using the verification script:

```bash
python verify_sentiment_deployment.py --environment dev
```

For a more thorough verification including LLM API calls:

```bash
python verify_sentiment_deployment.py --environment prod --detailed
```

The verification script checks:

1. Import availability of all required modules
2. Configuration values
3. API key availability
4. Component initialization
5. Detailed tests (if requested)

## Monitoring

The system includes several monitoring capabilities:

### Dashboard

The LLM Event Dashboard provides real-time visualization of:

- Sentiment consensus data
- Model performance metrics
- Detected events and signals
- Confidence calibration
- Historical accuracy

To launch the dashboard:

```bash
python run_dashboard.py --dashboard llm_event
```

### Logging

All system components use structured logging. Logs are stored in the `logs/` directory and include:

- Component initialization and shutdown
- API calls and responses
- Sentiment events and updates
- Performance metrics
- System health

To view logs in real time:

```bash
tail -f logs/sentiment_system.log
```

### Performance Testing

Regularly test the system's performance using:

```bash
python run_sentiment_tests.py
```

This runs comprehensive tests and generates a report in both JSON and HTML formats.

## Performance Tracking

The system includes a performance tracking component that:

1. Records predictions and their outcomes
2. Calculates accuracy, precision, and recall metrics
3. Calibrates confidence scores based on historical performance
4. Adjusts source weights adaptively

Performance data is stored in `data/performance/sentiment_performance.json` and includes:

- Direction accuracy (how often the direction prediction was correct)
- Value accuracy (how close the sentiment value was to actual market movement)
- Calibration error (difference between confidence and actual accuracy)
- Weighted accuracy (recency-weighted performance metric)

The performance tracker automatically adjusts the system to emphasize more accurate sources over time.

## Troubleshooting

### Common Issues

#### API Connection Problems

If experiencing API connection issues:

1. Verify API keys are set correctly in environment variables
2. Check network connectivity
3. Verify API rate limits haven't been exceeded
4. Look for specific error messages in the logs

#### Component Initialization Failures

If components fail to initialize:

1. Check logs for specific error messages
2. Verify all configuration files are valid
3. Ensure all dependencies are installed
4. Restart the system

#### Dashboard Not Displaying Data

If the dashboard isn't showing data:

1. Verify the sentiment system is running
2. Check event bus connections
3. Verify database connectivity
4. Check for JavaScript console errors

### Getting Help

For additional assistance:

1. Consult the component-specific documentation
2. Review the logs for detailed error messages
3. Check the deployment report for warnings or errors
4. Contact the sentiment system maintainer