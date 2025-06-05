#!/usr/bin/env bash
# System Overseer Deployment Script
# This script deploys the System Overseer service

# Exit on error
set -e

# Configuration
CONFIG_DIR="./config"
DATA_DIR="./data"
LOG_DIR="./logs"

# Create directories
mkdir -p "$CONFIG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 and try again."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Check for environment file
if [ ! -f ".env" ]; then
    echo "Creating default .env file..."
    cp .env-example .env
    echo "Please edit .env file with your API keys and settings."
fi

# Create default configuration
if [ ! -f "$CONFIG_DIR/system.json" ]; then
    echo "Creating default system configuration..."
    cat > "$CONFIG_DIR/system.json" << EOF
{
  "system": {
    "name": "Trading System Overseer",
    "version": "1.0.0"
  },
  "trading": {
    "default_pairs": ["BTCUSDC", "ETHUSDC", "SOLUSDC"],
    "risk_level": "moderate"
  },
  "notifications": {
    "level": "all",
    "frequency": "medium"
  }
}
EOF
fi

# Create default personality configuration
if [ ! -f "$CONFIG_DIR/personality.json" ]; then
    echo "Creating default personality configuration..."
    cat > "$CONFIG_DIR/personality.json" << EOF
{
  "traits": {
    "formality": 0.7,
    "verbosity": 0.6,
    "helpfulness": 0.9,
    "proactivity": 0.8
  },
  "memory": {
    "retention_period": 7,
    "importance_threshold": 0.5
  }
}
EOF
fi

# Create default plugins configuration
if [ ! -f "$CONFIG_DIR/plugins.json" ]; then
    echo "Creating default plugins configuration..."
    cat > "$CONFIG_DIR/plugins.json" << EOF
{
  "plugins": [
    {
      "id": "trading_analytics",
      "path": "plugins.trading_analytics.trading_analytics_plugin.TradingAnalyticsPlugin",
      "enabled": true,
      "config": {}
    }
  ]
}
EOF
fi

# Check if service is already running
if pgrep -f "python3 system_overseer_service.py" > /dev/null; then
    echo "System Overseer service is already running. Stopping it..."
    pkill -f "python3 system_overseer_service.py"
    sleep 2
fi

# Start the service
echo "Starting System Overseer service..."
nohup python3 system_overseer_service.py --config-dir "$CONFIG_DIR" --data-dir "$DATA_DIR" > "$LOG_DIR/service.log" 2>&1 &

# Check if service started successfully
sleep 2
if pgrep -f "python3 system_overseer_service.py" > /dev/null; then
    echo "System Overseer service started successfully!"
    echo "Log file: $LOG_DIR/service.log"
else
    echo "Failed to start System Overseer service. Check the log file for details."
    exit 1
fi

echo "Deployment complete!"
