# System Overseer Documentation

## Overview

The System Overseer is a comprehensive monitoring and control module for the Trading-Agent system. It provides a conversational interface through Telegram, allowing users to interact with the trading system using natural language, monitor system status, adjust parameters, and receive insights about market conditions and trading performance.

## Architecture

The System Overseer is built with a modular, plugin-based architecture that allows for easy extension and customization:

### Core Components

1. **Module Registry**: Central service locator that manages all system components
2. **Config Registry**: Manages system parameters with validation and persistence
3. **Event Bus**: Enables communication between components through events
4. **System Core**: Coordinates all components and manages system lifecycle

### Conversational Components

1. **LLM Client**: Interfaces with language models for natural language processing
2. **Dialogue Manager**: Handles conversation flow and context management
3. **Personality System**: Provides human-like interaction with customizable traits
4. **Telegram Integration**: Connects the system to Telegram for user interaction

### Plugin System

1. **Plugin Manager**: Loads and manages plugins dynamically
2. **Extension Points**: Predefined hooks for extending system functionality
3. **Analytics Plugins**: Provide market insights and trading analysis
4. **Visualization Plugins**: Generate charts and visual representations of data

## Installation

### Prerequisites

- Python 3.8+
- pip
- Telegram Bot Token (from BotFather)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Efficientlyy/AI-Trading-Agent.git
   cd AI-Trading-Agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## Configuration

The System Overseer is configured through a combination of environment variables and configuration files:

### Environment Variables

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `TELEGRAM_CHAT_ID`: Your Telegram chat ID
- `LLM_API_KEY`: API key for the language model provider
- `LLM_PROVIDER`: Language model provider (openai, anthropic, etc.)

### Configuration Files

- `config/system.json`: System-wide configuration
- `config/personality.json`: Personality traits configuration
- `config/plugins.json`: Plugin configuration

## Usage

### Starting the System

To start the System Overseer:

```bash
python system_overseer_service.py
```

This will start the service in the background and connect to Telegram.

### Telegram Commands

While the System Overseer primarily uses natural language, it also supports these commands:

- `/help`: Show available commands and capabilities
- `/status`: Show system status
- `/pairs`: List active trading pairs
- `/add_pair SYMBOL`: Add a trading pair (e.g., `/add_pair ETHUSDC`)
- `/remove_pair SYMBOL`: Remove a trading pair (e.g., `/remove_pair ETHUSDC`)
- `/notifications LEVEL`: Set notification level (all, signals, trades, errors, none)

### Natural Language Interaction

You can interact with the System Overseer using natural language:

- "How is the market doing today?"
- "What's the status of the trading system?"
- "Change the risk level to conservative"
- "Tell me about recent trades"
- "Are there any issues I should know about?"

## Extending the System

### Creating Custom Plugins

Custom plugins can be created by implementing the `Plugin` interface:

```python
from system_overseer.plugin_manager import Plugin

class CustomPlugin(Plugin):
    def __init__(self):
        super().__init__("custom_plugin", "Custom Plugin")
    
    def initialize(self, system_core):
        # Initialize plugin
        pass
    
    def start(self):
        # Start plugin
        pass
    
    def stop(self):
        # Stop plugin
        pass
```

### Registering Plugins

Plugins are registered in the `config/plugins.json` file:

```json
{
  "plugins": [
    {
      "id": "custom_plugin",
      "path": "plugins.custom_plugin.CustomPlugin",
      "enabled": true,
      "config": {
        "option1": "value1",
        "option2": "value2"
      }
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **Telegram Connection Issues**:
   - Verify your `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
   - Ensure the bot has been added to your chat
   - Check internet connectivity

2. **LLM Integration Issues**:
   - Verify your `LLM_API_KEY`
   - Check API rate limits
   - Ensure the selected model is available

3. **Plugin Loading Issues**:
   - Check plugin path in configuration
   - Verify plugin implements required interfaces
   - Check for Python import errors

### Logs

Logs are stored in the following locations:

- `logs/system_overseer.log`: Main system log
- `logs/telegram.log`: Telegram integration log
- `logs/llm.log`: Language model integration log

## Support

For support, please open an issue on the GitHub repository or contact the development team.
