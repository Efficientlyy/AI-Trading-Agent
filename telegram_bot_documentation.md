# Telegram Bot Documentation

## Overview

The Telegram bot component of the Trading-Agent system provides command-based control and notifications for the trading system. It allows users to manage active trading pairs, control notification levels, and receive real-time updates about trading signals and executed trades.

## Features

- **Command Processing**: Handles user commands to control the trading system
- **Notification System**: Sends real-time notifications for signals, trades, errors, and system events
- **Settings Management**: Allows configuration of active trading pairs and notification levels
- **Persistent Storage**: Saves settings to a JSON file for persistence across restarts

## Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Show available commands | `/help` |
| `/status` | Show system status | `/status` |
| `/pairs` | Show active trading pairs | `/pairs` |
| `/add_pair SYMBOL` | Add trading pair | `/add_pair ETHUSDC` |
| `/remove_pair SYMBOL` | Remove trading pair | `/remove_pair ETHUSDC` |
| `/notifications LEVEL` | Set notification level | `/notifications all` |

## Notification Levels

- `all`: All notifications (signals, trades, errors, system)
- `signals`: Only signal notifications
- `trades`: Only trade notifications
- `errors`: Only error notifications
- `none`: No notifications

## Implementation Details

### Components

1. **EnhancedTelegramNotifier**: Main class that handles notifications and commands
2. **TelegramSettings**: Class that manages and persists settings
3. **TelegramBotService**: Service wrapper for running the bot as a background process

### Files

- `improved_telegram_settings_command.py`: Enhanced implementation with better error handling
- `fixed_telegram_bot_service.py`: Service wrapper for running the bot in the background
- `telegram_settings.json`: Persistent storage for settings
- `telegram_bot_test.py`: Test script for verifying command functionality

### Environment Variables

The bot requires the following environment variables:

- `TELEGRAM_BOT_TOKEN`: Token for the Telegram bot (from BotFather)
- `TELEGRAM_USER_ID` or `TELEGRAM_CHAT_ID`: ID of the authorized chat

## Setup and Running

### Prerequisites

1. Create a Telegram bot using BotFather
2. Get the bot token and chat ID
3. Set the environment variables in `.env-secure/.env`

### Starting the Bot Service

```bash
# Make the service script executable
chmod +x fixed_telegram_bot_service.py

# Start the service in the background
nohup python3 fixed_telegram_bot_service.py > telegram_bot_service.log 2>&1 &
```

### Stopping the Bot Service

```bash
# Find the process ID
ps aux | grep python | grep telegram_bot_service

# Kill the process
kill <process_id>
```

## Troubleshooting

### Common Issues

1. **Commands not working**: Ensure the bot service is running in the background
2. **Unauthorized messages**: Verify the chat ID matches the one in the environment variables
3. **Missing notifications**: Check the notification level setting

### Logs

The bot service logs to:
- `telegram_bot_service.log`: Main service log
- `fixed_telegram_bot_service.out`: Output from the background process

## Future Improvements

1. **Web Interface**: Add a web dashboard for managing settings
2. **Multiple Chat Support**: Allow multiple authorized users/chats
3. **Advanced Commands**: Add more sophisticated commands for trading strategy control
4. **Scheduled Reports**: Add support for scheduled performance reports
5. **Interactive Charts**: Send interactive charts for market data and performance
