# Notification & Control System

## Overview

The Notification & Control System enables remote monitoring and interaction with the trading system through messaging platforms (WhatsApp and Telegram). It provides real-time alerts about system status and trading activities while allowing remote command execution for system control.

## Key Responsibilities

- Send configurable notifications about system events and trading activities
- Provide real-time status updates and performance metrics
- Process incoming commands from authorized users
- Implement secure authentication for remote control
- Format messages appropriately for different platforms
- Manage notification frequency to prevent flooding

## Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Notification & Control System             │
│                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│  │ Event       │   │ Notification│   │ Message     │    │
│  │ Listener    │──▶│ Manager     │──▶│ Formatter   │    │
│  └─────────────┘   └─────────────┘   └──────┬──────┘    │
│         ▲                                    │          │
│         │                                    ▼          │
│  ┌─────────────┐                    ┌─────────────┐     │
│  │ Command     │◀───────────────────│ Platform    │     │
│  │ Processor   │                    │ Connectors  │     │
│  └─────────────┘                    └─────────────┘     │
│         │                                               │
└─────────┼───────────────────────────────────────────────┘
          │
          ▼
   ┌─────────────┐
   │ System      │
   │ Controller  │
   └─────────────┘
```

## Subcomponents

### 1. Event Listener

Monitors system events and triggers notifications:

- Subscribes to system events from all components
- Filters events based on notification rules
- Processes event data for notification content
- Triggers appropriate notification types

### 2. Notification Manager

Manages notification logic and rules:

- Implements notification rules and priorities
- Handles notification throttling and batching
- Tracks notification delivery and confirmation
- Manages user notification preferences

### 3. Message Formatter

Creates platform-appropriate message content:

- Formats messages for different platforms
- Generates rich content (charts, tables)
- Implements message templates
- Handles message size limitations

### 4. Platform Connectors

Interfaces with messaging platforms:

- Connects to Telegram Bot API
- Integrates with WhatsApp Business API
- Handles authentication and connection management
- Processes incoming and outgoing messages

### 5. Command Processor

Processes incoming commands from users:

- Parses command syntax and parameters
- Validates command authorization
- Implements command workflows
- Provides command feedback and confirmation

### 6. System Controller

Executes commands and interfaces with other components:

- Translates commands to system actions
- Interacts with other system components
- Validates command execution
- Reports execution results

## Notification Types

The system supports several categories of notifications:

### Critical Alerts

High-priority notifications requiring immediate attention:

- System failures or component outages
- Significant losses or drawdowns
- Unusual market conditions
- Security-related alerts

### Trade Notifications

Updates related to trading activities:

- New trade signals and entries
- Take-profit and stop-loss triggers
- Position adjustments
- Trade completions with results

### Performance Updates

Regular updates about system performance:

- Daily/weekly performance summaries
- Milestone achievements
- Strategy performance metrics
- Account balance updates

### System Status

Information about system operations:

- Component status changes
- Configuration updates
- Scheduled maintenance
- Data quality issues

## Command Categories

The system supports various command types:

### Status Commands

Retrieve information about system state:

- `status`: Get overall system status
- `positions`: List open positions
- `performance`: Get performance metrics
- `balance`: Get account balance

### Trading Commands

Control trading activities:

- `pause trading`: Temporarily stop new trades
- `resume trading`: Resume normal trading
- `close position <id>`: Close specific position
- `close all`: Close all open positions

### Configuration Commands

Modify system settings:

- `set risk <level>`: Adjust risk level (low/medium/high)
- `enable agent <agent_id>`: Enable specific agent
- `disable agent <agent_id>`: Disable specific agent
- `set threshold <parameter> <value>`: Adjust thresholds

### Emergency Commands

Critical control commands:

- `emergency stop`: Halt all system operations
- `system restart`: Restart the entire system
- `force close all`: Close all positions immediately

## Message Format Examples

### Notification Examples

Trade Entry Notification:
```
🚨 NEW TRADE SIGNAL 🚨

Symbol: BTCUSDT
Direction: LONG
Entry Price: $38,245.50
Stop Loss: $37,950.00 (-0.77%)
Take Profit: $39,125.00 (+2.30%)
Confidence: 92.5%

Reasons:
- Strong support bounce with increased volume
- RSI bullish divergence
- Inverted H&S pattern completion

Time: 2023-05-15 14:26:25 UTC
```

Performance Update:
```
📊 DAILY PERFORMANCE SUMMARY 📊

Date: 2023-05-15

Trades Completed: 3
Win Rate: 2/3 (66.7%)
Net Profit: +$142.35 (+1.42%)

Current Positions: 2
Unrealized P&L: +$85.12 (+0.85%)
Account Balance: $10,227.47

Best Performer: ETHUSDT (+$98.45)
```

System Alert:
```
⚠️ SYSTEM ALERT ⚠️

Data Collection module experiencing connectivity issues with Binance API.

Impact: Delayed data updates
Status: Auto-reconnection in progress (attempt 2/5)
Action Required: None at this time

Time: 2023-05-15 19:42:18 UTC
```

### Command Response Examples

Status Command:
```
System Status: ONLINE ✅

Components:
- Data Collection: ONLINE
- Analysis Agents: ONLINE
- Decision Engine: ONLINE
- Virtual Trading: ONLINE

Current Positions: 2
Win Rate (7 days): 76.2%
Account Balance: $10,227.47

Last Updated: 2023-05-15 20:15:32 UTC
```

Trading Command:
```
Command Executed: close position BTC-LONG-12345 ✅

Position Closed:
Symbol: BTCUSDT
Direction: LONG
Entry Price: $38,245.50
Exit Price: $38,510.75
Profit/Loss: +$265.25 (+0.69%)
Duration: 8h 24m

Updated Balance: $10,492.72
```

## Security Model

The system implements a comprehensive security approach:

### Authentication

Multiple authentication factors:

- API token authentication
- User-specific access codes
- Device recognition
- Time-based authentication challenges

### Authorization

Role-based command permissions:

- Admin: Full system control
- Operator: Trading operations
- Viewer: Status and information only

### Access Controls

Restrictions to limit unauthorized access:

- IP-based restrictions
- Time-window limitations
- Command rate limiting
- Unusual pattern detection

### Audit Logging

Comprehensive tracking of all interactions:

- All commands logged with timestamps
- Authentication attempts recorded
- Command execution results
- Notification delivery status

## Configuration Options

The Notification & Control System is configurable through the `config/notification.yaml` file:

```yaml
platforms:
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    chat_ids:
      admin: "${ADMIN_CHAT_ID}"
      alerts: "${ALERTS_CHAT_ID}"
    
  whatsapp:
    enabled: true
    api_key: "${WHATSAPP_API_KEY}"
    phone_numbers:
      admin: "${ADMIN_PHONE}"
      alerts: "${ALERTS_PHONE}"

notifications:
  critical:
    enabled: true
    platforms: ["telegram", "whatsapp"]
    throttle_interval: 0  # no throttling for critical
    
  trade:
    enabled: true
    platforms: ["telegram"]
    throttle_interval: 300  # seconds
    include:
      entries: true
      exits: true
      signals: false  # don't notify about signals, only actual trades
    
  performance:
    enabled: true
    platforms: ["telegram"]
    schedule:
      daily: "20:00"
      weekly: "Monday 08:00"
    
  system:
    enabled: true
    platforms: ["telegram"]
    throttle_interval: 600  # seconds

commands:
  enabled: true
  allowed_platforms: ["telegram", "whatsapp"]
  
  authentication:
    token_required: true
    whitelist_users: ["${ADMIN_USER_ID}"]
    whitelist_ips: ["${ADMIN_IP}"]
    
  authorization:
    admin_users: ["${ADMIN_USER_ID}"]
    operator_users: ["${OPERATOR_USER_ID}"]
    viewer_users: ["${VIEWER_USER_ID}"]
```

## Integration Points

### Input Interfaces
- Event system for system events
- Configuration system for notification settings
- Authentication system for user validation

### Output Interfaces
- `send_notification(type, content, priority)`: Send notification to configured platforms
- `execute_command(command, params, user_id)`: Execute a remote command
- `get_notification_history(user_id, limit)`: Get recent notification history
- `get_command_history(user_id, limit)`: Get recent command history

## Error Handling

The system implements comprehensive error handling:

- Platform connection errors: Retry with backoff, switch to alternative platform
- Message delivery failures: Retry, store for later delivery
- Command parsing errors: Provide helpful error messages
- Authentication failures: Log attempt, notify administrators

## Implementation Guidelines

- Use secure API integration for messaging platforms
- Implement proper error handling for all external communications
- Create comprehensive logging of all notifications and commands
- Use asynchronous processing for notification delivery
- Implement proper message queue for reliability
- Design for testability with mocked platform APIs
- Create clear separation between command parsing and execution
- Implement proper security measures for all remote commands
