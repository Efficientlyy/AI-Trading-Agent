# Production Deployment Guide

This document provides comprehensive instructions for deploying the AI Crypto Trading System in a production environment. It covers system requirements, configuration, security considerations, monitoring setup, and best practices for reliability.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Security Configuration](#security-configuration)
3. [Environment Setup](#environment-setup)
4. [Running the System](#running-the-system)
5. [Monitoring & Alerting](#monitoring--alerting)
6. [Backup & Recovery](#backup--recovery)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)

## System Requirements

### Hardware Requirements

For optimal performance, we recommend:

- **CPU**: 4+ cores (8+ cores for high-frequency strategies)
- **RAM**: 16GB minimum (32GB+ recommended)
- **Storage**: SSD with at least 100GB free space
- **Network**: Stable, low-latency internet connection (under 100ms to exchange APIs)

### Software Requirements

- **Operating System**: Ubuntu 20.04 LTS or newer (recommended), or other Linux distributions
- **Python**: 3.8 or newer
- **Rust**: Latest stable version (if using Rust optimizations)
- **Database**: PostgreSQL 12+ (for historical data storage)
- **Redis**: 6.0+ (for caching and pub/sub)

### Python Dependencies

All dependencies are specified in `requirements.txt`, but core components include:

- numpy, pandas, scikit-learn
- aiohttp, websockets
- plotly, dash (for dashboard)
- sqlalchemy, asyncpg
- pydantic
- structlog, rich

## Security Configuration

### API Key Management

The system uses a secure API key management system:

1. **Setup the API Key Store**:

   ```bash
   # Create a secure directory for API keys
   mkdir -p ~/.trading_system/keys
   chmod 700 ~/.trading_system/keys
   ```

2. **Configure Key Storage**:

   Edit your `config/security.yaml` file to specify the key storage location:

   ```yaml
   security:
     api_keys:
       storage_type: file
       file_path: ~/.trading_system/keys
       encryption_enabled: true
   ```

3. **Add Exchange API Keys**:

   ```bash
   # Use the utility script to securely add exchange API keys
   python setup_exchange_credentials.py --exchange binance
   ```

### Network Security

1. **Firewall Configuration**:
   
   Configure your firewall to allow only necessary outbound connections:

   ```bash
   # Example UFW configuration
   sudo ufw default deny incoming
   sudo ufw default allow outgoing
   sudo ufw allow ssh
   # If exposing dashboard
   sudo ufw allow 8050/tcp
   sudo ufw enable
   ```

2. **VPN Usage** (Recommended):
   
   For additional security, run the trading system behind a VPN:

   ```bash
   # Install OpenVPN client
   sudo apt install openvpn
   # Configure VPN connection (specific to your provider)
   sudo openvpn --config /path/to/your-vpn-config.ovpn
   ```

3. **SSH Hardening**:

   If accessing the server remotely, secure SSH:

   ```bash
   # Edit SSH config
   sudo nano /etc/ssh/sshd_config
   
   # Recommended settings
   PermitRootLogin no
   PasswordAuthentication no
   X11Forwarding no
   ```

### Secrets Management

For sensitive data beyond API keys:

1. Environment variables (for simple deployments):
   
   ```bash
   export TRADING_SECURITY__SECRET_KEY="your-secret-key"
   ```

2. Or use a secrets management service like HashiCorp Vault for enterprise deployments.

## Environment Setup

### Setting Up Production Environment

1. **Create a dedicated user**:

   ```bash
   sudo adduser trading
   sudo usermod -aG sudo trading
   ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/ai-trading-agent.git
   cd ai-trading-agent
   ```

3. **Create and activate virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**:

   ```bash
   pip install --no-cache-dir -r requirements.txt
   pip install --no-cache-dir -r requirements-prod.txt
   ```

5. **Install Rust components** (if using):

   ```bash
   cd rust
   cargo build --release
   ```

6. **Initialize database** (if using):

   ```bash
   python -m src.data_collection.persistence.db_init
   ```

### Configuration for Production

Create a production configuration file:

```bash
cp config/development.yaml config/production.yaml
```

Edit `config/production.yaml` with production-specific settings:

```yaml
system:
  environment: production
  debug_mode: false
  log_level: INFO

data_collection:
  persistence:
    storage_type: database
    database_url: postgresql://user:password@localhost/trading_data

execution:
  paper_trading: false  # Set to true initially for testing
  retry_attempts: 5
  retry_delay: 1.5
  cancel_orders_on_shutdown: true
  
# Other production settings...
```

## Running the System

### Starting the System

Use the production launcher script:

```bash
python run_trading_system.py --env production
```

### Options and Flags

The launcher supports several options:

```
--config-dir DIR    Path to configuration directory (default: config)
--env ENV           Environment to run in (development, testing, production)
--log-level LEVEL   Override logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
--dry-run           Validate configuration without executing trades
--component COMP    Specific components to enable (can use multiple times)
--skip-component    Specific components to disable (can use multiple times)
--validate-keys     Validate API keys before starting
```

### Running as a Service

For reliable operation, configure the system as a systemd service:

1. Create a service file:

   ```bash
   sudo nano /etc/systemd/system/trading-system.service
   ```

2. Add the service configuration:

   ```ini
   [Unit]
   Description=AI Crypto Trading System
   After=network.target postgresql.service redis.service
   
   [Service]
   Type=simple
   User=trading
   WorkingDirectory=/home/trading/ai-trading-agent
   ExecStart=/home/trading/ai-trading-agent/venv/bin/python run_trading_system.py --env production
   Restart=on-failure
   RestartSec=10
   StandardOutput=journal
   StandardError=journal
   
   # Environment variables
   Environment=PYTHONUNBUFFERED=1
   
   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:

   ```bash
   sudo systemctl enable trading-system
   sudo systemctl start trading-system
   ```

4. Check the status:

   ```bash
   sudo systemctl status trading-system
   ```

## Monitoring & Alerting

### System Monitoring

1. **Logging**:

   Logs are stored in the `logs` directory. For production, consider forwarding logs to a centralized system like ELK Stack or Graylog.

2. **Metrics**:

   Use the built-in monitoring dashboard:

   ```bash
   python run_dashboard.py --monitoring
   ```

3. **Resource Monitoring**:

   Install and configure monitoring tools:

   ```bash
   sudo apt install htop iotop netstat
   ```

### Trading Monitoring

1. **Dashboard**:

   The system includes a comprehensive dashboard:

   ```bash
   python run_dashboard.py --port 8050
   ```

2. **Alerts Configuration**:

   Configure alerts in `config/monitoring.yaml`:

   ```yaml
   monitoring:
     alerts:
       email:
         enabled: true
         smtp_server: smtp.example.com
         smtp_port: 587
         username: your_email@example.com
         password: your_password
         recipients:
           - alerts@example.com
       
       # Other alert channels (Telegram, SMS, etc.)
   ```

## Backup & Recovery

### Data Backup

1. **Database Backup**:

   ```bash
   # For PostgreSQL
   pg_dump -U username trading_db > backup_$(date +%Y%m%d).sql
   ```

2. **Configuration Backup**:

   ```bash
   tar -czf config_backup_$(date +%Y%m%d).tar.gz config/
   ```

3. **Automated Backups**:

   Create a cron job:

   ```bash
   crontab -e
   
   # Add this line to back up daily at 1am
   0 1 * * * /home/trading/ai-trading-agent/scripts/backup.sh
   ```

### Recovery Procedures

1. **Configuration Recovery**:

   ```bash
   tar -xzf config_backup_YYYYMMDD.tar.gz -C /tmp/
   cp -R /tmp/config/* ./config/
   ```

2. **Database Recovery**:

   ```bash
   psql -U username trading_db < backup_YYYYMMDD.sql
   ```

## Troubleshooting

### Common Issues

1. **API Connection Failures**:

   - Check internet connectivity
   - Verify API keys are valid and have correct permissions
   - Look for IP restrictions on exchange accounts

2. **System Performance Issues**:

   - Check CPU and memory usage with `htop`
   - Check disk I/O with `iotop`
   - Look for Python processes consuming excessive resources

3. **Logging Issues**:

   If logs are not being written:

   - Check disk space with `df -h`
   - Verify permissions on log directory
   - Check for log rotation failures

### Diagnostic Commands

```bash
# View system logs
sudo journalctl -u trading-system -f

# Check service status
sudo systemctl status trading-system

# Resource monitoring
htop
```

## Performance Tuning

### Python Performance

1. **Use PyPy** for non-numeric components:

   ```bash
   pypy3 -m pip install -r requirements.txt
   pypy3 run_trading_system.py
   ```

2. **Enable Rust Extensions**:

   Set in config:

   ```yaml
   system:
     use_rust_extensions: true
   ```

### Process Prioritization

For critical tasks, adjust process priority:

```bash
# Run with higher priority
nice -n -10 python run_trading_system.py
```

### Network Optimization

1. **Reduce Network Latency**:

   ```bash
   # Measure latency to exchanges
   ping api.binance.com
   
   # Use traceroute to diagnose
   traceroute api.binance.com
   ```

2. **Use location-optimized servers**:

   Deploy in data centers close to exchange APIs where possible.

---

For further assistance, contact support or refer to the internal documentation. This guide will be updated as the system evolves.