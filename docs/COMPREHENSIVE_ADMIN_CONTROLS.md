# Comprehensive Admin Controls

This document provides an overview of the Comprehensive Admin Controls implemented in Phase 3 of the Real Data Integration project.

## Overview

The Comprehensive Admin Controls provide a unified administrative interface for managing the dashboard, data sources, and system configuration. It enables administrators to monitor system health, manage configurations, view logs, and control services through a user-friendly interface.

## Architecture

The Comprehensive Admin Controls consist of the following components:

1. **Admin Controller**: A Python class that provides administrative functionality.
2. **Admin UI**: User interface components for system administration.
3. **Configuration Management**: Tools for managing system configurations.
4. **Service Control**: Functionality for starting, stopping, and monitoring services.
5. **User Management**: Tools for managing user accounts and permissions.

## Features

### Advanced Configuration Management

- **Configuration Versioning**: Maintains a history of configuration changes.
- **Configuration Templates**: Provides templates for quick setup.
- **Import/Export Functionality**: Allows exporting and importing configurations.
- **Configuration Validation**: Validates configuration changes before applying them.

### System Monitoring and Control

- **Detailed System Resource Monitoring**: Monitors CPU, memory, disk usage, and more.
- **Service Control**: Provides controls for starting, stopping, and restarting services.
- **Performance Optimization Tools**: Helps identify and resolve performance bottlenecks.
- **System Alerts and Notifications**: Notifies administrators of system issues.

### User Management

- **Role-Based Access Control**: Controls access based on user roles.
- **User Activity Monitoring**: Tracks user actions for security and auditing.
- **User Preference Management**: Allows users to customize their experience.
- **Approval Workflows**: Requires approval for sensitive operations.

### Data Source Administration

- **Data Source Registration**: Allows registering and deregistering data sources.
- **Source-Specific Configuration**: Provides configuration options for each data source.
- **Testing Tools**: Enables testing data source connections.
- **Performance Analytics**: Monitors data source performance.

## Implementation Details

### Admin Controller

The `AdminController` class is the core component of the admin controls system. It provides methods for managing configurations, services, logs, and users.

```python
class AdminController:
    def __init__(self, 
                 config_dir=None,
                 data_dir=None,
                 log_dir=None,
                 backup_dir=None):
        # Set directories
        self.config_dir = config_dir or os.path.join(os.getcwd(), 'config')
        self.data_dir = data_dir or os.path.join(os.getcwd(), 'data')
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'logs')
        self.backup_dir = backup_dir or os.path.join(os.getcwd(), 'backups')
        
        # Create directories if they don't exist
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Initialize services
        self.services = {}
        
        # Initialize users
        self.users = {}
        
        # Initialize permissions
        self.permissions = {}
```

### Admin Result

The `AdminResult` class represents the result of an administrative operation. It includes information about whether the operation was successful, a message describing the result, and any data returned by the operation.

```python
class AdminResult:
    def __init__(self, 
                 success: bool, 
                 message: str = None,
                 data: Any = None):
        self.success = success
        self.message = message
        self.data = data
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp
        }
```

### System Status Monitoring

The admin controller provides methods for monitoring system status, including CPU usage, memory usage, disk usage, and service status.

```python
def get_system_status(self) -> AdminResult:
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        
        # Get process information
        process = psutil.Process()
        process_cpu = process.cpu_percent()
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Get service status
        services_status = self.get_services_status()
        
        # Create status data
        status_data = {
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used': memory.used / (1024 * 1024 * 1024),  # GB
                'memory_total': memory.total / (1024 * 1024 * 1024),  # GB
                'disk_percent': disk.percent,
                'disk_used': disk.used / (1024 * 1024 * 1024),  # GB
                'disk_total': disk.total / (1024 * 1024 * 1024),  # GB
                'boot_time': boot_time.isoformat(),
                'uptime': str(uptime)
            },
            'process': {
                'cpu_percent': process_cpu,
                'memory_mb': process_memory
            },
            'services': services_status
        }
        
        return AdminResult(
            success=True,
            message="System status retrieved successfully",
            data=status_data
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return AdminResult(
            success=False,
            message=f"Error getting system status: {str(e)}"
        )
```

### Service Management

The admin controller provides methods for managing services, including starting, stopping, and restarting services.

```python
def start_service(self, service_name: str) -> AdminResult:
    if service_name not in self.services:
        return AdminResult(
            success=False,
            message=f"Service '{service_name}' not found"
        )
    
    service_info = self.services[service_name]
    
    # Check if service is already running
    if service_info.get('pid'):
        try:
            process = psutil.Process(service_info['pid'])
            if process.is_running():
                return AdminResult(
                    success=False,
                    message=f"Service '{service_name}' is already running"
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Start service
    try:
        command = service_info.get('command')
        if not command:
            return AdminResult(
                success=False,
                message=f"No command specified for service '{service_name}'"
            )
        
        # Start process
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Update service info
        service_info['pid'] = process.pid
        service_info['start_time'] = datetime.now().isoformat()
        service_info['restart_count'] = service_info.get('restart_count', 0) + 1
        service_info['last_restart'] = datetime.now().isoformat()
        
        return AdminResult(
            success=True,
            message=f"Service '{service_name}' started successfully",
            data={
                'pid': process.pid,
                'start_time': service_info['start_time']
            }
        )
    except Exception as e:
        logger.error(f"Error starting service '{service_name}': {e}")
        return AdminResult(
            success=False,
            message=f"Error starting service '{service_name}': {str(e)}"
        )
```

### Configuration Management

The admin controller provides methods for managing configurations, including getting, saving, backing up, and restoring configurations.

```python
def get_config(self, config_name: str) -> AdminResult:
    config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
    
    # Check if configuration exists
    if not os.path.exists(config_path):
        return AdminResult(
            success=False,
            message=f"Configuration '{config_name}' not found"
        )
    
    # Read configuration
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return AdminResult(
            success=True,
            message=f"Configuration '{config_name}' retrieved successfully",
            data=config_data
        )
    except Exception as e:
        logger.error(f"Error reading configuration '{config_name}': {e}")
        return AdminResult(
            success=False,
            message=f"Error reading configuration '{config_name}': {str(e)}"
        )
```

## UI Components

The Comprehensive Admin Controls include UI components for system administration:

1. **Admin Panel**: The main container for admin controls.
2. **System Status Section**: Displays system resource usage and service status.
3. **Configuration Section**: Provides tools for managing configurations.
4. **Logs Section**: Displays system logs with filtering options.
5. **User Management Section**: Provides tools for managing users.
6. **Backup Section**: Allows backing up and restoring configurations.

### Admin Panel

The admin panel is the main container for admin controls. It includes a navigation bar for switching between different sections.

```html
<div class="admin-panel">
    <div class="admin-panel-header">
        <h3>Admin Controls</h3>
        <div class="actions">
            <button class="btn btn-primary refresh-admin">
                <i data-feather="refresh-cw"></i> Refresh
            </button>
        </div>
    </div>
    <div class="admin-panel-body">
        <div class="admin-nav">
            <div class="admin-nav-item active" data-section="system-status">System Status</div>
            <div class="admin-nav-item" data-section="configuration">Configuration</div>
            <div class="admin-nav-item" data-section="logs">Logs</div>
            <div class="admin-nav-item" data-section="users">Users</div>
            <div class="admin-nav-item" data-section="backups">Backups</div>
        </div>
        <div class="admin-content">
            <!-- Section content goes here -->
        </div>
    </div>
</div>
```

### System Status Section

The system status section displays system resource usage and service status.

```html
<div class="admin-section active" id="system-status">
    <div class="system-status">
        <div class="status-cards">
            <div class="status-card good">
                <div class="status-card-header">
                    <div class="status-card-icon">
                        <i data-feather="cpu"></i>
                    </div>
                    <h4 class="status-card-title">CPU Usage</h4>
                </div>
                <div class="status-card-value">25%</div>
                <div class="status-card-subtitle">4 cores, 3.2 GHz</div>
            </div>
            <div class="status-card good">
                <div class="status-card-header">
                    <div class="status-card-icon">
                        <i data-feather="database"></i>
                    </div>
                    <h4 class="status-card-title">Memory Usage</h4>
                </div>
                <div class="status-card-value">4.2 GB</div>
                <div class="status-card-subtitle">16 GB total (26%)</div>
            </div>
            <div class="status-card warning">
                <div class="status-card-header">
                    <div class="status-card-icon">
                        <i data-feather="hard-drive"></i>
                    </div>
                    <h4 class="status-card-title">Disk Usage</h4>
                </div>
                <div class="status-card-value">120 GB</div>
                <div class="status-card-subtitle">250 GB total (48%)</div>
            </div>
            <div class="status-card good">
                <div class="status-card-header">
                    <div class="status-card-icon">
                        <i data-feather="clock"></i>
                    </div>
                    <h4 class="status-card-title">Uptime</h4>
                </div>
                <div class="status-card-value">5d 12h 34m</div>
                <div class="status-card-subtitle">Since Mar 25, 2025</div>
            </div>
        </div>
    </div>
    <div class="services-list">
        <div class="services-header">
            <h4>Services</h4>
            <div class="actions">
                <button class="btn btn-sm btn-primary restart-all-services">
                    <i data-feather="refresh-cw"></i> Restart All
                </button>
            </div>
        </div>
        <div class="services-grid">
            <div class="service-card">
                <div class="service-card-header">
                    <h4 class="service-name">Data Collection Service</h4>
                    <div class="service-status running">Running</div>
                </div>
                <div class="service-details">
                    <div class="service-details-item">
                        <span>PID:</span>
                        <span>12345</span>
                    </div>
                    <div class="service-details-item">
                        <span>Started:</span>
                        <span>Mar 30, 2025 08:15:22</span>
                    </div>
                    <div class="service-details-item">
                        <span>Restarts:</span>
                        <span>2</span>
                    </div>
                </div>
                <div class="service-actions">
                    <button class="btn btn-sm btn-secondary restart-service" data-service="data_collection">
                        <i data-feather="refresh-cw"></i> Restart
                    </button>
                    <button class="btn btn-sm btn-danger stop-service" data-service="data_collection">
                        <i data-feather="square"></i> Stop
                    </button>
                </div>
            </div>
        </div>
    </div>
</div>
```

## Integration with Other Components

The Comprehensive Admin Controls integrate with other components of the Real Data Integration project:

1. **Real-time Data Updates**: Provides real-time monitoring of system status.
2. **Advanced Data Validation**: Allows configuring validation rules.
3. **Data Transformation Pipeline**: Enables managing transformation pipelines.

## Usage

### System Monitoring

```python
# Get system status
admin_controller = AdminController()
status_result = admin_controller.get_system_status()

if status_result.success:
    status_data = status_result.data
    
    # Check CPU usage
    cpu_percent = status_data['system']['cpu_percent']
    if cpu_percent > 80:
        print(f"Warning: High CPU usage ({cpu_percent}%)")
    
    # Check memory usage
    memory_percent = status_data['system']['memory_percent']
    if memory_percent > 80:
        print(f"Warning: High memory usage ({memory_percent}%)")
    
    # Check disk usage
    disk_percent = status_data['system']['disk_percent']
    if disk_percent > 80:
        print(f"Warning: High disk usage ({disk_percent}%)")
else:
    print(f"Error getting system status: {status_result.message}")
```

### Service Management

```python
# Register a service
admin_controller.register_service(
    service_name="data_collection",
    command="python data_collection_service.py",
    auto_start=True
)

# Start a service
start_result = admin_controller.start_service("data_collection")
if start_result.success:
    print(f"Service started with PID {start_result.data['pid']}")
else:
    print(f"Error starting service: {start_result.message}")

# Stop a service
stop_result = admin_controller.stop_service("data_collection")
if stop_result.success:
    print("Service stopped successfully")
else:
    print(f"Error stopping service: {stop_result.message}")

# Restart a service
restart_result = admin_controller.restart_service("data_collection")
if restart_result.success:
    print(f"Service restarted with PID {restart_result.data['pid']}")
else:
    print(f"Error restarting service: {restart_result.message}")
```

### Configuration Management

```python
# Get configuration
config_result = admin_controller.get_config("system")
if config_result.success:
    config_data = config_result.data
    
    # Modify configuration
    config_data['log_level'] = 'DEBUG'
    
    # Save configuration
    save_result = admin_controller.save_config("system", config_data)
    if save_result.success:
        print("Configuration saved successfully")
    else:
        print(f"Error saving configuration: {save_result.message}")
else:
    print(f"Error getting configuration: {config_result.message}")

# Backup configuration
backup_result = admin_controller.backup_config("system")
if backup_result.success:
    print(f"Configuration backed up to {backup_result.data['backup_path']}")
else:
    print(f"Error backing up configuration: {backup_result.message}")

# Restore configuration
restore_result = admin_controller.restore_config("system")
if restore_result.success:
    print(f"Configuration restored from {restore_result.data['backup_path']}")
else:
    print(f"Error restoring configuration: {restore_result.message}")
```

## Future Enhancements

Planned enhancements for the Comprehensive Admin Controls include:

1. **Remote Administration**: Enable administering the system remotely.
2. **Advanced Monitoring**: Implement more detailed system monitoring.
3. **Automated Maintenance**: Add automated maintenance tasks.
4. **Audit Logging**: Enhance logging of administrative actions.
5. **Multi-User Administration**: Support multiple administrators with different roles.

## Conclusion

The Comprehensive Admin Controls provide a powerful and user-friendly interface for administering the AI Trading Agent dashboard. By centralizing administrative functions and providing detailed monitoring and control capabilities, it enhances the manageability and reliability of the system.