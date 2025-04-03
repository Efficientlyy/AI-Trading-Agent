"""
Admin Controller

This module provides comprehensive administrative controls for the dashboard.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psutil
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("admin_controller")

class AdminAction(Enum):
    """Admin action enum"""
    VIEW = 1
    EDIT = 2
    CREATE = 3
    DELETE = 4
    RESTART = 5
    BACKUP = 6
    RESTORE = 7

class AdminPermission(Enum):
    """Admin permission enum"""
    SYSTEM = 1
    CONFIG = 2
    USER = 3
    DATA = 4
    LOG = 5

class AdminResult:
    """Admin result class"""
    
    def __init__(self, 
                 success: bool, 
                 message: str = None,
                 data: Any = None):
        """
        Initialize admin result.
        
        Args:
            success: Whether the action was successful
            message: Result message
            data: Result data
        """
        self.success = success
        self.message = message
        self.data = data
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert admin result to dictionary.
        
        Returns:
            Dictionary representation of admin result
        """
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        """
        String representation of admin result.
        
        Returns:
            String representation
        """
        return json.dumps(self.to_dict(), indent=2)

class AdminController:
    """
    Comprehensive administrative controller for the dashboard.
    """
    
    def __init__(self, 
                 config_dir: str = None,
                 data_dir: str = None,
                 log_dir: str = None,
                 backup_dir: str = None):
        """
        Initialize admin controller.
        
        Args:
            config_dir: Configuration directory
            data_dir: Data directory
            log_dir: Log directory
            backup_dir: Backup directory
        """
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
        
        logger.info("Admin controller initialized")
    
    def get_system_status(self) -> AdminResult:
        """
        Get system status.
        
        Returns:
            Admin result with system status
        """
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
    
    def get_services_status(self) -> Dict[str, Any]:
        """
        Get services status.
        
        Returns:
            Dictionary with services status
        """
        services_status = {}
        
        for service_name, service_info in self.services.items():
            # Check if service is running
            is_running = False
            pid = service_info.get('pid')
            
            if pid:
                try:
                    process = psutil.Process(pid)
                    is_running = process.is_running()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    is_running = False
            
            # Get service status
            services_status[service_name] = {
                'running': is_running,
                'pid': pid,
                'start_time': service_info.get('start_time'),
                'restart_count': service_info.get('restart_count', 0),
                'last_restart': service_info.get('last_restart')
            }
        
        return services_status
    
    def start_service(self, service_name: str) -> AdminResult:
        """
        Start a service.
        
        Args:
            service_name: Service name
            
        Returns:
            Admin result
        """
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
    
    def stop_service(self, service_name: str) -> AdminResult:
        """
        Stop a service.
        
        Args:
            service_name: Service name
            
        Returns:
            Admin result
        """
        if service_name not in self.services:
            return AdminResult(
                success=False,
                message=f"Service '{service_name}' not found"
            )
        
        service_info = self.services[service_name]
        
        # Check if service is running
        pid = service_info.get('pid')
        if not pid:
            return AdminResult(
                success=False,
                message=f"Service '{service_name}' is not running"
            )
        
        # Stop service
        try:
            process = psutil.Process(pid)
            process.terminate()
            
            # Wait for process to terminate
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                # Force kill if timeout
                process.kill()
            
            # Update service info
            service_info['pid'] = None
            
            return AdminResult(
                success=True,
                message=f"Service '{service_name}' stopped successfully"
            )
        except psutil.NoSuchProcess:
            # Process already terminated
            service_info['pid'] = None
            
            return AdminResult(
                success=True,
                message=f"Service '{service_name}' was not running"
            )
        except Exception as e:
            logger.error(f"Error stopping service '{service_name}': {e}")
            return AdminResult(
                success=False,
                message=f"Error stopping service '{service_name}': {str(e)}"
            )
    
    def restart_service(self, service_name: str) -> AdminResult:
        """
        Restart a service.
        
        Args:
            service_name: Service name
            
        Returns:
            Admin result
        """
        # Stop service
        stop_result = self.stop_service(service_name)
        if not stop_result.success and "not running" not in stop_result.message:
            return stop_result
        
        # Wait a moment
        time.sleep(1)
        
        # Start service
        return self.start_service(service_name)
    
    def get_config(self, config_name: str) -> AdminResult:
        """
        Get configuration.
        
        Args:
            config_name: Configuration name
            
        Returns:
            Admin result with configuration
        """
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
    
    def save_config(self, 
                    config_name: str, 
                    config_data: Dict[str, Any],
                    create_backup: bool = True) -> AdminResult:
        """
        Save configuration.
        
        Args:
            config_name: Configuration name
            config_data: Configuration data
            create_backup: Whether to create a backup of the existing configuration
            
        Returns:
            Admin result
        """
        config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        
        # Create backup if requested
        if create_backup and os.path.exists(config_path):
            backup_result = self.backup_config(config_name)
            if not backup_result.success:
                return backup_result
        
        # Save configuration
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            return AdminResult(
                success=True,
                message=f"Configuration '{config_name}' saved successfully"
            )
        except Exception as e:
            logger.error(f"Error saving configuration '{config_name}': {e}")
            return AdminResult(
                success=False,
                message=f"Error saving configuration '{config_name}': {str(e)}"
            )

# Create default admin controller instance
default_admin_controller = AdminController()
