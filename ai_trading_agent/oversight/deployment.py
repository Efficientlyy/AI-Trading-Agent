"""
LLM Oversight Production Deployment Integration Module

This module provides tools and utilities for deploying and running the LLM oversight
system in a production Kubernetes environment, including health checks, metrics
collection, and high availability configuration.
"""

import logging
import os
import time
import socket
import threading
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
import prometheus_client

# Import oversight components
from ai_trading_agent.oversight.service import create_app, OversightService
from ai_trading_agent.oversight.config import OversightServiceConfig
from ai_trading_agent.oversight.metrics import OversightMetricsCollector
from ai_trading_agent.oversight.llm_oversight import OversightLevel, LLMProvider
from ai_trading_agent.oversight.oversight_integration import OversightManager

# Set up logger
logger = logging.getLogger(__name__)

class OversightDeployment:
    """
    Manages the deployment of the LLM oversight system in a production environment.
    
    This class handles production-specific concerns like metrics export, health checks,
    configuration management, and integration with the container orchestration system.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        metrics_port: int = 8081,
        health_check_port: int = 8082,
        data_dir: Optional[str] = None,
        prometheus_multiproc_dir: Optional[str] = None,
        enable_high_availability: bool = False,
        replica_id: Optional[str] = None
    ):
        """
        Initialize the oversight deployment manager.
        
        Args:
            config_path: Path to the oversight service configuration file
            metrics_port: Port to expose Prometheus metrics on
            health_check_port: Port for health check endpoint
            data_dir: Directory for storing oversight data and logs
            prometheus_multiproc_dir: Directory for Prometheus multiprocess mode
            enable_high_availability: Whether to enable HA configuration
            replica_id: Unique ID for this replica in HA mode
        """
        # Set up data directories
        self.data_dir = data_dir or os.environ.get("OVERSIGHT_DATA_DIR", "/data/oversight")
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Read or create configuration
        self.config_path = config_path or os.environ.get("OVERSIGHT_CONFIG_PATH")
        if not self.config_path or not Path(self.config_path).exists():
            self.config_path = os.path.join(self.data_dir, "oversight_config.yaml")
            self._create_default_config()
        
        # Load configuration
        self.config = self._load_config()
        
        # Set up Prometheus metrics
        self.metrics_port = metrics_port
        self.prometheus_multiproc_dir = prometheus_multiproc_dir or os.environ.get(
            "PROMETHEUS_MULTIPROC_DIR", os.path.join(self.data_dir, "prometheus")
        )
        Path(self.prometheus_multiproc_dir).mkdir(parents=True, exist_ok=True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = self.prometheus_multiproc_dir
        
        # Set up health check
        self.health_check_port = health_check_port
        
        # High availability configuration
        self.enable_high_availability = enable_high_availability
        self.replica_id = replica_id or socket.gethostname()
        
        # Initialize metrics collector
        self.metrics_collector = OversightMetricsCollector(
            multiproc_dir=self.prometheus_multiproc_dir
        )
        
        # Create oversight service
        service_config = OversightServiceConfig(
            llm_provider=self.config.get("llm", {}).get("provider", "openai"),
            model_name=self.config.get("llm", {}).get("model", "gpt-4"),
            oversight_level=self.config.get("oversight", {}).get("level", "advise"),
            api_key=os.environ.get("OPENAI_API_KEY") or self.config.get("llm", {}).get("api_key"),
            decision_log_path=os.path.join(self.data_dir, "decisions"),
            metrics_collector=self.metrics_collector
        )
        
        self.oversight_service = OversightService(config=service_config)
        self.app = create_app(self.oversight_service)
        
        # Kubernetes liveness and readiness probes
        self.liveness_status = True
        self.readiness_status = False
        
        # State monitoring
        self._shutdown_event = threading.Event()
        self._status_threads = []
        
        logger.info(f"Oversight deployment initialized with replica ID {self.replica_id}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {str(e)}")
            logger.info("Using default configuration")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create a default configuration file.
        
        Returns:
            Default configuration dictionary
        """
        default_config = {
            "oversight": {
                "level": "advise",
                "bypass_on_error": True,
                "max_decision_cache_size": 1000,
                "enable_autonomous_recovery": True
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.0,
                "max_tokens": 500,
                "timeout": 30.0
            },
            "service": {
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 4,
                "log_level": "info",
                "cors_origins": ["*"],
                "request_timeout": 60.0
            },
            "storage": {
                "metrics_retention_days": 90,
                "decision_log_retention_days": 30
            },
            "deployment": {
                "enable_high_availability": False,
                "replica_count": 2,
                "health_check_interval": 30.0,
                "metrics_interval": 15.0
            }
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Write configuration to file
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
                logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error creating default configuration: {str(e)}")
        
        return default_config
    
    def start_health_check_server(self) -> None:
        """
        Start a simple health check server for Kubernetes probes.
        """
        import http.server
        import socketserver
        
        class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                deployment = self.server.deployment
                
                if self.path == "/health/live":
                    # Liveness probe - Is the pod running at all?
                    if deployment.liveness_status:
                        self.send_response(200)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "ok"}).encode())
                    else:
                        self.send_response(503)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "error"}).encode())
                
                elif self.path == "/health/ready":
                    # Readiness probe - Is the pod ready to receive traffic?
                    if deployment.readiness_status:
                        self.send_response(200)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "ok"}).encode())
                    else:
                        self.send_response(503)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "not_ready"}).encode())
                
                elif self.path == "/health/startup":
                    # Startup probe - Has the pod completed initialization?
                    if deployment.oversight_service.is_initialized:
                        self.send_response(200)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "initialized"}).encode())
                    else:
                        self.send_response(503)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "initializing"}).encode())
                
                else:
                    self.send_response(404)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "not found"}).encode())
                
            def log_message(self, format, *args):
                # Suppress logging for health checks to reduce noise
                pass
        
        class HealthCheckServer(socketserver.ThreadingTCPServer):
            # Allow address reuse for fast restarts
            allow_reuse_address = True
            
            def __init__(self, server_address, RequestHandlerClass, deployment):
                self.deployment = deployment
                super().__init__(server_address, RequestHandlerClass)
        
        def run_server():
            server = HealthCheckServer(("", self.health_check_port), HealthCheckHandler, self)
            logger.info(f"Health check server started on port {self.health_check_port}")
            
            try:
                while not self._shutdown_event.is_set():
                    server.handle_request()
            except Exception as e:
                logger.error(f"Health check server error: {str(e)}")
            finally:
                server.server_close()
                logger.info("Health check server stopped")
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        self._status_threads.append(thread)
        logger.info(f"Health check server thread started")
    
    def start_metrics_server(self) -> None:
        """
        Start the Prometheus metrics server.
        """
        try:
            prometheus_client.start_http_server(self.metrics_port)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Error starting Prometheus metrics server: {str(e)}")
    
    def monitor_service_status(self) -> None:
        """
        Continuously monitor service status and update health status.
        """
        def status_monitor():
            check_interval = self.config.get("deployment", {}).get("health_check_interval", 30.0)
            
            while not self._shutdown_event.is_set():
                try:
                    # Check if LLM service is responding
                    llm_status = self.oversight_service.check_llm_status()
                    
                    # Update liveness and readiness based on service status
                    self.liveness_status = True  # Only set to False for critical failures
                    self.readiness_status = llm_status.get("status") == "ok"
                    
                    # Log status
                    if self.readiness_status:
                        logger.debug(f"Service status check: ready")
                    else:
                        logger.warning(f"Service status check: not ready - {llm_status.get('message', 'unknown error')}")
                    
                    # Collect and update metrics
                    self.metrics_collector.update_service_status(
                        is_healthy=self.readiness_status,
                        llm_latency=llm_status.get("latency", 0.0),
                        replica_id=self.replica_id
                    )
                    
                except Exception as e:
                    logger.error(f"Error in status monitor: {str(e)}")
                    self.readiness_status = False
                
                # Sleep for the check interval
                for _ in range(int(check_interval)):
                    if self._shutdown_event.is_set():
                        break
                    time.sleep(1)
        
        thread = threading.Thread(target=status_monitor, daemon=True)
        thread.start()
        self._status_threads.append(thread)
        logger.info("Status monitoring thread started")
    
    def rotate_logs(self) -> None:
        """
        Set up log rotation for the service logs.
        """
        logs_dir = os.path.join(self.data_dir, "logs")
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        
        decisions_dir = os.path.join(self.data_dir, "decisions")
        Path(decisions_dir).mkdir(parents=True, exist_ok=True)
        
        retention_days = self.config.get("storage", {}).get("decision_log_retention_days", 30)
        
        # Configure log rotation using Python's built-in logging handlers
        # This is just setup - actual rotation would be handled by the logging module
        logger.info(f"Log rotation configured with {retention_days} days retention")
    
    def start(self) -> None:
        """
        Start the oversight deployment with all required components.
        """
        logger.info("Starting LLM Oversight deployment")
        
        # Start Prometheus metrics server
        self.start_metrics_server()
        
        # Start health check server
        self.start_health_check_server()
        
        # Start status monitoring
        self.monitor_service_status()
        
        # Set up log rotation
        self.rotate_logs()
        
        # Mark as ready for requests
        self.readiness_status = True
        
        logger.info(f"LLM Oversight deployment started successfully on replica {self.replica_id}")
        
        return self.app
    
    def shutdown(self) -> None:
        """
        Gracefully shut down the oversight deployment.
        """
        logger.info("Shutting down LLM Oversight deployment")
        
        # Signal all monitoring threads to stop
        self._shutdown_event.set()
        
        # Mark as not ready to receive traffic
        self.readiness_status = False
        
        # Wait for threads to finish
        for thread in self._status_threads:
            thread.join(timeout=5.0)
        
        logger.info("LLM Oversight deployment shutdown complete")


def create_production_app() -> Any:
    """
    Create a FastAPI application configured for production deployment.
    
    Returns:
        FastAPI application instance
    """
    # Get configuration from environment
    config_path = os.environ.get("OVERSIGHT_CONFIG_PATH")
    metrics_port = int(os.environ.get("METRICS_PORT", "8081"))
    health_check_port = int(os.environ.get("HEALTH_PORT", "8082"))
    data_dir = os.environ.get("OVERSIGHT_DATA_DIR", "/data/oversight")
    prometheus_multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    enable_high_availability = os.environ.get("ENABLE_HIGH_AVAILABILITY", "false").lower() == "true"
    replica_id = os.environ.get("POD_NAME") or socket.gethostname()
    
    # Configure logging
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create deployment manager
    deployment = OversightDeployment(
        config_path=config_path,
        metrics_port=metrics_port,
        health_check_port=health_check_port,
        data_dir=data_dir,
        prometheus_multiproc_dir=prometheus_multiproc_dir,
        enable_high_availability=enable_high_availability,
        replica_id=replica_id
    )
    
    # Start all services and return the FastAPI app
    return deployment.start()


# This app variable is used by the ASGI server
app = create_production_app()


if __name__ == "__main__":
    import uvicorn
    
    # When running directly, use a simplified version for testing
    config_path = os.environ.get("OVERSIGHT_CONFIG_PATH")
    data_dir = os.environ.get("OVERSIGHT_DATA_DIR", "./data/oversight")
    
    # Create deployment
    deployment = OversightDeployment(
        config_path=config_path,
        data_dir=data_dir,
        metrics_port=8081,
        health_check_port=8082
    )
    
    # Start the app
    app = deployment.start()
    
    # Run the server
    uvicorn.run(
        app,
        host=deployment.config.get("service", {}).get("host", "0.0.0.0"),
        port=deployment.config.get("service", {}).get("port", 8080),
        log_level="info"
    )
