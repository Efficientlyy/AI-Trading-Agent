"""Logging system for the entire application.

This module sets up structured logging with proper formatting,
rotation, and component-specific loggers.
"""

import gzip
import json
import logging
import os
import queue
import re
import shutil
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler, SysLogHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set, Union
from zoneinfo import ZoneInfo

import structlog
from structlog.types import Processor

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import logging as google_logging
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

from src.common.config import config
from src.common.datetime_utils import format_iso, utc_now

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Default log file
DEFAULT_LOG_FILE = LOG_DIR / "crypto_trading.log"

# Default sensitive fields to mask
DEFAULT_MASKED_FIELDS = {"password", "api_key", "api_secret", "secret", "token", "access_key", "private_key"}

# Thread-local storage for request IDs
_request_id_context = {}

# Collection of metrics
_performance_metrics = {}

# Rate limiting for log messages
_rate_limited_logs = {}

# Environment constants
ENVIRONMENT = os.environ.get("APP_ENVIRONMENT", "development").lower()
ENV_CONFIG_MAPPING = {
    "development": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "include_line_numbers": True,
        "console_colors": True,
        "log_format": "text"
    },
    "testing": {
        "console_level": "INFO",
        "file_level": "DEBUG",
        "include_line_numbers": True,
        "console_colors": True,
        "log_format": "text"
    },
    "staging": {
        "console_level": "INFO",
        "file_level": "INFO",
        "include_line_numbers": True,
        "console_colors": False,
        "log_format": "json"
    },
    "production": {
        "console_level": "WARNING",
        "file_level": "INFO",
        "include_line_numbers": False,
        "console_colors": False,
        "log_format": "json",
        "use_buffering": True,
        "buffer_size": 100,
        "compress_logs": True,
        "enable_syslog": True
    }
}

class BufferedHandler(logging.Handler):
    """Handler that buffers log records and flushes them in batches."""
    
    def __init__(self, target_handler, buffer_size=100, flush_interval=5.0):
        """
        Initialize the buffered handler.
        
        Args:
            target_handler: The handler to which buffered records will be sent
            buffer_size: Maximum number of records to buffer before flushing
            flush_interval: Maximum time in seconds before auto-flushing
        """
        super().__init__()
        self.target_handler = target_handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer = []
        self.lock = threading.Lock()
        self.last_flush = time.time()
        
        # Start the background flush thread
        self.shutdown_event = threading.Event()
        self.flush_thread = threading.Thread(target=self._flush_thread, daemon=True)
        self.flush_thread.start()
    
    def emit(self, record):
        """Add a record to the buffer, flushing if necessary."""
        with self.lock:
            self.buffer.append(record)
            
            # Flush if buffer is full
            if len(self.buffer) >= self.buffer_size:
                self.flush()
    
    def flush(self):
        """Flush buffered records to the target handler."""
        with self.lock:
            if not self.buffer:
                return
                
            # Process all buffered records
            for record in self.buffer:
                self.target_handler.handle(record)
                
            # Clear the buffer
            self.buffer = []
            self.last_flush = time.time()
    
    def close(self):
        """Close the handler, flushing any remaining records."""
        self.shutdown_event.set()
        self.flush_thread.join(timeout=2.0)
        self.flush()
        self.target_handler.close()
        super().close()
    
    def _flush_thread(self):
        """Background thread to periodically flush the buffer."""
        while not self.shutdown_event.is_set():
            time.sleep(0.5)  # Check every half second
            
            # Flush if enough time has passed
            now = time.time()
            if now - self.last_flush >= self.flush_interval:
                self.flush()


class CompressedRotatingFileHandler(RotatingFileHandler):
    """
    Extended version of RotatingFileHandler that compresses rotated logs.
    """
    def __init__(self, filename, **kwargs):
        """Initialize the handler."""
        compress_mode = kwargs.pop('compress_mode', 'gz')
        self.compress_mode = compress_mode
        super().__init__(filename, **kwargs)
        
        # Check if we need to compress any existing backup logs
        self._compress_existing_logs()
        
    def _compress_existing_logs(self):
        """Compress any existing log backups that aren't compressed yet."""
        for i in range(1, self.backupCount + 1):
            source = f"{self.baseFilename}.{i}"
            if os.path.exists(source):
                # Check if it's not already compressed
                if not self._is_compressed(source):
                    self._compress_log(source)
    
    def _is_compressed(self, filename):
        """Check if a file is already compressed."""
        if self.compress_mode == 'gz':
            return filename.endswith('.gz')
        return False
    
    def _compress_log(self, source):
        """Compress a log file."""
        dest = source + '.gz'
        with open(source, 'rb') as f_in:
            with gzip.open(dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)  # Remove the original file
    
    def doRollover(self):
        """Do a rollover, as described in __init__."""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Rotate the existing log files
        if os.path.exists(self.baseFilename):
            for i in range(self.backupCount - 1, 0, -1):
                source = f"{self.baseFilename}.{i}"
                if self.compress_mode == 'gz':
                    source_compressed = source + '.gz'
                else:
                    source_compressed = source
                    
                dest = f"{self.baseFilename}.{i+1}"
                if self.compress_mode == 'gz':
                    dest_compressed = dest + '.gz'
                else:
                    dest_compressed = dest
                    
                if os.path.exists(source_compressed):
                    if os.path.exists(dest_compressed):
                        os.remove(dest_compressed)
                    os.rename(source_compressed, dest_compressed)
            
            # Get the name for the first backup
            dest = f"{self.baseFilename}.1"
            
            # Rename and compress the current log
            if os.path.exists(self.baseFilename):
                os.rename(self.baseFilename, dest)
                if self.compress_mode == 'gz':
                    self._compress_log(dest)
        
        # Open the new file
        if not self.delay:
            self.stream = self._open()


_EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_CREDIT_CARD_PATTERN = re.compile(r'\b(?:\d{4}[ -]?){3}\d{4}\b')
_SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_IP_ADDRESS_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
_PHONE_NUMBER_PATTERN = re.compile(r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b')
_UK_POSTCODE_PATTERN = re.compile(r'\b[A-Z]{1,2}[0-9][A-Z0-9]? ?[0-9][A-Z]{2}\b', re.IGNORECASE)
_US_ZIPCODE_PATTERN = re.compile(r'\b\d{5}(?:-\d{4})?\b')
_DATE_OF_BIRTH_PATTERN = re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b')

# Default PII detection patterns with their replacements
_DEFAULT_PII_PATTERNS = {
    "email": (_EMAIL_PATTERN, "[EMAIL]"),
    "credit_card": (_CREDIT_CARD_PATTERN, "[CREDIT_CARD]"),
    "ssn": (_SSN_PATTERN, "[SSN]"),
    "ip_address": (_IP_ADDRESS_PATTERN, "[IP_ADDRESS]"),
    "phone": (_PHONE_NUMBER_PATTERN, "[PHONE_NUMBER]"),
    "uk_postcode": (_UK_POSTCODE_PATTERN, "[POSTCODE]"),
    "us_zipcode": (_US_ZIPCODE_PATTERN, "[ZIPCODE]"),
    "date_of_birth": (_DATE_OF_BIRTH_PATTERN, "[DOB]")
}


def _mask_processor(pii_patterns=None, sensitive_keys=None):
    """
    Create a processor that masks sensitive information.
    
    Args:
        pii_patterns: Dictionary of name to (pattern, replacement) tuples
        sensitive_keys: List of keys to mask values for
        
    Returns:
        Processor function for masking sensitive data
    """
    # Use default patterns if none provided
    pii_patterns = pii_patterns or _DEFAULT_PII_PATTERNS
    
    # Get list of sensitive keys to mask
    mask_keys = set(sensitive_keys or [])
    mask_keys.update(config.get("system.logging.sensitive_keys", []))
    
    # Add default sensitive keys
    mask_keys.update({
        "password", "secret", "token", "api_key", "apikey", "key", "access_token", 
        "refresh_token", "auth", "credential", "credentials", "secret_key", 
        "private_key", "authorization"
    })
    
    # Only get enabled PII patterns from config
    enabled_pii_types = config.get("system.logging.pii_detection.enabled_types", list(pii_patterns.keys()))
    active_pii_patterns = {k: v for k, v in pii_patterns.items() if k in enabled_pii_types}
    
    # Decide if we should check values for PII
    scan_values_for_pii = config.get("system.logging.pii_detection.scan_values", True)
    
    # Create mask replacer function
    def mask_text(text):
        if not isinstance(text, str):
            return text
            
        masked_text = text
        for name, (pattern, replacement) in active_pii_patterns.items():
            masked_text = pattern.sub(replacement, masked_text)
            
        return masked_text
    
    # Create value masker function
    def mask_value(key, value):
        # For sensitive keys, replace entire value
        if key.lower() in mask_keys:
            if isinstance(value, str):
                if value:  # Only mask non-empty strings
                    return "***MASKED***"
            return value
            
        # For normal values, scan for PII if enabled
        if scan_values_for_pii and isinstance(value, str):
            return mask_text(value)
            
        return value
    
    # Process dictionary recursively
    def process_dict(obj):
        result = {}
        for key, value in obj.items():
            # Process nested dictionaries
            if isinstance(value, dict):
                result[key] = process_dict(value)
            # Process lists
            elif isinstance(value, list):
                result[key] = [
                    process_dict(item) if isinstance(item, dict) 
                    else mask_value(key, item) 
                    for item in value
                ]
            # Process simple values
            else:
                result[key] = mask_value(key, value)
        return result
    
    # Main processor function
    def processor(logger, method_name, event_dict):
        try:
            # Skip masking if disabled
            if not config.get("system.logging.enable_masking", True):
                return event_dict
                
            # Process the event dictionary recursively
            return process_dict(event_dict)
        except Exception:
            # If anything goes wrong, return original to avoid breaking logging
            return event_dict
            
    return processor


def configure_logging() -> None:
    """Configure the logging system based on the application configuration."""
    # Get environment-specific defaults
    env_config = ENV_CONFIG_MAPPING.get(ENVIRONMENT, ENV_CONFIG_MAPPING["development"])
    
    # Get configuration values, with environment-specific defaults
    log_level_name = config.get("system.logging.level", env_config.get("console_level", "INFO"))
    file_log_level_name = config.get("system.logging.file_level", env_config.get("file_level", log_level_name))
    log_format = config.get("system.logging.format", env_config.get("log_format", "json"))
    include_timestamps = config.get("system.logging.include_timestamps", True)
    include_line_numbers = config.get("system.logging.include_line_numbers", env_config.get("include_line_numbers", True))
    max_log_size_mb = config.get("system.logging.max_file_size_mb", 10)
    log_backup_count = config.get("system.logging.backup_count", 5)
    console_colors = config.get("system.logging.console_colors", env_config.get("console_colors", True))
    enable_masking = config.get("system.logging.enable_masking", True)
    use_buffering = config.get("system.logging.buffering.enabled", env_config.get("use_buffering", False))
    buffer_size = config.get("system.logging.buffering.size", env_config.get("buffer_size", 100))
    flush_interval = config.get("system.logging.buffering.flush_interval", 5.0)
    compress_logs = config.get("system.logging.compress_logs", env_config.get("compress_logs", False))
    enable_syslog = config.get("system.logging.syslog.enabled", env_config.get("enable_syslog", False))
    
    # Convert log level strings to constants
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    file_log_level = getattr(logging, file_log_level_name.upper(), log_level)
    
    # Set up file handler
    if compress_logs:
        file_handler = CompressedRotatingFileHandler(
            DEFAULT_LOG_FILE,
            maxBytes=max_log_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=log_backup_count,
            compress_mode='gz'
        )
    else:
        file_handler = RotatingFileHandler(
            DEFAULT_LOG_FILE,
            maxBytes=max_log_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=log_backup_count
        )
        
    # Set file handler level
    file_handler.setLevel(file_log_level)
    
    # Configure handlers
    handlers = []
    
    # Apply buffering if enabled
    if use_buffering:
        # Add buffered console handler
        console_handler = logging.StreamHandler(sys.stdout)
        buffered_console = BufferedHandler(console_handler, buffer_size, flush_interval)
        handlers.append(buffered_console)
        
        # Add buffered file handler
        buffered_file = BufferedHandler(file_handler, buffer_size, flush_interval)
        handlers.append(buffered_file)
    else:
        # Use unbuffered handlers
        handlers.append(logging.StreamHandler(sys.stdout))
        handlers.append(file_handler)
    
    # Add syslog handler if configured
    if enable_syslog:
        syslog_host = config.get("system.logging.syslog.host", "localhost")
        syslog_port = config.get("system.logging.syslog.port", 514)
        syslog_facility = config.get("system.logging.syslog.facility", SysLogHandler.LOG_USER)
        
        syslog_handler = SysLogHandler(address=(syslog_host, syslog_port), facility=syslog_facility)
        handlers.append(syslog_handler)
    
    # Set up standard logging
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=handlers,
    )
    
    # Configure processors for structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
    ]
    
    # Add optional processors based on configuration
    if include_timestamps:
        processors.append(_add_timestamp)
    
    if include_line_numbers:
        # Use try-except to handle incompatible structlog versions
        try:
            processors.append(structlog.processors.CallsiteParameterAdder(
                parameters=["module", "function", "lineno"]
            ))
        except (TypeError, KeyError):
            # Fallback for older versions or incompatible structlog
            processors.append(structlog.processors.add_log_level)

    # Add component filter if specified
    component_filter = config.get("system.logging.component_filter", None)
    if component_filter:
        processors.append(_filter_by_component)
    
    # Add masking processor if enabled
    processors.append(_mask_processor())  # Use advanced mask processor
    
    # Add request ID processor if enabled
    processors.append(_add_request_id)
    
    # Add custom processors
    processors.append(_process_datetime_objects)
    
    # Add rate limiting processor
    processors.append(_rate_limit_processor)
    
    # Add standard processors for all formats
    processors.extend([
        structlog.processors.dict_tracebacks,
        structlog.processors.ExceptionPrettyPrinter(),
    ])
    
    # Add format-specific processors
    if log_format.lower() == "json":
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Use colored output for console if enabled
        if console_colors and sys.stdout.isatty():
            try:
                # Try to use the rich renderer if available
                from rich.console import Console
                from structlog.dev import _StructlogRichHandler
                
                console = Console()
                console_processor = _StructlogRichHandler(console=console)
                processors.append(console_processor)
            except ImportError:
                # Fall back to standard ConsoleRenderer with colors
                processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=False))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set up cloud logging if enabled
    _setup_cloud_logging()
    
    # Set up tracing if enabled
    configure_tracing()
    
    # Log the current environment
    logger = structlog.get_logger("system")
    logger.info(f"Logging configured for {ENVIRONMENT} environment")


def _setup_cloud_logging():
    """Set up cloud logging if enabled and available."""
    # AWS CloudWatch Logs
    if config.get("system.logging.cloudwatch.enabled", False) and AWS_AVAILABLE:
        try:
            aws_region = config.get("system.logging.cloudwatch.region", "us-east-1")
            log_group = config.get("system.logging.cloudwatch.log_group", "crypto-trading")
            log_stream = config.get("system.logging.cloudwatch.log_stream", "app-logs")
            
            # Create CloudWatch client
            client = boto3.client('logs', region_name=aws_region)
            
            # Create CloudWatch handler
            class CloudWatchHandler(logging.Handler):
                def __init__(self, client, log_group, log_stream):
                    super().__init__()
                    self.client = client
                    self.log_group = log_group
                    self.log_stream = log_stream
                    self.sequence_token = None
                    
                    # Create log group and stream if they don't exist
                    try:
                        self.client.create_log_group(logGroupName=self.log_group)
                    except Exception:
                        pass  # Group already exists
                        
                    try:
                        self.client.create_log_stream(
                            logGroupName=self.log_group,
                            logStreamName=self.log_stream
                        )
                    except Exception:
                        pass  # Stream already exists
                
                def emit(self, record):
                    try:
                        # Format the log message
                        if isinstance(record.msg, dict):
                            log_message = json.dumps(record.msg)
                        else:
                            log_message = self.format(record)
                        
                        # Build the log event
                        log_event = {
                            'timestamp': int(time.time() * 1000),
                            'message': log_message
                        }
                        
                        # Send the log event to CloudWatch
                        kwargs = {
                            'logGroupName': self.log_group,
                            'logStreamName': self.log_stream,
                            'logEvents': [log_event]
                        }
                        
                        if self.sequence_token:
                            kwargs["sequenceToken"] = self.sequence_token
                            
                        response = self.client.put_log_events(**kwargs)
                        self.sequence_token = response['nextSequenceToken']
                    except Exception:
                        self.handleError(record)
            
            # Add CloudWatch handler to the root logger
            cloudwatch_handler = CloudWatchHandler(client, log_group, log_stream)
            cloudwatch_handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(cloudwatch_handler)
            
            print(f"CloudWatch logging configured for log group {log_group}")
        except Exception as e:
            print(f"Failed to set up CloudWatch logging: {e}")
    
    # Google Cloud Logging
    if config.get("system.logging.google_cloud.enabled", False) and GOOGLE_CLOUD_AVAILABLE:
        try:
            project_id = config.get("system.logging.google_cloud.project_id")
            log_name = config.get("system.logging.google_cloud.log_name", "crypto-trading")
            
            # Set up Google Cloud Logging client
            client = google_logging.Client(project=project_id)
            
            # Define a handler that uses the Cloud Logging client
            class GoogleCloudHandler(logging.Handler):
                def __init__(self, client, log_name):
                    super().__init__()
                    self.client = client
                    self.logger = client.logger(log_name)
                
                def emit(self, record):
                    try:
                        # Format the log message
                        if isinstance(record.msg, dict):
                            payload = record.msg
                        else:
                            payload = {'message': self.format(record)}
                        
                        # Add severity based on log level
                        severity = record.levelname
                        
                        # Write the log entry
                        self.logger.log_struct(
                            payload,
                            severity=severity
                        )
                    except Exception:
                        self.handleError(record)
            
            # Add Google Cloud handler to the root logger
            google_handler = GoogleCloudHandler(client, log_name)
            google_handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(google_handler)
            
            print(f"Google Cloud logging configured for project {project_id}")
        except Exception as e:
            print(f"Failed to set up Google Cloud logging: {e}")


def _add_timestamp(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add a formatted timestamp to the log event."""
    tz_name = config.get("system.general.timezone", "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        # Fallback to UTC if the timezone is not found
        tz = ZoneInfo("UTC")
    
    now = datetime.now(tz)
    event_dict["timestamp"] = now.isoformat()
    return event_dict


def _filter_by_component(logger, log_method, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Filter log events by component name if a component filter is active."""
    component_filter = config.get("system.logging.component_filter", None)
    
    if not component_filter:
        return event_dict
    
    component = event_dict.get("component", "")
    
    # If the component doesn't match any of the filters, drop the event
    if component_filter and not any(component.startswith(c) for c in component_filter.split(",")):
        raise structlog.DropEvent
    
    return event_dict


def _mask_sensitive_data(
    event_dict: Dict[str, Any], 
    masked_fields: Set[str],
    mask_char: str = "*",
    preserve_length: bool = True
) -> Dict[str, Any]:
    """
    Mask sensitive data in log messages.
    
    Args:
        event_dict: The event dictionary to process
        masked_fields: Set of field names to mask
        mask_char: Character to use for masking
        preserve_length: Whether to preserve the length of masked values
        
    Returns:
        The processed event dictionary with sensitive data masked
    """
    return _process_sensitive_value(event_dict, masked_fields, mask_char, preserve_length)


def _process_sensitive_value(
    value: Any, 
    masked_fields: Set[str],
    mask_char: str = "*",
    preserve_length: bool = True
) -> Any:
    """
    Recursively process values to mask sensitive data.
    
    Args:
        value: The value to process
        masked_fields: Set of field names to mask
        mask_char: Character to use for masking
        preserve_length: Whether to preserve the length of masked values
        
    Returns:
        The processed value with sensitive data masked
    """
    if isinstance(value, dict):
        # Process dictionary
        result = {}
        for key, val in value.items():
            if key.lower() in masked_fields:
                # Mask the value
                if isinstance(val, str):
                    if preserve_length:
                        result[key] = mask_char * len(val)
                    else:
                        result[key] = mask_char * 8
                else:
                    result[key] = f"{mask_char * 8}"
            else:
                # Process nested structures
                result[key] = _process_sensitive_value(val, masked_fields, mask_char, preserve_length)
        return result
    elif isinstance(value, list):
        # Process list
        return [_process_sensitive_value(item, masked_fields, mask_char, preserve_length) for item in value]
    else:
        # Return other types unchanged
        return value


def _add_request_id(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add request ID to the log event if one is set."""
    request_id = _request_id_context.get("request_id")
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def _process_datetime_objects(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert datetime objects to ISO-8601 strings for proper serialization."""
    return _process_value(event_dict)


def _process_value(value: Any) -> Any:
    """Recursively process values to convert datetime objects to ISO strings."""
    if isinstance(value, datetime):
        return format_iso(value)
    elif isinstance(value, dict):
        return {k: _process_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_process_value(item) for item in value]
    return value


def _rate_limit_processor(logger, log_method, event_dict):
    """
    Process log events to apply rate limiting.
    
    This prevents high-volume log events from overwhelming the logs.
    """
    # Get the log key - use event plus all context values except timestamp and level
    log_key = event_dict.get("event", "")
    for key, value in sorted(event_dict.items()):
        if key not in ["timestamp", "level", "logger"]:
            log_key += f":{key}={value}"
    
    # Check if this log message needs to be rate limited
    rate_limit = event_dict.pop("_rate_limit", None)
    if rate_limit is None:
        # No rate limiting requested
        return event_dict
    
    # Check if we've exceeded the rate limit
    now = time.time()
    last_log = _rate_limited_logs.get(log_key, 0)
    if now - last_log < rate_limit:
        # Skip this log message (rate limited)
        return None
    
    # Update the last logged time
    _rate_limited_logs[log_key] = now
    
    # Include rate limit info in the log
    event_dict["rate_limit_seconds"] = rate_limit
    return event_dict


def rate_limited_log(logger, level, message, rate_limit_seconds, **kwargs):
    """
    Log a message with rate limiting.
    
    Args:
        logger: The logger to use
        level: The log level
        message: The log message
        rate_limit_seconds: The minimum time between identical log messages
        **kwargs: Additional log context
    """
    log_method = getattr(logger, level)
    kwargs["_rate_limit"] = rate_limit_seconds
    log_method(message, **kwargs)


@contextmanager
def request_context(request_id: Optional[str] = None):
    """
    Context manager for tracking a request across components.
    
    Args:
        request_id: Optional request ID to use. If not provided, a random UUID will be generated.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    # Store the request ID in thread-local storage
    old_request_id = _request_id_context.get("request_id")
    _request_id_context["request_id"] = request_id
    
    try:
        yield request_id
    finally:
        # Restore the previous request ID or clear it
        if old_request_id:
            _request_id_context["request_id"] = old_request_id
        else:
            _request_id_context.pop("request_id", None)


def get_logger(component: str, subcomponent: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger for a specific component.
    
    Args:
        component: The main component name
        subcomponent: Optional subcomponent name
        
    Returns:
        A bound logger with component context
    """
    logger = structlog.get_logger()
    
    if subcomponent:
        return logger.bind(component=f"{component}.{subcomponent}")
    
    return logger.bind(component=component)


class MetricsLogger:
    """Helper class for logging performance metrics."""
    
    def __init__(self, logger: structlog.BoundLogger, metric_name: str):
        """
        Initialize the metrics logger.
        
        Args:
            logger: The logger to use
            metric_name: The name of the metric to track
        """
        self.logger = logger
        self.metric_name = metric_name
        self._values = []
        self._count = 0
        self._sum = 0
        self._min = float('inf')
        self._max = float('-inf')
        self._last_value = None
        self._last_logged = 0
        
    def add_value(self, value: float) -> None:
        """
        Add a value to the metric.
        
        Args:
            value: The value to add
        """
        self._values.append(value)
        self._count += 1
        self._sum += value
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        self._last_value = value
        
    def reset(self) -> None:
        """Reset all metrics."""
        self._values = []
        self._count = 0
        self._sum = 0
        self._min = float('inf')
        self._max = float('-inf')
        self._last_value = None
    
    @property
    def count(self) -> int:
        """Get the number of values added."""
        return self._count
    
    @property
    def sum(self) -> float:
        """Get the sum of all values."""
        return self._sum
    
    @property
    def min(self) -> float:
        """Get the minimum value."""
        return self._min if self._count > 0 else 0
    
    @property
    def max(self) -> float:
        """Get the maximum value."""
        return self._max if self._count > 0 else 0
    
    @property
    def avg(self) -> float:
        """Get the average value."""
        return self._sum / self._count if self._count > 0 else 0
    
    @property
    def last(self) -> Optional[float]:
        """Get the last added value."""
        return self._last_value
    
    def log_metrics(self, reset: bool = False, level: str = "info") -> None:
        """
        Log the current metrics.
        
        Args:
            reset: Whether to reset the metrics after logging
            level: The log level to use
        """
        if self._count == 0:
            return
            
        log_method = getattr(self.logger, level)
        log_method(
            f"Metrics for {self.metric_name}",
            metric_name=self.metric_name,
            count=self._count,
            avg=self.avg,
            min=self._min,
            max=self._max,
            sum=self._sum,
            last=self._last_value
        )
        
        if reset:
            self.reset()


def get_metrics_logger(metric_name: str, component: str = "metrics") -> MetricsLogger:
    """
    Get a metrics logger for a specific metric.
    
    Args:
        metric_name: The name of the metric to track
        component: The component to associate with the metrics
        
    Returns:
        A MetricsLogger instance
    """
    logger = get_logger(component, metric_name)
    
    # Check if we already have a MetricsLogger for this metric
    if metric_name in _performance_metrics:
        return _performance_metrics[metric_name]
    
    # Create a new MetricsLogger
    metrics_logger = MetricsLogger(logger, metric_name)
    _performance_metrics[metric_name] = metrics_logger
    
    return metrics_logger


@contextmanager
def timed_operation(metric_name: str, component: str = "metrics"):
    """
    Context manager for timing operations and recording metrics.
    
    Args:
        metric_name: The name of the metric to track
        component: The component to associate with the metrics
    """
    metrics_logger = get_metrics_logger(metric_name, component)
    start_time = time.time()
    
    try:
        yield metrics_logger
    finally:
        duration = time.time() - start_time
        metrics_logger.add_value(duration)


class LogOperation:
    """Context manager for logging operations with timing information."""
    
    def __init__(self, logger: structlog.BoundLogger, operation: str, level: str = "info", **kwargs):
        """
        Initialize the context manager.
        
        Args:
            logger: The logger to use
            operation: The name of the operation being performed
            level: The log level to use (default: info)
            **kwargs: Additional context to include in log messages
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.kwargs = kwargs
        self.start_time = None
        
    def __enter__(self):
        """Log the start of the operation and record the start time."""
        self.start_time = utc_now()
        log_method = getattr(self.logger, self.level)
        log_method(f"Starting {self.operation}", **self.kwargs)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log the end of the operation with duration information."""
        end_time = utc_now()
        duration_ms = (end_time - self.start_time).total_seconds() * 1000
        
        log_method = getattr(self.logger, self.level)
        
        if exc_type:
            # If an exception occurred, log it with the error level
            self.logger.error(
                f"Error in {self.operation}", 
                error_type=exc_type.__name__,
                error=str(exc_val),
                duration_ms=duration_ms,
                **self.kwargs
            )
            # Don't suppress the exception
            return False
        
        # Log successful completion
        log_method(
            f"Completed {self.operation}",
            duration_ms=duration_ms,
            **self.kwargs
        )
        return True


class TraceContext:
    """Context manager for distributed tracing using OpenTelemetry."""
    
    def __init__(self, operation_name, attributes=None, parent=None, logger=None):
        """
        Initialize a new trace context.
        
        Args:
            operation_name: Name of the operation being traced
            attributes: Optional attributes to attach to the span
            parent: Optional parent span or span context
            logger: Optional logger to use for logging span events
        """
        self.operation_name = operation_name
        self.attributes = attributes or {}
        self.parent = parent
        self.logger = logger or structlog.get_logger()
        self.span = None
        
    def __enter__(self):
        """Start a new span when entering the context."""
        if not OPENTELEMETRY_AVAILABLE:
            self.logger.debug(f"Starting operation: {self.operation_name} (OpenTelemetry not available)")
            return self
            
        # Get current tracer
        tracer = trace.get_tracer(__name__)
        
        # Start a new span
        self.span = tracer.start_span(
            name=self.operation_name,
            context=self.parent,
            attributes=self.attributes
        )
        
        # Set current span as active
        self.span.__enter__()
        
        self.logger.debug(f"Starting traced operation: {self.operation_name}", 
                        trace_id=self.span.get_span_context().trace_id,
                        span_id=self.span.get_span_context().span_id)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the span when exiting the context."""
        if not OPENTELEMETRY_AVAILABLE or self.span is None:
            if exc_type:
                self.logger.error(f"Error in operation {self.operation_name}: {exc_val}")
            else:
                self.logger.debug(f"Completed operation: {self.operation_name}")
            return False
            
        # Record exception if one occurred
        if exc_type:
            self.span.record_exception(exc_val)
            self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
            self.logger.error(f"Error in traced operation: {self.operation_name}", 
                            error=str(exc_val),
                            trace_id=self.span.get_span_context().trace_id,
                            span_id=self.span.get_span_context().span_id)
        else:
            self.span.set_status(trace.Status(trace.StatusCode.OK))
            self.logger.debug(f"Completed traced operation: {self.operation_name}",
                            trace_id=self.span.get_span_context().trace_id,
                            span_id=self.span.get_span_context().span_id)
        
        # End the span
        self.span.__exit__(exc_type, exc_val, exc_tb)
        
        return False  # propagate exceptions
        
    def add_event(self, name, attributes=None):
        """Add an event to the current span."""
        if OPENTELEMETRY_AVAILABLE and self.span:
            self.span.add_event(name, attributes)
            
    def update_name(self, new_name):
        """Update the span name."""
        if OPENTELEMETRY_AVAILABLE and self.span:
            self.span.update_name(new_name)
            
    def set_attribute(self, key, value):
        """Set a span attribute."""
        if OPENTELEMETRY_AVAILABLE and self.span:
            self.span.set_attribute(key, value)


def configure_tracing():
    """Configure OpenTelemetry distributed tracing."""
    if not OPENTELEMETRY_AVAILABLE:
        return False
        
    if not config.get("system.tracing.enabled", False):
        return False
        
    try:
        # Service name and version for resource attribution
        service_name = config.get("system.tracing.service_name", "ai-trading-agent")
        service_version = config.get("system.tracing.service_version", "0.1.0")
        
        # Create resource with service info
        resource = Resource.create({
            "service.name": service_name,
            "service.version": service_version,
            "environment": ENVIRONMENT,
        })
        
        # Set up trace provider with the resource
        trace_provider = TracerProvider(resource=resource)
        
        # Configure exporters
        if config.get("system.tracing.otlp.enabled", False):
            # OTLP (OpenTelemetry Protocol) exporter for sending to collectors like Jaeger
            otlp_endpoint = config.get("system.tracing.otlp.endpoint", "localhost:4317")
            otlp_headers = config.get("system.tracing.otlp.headers", {})
            
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                headers=otlp_headers
            )
            
            # Add span processor with batch export
            trace_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        
        # Set global trace provider
        trace.set_tracer_provider(trace_provider)
        
        # Set up context propagation
        propagator = TraceContextTextMapPropagator()
        
        # Create a logger for tracing system
        logger = structlog.get_logger("tracing")
        logger.info("Distributed tracing configured", 
                   service_name=service_name, 
                   service_version=service_version)
        
        return True
    except Exception as e:
        logger = structlog.get_logger("tracing")
        logger.error("Failed to configure tracing", error=str(e))
        return False


def extract_trace_context(headers):
    """
    Extract trace context from headers.
    
    Args:
        headers: Dictionary of HTTP headers
        
    Returns:
        Trace context or None if not found/available
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None
        
    try:
        # Create context carrier from headers
        context = {}
        propagator = TraceContextTextMapPropagator()
        
        # Extract context
        return propagator.extract(headers)
    except Exception:
        return None


def inject_trace_context(headers):
    """
    Inject current trace context into headers.
    
    Args:
        headers: Dictionary of HTTP headers to inject context into
    """
    if not OPENTELEMETRY_AVAILABLE:
        return
        
    try:
        # Get propagator
        propagator = TraceContextTextMapPropagator()
        
        # Inject current context into headers
        propagator.inject(headers)
    except Exception:
        pass


# Configure logging when the module is imported
configure_logging()

# Default system logger
system_logger = get_logger("system")