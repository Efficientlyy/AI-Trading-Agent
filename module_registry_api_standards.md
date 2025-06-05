# Module Registry and API Interface Standards

This document defines the standards for module registration, discovery, and API interfaces in the System Overseer's modular architecture.

## 1. Module Registry Architecture

### Core Concepts

The Module Registry serves as the central hub for component discovery, dependency management, and lifecycle control in the System Overseer. It implements the following key patterns:

- **Service Locator Pattern**: Provides a centralized registry for locating services
- **Plugin Architecture**: Supports dynamic loading and unloading of components
- **Dependency Injection**: Manages component dependencies and initialization order
- **Event-Driven Communication**: Facilitates loose coupling between components

### Registry Structure

```
ModuleRegistry
├── ServiceLocator - Core service discovery mechanism
├── PluginManager - Handles plugin lifecycle and dependencies
├── EventBroker - Manages event subscriptions and routing
└── ConfigurationProvider - Supplies configuration to modules
```

### Module Types

The registry supports several types of modules with different lifecycle patterns:

1. **Core Modules**: Essential system components loaded at startup
2. **Service Modules**: Provide specific functionality to other modules
3. **Plugin Modules**: Optional extensions loaded dynamically
4. **Resource Modules**: Provide access to external resources

## 2. Module Interface Standards

### Base Module Interface

All modules must implement the `IModule` interface:

```python
class IModule:
    """Base interface for all modules in the system."""
    
    @property
    def module_id(self) -> str:
        """Get unique module identifier."""
        pass
    
    @property
    def name(self) -> str:
        """Get human-readable module name."""
        pass
    
    @property
    def version(self) -> str:
        """Get module version (semver format)."""
        pass
    
    @property
    def description(self) -> str:
        """Get module description."""
        pass
    
    @property
    def dependencies(self) -> List[str]:
        """Get list of module dependencies (module_ids)."""
        pass
    
    def initialize(self, registry: 'ModuleRegistry') -> bool:
        """Initialize module with registry reference.
        
        Args:
            registry: Module registry instance
            
        Returns:
            bool: True if initialization successful
        """
        pass
    
    def start(self) -> bool:
        """Start module operation.
        
        Returns:
            bool: True if start successful
        """
        pass
    
    def stop(self) -> bool:
        """Stop module operation.
        
        Returns:
            bool: True if stop successful
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status information.
        
        Returns:
            dict: Status information
        """
        pass
```

### Plugin Interface

Plugins extend the base module interface with additional capabilities:

```python
class IPlugin(IModule):
    """Interface for plugin modules."""
    
    @property
    def plugin_type(self) -> str:
        """Get plugin type identifier."""
        pass
    
    @property
    def capabilities(self) -> List[str]:
        """Get list of plugin capabilities."""
        pass
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure plugin with settings.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if configuration successful
        """
        pass
    
    def get_extension_points(self) -> Dict[str, Any]:
        """Get plugin extension points.
        
        Returns:
            dict: Extension point definitions
        """
        pass
```

### Service Interface

Services provide specific functionality to other modules:

```python
class IService(IModule):
    """Interface for service modules."""
    
    @property
    def service_type(self) -> str:
        """Get service type identifier."""
        pass
    
    def get_api(self) -> Any:
        """Get service API object.
        
        Returns:
            object: Service API implementation
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics.
        
        Returns:
            dict: Metrics dictionary
        """
        pass
```

## 3. API Versioning and Compatibility

### Versioning Strategy

All APIs follow Semantic Versioning (SemVer) principles:

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

### Version Declaration

APIs must declare their version in multiple ways:

1. **Module Version**: The overall module version
2. **API Version**: Specific version of each API interface
3. **Method Versioning**: Optional versioning of individual methods

Example:

```python
@api_version("2.0.0")
class MarketDataService(IService):
    """Market data service implementation v2.0.0."""
    
    @property
    def version(self) -> str:
        return "2.0.0"
    
    @method_version("1.0.0")
    def get_candles(self, symbol: str, interval: str) -> List[Dict]:
        """Get candlestick data."""
        pass
    
    @method_version("2.0.0")
    def get_candles_v2(self, symbol: str, interval: str, options: Dict = None) -> List[Dict]:
        """Get candlestick data with extended options."""
        pass
```

### Compatibility Handling

The system handles API compatibility through several mechanisms:

1. **Version Negotiation**: Clients can request specific API versions
2. **Compatibility Layers**: Adapters for backward compatibility
3. **Deprecation Markers**: Clear indication of deprecated features
4. **Feature Detection**: Runtime checking for optional capabilities

## 4. Module Registration Process

### Registration Workflow

1. **Discovery**: Module is discovered through filesystem, entry points, or explicit registration
2. **Validation**: Module interface and metadata are validated
3. **Dependency Resolution**: Module dependencies are checked and resolved
4. **Initialization**: Module is initialized with registry reference
5. **Configuration**: Module is configured with appropriate settings
6. **Activation**: Module is started if auto-start is enabled

### Registration Methods

Modules can be registered through several mechanisms:

1. **Explicit Registration**: Direct API call to register a module
2. **Entry Point Discovery**: Using Python entry points in setup.py
3. **Directory Scanning**: Automatic discovery in plugin directories
4. **Dynamic Loading**: Runtime loading from external sources

Example:

```python
# Explicit registration
registry.register_module(MarketDataService())

# Entry point in setup.py
setup(
    # ...
    entry_points={
        'trading_agent.plugins': [
            'market_data=trading_agent.plugins.market_data:MarketDataService',
        ],
    },
)
```

## 5. Service Discovery and Dependency Injection

### Service Locator

The `ServiceLocator` provides a centralized mechanism for discovering services:

```python
# Get service by type
market_data = service_locator.get_service("market_data")

# Get service by ID
market_data = service_locator.get_service_by_id("mexc_market_data")

# Get service with version constraint
market_data = service_locator.get_service("market_data", min_version="2.0.0")
```

### Dependency Injection

Modules can declare dependencies that are automatically injected:

```python
class SignalGenerator(IModule):
    """Trading signal generator module."""
    
    @property
    def dependencies(self) -> List[str]:
        return ["market_data", "technical_analysis"]
    
    def initialize(self, registry: 'ModuleRegistry') -> bool:
        self.market_data = registry.get_service("market_data")
        self.ta_service = registry.get_service("technical_analysis")
        return True
```

### Lazy Loading

Services can be lazily loaded to improve startup performance:

```python
# Get service reference that will be loaded on first use
market_data = service_locator.get_lazy_service("market_data")
```

## 6. Event-Based Communication

### Event Bus Interface

Modules communicate through a standardized event bus:

```python
class IEventBus:
    """Interface for event bus implementations."""
    
    def publish(self, event_type: str, data: Dict[str, Any], publisher_id: str = None, priority: int = 0) -> str:
        """Publish event to the bus.
        
        Args:
            event_type: Event type identifier
            data: Event data payload
            publisher_id: ID of publishing module
            priority: Event priority (0-10)
            
        Returns:
            str: Event ID
        """
        pass
    
    def subscribe(self, subscriber_id: str, event_type: str, callback: Callable, filter_func: Callable = None) -> str:
        """Subscribe to events.
        
        Args:
            subscriber_id: ID of subscribing module
            event_type: Event type to subscribe to (supports wildcards)
            callback: Callback function for events
            filter_func: Optional filter function
            
        Returns:
            str: Subscription ID
        """
        pass
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.
        
        Args:
            subscription_id: Subscription ID to remove
            
        Returns:
            bool: True if unsubscribed successfully
        """
        pass
```

### Event Structure

Events follow a standardized structure:

```python
{
    "event_id": "evt_1234567890",
    "event_type": "market_data.candle.closed",
    "timestamp": 1622547600.123,
    "publisher_id": "mexc_market_data",
    "priority": 0,
    "data": {
        "symbol": "BTCUSDC",
        "interval": "1m",
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50050.0,
        "volume": 10.5
    }
}
```

### Event Type Hierarchy

Event types follow a hierarchical naming convention:

- `domain.entity.action` (e.g., `market_data.candle.closed`)
- `module.event_name` (e.g., `telegram_bot.command_received`)
- `system.status.change` (e.g., `system.module.started`)

Wildcards are supported for subscriptions:

- `market_data.*` - All market data events
- `*.error` - All error events
- `system.**` - All system events including sub-categories

## 7. Configuration Management

### Configuration Registry Interface

Modules access configuration through a standardized registry:

```python
class IConfigRegistry:
    """Interface for configuration registry implementations."""
    
    def register_parameter(
        self,
        module_id: str,
        param_id: str,
        default_value: Any,
        param_type: type = None,
        description: str = None,
        min_value: Any = None,
        max_value: Any = None,
        enum_values: List[Any] = None,
        is_secret: bool = False,
        group: str = None
    ) -> bool:
        """Register configuration parameter.
        
        Args:
            module_id: Module identifier
            param_id: Parameter identifier
            default_value: Default parameter value
            param_type: Parameter type
            description: Parameter description
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            enum_values: List of allowed values
            is_secret: Whether parameter is sensitive
            group: Parameter group
            
        Returns:
            bool: True if registration successful
        """
        pass
    
    def get_parameter(self, module_id: str, param_id: str, default: Any = None) -> Any:
        """Get parameter value.
        
        Args:
            module_id: Module identifier
            param_id: Parameter identifier
            default: Default value if not found
            
        Returns:
            Any: Parameter value
        """
        pass
    
    def set_parameter(self, module_id: str, param_id: str, value: Any) -> bool:
        """Set parameter value.
        
        Args:
            module_id: Module identifier
            param_id: Parameter identifier
            value: Parameter value
            
        Returns:
            bool: True if set successfully
        """
        pass
    
    def get_module_parameters(self, module_id: str) -> Dict[str, Any]:
        """Get all parameters for a module.
        
        Args:
            module_id: Module identifier
            
        Returns:
            dict: Parameter dictionary
        """
        pass
```

### Parameter Namespacing

Parameters are namespaced to avoid conflicts:

- `module_id.param_id` - Full parameter identifier
- `module_id.*` - All parameters for a module
- `*.group_name` - All parameters in a group

### Configuration Sources

The configuration registry supports multiple sources with priority:

1. **User Settings**: Highest priority, set by users
2. **Environment Variables**: System environment configuration
3. **Configuration Files**: JSON/YAML configuration files
4. **Default Values**: Lowest priority, defined during registration

## 8. Extension Points

### Extension Point Definition

Modules can define extension points for customization:

```python
class ExtensionPoint:
    """Definition of an extension point."""
    
    def __init__(
        self,
        point_id: str,
        interface_class: type,
        description: str,
        multi: bool = False
    ):
        """Initialize extension point.
        
        Args:
            point_id: Extension point identifier
            interface_class: Required interface class
            description: Extension point description
            multi: Whether multiple extensions are allowed
        """
        self.point_id = point_id
        self.interface_class = interface_class
        self.description = description
        self.multi = multi
        self.extensions = []
```

### Extension Registration

Extensions are registered with their target extension points:

```python
# Define extension point
market_data_provider = ExtensionPoint(
    point_id="market_data_provider",
    interface_class=IMarketDataProvider,
    description="Provider of market data",
    multi=True
)

# Register extension
registry.register_extension(
    extension_point_id="market_data_provider",
    extension=MexcMarketDataProvider()
)
```

### Extension Discovery

Extensions can be discovered and used at runtime:

```python
# Get all extensions for a point
providers = registry.get_extensions("market_data_provider")

# Get specific extension
mexc_provider = registry.get_extension("market_data_provider", "mexc")
```

## 9. API Documentation Standards

### Documentation Format

All APIs must be documented using:

1. **Docstrings**: Detailed Python docstrings
2. **Type Hints**: Python type annotations
3. **OpenAPI**: REST API specifications
4. **Interface Definitions**: Clear interface classes

Example:

```python
class IMarketDataProvider:
    """Interface for market data providers.
    
    Market data providers supply price and order book data
    from cryptocurrency exchanges or other sources.
    """
    
    def get_candles(
        self, 
        symbol: str, 
        interval: str, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical candlestick data.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDC")
            interval: Candle interval (e.g., "1m", "1h", "1d")
            limit: Maximum number of candles to return
            
        Returns:
            list: List of candle dictionaries with keys:
                - open_time: Candle open time (timestamp)
                - open: Open price
                - high: High price
                - low: Low price
                - close: Close price
                - volume: Volume
                
        Raises:
            ValueError: If symbol or interval is invalid
            ConnectionError: If provider connection fails
        """
        pass
```

### API Stability Markers

APIs should use stability markers:

- `@stable` - API is stable and backward compatibility is guaranteed
- `@experimental` - API may change without notice
- `@deprecated` - API is deprecated and will be removed
- `@beta` - API is in beta testing phase

## 10. Error Handling and Logging

### Error Propagation

Modules should follow consistent error handling patterns:

1. **Exception Types**: Use specific exception types for different errors
2. **Error Codes**: Include error codes for programmatic handling
3. **Context Information**: Provide detailed context in exceptions
4. **Recovery Hints**: Include recovery suggestions when possible

### Standardized Logging

All modules should use the standardized logging framework:

```python
# Module-specific logger
logger = logging.getLogger("trading_agent.market_data")

# Consistent log format
logger.info("Processing candle data for %s", symbol)
logger.error("Failed to connect to exchange: %s", str(error))
```

### Health Reporting

Modules must implement health reporting:

```python
def get_health(self) -> Dict[str, Any]:
    """Get module health information.
    
    Returns:
        dict: Health information with keys:
            - status: "healthy", "degraded", or "unhealthy"
            - details: Detailed health information
            - last_check: Timestamp of last health check
    """
    return {
        "status": "healthy",
        "details": {
            "connection": "connected",
            "latency_ms": 50,
            "error_rate": 0.01
        },
        "last_check": time.time()
    }
```

## 11. Module Lifecycle Management

### Lifecycle States

Modules follow a defined lifecycle:

1. **Discovered**: Module is found but not loaded
2. **Registered**: Module is registered with the system
3. **Initialized**: Module is initialized with dependencies
4. **Started**: Module is actively running
5. **Stopped**: Module is stopped but still initialized
6. **Unregistered**: Module is removed from the system

### Lifecycle Hooks

Modules can implement hooks for lifecycle events:

```python
def on_before_start(self) -> bool:
    """Called before module is started."""
    pass

def on_after_start(self) -> None:
    """Called after module is successfully started."""
    pass

def on_before_stop(self) -> bool:
    """Called before module is stopped."""
    pass

def on_after_stop(self) -> None:
    """Called after module is successfully stopped."""
    pass
```

### Dependency Management

The registry handles module dependencies:

1. **Dependency Resolution**: Ensures dependencies are available
2. **Initialization Order**: Initializes modules in dependency order
3. **Start Order**: Starts modules in dependency order
4. **Stop Order**: Stops modules in reverse dependency order

## 12. Security and Access Control

### Authentication and Authorization

The module registry implements security controls:

1. **Module Authentication**: Verifies module identity
2. **API Authorization**: Controls access to sensitive APIs
3. **Capability-Based Security**: Restricts operations based on capabilities

### Secure Communication

Inter-module communication follows security best practices:

1. **Input Validation**: All inputs are validated
2. **Sensitive Data Handling**: Secure handling of credentials
3. **Audit Logging**: Security-relevant events are logged

## 13. Implementation Examples

### Module Registration Example

```python
# Define module
class MarketDataService(IService):
    def __init__(self):
        self.module_id = "market_data"
        self.name = "Market Data Service"
        self.version = "1.0.0"
        self.description = "Provides market data from exchanges"
        self.dependencies = ["config_registry", "event_bus"]
        
    def initialize(self, registry):
        self.config = registry.get_service("config_registry")
        self.event_bus = registry.get_service("event_bus")
        
        # Register configuration parameters
        self.config.register_parameter(
            module_id=self.module_id,
            param_id="default_exchange",
            default_value="mexc",
            param_type=str,
            description="Default exchange for market data"
        )
        
        return True
        
    def start(self):
        # Start background data collection
        return True
        
    def stop(self):
        # Stop background data collection
        return True
        
    def get_api(self):
        return self  # Return self as the API implementation
        
    def get_candles(self, symbol, interval, limit=100):
        # Implementation...
        pass

# Register with registry
registry = ModuleRegistry()
registry.register_module(MarketDataService())
```

### Event Communication Example

```python
# Publisher
def publish_new_candle(self, candle_data):
    self.event_bus.publish(
        event_type="market_data.candle.closed",
        data={
            "symbol": candle_data["symbol"],
            "interval": candle_data["interval"],
            "open": candle_data["open"],
            "high": candle_data["high"],
            "low": candle_data["low"],
            "close": candle_data["close"],
            "volume": candle_data["volume"],
            "timestamp": candle_data["timestamp"]
        },
        publisher_id=self.module_id
    )

# Subscriber
def initialize(self, registry):
    self.event_bus = registry.get_service("event_bus")
    
    # Subscribe to candle events
    self.event_bus.subscribe(
        subscriber_id=self.module_id,
        event_type="market_data.candle.closed",
        callback=self.on_candle_closed,
        filter_func=lambda e: e["data"]["symbol"] in self.watched_symbols
    )
    
def on_candle_closed(self, event):
    # Process new candle data
    symbol = event["data"]["symbol"]
    close_price = event["data"]["close"]
    # Implementation...
```

## 14. Module Registry Implementation

```python
class ModuleRegistry:
    """Central registry for system modules."""
    
    def __init__(self):
        """Initialize module registry."""
        self.modules = {}  # module_id -> module
        self.services = {}  # service_type -> [service]
        self.plugins = {}  # plugin_type -> [plugin]
        self.extension_points = {}  # point_id -> ExtensionPoint
        self.extensions = {}  # point_id -> [extension]
        self.dependencies = {}  # module_id -> [dependency_id]
        self.dependents = {}  # module_id -> [dependent_id]
        self.lock = threading.RLock()
        
    def register_module(self, module: IModule) -> bool:
        """Register a module with the registry.
        
        Args:
            module: Module to register
            
        Returns:
            bool: True if registration successful
        """
        with self.lock:
            module_id = module.module_id
            
            # Check if module already registered
            if module_id in self.modules:
                return False
            
            # Store module
            self.modules[module_id] = module
            
            # Register as service if applicable
            if isinstance(module, IService):
                service_type = module.service_type
                if service_type not in self.services:
                    self.services[service_type] = []
                self.services[service_type].append(module)
            
            # Register as plugin if applicable
            if isinstance(module, IPlugin):
                plugin_type = module.plugin_type
                if plugin_type not in self.plugins:
                    self.plugins[plugin_type] = []
                self.plugins[plugin_type].append(module)
            
            # Process dependencies
            for dep_id in module.dependencies:
                if dep_id not in self.dependents:
                    self.dependents[dep_id] = []
                self.dependents[dep_id].append(module_id)
                
                if module_id not in self.dependencies:
                    self.dependencies[module_id] = []
                self.dependencies[module_id].append(dep_id)
            
            return True
    
    def get_module(self, module_id: str) -> Optional[IModule]:
        """Get module by ID.
        
        Args:
            module_id: Module identifier
            
        Returns:
            IModule: Module instance or None if not found
        """
        return self.modules.get(module_id)
    
    def get_service(self, service_type: str, version: str = None) -> Optional[IService]:
        """Get service by type.
        
        Args:
            service_type: Service type identifier
            version: Optional version constraint
            
        Returns:
            IService: Service instance or None if not found
        """
        services = self.services.get(service_type, [])
        
        if not services:
            return None
            
        if version:
            # Filter by version constraint
            compatible_services = [
                s for s in services 
                if self._is_version_compatible(s.version, version)
            ]
            if compatible_services:
                # Return highest version
                return max(compatible_services, key=lambda s: self._parse_version(s.version))
            return None
        
        # Return first service (typically only one per type)
        return services[0]
    
    def initialize_module(self, module_id: str) -> bool:
        """Initialize a module.
        
        Args:
            module_id: Module identifier
            
        Returns:
            bool: True if initialization successful
        """
        module = self.get_module(module_id)
        if not module:
            return False
            
        # Initialize module with registry reference
        return module.initialize(self)
    
    def start_module(self, module_id: str) -> bool:
        """Start a module.
        
        Args:
            module_id: Module identifier
            
        Returns:
            bool: True if start successful
        """
        module = self.get_module(module_id)
        if not module:
            return False
            
        # Start module
        return module.start()
    
    def stop_module(self, module_id: str) -> bool:
        """Stop a module.
        
        Args:
            module_id: Module identifier
            
        Returns:
            bool: True if stop successful
        """
        module = self.get_module(module_id)
        if not module:
            return False
            
        # Stop module
        return module.stop()
    
    def register_extension_point(self, extension_point: ExtensionPoint) -> bool:
        """Register an extension point.
        
        Args:
            extension_point: Extension point to register
            
        Returns:
            bool: True if registration successful
        """
        with self.lock:
            point_id = extension_point.point_id
            
            # Check if extension point already registered
            if point_id in self.extension_points:
                return False
                
            # Store extension point
            self.extension_points[point_id] = extension_point
            self.extensions[point_id] = []
            
            return True
    
    def register_extension(self, extension_point_id: str, extension: Any) -> bool:
        """Register an extension.
        
        Args:
            extension_point_id: Extension point identifier
            extension: Extension implementation
            
        Returns:
            bool: True if registration successful
        """
        with self.lock:
            # Check if extension point exists
            if extension_point_id not in self.extension_points:
                return False
                
            extension_point = self.extension_points[extension_point_id]
            
            # Check if extension implements required interface
            if not isinstance(extension, extension_point.interface_class):
                return False
                
            # Check if multiple extensions are allowed
            if not extension_point.multi and self.extensions[extension_point_id]:
                return False
                
            # Store extension
            self.extensions[extension_point_id].append(extension)
            
            return True
    
    def get_extensions(self, extension_point_id: str) -> List[Any]:
        """Get all extensions for an extension point.
        
        Args:
            extension_point_id: Extension point identifier
            
        Returns:
            list: List of extensions
        """
        return self.extensions.get(extension_point_id, [])
    
    def _is_version_compatible(self, version: str, constraint: str) -> bool:
        """Check if version is compatible with constraint.
        
        Args:
            version: Version string
            constraint: Version constraint
            
        Returns:
            bool: True if compatible
        """
        # Simple implementation - in practice use semver library
        if constraint.startswith(">="):
            min_version = constraint[2:]
            return self._parse_version(version) >= self._parse_version(min_version)
        elif constraint.startswith(">"):
            min_version = constraint[1:]
            return self._parse_version(version) > self._parse_version(min_version)
        elif constraint.startswith("<="):
            max_version = constraint[2:]
            return self._parse_version(version) <= self._parse_version(max_version)
        elif constraint.startswith("<"):
            max_version = constraint[1:]
            return self._parse_version(version) < self._parse_version(max_version)
        elif constraint.startswith("=="):
            exact_version = constraint[2:]
            return version == exact_version
        else:
            # Assume exact match
            return version == constraint
    
    def _parse_version(self, version: str) -> Tuple[int, ...]:
        """Parse version string into comparable tuple.
        
        Args:
            version: Version string
            
        Returns:
            tuple: Version components as integers
        """
        # Simple implementation - in practice use semver library
        return tuple(int(x) for x in version.split("."))
```

## 15. Best Practices

1. **Interface Stability**: Design interfaces for long-term stability
2. **Minimal Dependencies**: Keep module dependencies to a minimum
3. **Graceful Degradation**: Handle missing dependencies gracefully
4. **Defensive Programming**: Validate all inputs and handle errors
5. **Clear Documentation**: Document all interfaces and extension points
6. **Performance Awareness**: Consider performance implications of designs
7. **Testing Support**: Design modules to be easily testable
8. **Versioning Discipline**: Follow semantic versioning strictly
9. **Security First**: Consider security implications in all designs
10. **User Experience**: Design APIs with developer experience in mind
