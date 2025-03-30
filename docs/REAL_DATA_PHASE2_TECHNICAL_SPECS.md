# Real Data Integration - Phase 2 Technical Specifications

## Overview

This document provides detailed technical specifications for the remaining components of Phase 2 in the Real Data Integration project. These specifications include architecture, implementation details, API definitions, UI designs, and testing strategies.

## Table of Contents

1. [Dashboard Settings Panel](#dashboard-settings-panel)
2. [Configuration UI](#configuration-ui)
3. [Status Monitoring Panel](#status-monitoring-panel)
4. [Admin Controls Integration](#admin-controls-integration)
5. [Integration Testing](#integration-testing)
6. [Deployment Strategy](#deployment-strategy)

## Dashboard Settings Panel

### Purpose

Create a dashboard settings panel that allows users to permanently configure data source settings within the UI, eliminating the need for command-line operations.

### Components

1. **Settings Modal**
   - Accessible via the settings icon in the dashboard header
   - Responsive design that works on desktop and tablet devices
   - Tab-based navigation for different setting categories

2. **Data Source Settings Tab**
   - Master toggle for enabling/disabling real data
   - Status indicator showing current configuration
   - Save button that persists changes to the configuration file
   - Reset button that reverts to the last saved configuration

3. **Settings API**
   - Endpoints for retrieving current settings
   - Endpoints for updating settings
   - Validation middleware to ensure valid configurations
   - Authentication checks to restrict access to authorized users

### Technical Details

#### Frontend Implementation

```javascript
// settings_panel.js structure
class SettingsPanel {
    constructor() {
        this.modal = document.getElementById('settings-modal');
        this.tabs = document.querySelectorAll('.settings-tab');
        this.saveButton = document.getElementById('save-settings');
        this.resetButton = document.getElementById('reset-settings');
        this.dataSourceToggle = document.getElementById('real-data-toggle');
        
        this.initEventListeners();
        this.loadCurrentSettings();
    }
    
    initEventListeners() {
        // Set up event handlers for UI interactions
        this.saveButton.addEventListener('click', this.saveSettings.bind(this));
        this.resetButton.addEventListener('click', this.resetSettings.bind(this));
        this.tabs.forEach(tab => {
            tab.addEventListener('click', this.switchTab.bind(this));
        });
    }
    
    loadCurrentSettings() {
        // Fetch current settings from the backend
        fetch('/api/settings/data-source')
            .then(response => response.json())
            .then(data => {
                this.updateUI(data);
            })
            .catch(error => {
                showNotification('Error loading settings', 'error');
                console.error('Error loading settings:', error);
            });
    }
    
    saveSettings() {
        // Collect settings from UI
        const settings = {
            realDataEnabled: this.dataSourceToggle.checked,
            // Other settings...
        };
        
        // Send to backend
        fetch('/api/settings/data-source', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(settings),
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification('Settings saved successfully', 'success');
                    this.closeModal();
                } else {
                    showNotification('Error saving settings: ' + data.message, 'error');
                }
            })
            .catch(error => {
                showNotification('Error saving settings', 'error');
                console.error('Error saving settings:', error);
            });
    }
    
    // Other methods...
}
```

#### Backend Implementation

```python
# settings_manager.py structure
class SettingsManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
            
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
        except IOError as e:
            logger.error(f"Error saving configuration: {e}")
            return False
            
    def get_data_source_settings(self):
        """Get data source settings"""
        return {
            'realDataEnabled': self.config.get('enabled', False),
            'connections': self.config.get('connections', {}),
            # Other settings...
        }
        
    def update_data_source_settings(self, settings):
        """Update data source settings"""
        # Update configuration
        self.config['enabled'] = settings.get('realDataEnabled', False)
        
        # Handle other settings...
        
        # Save to file
        if self._save_config():
            # Trigger any necessary system updates
            self._apply_settings()
            return {'success': True}
        else:
            return {
                'success': False, 
                'message': 'Failed to save configuration'
            }
            
    def _apply_settings(self):
        """Apply settings to the running system"""
        # Notify DataService about configuration changes
        # Update system state based on new settings
        # Maybe restart certain components if needed
```

#### API Endpoints

```python
# In modern_dashboard.py

@app.route('/api/settings/data-source', methods=['GET'])
def get_data_source_settings():
    """API endpoint to get data source settings"""
    # Verify user is authenticated and authorized
    if not current_user.is_authenticated:
        return jsonify({
            "success": False, 
            "message": "Authentication required"
        }), 401
        
    # Get settings from manager
    settings = settings_manager.get_data_source_settings()
    
    return jsonify(settings)
    
@app.route('/api/settings/data-source', methods=['POST'])
def update_data_source_settings():
    """API endpoint to update data source settings"""
    # Verify user is authenticated and authorized
    if not current_user.is_authenticated:
        return jsonify({
            "success": False, 
            "message": "Authentication required"
        }), 401
        
    # Validate request data
    if not request.is_json:
        return jsonify({
            "success": False, 
            "message": "Invalid request format"
        }), 400
        
    settings = request.json
    
    # Update settings
    result = settings_manager.update_data_source_settings(settings)
    
    return jsonify(result)
```

#### HTML Template

```html
<!-- settings_modal.html -->
<div id="settings-modal" class="modal">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Dashboard Settings</h5>
                <button type="button" class="btn-close" id="close-settings"></button>
            </div>
            <div class="modal-body">
                <div class="settings-tabs">
                    <button class="settings-tab active" data-tab="general">General</button>
                    <button class="settings-tab" data-tab="data-source">Data Sources</button>
                    <button class="settings-tab" data-tab="display">Display</button>
                    <button class="settings-tab" data-tab="notifications">Notifications</button>
                </div>
                
                <div class="settings-content">
                    <!-- General Settings Tab -->
                    <div class="settings-pane active" id="general-settings">
                        <!-- General settings content -->
                    </div>
                    
                    <!-- Data Source Settings Tab -->
                    <div class="settings-pane" id="data-source-settings">
                        <div class="setting-group">
                            <div class="setting-item">
                                <div class="setting-label">
                                    <span>Real Data</span>
                                    <span class="setting-description">Enable real data connections</span>
                                </div>
                                <div class="setting-control">
                                    <label class="toggle-switch">
                                        <input type="checkbox" id="real-data-toggle">
                                        <span class="toggle-slider"></span>
                                    </label>
                                </div>
                            </div>
                            
                            <div class="setting-item" id="real-data-options">
                                <div class="setting-label">
                                    <span>Connection Settings</span>
                                    <span class="setting-description">Configure individual data sources</span>
                                </div>
                                <div class="setting-control">
                                    <button id="edit-connections" class="btn btn-sm btn-secondary">
                                        Edit Connections
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Display Settings Tab -->
                    <div class="settings-pane" id="display-settings">
                        <!-- Display settings content -->
                    </div>
                    
                    <!-- Notifications Settings Tab -->
                    <div class="settings-pane" id="notifications-settings">
                        <!-- Notifications settings content -->
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" id="reset-settings">Reset</button>
                <button type="button" class="btn btn-primary" id="save-settings">Save Changes</button>
            </div>
        </div>
    </div>
</div>
```

### Testing Strategy

1. **Unit Tests**
   - Test UI components in isolation
   - Test settings manager with mock file system
   - Test API endpoints with mock requests

2. **Integration Tests**
   - Test end-to-end settings flow
   - Verify settings persist to file
   - Test system state changes with setting changes

## Configuration UI

### Purpose

Create a dedicated UI for editing the `real_data_config.json` file, allowing detailed configuration of real data sources without requiring manual file editing.

### Components

1. **Connection Editor Modal**
   - Launched from the data source settings tab
   - Form-based interface for editing connection properties
   - Validation for configuration values
   - Save and cancel buttons

2. **Connection List**
   - List of all available data source connections
   - Enable/disable toggle for each connection
   - Edit button for each connection
   - Status indicator for each connection

3. **Connection Detail Form**
   - Fields for connection properties (retry attempts, timeout, cache duration)
   - Advanced options section for expert users
   - Validation with immediate feedback
   - Reset to defaults button

### Technical Details

#### Frontend Implementation

```javascript
// data_source_config.js structure
class DataSourceConfig {
    constructor() {
        this.modal = document.getElementById('connection-editor-modal');
        this.connectionList = document.getElementById('connection-list');
        this.saveButton = document.getElementById('save-connections');
        this.cancelButton = document.getElementById('cancel-connections');
        
        this.connections = {};
        
        this.initEventListeners();
        this.loadConnections();
    }
    
    initEventListeners() {
        this.saveButton.addEventListener('click', this.saveConnections.bind(this));
        this.cancelButton.addEventListener('click', this.closeModal.bind(this));
        
        // Add event delegation for connection list
        this.connectionList.addEventListener('click', this.handleConnectionAction.bind(this));
    }
    
    loadConnections() {
        fetch('/api/settings/data-source/connections')
            .then(response => response.json())
            .then(data => {
                this.connections = data.connections;
                this.renderConnectionList();
            })
            .catch(error => {
                showNotification('Error loading connections', 'error');
                console.error('Error loading connections:', error);
            });
    }
    
    renderConnectionList() {
        this.connectionList.innerHTML = '';
        
        Object.entries(this.connections).forEach(([id, connection]) => {
            const item = document.createElement('div');
            item.className = 'connection-item';
            item.dataset.id = id;
            
            item.innerHTML = `
                <div class="connection-info">
                    <div class="connection-name">${this.formatConnectionName(id)}</div>
                    <div class="connection-status ${connection.enabled ? 'enabled' : 'disabled'}">
                        ${connection.enabled ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
                <div class="connection-actions">
                    <label class="toggle-switch">
                        <input type="checkbox" class="connection-toggle" ${connection.enabled ? 'checked' : ''}>
                        <span class="toggle-slider"></span>
                    </label>
                    <button class="btn btn-sm btn-secondary edit-connection">Edit</button>
                </div>
            `;
            
            this.connectionList.appendChild(item);
        });
    }
    
    handleConnectionAction(event) {
        const item = event.target.closest('.connection-item');
        if (!item) return;
        
        const connectionId = item.dataset.id;
        
        if (event.target.classList.contains('connection-toggle')) {
            this.toggleConnection(connectionId, event.target.checked);
        } else if (event.target.classList.contains('edit-connection')) {
            this.editConnection(connectionId);
        }
    }
    
    toggleConnection(id, enabled) {
        this.connections[id].enabled = enabled;
        // Update UI immediately for responsive feel
        const statusEl = document.querySelector(`.connection-item[data-id="${id}"] .connection-status`);
        if (statusEl) {
            statusEl.className = `connection-status ${enabled ? 'enabled' : 'disabled'}`;
            statusEl.textContent = enabled ? 'Enabled' : 'Disabled';
        }
    }
    
    editConnection(id) {
        // Open detail form for this connection
        const connection = this.connections[id];
        this.showConnectionDetailForm(id, connection);
    }
    
    showConnectionDetailForm(id, connection) {
        const detailForm = document.getElementById('connection-detail-form');
        // Populate form with connection details
        document.getElementById('connection-id').value = id;
        document.getElementById('connection-name').value = this.formatConnectionName(id);
        document.getElementById('connection-enabled').checked = connection.enabled;
        document.getElementById('retry-attempts').value = connection.retry_attempts || 3;
        document.getElementById('timeout-seconds').value = connection.timeout_seconds || 10;
        document.getElementById('cache-duration').value = connection.cache_duration_seconds || 60;
        
        // Show the form panel
        document.getElementById('connection-list-panel').style.display = 'none';
        detailForm.style.display = 'block';
    }
    
    saveConnectionDetail() {
        const id = document.getElementById('connection-id').value;
        
        // Get values from form
        const connection = {
            enabled: document.getElementById('connection-enabled').checked,
            retry_attempts: parseInt(document.getElementById('retry-attempts').value),
            timeout_seconds: parseInt(document.getElementById('timeout-seconds').value),
            cache_duration_seconds: parseInt(document.getElementById('cache-duration').value)
        };
        
        // Validate values
        if (connection.retry_attempts < 1 || connection.retry_attempts > 10) {
            showNotification('Retry attempts must be between 1 and 10', 'error');
            return;
        }
        
        if (connection.timeout_seconds < 1 || connection.timeout_seconds > 60) {
            showNotification('Timeout must be between 1 and 60 seconds', 'error');
            return;
        }
        
        if (connection.cache_duration_seconds < 0 || connection.cache_duration_seconds > 3600) {
            showNotification('Cache duration must be between 0 and 3600 seconds', 'error');
            return;
        }
        
        // Update the connection
        this.connections[id] = connection;
        
        // Return to list view
        document.getElementById('connection-detail-form').style.display = 'none';
        document.getElementById('connection-list-panel').style.display = 'block';
        
        // Update the list view
        this.renderConnectionList();
    }
    
    saveConnections() {
        fetch('/api/settings/data-source/connections', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                connections: this.connections
            }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification('Connections saved successfully', 'success');
                    this.closeModal();
                } else {
                    showNotification('Error saving connections: ' + data.message, 'error');
                }
            })
            .catch(error => {
                showNotification('Error saving connections', 'error');
                console.error('Error saving connections:', error);
            });
    }
    
    formatConnectionName(id) {
        // Convert snake_case to Title Case
        return id.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
    
    // Other methods...
}
```

#### Backend Implementation

```python
# config_manager.py structure
class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        
    def get_connections(self):
        """Get all data source connections"""
        return {
            'connections': self.config.get('connections', {})
        }
        
    def update_connections(self, connections_data):
        """Update data source connections"""
        # Update configuration
        self.config['connections'] = connections_data.get('connections', {})
        
        # Save to file
        if self._save_config():
            # Notify system of changes
            self._apply_connection_settings()
            return {'success': True}
        else:
            return {
                'success': False, 
                'message': 'Failed to save configuration'
            }
            
    def _apply_connection_settings(self):
        """Apply connection settings to the running system"""
        # Update DataService with new connection settings
        # May require restarting certain components
```

#### API Endpoints

```python
# In modern_dashboard.py

@app.route('/api/settings/data-source/connections', methods=['GET'])
def get_data_source_connections():
    """API endpoint to get data source connections"""
    if not current_user.is_authenticated:
        return jsonify({
            "success": False, 
            "message": "Authentication required"
        }), 401
        
    connections = config_manager.get_connections()
    
    return jsonify(connections)
    
@app.route('/api/settings/data-source/connections', methods=['POST'])
def update_data_source_connections():
    """API endpoint to update data source connections"""
    if not current_user.is_authenticated:
        return jsonify({
            "success": False, 
            "message": "Authentication required"
        }), 401
        
    if not request.is_json:
        return jsonify({
            "success": False, 
            "message": "Invalid request format"
        }), 400
        
    connections_data = request.json
    
    result = config_manager.update_connections(connections_data)
    
    return jsonify(result)
```

#### HTML Template

```html
<!-- connection_editor_modal.html -->
<div id="connection-editor-modal" class="modal">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Data Source Connections</h5>
                <button type="button" class="btn-close" id="close-connections"></button>
            </div>
            <div class="modal-body">
                <!-- Connection List Panel -->
                <div id="connection-list-panel">
                    <div class="panel-header">
                        <h6>Available Connections</h6>
                        <p class="panel-description">
                            Enable or disable individual data sources and configure their properties.
                        </p>
                    </div>
                    
                    <div id="connection-list" class="connection-list">
                        <!-- Connection items will be inserted here -->
                    </div>
                </div>
                
                <!-- Connection Detail Form -->
                <div id="connection-detail-form" style="display: none;">
                    <div class="panel-header">
                        <button class="btn btn-sm btn-secondary back-to-list">
                            <i class="fa fa-arrow-left"></i> Back to List
                        </button>
                        <h6 id="connection-detail-title">Edit Connection</h6>
                    </div>
                    
                    <div class="form-group">
                        <input type="hidden" id="connection-id">
                        
                        <div class="form-row">
                            <label for="connection-name">Name</label>
                            <input type="text" id="connection-name" class="form-control" disabled>
                        </div>
                        
                        <div class="form-row">
                            <label for="connection-enabled">Enabled</label>
                            <label class="toggle-switch">
                                <input type="checkbox" id="connection-enabled">
                                <span class="toggle-slider"></span>
                            </label>
                        </div>
                        
                        <div class="form-row">
                            <label for="retry-attempts">Retry Attempts</label>
                            <input type="number" id="retry-attempts" class="form-control" min="1" max="10">
                            <span class="form-hint">Number of times to retry failed requests (1-10)</span>
                        </div>
                        
                        <div class="form-row">
                            <label for="timeout-seconds">Timeout (seconds)</label>
                            <input type="number" id="timeout-seconds" class="form-control" min="1" max="60">
                            <span class="form-hint">Maximum time to wait for a response (1-60 seconds)</span>
                        </div>
                        
                        <div class="form-row">
                            <label for="cache-duration">Cache Duration (seconds)</label>
                            <input type="number" id="cache-duration" class="form-control" min="0" max="3600">
                            <span class="form-hint">How long to cache data (0-3600 seconds, 0 = no caching)</span>
                        </div>
                        
                        <div class="form-actions">
                            <button class="btn btn-secondary" id="cancel-detail">Cancel</button>
                            <button class="btn btn-primary" id="save-detail">Save</button>
                        </div>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" id="cancel-connections">Cancel</button>
                <button type="button" class="btn btn-primary" id="save-connections">Save All</button>
            </div>
        </div>
    </div>
</div>
```

### Testing Strategy

1. **Unit Tests**
   - Test form validation
   - Test connection toggling
   - Test configuration saving

2. **Integration Tests**
   - Test configuration persistence
   - Test system behavior after configuration changes
   - Test UI updates when configurations change

## Status Monitoring Panel

### Purpose

Create a dedicated panel for monitoring the health and status of real data connections, providing users with insights into data source reliability and performance.

### Components

1. **Status Dashboard**
   - Overview of all data sources with health indicators
   - Summary statistics for connection performance
   - Quick actions for troubleshooting

2. **Source Detail View**
   - Detailed information for each data source
   - Error history with timestamps
   - Performance metrics
   - Connection test functionality

3. **Health Metrics**
   - Uptime percentage
   - Response time trends
   - Error rate visualization
   - Data freshness indicators

### Technical Details

#### Frontend Implementation

```javascript
// data_source_status.js structure
class DataSourceStatus {
    constructor() {
        this.statusContainer = document.getElementById('data-source-status-container');
        this.refreshButton = document.getElementById('refresh-status');
        this.sourceDetailView = document.getElementById('source-detail-view');
        
        this.statusData = {};
        this.updateInterval = null;
        
        this.initEventListeners();
        this.loadStatusData();
        this.startAutoRefresh();
    }
    
    initEventListeners() {
        this.refreshButton.addEventListener('click', this.loadStatusData.bind(this));
        
        // Event delegation for source details
        this.statusContainer.addEventListener('click', event => {
            const sourceItem = event.target.closest('.source-status-item');
            if (sourceItem) {
                this.showSourceDetail(sourceItem.dataset.source);
            }
        });
    }
    
    loadStatusData() {
        // Show loading indicator
        this.statusContainer.classList.add('loading');
        
        fetch('/api/system/data-source/status')
            .then(response => response.json())
            .then(data => {
                this.statusData = data;
                this.renderStatusOverview();
                
                // Hide loading indicator
                this.statusContainer.classList.remove('loading');
            })
            .catch(error => {
                showNotification('Error loading status data', 'error');
                console.error('Error loading status data:', error);
                
                // Hide loading indicator
                this.statusContainer.classList.remove('loading');
            });
    }
    
    renderStatusOverview() {
        // Clear existing content
        this.statusContainer.innerHTML = '';
        
        // Create summary header
        const summary = document.createElement('div');
        summary.className = 'status-summary';
        
        const healthyCount = Object.values(this.statusData.sources).filter(
            source => source.health === 'HEALTHY'
        ).length;
        
        const totalCount = Object.values(this.statusData.sources).length;
        
        summary.innerHTML = `
            <div class="summary-header">
                <h3>Data Source Health</h3>
                <span class="timestamp">Last updated: ${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-value">${healthyCount}/${totalCount}</span>
                    <span class="stat-label">Healthy Sources</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${this.statusData.system_health || 'Unknown'}</span>
                    <span class="stat-label">System Health</span>
                </div>
            </div>
        `;
        
        this.statusContainer.appendChild(summary);
        
        // Create source items
        const sourceList = document.createElement('div');
        sourceList.className = 'source-list';
        
        Object.entries(this.statusData.sources).forEach(([id, source]) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = `source-status-item ${source.health.toLowerCase()}`;
            sourceItem.dataset.source = id;
            
            sourceItem.innerHTML = `
                <div class="source-info">
                    <div class="source-name">${this.formatSourceName(id)}</div>
                    <div class="source-health ${source.health.toLowerCase()}">
                        ${source.health}
                    </div>
                </div>
                <div class="source-metrics">
                    <div class="metric">
                        <span class="metric-value">${source.error_count || 0}</span>
                        <span class="metric-label">Errors</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">${source.response_time || 'N/A'}</span>
                        <span class="metric-label">Resp. Time</span>
                    </div>
                </div>
                <div class="source-actions">
                    <button class="btn btn-sm btn-secondary view-details">
                        Details
                    </button>
                    <button class="btn btn-sm btn-primary test-connection">
                        Test
                    </button>
                </div>
            `;
            
            sourceList.appendChild(sourceItem);
        });
        
        this.statusContainer.appendChild(sourceList);
    }
    
    showSourceDetail(sourceId) {
        const source = this.statusData.sources[sourceId];
        if (!source) return;
        
        // Update source detail view
        this.sourceDetailView.innerHTML = `
            <div class="detail-header">
                <h4>${this.formatSourceName(sourceId)}</h4>
                <span class="health-badge ${source.health.toLowerCase()}">
                    ${source.health}
                </span>
                <button class="btn btn-sm btn-secondary close-detail">
                    <i class="fa fa-times"></i>
                </button>
            </div>
            
            <div class="detail-metrics">
                <div class="metric-card">
                    <div class="metric-title">Uptime</div>
                    <div class="metric-value">${source.uptime || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Response Time</div>
                    <div class="metric-value">${source.response_time || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Error Rate</div>
                    <div class="metric-value">${source.error_rate || '0%'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Last Success</div>
                    <div class="metric-value">${source.last_success || 'Never'}</div>
                </div>
            </div>
            
            <div class="detail-section">
                <h5>Error History</h5>
                <div class="error-history">
                    ${this.renderErrorHistory(source.error_history)}
                </div>
            </div>
            
            <div class="detail-section">
                <h5>Performance Trend</h5>
                <div class="performance-chart" id="performance-chart-${sourceId}">
                    <!-- Chart will be rendered here -->
                </div>
            </div>
            
            <div class="detail-actions">
                <button class="btn btn-primary test-connection-detail" data-source="${sourceId}">
                    Test Connection
                </button>
                <button class="btn btn-secondary reset-source" data-source="${sourceId}">
                    Reset Statistics
                </button>
            </div>
        `;
        
        // Show the detail view
        this.sourceDetailView.style.display = 'block';
        
        // Render chart
        this.renderPerformanceChart(sourceId, source.performance_history);
        
        // Add event listeners
        document.querySelector('.close-detail').addEventListener('click', () => {
            this.sourceDetailView.style.display = 'none';
        });
        
        document.querySelector('.test-connection-detail').addEventListener('click', () => {
            this.testConnection(sourceId);
        });
        
        document.querySelector('.reset-source').addEventListener('click', () => {
            this.resetSourceStats(sourceId);
        });
    }
    
    renderErrorHistory(errorHistory) {
        if (!errorHistory || errorHistory.length === 0) {
            return '<div class="no-data">No errors recorded</div>';
        }
        
        return errorHistory.map(error => `
            <div class="error-item">
                <div class="error-time">${error.timestamp}</div>
                <div class="error-message">${error.message}</div>
            </div>
        `).join('');
    }
    
    renderPerformanceChart(sourceId, performanceHistory) {
        if (!performanceHistory || performanceHistory.length === 0) {
            document.getElementById(`performance-chart-${sourceId}`).innerHTML =
                '<div class="no-data">No performance data available</div>';
            return;
        }
        
        // Render chart using Plotly or Chart.js
        // This is a placeholder for the actual chart rendering code
    }
    
    testConnection(sourceId) {
        // Show loading indicator
        const button = document.querySelector(`.test-connection-detail[data-source="${sourceId}"]`);
        button.disabled = true;
        button.innerHTML = 'Testing...';
        
        fetch(`/api/system/data-source/test?source=${sourceId}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification(`Connection test successful: ${data.message}`, 'success');
                } else {
                    showNotification(`Connection test failed: ${data.message}`, 'error');
                }
                
                // Refresh status data
                this.loadStatusData();
                
                // Reset button
                button.disabled = false;
                button.innerHTML = 'Test Connection';
            })
            .catch(error => {
                showNotification('Error testing connection', 'error');
                console.error('Error testing connection:', error);
                
                // Reset button
                button.disabled = false;
                button.innerHTML = 'Test Connection';
            });
    }
    
    resetSourceStats(sourceId) {
        fetch(`/api/system/data-source/reset?source=${sourceId}`, {
            method: 'POST'
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification('Source statistics reset successfully', 'success');
                    
                    // Refresh status data
                    this.loadStatusData();
                    
                    // Hide detail view
                    this.sourceDetailView.style.display = 'none';
                } else {
                    showNotification(`Failed to reset source statistics: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showNotification('Error resetting source statistics', 'error');
                console.error('Error resetting source statistics:', error);
            });
    }
    
    startAutoRefresh() {
        this.updateInterval = setInterval(() => {
            this.loadStatusData();
        }, 30000); // Refresh every 30 seconds
    }
    
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    formatSourceName(id) {
        // Convert snake_case to Title Case
        return id.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
    
    // Other methods...
}
```

#### Backend Implementation

```python
# status_reporter.py structure
class StatusReporter:
    def __init__(self, data_service):
        self.data_service = data_service
        self.status_history = {}
        self.performance_history = {}
        self.error_history = {}
        
    def get_system_status(self):
        """Get comprehensive system status"""
        sources = {}
        
        # Get status for each data source
        for source_id, source in self.data_service.data_sources.items():
            sources[source_id] = {
                'health': source.health_status,
                'error_count': source.error_count,
                'response_time': self._format_response_time(source.avg_response_time),
                'uptime': self._calculate_uptime(source_id),
                'error_rate': self._calculate_error_rate(source_id),
                'last_success': self._format_timestamp(source.last_success_time),
                'error_history': self._get_error_history(source_id),
                'performance_history': self._get_performance_history(source_id)
            }
            
        # Calculate overall system health
        healthy_sources = sum(1 for source in sources.values() if source['health'] == 'HEALTHY')
        total_sources = len(sources)
        
        if healthy_sources == total_sources:
            system_health = 'HEALTHY'
        elif healthy_sources >= total_sources * 0.7:
            system_health = 'DEGRADED'
        else:
            system_health = 'UNHEALTHY'
            
        return {
            'sources': sources,
            'system_health': system_health,
            'timestamp': datetime.now().isoformat()
        }
        
    def test_source_connection(self, source_id):
        """Test connection to a specific data source"""
        if source_id not in self.data_service.data_sources:
            return {
                'success': False,
                'message': f'Unknown data source: {source_id}'
            }
            
        source = self.data_service.data_sources[source_id]
        
        try:
            # Attempt to fetch minimal data from this source
            start_time = time.time()
            test_result = source.test_connection()
            response_time = time.time() - start_time
            
            if test_result['success']:
                return {
                    'success': True,
                    'message': f'Connection successful (response time: {response_time:.2f}s)',
                    'response_time': response_time
                }
            else:
                return {
                    'success': False,
                    'message': f'Connection failed: {test_result["message"]}'
                }
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection error: {str(e)}'
            }
            
    def reset_source_stats(self, source_id):
        """Reset statistics for a specific data source"""
        if source_id not in self.data_service.data_sources:
            return {
                'success': False,
                'message': f'Unknown data source: {source_id}'
            }
            
        source = self.data_service.data_sources[source_id]
        
        try:
            # Reset source statistics
            source.reset_stats()
            
            # Clear history
            if source_id in self.status_history:
                self.status_history[source_id] = []
            if source_id in self.performance_history:
                self.performance_history[source_id] = []
            if source_id in self.error_history:
                self.error_history[source_id] = []
                
            return {
                'success': True,
                'message': f'Statistics for {source_id} have been reset'
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to reset statistics: {str(e)}'
            }
            
    def _format_response_time(self, response_time):
        """Format response time for display"""
        if response_time is None:
            return 'N/A'
        return f'{response_time:.2f}s'
        
    def _calculate_uptime(self, source_id):
        """Calculate uptime percentage for a data source"""
        # Implementation depends on how uptime is tracked
        # This is a placeholder
        return '99.5%'
        
    def _calculate_error_rate(self, source_id):
        """Calculate error rate for a data source"""
        # Implementation depends on how errors are tracked
        # This is a placeholder
        return '0.5%'
        
    def _format_timestamp(self, timestamp):
        """Format timestamp for display"""
        if timestamp is None:
            return 'Never'
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
    def _get_error_history(self, source_id):
        """Get error history for a data source"""
        # Implementation depends on how error history is tracked
        # This is a placeholder
        return []
        
    def _get_performance_history(self, source_id):
        """Get performance history for a data source"""
        # Implementation depends on how performance history is tracked
        # This is a placeholder
        return []
```

#### API Endpoints

```python
# In modern_dashboard.py

@app.route('/api/system/data-source/status', methods=['GET'])
def get_data_source_status():
    """API endpoint to get data source status"""
    if not current_user.is_authenticated:
        return jsonify({
            "success": False, 
            "message": "Authentication required"
        }), 401
        
    status = status_reporter.get_system_status()
    
    return jsonify(status)
    
@app.route('/api/system/data-source/test', methods=['GET'])
def test_data_source_connection():
    """API endpoint to test a data source connection"""
    if not current_user.is_authenticated:
        return jsonify({
            "success": False, 
            "message": "Authentication required"
        }), 401
        
    source_id = request.args.get('source')
    if not source_id:
        return jsonify({
            "success": False, 
            "message": "Source ID is required"
        }), 400
        
    result = status_reporter.test_source_connection(source_id)
    
    return jsonify(result)
    
@app.route('/api/system/data-source/reset', methods=['POST'])
def reset_data_source_stats():
    """API endpoint to reset data source statistics"""
    if not current_user.is_authenticated:
        return jsonify({
            "success": False, 
            "message": "Authentication required"
        }), 401
        
    source_id = request.args.get('source')
    if not source_id:
        return jsonify({
            "success": False, 
            "message": "Source ID is required"
        }), 400
        
    result = status_reporter.reset_source_stats(source_id)
    
    return jsonify(result)
```

#### HTML Template

```html
<!-- data_source_status_panel.html -->
<div class="status-panel-container">
    <div class="status-panel-header">
        <h2>Data Source Status</h2>
        <div class="status-actions">
            <button id="refresh-status" class="btn btn-sm btn-secondary">
                <i class="fa fa-refresh"></i> Refresh
            </button>
        </div>
    </div>
    
    <div id="data-source-status-container" class="status-content">
        <!-- Status content will be inserted here -->
    </div>
    
    <div id="source-detail-view" class="source-detail-panel" style="display: none;">
        <!-- Source detail content will be inserted here -->
    </div>
</div>
```

### Testing Strategy

1. **Unit Tests**
   - Test status data formatting
   - Test error history tracking
   - Test performance metrics calculation

2. **Integration Tests**
   - Test status reporting with simulated data sources
   - Test connection testing functionality
   - Test statistics reset functionality

## Admin Controls Integration

### Purpose

Integrate command-line functionality into the dashboard admin interface, allowing administrators to perform system operations without using the command line.

### Components

1. **Admin Dashboard**
   - Overview of system status
   - Quick actions for common tasks
   - Access to advanced configuration options

2. **System Operations**
   - Start/stop system components
   - Run diagnostic tests
   - View logs and error reports
   - Manage configurations

3. **User Management**
   - Add/remove users
   - Assign roles and permissions
   - View user activity logs

### Technical Details

#### Frontend Implementation

```javascript
// admin_controls.js structure
class AdminControls {
    constructor() {
        this.adminPanel = document.getElementById('admin-panel');
        this.systemTab = document.getElementById('system-tab');
        this.usersTab = document.getElementById('users-tab');
        this.logsTab = document.getElementById('logs-tab');
        this.configTab = document.getElementById('config-tab');
        
        this.currentTab = 'system';
        
        this.initEventListeners();
        this.loadSystemStatus();
    }
    
    initEventListeners() {
        // Tab switching
        document.querySelectorAll('.admin-tab').forEach(tab => {
            tab.addEventListener('click', this.switchTab.bind(this));
        });
        
        // System operations
        document.getElementById('run-diagnostics').addEventListener('click', this.runDiagnostics.bind(this));
        document.getElementById('enable-real-data').addEventListener('click', this.enableRealData.bind(this));
        document.getElementById('disable-real-data').addEventListener('click', this.disableRealData.bind(this));
        document.getElementById('restart-services').addEventListener('click', this.restartServices.bind(this));
        
        // Log filtering
        document.getElementById('log-filter-form').addEventListener('submit', event => {
            event.preventDefault();
            this.filterLogs();
        });
        
        // Config management
        document.getElementById('export-config').addEventListener('click', this.exportConfig.bind(this));
        document.getElementById('import-config').addEventListener('click', this.importConfig.bind(this));
    }
    
    switchTab(event) {
        const tabId = event.target.dataset.tab;
        if (!tabId) return;
        
        // Update active tab
        document.querySelectorAll('.admin-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        event.target.classList.add('active');
        
        // Show selected tab content
        document.querySelectorAll('.admin-tab-content').forEach(content => {
            content.style.display = 'none';
        });
        document.getElementById(`${tabId}-tab-content`).style.display = 'block';
        
        this.currentTab = tabId;
        
        // Load tab-specific data
        if (tabId === 'system') {
            this.loadSystemStatus();
        } else if (tabId === 'users') {
            this.loadUsers();
        } else if (tabId === 'logs') {
            this.loadLogs();
        } else if (tabId === 'config') {
            this.loadConfigurations();
        }
    }
    
    loadSystemStatus() {
        fetch('/api/admin/system/status')
            .then(response => response.json())
            .then(data => {
                this.updateSystemStatus(data);
            })
            .catch(error => {
                showNotification('Error loading system status', 'error');
                console.error('Error loading system status:', error);
            });
    }
    
    updateSystemStatus(data) {
        const statusContainer = document.getElementById('system-status');
        
        statusContainer.innerHTML = `
            <div class="status-card">
                <div class="status-title">System Status</div>
                <div class="status-value ${data.system_status.toLowerCase()}">${data.system_status}</div>
                <div class="status-details">
                    <div class="detail-item">
                        <span class="detail-label">Uptime:</span>
                        <span class="detail-value">${data.uptime}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">CPU Usage:</span>
                        <span class="detail-value">${data.cpu_usage}%</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Memory Usage:</span>
                        <span class="detail-value">${data.memory_usage}%</span>
                    </div>
                </div>
            </div>
            
            <div class="status-card">
                <div class="status-title">Data Sources</div>
                <div class="status-value ${data.data_sources_status.toLowerCase()}">${data.data_sources_status}</div>
                <div class="status-details">
                    <div class="detail-item">
                        <span class="detail-label">Healthy:</span>
                        <span class="detail-value">${data.healthy_sources}/${data.total_sources}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Real Data:</span>
                        <span class="detail-value">${data.real_data_enabled ? 'Enabled' : 'Disabled'}</span>
                    </div>
                </div>
            </div>
            
            <div class="status-card">
                <div class="status-title">Services</div>
                <div class="status-value ${data.services_status.toLowerCase()}">${data.services_status}</div>
                <div class="status-details">
                    <div class="detail-item">
                        <span class="detail-label">Running:</span>
                        <span class="detail-value">${data.running_services}/${data.total_services}</span>
                    </div>
                </div>
            </div>
        `;
        
        // Update quick actions
        document.getElementById('enable-real-data').disabled = data.real_data_enabled;
        document.getElementById('disable-real-data').disabled = !data.real_data_enabled;
    }
    
    runDiagnostics() {
        // Show loading indicator
        const button = document.getElementById('run-diagnostics');
        button.disabled = true;
        button.innerHTML = 'Running Diagnostics...';
        
        fetch('/api/admin/system/diagnostics', {
            method: 'POST'
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification('Diagnostics completed', 'success');
                    
                    // Show diagnostic results
                    document.getElementById('diagnostic-results').innerHTML = `
                        <h4>Diagnostic Results</h4>
                        <div class="diagnostic-summary">
                            <div class="diagnostic-status ${data.status.toLowerCase()}">
                                ${data.status}
                            </div>
                            <div class="diagnostic-timestamp">
                                ${data.timestamp}
                            </div>
                        </div>
                        <div class="diagnostic-details">
                            ${this.renderDiagnosticChecks(data.checks)}
                        </div>
                    `;
                    
                    // Refresh system status
                    this.loadSystemStatus();
                } else {
                    showNotification(`Diagnostics failed: ${data.message}`, 'error');
                }
                
                // Reset button
                button.disabled = false;
                button.innerHTML = 'Run Diagnostics';
            })
            .catch(error => {
                showNotification('Error running diagnostics', 'error');
                console.error('Error running diagnostics:', error);
                
                // Reset button
                button.disabled = false;
                button.innerHTML = 'Run Diagnostics';
            });
    }
    
    renderDiagnosticChecks(checks) {
        return checks.map(check => `
            <div class="diagnostic-check ${check.status.toLowerCase()}">
                <div class="check-name">${check.name}</div>
                <div class="check-status">${check.status}</div>
                <div class="check-message">${check.message}</div>
            </div>
        `).join('');
    }
    
    enableRealData() {
        this.updateRealDataStatus(true);
    }
    
    disableRealData() {
        this.updateRealDataStatus(false);
    }
    
    updateRealDataStatus(enable) {
        // Show loading indicator
        const button = enable ? 
            document.getElementById('enable-real-data') :
            document.getElementById('disable-real-data');
        
        button.disabled = true;
        
        fetch('/api/admin/system/real-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                enabled: enable
            }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification(`Real data ${enable ? 'enabled' : 'disabled'} successfully`, 'success');
                    
                    // Refresh system status
                    this.loadSystemStatus();
                } else {
                    showNotification(`Failed to ${enable ? 'enable' : 'disable'} real data: ${data.message}`, 'error');
                }
                
                // Reset button
                button.disabled = false;
            })
            .catch(error => {
                showNotification(`Error ${enable ? 'enabling' : 'disabling'} real data`, 'error');
                console.error(`Error ${enable ? 'enabling' : 'disabling'} real data:`, error);
                
                // Reset button
                button.disabled = false;
            });
    }
    
    restartServices() {
        // Show confirmation dialog
        if (!confirm('Are you sure you want to restart all services? This may disrupt ongoing operations.')) {
            return;
        }
        
        // Show loading indicator
        const button = document.getElementById('restart-services');
        button.disabled = true;
        button.innerHTML = 'Restarting...';
        
        fetch('/api/admin/system/restart', {
            method: 'POST'
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification('Services restarted successfully', 'success');
                    
                    // Refresh system status after a delay to allow services to start
                    setTimeout(() => {
                        this.loadSystemStatus();
                    }, 5000);
                } else {
                    showNotification(`Failed to restart services: ${data.message}`, 'error');
                }
                
                // Reset button
                button.disabled = false;
                button.innerHTML = 'Restart Services';
            })
            .catch(error => {
                showNotification('Error restarting services', 'error');
                console.error('Error restarting services:', error);
                
                // Reset button
                button.disabled = false;
                button.innerHTML = 'Restart Services';
            });
    }
    
    loadLogs() {
        // Default to last 100 lines
        fetch('/api/admin/logs?lines=100')
            .then(response => response.json())
            .then(data => {
                this.updateLogView(data);
            })
            .catch(error => {
                showNotification('Error loading logs', 'error');
                console.error('Error loading logs:', error);
            });
    }
    
    filterLogs() {
        const level = document.getElementById('log-level').value;
        const component = document.getElementById('log-component').value;
        const lines = document.getElementById('log-lines').value;
        const query = document.getElementById('log-query').value;
        
        const params = new URLSearchParams();
        if (level) params.append('level', level);
        if (component) params.append('component', component);
        if (lines) params.append('lines', lines);
        if (query) params.append('query', query);
        
        fetch(`/api/admin/logs?${params.toString()}`)
            .then(response => response.json())
            .then(data => {
                this.updateLogView(data);
            })
            .catch(error => {
                showNotification('Error filtering logs', 'error');
                console.error('Error filtering logs:', error);
            });
    }
    
    updateLogView(data) {
        const logContainer = document.getElementById('log-content');
        
        if (data.logs.length === 0) {
            logContainer.innerHTML = '<div class="no-logs">No logs found matching the criteria</div>';
            return;
        }
        
        logContainer.innerHTML = '';
        
        data.logs.forEach(log => {
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry ${log.level.toLowerCase()}`;
            
            logEntry.innerHTML = `
                <div class="log-timestamp">${log.timestamp}</div>
                <div class="log-level ${log.level.toLowerCase()}">${log.level}</div>
                <div class="log-component">${log.component}</div>
                <div class="log-message">${this.escapeHtml(log.message)}</div>
            `;
            
            logContainer.appendChild(logEntry);
        });
    }
    
    loadConfigurations() {
        fetch('/api/admin/config')
            .then(response => response.json())
            .then(data => {
                this.updateConfigView(data);
            })
            .catch(error => {
                showNotification('Error loading configurations', 'error');
                console.error('Error loading configurations:', error);
            });
    }
    
    updateConfigView(data) {
        const configListContainer = document.getElementById('config-list');
        
        configListContainer.innerHTML = '';
        
        data.configs.forEach(config => {
            const configItem = document.createElement('div');
            configItem.className = 'config-item';
            configItem.dataset.path = config.path;
            
            configItem.innerHTML = `
                <div class="config-name">${config.name}</div>
                <div class="config-path">${config.path}</div>
                <div class="config-actions">
                    <button class="btn btn-sm btn-secondary edit-config">Edit</button>
                    <button class="btn btn-sm btn-secondary view-config">View</button>
                </div>
            `;
            
            configListContainer.appendChild(configItem);
        });
        
        // Add event listeners
        document.querySelectorAll('.edit-config').forEach(button => {
            button.addEventListener('click', event => {
                const configPath = event.target.closest('.config-item').dataset.path;
                this.editConfig(configPath);
            });
        });
        
        document.querySelectorAll('.view-config').forEach(button => {
            button.addEventListener('click', event => {
                const configPath = event.target.closest('.config-item').dataset.path;
                this.viewConfig(configPath);
            });
        });
    }
    
    editConfig(configPath) {
        fetch(`/api/admin/config?path=${encodeURIComponent(configPath)}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.showConfigEditor(configPath, data.content);
                } else {
                    showNotification(`Failed to load configuration: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showNotification('Error loading configuration', 'error');
                console.error('Error loading configuration:', error);
            });
    }
    
    viewConfig(configPath) {
        fetch(`/api/admin/config?path=${encodeURIComponent(configPath)}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.showConfigViewer(configPath, data.content);
                } else {
                    showNotification(`Failed to load configuration: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showNotification('Error loading configuration', 'error');
                console.error('Error loading configuration:', error);
            });
    }
    
    showConfigEditor(configPath, content) {
        const editorModal = document.getElementById('config-editor-modal');
        const editor = document.getElementById('config-editor');
        const saveButton = document.getElementById('save-config');
        
        // Set modal title
        document.getElementById('editor-title').textContent = `Edit: ${configPath}`;
        
        // Set editor content
        editor.value = content;
        
        // Show modal
        editorModal.style.display = 'block';
        
        // Setup save handler
        saveButton.onclick = () => {
            this.saveConfig(configPath, editor.value);
        };
    }
    
    showConfigViewer(configPath, content) {
        const viewerModal = document.getElementById('config-viewer-modal');
        const viewer = document.getElementById('config-viewer');
        
        // Set modal title
        document.getElementById('viewer-title').textContent = `View: ${configPath}`;
        
        // Set viewer content
        viewer.textContent = content;
        
        // Show modal
        viewerModal.style.display = 'block';
    }
    
    saveConfig(configPath, content) {
        fetch('/api/admin/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                path: configPath,
                content: content
            }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification('Configuration saved successfully', 'success');
                    
                    // Hide modal
                    document.getElementById('config-editor-modal').style.display = 'none';
                } else {
                    showNotification(`Failed to save configuration: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showNotification('Error saving configuration', 'error');
                console.error('Error saving configuration:', error);
            });
    }
    
    exportConfig() {
        window.location.href = '/api/admin/config/export';
    }
    
    importConfig() {
        document.getElementById('config-import-file').click();
    }
    
    handleConfigImport(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('config_file', file);
        
        fetch('/api/admin/config/import', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification('Configuration imported successfully', 'success');
                    
                    // Refresh configs
                    this.loadConfigurations();
                } else {
                    showNotification(`Failed to import configuration: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showNotification('Error importing configuration', 'error');
                console.error('Error importing configuration:', error);
            });
            
        // Clear the file input
        event.target.value = '';
    }
    
    loadUsers() {
        fetch('/api/admin/users')
            .then(response => response.json())
            .then(data => {
                this.updateUserList(data);
            })
            .catch(error => {
                showNotification('Error loading users', 'error');
                console.error('Error loading users:', error);
            });
    }
    
    updateUserList(data) {
        const userListContainer = document.getElementById('user-list');
        
        userListContainer.innerHTML = '';
        
        data.users.forEach(user => {
            const userItem = document.createElement('div');
            userItem.className = 'user-item';
            userItem.dataset.id = user.id;
            
            userItem.innerHTML = `
                <div class="user-avatar">
                    <img src="${user.avatar || '/static/img/default-avatar.png'}" alt="${user.username}">
                </div>
                <div class="user-info">
                    <div class="user-name">${user.username}</div>
                    <div class="user-role ${user.role.toLowerCase()}">${user.role}</div>
                    <div class="user-status ${user.active ? 'active' : 'inactive'}">
                        ${user.active ? 'Active' : 'Inactive'}
                    </div>
                </div>
                <div class="user-actions">
                    <button class="btn btn-sm btn-secondary edit-user">Edit</button>
                    <button class="btn btn-sm btn-danger ${user.active ? 'deactivate-user' : 'activate-user'}">
                        ${user.active ? 'Deactivate' : 'Activate'}
                    </button>
                </div>
            `;
            
            userListContainer.appendChild(userItem);
        });
        
        // Add event listeners
        document.querySelectorAll('.edit-user').forEach(button => {
            button.addEventListener('click', event => {
                const userId = event.target.closest('.user-item').dataset.id;
                this.editUser(userId);
            });
        });
        
        document.querySelectorAll('.deactivate-user').forEach(button => {
            button.addEventListener('click', event => {
                const userId = event.target.closest('.user-item').dataset.id;
                this.deactivateUser(userId);
            });
        });
        
        document.querySelectorAll('.activate-user').forEach(button => {
            button.addEventListener('click', event => {
                const userId = event.target.closest('.user-item').dataset.id;
                this.activateUser(userId);
            });
        });
    }
    
    // Other methods...
    
    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
}
```

#### Backend Implementation

```python
# admin_endpoints.py structure
class AdminEndpoints:
    def __init__(self, app, data_service, settings_manager, config_manager):
        self.app = app
        self.data_service = data_service
        self.settings_manager = settings_manager
        self.config_manager = config_manager
        
        self.register_endpoints()
        
    def register_endpoints(self):
        """Register all admin endpoints"""
        self.app.route('/api/admin/system/status', methods=['GET'])(self.get_system_status)
        self.app.route('/api/admin/system/diagnostics', methods=['POST'])(self.run_diagnostics)
        self.app.route('/api/admin/system/real-data', methods=['POST'])(self.update_real_data)
        self.app.route('/api/admin/system/restart', methods=['POST'])(self.restart_services)
        self.app.route('/api/admin/logs', methods=['GET'])(self.get_logs)
        self.app.route('/api/admin/config', methods=['GET', 'POST'])(self.handle_config)
        self.app.route('/api/admin/config/export', methods=['GET'])(self.export_config)
        self.app.route('/api/admin/config/import', methods=['POST'])(self.import_config)
        self.app.route('/api/admin/users', methods=['GET'])(self.get_users)
        
    def get_system_status(self):
        """Get system status information"""
        # Check authentication
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({
                "success": False, 
                "message": "Admin access required"
            }), 403
            
        # Get system status
        status = {
            "system_status": "HEALTHY",  # Example value
            "uptime": "5d 12h 34m",
            "cpu_usage": 45.2,
            "memory_usage": 68.7,
            "data_sources_status": "HEALTHY",
            "healthy_sources": 5,
            "total_sources": 6,
            "real_data_enabled": self.settings_manager.is_real_data_enabled(),
            "services_status": "HEALTHY",
            "running_services": 12,
            "total_services": 12
        }
        
        return jsonify(status)
        
    def run_diagnostics(self):
        """Run system diagnostics"""
        # Check authentication
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({
                "success": False, 
                "message": "Admin access required"
            }), 403
            
        try:
            # Run diagnostics
            # In a real implementation, this would run actual diagnostic checks
            
            # Example response
            diagnostic_results = {
                "success": True,
                "status": "PASSED",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "checks": [
                    {
                        "name": "Data Service Connectivity",
                        "status": "PASSED",
                        "message": "All data sources are accessible"
                    },
                    {
                        "name": "Database Health",
                        "status": "PASSED",
                        "message": "Database connection is healthy"
                    },
                    {
                        "name": "File System Access",
                        "status": "PASSED",
                        "message": "File system is accessible and has sufficient space"
                    },
                    {
                        "name": "Memory Usage",
                        "status": "WARNING",
                        "message": "Memory usage is above 75%"
                    }
                ]
            }
            
            return jsonify(diagnostic_results)
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Failed to run diagnostics: {str(e)}"
            })
            
    def update_real_data(self):
        """Enable or disable real data"""
        # Check authentication
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({
                "success": False, 
                "message": "Admin access required"
            }), 403
            
        if not request.is_json:
            return jsonify({
                "success": False, 
                "message": "Invalid request format"
            }), 400
            
        data = request.json
        enabled = data.get('enabled', False)
        
        try:
            # Update real data settings
            if enabled:
                result = self.settings_manager.enable_real_data()
            else:
                result = self.settings_manager.disable_real_data()
                
            return jsonify({
                "success": result,
                "message": f"Real data {'enabled' if enabled else 'disabled'} successfully"
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Failed to {'enable' if enabled else 'disable'} real data: {str(e)}"
            })
            
    def restart_services(self):
        """Restart system services"""
        # Check authentication
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({
                "success": False, 
                "message": "Admin access required"
            }), 403
            
        try:
            # Restart services
            # In a real implementation, this would restart actual services
            
            return jsonify({
                "success": True,
                "message": "Services restarted successfully"
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Failed to restart services: {str(e)}"
            })
            
    def get_logs(self):
        """Get system logs"""
        # Check authentication
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({
                "success": False, 
                "message": "Admin access required"
            }), 403
            
        # Parse query parameters
        level = request.args.get('level')
        component = request.args.get('component')
        lines = request.args.get('lines', 100)
        query = request.args.get('query')
        
        try:
            lines = int(lines)
        except (TypeError, ValueError):
            lines = 100
            
        try:
            # Get logs
            # In a real implementation, this would fetch actual logs from log files
            
            # Example response
            logs = [
                {
                    "timestamp": "2025-03-30 10:15:23",
                    "level": "INFO",
                    "component": "DataService",
                    "message": "Successfully initialized data sources"
                },
                {
                    "timestamp": "2025-03-30 10:15:24",
                    "level": "DEBUG",
                    "component": "PerformanceTracker",
                    "message": "Loaded trade history from data/trades/history.json"
                },
                {
                    "timestamp": "2025-03-30 10:20:15",
                    "level": "WARNING",
                    "component": "DataService",
                    "message": "Slow response from exchange_api data source (2.5s)"
                },
                {
                    "timestamp": "2025-03-30 10:25:42",
                    "level": "ERROR",
                    "component": "SystemMonitor",
                    "message": "Failed to update system metrics: Connection timeout"
                }
            ]
            
            # Filter logs
            if level:
                logs = [log for log in logs if log['level'] == level.upper()]
                
            if component:
                logs = [log for log in logs if component.lower() in log['component'].lower()]
                
            if query:
                logs = [log for log in logs if query.lower() in log['message'].lower()]
                
            return jsonify({
                "success": True,
                "logs": logs[:lines]
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Failed to get logs: {str(e)}"
            })
            
    def handle_config(self):
        """Handle configuration operations"""
        # Check authentication
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({
                "success": False, 
                "message": "Admin access required"
            }), 403
            
        if request.method == 'GET':
            path = request.args.get('path')
            
            if path:
                # Get specific config file
                try:
                    content = self.config_manager.get_config_content(path)
                    return jsonify({
                        "success": True,
                        "content": content
                    })
                except Exception as e:
                    return jsonify({
                        "success": False,
                        "message": f"Failed to get configuration: {str(e)}"
                    })
            else:
                # List all config files
                try:
                    configs = self.config_manager.get_available_configs()
                    return jsonify({
                        "success": True,
                        "configs": configs
                    })
                except Exception as e:
                    return jsonify({
                        "success": False,
                        "message": f"Failed to list configurations: {str(e)}"
                    })
        elif request.method == 'POST':
            if not request.is_json:
                return jsonify({
                    "success": False, 
                    "message": "Invalid request format"
                }), 400
                
            data = request.json
            path = data.get('path')
            content = data.get('content')
            
            if not path or content is None:
                return jsonify({
                    "success": False, 
                    "message": "Path and content are required"
                }), 400
                
            try:
                success = self.config_manager.update_config(path, content)
                if success:
                    return jsonify({
                        "success": True,
                        "message": "Configuration updated successfully"
                    })
                else:
                    return jsonify({
                        "success": False,
                        "message": "Failed to update configuration"
                    })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "message": f"Failed to update configuration: {str(e)}"
                })
                
    def export_config(self):
        """Export all configurations as a zip file"""
        # Check authentication
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({
                "success": False, 
                "message": "Admin access required"
            }), 403
            
        try:
            # Generate zip file
            zip_file = self.config_manager.export_configs()
            
            # Return the zip file as a download
            return send_file(
                zip_file,
                as_attachment=True,
                download_name='config_export.zip',
                mimetype='application/zip'
            )
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Failed to export configurations: {str(e)}"
            })
            
    def import_config(self):
        """Import configurations from a zip file"""
        # Check authentication
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({
                "success": False, 
                "message": "Admin access required"
            }), 403
            
        if 'config_file' not in request.files:
            return jsonify({
                "success": False, 
                "message": "No file provided"
            }), 400
            
        file = request.files['config_file']
        
        if file.filename == '':
            return jsonify({
                "success": False, 
                "message": "No file selected"
            }), 400
            
        try:
            success = self.config_manager.import_configs(file)
            if success:
                return jsonify({
                    "success": True,
                    "message": "Configurations imported successfully"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to import configurations"
                })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Failed to import configurations: {str(e)}"
            })
            
    def get_users(self):
        """Get all users"""
        # Check authentication
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({
                "success": False, 
                "message": "Admin access required"
            }), 403
            
        try:
            # Get users
            # In a real implementation, this would fetch actual users from the database
            
            # Example response
            users = [
                {
                    "id": 1,
                    "username": "admin",
                    "email": "admin@example.com",
                    "role": "admin",
                    "active": True,
                    "last_login": "2025-03-30 08:45:12"
                },
                {
                    "id": 2,
                    "username": "analyst",
                    "email": "analyst@example.com",
                    "role": "analyst",
                    "active": True,
                    "last_login": "2025-03-29 16:20:33"
                },
                {
                    "id": 3,
                    "username": "viewer",
                    "email": "viewer@example.com",
                    "role": "viewer",
                    "active": False,
                    "last_login": "2025-03-15 11:10:45"
                }
            ]
            
            return jsonify({
                "success": True,
                "users": users
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Failed to get users: {str(e)}"
            })
```

#### HTML Template

```html
<!-- admin_panel.html -->
<div class="admin-panel-container">
    <div class="admin-tabs">
        <button class="admin-tab active" data-tab="system">System</button>
        <button class="admin-tab" data-tab="users">Users</button>
        <button class="admin-tab" data-tab="logs">Logs</button>
        <button class="admin-tab" data-tab="config">Configuration</button>
    </div>
    
    <div class="admin-content">
        <!-- System Tab -->
        <div class="admin-tab-content" id="system-tab-content">
            <div class="section-header">
                <h3>System Status</h3>
                <div class="section-actions">
                    <button id="run-diagnostics" class="btn btn-primary">Run Diagnostics</button>
                </div>
            </div>
            
            <div id="system-status" class="status-cards">
                <!-- Status cards will be inserted here -->
            </div>
            
            <div id="diagnostic-results" class="diagnostic-results">
                <!-- Diagnostic results will be inserted here -->
            </div>
            
            <div class="section-header">
                <h3>Quick Actions</h3>
            </div>
            
            <div class="action-buttons">
                <button id="enable-real-data" class="btn btn-primary">Enable Real Data</button>
                <button id="disable-real-data" class="btn btn-secondary">Disable Real Data</button>
                <button id="restart-services" class="btn btn-warning">Restart Services</button>
            </div>
        </div>
        
        <!-- Users Tab -->
        <div class="admin-tab-content" id="users-tab-content" style="display: none;">
            <div class="section-header">
                <h3>User Management</h3>
                <div class="section-actions">
                    <button id="add-user" class="btn btn-primary">Add User</button>
                </div>
            </div>
            
            <div id="user-list" class="user-list">
                <!-- User items will be inserted here -->
            </div>
        </div>
        
        <!-- Logs Tab -->
        <div class="admin-tab-content" id="logs-tab-content" style="display: none;">
            <div class="section-header">
                <h3>System Logs</h3>
                <div class="section-actions">
                    <button id="refresh-logs" class="btn btn-secondary">Refresh</button>
                    <button id="download-logs" class="btn btn-secondary">Download</button>
                </div>
            </div>
            
            <form id="log-filter-form" class="log-filter">
                <div class="filter-group">
                    <label for="log-level">Level</label>
                    <select id="log-level" class="form-select">
                        <option value="">All Levels</option>
                        <option value="DEBUG">Debug</option>
                        <option value="INFO">Info</option>
                        <option value="WARNING">Warning</option>
                        <option value="ERROR">Error</option>
                        <option value="CRITICAL">Critical</option>
                    </select>
                </div>
                
                <div class="filter-group">
                    <label for="log-component">Component</label>
                    <select id="log-component" class="form-select">
                        <option value="">All Components</option>
                        <option value="DataService">Data Service</option>
                        <option value="SystemMonitor">System Monitor</option>
                        <option value="PerformanceTracker">Performance Tracker</option>
                        <option value="Dashboard">Dashboard</option>
                    </select>
                </div>
                
                <div class="filter-group">
                    <label for="log-lines">Lines</label>
                    <select id="log-lines" class="form-select">
                        <option value="100">100</option>
                        <option value="500">500</option>
                        <option value="1000">1000</option>
                    </select>
                </div>
                
                <div class="filter-group">
                    <label for="log-query">Search</label>
                    <input type="text" id="log-query" class="form-control" placeholder="Search logs...">
                </div>
                
                <button type="submit" class="btn btn-primary">Apply</button>
            </form>
            
            <div id="log-content" class="log-content">
                <!-- Log entries will be inserted here -->
            </div>
        </div>
        
        <!-- Configuration Tab -->
        <div class="admin-tab-content" id="config-tab-content" style="display: none;">
            <div class="section-header">
                <h3>System Configuration</h3>
                <div class="section-actions">
                    <button id="export-config" class="btn btn-secondary">Export All</button>
                    <button id="import-config" class="btn btn-secondary">Import</button>
                    <input type="file" id="config-import-file" style="display: none;">
                </div>
            </div>
            
            <div id="config-list" class="config-list">
                <!-- Config items will be inserted here -->
            </div>
        </div>
    </div>
</div>

<!-- Config Editor Modal -->
<div id="config-editor-modal" class="modal">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="editor-title">Edit Configuration</h5>
                <button type="button" class="btn-close" id="close-editor"></button>
            </div>
            <div class="modal-body">
                <textarea id="config-editor" class="code-editor"></textarea>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" id="cancel-config">Cancel</button>
                <button type="button" class="btn btn-primary" id="save-config">Save Changes</button>
            </div>
        </div>
    </div>
</div>

<!-- Config Viewer Modal -->
<div id="config-viewer-modal" class="modal">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="viewer-title">View Configuration</h5>
                <button type="button" class="btn-close" id="close-viewer"></button>
            </div>
            <div class="modal-body">
                <pre id="config-viewer" class="code-viewer"></pre>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" id="close-view">Close</button>
            </div>
        </div>
    </div>
</div>
```

### Testing Strategy

1. **Unit Tests**
   - Test API endpoints with mock requests
   - Test admin control operations
   - Test configuration handling

2. **Integration Tests**
   - Test end-to-end admin workflows
   - Test user management
   - Test configuration import/export

## Integration Testing

This section outlines the integration testing strategy for all Phase 2 components, ensuring they work together seamlessly.

### Test Categories

1. **Component Integration Tests**
   - Test interactions between frontend components
   - Test communication between frontend and backend
   - Test data flow between components

2. **End-to-End Workflow Tests**
   - Test complete user workflows
   - Test system behavior under different configurations
   - Test error handling and recovery

3. **Regression Tests**
   - Test for unintended side effects
   - Verify existing functionality still works
   - Ensure backward compatibility

### Test Scenarios

1. **Data Source Configuration Workflow**
   - Enable real data through settings panel
   - Verify configuration is saved and persisted
   - Restart the system and verify settings are maintained
   - Check that real data is actually being used

2. **Status Monitoring and Configuration**
   - Artificially degrade a data source
   - Verify status panel shows the degradation
   - Use status panel to test connection
   - Verify test results are displayed correctly

3. **Admin Operations**
   - Import a modified configuration
   - Verify configuration changes are applied
   - Use admin panel to restart services
   - Verify services restart correctly

### Automated Testing

1. **Setup Test Environment**
   - Create isolated testing environment
   - Pre-populate with test data
   - Configure mock data sources

2. **Write Test Scripts**
   - Use pytest for backend testing
   - Use Jest for frontend testing
   - Use Selenium for end-to-end testing

3. **Continuous Integration**
   - Run tests on every pull request
   - Automate deployment to test environment
   - Generate test reports

## Deployment Strategy

This section outlines the strategy for deploying the Phase 2 components to production.

### Deployment Steps

1. **Pre-Deployment Preparation**
   - Freeze development on Phase 2 components
   - Run complete test suite
   - Create deployment package

2. **Staging Deployment**
   - Deploy to staging environment
   - Conduct user acceptance testing
   - Verify performance and stability

3. **Production Deployment**
   - Schedule deployment window
   - Create backup of current configuration
   - Deploy new components
   - Verify successful deployment

4. **Post-Deployment Activities**
   - Monitor system performance
   - Collect user feedback
   - Address any issues or bugs

### Rollback Plan

1. **Rollback Triggers**
   - Critical functionality is broken
   - Performance degradation exceeds threshold
   - Security vulnerability is discovered

2. **Rollback Process**
   - Stop the system
   - Restore configuration from backup
   - Restart with previous version
   - Verify functionality

3. **Communication Plan**
   - Notify users of rollback
   - Provide estimated resolution time
   - Update stakeholders on recovery progress