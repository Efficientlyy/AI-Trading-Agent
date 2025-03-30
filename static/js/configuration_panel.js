/**
 * Configuration Panel JavaScript
 * 
 * Handles the configuration panel functionality, including:
 * - Loading configuration from the server
 * - Updating the UI based on the loaded configuration
 * - Saving configuration to the server
 * - Handling configuration panel tabs
 * - Managing data source connections
 */

// Configuration Panel Controller
class ConfigurationPanelController {
    constructor() {
        // DOM Elements
        this.container = document.querySelector('.configuration-panel-container');
        this.saveButton = document.getElementById('save-configuration');
        this.resetButton = document.getElementById('reset-configuration');
        this.tabButtons = document.querySelectorAll('.configuration-tab');
        this.configurationPanes = document.querySelectorAll('.configuration-pane');

        // General settings
        this.enableRealDataToggle = document.getElementById('enable-real-data');
        this.autoRecoveryToggle = document.getElementById('auto-recovery');
        this.recoveryAttemptsSelect = document.getElementById('recovery-attempts');
        this.healthCheckIntervalSelect = document.getElementById('health-check-interval');

        // Connections
        this.connectionList = document.getElementById('connection-list');
        this.editConnectionsButton = document.getElementById('edit-connections');
        this.testAllConnectionsButton = document.getElementById('test-all-connections');

        // Fallback strategy
        this.fallbackStrategySelect = document.getElementById('fallback-strategy');
        this.useCachedDataToggle = document.getElementById('use-cached-data');
        this.cacheExpirySelect = document.getElementById('cache-expiry');
        this.useMockDataToggle = document.getElementById('use-mock-data');

        // Advanced settings
        this.loggingLevelSelect = document.getElementById('logging-level');
        this.performanceMetricsToggle = document.getElementById('performance-metrics');
        this.metricsRetentionSelect = document.getElementById('metrics-retention');
        this.concurrentRequestsSelect = document.getElementById('concurrent-requests');
        this.requestTimeoutSelect = document.getElementById('request-timeout');

        // Templates
        this.connectionItemTemplate = document.getElementById('connection-item-template');

        // Current configuration
        this.configuration = {};

        // Initialize
        this.init();
    }

    /**
     * Initialize the configuration panel
     */
    init() {
        // Add event listeners
        this.addEventListeners();

        // Load configuration
        this.loadConfiguration();
    }

    /**
     * Add event listeners
     */
    addEventListeners() {
        // Save button
        if (this.saveButton) {
            this.saveButton.addEventListener('click', () => {
                this.saveConfiguration();
            });
        }

        // Reset button
        if (this.resetButton) {
            this.resetButton.addEventListener('click', () => {
                this.resetConfiguration();
            });
        }

        // Tab buttons
        this.tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tab = button.getAttribute('data-tab');
                this.showTab(tab);
            });
        });

        // Edit connections button
        if (this.editConnectionsButton) {
            this.editConnectionsButton.addEventListener('click', () => {
                this.openConnectionEditor();
            });
        }

        // Test all connections button
        if (this.testAllConnectionsButton) {
            this.testAllConnectionsButton.addEventListener('click', () => {
                this.testAllConnections();
            });
        }

        // Fallback strategy select
        if (this.fallbackStrategySelect) {
            this.fallbackStrategySelect.addEventListener('change', () => {
                this.updateFallbackControls();
            });
        }
    }

    /**
     * Show a specific tab
     * @param {string} tabName - The name of the tab to show
     */
    showTab(tabName) {
        // Update tab buttons
        this.tabButtons.forEach(button => {
            if (button.getAttribute('data-tab') === tabName) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });

        // Update tab panes
        this.configurationPanes.forEach(pane => {
            if (pane.id === `${tabName}-configuration`) {
                pane.classList.add('active');
            } else {
                pane.classList.remove('active');
            }
        });
    }

    /**
     * Load configuration from the server
     */
    loadConfiguration() {
        // Show loading indicator
        this.showLoading();

        // Fetch configuration from server
        fetch('/api/settings/real-data-config')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load configuration');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    // Store configuration
                    this.configuration = data.config;

                    // Update UI
                    this.updateUI();

                    // Hide loading indicator
                    this.hideLoading();
                } else {
                    throw new Error(data.message || 'Failed to load configuration');
                }
            })
            .catch(error => {
                console.error('Error loading configuration:', error);
                this.showError('Failed to load configuration');
                this.hideLoading();
            });
    }

    /**
     * Update the UI based on the current configuration
     */
    updateUI() {
        // General settings
        this.updateGeneralSettings();

        // Connections
        this.updateConnections();

        // Fallback strategy
        this.updateFallbackStrategy();

        // Advanced settings
        this.updateAdvancedSettings();
    }

    /**
     * Update general settings UI
     */
    updateGeneralSettings() {
        if (this.enableRealDataToggle) {
            this.enableRealDataToggle.checked = this.configuration.enabled || false;
        }

        if (this.autoRecoveryToggle) {
            this.autoRecoveryToggle.checked = this.configuration.auto_recovery !== false;
        }

        if (this.recoveryAttemptsSelect) {
            this.recoveryAttemptsSelect.value = this.configuration.recovery_attempts || '3';
        }

        if (this.healthCheckIntervalSelect) {
            this.healthCheckIntervalSelect.value = this.configuration.health_check_interval || '60';
        }
    }

    /**
     * Update connections UI
     */
    updateConnections() {
        if (!this.connectionList || !this.connectionItemTemplate) {
            return;
        }

        // Clear connection list
        this.connectionList.innerHTML = '';

        // Get connections
        const connections = this.configuration.connections || {};

        // Add connection items
        Object.entries(connections).forEach(([id, connection]) => {
            // Clone template
            const item = this.connectionItemTemplate.content.cloneNode(true);

            // Update item data
            const connectionName = item.querySelector('.connection-name');
            const connectionStatus = item.querySelector('.connection-status');
            const connectionToggle = item.querySelector('.connection-toggle');
            const testButton = item.querySelector('.test-connection');

            if (connectionName) {
                connectionName.textContent = this.formatConnectionName(id);
            }

            if (connectionStatus) {
                connectionStatus.textContent = connection.enabled ? 'Enabled' : 'Disabled';
                connectionStatus.className = `connection-status ${connection.enabled ? 'enabled' : 'disabled'}`;
            }

            if (connectionToggle) {
                connectionToggle.checked = connection.enabled || false;
                connectionToggle.dataset.id = id;

                // Add event listener
                connectionToggle.addEventListener('change', (event) => {
                    this.toggleConnection(id, event.target.checked);
                });
            }

            if (testButton) {
                testButton.dataset.id = id;

                // Add event listener
                testButton.addEventListener('click', (event) => {
                    this.testConnection(id, testButton);
                });
            }

            // Create container
            const container = document.createElement('div');
            container.className = 'connection-item-container';
            container.dataset.id = id;
            container.appendChild(item);

            // Add to list
            this.connectionList.appendChild(container);
        });
    }

    /**
     * Update fallback strategy UI
     */
    updateFallbackStrategy() {
        const fallback = this.configuration.fallback_strategy || {};

        if (this.fallbackStrategySelect) {
            this.fallbackStrategySelect.value = fallback.strategy || 'cache_then_mock';
        }

        if (this.useCachedDataToggle) {
            this.useCachedDataToggle.checked = fallback.use_cached_data !== false;
        }

        if (this.cacheExpirySelect) {
            this.cacheExpirySelect.value = fallback.cache_expiry_seconds || '3600';
        }

        if (this.useMockDataToggle) {
            this.useMockDataToggle.checked = fallback.use_mock_data_on_failure !== false;
        }

        // Update fallback controls based on selected strategy
        this.updateFallbackControls();
    }

    /**
     * Update advanced settings UI
     */
    updateAdvancedSettings() {
        const advanced = this.configuration.advanced || {};

        if (this.loggingLevelSelect) {
            this.loggingLevelSelect.value = advanced.logging_level || 'info';
        }

        if (this.performanceMetricsToggle) {
            this.performanceMetricsToggle.checked = advanced.collect_performance_metrics !== false;
        }

        if (this.metricsRetentionSelect) {
            this.metricsRetentionSelect.value = advanced.metrics_retention_days || '7';
        }

        if (this.concurrentRequestsSelect) {
            this.concurrentRequestsSelect.value = advanced.max_concurrent_requests || '5';
        }

        if (this.requestTimeoutSelect) {
            this.requestTimeoutSelect.value = advanced.default_timeout_seconds || '10';
        }
    }

    /**
     * Update fallback controls based on selected strategy
     */
    updateFallbackControls() {
        if (!this.fallbackStrategySelect) {
            return;
        }

        const strategy = this.fallbackStrategySelect.value;

        // Update cache controls
        if (this.useCachedDataToggle && this.cacheExpirySelect) {
            const useCacheControls = strategy === 'cache_then_mock' || strategy === 'cache_only';

            this.useCachedDataToggle.disabled = !useCacheControls;
            this.cacheExpirySelect.disabled = !useCacheControls;

            if (!useCacheControls) {
                this.useCachedDataToggle.checked = strategy === 'cache_only';
            }
        }

        // Update mock data controls
        if (this.useMockDataToggle) {
            const useMockControls = strategy === 'cache_then_mock' || strategy === 'mock_only';

            this.useMockDataToggle.disabled = !useMockControls;

            if (!useMockControls) {
                this.useMockDataToggle.checked = strategy === 'mock_only';
            }
        }
    }

    /**
     * Save configuration to the server
     */
    saveConfiguration() {
        // Get configuration from UI
        const config = this.getConfigurationFromUI();

        // Show loading indicator
        this.showLoading();

        // Send configuration to server
        fetch('/api/settings/real-data-config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                config: config
            })
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to save configuration');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    // Update configuration
                    this.configuration = config;

                    // Show success message
                    this.showSuccess('Configuration saved successfully');

                    // Check if reload is required
                    if (data.reload_required) {
                        this.showReloadNotification();
                    }
                } else {
                    throw new Error(data.message || 'Failed to save configuration');
                }

                // Hide loading indicator
                this.hideLoading();
            })
            .catch(error => {
                console.error('Error saving configuration:', error);
                this.showError('Failed to save configuration');
                this.hideLoading();
            });
    }

    /**
     * Reset configuration to defaults
     */
    resetConfiguration() {
        if (confirm('Are you sure you want to reset the configuration to defaults?')) {
            // Show loading indicator
            this.showLoading();

            // Reset configuration on server
            fetch('/api/settings/real-data-config/reset', {
                method: 'POST'
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to reset configuration');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        // Update configuration
                        this.configuration = data.config;

                        // Update UI
                        this.updateUI();

                        // Show success message
                        this.showSuccess('Configuration reset to defaults');
                    } else {
                        throw new Error(data.message || 'Failed to reset configuration');
                    }

                    // Hide loading indicator
                    this.hideLoading();
                })
                .catch(error => {
                    console.error('Error resetting configuration:', error);
                    this.showError('Failed to reset configuration');
                    this.hideLoading();
                });
        }
    }

    /**
     * Get configuration from UI
     * @returns {Object} The configuration object
     */
    getConfigurationFromUI() {
        // General settings
        const enabled = this.enableRealDataToggle ? this.enableRealDataToggle.checked : false;
        const autoRecovery = this.autoRecoveryToggle ? this.autoRecoveryToggle.checked : true;
        const recoveryAttempts = this.recoveryAttemptsSelect ? parseInt(this.recoveryAttemptsSelect.value, 10) : 3;
        const healthCheckInterval = this.healthCheckIntervalSelect ? parseInt(this.healthCheckIntervalSelect.value, 10) : 60;

        // Connections
        const connections = this.getConnectionsFromUI();

        // Fallback strategy
        const fallbackStrategy = this.fallbackStrategySelect ? this.fallbackStrategySelect.value : 'cache_then_mock';
        const useCachedData = this.useCachedDataToggle ? this.useCachedDataToggle.checked : true;
        const cacheExpiry = this.cacheExpirySelect ? parseInt(this.cacheExpirySelect.value, 10) : 3600;
        const useMockData = this.useMockDataToggle ? this.useMockDataToggle.checked : true;

        // Advanced settings
        const loggingLevel = this.loggingLevelSelect ? this.loggingLevelSelect.value : 'info';
        const performanceMetrics = this.performanceMetricsToggle ? this.performanceMetricsToggle.checked : true;
        const metricsRetention = this.metricsRetentionSelect ? parseInt(this.metricsRetentionSelect.value, 10) : 7;
        const concurrentRequests = this.concurrentRequestsSelect ? parseInt(this.concurrentRequestsSelect.value, 10) : 5;
        const requestTimeout = this.requestTimeoutSelect ? parseInt(this.requestTimeoutSelect.value, 10) : 10;

        // Build configuration object
        return {
            enabled: enabled,
            auto_recovery: autoRecovery,
            recovery_attempts: recoveryAttempts,
            health_check_interval: healthCheckInterval,
            connections: connections,
            fallback_strategy: {
                strategy: fallbackStrategy,
                use_cached_data: useCachedData,
                cache_expiry_seconds: cacheExpiry,
                use_mock_data_on_failure: useMockData
            },
            advanced: {
                logging_level: loggingLevel,
                collect_performance_metrics: performanceMetrics,
                metrics_retention_days: metricsRetention,
                max_concurrent_requests: concurrentRequests,
                default_timeout_seconds: requestTimeout
            }
        };
    }

    /**
     * Get connections from UI
     * @returns {Object} The connections object
     */
    getConnectionsFromUI() {
        // Get current connections
        const connections = { ...this.configuration.connections } || {};

        // Update connection enabled state
        const toggles = document.querySelectorAll('.connection-toggle');
        toggles.forEach(toggle => {
            const id = toggle.dataset.id;
            if (id && connections[id]) {
                connections[id].enabled = toggle.checked;
            }
        });

        return connections;
    }

    /**
     * Toggle a connection's enabled state
     * @param {string} id - The connection ID
     * @param {boolean} enabled - Whether the connection is enabled
     */
    toggleConnection(id, enabled) {
        // Update connection in configuration
        if (this.configuration.connections && this.configuration.connections[id]) {
            this.configuration.connections[id].enabled = enabled;

            // Update UI
            const statusElement = document.querySelector(`.connection-item-container[data-id="${id}"] .connection-status`);
            if (statusElement) {
                statusElement.textContent = enabled ? 'Enabled' : 'Disabled';
                statusElement.className = `connection-status ${enabled ? 'enabled' : 'disabled'}`;
            }
        }
    }

    /**
     * Test a connection
     * @param {string} id - The connection ID
     * @param {HTMLElement} button - The test button
     */
    testConnection(id, button) {
        // Show loading state
        if (button) {
            button.disabled = true;
            button.classList.add('loading');
        }

        // Test connection
        fetch(`/api/system/test-connection?source=${id}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to test connection');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    this.showSuccess(`Connection to ${this.formatConnectionName(id)} successful`);
                } else {
                    this.showError(`Connection to ${this.formatConnectionName(id)} failed: ${data.message}`);
                }

                // Reset button state
                if (button) {
                    button.disabled = false;
                    button.classList.remove('loading');
                }
            })
            .catch(error => {
                console.error('Error testing connection:', error);
                this.showError(`Failed to test connection to ${this.formatConnectionName(id)}`);

                // Reset button state
                if (button) {
                    button.disabled = false;
                    button.classList.remove('loading');
                }
            });
    }

    /**
     * Test all connections
     */
    testAllConnections() {
        // Show loading state
        if (this.testAllConnectionsButton) {
            this.testAllConnectionsButton.disabled = true;
            this.testAllConnectionsButton.classList.add('loading');
        }

        // Get enabled connections
        const connections = this.configuration.connections || {};
        const enabledConnections = Object.entries(connections)
            .filter(([_, connection]) => connection.enabled)
            .map(([id, _]) => id);

        if (enabledConnections.length === 0) {
            this.showWarning('No enabled connections to test');

            // Reset button state
            if (this.testAllConnectionsButton) {
                this.testAllConnectionsButton.disabled = false;
                this.testAllConnectionsButton.classList.remove('loading');
            }

            return;
        }

        // Test each connection
        let completed = 0;
        let successful = 0;
        let failed = 0;

        enabledConnections.forEach(id => {
            fetch(`/api/system/test-connection?source=${id}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to test connection');
                    }
                    return response.json();
                })
                .then(data => {
                    completed++;

                    if (data.success) {
                        successful++;
                    } else {
                        failed++;
                    }

                    // Check if all tests are complete
                    if (completed === enabledConnections.length) {
                        this.showTestResults(successful, failed);
                    }
                })
                .catch(error => {
                    console.error(`Error testing connection ${id}:`, error);

                    completed++;
                    failed++;

                    // Check if all tests are complete
                    if (completed === enabledConnections.length) {
                        this.showTestResults(successful, failed);
                    }
                });
        });
    }

    /**
     * Show test results
     * @param {number} successful - Number of successful tests
     * @param {number} failed - Number of failed tests
     */
    showTestResults(successful, failed) {
        // Reset button state
        if (this.testAllConnectionsButton) {
            this.testAllConnectionsButton.disabled = false;
            this.testAllConnectionsButton.classList.remove('loading');
        }

        // Show results
        if (failed === 0) {
            this.showSuccess(`All ${successful} connections tested successfully`);
        } else if (successful === 0) {
            this.showError(`All ${failed} connections failed`);
        } else {
            this.showWarning(`${successful} connections successful, ${failed} connections failed`);
        }
    }

    /**
     * Open the connection editor
     */
    openConnectionEditor() {
        // Check if connection editor is available
        if (typeof window.connectionEditor === 'undefined') {
            // Load connection editor
            fetch('/api/templates/connection_editor_modal.html')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to load connection editor');
                    }
                    return response.text();
                })
                .then(html => {
                    // Create temporary container
                    const container = document.createElement('div');
                    container.innerHTML = html;

                    // Append to body
                    document.body.appendChild(container.firstChild);

                    // Load connection editor script
                    const script = document.createElement('script');
                    script.src = '/static/js/connection_editor.js';
                    script.onload = () => {
                        // Initialize connection editor
                        if (window.connectionEditor) {
                            window.connectionEditor.openModal();
                        }
                    };
                    document.body.appendChild(script);
                })
                .catch(error => {
                    console.error('Error loading connection editor:', error);
                    this.showError('Failed to load connection editor');
                });
        } else {
            // Open connection editor
            window.connectionEditor.openModal();
        }
    }

    /**
     * Format a connection ID as a readable name
     * @param {string} id - The connection ID
     * @returns {string} The formatted name
     */
    formatConnectionName(id) {
        return id.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    /**
     * Show loading indicator
     */
    showLoading() {
        // Disable buttons
        if (this.saveButton) {
            this.saveButton.disabled = true;
        }

        if (this.resetButton) {
            this.resetButton.disabled = true;
        }

        // Add loading class to container
        if (this.container) {
            this.container.classList.add('loading');
        }
    }

    /**
     * Hide loading indicator
     */
    hideLoading() {
        // Enable buttons
        if (this.saveButton) {
            this.saveButton.disabled = false;
        }

        if (this.resetButton) {
            this.resetButton.disabled = false;
        }

        // Remove loading class from container
        if (this.container) {
            this.container.classList.remove('loading');
        }
    }

    /**
     * Show success message
     * @param {string} message - The success message
     */
    showSuccess(message) {
        if (window.showToast) {
            window.showToast(message, 'success');
        } else {
            alert(message);
        }
    }

    /**
     * Show warning message
     * @param {string} message - The warning message
     */
    showWarning(message) {
        if (window.showToast) {
            window.showToast(message, 'warning');
        } else {
            alert(`Warning: ${message}`);
        }
    }

    /**
     * Show error message
     * @param {string} message - The error message
     */
    showError(message) {
        if (window.showToast) {
            window.showToast(message, 'error');
        } else {
            alert(`Error: ${message}`);
        }
    }

    /**
     * Show reload notification
     */
    showReloadNotification() {
        if (confirm('Some settings require a page reload to take effect. Reload now?')) {
            window.location.reload();
        }
    }
}

// Initialize configuration panel when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global configuration panel instance
    window.configurationPanel = new ConfigurationPanelController();
});