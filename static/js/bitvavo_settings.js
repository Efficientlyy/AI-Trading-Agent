/**
 * Bitvavo API Settings Component
 * 
 * This module provides a UI component for managing Bitvavo API credentials
 * and connection settings.
 */

class BitvavoSettingsComponent {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.isLoading = false;
        this.connectionStatus = 'unknown';
        this.apiKey = '';
        this.apiSecret = '';
        this.initialized = false;
    }

    /**
     * Initialize the component
     */
    async initialize() {
        if (this.initialized) return;

        // Render the initial UI
        this.render();

        // Set up event listeners
        this.setupEventListeners();

        // Check connection status
        await this.checkConnectionStatus();

        this.initialized = true;
    }

    /**
     * Render the component
     */
    render() {
        if (!this.container) {
            console.error(`Container with ID ${this.containerId} not found`);
            return;
        }

        const statusClass = this.getStatusClass();
        const statusText = this.getStatusText();

        this.container.innerHTML = `
            <div class="card mb-4">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">Bitvavo API Settings</h5>
                    <span class="badge ${statusClass}">${statusText}</span>
                </div>
                <div class="card-body">
                    <form id="bitvavo-api-form">
                        <div class="mb-3">
                            <label for="bitvavo-api-key" class="form-label">API Key</label>
                            <input type="text" class="form-control" id="bitvavo-api-key" 
                                placeholder="Enter Bitvavo API Key" value="${this.apiKey}">
                            <div class="form-text">Your Bitvavo API key (keep this secure)</div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="bitvavo-api-secret" class="form-label">API Secret</label>
                            <input type="password" class="form-control" id="bitvavo-api-secret" 
                                placeholder="Enter Bitvavo API Secret" value="${this.apiSecret}">
                            <div class="form-text">Your Bitvavo API secret (keep this secure)</div>
                        </div>
                        
                        <div id="bitvavo-status-message"></div>
                        
                        <div class="d-flex gap-2">
                            <button type="button" class="btn btn-secondary" id="bitvavo-test-btn">
                                <i class="fas fa-plug me-1"></i> Test Connection
                            </button>
                            <button type="button" class="btn btn-primary" id="bitvavo-save-btn">
                                <i class="fas fa-save me-1"></i> Save Credentials
                            </button>
                        </div>
                    </form>
                </div>
                <div class="card-footer">
                    <small class="text-muted">
                        <i class="fas fa-info-circle me-1"></i>
                        Bitvavo is a Netherlands-based cryptocurrency exchange with competitive fees.
                    </small>
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h5 class="mb-0">Connection Settings</h5>
                </div>
                <div class="card-body">
                    <form id="bitvavo-settings-form">
                        <div class="mb-3">
                            <label for="bitvavo-retry-attempts" class="form-label">Retry Attempts</label>
                            <input type="number" class="form-control" id="bitvavo-retry-attempts" 
                                min="1" max="10" value="3">
                            <div class="form-text">Number of retry attempts for failed API calls</div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="bitvavo-timeout" class="form-label">Timeout (seconds)</label>
                            <input type="number" class="form-control" id="bitvavo-timeout" 
                                min="1" max="60" value="10">
                            <div class="form-text">API call timeout in seconds</div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="bitvavo-cache-duration" class="form-label">Cache Duration (seconds)</label>
                            <input type="number" class="form-control" id="bitvavo-cache-duration" 
                                min="0" max="3600" value="60">
                            <div class="form-text">Duration to cache API responses (0 to disable)</div>
                        </div>
                        
                        <div id="bitvavo-settings-message"></div>
                        
                        <button type="button" class="btn btn-primary" id="bitvavo-settings-save-btn">
                            <i class="fas fa-save me-1"></i> Save Settings
                        </button>
                    </form>
                </div>
            </div>
        `;
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        const testBtn = document.getElementById('bitvavo-test-btn');
        const saveBtn = document.getElementById('bitvavo-save-btn');
        const settingsSaveBtn = document.getElementById('bitvavo-settings-save-btn');

        if (testBtn) {
            testBtn.addEventListener('click', () => this.testConnection());
        }

        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveCredentials());
        }

        if (settingsSaveBtn) {
            settingsSaveBtn.addEventListener('click', () => this.saveSettings());
        }
    }

    /**
     * Get status CSS class based on connection status
     */
    getStatusClass() {
        switch (this.connectionStatus) {
            case 'connected':
                return 'bg-success';
            case 'error':
                return 'bg-danger';
            case 'configuring':
                return 'bg-warning';
            default:
                return 'bg-secondary';
        }
    }

    /**
     * Get status text based on connection status
     */
    getStatusText() {
        switch (this.connectionStatus) {
            case 'connected':
                return 'Connected';
            case 'error':
                return 'Connection Error';
            case 'configuring':
                return 'Configuring';
            default:
                return 'Not Configured';
        }
    }

    /**
     * Check connection status
     */
    async checkConnectionStatus() {
        try {
            this.setLoading(true);

            const response = await fetch('/api/settings/bitvavo/status');
            const data = await response.json();

            if (data.configured) {
                this.connectionStatus = data.connected ? 'connected' : 'error';
            } else {
                this.connectionStatus = 'unknown';
            }

            this.render();
        } catch (error) {
            console.error('Error checking Bitvavo connection status:', error);
            this.connectionStatus = 'error';
            this.render();
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Test connection with current credentials
     */
    async testConnection() {
        try {
            this.setLoading(true);
            this.clearMessages();

            const apiKey = document.getElementById('bitvavo-api-key').value;
            const apiSecret = document.getElementById('bitvavo-api-secret').value;

            if (!apiKey || !apiSecret) {
                this.showMessage('bitvavo-status-message', 'Please enter both API key and secret', 'danger');
                return;
            }

            const response = await fetch('/api/settings/bitvavo/test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    apiKey,
                    apiSecret
                })
            });

            const data = await response.json();

            if (data.success) {
                this.showMessage('bitvavo-status-message', data.message, 'success');
                this.connectionStatus = 'connected';
            } else {
                this.showMessage('bitvavo-status-message', data.message, 'danger');
                this.connectionStatus = 'error';
            }

            this.render();
        } catch (error) {
            console.error('Error testing Bitvavo connection:', error);
            this.showMessage('bitvavo-status-message', 'Error testing connection', 'danger');
            this.connectionStatus = 'error';
            this.render();
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Save API credentials
     */
    async saveCredentials() {
        try {
            this.setLoading(true);
            this.clearMessages();

            const apiKey = document.getElementById('bitvavo-api-key').value;
            const apiSecret = document.getElementById('bitvavo-api-secret').value;

            if (!apiKey || !apiSecret) {
                this.showMessage('bitvavo-status-message', 'Please enter both API key and secret', 'danger');
                return;
            }

            const response = await fetch('/api/settings/bitvavo/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    apiKey,
                    apiSecret
                })
            });

            const data = await response.json();

            if (data.success) {
                this.showMessage('bitvavo-status-message', data.message, 'success');
                this.apiKey = apiKey;
                this.apiSecret = apiSecret;
                this.connectionStatus = 'connected';
            } else {
                this.showMessage('bitvavo-status-message', data.message, 'danger');
            }

            this.render();
        } catch (error) {
            console.error('Error saving Bitvavo credentials:', error);
            this.showMessage('bitvavo-status-message', 'Error saving credentials', 'danger');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Save connection settings
     */
    async saveSettings() {
        try {
            this.setLoading(true);
            this.clearMessages();

            const retryAttempts = document.getElementById('bitvavo-retry-attempts').value;
            const timeout = document.getElementById('bitvavo-timeout').value;
            const cacheDuration = document.getElementById('bitvavo-cache-duration').value;

            const response = await fetch('/api/settings/bitvavo/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    retryAttempts: parseInt(retryAttempts, 10),
                    timeoutSeconds: parseInt(timeout, 10),
                    cacheDurationSeconds: parseInt(cacheDuration, 10)
                })
            });

            const data = await response.json();

            if (data.success) {
                this.showMessage('bitvavo-settings-message', data.message, 'success');
            } else {
                this.showMessage('bitvavo-settings-message', data.message, 'danger');
            }
        } catch (error) {
            console.error('Error saving Bitvavo settings:', error);
            this.showMessage('bitvavo-settings-message', 'Error saving settings', 'danger');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Show a message in the specified container
     */
    showMessage(containerId, message, type) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="alert alert-${type} mt-3">
                    ${message}
                </div>
            `;
        }
    }

    /**
     * Clear all message containers
     */
    clearMessages() {
        const containers = [
            document.getElementById('bitvavo-status-message'),
            document.getElementById('bitvavo-settings-message')
        ];

        containers.forEach(container => {
            if (container) {
                container.innerHTML = '';
            }
        });
    }

    /**
     * Set loading state
     */
    setLoading(isLoading) {
        this.isLoading = isLoading;

        const buttons = [
            document.getElementById('bitvavo-test-btn'),
            document.getElementById('bitvavo-save-btn'),
            document.getElementById('bitvavo-settings-save-btn')
        ];

        buttons.forEach(button => {
            if (button) {
                button.disabled = isLoading;
            }
        });
    }
}

// Initialize the component when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if the container exists
    if (document.getElementById('bitvavo-settings-container')) {
        const bitvavoSettings = new BitvavoSettingsComponent('bitvavo-settings-container');
        bitvavoSettings.initialize();
    }
});