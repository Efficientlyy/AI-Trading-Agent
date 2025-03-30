/**
 * Admin Controls JavaScript
 * 
 * Handles the admin controls panel functionality, including:
 * - System status and diagnostics
 * - User management
 * - Log viewing and filtering
 * - Configuration management
 */

// Admin Controls Controller
class AdminControlsController {
    constructor() {
        // DOM Elements
        this.container = document.querySelector('.admin-panel-container');
        this.tabs = document.querySelectorAll('.admin-tab');
        this.tabContents = document.querySelectorAll('.admin-tab-content');

        // System Tab Elements
        this.systemStatus = document.getElementById('system-status');
        this.diagnosticResults = document.getElementById('diagnostic-results');
        this.runDiagnosticsButton = document.getElementById('run-diagnostics');
        this.enableRealDataButton = document.getElementById('enable-real-data');
        this.disableRealDataButton = document.getElementById('disable-real-data');
        this.restartServicesButton = document.getElementById('restart-services');

        // Templates
        this.statusCardTemplate = document.getElementById('status-card-template');
        this.userItemTemplate = document.getElementById('user-item-template');
        this.logEntryTemplate = document.getElementById('log-entry-template');
        this.configItemTemplate = document.getElementById('config-item-template');
        this.diagnosticCheckTemplate = document.getElementById('diagnostic-check-template');

        // State
        this.currentTab = 'system';

        // Initialize
        this.init();
    }

    /**
     * Initialize the admin controls panel
     */
    init() {
        // Add event listeners
        this.addEventListeners();

        // Load initial data
        this.loadSystemStatus();
    }

    /**
     * Add event listeners
     */
    addEventListeners() {
        // Tab switching
        this.tabs.forEach(tab => {
            tab.addEventListener('click', this.switchTab.bind(this));
        });

        // System tab
        if (this.runDiagnosticsButton) {
            this.runDiagnosticsButton.addEventListener('click', this.runDiagnostics.bind(this));
        }

        if (this.enableRealDataButton) {
            this.enableRealDataButton.addEventListener('click', this.enableRealData.bind(this));
        }

        if (this.disableRealDataButton) {
            this.disableRealDataButton.addEventListener('click', this.disableRealData.bind(this));
        }

        if (this.restartServicesButton) {
            this.restartServicesButton.addEventListener('click', this.restartServices.bind(this));
        }
    }

    /**
     * Switch between tabs
     * @param {Event} event - The click event
     */
    switchTab(event) {
        const tabId = event.target.dataset.tab;
        if (!tabId) return;

        // Update active tab
        this.tabs.forEach(tab => {
            tab.classList.remove('active');
        });
        event.target.classList.add('active');

        // Show selected tab content
        this.tabContents.forEach(content => {
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

    /**
     * Load system status
     */
    loadSystemStatus() {
        if (!this.systemStatus) return;

        // Show loading state
        this.systemStatus.innerHTML = `
            <div class="loading-indicator">
                <div class="spinner"></div>
                <div class="loading-text">Loading system status...</div>
            </div>
        `;

        // Clear diagnostic results
        if (this.diagnosticResults) {
            this.diagnosticResults.innerHTML = '';
        }

        // Fetch system status
        fetch('/api/admin/system/status')
            .then(response => response.json())
            .then(data => {
                this.updateSystemStatus(data);

                // Update real data buttons
                if (this.enableRealDataButton && this.disableRealDataButton) {
                    this.enableRealDataButton.disabled = data.real_data_enabled;
                    this.disableRealDataButton.disabled = !data.real_data_enabled;
                }
            })
            .catch(error => {
                console.error('Error loading system status:', error);
                this.showError('Failed to load system status');
            });
    }

    /**
     * Update system status UI
     * @param {Object} data - The system status data
     */
    updateSystemStatus(data) {
        if (!this.systemStatus || !this.statusCardTemplate) return;

        // Clear system status
        this.systemStatus.innerHTML = '';

        // Create system status card
        const systemCard = this.statusCardTemplate.content.cloneNode(true);
        const systemTitle = systemCard.querySelector('.status-title');
        const systemValue = systemCard.querySelector('.status-value');
        const systemDetails = systemCard.querySelector('.status-details');

        if (systemTitle) {
            systemTitle.textContent = 'System Status';
        }

        if (systemValue) {
            systemValue.textContent = data.system_health;
            systemValue.className = `status-value ${data.system_health.toLowerCase()}`;
        }

        if (systemDetails) {
            systemDetails.innerHTML = `
                <div class="detail-item">
                    <span class="detail-label">Uptime:</span>
                    <span class="detail-value">${data.uptime || 'N/A'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">CPU Usage:</span>
                    <span class="detail-value">${data.cpu_usage ? data.cpu_usage + '%' : 'N/A'}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Memory Usage:</span>
                    <span class="detail-value">${data.memory_usage ? data.memory_usage + '%' : 'N/A'}</span>
                </div>
            `;
        }

        this.systemStatus.appendChild(systemCard);

        // Add more status cards as needed
    }

    /**
     * Run system diagnostics
     */
    runDiagnostics() {
        if (!this.runDiagnosticsButton || !this.diagnosticResults) return;

        // Show loading state
        this.runDiagnosticsButton.disabled = true;
        this.runDiagnosticsButton.classList.add('loading');

        // Run diagnostics
        fetch('/api/admin/system/diagnostics', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.showSuccess('Diagnostics completed');
                    this.updateDiagnosticResults(data);
                } else {
                    this.showError(`Diagnostics failed: ${data.message}`);
                }

                // Reset button state
                this.runDiagnosticsButton.disabled = false;
                this.runDiagnosticsButton.classList.remove('loading');

                // Refresh system status
                this.loadSystemStatus();
            })
            .catch(error => {
                console.error('Error running diagnostics:', error);
                this.showError('Failed to run diagnostics');

                // Reset button state
                this.runDiagnosticsButton.disabled = false;
                this.runDiagnosticsButton.classList.remove('loading');
            });
    }

    /**
     * Enable real data
     */
    enableRealData() {
        this.updateRealDataStatus(true);
    }

    /**
     * Disable real data
     */
    disableRealData() {
        this.updateRealDataStatus(false);
    }

    /**
     * Update real data status
     * @param {boolean} enable - Whether to enable or disable real data
     */
    updateRealDataStatus(enable) {
        // Show loading state
        const button = enable ? this.enableRealDataButton : this.disableRealDataButton;
        if (button) {
            button.disabled = true;
            button.classList.add('loading');
        }

        // Update real data status
        fetch('/api/admin/system/real-data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: enable })
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.showSuccess(`Real data ${enable ? 'enabled' : 'disabled'} successfully`);
                    this.loadSystemStatus();
                } else {
                    this.showError(`Failed to ${enable ? 'enable' : 'disable'} real data: ${data.message}`);
                }

                // Reset button state
                if (button) {
                    button.disabled = false;
                    button.classList.remove('loading');
                }
            })
            .catch(error => {
                console.error(`Error ${enable ? 'enabling' : 'disabling'} real data:`, error);
                this.showError(`Failed to ${enable ? 'enable' : 'disable'} real data`);

                // Reset button state
                if (button) {
                    button.disabled = false;
                    button.classList.remove('loading');
                }
            });
    }

    /**
     * Restart services
     */
    restartServices() {
        if (!this.restartServicesButton) return;

        // Show confirmation dialog
        if (!confirm('Are you sure you want to restart all services? This may disrupt ongoing operations.')) {
            return;
        }

        // Show loading state
        this.restartServicesButton.disabled = true;
        this.restartServicesButton.classList.add('loading');

        // Restart services
        fetch('/api/admin/system/restart', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.showSuccess('Services restarted successfully');
                    setTimeout(() => this.loadSystemStatus(), 5000);
                } else {
                    this.showError(`Failed to restart services: ${data.message}`);
                }

                // Reset button state
                this.restartServicesButton.disabled = false;
                this.restartServicesButton.classList.remove('loading');
            })
            .catch(error => {
                console.error('Error restarting services:', error);
                this.showError('Failed to restart services');

                // Reset button state
                this.restartServicesButton.disabled = false;
                this.restartServicesButton.classList.remove('loading');
            });
    }

    /**
     * Load users
     */
    loadUsers() {
        // Implementation for loading users
        console.log('Loading users...');
    }

    /**
     * Load logs
     */
    loadLogs() {
        // Implementation for loading logs
        console.log('Loading logs...');
    }

    /**
     * Load configurations
     */
    loadConfigurations() {
        // Implementation for loading configurations
        console.log('Loading configurations...');
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
}

// Initialize admin controls when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global admin controls instance
    window.adminControls = new AdminControlsController();
});
