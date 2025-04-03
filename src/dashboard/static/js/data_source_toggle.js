/**
 * Data Source Toggle Component
 * 
 * This component provides a toggle between mock and real data sources,
 * with status indicators for real data components.
 */

class DataSourceToggle {
    constructor(options = {}) {
        this.options = {
            containerSelector: '.data-source-toggle',
            mockButtonId: 'mock-data-btn',
            realButtonId: 'real-data-btn',
            statusEndpoint: '/api/system/data-source-status',
            toggleEndpoint: '/api/system/data-source',
            componentStatusEndpoint: '/api/dashboard/components',
            refreshInterval: 30000, // 30 seconds
            onToggle: null, // Callback function when data source is toggled
            ...options
        };

        this.container = null;
        this.mockButton = null;
        this.realButton = null;
        this.statusIndicator = null;
        this.componentsContainer = null;
        this.refreshTimer = null;
        this.currentSource = null;

        this.init();
    }

    /**
     * Initialize the component
     */
    init() {
        // Create the component structure
        this.createToggle();

        // Fetch initial status
        this.fetchStatus();

        // Set up refresh interval
        this.startRefreshTimer();

        // Add event listeners
        this.addEventListeners();
    }

    /**
     * Create the toggle component structure
     */
    createToggle() {
        // Find or create container
        this.container = document.querySelector(this.options.containerSelector);
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.className = 'data-source-toggle';
            document.querySelector('.system-controls').appendChild(this.container);
        }

        // Create component HTML
        this.container.innerHTML = `
            <div class="data-source-label">Data Source:</div>
            <div class="data-source-buttons">
                <button id="${this.options.mockButtonId}" class="data-source-btn">Mock</button>
                <button id="${this.options.realButtonId}" class="data-source-btn">Real</button>
            </div>
            <div class="data-source-status">
                <span class="status-text">Loading...</span>
            </div>
            <div class="real-data-components">
                <div class="real-data-components-title">Real Data Components</div>
                <div class="component-list"></div>
            </div>
        `;

        // Store references to elements
        this.mockButton = document.getElementById(this.options.mockButtonId);
        this.realButton = document.getElementById(this.options.realButtonId);
        this.statusIndicator = this.container.querySelector('.data-source-status');
        this.componentsContainer = this.container.querySelector('.component-list');
    }

    /**
     * Add event listeners to buttons
     */
    addEventListeners() {
        this.mockButton.addEventListener('click', () => this.toggleDataSource('mock'));
        this.realButton.addEventListener('click', () => this.toggleDataSource('real'));
    }

    /**
     * Toggle between mock and real data sources
     * @param {string} source - The data source to switch to ('mock' or 'real')
     */
    toggleDataSource(source) {
        if (this.currentSource === source) return;

        // Disable buttons during toggle
        this.mockButton.disabled = true;
        this.realButton.disabled = true;

        // Update UI to show loading state
        this.statusIndicator.innerHTML = '<span class="status-text">Switching...</span>';

        // Send request to toggle data source
        fetch(this.options.toggleEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ source })
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.updateToggleState(source);

                    // Call callback if provided
                    if (typeof this.options.onToggle === 'function') {
                        this.options.onToggle(source);
                    }

                    // Show success notification
                    this.showNotification('success', `Switched to ${source} data source`);

                    // Refresh the page data
                    this.refreshPageData();
                } else {
                    // Show error notification
                    this.showNotification('error', data.message || `Failed to switch to ${source} data`);

                    // Revert UI to previous state
                    this.updateToggleState(this.currentSource);
                }
            })
            .catch(error => {
                console.error('Error toggling data source:', error);
                this.showNotification('error', 'Error switching data source');
                this.updateToggleState(this.currentSource);
            })
            .finally(() => {
                // Re-enable buttons
                this.mockButton.disabled = false;
                this.realButton.disabled = false;
            });
    }

    /**
     * Update the toggle state based on current data source
     * @param {string} source - The current data source ('mock' or 'real')
     */
    updateToggleState(source) {
        this.currentSource = source;

        // Update button states
        this.mockButton.classList.toggle('active', source === 'mock');
        this.realButton.classList.toggle('active', source === 'real');

        // Update status indicator
        this.statusIndicator.innerHTML = `
            <span class="status-text">${source === 'mock' ? 'Using Mock Data' : 'Using Real Data'}</span>
        `;

        // Update components visibility
        if (source === 'real') {
            this.fetchComponentStatus();
        }
    }

    /**
     * Fetch the current data source status
     */
    fetchStatus() {
        fetch(this.options.statusEndpoint)
            .then(response => response.json())
            .then(data => {
                this.updateToggleState(data.source);
            })
            .catch(error => {
                console.error('Error fetching data source status:', error);
                // Default to mock if status can't be determined
                this.updateToggleState('mock');
            });
    }

    /**
     * Fetch component status for real data
     */
    fetchComponentStatus() {
        fetch(this.options.componentStatusEndpoint)
            .then(response => response.json())
            .then(components => {
                this.updateComponentStatus(components);
            })
            .catch(error => {
                console.error('Error fetching component status:', error);
                this.componentsContainer.innerHTML = '<div class="component-status">Error loading components</div>';
            });
    }

    /**
     * Update component status display
     * @param {Array} components - Array of component status objects
     */
    updateComponentStatus(components) {
        if (!components || !components.length) {
            this.componentsContainer.innerHTML = '<div class="component-status">No components found</div>';
            return;
        }

        this.componentsContainer.innerHTML = '';

        components.forEach(component => {
            const isAvailable = component.status === 'operational';
            const statusItem = document.createElement('div');
            statusItem.className = `component-status ${isAvailable ? 'available' : ''}`;

            statusItem.innerHTML = `
                <div class="status-icon">${isAvailable ? '✓' : '✗'}</div>
                <div class="component-name">${component.name}</div>
            `;

            this.componentsContainer.appendChild(statusItem);
        });
    }

    /**
     * Start the refresh timer to periodically update status
     */
    startRefreshTimer() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
        }

        this.refreshTimer = setInterval(() => {
            this.fetchStatus();
            if (this.currentSource === 'real') {
                this.fetchComponentStatus();
            }
        }, this.options.refreshInterval);
    }

    /**
     * Stop the refresh timer
     */
    stopRefreshTimer() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    /**
     * Show a notification
     * @param {string} type - Notification type ('success', 'error', 'info', 'warning')
     * @param {string} message - Notification message
     */
    showNotification(type, message) {
        // Check if notification container exists, create if not
        let container = document.getElementById('notification-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notification-container';
            document.body.appendChild(container);
        }

        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            ${message}
            <button class="close-btn">&times;</button>
        `;

        // Add close button functionality
        const closeBtn = notification.querySelector('.close-btn');
        closeBtn.addEventListener('click', () => {
            notification.remove();
        });

        // Add to container
        container.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    /**
     * Refresh the page data after toggling data source
     */
    refreshPageData() {
        // Trigger refresh for all data panels
        document.querySelectorAll('.refresh-button').forEach(button => {
            // Simulate click on all refresh buttons
            button.click();
        });
    }

    /**
     * Destroy the component and clean up
     */
    destroy() {
        this.stopRefreshTimer();

        // Remove event listeners
        this.mockButton.removeEventListener('click', this.toggleDataSource);
        this.realButton.removeEventListener('click', this.toggleDataSource);

        // Remove component from DOM
        if (this.container) {
            this.container.remove();
        }
    }
}

// Initialize the component when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create the data source toggle component
    window.dataSourceToggle = new DataSourceToggle({
        onToggle: (source) => {
            console.log(`Data source switched to: ${source}`);
            // Additional callback actions can be added here
        }
    });
});
