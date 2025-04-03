/**
 * Connection Editor
 * 
 * This script handles the data source connection editor functionality,
 * allowing users to configure individual data source connections.
 */

document.addEventListener('DOMContentLoaded', function () {
    // Initialize the connection editor
    const connectionEditor = new ConnectionEditor();
    connectionEditor.init();

    // Make it globally accessible
    window.connectionEditor = connectionEditor;
});

/**
 * Connection Editor Class
 */
class ConnectionEditor {
    constructor() {
        // Modal elements
        this.modal = null;
        this.connectionList = null;
        this.connectionDetailForm = null;
        this.saveButton = null;
        this.cancelButton = null;
        this.closeButton = null;

        // Current connections
        this.connections = {};

        // Currently editing connection
        this.currentConnectionId = null;

        // Flag to track if connections have changed
        this.connectionsChanged = false;
    }

    /**
     * Initialize the connection editor
     */
    init() {
        // Load the connection editor modal template
        this.loadConnectionEditorModal();
    }

    /**
     * Load the connection editor modal template
     */
    loadConnectionEditorModal() {
        // Check if the connection editor modal is already in the DOM
        if (document.getElementById('connection-editor-modal')) {
            this.initializeModalElements();
            return;
        }

        // Fetch the connection editor modal template
        fetch('/api/templates/connection_editor_modal.html')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load connection editor modal template');
                }
                return response.text();
            })
            .then(html => {
                // Insert the modal HTML into the DOM
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = html;
                document.body.appendChild(tempDiv.firstElementChild);

                // Initialize modal elements
                this.initializeModalElements();
            })
            .catch(error => {
                console.error('Error loading connection editor modal:', error);
                // Show error notification
                if (typeof showNotification === 'function') {
                    showNotification('Error loading connection editor', 'error');
                }
            });
    }

    /**
     * Initialize modal elements and event listeners
     */
    initializeModalElements() {
        // Get modal elements
        this.modal = document.getElementById('connection-editor-modal');
        this.connectionList = document.getElementById('connection-list');
        this.connectionDetailForm = document.getElementById('connection-detail-form');
        this.saveButton = document.getElementById('save-connections');
        this.cancelButton = document.getElementById('cancel-connections');
        this.closeButton = document.getElementById('close-connections');

        // Set up event listeners
        if (this.saveButton) {
            this.saveButton.addEventListener('click', this.saveConnections.bind(this));
        }

        if (this.cancelButton) {
            this.cancelButton.addEventListener('click', this.closeModal.bind(this));
        }

        if (this.closeButton) {
            this.closeButton.addEventListener('click', this.closeModal.bind(this));
        }

        // Set up connection detail form event listeners
        const backToListButton = document.querySelector('.back-to-list');
        if (backToListButton) {
            backToListButton.addEventListener('click', this.showConnectionList.bind(this));
        }

        const saveDetailButton = document.getElementById('save-detail');
        if (saveDetailButton) {
            saveDetailButton.addEventListener('click', this.saveConnectionDetail.bind(this));
        }

        const cancelDetailButton = document.getElementById('cancel-detail');
        if (cancelDetailButton) {
            cancelDetailButton.addEventListener('click', this.showConnectionList.bind(this));
        }
    }

    /**
     * Open the connection editor modal
     */
    openModal() {
        if (!this.modal) {
            console.error('Connection editor modal not initialized');
            return;
        }

        // Reset the connections changed flag
        this.connectionsChanged = false;

        // Show the modal
        this.modal.classList.add('show');

        // Load connections
        this.loadConnections();

        // Initialize Feather icons if available
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }

    /**
     * Close the connection editor modal
     */
    closeModal() {
        if (!this.modal) {
            return;
        }

        // Check if connections have changed
        if (this.connectionsChanged) {
            if (confirm('You have unsaved changes. Are you sure you want to close without saving?')) {
                this.modal.classList.remove('show');
            }
        } else {
            this.modal.classList.remove('show');
        }
    }

    /**
     * Load connections from the server
     */
    loadConnections() {
        // Show loading state
        if (this.connectionList) {
            this.connectionList.classList.add('loading');
        }

        // Fetch connections from the server
        fetch('/api/settings/data-source/connections')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load connections');
                }
                return response.json();
            })
            .then(data => {
                // Update connections
                this.connections = data.connections || {};

                // Render connection list
                this.renderConnectionList();

                // Hide loading state
                if (this.connectionList) {
                    this.connectionList.classList.remove('loading');
                }
            })
            .catch(error => {
                console.error('Error loading connections:', error);

                // Show error notification
                if (typeof showNotification === 'function') {
                    showNotification('Error loading connections', 'error');
                }

                // Hide loading state
                if (this.connectionList) {
                    this.connectionList.classList.remove('loading');
                }
            });
    }

    /**
     * Render the connection list
     */
    renderConnectionList() {
        if (!this.connectionList) {
            return;
        }

        // Clear the connection list
        this.connectionList.innerHTML = '';

        // Add connection items
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

            // Add event listeners
            const toggleInput = item.querySelector('.connection-toggle');
            if (toggleInput) {
                toggleInput.addEventListener('change', (event) => {
                    this.toggleConnection(id, event.target.checked);
                });
            }

            const editButton = item.querySelector('.edit-connection');
            if (editButton) {
                editButton.addEventListener('click', () => {
                    this.editConnection(id);
                });
            }
        });

        // Show the connection list panel
        this.showConnectionList();
    }

    /**
     * Show the connection list panel
     */
    showConnectionList() {
        if (this.connectionDetailForm) {
            this.connectionDetailForm.style.display = 'none';
        }

        const listPanel = document.getElementById('connection-list-panel');
        if (listPanel) {
            listPanel.style.display = 'block';
        }

        // Reset current connection ID
        this.currentConnectionId = null;
    }

    /**
     * Toggle a connection's enabled state
     * 
     * @param {string} id - The connection ID
     * @param {boolean} enabled - Whether the connection is enabled
     */
    toggleConnection(id, enabled) {
        if (!this.connections[id]) {
            return;
        }

        // Update the connection
        this.connections[id].enabled = enabled;

        // Update the UI
        const statusElement = document.querySelector(`.connection-item[data-id="${id}"] .connection-status`);
        if (statusElement) {
            statusElement.className = `connection-status ${enabled ? 'enabled' : 'disabled'}`;
            statusElement.textContent = enabled ? 'Enabled' : 'Disabled';
        }

        // Mark connections as changed
        this.connectionsChanged = true;
    }

    /**
     * Edit a connection
     * 
     * @param {string} id - The connection ID
     */
    editConnection(id) {
        if (!this.connections[id]) {
            return;
        }

        // Set current connection ID
        this.currentConnectionId = id;

        // Get the connection
        const connection = this.connections[id];

        // Update the form
        const nameInput = document.getElementById('connection-name');
        const enabledInput = document.getElementById('connection-enabled');
        const retryAttemptsInput = document.getElementById('retry-attempts');
        const timeoutSecondsInput = document.getElementById('timeout-seconds');
        const cacheDurationInput = document.getElementById('cache-duration');

        if (nameInput) {
            nameInput.value = this.formatConnectionName(id);
        }

        if (enabledInput) {
            enabledInput.checked = connection.enabled;
        }

        if (retryAttemptsInput) {
            retryAttemptsInput.value = connection.retry_attempts || 3;
        }

        if (timeoutSecondsInput) {
            timeoutSecondsInput.value = connection.timeout_seconds || 10;
        }

        if (cacheDurationInput) {
            cacheDurationInput.value = connection.cache_duration_seconds || 60;
        }

        // Update the title
        const titleElement = document.getElementById('connection-detail-title');
        if (titleElement) {
            titleElement.textContent = `Edit ${this.formatConnectionName(id)}`;
        }

        // Show the connection detail form
        this.showConnectionDetailForm();
    }

    /**
     * Show the connection detail form
     */
    showConnectionDetailForm() {
        const listPanel = document.getElementById('connection-list-panel');
        if (listPanel) {
            listPanel.style.display = 'none';
        }

        if (this.connectionDetailForm) {
            this.connectionDetailForm.style.display = 'block';
        }
    }

    /**
     * Save the current connection detail
     */
    saveConnectionDetail() {
        if (!this.currentConnectionId || !this.connections[this.currentConnectionId]) {
            return;
        }

        // Get form values
        const enabledInput = document.getElementById('connection-enabled');
        const retryAttemptsInput = document.getElementById('retry-attempts');
        const timeoutSecondsInput = document.getElementById('timeout-seconds');
        const cacheDurationInput = document.getElementById('cache-duration');

        // Validate inputs
        if (retryAttemptsInput) {
            const retryAttempts = parseInt(retryAttemptsInput.value);
            if (isNaN(retryAttempts) || retryAttempts < 1 || retryAttempts > 10) {
                if (typeof showNotification === 'function') {
                    showNotification('Retry attempts must be between 1 and 10', 'error');
                }
                return;
            }
        }

        if (timeoutSecondsInput) {
            const timeoutSeconds = parseInt(timeoutSecondsInput.value);
            if (isNaN(timeoutSeconds) || timeoutSeconds < 1 || timeoutSeconds > 60) {
                if (typeof showNotification === 'function') {
                    showNotification('Timeout must be between 1 and 60 seconds', 'error');
                }
                return;
            }
        }

        if (cacheDurationInput) {
            const cacheDuration = parseInt(cacheDurationInput.value);
            if (isNaN(cacheDuration) || cacheDuration < 0 || cacheDuration > 3600) {
                if (typeof showNotification === 'function') {
                    showNotification('Cache duration must be between 0 and 3600 seconds', 'error');
                }
                return;
            }
        }

        // Update the connection
        const connection = this.connections[this.currentConnectionId];

        if (enabledInput) {
            connection.enabled = enabledInput.checked;
        }

        if (retryAttemptsInput) {
            connection.retry_attempts = parseInt(retryAttemptsInput.value);
        }

        if (timeoutSecondsInput) {
            connection.timeout_seconds = parseInt(timeoutSecondsInput.value);
        }

        if (cacheDurationInput) {
            connection.cache_duration_seconds = parseInt(cacheDurationInput.value);
        }

        // Mark connections as changed
        this.connectionsChanged = true;

        // Show the connection list
        this.renderConnectionList();
    }

    /**
     * Save all connections to the server
     */
    saveConnections() {
        // Show loading state
        this.setLoadingState(true);

        // Save connections to the server
        fetch('/api/settings/data-source/connections', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                connections: this.connections
            }),
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to save connections');
                }
                return response.json();
            })
            .then(data => {
                // Reset connections changed flag
                this.connectionsChanged = false;

                // Show success notification
                if (typeof showNotification === 'function') {
                    showNotification('Connections saved successfully', 'success');
                }

                // Hide loading state
                this.setLoadingState(false);

                // Close the modal
                this.closeModal();
            })
            .catch(error => {
                console.error('Error saving connections:', error);

                // Show error notification
                if (typeof showNotification === 'function') {
                    showNotification('Error saving connections', 'error');
                }

                // Hide loading state
                this.setLoadingState(false);
            });
    }

    /**
     * Format a connection ID as a readable name
     * 
     * @param {string} id - The connection ID
     * @returns {string} - The formatted name
     */
    formatConnectionName(id) {
        return id.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    /**
     * Set loading state for the modal
     * 
     * @param {boolean} isLoading - Whether the modal is in loading state
     */
    setLoadingState(isLoading) {
        if (this.saveButton) {
            this.saveButton.disabled = isLoading;
        }

        if (this.cancelButton) {
            this.cancelButton.disabled = isLoading;
        }

        if (this.closeButton) {
            this.closeButton.disabled = isLoading;
        }

        // Add loading indicator if needed
        if (isLoading) {
            this.modal.classList.add('loading');
        } else {
            this.modal.classList.remove('loading');
        }
    }
}