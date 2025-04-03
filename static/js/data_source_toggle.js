/**
 * Data Source Toggle Functionality
 * 
 * This script handles the data source toggle button functionality,
 * allowing users to switch between mock and real data sources.
 */

document.addEventListener('DOMContentLoaded', function () {
    // Get the data source buttons
    const mockButton = document.querySelector('.data-source-button[data-source="mock"]');
    const realButton = document.querySelector('.data-source-button[data-source="real"]');

    if (!mockButton || !realButton) {
        console.error('Data source buttons not found');
        return;
    }

    // Add event listeners for button clicks
    mockButton.addEventListener('click', function () {
        if (!this.classList.contains('active')) {
            toggleDataSource('mock');
        }
    });

    realButton.addEventListener('click', function () {
        if (!this.classList.contains('active')) {
            toggleDataSource('real');
        }
    });

    /**
     * Toggle between mock and real data sources
     * 
     * @param {string} dataSource - The data source to use ('mock' or 'real')
     */
    function toggleDataSource(dataSource) {
        // Disable buttons while processing
        mockButton.disabled = true;
        realButton.disabled = true;

        // Show loading indicator
        const controlGroup = mockButton.closest('.control-group');
        if (controlGroup) {
            controlGroup.classList.add('loading');
        }

        // Make an API call to change the data source
        fetch('/api/system/data-source', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                source: dataSource
            }),
        })
            .then(response => response.json())
            .then(data => {
                // Re-enable the buttons
                mockButton.disabled = false;
                realButton.disabled = false;

                // Remove loading indicator
                if (controlGroup) {
                    controlGroup.classList.remove('loading');
                }

                // Check if the operation was successful
                if (data.success) {
                    // Update button states
                    if (data.data_source === 'mock') {
                        mockButton.classList.add('active');
                        realButton.classList.remove('active');
                    } else {
                        mockButton.classList.remove('active');
                        realButton.classList.add('active');
                    }

                    // Update status indicator
                    const statusIndicator = document.querySelector('.data-source-status .status-indicator');
                    const statusText = document.querySelector('.data-source-status .status-text');

                    if (statusIndicator && statusText) {
                        statusIndicator.className = 'status-indicator ' + data.data_source;
                        statusText.textContent = data.data_source.charAt(0).toUpperCase() + data.data_source.slice(1);
                    }

                    // Show success message
                    showNotification('Data source changed to ' + data.data_source, 'success');

                    // Refresh the dashboard data
                    refreshDashboardData();
                } else {
                    // Show error message with more details
                    showNotification('Error: ' + data.message, 'error');

                    // If real data is not available, disable the real button and show detailed error
                    if (data.message.includes('not available') && dataSource === 'real') {
                        realButton.classList.add('disabled');
                        realButton.setAttribute('title', data.message);

                        // Create a more detailed error message
                        const errorDetails = document.createElement('div');
                        errorDetails.className = 'error-details';
                        errorDetails.innerHTML = `
                            <div class="error-title">Real Data Connection Failed</div>
                            <div class="error-message">${data.message}</div>
                            <div class="error-help">
                                <p>Possible solutions:</p>
                                <ul>
                                    <li>Check if data files exist in the expected locations</li>
                                    <li>Verify that the data service is properly configured</li>
                                    <li>Ensure that the REAL_DATA_AVAILABLE flag is set to True</li>
                                    <li>Check the logs for more detailed error information</li>
                                </ul>
                            </div>
                        `;

                        // Show the error details in a modal or append to a designated area
                        const errorContainer = document.getElementById('errorContainer') || document.body;
                        errorContainer.appendChild(errorDetails);

                        // Auto-remove after 10 seconds
                        setTimeout(() => {
                            errorContainer.removeChild(errorDetails);
                        }, 10000);
                    }
                }
            })
            .catch(error => {
                console.error('Error changing data source:', error);

                // Re-enable the buttons
                mockButton.disabled = false;
                realButton.disabled = false;

                // Remove loading indicator
                if (controlGroup) {
                    controlGroup.classList.remove('loading');
                }

                // Show error message
                showNotification('Error changing data source', 'error');
            });
    }

    /**
     * Show a notification message
     * 
     * @param {string} message - The message to show
     * @param {string} type - The type of notification (success, error, warning, info)
     */
    function showNotification(message, type) {
        // Check if the notification system exists
        if (typeof showToast === 'function') {
            showToast(message, type);
        } else {
            // Create a simple toast notification
            const toast = document.createElement('div');
            toast.className = 'toast-notification ' + type;
            toast.textContent = message;

            document.body.appendChild(toast);

            // Show the toast
            setTimeout(() => {
                toast.classList.add('show');
            }, 10);

            // Hide the toast after 3 seconds
            setTimeout(() => {
                toast.classList.remove('show');
                setTimeout(() => {
                    document.body.removeChild(toast);
                }, 300);
            }, 3000);
        }
    }

    /**
     * Refresh the dashboard data
     */
    function refreshDashboardData() {
        // Check if the refresh function exists
        if (typeof refreshAllDashboardData === 'function') {
            refreshAllDashboardData();
        } else if (typeof fetchDashboardData === 'function') {
            fetchDashboardData();
        } else {
            // Reload the page as a last resort
            window.location.reload();
        }
    }
});