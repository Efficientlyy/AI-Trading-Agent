/**
 * Status Monitoring Panel JavaScript
 * 
 * Handles the status monitoring panel functionality, including:
 * - Loading status data from the server
 * - Displaying source health and metrics
 * - Testing connections
 * - Viewing detailed source information
 * - Resetting source statistics
 */

// Status Monitoring Controller
class StatusMonitoringController {
    constructor() {
        // DOM Elements
        this.container = document.querySelector('.status-panel-container');
        this.sourceList = document.getElementById('source-list');
        this.sourceDetailPanel = document.getElementById('source-detail-panel');
        this.refreshButton = document.getElementById('refresh-status');
        this.testAllButton = document.getElementById('test-all-sources');
        this.resetAllButton = document.getElementById('reset-all-stats');

        // Templates
        this.sourceItemTemplate = document.getElementById('source-item-template');
        this.errorItemTemplate = document.getElementById('error-item-template');

        // State
        this.statusData = null;
        this.currentSourceId = null;
        this.updateInterval = null;
        this.charts = {};

        // Initialize
        this.init();
    }

    /**
     * Initialize the status monitoring panel
     */
    init() {
        // Add event listeners
        this.addEventListeners();

        // Load status data
        this.loadStatusData();

        // Start auto-refresh
        this.startAutoRefresh();
    }

    /**
     * Add event listeners
     */
    addEventListeners() {
        // Refresh button
        if (this.refreshButton) {
            this.refreshButton.addEventListener('click', () => {
                this.loadStatusData();
            });
        }

        // Test all button
        if (this.testAllButton) {
            this.testAllButton.addEventListener('click', () => {
                this.testAllSources();
            });
        }

        // Reset all button
        if (this.resetAllButton) {
            this.resetAllButton.addEventListener('click', () => {
                this.resetAllStats();
            });
        }

        // Source list event delegation
        if (this.sourceList) {
            this.sourceList.addEventListener('click', (event) => {
                const sourceItem = event.target.closest('.source-item');
                if (!sourceItem) return;

                const sourceId = sourceItem.dataset.id;

                if (event.target.classList.contains('view-details')) {
                    this.showSourceDetail(sourceId);
                } else if (event.target.classList.contains('test-source')) {
                    this.testSource(sourceId, event.target);
                }
            });
        }

        // Source detail panel event listeners
        if (this.sourceDetailPanel) {
            // Close detail panel
            const closeButton = this.sourceDetailPanel.querySelector('.close-detail');
            if (closeButton) {
                closeButton.addEventListener('click', () => {
                    this.hideSourceDetail();
                });
            }

            // Test source button
            const testButton = document.getElementById('test-source');
            if (testButton) {
                testButton.addEventListener('click', () => {
                    if (this.currentSourceId) {
                        this.testSource(this.currentSourceId, testButton);
                    }
                });
            }

            // Reset source stats button
            const resetButton = document.getElementById('reset-source-stats');
            if (resetButton) {
                resetButton.addEventListener('click', () => {
                    if (this.currentSourceId) {
                        this.resetSourceStats(this.currentSourceId);
                    }
                });
            }

            // Edit source button
            const editButton = document.getElementById('edit-source');
            if (editButton) {
                editButton.addEventListener('click', () => {
                    if (this.currentSourceId) {
                        this.editSource(this.currentSourceId);
                    }
                });
            }
        }
    }

    /**
     * Load status data from the server
     */
    loadStatusData() {
        // Show loading state
        this.showLoading();

        // Fetch status data
        fetch('/api/system/data-source-status')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load status data');
                }
                return response.json();
            })
            .then(data => {
                // Store status data
                this.statusData = data;

                // Update UI
                this.updateUI();

                // Hide loading state
                this.hideLoading();
            })
            .catch(error => {
                console.error('Error loading status data:', error);
                this.showError('Failed to load status data');

                // Hide loading state
                this.hideLoading();
            });
    }

    /**
     * Update the UI with status data
     */
    updateUI() {
        // Update summary
        this.updateSummary();

        // Update source list
        this.updateSourceList();

        // Update source detail if open
        if (this.currentSourceId && this.sourceDetailPanel.style.display !== 'none') {
            this.updateSourceDetail(this.currentSourceId);
        }
    }

    /**
     * Update the summary section
     */
    updateSummary() {
        if (!this.statusData) return;

        // Update last updated timestamp
        const lastUpdated = document.getElementById('last-updated');
        if (lastUpdated) {
            const timestamp = this.statusData.timestamp ? new Date(this.statusData.timestamp) : new Date();
            lastUpdated.textContent = `Last updated: ${timestamp.toLocaleTimeString()}`;
        }

        // Calculate summary metrics
        const sources = this.statusData.sources || {};
        const sourceCount = Object.keys(sources).length;
        const healthySources = Object.values(sources).filter(source => source.health === 'HEALTHY').length;

        // Update healthy sources count
        const healthySourcesElement = document.getElementById('healthy-sources');
        if (healthySourcesElement) {
            healthySourcesElement.textContent = `${healthySources}/${sourceCount}`;

            // Add color class based on health ratio
            healthySourcesElement.className = 'stat-value';
            if (healthySources === sourceCount) {
                healthySourcesElement.classList.add('healthy');
            } else if (healthySources >= sourceCount * 0.7) {
                healthySourcesElement.classList.add('degraded');
            } else {
                healthySourcesElement.classList.add('unhealthy');
            }
        }

        // Update system health
        const systemHealthElement = document.getElementById('system-health');
        if (systemHealthElement) {
            const systemHealth = this.statusData.system_health || 'UNKNOWN';
            systemHealthElement.textContent = systemHealth;

            // Add color class based on health
            systemHealthElement.className = 'stat-value';
            systemHealthElement.classList.add(systemHealth.toLowerCase());
        }

        // Calculate average response time
        let totalResponseTime = 0;
        let responseTimeCount = 0;

        Object.values(sources).forEach(source => {
            if (source.response_time) {
                const responseTime = this.parseResponseTime(source.response_time);
                if (!isNaN(responseTime)) {
                    totalResponseTime += responseTime;
                    responseTimeCount++;
                }
            }
        });

        const avgResponseTime = responseTimeCount > 0 ? totalResponseTime / responseTimeCount : 0;

        // Update average response time
        const avgResponseTimeElement = document.getElementById('avg-response-time');
        if (avgResponseTimeElement) {
            avgResponseTimeElement.textContent = responseTimeCount > 0 ? `${avgResponseTime.toFixed(2)}ms` : '--';
        }

        // Calculate error rate
        let totalErrors = 0;
        let totalRequests = 0;

        Object.values(sources).forEach(source => {
            if (source.error_count) {
                totalErrors += parseInt(source.error_count, 10) || 0;
            }
            if (source.request_count) {
                totalRequests += parseInt(source.request_count, 10) || 0;
            }
        });

        const errorRate = totalRequests > 0 ? (totalErrors / totalRequests) * 100 : 0;

        // Update error rate
        const errorRateElement = document.getElementById('error-rate');
        if (errorRateElement) {
            errorRateElement.textContent = totalRequests > 0 ? `${errorRate.toFixed(1)}%` : '--';

            // Add color class based on error rate
            errorRateElement.className = 'stat-value';
            if (errorRate < 1) {
                errorRateElement.classList.add('healthy');
            } else if (errorRate < 5) {
                errorRateElement.classList.add('degraded');
            } else {
                errorRateElement.classList.add('unhealthy');
            }
        }
    }

    /**
     * Update the source list
     */
    updateSourceList() {
        if (!this.statusData || !this.sourceList || !this.sourceItemTemplate) return;

        // Clear source list
        this.sourceList.innerHTML = '';

        // Get sources
        const sources = this.statusData.sources || {};

        // Check if there are any sources
        if (Object.keys(sources).length === 0) {
            const noData = document.createElement('div');
            noData.className = 'no-data';
            noData.textContent = 'No data sources available';
            this.sourceList.appendChild(noData);
            return;
        }

        // Add source items
        Object.entries(sources).forEach(([id, source]) => {
            // Clone template
            const item = this.sourceItemTemplate.content.cloneNode(true);
            const sourceItem = item.querySelector('.source-item');

            // Set source ID
            sourceItem.dataset.id = id;

            // Update source name
            const sourceName = item.querySelector('.source-name');
            if (sourceName) {
                sourceName.textContent = this.formatSourceName(id);
            }

            // Update source health
            const sourceHealth = item.querySelector('.source-health');
            if (sourceHealth) {
                const health = source.health || 'UNKNOWN';
                sourceHealth.textContent = health;
                sourceHealth.classList.add(health.toLowerCase());
            }

            // Update response time
            const responseTime = item.querySelector('.response-time');
            if (responseTime) {
                responseTime.textContent = source.response_time || '--';
            }

            // Update error count
            const errorCount = item.querySelector('.error-count');
            if (errorCount) {
                errorCount.textContent = source.error_count || '0';
            }

            // Add to source list
            this.sourceList.appendChild(item);
        });
    }

    /**
     * Show source detail panel
     * @param {string} sourceId - The source ID
     */
    showSourceDetail(sourceId) {
        if (!this.statusData || !this.sourceDetailPanel) return;

        // Get source data
        const sources = this.statusData.sources || {};
        const source = sources[sourceId];

        if (!source) {
            this.showError(`Source ${sourceId} not found`);
            return;
        }

        // Store current source ID
        this.currentSourceId = sourceId;

        // Update source detail
        this.updateSourceDetail(sourceId);

        // Show source detail panel
        this.sourceDetailPanel.style.display = 'block';
    }

    /**
     * Update source detail panel
     * @param {string} sourceId - The source ID
     */
    updateSourceDetail(sourceId) {
        if (!this.statusData || !this.sourceDetailPanel) return;

        // Get source data
        const sources = this.statusData.sources || {};
        const source = sources[sourceId];

        if (!source) {
            this.showError(`Source ${sourceId} not found`);
            return;
        }

        // Update source name
        const sourceName = document.getElementById('detail-source-name');
        if (sourceName) {
            sourceName.textContent = this.formatSourceName(sourceId);
        }

        // Update source health
        const sourceHealth = document.getElementById('detail-source-health');
        if (sourceHealth) {
            const health = source.health || 'UNKNOWN';
            sourceHealth.textContent = health;
            sourceHealth.className = 'health-badge ' + health.toLowerCase();
        }

        // Update metrics
        document.getElementById('detail-uptime').textContent = source.uptime || '--';
        document.getElementById('detail-response-time').textContent = source.response_time || '--';
        document.getElementById('detail-error-rate').textContent = source.error_rate || '--';
        document.getElementById('detail-last-success').textContent = source.last_success || '--';

        // Update connection details
        document.getElementById('detail-type').textContent = source.type || '--';
        document.getElementById('detail-endpoint').textContent = source.endpoint || '--';
        document.getElementById('detail-retry-attempts').textContent = source.retry_attempts || '--';
        document.getElementById('detail-timeout').textContent = source.timeout ? `${source.timeout}s` : '--';
        document.getElementById('detail-cache-duration').textContent = source.cache_duration ? `${source.cache_duration}s` : '--';

        // Update error history
        this.updateErrorHistory(sourceId, source.error_history);

        // Update performance chart
        this.updatePerformanceChart(sourceId, source.performance_history);
    }

    /**
     * Hide source detail panel
     */
    hideSourceDetail() {
        if (this.sourceDetailPanel) {
            this.sourceDetailPanel.style.display = 'none';
            this.currentSourceId = null;
        }
    }

    /**
     * Update error history
     * @param {string} sourceId - The source ID
     * @param {Array} errorHistory - The error history
     */
    updateErrorHistory(sourceId, errorHistory) {
        const errorHistoryContainer = document.getElementById('error-history');
        if (!errorHistoryContainer || !this.errorItemTemplate) return;

        // Clear error history
        errorHistoryContainer.innerHTML = '';

        // Check if there are any errors
        if (!errorHistory || errorHistory.length === 0) {
            const noData = document.createElement('div');
            noData.className = 'no-data';
            noData.textContent = 'No errors recorded';
            errorHistoryContainer.appendChild(noData);
            return;
        }

        // Add error items
        errorHistory.forEach(error => {
            // Clone template
            const item = this.errorItemTemplate.content.cloneNode(true);

            // Update error time
            const errorTime = item.querySelector('.error-time');
            if (errorTime) {
                errorTime.textContent = error.timestamp || '--';
            }

            // Update error message
            const errorMessage = item.querySelector('.error-message');
            if (errorMessage) {
                errorMessage.textContent = error.message || '--';
            }

            // Add to error history
            errorHistoryContainer.appendChild(item);
        });
    }

    /**
     * Update performance chart
     * @param {string} sourceId - The source ID
     * @param {Array} performanceHistory - The performance history
     */
    updatePerformanceChart(sourceId, performanceHistory) {
        const chartContainer = document.getElementById('performance-chart');
        if (!chartContainer) return;

        // Check if there is any performance history
        if (!performanceHistory || performanceHistory.length === 0) {
            chartContainer.innerHTML = '<div class="no-data">No performance data available</div>';
            return;
        }

        // Prepare chart data
        const timestamps = [];
        const responseTimes = [];

        performanceHistory.forEach(entry => {
            timestamps.push(entry.timestamp);
            responseTimes.push(this.parseResponseTime(entry.response_time));
        });

        // Create chart
        if (window.Plotly) {
            const data = [{
                x: timestamps,
                y: responseTimes,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Response Time (ms)',
                line: {
                    color: '#4CAF50',
                    width: 2
                },
                marker: {
                    color: '#4CAF50',
                    size: 6
                }
            }];

            const layout = {
                title: 'Response Time Trend',
                xaxis: {
                    title: 'Time',
                    showgrid: false
                },
                yaxis: {
                    title: 'Response Time (ms)',
                    showgrid: true
                },
                margin: {
                    l: 50,
                    r: 20,
                    t: 40,
                    b: 50
                },
                height: 250,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {
                    color: '#333'
                }
            };

            const config = {
                responsive: true,
                displayModeBar: false
            };

            // Check if chart already exists
            if (this.charts[sourceId]) {
                // Update existing chart
                Plotly.react(chartContainer, data, layout, config);
            } else {
                // Create new chart
                Plotly.newPlot(chartContainer, data, layout, config);
                this.charts[sourceId] = true;
            }
        } else {
            // Fallback if Plotly is not available
            chartContainer.innerHTML = '<div class="chart-fallback">Chart library not available</div>';
        }
    }

    /**
     * Test a source connection
     * @param {string} sourceId - The source ID
     * @param {HTMLElement} button - The button element
     */
    testSource(sourceId, button) {
        // Show loading state
        if (button) {
            button.disabled = true;
            button.classList.add('loading');
        }

        // Test connection
        fetch(`/api/system/test-connection?source=${sourceId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to test connection');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    this.showSuccess(`Connection to ${this.formatSourceName(sourceId)} successful`);
                } else {
                    this.showError(`Connection to ${this.formatSourceName(sourceId)} failed: ${data.message}`);
                }

                // Reload status data
                this.loadStatusData();

                // Reset button state
                if (button) {
                    button.disabled = false;
                    button.classList.remove('loading');
                }
            })
            .catch(error => {
                console.error('Error testing connection:', error);
                this.showError(`Failed to test connection to ${this.formatSourceName(sourceId)}`);

                // Reset button state
                if (button) {
                    button.disabled = false;
                    button.classList.remove('loading');
                }
            });
    }

    /**
     * Test all sources
     */
    testAllSources() {
        if (!this.statusData) return;

        // Show loading state
        if (this.testAllButton) {
            this.testAllButton.disabled = true;
            this.testAllButton.classList.add('loading');
        }

        // Get sources
        const sources = this.statusData.sources || {};
        const sourceIds = Object.keys(sources);

        // Check if there are any sources
        if (sourceIds.length === 0) {
            this.showWarning('No sources to test');

            // Reset button state
            if (this.testAllButton) {
                this.testAllButton.disabled = false;
                this.testAllButton.classList.remove('loading');
            }

            return;
        }

        // Test each source
        let completed = 0;
        let successful = 0;
        let failed = 0;

        sourceIds.forEach(sourceId => {
            fetch(`/api/system/test-connection?source=${sourceId}`)
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
                    if (completed === sourceIds.length) {
                        this.showTestResults(successful, failed);
                    }
                })
                .catch(error => {
                    console.error(`Error testing connection ${sourceId}:`, error);

                    completed++;
                    failed++;

                    // Check if all tests are complete
                    if (completed === sourceIds.length) {
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
        if (this.testAllButton) {
            this.testAllButton.disabled = false;
            this.testAllButton.classList.remove('loading');
        }

        // Show results
        if (failed === 0) {
            this.showSuccess(`All ${successful} connections tested successfully`);
        } else if (successful === 0) {
            this.showError(`All ${failed} connections failed`);
        } else {
            this.showWarning(`${successful} connections successful, ${failed} connections failed`);
        }

        // Reload status data
        this.loadStatusData();
    }

    /**
     * Reset source statistics
     * @param {string} sourceId - The source ID
     */
    resetSourceStats(sourceId) {
        // Show confirmation dialog
        if (!confirm(`Are you sure you want to reset statistics for ${this.formatSourceName(sourceId)}?`)) {
            return;
        }

        // Reset statistics
        fetch(`/api/system/reset-source-stats?source=${sourceId}`, {
            method: 'POST'
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to reset statistics');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    this.showSuccess(`Statistics for ${this.formatSourceName(sourceId)} reset successfully`);

                    // Reload status data
                    this.loadStatusData();

                    // Hide source detail panel
                    this.hideSourceDetail();
                } else {
                    this.showError(`Failed to reset statistics for ${this.formatSourceName(sourceId)}: ${data.message}`);
                }
            })
            .catch(error => {
                console.error('Error resetting statistics:', error);
                this.showError(`Failed to reset statistics for ${this.formatSourceName(sourceId)}`);
            });
    }

    /**
     * Reset all source statistics
     */
    resetAllStats() {
        // Show confirmation dialog
        if (!confirm('Are you sure you want to reset statistics for all sources?')) {
            return;
        }

        // Show loading state
        if (this.resetAllButton) {
            this.resetAllButton.disabled = true;
            this.resetAllButton.classList.add('loading');
        }

        // Reset all statistics
        fetch('/api/system/reset-source-stats', {
            method: 'POST'
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to reset statistics');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    this.showSuccess('Statistics for all sources reset successfully');

                    // Reload status data
                    this.loadStatusData();
                } else {
                    this.showError(`Failed to reset statistics: ${data.message}`);
                }

                // Reset button state
                if (this.resetAllButton) {
                    this.resetAllButton.disabled = false;
                    this.resetAllButton.classList.remove('loading');
                }
            })
            .catch(error => {
                console.error('Error resetting statistics:', error);
                this.showError('Failed to reset statistics');

                // Reset button state
                if (this.resetAllButton) {
                    this.resetAllButton.disabled = false;
                    this.resetAllButton.classList.remove('loading');
                }
            });
    }

    /**
     * Edit source configuration
     * @param {string} sourceId - The source ID
     */
    editSource(sourceId) {
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
                            window.connectionEditor.openModal(sourceId);
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
            window.connectionEditor.openModal(sourceId);
        }
    }

    /**
     * Start auto-refresh
     */
    startAutoRefresh() {
        // Clear existing interval
        this.stopAutoRefresh();

        // Start new interval
        this.updateInterval = setInterval(() => {
            this.loadStatusData();
        }, 30000); // Refresh every 30 seconds
    }

    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    /**
     * Show loading state
     */
    showLoading() {
        if (this.container) {
            this.container.classList.add('loading');
        }
    }

    /**
     * Hide loading state
     */
    hideLoading() {
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
     * Format source name
     * @param {string} sourceId - The source ID
     * @returns {string} The formatted source name
     */
    formatSourceName(sourceId) {
        return sourceId.split('_').map(word =>
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    /**
     * Parse response time
     * @param {string} responseTime - The response time string
     * @returns {number} The response time in milliseconds
     */
    parseResponseTime(responseTime) {
        if (!responseTime) return 0;

        // Remove units and convert to number
        const value = parseFloat(responseTime.replace(/[^\d.]/g, ''));

        // Convert to milliseconds if needed
        if (responseTime.includes('s')) {
            return value * 1000;
        }

        return value;
    }
}

// Initialize status monitoring when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global status monitoring instance
    window.statusMonitoring = new StatusMonitoringController();
});