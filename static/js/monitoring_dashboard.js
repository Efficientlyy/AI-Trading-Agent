/**
 * Monitoring Dashboard JavaScript
 * 
 * Handles the monitoring dashboard functionality, including:
 * - Loading monitoring data from the server
 * - Updating the UI based on the loaded data
 * - Auto-refreshing the data
 * - Rendering charts and visualizations
 */

// Monitoring Dashboard Controller
class MonitoringDashboardController {
    constructor() {
        // DOM Elements
        this.refreshButton = document.getElementById('refresh-monitoring');
        this.autoRefreshCheckbox = document.getElementById('auto-refresh-monitoring');

        // Charts
        this.rateLimitChart = null;
        this.errorDistributionChart = null;
        this.cachePerformanceChart = null;
        this.requestPerformanceChart = null;

        // Data
        this.monitoringData = null;
        this.alertHistory = [];
        this.rateLimitHistory = [];
        this.errorHistory = [];

        // Auto-refresh
        this.autoRefreshInterval = null;
        this.autoRefreshTime = 30000; // 30 seconds

        // Initialize
        this.initialize();
    }

    /**
     * Initialize the monitoring dashboard
     */
    initialize() {
        // Initialize event listeners
        this.initEventListeners();

        // Initialize charts
        this.initCharts();

        // Load initial data
        this.loadMonitoringData();

        // Start auto-refresh if enabled
        if (this.autoRefreshCheckbox.checked) {
            this.startAutoRefresh();
        }
    }

    /**
     * Initialize event listeners
     */
    initEventListeners() {
        // Refresh button
        if (this.refreshButton) {
            this.refreshButton.addEventListener('click', () => {
                this.loadMonitoringData();
            });
        }

        // Auto-refresh checkbox
        if (this.autoRefreshCheckbox) {
            this.autoRefreshCheckbox.addEventListener('change', () => {
                if (this.autoRefreshCheckbox.checked) {
                    this.startAutoRefresh();
                } else {
                    this.stopAutoRefresh();
                }
            });
        }

        // Tab change events
        const tabs = document.querySelectorAll('#monitoringTabs button[data-bs-toggle="tab"]');
        tabs.forEach(tab => {
            tab.addEventListener('shown.bs.tab', (event) => {
                // Refresh charts when tab is shown
                this.refreshCharts();
            });
        });
    }

    /**
     * Initialize charts
     */
    initCharts() {
        // Rate Limit Chart
        const rateLimitChartEl = document.getElementById('rate-limit-chart');
        if (rateLimitChartEl) {
            this.rateLimitChart = new Chart(rateLimitChartEl, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Usage Percentage',
                        data: [],
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Usage (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        }
                    }
                }
            });
        }

        // Error Distribution Chart
        const errorDistributionChartEl = document.getElementById('error-distribution-chart');
        if (errorDistributionChartEl) {
            this.errorDistributionChart = new Chart(errorDistributionChartEl, {
                type: 'pie',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.7)',
                            'rgba(54, 162, 235, 0.7)',
                            'rgba(255, 206, 86, 0.7)',
                            'rgba(75, 192, 192, 0.7)',
                            'rgba(153, 102, 255, 0.7)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'right'
                        }
                    }
                }
            });
        }

        // Cache Performance Chart
        const cachePerformanceChartEl = document.getElementById('cache-performance-chart');
        if (cachePerformanceChartEl) {
            this.cachePerformanceChart = new Chart(cachePerformanceChartEl, {
                type: 'bar',
                data: {
                    labels: ['Memory Hits', 'Disk Hits', 'Misses'],
                    datasets: [{
                        label: 'Cache Performance',
                        data: [0, 0, 0],
                        backgroundColor: [
                            'rgba(75, 192, 192, 0.7)',
                            'rgba(54, 162, 235, 0.7)',
                            'rgba(255, 99, 132, 0.7)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Count'
                            }
                        }
                    }
                }
            });
        }

        // Request Performance Chart
        const requestPerformanceChartEl = document.getElementById('request-performance-chart');
        if (requestPerformanceChartEl) {
            this.requestPerformanceChart = new Chart(requestPerformanceChartEl, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Average Response Time (ms)',
                        data: [],
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Time (ms)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Endpoint'
                            }
                        }
                    }
                }
            });
        }
    }

    /**
     * Refresh charts with current data
     */
    refreshCharts() {
        if (!this.monitoringData) return;

        // Update Rate Limit Chart
        if (this.rateLimitChart && this.rateLimitHistory.length > 0) {
            const labels = this.rateLimitHistory.map(item => {
                const date = new Date(item.timestamp);
                return date.toLocaleTimeString();
            });

            const data = this.rateLimitHistory.map(item => item.usage_percentage * 100);

            this.rateLimitChart.data.labels = labels;
            this.rateLimitChart.data.datasets[0].data = data;
            this.rateLimitChart.update();
        }

        // Update Error Distribution Chart
        if (this.errorDistributionChart && this.monitoringData.errors) {
            const errorStats = this.monitoringData.errors.error_stats || {};
            const errorTypes = [];
            const errorCounts = [];

            for (const source in errorStats) {
                const sourceErrors = errorStats[source];
                if (sourceErrors.error_types) {
                    for (const type in sourceErrors.error_types) {
                        const count = sourceErrors.error_types[type];
                        errorTypes.push(`${source}: ${type}`);
                        errorCounts.push(count);
                    }
                }
            }

            this.errorDistributionChart.data.labels = errorTypes;
            this.errorDistributionChart.data.datasets[0].data = errorCounts;
            this.errorDistributionChart.update();
        }

        // Update Cache Performance Chart
        if (this.cachePerformanceChart && this.monitoringData.cache) {
            const cacheStats = this.monitoringData.cache.stats || {};
            const data = [
                cacheStats.memory_hits || 0,
                cacheStats.disk_hits || 0,
                cacheStats.misses || 0
            ];

            this.cachePerformanceChart.data.datasets[0].data = data;
            this.cachePerformanceChart.update();
        }

        // Update Request Performance Chart
        if (this.requestPerformanceChart && this.monitoringData.performance) {
            const requestTiming = this.monitoringData.performance.request_timing || {};
            const endpoints = [];
            const avgTimes = [];

            for (const endpoint in requestTiming) {
                endpoints.push(endpoint);
                avgTimes.push(requestTiming[endpoint].avg_time || 0);
            }

            this.requestPerformanceChart.data.labels = endpoints;
            this.requestPerformanceChart.data.datasets[0].data = avgTimes;
            this.requestPerformanceChart.update();
        }
    }

    /**
     * Start auto-refresh
     */
    startAutoRefresh() {
        this.stopAutoRefresh();

        this.autoRefreshInterval = setInterval(() => {
            this.loadMonitoringData();
        }, this.autoRefreshTime);

        console.log(`Auto-refresh started (${this.autoRefreshTime / 1000}s interval)`);
    }

    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;

            console.log('Auto-refresh stopped');
        }
    }

    /**
     * Load monitoring data from the server
     */
    loadMonitoringData() {
        console.log('Loading monitoring data...');

        // Show loading state
        this.showLoading();

        // Fetch monitoring data
        fetch('/api/monitoring/data')
            .then(response => response.json())
            .then(data => {
                this.monitoringData = data;

                // Update UI
                this.updateUI();

                // Update history
                this.updateHistory(data);

                // Hide loading state
                this.hideLoading();

                console.log('Monitoring data loaded');
            })
            .catch(error => {
                console.error('Error loading monitoring data:', error);

                // Hide loading state
                this.hideLoading();

                // Show error
                this.showError('Failed to load monitoring data');
            });
    }

    /**
     * Update UI with monitoring data
     */
    updateUI() {
        if (!this.monitoringData) return;

        // Update overview
        this.updateOverview();

        // Update rate limits
        this.updateRateLimits();

        // Update errors
        this.updateErrors();

        // Update cache
        this.updateCache();

        // Update performance
        this.updatePerformance();

        // Refresh charts
        this.refreshCharts();
    }

    /**
     * Update overview section
     */
    updateOverview() {
        // Rate Limit Progress
        const rateLimitProgress = document.getElementById('rate-limit-progress');
        const rateLimitText = document.getElementById('rate-limit-text');

        if (rateLimitProgress && rateLimitText && this.monitoringData.rate_limits) {
            const rateLimitStatus = this.monitoringData.rate_limits.status || {};
            const bitvavo = rateLimitStatus.bitvavo || {};

            const usagePercentage = bitvavo.usage_percentage || 0;
            const remaining = bitvavo.remaining || 0;
            const limit = bitvavo.limit || 0;

            // Update progress circle
            rateLimitProgress.setAttribute('data-value', Math.round(usagePercentage * 100));
            rateLimitProgress.querySelector('.progress-circle-value').textContent = `${Math.round(usagePercentage * 100)}%`;

            // Update text
            rateLimitText.textContent = `${remaining}/${limit} requests remaining`;

            // Update color based on usage
            if (usagePercentage > 0.8) {
                rateLimitProgress.classList.add('danger');
                rateLimitProgress.classList.remove('warning', 'success');
            } else if (usagePercentage > 0.5) {
                rateLimitProgress.classList.add('warning');
                rateLimitProgress.classList.remove('danger', 'success');
            } else {
                rateLimitProgress.classList.add('success');
                rateLimitProgress.classList.remove('danger', 'warning');
            }
        }

        // Cache Hit Progress
        const cacheHitProgress = document.getElementById('cache-hit-progress');
        const cacheHitText = document.getElementById('cache-hit-text');

        if (cacheHitProgress && cacheHitText && this.monitoringData.cache) {
            const cacheStats = this.monitoringData.cache.stats || {};

            const hitRatio = cacheStats.hit_ratio || 0;
            const totalHits = cacheStats.total_hits || 0;
            const totalRequests = totalHits + (cacheStats.misses || 0);

            // Update progress circle
            cacheHitProgress.setAttribute('data-value', Math.round(hitRatio * 100));
            cacheHitProgress.querySelector('.progress-circle-value').textContent = `${Math.round(hitRatio * 100)}%`;

            // Update text
            cacheHitText.textContent = `${totalHits}/${totalRequests} cache hits`;

            // Update color based on hit ratio
            if (hitRatio > 0.8) {
                cacheHitProgress.classList.add('success');
                cacheHitProgress.classList.remove('warning', 'danger');
            } else if (hitRatio > 0.5) {
                cacheHitProgress.classList.add('warning');
                cacheHitProgress.classList.remove('danger', 'success');
            } else {
                cacheHitProgress.classList.add('danger');
                cacheHitProgress.classList.remove('warning', 'success');
            }
        }

        // Error Rate Progress
        const errorRateProgress = document.getElementById('error-rate-progress');
        const errorRateText = document.getElementById('error-rate-text');

        if (errorRateProgress && errorRateText && this.monitoringData.errors) {
            const errorStats = this.monitoringData.errors.error_stats || {};

            let totalErrors = 0;
            for (const source in errorStats) {
                totalErrors += errorStats[source].total_errors || 0;
            }

            // Calculate error rate (assuming 100 requests per hour is normal)
            const errorRate = Math.min(totalErrors / 100, 1);

            // Update progress circle
            errorRateProgress.setAttribute('data-value', Math.round(errorRate * 100));
            errorRateProgress.querySelector('.progress-circle-value').textContent = `${Math.round(errorRate * 100)}%`;

            // Update text
            errorRateText.textContent = `${totalErrors} errors in last hour`;

            // Update color based on error rate
            if (errorRate > 0.1) {
                errorRateProgress.classList.add('danger');
                errorRateProgress.classList.remove('warning', 'success');
            } else if (errorRate > 0.05) {
                errorRateProgress.classList.add('warning');
                errorRateProgress.classList.remove('danger', 'success');
            } else {
                errorRateProgress.classList.add('success');
                errorRateProgress.classList.remove('danger', 'warning');
            }
        }

        // Circuit Breaker Indicator
        const circuitBreakerIndicator = document.getElementById('circuit-breaker-indicator');
        const circuitBreakerText = document.getElementById('circuit-breaker-text');

        if (circuitBreakerIndicator && circuitBreakerText && this.monitoringData.errors) {
            const circuitBreakers = this.monitoringData.errors.circuit_breakers || {};
            const circuitBreakerCount = Object.keys(circuitBreakers).length;

            if (circuitBreakerCount > 0) {
                // Update indicator
                circuitBreakerIndicator.innerHTML = '<i class="fas fa-exclamation-triangle text-danger fa-3x"></i>';

                // Update text
                circuitBreakerText.textContent = `${circuitBreakerCount} circuit breaker(s) open`;
                circuitBreakerText.classList.add('text-danger');
                circuitBreakerText.classList.remove('text-success');
            } else {
                // Update indicator
                circuitBreakerIndicator.innerHTML = '<i class="fas fa-check-circle text-success fa-3x"></i>';

                // Update text
                circuitBreakerText.textContent = 'All systems operational';
                circuitBreakerText.classList.add('text-success');
                circuitBreakerText.classList.remove('text-danger');
            }
        }

        // Recent Alerts
        const recentAlertsBody = document.getElementById('recent-alerts-body');

        if (recentAlertsBody && this.alertHistory.length > 0) {
            let html = '';

            for (const alert of this.alertHistory.slice(0, 5)) {
                const date = new Date(alert.timestamp);
                const time = date.toLocaleTimeString();

                html += `
                    <tr>
                        <td>${time}</td>
                        <td>${alert.type}</td>
                        <td>${alert.source}</td>
                        <td>${alert.message}</td>
                    </tr>
                `;
            }

            recentAlertsBody.innerHTML = html;
        } else if (recentAlertsBody) {
            recentAlertsBody.innerHTML = '<tr><td colspan="4" class="text-center">No recent alerts</td></tr>';
        }

        // System Status
        const systemStatusBody = document.getElementById('system-status-body');

        if (systemStatusBody && this.monitoringData.system) {
            const systemStatus = this.monitoringData.system.status || {};

            let html = '';

            for (const component in systemStatus) {
                const status = systemStatus[component];
                const statusClass = status.operational ? 'bg-success' : 'bg-danger';
                const statusText = status.operational ? 'Operational' : 'Down';
                const lastUpdated = new Date(status.last_updated).toLocaleTimeString();

                html += `
                    <tr>
                        <td>${component}</td>
                        <td><span class="badge ${statusClass}">${statusText}</span></td>
                        <td>${lastUpdated}</td>
                    </tr>
                `;
            }

            systemStatusBody.innerHTML = html;
        }
    }

    /**
     * Update rate limits section
     */
    updateRateLimits() {
        // Rate Limits Table
        const rateLimitsBody = document.getElementById('rate-limits-body');

        if (rateLimitsBody && this.monitoringData.rate_limits) {
            const rateLimitStatus = this.monitoringData.rate_limits.status || {};

            let html = '';

            for (const connector in rateLimitStatus) {
                const status = rateLimitStatus[connector];
                const usagePercentage = status.usage_percentage || 0;
                const usageClass = usagePercentage > 0.8 ? 'text-danger' : (usagePercentage > 0.5 ? 'text-warning' : 'text-success');
                const resetTime = status.reset_time ? new Date(status.reset_time).toLocaleTimeString() : 'N/A';

                html += `
                    <tr>
                        <td>${connector}</td>
                        <td>${status.limit || 0}</td>
                        <td>${status.remaining || 0}</td>
                        <td class="${usageClass}">${Math.round(usagePercentage * 100)}%</td>
                        <td>${resetTime}</td>
                    </tr>
                `;
            }

            if (html) {
                rateLimitsBody.innerHTML = html;
            } else {
                rateLimitsBody.innerHTML = '<tr><td colspan="5" class="text-center">No rate limit data available</td></tr>';
            }
        }

        // Rate Limit History Table
        const rateLimitHistoryBody = document.getElementById('rate-limit-history-body');

        if (rateLimitHistoryBody && this.rateLimitHistory.length > 0) {
            let html = '';

            for (const item of this.rateLimitHistory.slice(0, 10)) {
                const date = new Date(item.timestamp);
                const time = date.toLocaleTimeString();
                const usagePercentage = item.usage_percentage || 0;
                const usageClass = usagePercentage > 0.8 ? 'text-danger' : (usagePercentage > 0.5 ? 'text-warning' : 'text-success');
                const resetTime = item.reset_time ? new Date(item.reset_time).toLocaleTimeString() : 'N/A';

                html += `
                    <tr>
                        <td>${time}</td>
                        <td>${item.connector}</td>
                        <td>${item.limit || 0}</td>
                        <td>${item.remaining || 0}</td>
                        <td class="${usageClass}">${Math.round(usagePercentage * 100)}%</td>
                        <td>${resetTime}</td>
                    </tr>
                `;
            }

            rateLimitHistoryBody.innerHTML = html;
        } else if (rateLimitHistoryBody) {
            rateLimitHistoryBody.innerHTML = '<tr><td colspan="6" class="text-center">No rate limit history available</td></tr>';
        }
    }

    /**
     * Update errors section
     */
    updateErrors() {
        // Circuit Breakers Table
        const circuitBreakersBody = document.getElementById('circuit-breakers-body');

        if (circuitBreakersBody && this.monitoringData.errors) {
            const circuitBreakers = this.monitoringData.errors.circuit_breakers || {};

            let html = '';

            for (const key in circuitBreakers) {
                const breaker = circuitBreakers[key];
                const trippedAt = new Date(breaker.tripped_at).toLocaleTimeString();
                const resetTime = new Date(breaker.reset_time * 1000).toLocaleTimeString();

                html += `
                    <tr>
                        <td>${breaker.source}</td>
                        <td>${breaker.error_type}</td>
                        <td>${trippedAt}</td>
                        <td>${resetTime}</td>
                        <td><span class="badge bg-danger">Open</span></td>
                    </tr>
                `;
            }

            if (html) {
                circuitBreakersBody.innerHTML = html;
            } else {
                circuitBreakersBody.innerHTML = '<tr><td colspan="5" class="text-center">No circuit breakers active</td></tr>';
            }
        }

        // Recent Errors Table
        const recentErrorsBody = document.getElementById('recent-errors-body');

        if (recentErrorsBody && this.errorHistory.length > 0) {
            let html = '';

            for (const error of this.errorHistory.slice(0, 10)) {
                const date = new Date(error.timestamp);
                const time = date.toLocaleTimeString();

                html += `
                    <tr>
                        <td>${time}</td>
                        <td>${error.source}</td>
                        <td>${error.type}</td>
                        <td>${error.message}</td>
                    </tr>
                `;
            }

            recentErrorsBody.innerHTML = html;
        } else if (recentErrorsBody) {
            recentErrorsBody.innerHTML = '<tr><td colspan="4" class="text-center">No recent errors</td></tr>';
        }
    }

    /**
     * Update cache section
     */
    updateCache() {
        // Cache Stats Table
        const cacheStatsBody = document.getElementById('cache-stats-body');

        if (cacheStatsBody && this.monitoringData.cache) {
            const cacheStats = this.monitoringData.cache.stats || {};

            const rows = [
                { label: 'Memory Entries', value: cacheStats.memory_entries || 0 },
                { label: 'Memory Hits', value: cacheStats.memory_hits || 0 },
                { label: 'Disk Hits', value: cacheStats.disk_hits || 0 },
                { label: 'Total Hits', value: cacheStats.total_hits || 0 },
                { label: 'Misses', value: cacheStats.misses || 0 },
                { label: 'Hit Ratio', value: `${Math.round((cacheStats.hit_ratio || 0) * 100)}%` },
                { label: 'Sets', value: cacheStats.sets || 0 },
                { label: 'Invalidations', value: cacheStats.invalidations || 0 },
                { label: 'Prunes', value: cacheStats.prunes || 0 }
            ];

            let html = '';

            for (const row of rows) {
                html += `
                    <tr>
                        <td>${row.label}</td>
                        <td>${row.value}</td>
                    </tr>
                `;
            }

            cacheStatsBody.innerHTML = html;
        }

        // Cache Entries Table
        const cacheEntriesBody = document.getElementById('cache-entries-body');

        if (cacheEntriesBody && this.monitoringData.cache) {
            const cacheEntries = this.monitoringData.cache.entries || {};

            let html = '';

            for (const endpoint in cacheEntries) {
                const entry = cacheEntries[endpoint];
                const hitRatio = entry.hits / (entry.hits + entry.misses) || 0;
                const hitRatioClass = hitRatio > 0.8 ? 'text-success' : (hitRatio > 0.5 ? 'text-warning' : 'text-danger');
                const lastUpdated = new Date(entry.last_updated).toLocaleTimeString();

                html += `
                    <tr>
                        <td>${endpoint}</td>
                        <td>${entry.hits || 0}</td>
                        <td>${entry.misses || 0}</td>
                        <td class="${hitRatioClass}">${Math.round(hitRatio * 100)}%</td>
                        <td>${entry.ttl || 0}s</td>
                        <td>${lastUpdated}</td>
                    </tr>
                `;
            }

            if (html) {
                cacheEntriesBody.innerHTML = html;
            } else {
                cacheEntriesBody.innerHTML = '<tr><td colspan="6" class="text-center">No cache entries available</td></tr>';
            }
        }
    }

    /**
     * Update performance section
     */
    updatePerformance() {
        // Connection Pool Stats Table
        const connectionPoolStatsBody = document.getElementById('connection-pool-stats-body');

        if (connectionPoolStatsBody && this.monitoringData.performance) {
            const poolStats = this.monitoringData.performance.connection_pool || {};

            const rows = [
                { label: 'Pool Size', value: poolStats.pool_size || 0 },
                { label: 'Active Sessions', value: poolStats.active_sessions || 0 },
                { label: 'Requests', value: poolStats.requests || 0 },
                { label: 'Retries', value: poolStats.retries || 0 },
                { label: 'Errors', value: poolStats.errors || 0 },
                { label: 'Timeouts', value: poolStats.timeouts || 0 }
            ];

            let html = '';

            for (const row of rows) {
                html += `
                    <tr>
                        <td>${row.label}</td>
                        <td>${row.value}</td>
                    </tr>
                `;
            }

            connectionPoolStatsBody.innerHTML = html;
        }

        // Request Timing Table
        const requestTimingBody = document.getElementById('request-timing-body');

        if (requestTimingBody && this.monitoringData.performance) {
            const requestTiming = this.monitoringData.performance.request_timing || {};

            let html = '';

            for (const endpoint in requestTiming) {
                const timing = requestTiming[endpoint];
                const lastRequest = new Date(timing.last_request).toLocaleTimeString();

                html += `
                    <tr>
                        <td>${endpoint}</td>
                        <td>${timing.requests || 0}</td>
                        <td>${Math.round(timing.avg_time || 0)}</td>
                        <td>${Math.round(timing.min_time || 0)}</td>
                        <td>${Math.round(timing.max_time || 0)}</td>
                        <td>${lastRequest}</td>
                    </tr>
                `;
            }

            if (html) {
                requestTimingBody.innerHTML = html;
            } else {
                requestTimingBody.innerHTML = '<tr><td colspan="6" class="text-center">No request timing data available</td></tr>';
            }
        }
    }

    /**
     * Update history with new data
     */
    updateHistory(data) {
        // Update rate limit history
        if (data.rate_limits && data.rate_limits.status) {
            const timestamp = new Date().toISOString();

            for (const connector in data.rate_limits.status) {
                const status = data.rate_limits.status[connector];

                this.rateLimitHistory.unshift({
                    timestamp,
                    connector,
