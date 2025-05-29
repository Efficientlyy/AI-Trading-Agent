/**
 * Main Dashboard JavaScript
 * 
 * Handles dashboard initialization, data refresh, and common functionality
 */

// Dashboard state
const dashboardState = {
    lastUpdate: null,
    refreshInterval: 5000, // 5 seconds
    autoRefresh: true,
    systemHealth: {},
    componentHealth: {},
    activeAlerts: [],
    metrics: {},
    recoveryHistory: [],
    refreshTimer: null
};

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initializing...');
    
    // Set up event listeners
    setupEventListeners();
    
    // Load initial data
    refreshDashboardData();
    
    // Start auto-refresh
    startAutoRefresh();
});

/**
 * Set up event listeners for dashboard controls
 */
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            refreshDashboardData();
        });
    }
    
    // Auto-recovery toggle
    const autoRecoveryToggle = document.getElementById('auto-recovery');
    if (autoRecoveryToggle) {
        autoRecoveryToggle.addEventListener('change', (e) => {
            toggleAutoRecovery(e.target.checked);
        });
    }
    
    // Start all agents button
    const startAllBtn = document.getElementById('start-all');
    if (startAllBtn) {
        startAllBtn.addEventListener('click', () => {
            startAllAgents();
        });
    }
    
    // Stop all agents button
    const stopAllBtn = document.getElementById('stop-all');
    if (stopAllBtn) {
        stopAllBtn.addEventListener('click', () => {
            stopAllAgents();
        });
    }
    
    // Tab change events
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Get section id from href
            const sectionId = link.getAttribute('href').replace('#', '');
            
            // Scroll to section
            const section = document.getElementById(sectionId);
            if (section) {
                section.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

/**
 * Start automatic dashboard refresh
 */
function startAutoRefresh() {
    if (dashboardState.refreshTimer) {
        clearInterval(dashboardState.refreshTimer);
    }
    
    dashboardState.refreshTimer = setInterval(() => {
        if (dashboardState.autoRefresh) {
            refreshDashboardData();
        }
    }, dashboardState.refreshInterval);
    
    console.log(`Auto-refresh started (every ${dashboardState.refreshInterval / 1000} seconds)`);
}

/**
 * Refresh all dashboard data
 */
function refreshDashboardData() {
    console.log('Refreshing dashboard data...');
    
    fetch('/api/dashboard-data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Update dashboard state
            dashboardState.systemHealth = data.system_health || {};
            dashboardState.componentHealth = data.component_health || {};
            dashboardState.activeAlerts = data.active_alerts || [];
            dashboardState.metrics = data.metrics || {};
            dashboardState.recoveryHistory = data.recovery_history || [];
            dashboardState.lastUpdate = data.last_update || Date.now() / 1000;
            
            // Update UI with new data
            updateDashboardUI();
        })
        .catch(error => {
            console.error('Error fetching dashboard data:', error);
            showErrorMessage(`Failed to refresh dashboard data: ${error.message}`);
        });
}

/**
 * Update dashboard UI with current state data
 */
function updateDashboardUI() {
    // Update last update time
    updateLastUpdateTime();
    
    // Update system health overview
    updateSystemHealthOverview();
    
    // Update components table (via components.js)
    if (typeof updateComponentsTable === 'function') {
        updateComponentsTable(dashboardState.componentHealth);
    }
    
    // Update alerts table (via alerts.js)
    if (typeof updateAlertsTable === 'function') {
        updateAlertsTable(dashboardState.activeAlerts);
    }
    
    // Update metrics (via metrics.js)
    if (typeof updateMetricsCharts === 'function') {
        updateMetricsCharts(dashboardState.metrics);
    }
}

/**
 * Update the last update time display
 */
function updateLastUpdateTime() {
    const lastUpdateElement = document.getElementById('last-update-time');
    if (lastUpdateElement && dashboardState.lastUpdate) {
        const date = new Date(dashboardState.lastUpdate * 1000);
        lastUpdateElement.textContent = date.toLocaleString();
    }
}

/**
 * Update the system health overview section
 */
function updateSystemHealthOverview() {
    const systemHealth = dashboardState.systemHealth;
    
    // Update overall status
    const overallStatusElement = document.getElementById('overall-status');
    if (overallStatusElement) {
        overallStatusElement.textContent = systemHealth.overall_status || 'Unknown';
        
        // Clear existing classes
        overallStatusElement.className = '';
        
        // Add status-specific class
        overallStatusElement.classList.add(systemHealth.overall_status || 'unknown');
    }
    
    // Update status indicator in navbar
    const statusIndicator = document.getElementById('system-status-indicator');
    if (statusIndicator) {
        const badge = statusIndicator.querySelector('.status-badge');
        if (badge) {
            badge.textContent = systemHealth.overall_status || 'Unknown';
            
            // Clear existing classes
            badge.className = 'status-badge';
            
            // Add status-specific class
            badge.classList.add(systemHealth.overall_status || 'unknown');
        }
    }
    
    // Update component counts
    if (systemHealth.status_counts) {
        const counts = systemHealth.status_counts;
        
        updateCountElement('healthy-count', counts.healthy || 0);
        updateCountElement('degraded-count', counts.degraded || 0);
        updateCountElement('unhealthy-count', counts.unhealthy || 0);
        updateCountElement('critical-count', counts.critical || 0);
    }
}

/**
 * Update a count element with a numeric value
 */
function updateCountElement(elementId, count) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = count;
    }
}

/**
 * Toggle auto recovery mode
 */
function toggleAutoRecovery(enabled) {
    console.log(`Setting auto-recovery to: ${enabled}`);
    
    fetch('/api/toggle-auto-recovery', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ enabled: enabled })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Failed to toggle auto-recovery: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showSuccessMessage(`Auto-recovery ${enabled ? 'enabled' : 'disabled'}`);
        } else {
            showErrorMessage(`Failed to ${enabled ? 'enable' : 'disable'} auto-recovery: ${data.error}`);
            // Revert toggle
            const toggle = document.getElementById('auto-recovery');
            if (toggle) {
                toggle.checked = !enabled;
            }
        }
    })
    .catch(error => {
        console.error('Error toggling auto-recovery:', error);
        showErrorMessage(`Error toggling auto-recovery: ${error.message}`);
        // Revert toggle
        const toggle = document.getElementById('auto-recovery');
        if (toggle) {
            toggle.checked = !enabled;
        }
    });
}

/**
 * Start all trading agents
 */
function startAllAgents() {
    console.log('Starting all agents...');
    
    fetch('/api/start-all-agents', {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Failed to start agents: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showSuccessMessage('All agents started successfully');
            // Refresh data after a short delay
            setTimeout(() => refreshDashboardData(), 1000);
        } else {
            showErrorMessage(`Failed to start agents: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Error starting agents:', error);
        showErrorMessage(`Error starting agents: ${error.message}`);
    });
}

/**
 * Stop all trading agents
 */
function stopAllAgents() {
    console.log('Stopping all agents...');
    
    fetch('/api/stop-all-agents', {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Failed to stop agents: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showSuccessMessage('All agents stopped successfully');
            // Refresh data after a short delay
            setTimeout(() => refreshDashboardData(), 1000);
        } else {
            showErrorMessage(`Failed to stop agents: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Error stopping agents:', error);
        showErrorMessage(`Error stopping agents: ${error.message}`);
    });
}

/**
 * Show a success message to the user
 */
function showSuccessMessage(message) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show';
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to page
    const container = document.querySelector('.container-fluid');
    if (container) {
        container.prepend(alertDiv);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 500);
        }, 5000);
    }
}

/**
 * Show an error message to the user
 */
function showErrorMessage(message) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to page
    const container = document.querySelector('.container-fluid');
    if (container) {
        container.prepend(alertDiv);
        
        // Auto-dismiss after 8 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 500);
        }, 8000);
    }
}
