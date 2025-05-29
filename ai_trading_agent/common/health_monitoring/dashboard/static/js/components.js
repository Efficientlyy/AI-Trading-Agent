/**
 * Components JavaScript
 * 
 * Handles component health status display and component-specific controls
 */

// Component status constants
const COMPONENT_STATUS = {
    HEALTHY: 'healthy',
    DEGRADED: 'degraded',
    UNHEALTHY: 'unhealthy',
    CRITICAL: 'critical',
    UNKNOWN: 'unknown'
};

// Agent status constants
const AGENT_STATUS = {
    RUNNING: 'running',
    STOPPED: 'stopped',
    INITIALIZING: 'initializing',
    ERROR: 'error',
    UNKNOWN: 'unknown'
};

// Map of component types to display names
const COMPONENT_TYPES = {
    'market_data_agent': 'Market Data Agent',
    'strategy_agent': 'Strategy Agent',
    'decision_agent': 'Decision Agent',
    'execution_agent': 'Execution Agent',
    'risk_management_agent': 'Risk Management Agent',
    'orchestrator': 'Orchestrator',
    'monitor': 'Monitor'
};

// Initialize components module
document.addEventListener('DOMContentLoaded', () => {
    console.log('Components module initializing...');
});

/**
 * Update the components table with fresh component health data
 * 
 * @param {Object} componentHealthData - Component health data from API
 */
function updateComponentsTable(componentHealthData) {
    console.log('Updating components table...');
    
    const componentsTableBody = document.getElementById('components-tbody');
    if (!componentsTableBody) {
        console.error('Components table body not found');
        return;
    }
    
    // Clear existing rows
    componentsTableBody.innerHTML = '';
    
    // Check if we have component data
    if (!componentHealthData || Object.keys(componentHealthData).length === 0) {
        const noComponentsRow = document.createElement('tr');
        noComponentsRow.innerHTML = '<td colspan="6" class="text-center">No components registered</td>';
        componentsTableBody.appendChild(noComponentsRow);
        return;
    }
    
    // Sort components by type and then id
    const sortedComponents = Object.entries(componentHealthData).sort((a, b) => {
        // First sort by component type
        const typeA = a[1].type || '';
        const typeB = b[1].type || '';
        if (typeA !== typeB) {
            return typeA.localeCompare(typeB);
        }
        
        // Then by component ID
        return a[0].localeCompare(b[0]);
    });
    
    // Group components by type
    const componentGroups = {};
    
    sortedComponents.forEach(([componentId, component]) => {
        const type = component.type || 'unknown';
        
        if (!componentGroups[type]) {
            componentGroups[type] = [];
        }
        
        componentGroups[type].push({
            id: componentId,
            ...component
        });
    });
    
    // Add components to table, grouped by type
    Object.entries(componentGroups).forEach(([type, components]) => {
        // Add group header
        const groupHeaderRow = document.createElement('tr');
        groupHeaderRow.className = 'table-secondary';
        
        // Get readable name for the group
        const groupDisplayName = COMPONENT_TYPES[type] || type.charAt(0).toUpperCase() + type.slice(1).replace(/_/g, ' ');
        
        groupHeaderRow.innerHTML = `
            <td colspan="6">
                <strong>${groupDisplayName}</strong> (${components.length} components)
            </td>
        `;
        componentsTableBody.appendChild(groupHeaderRow);
        
        // Add each component in the group
        components.forEach(component => {
            addComponentRow(componentsTableBody, component);
        });
    });
    
    // Re-initialize any tooltips or popovers
    initializeComponentTooltips();
}

/**
 * Add a single component row to the components table
 * 
 * @param {HTMLElement} tableBody - The table body element
 * @param {Object} component - Component data
 */
function addComponentRow(tableBody, component) {
    const row = document.createElement('tr');
    row.className = 'component-row';
    row.dataset.componentId = component.id;
    
    // Calculate time since last heartbeat
    const lastHeartbeatText = getLastHeartbeatText(component.last_heartbeat);
    
    // Format uptime
    const uptimeText = formatUptime(component.uptime);
    
    // Create status badge
    const statusBadge = getStatusBadgeHTML(component.status);
    
    row.innerHTML = `
        <td>${component.id}</td>
        <td>${component.description || component.id}</td>
        <td>${statusBadge}</td>
        <td>${lastHeartbeatText}</td>
        <td>${uptimeText}</td>
        <td class="component-actions">
            ${getComponentActionButtons(component)}
        </td>
    `;
    
    // Add click handler to expand component details
    row.addEventListener('click', (e) => {
        // Ignore clicks on action buttons
        if (e.target.closest('.btn')) {
            return;
        }
        
        toggleComponentDetails(component.id);
    });
    
    tableBody.appendChild(row);
    
    // Add component details row
    const detailsRow = document.createElement('tr');
    detailsRow.className = 'component-details-row';
    detailsRow.style.display = 'none';  // Initially hidden
    
    detailsRow.innerHTML = `
        <td colspan="6">
            <div class="component-details" id="details-${component.id}">
                ${getComponentDetailsHTML(component)}
            </div>
        </td>
    `;
    
    tableBody.appendChild(detailsRow);
}

/**
 * Format the component action buttons based on component type and status
 * 
 * @param {Object} component - The component data
 * @returns {string} HTML for the action buttons
 */
function getComponentActionButtons(component) {
    // Check if this is an agent type
    const isAgent = component.type && (
        component.type.includes('agent') || 
        component.type === 'orchestrator'
    );
    
    if (!isAgent) {
        // For non-agent components, just show details button
        return `<button class="btn btn-sm btn-outline-secondary view-details-btn" data-component-id="${component.id}">
            <i class="bi bi-info-circle"></i> Details
        </button>`;
    }
    
    // For agents, show start/stop/details buttons
    const isRunning = component.agent_status === AGENT_STATUS.RUNNING;
    
    return `
        <div class="btn-group btn-group-sm" role="group">
            ${isRunning ? 
                `<button class="btn btn-sm btn-outline-danger stop-agent-btn" data-component-id="${component.id}">
                    Stop
                </button>` :
                `<button class="btn btn-sm btn-outline-success start-agent-btn" data-component-id="${component.id}">
                    Start
                </button>`
            }
            <button class="btn btn-sm btn-outline-secondary view-details-btn" data-component-id="${component.id}">
                Details
            </button>
        </div>
    `;
}

/**
 * Generate HTML for the component details section
 * 
 * @param {Object} component - The component data
 * @returns {string} HTML for the component details
 */
function getComponentDetailsHTML(component) {
    // Create metrics table
    let metricsHTML = '<h6>Metrics</h6>';
    
    if (component.metrics && Object.keys(component.metrics).length > 0) {
        metricsHTML += '<table class="table table-sm">';
        metricsHTML += '<thead><tr><th>Metric</th><th>Value</th><th>Status</th></tr></thead>';
        metricsHTML += '<tbody>';
        
        Object.entries(component.metrics).forEach(([metricName, metric]) => {
            const metricStatus = metric.status || 'unknown';
            const statusBadge = getStatusBadgeHTML(metricStatus);
            
            metricsHTML += `<tr>
                <td>${metricName}</td>
                <td>${metric.value !== undefined ? metric.value : 'N/A'}</td>
                <td>${statusBadge}</td>
            </tr>`;
        });
        
        metricsHTML += '</tbody></table>';
    } else {
        metricsHTML += '<p>No metrics available</p>';
    }
    
    // Create heartbeat history
    let heartbeatHTML = '<h6>Heartbeat History</h6>';
    
    if (component.heartbeat_history && component.heartbeat_history.length > 0) {
        heartbeatHTML += '<div class="heartbeat-history">';
        
        component.heartbeat_history.forEach((heartbeat, index) => {
            const timestamp = new Date(heartbeat.timestamp * 1000).toLocaleString();
            const status = heartbeat.missed ? 'missed' : 'received';
            
            heartbeatHTML += `<div class="heartbeat-item ${status}">
                <small>${timestamp}: ${status}</small>
            </div>`;
            
            // Limit to last 5 heartbeats
            if (index >= 4) {
                return;
            }
        });
        
        heartbeatHTML += '</div>';
    } else {
        heartbeatHTML += '<p>No heartbeat history</p>';
    }
    
    // Agent-specific details
    let agentHTML = '';
    if (component.type && component.type.includes('agent')) {
        const agentStatus = component.agent_status || 'unknown';
        
        agentHTML = `
            <h6>Agent Details</h6>
            <div class="agent-details">
                <div>
                    <strong>Status:</strong> 
                    <span class="agent-status-${agentStatus.toLowerCase()}">${agentStatus}</span>
                </div>
                <div>
                    <strong>Last Execution:</strong> 
                    ${component.last_execution ? new Date(component.last_execution * 1000).toLocaleString() : 'Never'}
                </div>
                <div>
                    <strong>Execution Count:</strong> 
                    ${component.execution_count || 0}
                </div>
            </div>
        `;
    }
    
    // Combine all sections
    return `
        <div class="row">
            <div class="col-md-6">
                ${metricsHTML}
            </div>
            <div class="col-md-6">
                ${heartbeatHTML}
                ${agentHTML}
            </div>
        </div>
    `;
}

/**
 * Create a status badge HTML element
 * 
 * @param {string} status - The status value 
 * @returns {string} HTML for the status badge
 */
function getStatusBadgeHTML(status) {
    status = status ? status.toLowerCase() : 'unknown';
    
    return `<span class="status-badge ${status}">${status}</span>`;
}

/**
 * Format last heartbeat time as readable text
 * 
 * @param {number} timestamp - Unix timestamp of last heartbeat
 * @returns {string} Formatted time string
 */
function getLastHeartbeatText(timestamp) {
    if (!timestamp) {
        return 'Never';
    }
    
    const now = Math.floor(Date.now() / 1000);
    const diff = now - timestamp;
    
    if (diff < 60) {
        return `${diff} seconds ago`;
    } else if (diff < 3600) {
        return `${Math.floor(diff / 60)} minutes ago`;
    } else if (diff < 86400) {
        return `${Math.floor(diff / 3600)} hours ago`;
    } else {
        return new Date(timestamp * 1000).toLocaleString();
    }
}

/**
 * Format uptime as readable text
 * 
 * @param {number} uptime - Uptime in seconds
 * @returns {string} Formatted uptime string
 */
function formatUptime(uptime) {
    if (!uptime) {
        return 'N/A';
    }
    
    if (uptime < 60) {
        return `${Math.floor(uptime)} seconds`;
    } else if (uptime < 3600) {
        return `${Math.floor(uptime / 60)} minutes`;
    } else if (uptime < 86400) {
        return `${Math.floor(uptime / 3600)} hours`;
    } else {
        const days = Math.floor(uptime / 86400);
        const hours = Math.floor((uptime % 86400) / 3600);
        return `${days} days, ${hours} hours`;
    }
}

/**
 * Toggle showing/hiding component details
 * 
 * @param {string} componentId - ID of the component to toggle details for
 */
function toggleComponentDetails(componentId) {
    const detailsRow = document.querySelector(`.component-details-row[data-component-id="${componentId}"]`);
    const detailsDiv = document.getElementById(`details-${componentId}`);
    
    if (detailsRow && detailsDiv) {
        if (detailsRow.style.display === 'none') {
            // Show details
            detailsRow.style.display = '';
            detailsDiv.style.display = 'block';
        } else {
            // Hide details
            detailsRow.style.display = 'none';
            detailsDiv.style.display = 'none';
        }
    }
}

/**
 * Initialize tooltips and other component UI elements
 */
function initializeComponentTooltips() {
    // Add event listeners for action buttons
    document.querySelectorAll('.view-details-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const componentId = btn.dataset.componentId;
            toggleComponentDetails(componentId);
        });
    });
    
    document.querySelectorAll('.start-agent-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const componentId = btn.dataset.componentId;
            startAgent(componentId);
        });
    });
    
    document.querySelectorAll('.stop-agent-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const componentId = btn.dataset.componentId;
            stopAgent(componentId);
        });
    });
}

/**
 * Start a specific agent
 * 
 * @param {string} agentId - ID of the agent to start
 */
function startAgent(agentId) {
    console.log(`Starting agent: ${agentId}`);
    
    fetch(`/api/start-agent/${agentId}`, {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Failed to start agent: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showSuccessMessage(`Started agent ${agentId}`);
            // Refresh data after a short delay
            setTimeout(() => refreshDashboardData(), 1000);
        } else {
            showErrorMessage(`Failed to start agent ${agentId}: ${data.error}`);
        }
    })
    .catch(error => {
        console.error(`Error starting agent ${agentId}:`, error);
        showErrorMessage(`Error starting agent ${agentId}: ${error.message}`);
    });
}

/**
 * Stop a specific agent
 * 
 * @param {string} agentId - ID of the agent to stop
 */
function stopAgent(agentId) {
    console.log(`Stopping agent: ${agentId}`);
    
    fetch(`/api/stop-agent/${agentId}`, {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Failed to stop agent: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showSuccessMessage(`Stopped agent ${agentId}`);
            // Refresh data after a short delay
            setTimeout(() => refreshDashboardData(), 1000);
        } else {
            showErrorMessage(`Failed to stop agent ${agentId}: ${data.error}`);
        }
    })
    .catch(error => {
        console.error(`Error stopping agent ${agentId}:`, error);
        showErrorMessage(`Error stopping agent ${agentId}: ${error.message}`);
    });
}
