/**
 * Alerts JavaScript
 * 
 * Handles alert display, filtering and management
 */

// Alert severity constants
const ALERT_SEVERITY = {
    CRITICAL: 'critical',
    ERROR: 'error',
    WARNING: 'warning',
    INFO: 'info'
};

// Initialize alerts module
document.addEventListener('DOMContentLoaded', () => {
    console.log('Alerts module initializing...');
});

/**
 * Update the alerts table with fresh alerts data
 * 
 * @param {Array} alertsData - Alerts data from API
 */
function updateAlertsTable(alertsData) {
    console.log('Updating alerts table...');
    
    const alertsTableBody = document.getElementById('alerts-tbody');
    if (!alertsTableBody) {
        console.error('Alerts table body not found');
        return;
    }
    
    // Clear existing rows
    alertsTableBody.innerHTML = '';
    
    // Check if we have alerts data
    if (!alertsData || alertsData.length === 0) {
        const noAlertsRow = document.createElement('tr');
        noAlertsRow.innerHTML = '<td colspan="6" class="text-center">No active alerts</td>';
        alertsTableBody.appendChild(noAlertsRow);
        return;
    }
    
    // Sort alerts by severity (critical first) and then by timestamp (newest first)
    const sortedAlerts = [...alertsData].sort((a, b) => {
        // First sort by severity
        const severityOrder = {
            'critical': 0,
            'error': 1,
            'warning': 2,
            'info': 3
        };
        
        const severityA = severityOrder[a.severity.toLowerCase()] || 4;
        const severityB = severityOrder[b.severity.toLowerCase()] || 4;
        
        if (severityA !== severityB) {
            return severityA - severityB;
        }
        
        // Then sort by timestamp, newer first
        return b.timestamp - a.timestamp;
    });
    
    // Add alerts to table
    sortedAlerts.forEach(alert => {
        addAlertRow(alertsTableBody, alert);
    });
    
    // Initialize alert action buttons
    initializeAlertActions();
}

/**
 * Add a single alert row to the alerts table
 * 
 * @param {HTMLElement} tableBody - The table body element
 * @param {Object} alert - Alert data
 */
function addAlertRow(tableBody, alert) {
    const row = document.createElement('tr');
    row.className = `alert-row alert-${alert.severity.toLowerCase()}`;
    row.dataset.alertId = alert.id;
    
    // Format timestamp
    const timestamp = new Date(alert.timestamp * 1000).toLocaleString();
    
    // Get severity badge
    const severityBadge = getAlertSeverityBadge(alert.severity);
    
    row.innerHTML = `
        <td>${alert.id}</td>
        <td>${alert.component_id}</td>
        <td>${severityBadge}</td>
        <td>${alert.message}</td>
        <td>${timestamp}</td>
        <td class="alert-actions">
            <div class="btn-group btn-group-sm">
                <button class="btn btn-sm btn-outline-primary acknowledge-btn" data-alert-id="${alert.id}">
                    Acknowledge
                </button>
                <button class="btn btn-sm btn-outline-secondary details-btn" data-alert-id="${alert.id}">
                    Details
                </button>
            </div>
        </td>
    `;
    
    tableBody.appendChild(row);
}

/**
 * Get alert severity badge HTML
 * 
 * @param {string} severity - Alert severity
 * @returns {string} HTML for the severity badge
 */
function getAlertSeverityBadge(severity) {
    severity = severity ? severity.toLowerCase() : 'info';
    
    return `<span class="status-badge ${severity}">${severity}</span>`;
}

/**
 * Initialize event listeners for alert action buttons
 */
function initializeAlertActions() {
    // Acknowledge buttons
    document.querySelectorAll('.acknowledge-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const alertId = btn.dataset.alertId;
            acknowledgeAlert(alertId);
        });
    });
    
    // Details buttons
    document.querySelectorAll('.details-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const alertId = btn.dataset.alertId;
            showAlertDetails(alertId);
        });
    });
}

/**
 * Acknowledge an alert
 * 
 * @param {string} alertId - ID of the alert to acknowledge
 */
function acknowledgeAlert(alertId) {
    console.log(`Acknowledging alert: ${alertId}`);
    
    fetch(`/api/acknowledge-alert/${alertId}`, {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Failed to acknowledge alert: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showSuccessMessage(`Alert ${alertId} acknowledged`);
            
            // Remove alert from table
            const alertRow = document.querySelector(`.alert-row[data-alert-id="${alertId}"]`);
            if (alertRow) {
                alertRow.remove();
            }
            
            // Check if table is now empty
            const alertsTableBody = document.getElementById('alerts-tbody');
            if (alertsTableBody && alertsTableBody.children.length === 0) {
                const noAlertsRow = document.createElement('tr');
                noAlertsRow.innerHTML = '<td colspan="6" class="text-center">No active alerts</td>';
                alertsTableBody.appendChild(noAlertsRow);
            }
        } else {
            showErrorMessage(`Failed to acknowledge alert: ${data.error}`);
        }
    })
    .catch(error => {
        console.error(`Error acknowledging alert ${alertId}:`, error);
        showErrorMessage(`Error acknowledging alert: ${error.message}`);
    });
}

/**
 * Show detailed information for an alert
 * 
 * @param {string} alertId - ID of the alert to show details for
 */
function showAlertDetails(alertId) {
    console.log(`Showing details for alert: ${alertId}`);
    
    fetch(`/api/alert-details/${alertId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to get alert details: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.alert) {
                showAlertDetailsModal(data.alert);
            } else {
                showErrorMessage(`Alert details not found for ID: ${alertId}`);
            }
        })
        .catch(error => {
            console.error(`Error fetching alert details for ${alertId}:`, error);
            showErrorMessage(`Error fetching alert details: ${error.message}`);
        });
}

/**
 * Show a modal with alert details
 * 
 * @param {Object} alert - Alert data
 */
function showAlertDetailsModal(alert) {
    // Create modal HTML
    const modalHtml = `
        <div class="modal fade" id="alertDetailsModal" tabindex="-1" aria-labelledby="alertDetailsModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="alertDetailsModalLabel">Alert Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="alert-details">
                            <div class="row mb-3">
                                <div class="col-md-3"><strong>Alert ID:</strong></div>
                                <div class="col-md-9">${alert.id}</div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-md-3"><strong>Component:</strong></div>
                                <div class="col-md-9">${alert.component_id}</div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-md-3"><strong>Severity:</strong></div>
                                <div class="col-md-9">${getAlertSeverityBadge(alert.severity)}</div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-md-3"><strong>Timestamp:</strong></div>
                                <div class="col-md-9">${new Date(alert.timestamp * 1000).toLocaleString()}</div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-md-3"><strong>Message:</strong></div>
                                <div class="col-md-9">${alert.message}</div>
                            </div>
                            ${alert.details ? `
                                <div class="row mb-3">
                                    <div class="col-md-3"><strong>Details:</strong></div>
                                    <div class="col-md-9">
                                        <pre class="alert-details-code">${JSON.stringify(alert.details, null, 2)}</pre>
                                    </div>
                                </div>
                            ` : ''}
                            ${alert.related_metrics ? `
                                <div class="row mb-3">
                                    <div class="col-md-3"><strong>Related Metrics:</strong></div>
                                    <div class="col-md-9">
                                        <ul class="list-group">
                                            ${Object.entries(alert.related_metrics).map(([name, value]) => `
                                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                                    ${name}
                                                    <span class="badge bg-primary rounded-pill">${value}</span>
                                                </li>
                                            `).join('')}
                                        </ul>
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-primary acknowledge-modal-btn" data-alert-id="${alert.id}">
                            Acknowledge
                        </button>
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to page
    const modalContainer = document.createElement('div');
    modalContainer.innerHTML = modalHtml;
    document.body.appendChild(modalContainer);
    
    // Initialize modal
    const modal = new bootstrap.Modal(document.getElementById('alertDetailsModal'));
    modal.show();
    
    // Add event listener for acknowledge button
    const acknowledgeBtn = document.querySelector('.acknowledge-modal-btn');
    if (acknowledgeBtn) {
        acknowledgeBtn.addEventListener('click', () => {
            const alertId = acknowledgeBtn.dataset.alertId;
            acknowledgeAlert(alertId);
            modal.hide();
        });
    }
    
    // Clean up modal when hidden
    document.getElementById('alertDetailsModal').addEventListener('hidden.bs.modal', function () {
        document.body.removeChild(modalContainer);
    });
}
