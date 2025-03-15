/**
 * Backtest Interface JavaScript
 * Provides interactive features and visualizations for the backtest interface
 */

document.addEventListener('DOMContentLoaded', function() {
    // Strategy descriptions for tooltips
    const strategyDescriptions = {
        'trend_following': 'Follows the direction of the market by buying in uptrends and selling in downtrends.',
        'mean_reversion': 'Takes advantage of price reversals by buying when prices are low and selling when prices are high.',
        'momentum': 'Buys assets that have performed well in the past and sells assets that have performed poorly.',
        'volatility_based': 'Adjusts position sizes based on market volatility, taking smaller positions in high volatility regimes.',
        'regime_adaptive': 'Dynamically adapts the strategy based on the detected market regime.'
    };

    // Position sizing descriptions
    const positionSizingDescriptions = {
        'fixed': 'Uses a fixed fraction of capital for all trades.',
        'percent': 'Allocates a percentage of current equity for each trade.',
        'kelly': 'Uses the Kelly criterion to optimize position sizes based on win rate and win/loss ratio.',
        'volatility': 'Adjusts position sizes based on market volatility.'
    };

    // Update strategy description when strategy changes
    const strategySelect = document.getElementById('strategy');
    const strategyDescription = document.querySelector('.strategy-description');
    
    if (strategySelect && strategyDescription) {
        strategySelect.addEventListener('change', function() {
            strategyDescription.textContent = strategyDescriptions[this.value] || '';
        });
        
        // Set initial description
        strategyDescription.textContent = strategyDescriptions[strategySelect.value] || '';
    }

    // Update position sizing description when position sizing method changes
    const positionSizingSelect = document.getElementById('position_sizing');
    const positionSizingDescription = document.querySelector('.position-sizing-description');
    
    if (positionSizingSelect && positionSizingDescription) {
        positionSizingSelect.addEventListener('change', function() {
            positionSizingDescription.textContent = positionSizingDescriptions[this.value] || '';
        });
        
        // Set initial description
        positionSizingDescription.textContent = positionSizingDescriptions[positionSizingSelect.value] || '';
    }

    // Show loading indicator when form is submitted
    const backtestForm = document.querySelector('form');
    if (backtestForm) {
        backtestForm.addEventListener('submit', function() {
            showLoadingOverlay();
        });
    }

    // Download report functionality
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', function() {
            generatePDF();
        });
    }

    // Initialize tooltips
    initializeTooltips();
});

/**
 * Shows a loading overlay while the backtest is running
 */
function showLoadingOverlay() {
    // Create overlay element
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    
    // Create spinner
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    
    // Create message
    const message = document.createElement('div');
    message.className = 'loading-message';
    message.textContent = 'Running backtest...';
    message.style.marginLeft = '15px';
    message.style.fontSize = '18px';
    message.style.fontWeight = '500';
    
    // Create container for spinner and message
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.appendChild(spinner);
    container.appendChild(message);
    
    overlay.appendChild(container);
    document.body.appendChild(overlay);
}

/**
 * Initializes Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Generates a PDF report of the backtest results
 */
function generatePDF() {
    // Show loading message
    const downloadBtn = document.getElementById('downloadReportBtn');
    const originalText = downloadBtn.innerHTML;
    downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating PDF...';
    downloadBtn.disabled = true;
    
    // Get backtest data
    const symbol = document.querySelector('h2').textContent.split(':')[1].trim();
    const strategy = document.querySelector('p.text-muted').textContent.split(' ')[0];
    
    // Collect performance metrics
    const metrics = {};
    const metricsTable = document.querySelector('.table-striped tbody');
    if (metricsTable) {
        const rows = metricsTable.querySelectorAll('tr');
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 2) {
                const key = cells[0].textContent.trim();
                const value = cells[1].textContent.trim();
                metrics[key] = value;
            }
        });
    }
    
    // Get chart images
    const charts = document.querySelectorAll('.chart-container img');
    const chartUrls = Array.from(charts).map(img => img.src);
    
    // Prepare data for the server
    const reportData = {
        symbol: symbol,
        strategy: strategy,
        metrics: metrics,
        chartUrls: chartUrls
    };
    
    // Send request to generate PDF
    fetch('/generate_pdf', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(reportData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.blob();
    })
    .then(blob => {
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `${symbol}_${strategy}_backtest_report.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        
        // Reset button
        downloadBtn.innerHTML = originalText;
        downloadBtn.disabled = false;
    })
    .catch(error => {
        console.error('Error generating PDF:', error);
        alert('Error generating PDF report. Please try again.');
        
        // Reset button
        downloadBtn.innerHTML = originalText;
        downloadBtn.disabled = false;
    });
}

/**
 * Toggles the visibility of trade details
 * @param {number} tradeId - The ID of the trade to toggle
 */
function toggleTradeDetails(tradeId) {
    const detailsRow = document.getElementById(`trade-details-${tradeId}`);
    if (detailsRow) {
        detailsRow.classList.toggle('d-none');
    }
}

/**
 * Filters the trade list based on user input
 */
function filterTrades() {
    const filterInput = document.getElementById('trade-filter');
    const filterValue = filterInput.value.toLowerCase();
    const tradeRows = document.querySelectorAll('.trade-row');
    
    tradeRows.forEach(row => {
        const text = row.textContent.toLowerCase();
        if (text.includes(filterValue)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

/**
 * Sorts the trade table by the specified column
 * @param {string} column - The column to sort by
 */
function sortTradeTable(column) {
    const table = document.querySelector('.trade-table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr.trade-row'));
    
    // Get current sort direction
    const currentDir = table.getAttribute('data-sort-dir') || 'asc';
    const newDir = currentDir === 'asc' ? 'desc' : 'asc';
    
    // Update sort indicators
    const headers = table.querySelectorAll('th');
    headers.forEach(header => {
        header.classList.remove('sort-asc', 'sort-desc');
    });
    
    const header = table.querySelector(`th[data-column="${column}"]`);
    if (header) {
        header.classList.add(newDir === 'asc' ? 'sort-asc' : 'sort-desc');
    }
    
    // Sort rows
    rows.sort((a, b) => {
        const aValue = a.querySelector(`td[data-column="${column}"]`).textContent;
        const bValue = b.querySelector(`td[data-column="${column}"]`).textContent;
        
        // Handle numeric values
        if (!isNaN(parseFloat(aValue)) && !isNaN(parseFloat(bValue))) {
            return newDir === 'asc' 
                ? parseFloat(aValue) - parseFloat(bValue)
                : parseFloat(bValue) - parseFloat(aValue);
        }
        
        // Handle date values
        if (aValue.match(/^\d{4}-\d{2}-\d{2}/) && bValue.match(/^\d{4}-\d{2}-\d{2}/)) {
            return newDir === 'asc'
                ? new Date(aValue) - new Date(bValue)
                : new Date(bValue) - new Date(aValue);
        }
        
        // Handle text values
        return newDir === 'asc'
            ? aValue.localeCompare(bValue)
            : bValue.localeCompare(aValue);
    });
    
    // Update table
    rows.forEach(row => {
        tbody.appendChild(row);
        
        // Also move any detail rows
        const id = row.getAttribute('data-trade-id');
        const detailRow = document.getElementById(`trade-details-${id}`);
        if (detailRow) {
            tbody.appendChild(detailRow);
        }
    });
    
    // Update sort direction attribute
    table.setAttribute('data-sort-dir', newDir);
    table.setAttribute('data-sort-column', column);
} 