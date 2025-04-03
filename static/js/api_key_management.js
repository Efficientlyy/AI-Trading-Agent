/**
 * API Key Management JavaScript Module
 * Handles API key management functionality in the dashboard
 */

// Initialize after document is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Element references
    const apiKeysBtn = document.getElementById('apiKeysBtn');
    const apiKeysPanel = document.getElementById('apiKeysPanel');
    const apiKeysOverlay = document.getElementById('apiKeysOverlay');
    const closeApiKeys = document.getElementById('closeApiKeys');
    const apiKeysList = document.getElementById('apiKeysList');
    const refreshApiKeys = document.getElementById('refreshApiKeys');
    const addApiKeyForm = document.getElementById('addApiKeyForm');
    const exchangeSelect = document.getElementById('exchangeSelect');
    const passphraseGroup = document.getElementById('passphraseGroup');
    const validateApiKey = document.getElementById('validateApiKey');
    const apiKeyDetailModal = document.getElementById('apiKeyDetailModal');
    const modalCloseBtn = apiKeyDetailModal.querySelector('.close-modal');
    const modalCloseFooterBtn = apiKeyDetailModal.querySelector('.close-btn');
    const deleteApiKeyBtn = document.getElementById('deleteApiKey');
    const modalExchangeTitle = document.getElementById('modalExchangeTitle');
    const apiKeyDetailContent = document.getElementById('apiKeyDetailContent');

    // If any element is missing, don't initialize
    if (!apiKeysBtn || !apiKeysPanel || !apiKeysOverlay || !closeApiKeys) {
        console.warn('API Key Management: Required elements not found');
        return;
    }

    // Exchange icons mapping
    const exchangeIcons = {
        binance: 'activity',
        coinbase: 'globe',
        kraken: 'anchor',
        ftx: 'bar-chart-2',
        kucoin: 'box',
        twitter: 'twitter',
        newsapi: 'rss',
        cryptocompare: 'database'
    };

    // Track current exchange for delete operation
    let currentExchange = null;

    // Open API Keys panel
    function openApiKeysPanel() {
        apiKeysPanel.classList.add('active');
        apiKeysOverlay.classList.add('active');
        loadApiKeys();
    }

    // Close API Keys panel
    function closeApiKeysPanel() {
        apiKeysPanel.classList.remove('active');
        apiKeysOverlay.classList.remove('active');
    }

    // Toggle display of passphrase field based on exchange
    function togglePassphraseField() {
        const exchange = exchangeSelect.value;
        if (['coinbase', 'kraken', 'kucoin'].includes(exchange)) {
            passphraseGroup.style.display = 'block';
        } else {
            passphraseGroup.style.display = 'none';
        }
    }

    // Format API key for display (mask most of it)
    function formatApiKey(key) {
        if (!key) return '****';
        if (key.length <= 8) return '****' + key.slice(-4);
        return key.slice(0, 4) + '****' + key.slice(-4);
    }

    // Current pagination state
    let currentPage = 1;
    let totalPages = 1;
    let itemsPerPage = 10;

    // Load API keys from the server with pagination
    function loadApiKeys(page = 1) {
        // Update current page
        currentPage = page;
        
        // Show loading indicator
        apiKeysList.innerHTML = `
            <div class="loading-indicator">
                <div class="spinner"></div>
                <p>Loading API keys...</p>
            </div>
        `;

        // Fetch API keys with pagination
        fetch(`/api/api_keys?page=${page}&per_page=${itemsPerPage}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load API keys');
                }
                return response.json();
            })
            .then(data => {
                // Update pagination state
                if (data.pagination) {
                    currentPage = data.pagination.page;
                    totalPages = data.pagination.total_pages;
                    itemsPerPage = data.pagination.per_page;
                }
                
                // Render API keys
                renderApiKeys(data.items, data.pagination);
            })
            .catch(error => {
                console.error('Error loading API keys:', error);
                apiKeysList.innerHTML = `
                    <div class="error-message">
                        <p>Failed to load API keys. ${error.message}</p>
                        <button id="retryLoadKeys" class="retry-btn">Retry</button>
                    </div>
                `;
                document.getElementById('retryLoadKeys').addEventListener('click', () => loadApiKeys(currentPage));
            });
    }

    // Render API keys list with pagination controls
    function renderApiKeys(items, pagination) {
        if (!items || items.length === 0) {
            apiKeysList.innerHTML = `
                <div class="empty-state">
                    <p>No API keys configured yet</p>
                    <p class="empty-hint">Add your first API key using the form below</p>
                </div>
            `;
            return;
        }

        // Create list of API keys
        let html = '';
        items.forEach(key => {
            const template = document.getElementById('apiKeyItemTemplate').innerHTML;
            const icon = exchangeIcons[key.exchange] || 'key';
            const displayName = key.exchange.charAt(0).toUpperCase() + key.exchange.slice(1);
            const testnetDisplay = key.is_testnet ? 'inline-flex' : 'none';
            const description = key.description || 'No description';
            const maskedKey = formatApiKey(key.key);

            html += template
                .replace(/{exchange}/g, key.exchange)
                .replace(/{icon}/g, icon)
                .replace(/{displayName}/g, displayName)
                .replace(/{testnetDisplay}/g, testnetDisplay)
                .replace(/{description}/g, description)
                .replace(/{maskedKey}/g, maskedKey);
        });

        // Add pagination controls if there are multiple pages
        if (pagination && pagination.total_pages > 1) {
            html += renderPaginationControls(pagination);
        }

        apiKeysList.innerHTML = html;

        // Initialize Feather icons for the new elements
        if (typeof feather !== 'undefined') {
            feather.replace();
        }

        // Add event listeners to the new elements
        document.querySelectorAll('.api-key-item .details-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const exchange = this.closest('.api-key-item').dataset.exchange;
                showApiKeyDetails(exchange);
            });
        });

        document.querySelectorAll('.api-key-item .validate-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const exchange = this.closest('.api-key-item').dataset.exchange;
                validateApiKeyByExchange(exchange);
            });
        });

        // Add event listeners to pagination controls
        document.querySelectorAll('.pagination-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const page = parseInt(this.dataset.page);
                loadApiKeys(page);
            });
        });
    }
    
    // Render pagination controls
    function renderPaginationControls(pagination) {
        const { page, total_pages, total_items } = pagination;
        
        let paginationHtml = `
            <div class="pagination-controls">
                <div class="pagination-info">
                    Showing page ${page} of ${total_pages} (${total_items} total keys)
                </div>
                <div class="pagination-buttons">
        `;
        
        // Previous button
        paginationHtml += `
            <button class="pagination-btn ${page <= 1 ? 'disabled' : ''}" 
                    ${page <= 1 ? 'disabled' : `data-page="${page - 1}"`}
                    title="Previous page">
                <i data-feather="chevron-left"></i>
            </button>
        `;
        
        // Page buttons
        const startPage = Math.max(1, page - 2);
        const endPage = Math.min(total_pages, page + 2);
        
        // First page button if not in range
        if (startPage > 1) {
            paginationHtml += `
                <button class="pagination-btn" data-page="1">1</button>
                ${startPage > 2 ? '<span class="pagination-ellipsis">...</span>' : ''}
            `;
        }
        
        // Page numbers
        for (let i = startPage; i <= endPage; i++) {
            paginationHtml += `
                <button class="pagination-btn ${i === page ? 'active' : ''}" 
                        data-page="${i}">${i}</button>
            `;
        }
        
        // Last page button if not in range
        if (endPage < total_pages) {
            paginationHtml += `
                ${endPage < total_pages - 1 ? '<span class="pagination-ellipsis">...</span>' : ''}
                <button class="pagination-btn" data-page="${total_pages}">${total_pages}</button>
            `;
        }
        
        // Next button
        paginationHtml += `
            <button class="pagination-btn ${page >= total_pages ? 'disabled' : ''}" 
                    ${page >= total_pages ? 'disabled' : `data-page="${page + 1}"`}
                    title="Next page">
                <i data-feather="chevron-right"></i>
            </button>
        `;
        
        paginationHtml += `
                </div>
            </div>
        `;
        
        return paginationHtml;
    }

    // Show API key details in modal
    function showApiKeyDetails(exchange) {
        currentExchange = exchange;
        modalExchangeTitle.textContent = `${exchange.charAt(0).toUpperCase() + exchange.slice(1)} API Key Details`;
        
        // Show loading state
        apiKeyDetailContent.innerHTML = `
            <div class="loading-indicator">
                <div class="spinner"></div>
                <p>Loading details...</p>
            </div>
        `;
        
        // Show modal
        apiKeyDetailModal.classList.add('active');
        
        // Fetch API key details
        fetch(`/api/api_keys/${exchange}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load API key details');
                }
                return response.json();
            })
            .then(data => {
                renderApiKeyDetails(data);
            })
            .catch(error => {
                console.error('Error loading API key details:', error);
                apiKeyDetailContent.innerHTML = `
                    <div class="error-message">
                        <p>Failed to load API key details. ${error.message}</p>
                    </div>
                `;
            });
    }

    // Render API key details in modal
    function renderApiKeyDetails(data) {
        let html = '';
        
        // Key field
        html += `
            <div class="key-detail-row">
                <div class="key-detail-label">API Key</div>
                <div class="key-detail-value">${data.key}</div>
            </div>
        `;
        
        // Secret field (masked)
        html += `
            <div class="key-detail-row">
                <div class="key-detail-label">API Secret</div>
                <div class="key-detail-value">••••••••••••••••</div>
            </div>
        `;
        
        // Passphrase (if exists)
        if (data.passphrase) {
            html += `
                <div class="key-detail-row">
                    <div class="key-detail-label">Passphrase</div>
                    <div class="key-detail-value">••••••••</div>
                </div>
            `;
        }
        
        // Description
        html += `
            <div class="key-detail-row">
                <div class="key-detail-label">Description</div>
                <div class="key-detail-value">${data.description || 'No description'}</div>
            </div>
        `;
        
        // Testnet status
        html += `
            <div class="key-detail-row">
                <div class="key-detail-label">Testnet/Sandbox</div>
                <div class="key-detail-value">${data.is_testnet ? 'Yes' : 'No'}</div>
            </div>
        `;
        
        // Last validated
        if (data.last_validated) {
            html += `
                <div class="key-detail-row">
                    <div class="key-detail-label">Last Validated</div>
                    <div class="key-detail-value">${data.last_validated}</div>
                </div>
            `;
        }
        
        // Validation status
        if (data.is_valid !== undefined) {
            const validationStatus = data.is_valid 
                ? '<span class="validation-status valid">Valid</span>' 
                : '<span class="validation-status invalid">Invalid</span>';
            
            html += `
                <div class="key-detail-row">
                    <div class="key-detail-label">Validation Status</div>
                    <div class="key-detail-value">${validationStatus}</div>
                </div>
            `;
        }
        
        apiKeyDetailContent.innerHTML = html;
    }

    // Delete an API key
    function deleteApiKey(exchange) {
        if (!exchange) return;
        
        fetch(`/api/api_keys/${exchange}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to delete API key');
            }
            return response.json();
        })
        .then(data => {
            // Close modal and refresh
            apiKeyDetailModal.classList.remove('active');
            showNotification('API key deleted successfully', 'success');
            loadApiKeys();
        })
        .catch(error => {
            console.error('Error deleting API key:', error);
            showNotification(`Failed to delete API key: ${error.message}`, 'error');
        });
    }

    // Validate API key
    function validateApiKeyByExchange(exchange) {
        showNotification('Validating API key...', 'info');
        
        fetch(`/api/api_keys/${exchange}/validate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Validation failed');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                showNotification('API key is valid', 'success');
            } else {
                showNotification(`API key validation failed: ${data.message}`, 'error');
            }
        })
        .catch(error => {
            console.error('Error validating API key:', error);
            showNotification(`Validation error: ${error.message}`, 'error');
        });
    }

    // Validate API key from form
    function validateFormApiKey() {
        const exchange = exchangeSelect.value;
        const apiKey = document.getElementById('apiKey').value;
        const apiSecret = document.getElementById('apiSecret').value;
        const apiPassphrase = document.getElementById('apiPassphrase').value;
        
        if (!exchange || !apiKey || !apiSecret) {
            showNotification('Please fill in the required fields', 'warning');
            return;
        }
        
        showNotification('Validating API key...', 'info');
        
        const data = {
            exchange,
            key: apiKey,
            secret: apiSecret
        };
        
        if (apiPassphrase) {
            data.passphrase = apiPassphrase;
        }
        
        fetch('/api/api_keys/validate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Validation failed');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                showNotification('API key is valid', 'success');
            } else {
                showNotification(`API key validation failed: ${data.message}`, 'error');
            }
        })
        .catch(error => {
            console.error('Error validating API key:', error);
            showNotification(`Validation error: ${error.message}`, 'error');
        });
    }

    // Add a new API key
    function addApiKey(event) {
        event.preventDefault();
        
        const exchange = exchangeSelect.value;
        const apiKey = document.getElementById('apiKey').value;
        const apiSecret = document.getElementById('apiSecret').value;
        const apiPassphrase = document.getElementById('apiPassphrase').value;
        const apiDescription = document.getElementById('apiDescription').value;
        const isTestnet = document.getElementById('isTestnet').checked;
        
        if (!exchange || !apiKey || !apiSecret) {
            showNotification('Please fill in the required fields', 'warning');
            return;
        }
        
        const data = {
            exchange,
            key: apiKey,
            secret: apiSecret,
            description: apiDescription,
            is_testnet: isTestnet
        };
        
        if (apiPassphrase) {
            data.passphrase = apiPassphrase;
        }
        
        fetch('/api/api_keys', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to add API key');
            }
            return response.json();
        })
        .then(data => {
            showNotification('API key added successfully', 'success');
            addApiKeyForm.reset();
            loadApiKeys();
        })
        .catch(error => {
            console.error('Error adding API key:', error);
            showNotification(`Failed to add API key: ${error.message}`, 'error');
        });
    }

    // Show notification
    function showNotification(message, type = 'info') {
        // Check if notification system exists
        if (typeof window.showToast === 'function') {
            window.showToast(message, type);
        } else {
            alert(message);
        }
    }

    // Event Listeners
    apiKeysBtn.addEventListener('click', openApiKeysPanel);
    closeApiKeys.addEventListener('click', closeApiKeysPanel);
    apiKeysOverlay.addEventListener('click', closeApiKeysPanel);
    refreshApiKeys.addEventListener('click', loadApiKeys);
    exchangeSelect.addEventListener('change', togglePassphraseField);
    validateApiKey.addEventListener('click', validateFormApiKey);
    addApiKeyForm.addEventListener('submit', addApiKey);
    
    // Modal close buttons
    modalCloseBtn.addEventListener('click', () => {
        apiKeyDetailModal.classList.remove('active');
    });
    
    modalCloseFooterBtn.addEventListener('click', () => {
        apiKeyDetailModal.classList.remove('active');
    });
    
    // Delete API key button
    deleteApiKeyBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to delete this API key?')) {
            deleteApiKey(currentExchange);
        }
    });

    // Set initial state
    togglePassphraseField();
});