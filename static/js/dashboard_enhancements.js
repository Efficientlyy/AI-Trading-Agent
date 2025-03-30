/**
 * Dashboard Enhancements
 * 
 * This script provides additional functionality for the dashboard:
 * 1. Global loading indicator
 * 2. Persistent cache for dashboard data
 * 3. Enhanced error handling
 */

document.addEventListener('DOMContentLoaded', function () {
    // Initialize the dashboard enhancements
    const dashboardEnhancements = new DashboardEnhancements();
    dashboardEnhancements.init();

    // Make it globally accessible
    window.dashboardEnhancements = dashboardEnhancements;
});

/**
 * Dashboard Enhancements Class
 */
class DashboardEnhancements {
    constructor() {
        // Cache configuration
        this.cacheConfig = {
            // Cache expiration times in seconds
            expirationTimes: {
                'system_health': 5,
                'component_status': 10,
                'trading_performance': 30,
                'current_positions': 30,
                'recent_trades': 30,
                'system_alerts': 10,
                'equity_curve': 60,
                'market_regime': 60,
                'sentiment': 60,
                'risk_management': 60,
                'performance_analytics': 300,
                'logs_monitoring': 10
            },
            // Default expiration time if not specified
            defaultExpiration: 60,
            // Whether to use persistent cache
            usePersistentCache: true,
            // Prefix for localStorage keys
            storagePrefix: 'dashboard_cache_'
        };

        // In-memory cache
        this.memoryCache = {};

        // Loading state
        this.isLoading = false;
        this.pendingRequests = 0;
    }

    /**
     * Initialize the dashboard enhancements
     */
    init() {
        // Create the global loading indicator
        this.createLoadingIndicator();

        // Intercept fetch requests to show loading indicator
        this.interceptFetchRequests();

        // Add error container for detailed error messages
        this.createErrorContainer();
    }

    /**
     * Create the global loading indicator
     */
    createLoadingIndicator() {
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'global-loading-indicator hidden';
        loadingIndicator.id = 'globalLoadingIndicator';
        document.body.appendChild(loadingIndicator);
    }

    /**
     * Create error container for detailed error messages
     */
    createErrorContainer() {
        const errorContainer = document.createElement('div');
        errorContainer.id = 'errorContainer';
        errorContainer.style.display = 'none';
        document.body.appendChild(errorContainer);
    }

    /**
     * Intercept fetch requests to show loading indicator
     */
    interceptFetchRequests() {
        const originalFetch = window.fetch;
        const self = this;

        window.fetch = function () {
            const fetchArgs = arguments;
            const url = typeof fetchArgs[0] === 'string' ? fetchArgs[0] : fetchArgs[0].url;

            // Check if this is an API request
            if (url.includes('/api/')) {
                // Show loading indicator
                self.showLoading();

                // Try to get from cache first
                const cachedData = self.getFromCache(url);
                if (cachedData) {
                    // Return cached data immediately
                    const response = new Response(JSON.stringify(cachedData.data), {
                        status: 200,
                        headers: { 'Content-Type': 'application/json' }
                    });

                    // Hide loading indicator
                    self.hideLoading();

                    // Return a resolved promise with the cached response
                    return Promise.resolve(response);
                }

                // If not in cache or expired, make the actual request
                return originalFetch.apply(this, fetchArgs)
                    .then(response => {
                        // Clone the response to use it twice
                        const clone = response.clone();

                        // Process the response
                        clone.json().then(data => {
                            // Cache the response
                            self.saveToCache(url, data);
                        }).catch(err => {
                            console.error('Error parsing JSON from response:', err);
                        });

                        // Hide loading indicator
                        self.hideLoading();

                        return response;
                    })
                    .catch(error => {
                        // Hide loading indicator
                        self.hideLoading();

                        // Show error message
                        self.showError('Network Error', 'Failed to fetch data from the server. Please check your connection and try again.');

                        // Rethrow the error
                        throw error;
                    });
            }

            // For non-API requests, just use the original fetch
            return originalFetch.apply(this, fetchArgs);
        };
    }

    /**
     * Show the loading indicator
     */
    showLoading() {
        this.pendingRequests++;
        this.isLoading = true;

        const loadingIndicator = document.getElementById('globalLoadingIndicator');
        if (loadingIndicator) {
            loadingIndicator.classList.remove('hidden');
        }
    }

    /**
     * Hide the loading indicator
     */
    hideLoading() {
        this.pendingRequests--;

        if (this.pendingRequests <= 0) {
            this.pendingRequests = 0;
            this.isLoading = false;

            const loadingIndicator = document.getElementById('globalLoadingIndicator');
            if (loadingIndicator) {
                loadingIndicator.classList.add('hidden');
            }
        }
    }

    /**
     * Show an error message
     * 
     * @param {string} title - The error title
     * @param {string} message - The error message
     * @param {Object} details - Additional error details
     */
    showError(title, message, details = null) {
        // Create error element
        const errorElement = document.createElement('div');
        errorElement.className = 'error-details';

        // Create error content
        let errorContent = `
            <div class="error-title">${title}</div>
            <div class="error-message">${message}</div>
        `;

        // Add details if provided
        if (details) {
            errorContent += `
                <div class="error-details-content">
                    <pre>${JSON.stringify(details, null, 2)}</pre>
                </div>
            `;
        }

        // Add close button
        errorContent += `
            <button class="error-close-btn">Close</button>
        `;

        // Set error content
        errorElement.innerHTML = errorContent;

        // Add close button event listener
        errorElement.querySelector('.error-close-btn').addEventListener('click', function () {
            document.body.removeChild(errorElement);
        });

        // Add to body
        document.body.appendChild(errorElement);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (document.body.contains(errorElement)) {
                document.body.removeChild(errorElement);
            }
        }, 10000);
    }

    /**
     * Get data from cache
     * 
     * @param {string} key - The cache key
     * @returns {Object|null} - The cached data or null if not found or expired
     */
    getFromCache(key) {
        // Try memory cache first
        if (this.memoryCache[key]) {
            const { data, timestamp, expiration } = this.memoryCache[key];

            // Check if expired
            if (Date.now() - timestamp < expiration * 1000) {
                return { data, timestamp };
            }
        }

        // If not in memory cache or expired, try persistent cache
        if (this.cacheConfig.usePersistentCache) {
            try {
                const storageKey = this.cacheConfig.storagePrefix + this.hashKey(key);
                const cachedItem = localStorage.getItem(storageKey);

                if (cachedItem) {
                    const { data, timestamp, expiration } = JSON.parse(cachedItem);

                    // Check if expired
                    if (Date.now() - timestamp < expiration * 1000) {
                        // Update memory cache
                        this.memoryCache[key] = { data, timestamp, expiration };
                        return { data, timestamp };
                    }

                    // If expired, remove from localStorage
                    localStorage.removeItem(storageKey);
                }
            } catch (error) {
                console.error('Error reading from persistent cache:', error);
            }
        }

        return null;
    }

    /**
     * Save data to cache
     * 
     * @param {string} key - The cache key
     * @param {Object} data - The data to cache
     */
    saveToCache(key, data) {
        // Determine expiration time
        let expiration = this.cacheConfig.defaultExpiration;

        // Check if key contains a known data type
        for (const dataType in this.cacheConfig.expirationTimes) {
            if (key.includes(dataType)) {
                expiration = this.cacheConfig.expirationTimes[dataType];
                break;
            }
        }

        // Save to memory cache
        const timestamp = Date.now();
        this.memoryCache[key] = { data, timestamp, expiration };

        // Save to persistent cache if enabled
        if (this.cacheConfig.usePersistentCache) {
            try {
                const storageKey = this.cacheConfig.storagePrefix + this.hashKey(key);
                localStorage.setItem(storageKey, JSON.stringify({ data, timestamp, expiration }));
            } catch (error) {
                console.error('Error saving to persistent cache:', error);

                // If localStorage is full, clear old items
                if (error instanceof DOMException && error.name === 'QuotaExceededError') {
                    this.clearOldCacheItems();

                    // Try again
                    try {
                        const storageKey = this.cacheConfig.storagePrefix + this.hashKey(key);
                        localStorage.setItem(storageKey, JSON.stringify({ data, timestamp, expiration }));
                    } catch (retryError) {
                        console.error('Error saving to persistent cache after clearing old items:', retryError);
                    }
                }
            }
        }
    }

    /**
     * Clear old cache items
     */
    clearOldCacheItems() {
        // Get all cache keys
        const cacheKeys = [];
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key.startsWith(this.cacheConfig.storagePrefix)) {
                cacheKeys.push(key);
            }
        }

        // Sort by timestamp (oldest first)
        cacheKeys.sort((a, b) => {
            const aData = JSON.parse(localStorage.getItem(a));
            const bData = JSON.parse(localStorage.getItem(b));
            return aData.timestamp - bData.timestamp;
        });

        // Remove oldest 50% of items
        const itemsToRemove = Math.ceil(cacheKeys.length / 2);
        for (let i = 0; i < itemsToRemove; i++) {
            localStorage.removeItem(cacheKeys[i]);
        }
    }

    /**
     * Hash a key for storage
     * 
     * @param {string} key - The key to hash
     * @returns {string} - The hashed key
     */
    hashKey(key) {
        let hash = 0;
        for (let i = 0; i < key.length; i++) {
            const char = key.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return hash.toString(16);
    }

    /**
     * Clear all cache
     */
    clearCache() {
        // Clear memory cache
        this.memoryCache = {};

        // Clear persistent cache
        if (this.cacheConfig.usePersistentCache) {
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key.startsWith(this.cacheConfig.storagePrefix)) {
                    localStorage.removeItem(key);
                }
            }
        }
    }
}