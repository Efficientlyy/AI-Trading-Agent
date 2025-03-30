/**
 * Settings Panel JavaScript
 * 
 * Handles the settings modal functionality, including:
 * - Loading settings from the server
 * - Updating the UI based on the loaded settings
 * - Saving settings to the server
 * - Handling the settings modal tabs
 * - Toggling the real data option
 * - Opening the connection editor modal
 */

// Settings Panel Controller
class SettingsPanelController {
    constructor() {
        // DOM Elements
        this.modal = document.getElementById('settings-modal');
        this.closeButton = document.getElementById('close-settings');
        this.saveButton = document.getElementById('save-settings');
        this.resetButton = document.getElementById('reset-settings');
        this.tabButtons = document.querySelectorAll('.settings-tab');
        this.settingsPanes = document.querySelectorAll('.settings-pane');

        // Theme settings
        this.themeOptions = document.querySelectorAll('.theme-option');

        // Auto-refresh settings
        this.autoRefreshToggle = document.getElementById('auto-refresh-toggle');
        this.refreshIntervalSelect = document.getElementById('refresh-interval');
        this.refreshIntervalContainer = document.getElementById('refresh-interval-container');

        // Real data settings
        this.realDataToggle = document.getElementById('real-data-toggle');
        this.realDataOptions = document.getElementById('real-data-options');
        this.editConnectionsButton = document.getElementById('edit-connections');
        this.fallbackStrategySelect = document.getElementById('fallback-strategy');
        this.cacheDurationSelect = document.getElementById('cache-duration');

        // Display settings
        this.chartStyleSelect = document.getElementById('chart-style');
        this.defaultTimeRangeSelect = document.getElementById('default-time-range');
        this.decimalPlacesSelect = document.getElementById('decimal-places');

        // Notification settings
        this.desktopNotificationsToggle = document.getElementById('desktop-notifications-toggle');
        this.notificationLevelSelect = document.getElementById('notification-level');
        this.soundAlertsToggle = document.getElementById('sound-alerts-toggle');

        // Current settings
        this.currentSettings = {};

        // Initialize
        this.init();
    }

    /**
     * Initialize the settings panel
     */
    init() {
        // Load settings from server
        this.loadSettings();

        // Add event listeners
        this.addEventListeners();
    }

    /**
     * Add event listeners to the settings panel elements
     */
    addEventListeners() {
        // Close button
        this.closeButton.addEventListener('click', () => {
            this.hideModal();
        });

        // Save button
        this.saveButton.addEventListener('click', () => {
            this.saveSettings();
        });

        // Reset button
        this.resetButton.addEventListener('click', () => {
            this.resetSettings();
        });

        // Tab buttons
        this.tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tab = button.getAttribute('data-tab');
                this.showTab(tab);
            });
        });

        // Theme options
        this.themeOptions.forEach(option => {
            option.addEventListener('click', () => {
                const theme = option.getAttribute('data-theme');
                this.setTheme(theme);
            });
        });

        // Auto-refresh toggle
        this.autoRefreshToggle.addEventListener('change', () => {
            this.toggleAutoRefresh();
        });

        // Real data toggle
        this.realDataToggle.addEventListener('change', () => {
            this.toggleRealData();
        });

        // Edit connections button
        this.editConnectionsButton.addEventListener('click', () => {
            this.openConnectionEditor();
        });
    }

    /**
     * Show the settings modal
     */
    showModal() {
        this.modal.style.display = 'block';
        // Reload settings to ensure we have the latest
        this.loadSettings();
    }

    /**
     * Hide the settings modal
     */
    hideModal() {
        this.modal.style.display = 'none';
    }

    /**
     * Show a specific tab
     * @param {string} tabName - The name of the tab to show
     */
    showTab(tabName) {
        // Update tab buttons
        this.tabButtons.forEach(button => {
            if (button.getAttribute('data-tab') === tabName) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });

        // Update tab panes
        this.settingsPanes.forEach(pane => {
            if (pane.id === `${tabName}-settings`) {
                pane.classList.add('active');
            } else {
                pane.classList.remove('active');
            }
        });
    }

    /**
     * Load settings from the server
     */
    loadSettings() {
        // Show loading indicator
        this.showLoading();

        // Fetch settings from server
        fetch('/api/settings')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load settings');
                }
                return response.json();
            })
            .then(settings => {
                this.currentSettings = settings;
                this.updateUI();
                this.hideLoading();
            })
            .catch(error => {
                console.error('Error loading settings:', error);
                this.showError('Failed to load settings');
                this.hideLoading();
            });
    }

    /**
     * Save settings to the server
     */
    saveSettings() {
        // Get settings from UI
        const settings = this.getSettingsFromUI();

        // Show loading indicator
        this.showLoading();

        // Send settings to server
        fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to save settings');
                }
                return response.json();
            })
            .then(result => {
                if (result.success) {
                    this.showSuccess('Settings saved successfully');

                    // Check if reload is required
                    if (result.reloadRequired) {
                        this.showReloadNotification();
                    }

                    // Hide modal
                    this.hideModal();
                } else {
                    this.showError(result.message || 'Failed to save settings');
                }
                this.hideLoading();
            })
            .catch(error => {
                console.error('Error saving settings:', error);
                this.showError('Failed to save settings');
                this.hideLoading();
            });
    }

    /**
     * Reset settings to defaults
     */
    resetSettings() {
        if (confirm('Are you sure you want to reset all settings to defaults?')) {
            // Show loading indicator
            this.showLoading();

            // Reset settings on server
            fetch('/api/settings/reset', {
                method: 'POST'
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to reset settings');
                    }
                    return response.json();
                })
                .then(result => {
                    if (result.success) {
                        this.currentSettings = result.settings;
                        this.updateUI();
                        this.showSuccess('Settings reset to defaults');
                    } else {
                        this.showError(result.message || 'Failed to reset settings');
                    }
                    this.hideLoading();
                })
                .catch(error => {
                    console.error('Error resetting settings:', error);
                    this.showError('Failed to reset settings');
                    this.hideLoading();
                });
        }
    }

    /**
     * Update the UI based on the current settings
     */
    updateUI() {
        // Theme
        this.updateThemeUI();

        // Auto-refresh
        this.updateAutoRefreshUI();

        // Real data
        this.updateRealDataUI();

        // Display settings
        this.updateDisplayUI();

        // Notification settings
        this.updateNotificationsUI();
    }

    /**
     * Update the theme UI based on the current settings
     */
    updateThemeUI() {
        const theme = this.currentSettings.theme || 'system';

        this.themeOptions.forEach(option => {
            const optionTheme = option.getAttribute('data-theme');
            if (optionTheme === theme) {
                option.classList.add('active');
            } else {
                option.classList.remove('active');
            }
        });
    }

    /**
     * Update the auto-refresh UI based on the current settings
     */
    updateAutoRefreshUI() {
        const autoRefresh = this.currentSettings.autoRefresh !== false;
        const refreshInterval = this.currentSettings.refreshInterval || 30;

        this.autoRefreshToggle.checked = autoRefresh;
        this.refreshIntervalSelect.value = refreshInterval.toString();

        // Show/hide refresh interval based on auto-refresh setting
        if (autoRefresh) {
            this.refreshIntervalContainer.style.display = 'flex';
        } else {
            this.refreshIntervalContainer.style.display = 'none';
        }
    }

    /**
     * Update the real data UI based on the current settings
     */
    updateRealDataUI() {
        const realData = this.currentSettings.realData || {};
        const enabled = realData.enabled || false;
        const fallbackStrategy = realData.fallbackStrategy || 'cache_then_mock';
        const cacheDuration = realData.cacheDuration || 3600;

        this.realDataToggle.checked = enabled;
        this.fallbackStrategySelect.value = fallbackStrategy;
        this.cacheDurationSelect.value = cacheDuration.toString();

        // Show/hide real data options based on enabled setting
        if (enabled) {
            this.realDataOptions.style.display = 'flex';
        } else {
            this.realDataOptions.style.display = 'none';
        }
    }

    /**
     * Update the display UI based on the current settings
     */
    updateDisplayUI() {
        const display = this.currentSettings.display || {};
        const chartStyle = display.chartStyle || 'modern';
        const defaultTimeRange = display.defaultTimeRange || '1w';
        const decimalPlaces = display.decimalPlaces || 2;

        this.chartStyleSelect.value = chartStyle;
        this.defaultTimeRangeSelect.value = defaultTimeRange;
        this.decimalPlacesSelect.value = decimalPlaces.toString();
    }

    /**
     * Update the notifications UI based on the current settings
     */
    updateNotificationsUI() {
        const notifications = this.currentSettings.notifications || {};
        const desktopNotifications = notifications.desktop !== false;
        const notificationLevel = notifications.level || 'warning';
        const soundAlerts = notifications.sound || false;

        this.desktopNotificationsToggle.checked = desktopNotifications;
        this.notificationLevelSelect.value = notificationLevel;
        this.soundAlertsToggle.checked = soundAlerts;
    }

    /**
     * Get settings from the UI
     * @returns {Object} The settings object
     */
    getSettingsFromUI() {
        // Theme
        const theme = this.getActiveTheme();

        // Auto-refresh
        const autoRefresh = this.autoRefreshToggle.checked;
        const refreshInterval = parseInt(this.refreshIntervalSelect.value, 10);

        // Real data
        const realDataEnabled = this.realDataToggle.checked;
        const fallbackStrategy = this.fallbackStrategySelect.value;
        const cacheDuration = parseInt(this.cacheDurationSelect.value, 10);

        // Display settings
        const chartStyle = this.chartStyleSelect.value;
        const defaultTimeRange = this.defaultTimeRangeSelect.value;
        const decimalPlaces = parseInt(this.decimalPlacesSelect.value, 10);

        // Notification settings
        const desktopNotifications = this.desktopNotificationsToggle.checked;
        const notificationLevel = this.notificationLevelSelect.value;
        const soundAlerts = this.soundAlertsToggle.checked;

        // Construct settings object
        return {
            theme,
            autoRefresh,
            refreshInterval,
            realData: {
                enabled: realDataEnabled,
                fallbackStrategy,
                cacheDuration
            },
            display: {
                chartStyle,
                defaultTimeRange,
                decimalPlaces
            },
            notifications: {
                desktop: desktopNotifications,
                level: notificationLevel,
                sound: soundAlerts
            }
        };
    }

    /**
     * Get the active theme
     * @returns {string} The active theme
     */
    getActiveTheme() {
        let activeTheme = 'system';

        this.themeOptions.forEach(option => {
            if (option.classList.contains('active')) {
                activeTheme = option.getAttribute('data-theme');
            }
        });

        return activeTheme;
    }

    /**
     * Set the theme
     * @param {string} theme - The theme to set
     */
    setTheme(theme) {
        this.themeOptions.forEach(option => {
            const optionTheme = option.getAttribute('data-theme');
            if (optionTheme === theme) {
                option.classList.add('active');
            } else {
                option.classList.remove('active');
            }
        });
    }

    /**
     * Toggle auto-refresh
     */
    toggleAutoRefresh() {
        const autoRefresh = this.autoRefreshToggle.checked;

        if (autoRefresh) {
            this.refreshIntervalContainer.style.display = 'flex';
        } else {
            this.refreshIntervalContainer.style.display = 'none';
        }
    }

    /**
     * Toggle real data
     */
    toggleRealData() {
        const realData = this.realDataToggle.checked;

        if (realData) {
            this.realDataOptions.style.display = 'flex';
        } else {
            this.realDataOptions.style.display = 'none';
        }
    }

    /**
     * Open the connection editor modal
     */
    openConnectionEditor() {
        // Hide settings modal
        this.hideModal();

        // Fetch connection editor template
        fetch('/api/templates/connection_editor_modal.html')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load connection editor template');
                }
                return response.text();
            })
            .then(html => {
                // Create temporary container
                const container = document.createElement('div');
                container.innerHTML = html;

                // Append to body
                document.body.appendChild(container.firstChild);

                // Initialize connection editor
                if (window.connectionEditor) {
                    window.connectionEditor.init();
                } else {
                    console.error('Connection editor not found');
                }
            })
            .catch(error => {
                console.error('Error loading connection editor:', error);
                this.showError('Failed to load connection editor');

                // Show settings modal again
                this.showModal();
            });
    }

    /**
     * Show loading indicator
     */
    showLoading() {
        // Disable save and reset buttons
        this.saveButton.disabled = true;
        this.resetButton.disabled = true;

        // Add loading class to save button
        this.saveButton.classList.add('loading');
    }

    /**
     * Hide loading indicator
     */
    hideLoading() {
        // Enable save and reset buttons
        this.saveButton.disabled = false;
        this.resetButton.disabled = false;

        // Remove loading class from save button
        this.saveButton.classList.remove('loading');
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
     * Show reload notification
     */
    showReloadNotification() {
        if (confirm('Some settings require a page reload to take effect. Reload now?')) {
            window.location.reload();
        }
    }
}

// Initialize settings panel when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global settings panel instance
    window.settingsPanel = new SettingsPanelController();

    // Add event listener to settings button
    const settingsButton = document.getElementById('settings-button');
    if (settingsButton) {
        settingsButton.addEventListener('click', () => {
            window.settingsPanel.showModal();
        });
    }
});