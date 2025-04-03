/**
 * Advanced Data Validation JavaScript
 * 
 * Handles the advanced data validation UI components and functionality
 */

// Data Validation Controller
class DataValidationController {
    constructor() {
        // DOM Elements
        this.validationStatusIndicators = document.querySelectorAll('.validation-status');
        this.validationResultsPanels = document.querySelectorAll('.validation-results');
        this.validationRulesPanels = document.querySelectorAll('.validation-rules');
        this.validationSchemaEditors = document.querySelectorAll('.validation-schema-editor');
        this.validationLevelSelectors = document.querySelectorAll('.validation-level-selector');

        // State
        this.validationResults = {};
        this.validationRules = {};
        this.validationSchemas = {};
        this.validationLevels = ['BASIC', 'STANDARD', 'STRICT'];
        this.currentValidationLevel = 'STANDARD';

        // Initialize
        this.init();
    }

    /**
     * Initialize the data validation controller
     */
    init() {
        // Initialize validation status indicators
        this.initValidationStatusIndicators();

        // Initialize validation results panels
        this.initValidationResultsPanels();

        // Initialize validation rules panels
        this.initValidationRulesPanels();

        // Initialize validation schema editors
        this.initValidationSchemaEditors();

        // Initialize validation level selectors
        this.initValidationLevelSelectors();

        // Load initial validation data
        this.loadValidationData();

        console.log('Data validation controller initialized');
    }

    /**
     * Initialize validation status indicators
     */
    initValidationStatusIndicators() {
        this.validationStatusIndicators.forEach(indicator => {
            // Set initial status
            const initialStatus = indicator.dataset.initialStatus || 'pending';
            this.updateValidationStatus(indicator, initialStatus);

            // Add click handler to show validation details
            indicator.addEventListener('click', () => {
                const targetId = indicator.dataset.target;
                if (targetId) {
                    const target = document.getElementById(targetId);
                    if (target) {
                        target.style.display = target.style.display === 'none' ? 'block' : 'none';
                    }
                }
            });
        });
    }

    /**
     * Update validation status
     * @param {HTMLElement} indicator - The validation status indicator element
     * @param {string} status - The validation status (valid, warning, invalid, pending)
     */
    updateValidationStatus(indicator, status) {
        // Remove existing status classes
        indicator.classList.remove('valid', 'warning', 'invalid', 'pending');

        // Add new status class
        indicator.classList.add(status);

        // Update text content
        indicator.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }

    /**
     * Initialize validation results panels
     */
    initValidationResultsPanels() {
        this.validationResultsPanels.forEach(panel => {
            // Get panel elements
            const refreshButton = panel.querySelector('.refresh-validation');
            const clearButton = panel.querySelector('.clear-validation');
            const exportButton = panel.querySelector('.export-validation');
            const filterItems = panel.querySelectorAll('.filter-item input');

            // Add refresh button handler
            if (refreshButton) {
                refreshButton.addEventListener('click', () => {
                    const dataSource = panel.dataset.source;
                    if (dataSource) {
                        this.refreshValidation(dataSource);
                    }
                });
            }

            // Add filter handlers
            filterItems.forEach(filterItem => {
                filterItem.addEventListener('change', () => {
                    this.applyValidationFilter(panel);
                });
            });
        });
    }

    /**
     * Initialize validation rules panels
     */
    initValidationRulesPanels() {
        this.validationRulesPanels.forEach(panel => {
            // Get panel elements
            const addRuleButton = panel.querySelector('.add-rule');
            const ruleToggles = panel.querySelectorAll('.validation-rule-toggle');

            // Add rule toggle handlers
            ruleToggles.forEach(toggle => {
                toggle.addEventListener('change', () => {
                    const ruleId = toggle.dataset.ruleId;
                    const isEnabled = toggle.checked;

                    if (ruleId) {
                        this.toggleValidationRule(ruleId, isEnabled);
                    }
                });
            });
        });
    }

    /**
     * Initialize validation schema editors
     */
    initValidationSchemaEditors() {
        this.validationSchemaEditors.forEach(editor => {
            // Get editor elements
            const textarea = editor.querySelector('.validation-schema-editor-textarea');
            const saveButton = editor.querySelector('.save-schema');
            const resetButton = editor.querySelector('.reset-schema');
            const validateButton = editor.querySelector('.validate-schema');

            // Add save button handler
            if (saveButton && textarea) {
                saveButton.addEventListener('click', () => {
                    const schemaId = editor.dataset.schemaId;
                    const schemaContent = textarea.value;

                    if (schemaId && schemaContent) {
                        this.saveValidationSchema(schemaId, schemaContent);
                    }
                });
            }
        });
    }

    /**
     * Initialize validation level selectors
     */
    initValidationLevelSelectors() {
        this.validationLevelSelectors.forEach(selector => {
            // Get level options
            const levelOptions = selector.querySelectorAll('.validation-level-option');

            // Add level option handlers
            levelOptions.forEach(option => {
                option.addEventListener('click', () => {
                    const level = option.dataset.level;

                    if (level) {
                        // Update active option
                        levelOptions.forEach(opt => {
                            opt.classList.remove('active');
                        });
                        option.classList.add('active');

                        // Update validation level
                        this.setValidationLevel(level);
                    }
                });
            });
        });
    }

    /**
     * Load validation data
     */
    loadValidationData() {
        // Load validation results
        this.loadValidationResults();

        // Load validation rules
        this.loadValidationRules();

        // Load validation schemas
        this.loadValidationSchemas();
    }

    /**
     * Load validation results
     */
    loadValidationResults() {
        fetch('/api/validation/results')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.validationResults = data.results || {};
                    this.updateValidationResultsPanels();
                }
            })
            .catch(error => {
                console.error('Error loading validation results:', error);
                // Use mock data for demonstration
                this.validationResults = {
                    'market_data': {
                        is_valid: true,
                        errors: [],
                        warnings: [
                            {
                                message: 'Price change exceeds normal range',
                                path: 'price',
                                severity: 'WARNING',
                                details: { current: 45000, previous: 42000 }
                            }
                        ]
                    }
                };
                this.updateValidationResultsPanels();
            });
    }

    /**
     * Update validation results panels
     */
    updateValidationResultsPanels() {
        this.validationResultsPanels.forEach(panel => {
            const dataSource = panel.dataset.source;

            if (dataSource && this.validationResults[dataSource]) {
                const result = this.validationResults[dataSource];

                // Update validation status
                const statusIndicator = panel.querySelector('.validation-status');
                if (statusIndicator) {
                    const status = !result.is_valid ? 'invalid' : (result.warnings.length > 0 ? 'warning' : 'valid');
                    this.updateValidationStatus(statusIndicator, status);
                }

                // Update validation issues list
                const issuesList = panel.querySelector('.validation-issues-list');
                if (issuesList) {
                    // Clear existing issues
                    issuesList.innerHTML = '';

                    // Add errors
                    result.errors.forEach(error => {
                        this.addValidationIssue(issuesList, error, 'error');
                    });

                    // Add warnings
                    result.warnings.forEach(warning => {
                        this.addValidationIssue(issuesList, warning, 'warning');
                    });
                }
            }
        });
    }

    /**
     * Add validation issue
     * @param {HTMLElement} container - The container element
     * @param {Object} issue - The validation issue data
     * @param {string} type - The issue type (error, warning, info)
     */
    addValidationIssue(container, issue, type) {
        // Create issue element
        const issueElement = document.createElement('div');
        issueElement.className = `validation-issue ${type}`;

        // Create issue content
        issueElement.innerHTML = `
            <div class="validation-issue-icon">
                <i data-feather="${type === 'error' ? 'alert-circle' : 'alert-triangle'}"></i>
            </div>
            <div class="validation-issue-content">
                <div class="validation-issue-header">
                    <h4 class="validation-issue-title">${issue.severity}</h4>
                    ${issue.path ? `<div class="validation-issue-path">${issue.path}</div>` : ''}
                </div>
                <div class="validation-issue-message">${issue.message}</div>
            </div>
        `;

        // Add issue to container
        container.appendChild(issueElement);

        // Initialize Feather icons
        if (window.feather) {
            window.feather.replace();
        }
    }

    /**
     * Set validation level
     * @param {string} level - The validation level
     */
    setValidationLevel(level) {
        if (this.validationLevels.includes(level)) {
            this.currentValidationLevel = level;

            fetch('/api/validation/level', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ level })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log(`Validation level set to ${level}`);
                        if (window.showToast) {
                            window.showToast(`Validation level set to ${level}`, 'success');
                        }
                    }
                })
                .catch(error => {
                    console.error('Error updating validation level:', error);
                });
        }
    }
}

// Initialize data validation controller when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global data validation instance
    window.dataValidation = new DataValidationController();
});
