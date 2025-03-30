/**
 * Data Transformation Pipeline JavaScript
 * 
 * Handles the data transformation pipeline UI components and functionality
 */

// Data Transformation Pipeline Controller
class TransformationPipelineController {
    constructor() {
        // DOM Elements
        this.pipelineContainers = document.querySelectorAll('.transformation-pipeline');
        this.pipelineVisualizations = document.querySelectorAll('.pipeline-visualization');
        this.transformationStepsList = document.querySelectorAll('.transformation-steps-list');
        this.addStepForms = document.querySelectorAll('.add-step-form');
        this.transformationPreviews = document.querySelectorAll('.transformation-preview');

        // State
        this.transformers = {};
        this.activeTransformer = null;
        this.previewData = {};
        this.transformationStepTemplates = {};

        // Initialize
        this.init();
    }

    /**
     * Initialize the transformation pipeline controller
     */
    init() {
        // Initialize pipeline containers
        this.initPipelineContainers();

        // Initialize transformation steps lists
        this.initTransformationStepsLists();

        // Initialize add step forms
        this.initAddStepForms();

        // Initialize transformation previews
        this.initTransformationPreviews();

        // Load initial data
        this.loadTransformers();

        console.log('Transformation pipeline controller initialized');
    }

    /**
     * Initialize pipeline containers
     */
    initPipelineContainers() {
        this.pipelineContainers.forEach(container => {
            // Get container elements
            const refreshButton = container.querySelector('.refresh-pipeline');
            const clearButton = container.querySelector('.clear-pipeline');

            // Add refresh button handler
            if (refreshButton) {
                refreshButton.addEventListener('click', () => {
                    const transformerId = container.dataset.transformerId;
                    if (transformerId) {
                        this.refreshPipeline(transformerId);
                    }
                });
            }

            // Add clear button handler
            if (clearButton) {
                clearButton.addEventListener('click', () => {
                    const transformerId = container.dataset.transformerId;
                    if (transformerId) {
                        this.clearPipeline(transformerId);
                    }
                });
            }
        });
    }

    /**
     * Initialize transformation steps lists
     */
    initTransformationStepsLists() {
        this.transformationStepsList.forEach(list => {
            // Add event listeners to step actions
            list.addEventListener('click', (event) => {
                const stepElement = event.target.closest('.transformation-step');
                if (!stepElement) return;

                const transformerId = list.closest('.transformation-pipeline').dataset.transformerId;
                const stepId = stepElement.dataset.stepId;

                if (!transformerId || !stepId) return;

                // Handle edit button
                if (event.target.closest('.edit-step')) {
                    this.editTransformationStep(transformerId, stepId);
                }

                // Handle delete button
                if (event.target.closest('.delete-step')) {
                    this.deleteTransformationStep(transformerId, stepId);
                }

                // Handle toggle
                if (event.target.closest('.transformation-step-toggle')) {
                    const toggle = event.target.closest('.transformation-step-toggle');
                    this.toggleTransformationStep(transformerId, stepId, toggle.checked);
                }
            });
        });
    }

    /**
     * Initialize add step forms
     */
    initAddStepForms() {
        this.addStepForms.forEach(form => {
            // Get form elements
            const addButton = form.querySelector('.add-step-button');
            const cancelButton = form.querySelector('.cancel-step-button');
            const stepTypeSelect = form.querySelector('.step-type-select');

            // Add add button handler
            if (addButton) {
                addButton.addEventListener('click', () => {
                    const transformerId = form.closest('.transformation-pipeline').dataset.transformerId;
                    if (!transformerId) return;

                    // Get form values
                    const stepName = form.querySelector('.step-name-input').value;
                    const stepType = stepTypeSelect ? stepTypeSelect.value : '';
                    const stepStage = form.querySelector('.step-stage-select').value;
                    const stepConfig = form.querySelector('.step-config-textarea').value;

                    // Validate form
                    if (!stepName || !stepType || !stepStage) {
                        if (window.showToast) {
                            window.showToast('Please fill in all required fields', 'error');
                        }
                        return;
                    }

                    // Parse config
                    let config;
                    try {
                        config = JSON.parse(stepConfig);
                    } catch (error) {
                        if (window.showToast) {
                            window.showToast(`Invalid JSON configuration: ${error.message}`, 'error');
                        }
                        return;
                    }

                    // Add step
                    this.addTransformationStep(transformerId, {
                        name: stepName,
                        type: stepType,
                        stage: stepStage,
                        config: config
                    });

                    // Reset form
                    form.reset();
                });
            }
        });
    }

    /**
     * Initialize transformation previews
     */
    initTransformationPreviews() {
        this.transformationPreviews.forEach(preview => {
            // Get preview elements
            const tabs = preview.querySelectorAll('.transformation-preview-tab');
            const contents = preview.querySelectorAll('.transformation-preview-content > div');

            // Add tab click handlers
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Update active tab
                    tabs.forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');

                    // Show corresponding content
                    const tabId = tab.dataset.tabId;
                    contents.forEach(content => {
                        content.style.display = content.classList.contains(tabId) ? 'block' : 'none';
                    });
                });
            });
        });
    }

    /**
     * Load transformers
     */
    loadTransformers() {
        // In a real implementation, this would load transformers from the server
        // For now, we'll just use mock data

        fetch('/api/transformers')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.transformers = data.transformers || {};
                    this.transformationStepTemplates = data.step_templates || {};
                    this.updateTransformerUI();
                }
            })
            .catch(error => {
                console.error('Error loading transformers:', error);

                // Use mock data for demonstration
                this.transformers = {
                    'market_data': {
                        name: 'Market Data Transformer',
                        enabled: true,
                        steps: [
                            {
                                id: 'step1',
                                name: 'Normalize Field Names',
                                type: 'normalize_field_names',
                                stage: 'NORMALIZATION',
                                enabled: true,
                                config: {
                                    field_mapping: {
                                        's': 'symbol',
                                        'p': 'price',
                                        'v': 'volume',
                                        't': 'timestamp'
                                    }
                                }
                            },
                            {
                                id: 'step2',
                                name: 'Standardize Datetime',
                                type: 'standardize_datetime',
                                stage: 'STANDARDIZATION',
                                enabled: true,
                                config: {
                                    fields: ['timestamp'],
                                    output_format: '%Y-%m-%dT%H:%M:%S.%fZ'
                                }
                            }
                        ]
                    }
                };

                this.updateTransformerUI();
            });
    }

    /**
     * Update transformer UI
     */
    updateTransformerUI() {
        this.pipelineContainers.forEach(container => {
            const transformerId = container.dataset.transformerId;
            if (transformerId && this.transformers[transformerId]) {
                const transformer = this.transformers[transformerId];

                // Update pipeline visualization
                this.updatePipelineVisualization(container, transformer);

                // Update transformation steps list
                this.updateTransformationStepsList(container, transformer);
            }
        });
    }

    /**
     * Update pipeline visualization
     * @param {HTMLElement} container - The pipeline container
     * @param {Object} transformer - The transformer data
     */
    updatePipelineVisualization(container, transformer) {
        const visualization = container.querySelector('.pipeline-visualization');
        if (!visualization) return;

        // Clear visualization
        visualization.innerHTML = '';

        // Get steps by stage
        const stepsByStage = {
            'NORMALIZATION': [],
            'STANDARDIZATION': [],
            'ENRICHMENT': [],
            'OPTIMIZATION': []
        };

        transformer.steps.forEach(step => {
            if (stepsByStage[step.stage]) {
                stepsByStage[step.stage].push(step);
            }
        });

        // Create stage elements
        const stages = [
            { id: 'NORMALIZATION', name: 'Normalization', description: 'Standardize field names and formats' },
            { id: 'STANDARDIZATION', name: 'Standardization', description: 'Ensure consistent data formats' },
            { id: 'ENRICHMENT', name: 'Enrichment', description: 'Add calculated fields and context' },
            { id: 'OPTIMIZATION', name: 'Optimization', description: 'Optimize data size and structure' }
        ];

        stages.forEach((stage, index) => {
            // Create stage element
            const stageElement = document.createElement('div');
            stageElement.className = 'pipeline-stage';

            // Add active class if stage has steps
            if (stepsByStage[stage.id].length > 0) {
                stageElement.classList.add('active');
            }

            // Create stage content
            stageElement.innerHTML = `
                <div class="pipeline-stage-icon">
                    <i data-feather="${this.getStageIcon(stage.id)}"></i>
                </div>
                <div class="pipeline-stage-content">
                    <div class="pipeline-stage-name">${stage.name}</div>
                    <div class="pipeline-stage-description">${stage.description}</div>
                </div>
            `;

            // Add stage to visualization
            visualization.appendChild(stageElement);

            // Add connector if not last stage
            if (index < stages.length - 1) {
                const connector = document.createElement('div');
                connector.className = 'pipeline-connector';
                visualization.appendChild(connector);
            }
        });

        // Initialize Feather icons
        if (window.feather) {
            window.feather.replace();
        }
    }

    /**
     * Get stage icon
     * @param {string} stage - The stage ID
     * @returns {string} The icon name
     */
    getStageIcon(stage) {
        switch (stage) {
            case 'NORMALIZATION':
                return 'edit';
            case 'STANDARDIZATION':
                return 'check-square';
            case 'ENRICHMENT':
                return 'plus-circle';
            case 'OPTIMIZATION':
                return 'zap';
            default:
                return 'circle';
        }
    }

    /**
     * Update transformation steps list
     * @param {HTMLElement} container - The pipeline container
     * @param {Object} transformer - The transformer data
     */
    updateTransformationStepsList(container, transformer) {
        const stepsList = container.querySelector('.transformation-steps-list');
        if (!stepsList) return;

        // Clear steps list
        stepsList.innerHTML = '';

        // Add steps
        transformer.steps.forEach(step => {
            // Create step element
            const stepElement = document.createElement('div');
            stepElement.className = 'transformation-step';
            stepElement.dataset.stepId = step.id;

            // Create step content
            stepElement.innerHTML = `
                <div class="transformation-step-toggle">
                    <input type="checkbox" id="step-${step.id}" ${step.enabled ? 'checked' : ''}>
                </div>
                <div class="transformation-step-content">
                    <div class="transformation-step-header">
                        <h4 class="transformation-step-name">${step.name}</h4>
                        <span class="transformation-step-stage ${step.stage.toLowerCase()}">${step.stage}</span>
                    </div>
                </div>
                <div class="transformation-step-actions">
                    <button class="btn btn-sm btn-secondary edit-step">
                        <i data-feather="edit-2"></i>
                    </button>
                    <button class="btn btn-sm btn-danger delete-step">
                        <i data-feather="trash-2"></i>
                    </button>
                </div>
            `;

            // Add step to list
            stepsList.appendChild(stepElement);
        });

        // Initialize Feather icons
        if (window.feather) {
            window.feather.replace();
        }
    }
}

// Initialize transformation pipeline controller when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global transformation pipeline instance
    window.transformationPipeline = new TransformationPipelineController();
});
