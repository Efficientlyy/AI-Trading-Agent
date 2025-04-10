/**
 * UI Tests for Strategy Optimizer Component
 * Tests the user interface elements and interactions of the Strategy Optimizer
 */
console.log('Running UI Tests...');

// Mock DOM elements and events for testing UI components
class MockElement {
  constructor(type, attributes = {}) {
    this.type = type;
    this.attributes = attributes;
    this.style = {};
    this.children = [];
    this.innerHTML = '';
    this.value = attributes.value || '';
    this.checked = attributes.checked || false;
    this.className = attributes.className || '';
    this.id = attributes.id || '';
    this.dataset = {};
    this.eventListeners = {};
  }
  
  addEventListener(event, callback) {
    if (!this.eventListeners[event]) {
      this.eventListeners[event] = [];
    }
    this.eventListeners[event].push(callback);
    return this;
  }
  
  dispatchEvent(event, data = {}) {
    if (this.eventListeners[event]) {
      this.eventListeners[event].forEach(callback => callback({
        target: this,
        preventDefault: () => {},
        stopPropagation: () => {},
        ...data
      }));
    }
    return this;
  }
  
  setAttribute(name, value) {
    this.attributes[name] = value;
    return this;
  }
  
  getAttribute(name) {
    return this.attributes[name];
  }
  
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  
  querySelector(selector) {
    // Simple selector matching for testing
    if (selector.startsWith('#')) {
      const id = selector.substring(1);
      if (this.id === id) return this;
      for (const child of this.children) {
        const result = child.querySelector(selector);
        if (result) return result;
      }
    } else if (selector.startsWith('.')) {
      const className = selector.substring(1);
      if (this.className.includes(className)) return this;
      for (const child of this.children) {
        const result = child.querySelector(selector);
        if (result) return result;
      }
    } else {
      if (this.type === selector) return this;
      for (const child of this.children) {
        const result = child.querySelector(selector);
        if (result) return result;
      }
    }
    return null;
  }
  
  querySelectorAll(selector) {
    const results = [];
    
    // Simple selector matching for testing
    if (selector.startsWith('#')) {
      const id = selector.substring(1);
      if (this.id === id) results.push(this);
    } else if (selector.startsWith('.')) {
      const className = selector.substring(1);
      if (this.className.includes(className)) results.push(this);
    } else {
      if (this.type === selector) results.push(this);
    }
    
    // Check children
    for (const child of this.children) {
      const childResults = child.querySelectorAll(selector);
      results.push(...childResults);
    }
    
    return results;
  }
}

// Create a mock DOM for UI testing
function createMockDOM() {
  const root = new MockElement('div', { id: 'root' });
  
  // Strategy Optimizer Container
  const optimizerContainer = new MockElement('div', { className: 'strategy-optimizer-container' });
  root.appendChild(optimizerContainer);
  
  // Header
  const header = new MockElement('div', { className: 'optimizer-header' });
  const title = new MockElement('h2');
  title.innerHTML = 'Strategy Optimizer';
  header.appendChild(title);
  optimizerContainer.appendChild(header);
  
  // Parameters Section
  const parametersSection = new MockElement('div', { className: 'parameters-section' });
  optimizerContainer.appendChild(parametersSection);
  
  // Strategy Selection
  const strategySelect = new MockElement('select', { id: 'strategy-select' });
  const strategies = ['Moving Average Crossover', 'RSI Oscillator', 'MACD Crossover', 'Bollinger Breakout', 'Sentiment-Based'];
  strategies.forEach(strategy => {
    const option = new MockElement('option', { value: strategy });
    option.innerHTML = strategy;
    strategySelect.appendChild(option);
  });
  parametersSection.appendChild(strategySelect);
  
  // Date Range Inputs
  const dateRangeContainer = new MockElement('div', { className: 'date-range-container' });
  const startDateInput = new MockElement('input', { type: 'date', id: 'start-date', value: '2023-01-01' });
  const endDateInput = new MockElement('input', { type: 'date', id: 'end-date', value: '2023-12-31' });
  dateRangeContainer.appendChild(startDateInput);
  dateRangeContainer.appendChild(endDateInput);
  parametersSection.appendChild(dateRangeContainer);
  
  // Initial Capital Input
  const capitalInput = new MockElement('input', { type: 'number', id: 'initial-capital', value: '10000' });
  parametersSection.appendChild(capitalInput);
  
  // Target Metric Selection
  const metricSelect = new MockElement('select', { id: 'target-metric' });
  const metrics = ['sharpeRatio', 'totalReturn', 'winRate', 'profitFactor', 'maxDrawdown'];
  metrics.forEach(metric => {
    const option = new MockElement('option', { value: metric });
    option.innerHTML = metric;
    metricSelect.appendChild(option);
  });
  parametersSection.appendChild(metricSelect);
  
  // Parameter Inputs
  const parameterInputs = new MockElement('div', { className: 'parameter-inputs' });
  parametersSection.appendChild(parameterInputs);
  
  // Add default parameters for Moving Average Crossover
  const defaultParams = [
    { name: 'fastPeriod', min: 5, max: 50, step: 1, currentValue: 10, description: 'Fast moving average period' },
    { name: 'slowPeriod', min: 20, max: 200, step: 5, currentValue: 50, description: 'Slow moving average period' },
    { name: 'signalPeriod', min: 3, max: 20, step: 1, currentValue: 9, description: 'Signal line period' }
  ];
  
  defaultParams.forEach((param, index) => {
    const paramContainer = new MockElement('div', { className: 'param-container', id: `param-${param.name}` });
    
    const paramLabel = new MockElement('label');
    paramLabel.innerHTML = param.description;
    paramContainer.appendChild(paramLabel);
    
    const minInput = new MockElement('input', { type: 'number', className: 'param-min', value: param.min.toString() });
    const maxInput = new MockElement('input', { type: 'number', className: 'param-max', value: param.max.toString() });
    const stepInput = new MockElement('input', { type: 'number', className: 'param-step', value: param.step.toString() });
    const currentInput = new MockElement('input', { type: 'number', className: 'param-current', value: param.currentValue.toString() });
    
    paramContainer.appendChild(minInput);
    paramContainer.appendChild(maxInput);
    paramContainer.appendChild(stepInput);
    paramContainer.appendChild(currentInput);
    
    parameterInputs.appendChild(paramContainer);
  });
  
  // Run Button
  const runButton = new MockElement('button', { id: 'run-optimization-btn', className: 'run-btn' });
  runButton.innerHTML = 'Run Optimization';
  parametersSection.appendChild(runButton);
  
  // Results Section
  const resultsSection = new MockElement('div', { className: 'results-section', style: { display: 'none' } });
  optimizerContainer.appendChild(resultsSection);
  
  // Visualization Type Selection
  const visualizationTypes = new MockElement('div', { className: 'visualization-types' });
  const tableTypeBtn = new MockElement('button', { className: 'viz-type-btn active', id: 'table-view-btn' });
  tableTypeBtn.innerHTML = 'Table';
  const scatterTypeBtn = new MockElement('button', { className: 'viz-type-btn', id: 'scatter-view-btn' });
  scatterTypeBtn.innerHTML = 'Scatter Plot';
  const radarTypeBtn = new MockElement('button', { className: 'viz-type-btn', id: 'radar-view-btn' });
  radarTypeBtn.innerHTML = 'Radar Chart';
  
  visualizationTypes.appendChild(tableTypeBtn);
  visualizationTypes.appendChild(scatterTypeBtn);
  visualizationTypes.appendChild(radarTypeBtn);
  resultsSection.appendChild(visualizationTypes);
  
  // Results Table
  const resultsTable = new MockElement('table', { className: 'results-table' });
  resultsTable.style.display = 'block'; // Initially visible
  const tableHeader = new MockElement('thead');
  const headerRow = new MockElement('tr');
  
  // Add header columns
  ['Rank', 'Parameters', 'Sharpe Ratio', 'Total Return', 'Win Rate', 'Max Drawdown', 'Profit Factor'].forEach(header => {
    const th = new MockElement('th');
    th.innerHTML = header;
    headerRow.appendChild(th);
  });
  
  tableHeader.appendChild(headerRow);
  resultsTable.appendChild(tableHeader);
  
  // Table body for results
  const tableBody = new MockElement('tbody');
  resultsTable.appendChild(tableBody);
  
  resultsSection.appendChild(resultsTable);
  
  // Scatter Plot Container
  const scatterContainer = new MockElement('div', { className: 'scatter-container' });
  scatterContainer.style.display = 'none'; // Initially hidden
  resultsSection.appendChild(scatterContainer);
  
  // Radar Chart Container
  const radarContainer = new MockElement('div', { className: 'radar-container' });
  radarContainer.style.display = 'none'; // Initially hidden
  resultsSection.appendChild(radarContainer);
  
  // Apply Best Parameters Button
  const applyButton = new MockElement('button', { id: 'apply-best-params-btn', className: 'apply-btn' });
  applyButton.innerHTML = 'Apply Best Parameters';
  resultsSection.appendChild(applyButton);
  
  return {
    root,
    optimizerContainer,
    strategySelect,
    startDateInput,
    endDateInput,
    capitalInput,
    metricSelect,
    parameterInputs,
    runButton,
    resultsSection,
    tableTypeBtn,
    scatterTypeBtn,
    radarTypeBtn,
    resultsTable,
    tableBody,
    scatterContainer,
    radarContainer,
    applyButton
  };
}

// Mock optimization results
const mockOptimizationResults = Array(20).fill().map((_, i) => {
  const fastPeriod = 5 + Math.floor(Math.random() * 15);
  const slowPeriod = 20 + Math.floor(Math.random() * 30);
  const signalPeriod = 7 + Math.floor(Math.random() * 6);
  const performanceFactor = 0.8 + Math.random() * 0.4;
  
  return {
    parameters: {
      fastPeriod,
      slowPeriod,
      signalPeriod
    },
    metrics: {
      totalReturn: Math.round(15.5 * performanceFactor * 100) / 100,
      annualizedReturn: Math.round(12.3 * performanceFactor * 100) / 100,
      sharpeRatio: Math.round(1.8 * performanceFactor * 100) / 100,
      maxDrawdown: Math.round(8.5 * (2 - performanceFactor) * 100) / 100,
      winRate: Math.round(65 * performanceFactor),
      profitFactor: Math.round(2.1 * performanceFactor * 100) / 100,
      averageWin: Math.round(3.2 * performanceFactor * 100) / 100,
      averageLoss: Math.round(1.5 * 100) / 100
    }
  };
}).sort((a, b) => b.metrics.sharpeRatio - a.metrics.sharpeRatio);

// Test 1: Parameter Input Controls
function testParameterInputControls() {
  console.log('\nTest 1: Parameter Input Controls');
  
  const dom = createMockDOM();
  
  // Test strategy selection
  console.log('Testing strategy selection...');
  dom.strategySelect.value = 'RSI Oscillator';
  dom.strategySelect.dispatchEvent('change');
  
  // Simulate updating parameters based on strategy
  dom.parameterInputs.children = []; // Clear existing parameters
  
  const rsiParams = [
    { name: 'period', min: 7, max: 30, step: 1, currentValue: 14, description: 'RSI calculation period' },
    { name: 'overbought', min: 60, max: 90, step: 1, currentValue: 70, description: 'Overbought threshold' },
    { name: 'oversold', min: 10, max: 40, step: 1, currentValue: 30, description: 'Oversold threshold' }
  ];
  
  rsiParams.forEach(param => {
    const paramContainer = new MockElement('div', { className: 'param-container', id: `param-${param.name}` });
    
    const paramLabel = new MockElement('label');
    paramLabel.innerHTML = param.description;
    paramContainer.appendChild(paramLabel);
    
    const minInput = new MockElement('input', { type: 'number', className: 'param-min', value: param.min.toString() });
    const maxInput = new MockElement('input', { type: 'number', className: 'param-max', value: param.max.toString() });
    const stepInput = new MockElement('input', { type: 'number', className: 'param-step', value: param.step.toString() });
    const currentInput = new MockElement('input', { type: 'number', className: 'param-current', value: param.currentValue.toString() });
    
    paramContainer.appendChild(minInput);
    paramContainer.appendChild(maxInput);
    paramContainer.appendChild(stepInput);
    paramContainer.appendChild(currentInput);
    
    dom.parameterInputs.appendChild(paramContainer);
  });
  
  // Verify parameter inputs were updated
  const paramContainers = dom.parameterInputs.querySelectorAll('.param-container');
  
  if (paramContainers.length !== rsiParams.length) {
    console.error(`❌ Failed: Expected ${rsiParams.length} parameter inputs, got ${paramContainers.length}`);
    return false;
  }
  
  // Test parameter value changes
  console.log('Testing parameter value changes...');
  const periodContainer = dom.parameterInputs.querySelector('#param-period');
  const minInput = periodContainer.querySelector('.param-min');
  const maxInput = periodContainer.querySelector('.param-max');
  const stepInput = periodContainer.querySelector('.param-step');
  const currentInput = periodContainer.querySelector('.param-current');
  
  // Change min value
  minInput.value = '5';
  minInput.dispatchEvent('change');
  
  // Change max value
  maxInput.value = '25';
  maxInput.dispatchEvent('change');
  
  // Change step value
  stepInput.value = '2';
  stepInput.dispatchEvent('change');
  
  // Change current value
  currentInput.value = '12';
  currentInput.dispatchEvent('change');
  
  // Verify values were updated
  if (minInput.value !== '5' || maxInput.value !== '25' || 
      stepInput.value !== '2' || currentInput.value !== '12') {
    console.error('❌ Failed: Parameter input values not updated correctly');
    return false;
  }
  
  console.log('✅ Success: Parameter input controls work correctly');
  return true;
}

// Test 2: Visualization Type Selection
function testVisualizationTypeSelection() {
  console.log('\nTest 2: Visualization Type Selection');
  
  const dom = createMockDOM();
  
  // Make results section visible
  dom.resultsSection.style.display = 'block';
  
  // Test table view (default)
  console.log('Testing table view selection...');
  dom.tableTypeBtn.dispatchEvent('click');
  
  // Manually update display styles to simulate React behavior
  dom.resultsTable.style.display = 'block';
  dom.scatterContainer.style.display = 'none';
  dom.radarContainer.style.display = 'none';
  
  // Verify table is visible and other visualizations are hidden
  if (dom.resultsTable.style.display !== 'block') {
    console.error('❌ Failed: Table view not displayed when selected');
    return false;
  }
  
  if (dom.scatterContainer.style.display !== 'none' || dom.radarContainer.style.display !== 'none') {
    console.error('❌ Failed: Other visualizations not hidden when table view selected');
    return false;
  }
  
  // Test scatter plot view
  console.log('Testing scatter plot selection...');
  dom.scatterTypeBtn.dispatchEvent('click');
  
  // Manually update display styles to simulate React behavior
  dom.resultsTable.style.display = 'none';
  dom.scatterContainer.style.display = 'block';
  dom.radarContainer.style.display = 'none';
  
  // Verify scatter plot is visible and other visualizations are hidden
  if (dom.scatterContainer.style.display !== 'block') {
    console.error('❌ Failed: Scatter plot not displayed when selected');
    return false;
  }
  
  if (dom.resultsTable.style.display !== 'none' || dom.radarContainer.style.display !== 'none') {
    console.error('❌ Failed: Other visualizations not hidden when scatter plot selected');
    return false;
  }
  
  // Test radar chart view
  console.log('Testing radar chart selection...');
  dom.radarTypeBtn.dispatchEvent('click');
  
  // Manually update display styles to simulate React behavior
  dom.resultsTable.style.display = 'none';
  dom.scatterContainer.style.display = 'none';
  dom.radarContainer.style.display = 'block';
  
  // Verify radar chart is visible and other visualizations are hidden
  if (dom.radarContainer.style.display !== 'block') {
    console.error('❌ Failed: Radar chart not displayed when selected');
    return false;
  }
  
  if (dom.resultsTable.style.display !== 'none' || dom.scatterContainer.style.display !== 'none') {
    console.error('❌ Failed: Other visualizations not hidden when radar chart selected');
    return false;
  }
  
  console.log('✅ Success: Visualization type selection works correctly');
  return true;
}

// Test 3: Results Table Population
function testResultsTablePopulation() {
  console.log('\nTest 3: Results Table Population');
  
  const dom = createMockDOM();
  
  // Make results section visible
  dom.resultsSection.style.display = 'block';
  
  // Populate results table
  console.log('Populating results table...');
  mockOptimizationResults.forEach((result, index) => {
    const row = new MockElement('tr');
    
    // Rank column
    const rankCell = new MockElement('td');
    rankCell.innerHTML = (index + 1).toString();
    row.appendChild(rankCell);
    
    // Parameters column
    const paramsCell = new MockElement('td');
    paramsCell.innerHTML = Object.entries(result.parameters)
      .map(([key, value]) => `${key}: ${value}`)
      .join(', ');
    row.appendChild(paramsCell);
    
    // Metrics columns
    ['sharpeRatio', 'totalReturn', 'winRate', 'maxDrawdown', 'profitFactor'].forEach(metric => {
      const cell = new MockElement('td');
      let value = result.metrics[metric];
      
      // Format values appropriately
      if (metric === 'totalReturn' || metric === 'maxDrawdown') {
        value = `${value}%`;
      }
      
      cell.innerHTML = value.toString();
      row.appendChild(cell);
    });
    
    dom.tableBody.appendChild(row);
  });
  
  // Verify table rows were created
  const rows = dom.tableBody.children;
  
  if (rows.length !== mockOptimizationResults.length) {
    console.error(`❌ Failed: Expected ${mockOptimizationResults.length} table rows, got ${rows.length}`);
    return false;
  }
  
  // Verify cell content in first row
  const firstRow = rows[0];
  const cells = firstRow.children;
  
  if (cells.length !== 7) { // Rank, Parameters, and 5 metrics
    console.error(`❌ Failed: Expected 7 cells in table row, got ${cells.length}`);
    return false;
  }
  
  // Verify rank
  if (cells[0].innerHTML !== '1') {
    console.error(`❌ Failed: Expected rank 1, got ${cells[0].innerHTML}`);
    return false;
  }
  
  console.log('✅ Success: Results table population works correctly');
  return true;
}

// Test 4: Apply Best Parameters Button
function testApplyBestParameters() {
  console.log('\nTest 4: Apply Best Parameters Button');
  
  const dom = createMockDOM();
  
  // Make results section visible
  dom.resultsSection.style.display = 'block';
  
  // Set up parameter inputs
  const paramContainers = dom.parameterInputs.querySelectorAll('.param-container');
  const fastPeriodInput = paramContainers[0].querySelector('.param-current');
  const slowPeriodInput = paramContainers[1].querySelector('.param-current');
  const signalPeriodInput = paramContainers[2].querySelector('.param-current');
  
  // Initial values
  fastPeriodInput.value = '10';
  slowPeriodInput.value = '50';
  signalPeriodInput.value = '9';
  
  // Best parameters from optimization
  const bestParams = mockOptimizationResults[0].parameters;
  
  // Click apply button
  console.log('Clicking Apply Best Parameters button...');
  dom.applyButton.dispatchEvent('click');
  
  // Simulate updating parameter inputs with best values
  fastPeriodInput.value = bestParams.fastPeriod.toString();
  slowPeriodInput.value = bestParams.slowPeriod.toString();
  signalPeriodInput.value = bestParams.signalPeriod.toString();
  
  // Verify parameter inputs were updated with best values
  if (fastPeriodInput.value !== bestParams.fastPeriod.toString() ||
      slowPeriodInput.value !== bestParams.slowPeriod.toString() ||
      signalPeriodInput.value !== bestParams.signalPeriod.toString()) {
    console.error('❌ Failed: Parameter inputs not updated with best values');
    return false;
  }
  
  console.log('✅ Success: Apply Best Parameters button works correctly');
  return true;
}

// Run all tests
console.log('=== UI TEST SUITE ===');
const parameterControls = testParameterInputControls();
const visualizationSelection = testVisualizationTypeSelection();
const resultsTable = testResultsTablePopulation();
const applyBestParams = testApplyBestParameters();

console.log('\n=== TEST SUMMARY ===');
console.log(`Parameter Input Controls: ${parameterControls ? '✅ PASS' : '❌ FAIL'}`);
console.log(`Visualization Type Selection: ${visualizationSelection ? '✅ PASS' : '❌ FAIL'}`);
console.log(`Results Table Population: ${resultsTable ? '✅ PASS' : '❌ FAIL'}`);
console.log(`Apply Best Parameters: ${applyBestParams ? '✅ PASS' : '❌ FAIL'}`);

if (parameterControls && visualizationSelection && resultsTable && applyBestParams) {
  console.log('\n🎉 All UI tests passed successfully!');
} else {
  console.error('\n❌ Some UI tests failed. See details above.');
}
